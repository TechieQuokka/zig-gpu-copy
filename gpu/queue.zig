//! GPFIFO channel: submission, fences and channel fault reporting.
//!
//! Every submission ends with a fence release written by the caller. The queue
//! reuses GPFIFO entries and pushbuffer space only after the fence of the work
//! using them has passed, so it never overwrites anything the GPU may still read.

const std = @import("std");
const abi = @import("rm/abi.zig");
const sys = @import("rm/sys.zig");
const Client = @import("rm/client.zig").Client;
const memory = @import("memory.zig");
const Buffer = memory.Buffer;
const Handle = abi.Handle;

pub const Error = sys.Error || error{ FenceTimeout, ChannelFault, AddressTooLarge, SubmissionTooLarge };

const entry_count: u32 = 64;
const pushbuffer_size: u64 = 64 * 1024;
/// Long enough for the largest benchmark. Faults are reported much sooner, through
/// the error notifier.
const default_timeout_ns: u64 = 30 * std.time.ns_per_s;

/// clc56f.h: NVC56F_GP_ENTRY0 and NVC56F_GP_ENTRY1.
pub const GpEntry = packed struct(u64) {
    fetch: u1 = 0,
    reserved: u1 = 0,
    get_lo: u30,
    get_hi: u8,
    priv: u1 = 0,
    level: u1 = 0,
    length_words: u21,
    sync: u1 = 0,

    pub fn init(gpu_va: u64, length_words: u21) error{AddressTooLarge}!GpEntry {
        std.debug.assert(gpu_va & 3 == 0);
        if (gpu_va >> 40 != 0) return error.AddressTooLarge;
        return .{
            .get_lo = @truncate(gpu_va >> 2),
            .get_hi = @truncate(gpu_va >> 32),
            .length_words = length_words,
        };
    }
};

pub const Fence = struct { value: u32 };

/// Space in the pushbuffer, and the fence the recorded work must release.
pub const Recording = struct {
    words: []u32,
    fence_va: u64,
    fence_value: u32,
    pb_offset: u64,
};

const InFlight = struct { pb_start: u64, pb_end: u64, fence: u32 };

/// Orders CPU writes before the GPU is told about them, and is a compiler
/// barrier as well.
inline fn writeBarrier() void {
    asm volatile ("mfence" ::: .{ .memory = true });
}

/// Orders reads of GPU-written data after the fence that announced them.
inline fn acquireBarrier() void {
    asm volatile ("lfence" ::: .{ .memory = true });
}

pub const Queue = struct {
    gpfifo: Buffer,
    userd: Buffer,
    pushbuffer: Buffer,
    fence: Buffer,
    notifier: Buffer,
    doorbell: memory.CpuMapping,
    h_usermode: Handle,
    h_channel: Handle,
    token: u32,

    gp_put: u32,
    pb_head: u64,
    next_fence: u32,
    in_flight: [entry_count]InFlight,
    oldest: u32,
    in_flight_count: u32,

    pub fn create(client: *Client, h_vaspace: Handle, engine: u32) sys.Error!Queue {
        var gpfifo = try Buffer.create(client, h_vaspace, .sysmem, entry_count * @sizeOf(GpEntry));
        errdefer gpfifo.destroy(client);
        var userd = try Buffer.create(client, h_vaspace, .sysmem, abi.userd_size);
        errdefer userd.destroy(client);
        var pushbuffer = try Buffer.create(client, h_vaspace, .sysmem, pushbuffer_size);
        errdefer pushbuffer.destroy(client);
        var fence = try Buffer.create(client, h_vaspace, .sysmem, 4096);
        errdefer fence.destroy(client);
        var notifier = try Buffer.create(client, h_vaspace, .sysmem, abi.notifier_slots * @sizeOf(abi.Notification));
        errdefer notifier.destroy(client);

        @memset(fence.slice(u8), 0);
        @memset(notifier.slice(u8), 0);

        const h_usermode = try client.alloc(client.h_subdevice, abi.class.usermode, null, "alloc usermode doorbell");
        errdefer client.free(h_usermode, client.h_subdevice);
        var doorbell = try memory.CpuMapping.create(client, .gpu_bar, h_usermode, abi.usermode_size, abi.map_access_write_only);
        errdefer doorbell.destroy(client);

        var params = std.mem.zeroes(abi.ChannelAlloc);
        params.hObjectError = notifier.h_physical;
        params.hObjectBuffer = gpfifo.h_physical;
        params.gpFifoOffset = gpfifo.gpu_va;
        params.gpFifoEntries = entry_count;
        params.hVASpace = h_vaspace;
        params.hUserdMemory[0] = userd.h_physical;
        params.engineType = engine;
        params.subDeviceId = 1; // One-hot mask of subdevice 0.
        const h_channel = try client.alloc(client.h_device, abi.class.channel_gpfifo, &params, "alloc channel");

        return .{
            .gpfifo = gpfifo,
            .userd = userd,
            .pushbuffer = pushbuffer,
            .fence = fence,
            .notifier = notifier,
            .doorbell = doorbell,
            .h_usermode = h_usermode,
            .h_channel = h_channel,
            .token = 0,
            .gp_put = 0,
            .pb_head = 0,
            .next_fence = 1,
            .in_flight = undefined,
            .oldest = 0,
            .in_flight_count = 0,
        };
    }

    /// Binds an engine object to the channel. Call before `start`.
    pub fn bind(self: *Queue, client: *Client, class: u32, params: anytype, what: []const u8) sys.Error!void {
        _ = try client.alloc(self.h_channel, class, params, what);
    }

    /// Fetches the doorbell token and enables scheduling.
    pub fn start(self: *Queue, client: *Client) sys.Error!void {
        var token = std.mem.zeroes(abi.WorkSubmitToken);
        try client.control(self.h_channel, abi.ctrl.gpfifo_get_work_submit_token, &token, "get work submit token");
        self.token = token.workSubmitToken;

        var schedule = std.mem.zeroes(abi.GpfifoSchedule);
        schedule.bEnable = 1;
        try client.control(self.h_channel, abi.ctrl.gpfifo_schedule, &schedule, "schedule channel");
    }

    pub fn destroy(self: *Queue, client: *Client) void {
        // Freeing the channel stops it and releases the objects bound to it.
        client.free(self.h_channel, client.h_device);
        self.doorbell.destroy(client);
        client.free(self.h_usermode, client.h_subdevice);
        self.notifier.destroy(client);
        self.fence.destroy(client);
        self.pushbuffer.destroy(client);
        self.userd.destroy(client);
        self.gpfifo.destroy(client);
    }

    /// Reserves pushbuffer space for up to `max_words`, waiting for earlier work
    /// when the GPFIFO or the pushbuffer is full.
    pub fn begin(self: *Queue, max_words: usize) Error!Recording {
        const length = max_words * @sizeOf(u32);
        if (length > pushbuffer_size) return error.SubmissionTooLarge;

        self.retire();
        // One entry stays empty, because GP_PUT == GP_GET means an empty ring.
        while (self.in_flight_count >= entry_count - 1) try self.waitOldest();

        const start_offset = if (self.pb_head + length > pushbuffer_size) 0 else self.pb_head;
        while (self.overlapsInFlight(start_offset, start_offset + length)) try self.waitOldest();

        return .{
            .words = self.pushbuffer.slice(u32)[start_offset / 4..][0..max_words],
            .fence_va = self.fence.gpu_va,
            .fence_value = self.next_fence,
            .pb_offset = start_offset,
        };
    }

    /// Submits the first `used_words` of `recording`. The recorded work must
    /// release `recording.fence_value` at `recording.fence_va`.
    pub fn submit(self: *Queue, recording: Recording, used_words: usize) Error!Fence {
        std.debug.assert(used_words <= recording.words.len);
        const length = used_words * @sizeOf(u32);
        const entry = try GpEntry.init(self.pushbuffer.gpu_va + recording.pb_offset, @intCast(used_words));

        const entries: [*]volatile GpEntry = @ptrCast(self.gpfifo.slice(u64).ptr);
        entries[self.gp_put] = entry;

        const slot = (self.oldest + self.in_flight_count) % entry_count;
        self.in_flight[slot] = .{
            .pb_start = recording.pb_offset,
            .pb_end = recording.pb_offset + length,
            .fence = recording.fence_value,
        };
        self.in_flight_count += 1;
        self.pb_head = recording.pb_offset + length;
        self.next_fence +%= 1;
        self.gp_put = (self.gp_put + 1) % entry_count;

        writeBarrier();
        self.userdWord(abi.userd_gp_put).* = self.gp_put;
        writeBarrier();

        const bell: *volatile u32 = @ptrCast(@alignCast(self.doorbell.ptr + abi.usermode_notify_channel_pending));
        bell.* = self.token;

        return .{ .value = recording.fence_value };
    }

    /// Waits for a fence. Returns error.ChannelFault as soon as RM reports a
    /// fault on this channel, rather than waiting for the timeout.
    pub fn wait(self: *Queue, fence: Fence) Error!void {
        const start_ns = sys.monotonicNs();
        while (!self.passed(fence.value)) {
            try self.checkFault();
            if (sys.monotonicNs() - start_ns > default_timeout_ns) {
                std.debug.print("error: fence {d} timed out: current={d} GP_GET={d} GP_PUT={d}\n", .{
                    fence.value, self.fenceWord().*, self.userdWord(abi.userd_gp_get).*, self.gp_put,
                });
                return error.FenceTimeout;
            }
            std.atomic.spinLoopHint();
        }
        // A fault can be reported after the fence passes, so this catches only those
        // already reported. Run `checkFault` again after a batch of work to be sure.
        try self.checkFault();
        // Data written before the release may now be read by this thread.
        acquireBarrier();
    }

    /// Returns error.ChannelFault if RM has reported a fault on this channel.
    pub fn checkFault(self: *Queue) error{ChannelFault}!void {
        const notification: *const volatile abi.Notification = @ptrCast(@alignCast(self.notifier.slice(u8).ptr));
        const code = notification.info32;
        if (code != 0 or notification.status != 0) {
            std.debug.print("error: channel fault: error code {d} (Xid), status 0x{x}\n", .{ code, notification.status });
            return error.ChannelFault;
        }
    }

    fn passed(self: *Queue, value: u32) bool {
        // Signed distance, so that the comparison survives wraparound.
        const distance: i32 = @bitCast(self.fenceWord().* -% value);
        return distance >= 0;
    }

    fn retire(self: *Queue) void {
        while (self.in_flight_count > 0 and self.passed(self.in_flight[self.oldest].fence)) {
            self.oldest = (self.oldest + 1) % entry_count;
            self.in_flight_count -= 1;
        }
    }

    fn waitOldest(self: *Queue) Error!void {
        std.debug.assert(self.in_flight_count > 0);
        try self.wait(.{ .value = self.in_flight[self.oldest].fence });
        self.retire();
    }

    fn overlapsInFlight(self: *const Queue, start_offset: u64, end_offset: u64) bool {
        for (0..self.in_flight_count) |i| {
            const item = self.in_flight[(self.oldest + i) % entry_count];
            if (start_offset < item.pb_end and item.pb_start < end_offset) return true;
        }
        return false;
    }

    fn fenceWord(self: *Queue) *volatile u32 {
        return @ptrCast(self.fence.slice(u32).ptr);
    }

    fn userdWord(self: *Queue, offset: usize) *volatile u32 {
        return @ptrCast(@alignCast(self.userd.slice(u8).ptr + offset));
    }
};

test "GPFIFO entry layout" {
    const entry = try GpEntry.init(0x12_3456_7000, 17);
    const raw: u64 = @bitCast(entry);
    try std.testing.expectEqual(@as(u32, 0x3456_7000), @as(u32, @truncate(raw)));
    try std.testing.expectEqual(@as(u32, (17 << 10) | 0x12), @as(u32, @truncate(raw >> 32)));
}

test "GPFIFO entry rejects an address above 40 bits" {
    try std.testing.expectError(error.AddressTooLarge, GpEntry.init(1 << 40, 1));
}
