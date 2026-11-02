//! Compute dispatch on AMPERE_COMPUTE_B (0xC7C0): QMD construction, methods and
//! kernel binding. The QMD layout is from clc7c0qmd.h. The dispatch sequence
//! follows Mesa's NVK driver, which is the public example of driving this class.

const std = @import("std");
const abi = @import("rm/abi.zig");
const sys = @import("rm/sys.zig");
const Client = @import("rm/client.zig").Client;
const memory = @import("memory.zig");
const Buffer = memory.Buffer;
const queue_mod = @import("queue.zig");
const Queue = queue_mod.Queue;
const methods = @import("methods.zig");
const Method = methods.Method;
const kernel_elf = @import("kernel_elf.zig");

pub const Error = queue_mod.Error;

pub const Engine = struct {
    pub const subchannel = 1;

    pub const set_object = Method(Engine, 0x0000, u32);
    // Host methods, handled by the channel for any subchannel (clc56f.h).
    pub const mem_op_a = Method(Engine, 0x0028, u32);
    pub const mem_op_b = Method(Engine, 0x002c, u32);
    pub const mem_op_c = Method(Engine, 0x0030, u32);
    pub const mem_op_d = Method(Engine, 0x0034, u32);

    pub const invalidate_shader_caches = Method(Engine, 0x021c, InvalidateShaderCaches);
    pub const invalidate_sked_caches = Method(Engine, 0x0298, u32);
    pub const set_shared_memory_window = Method(Engine, 0x02a0, u64);
    pub const send_pcas_a = Method(Engine, 0x02b4, u32);
    pub const send_signaling_pcas2_b = Method(Engine, 0x02c0, u32);
    pub const set_local_memory_window = Method(Engine, 0x07b0, u64);
    pub const set_report_semaphore = Method(Engine, 0x1b00, u64);
    pub const set_report_semaphore_c = Method(Engine, 0x1b08, u32);
    pub const set_report_semaphore_d = Method(Engine, 0x1b0c, ReportSemaphoreD);
};

pub const InvalidateShaderCaches = packed struct(u32) {
    instruction: bool = false,
    locks: bool = false,
    flush_data: bool = false,
    reserved0: u1 = 0,
    data: bool = false,
    reserved1: u7 = 0,
    constant: bool = false,
    reserved2: u19 = 0,
};

pub const ReportSemaphoreD = packed struct(u32) {
    operation: u2 = 0, // RELEASE
    flush_disable: bool = false,
    reserved0: u25 = 0,
    one_word: bool = false,
    reserved1: u3 = 0,
};

const pcas_action_invalidate_copy_schedule: u32 = 3;
const mem_op_d_membar: u32 = 5 << 27;
const mem_op_c_sys_membar: u32 = 0;

/// Addresses reserved for shared and local memory windows. They must not collide
/// with buffers, which RM places above 4 GiB. Values as in NVK.
const shared_window: u64 = 0xfe << 24;
const local_window: u64 = 0xff << 24;

pub const ConstantBufferDesc = packed struct(u64) {
    addr_lower: u32 = 0,
    addr_upper: u17 = 0,
    prefetch_post: u1 = 0,
    invalidate: bool = false,
    size_shifted4: u13 = 0,
};

/// QMD version 3.0: the descriptor that tells the scheduler what to run.
pub const Qmd = packed struct(u2048) {
    reserved0: u128 = 0,
    qmd_group_id: u6 = 0,
    sm_global_caching_enable: bool = true,
    reserved1: u51 = 0,
    invalidate_texture_header_cache: bool = false,
    invalidate_texture_sampler_cache: bool = false,
    invalidate_texture_data_cache: bool = false,
    invalidate_shader_data_cache: bool = true,
    invalidate_instruction_cache: bool = true,
    invalidate_shader_constant_cache: bool = true,
    reserved2: u186 = 0,
    api_visible_call_limit_no_check: bool = true,
    reserved3: u3 = 0,
    sampler_index: u1 = 0,
    disable_auto_invalidate: bool = false,
    cta_raster_width: u32 = 0,
    cta_raster_height: u16 = 0,
    reserved4: u16 = 0,
    cta_raster_depth: u16 = 0,
    reserved5: u80 = 0,
    shared_memory_size: u18 = 0,
    min_sm_config_shared_mem_size: u6 = 0,
    reserved6: u1 = 0,
    max_sm_config_shared_mem_size: u6 = 0,
    reserved7: u1 = 0,
    qmd_version: u4 = 0,
    qmd_major_version: u4 = 3,
    reserved8: u8 = 0,
    cta_thread_dimension0: u16 = 0,
    cta_thread_dimension1: u16 = 0,
    cta_thread_dimension2: u16 = 0,
    constant_buffer_valid: u8 = 0,
    register_count: u9 = 0,
    target_sm_config_shared_mem_size: u6 = 0,
    reserved9: u73 = 0,
    shader_local_memory_low_size: u24 = 0,
    reserved10: u3 = 0,
    barrier_count: u5 = 0,
    reserved11: u256 = 0,
    constant_buffer0: ConstantBufferDesc = .{},
    reserved12: u448 = 0,
    program_address_lower: u32 = 0,
    program_address_upper: u17 = 0,
    reserved13: u15 = 0,
    shader_local_memory_high_size: u24 = 0,
    reserved14: u424 = 0,

    pub const words = @bitSizeOf(Qmd) / 32;

    pub fn toWords(self: Qmd) [words]u32 {
        return @bitCast(self);
    }

    comptime {
        // Bit positions from NVC7C0_QMDV03_00_* in clc7c0qmd.h.
        const expected = .{
            .sm_global_caching_enable = 134,
            .invalidate_shader_data_cache = 189,
            .invalidate_instruction_cache = 190,
            .invalidate_shader_constant_cache = 191,
            .api_visible_call_limit_no_check = 378,
            .cta_raster_width = 384,
            .cta_raster_height = 416,
            .cta_raster_depth = 448,
            .shared_memory_size = 544,
            .min_sm_config_shared_mem_size = 562,
            .max_sm_config_shared_mem_size = 569,
            .qmd_version = 576,
            .qmd_major_version = 580,
            .cta_thread_dimension0 = 592,
            .cta_thread_dimension1 = 608,
            .cta_thread_dimension2 = 624,
            .constant_buffer_valid = 640,
            .register_count = 648,
            .target_sm_config_shared_mem_size = 657,
            .shader_local_memory_low_size = 736,
            .barrier_count = 763,
            .constant_buffer0 = 1024,
            .program_address_lower = 1536,
            .program_address_upper = 1568,
            .shader_local_memory_high_size = 1600,
        };
        for (std.meta.fieldNames(@TypeOf(expected))) |name| {
            const want: usize = @field(expected, name);
            if (@bitOffsetOf(Qmd, name) != want) {
                @compileError(std.fmt.comptimePrint("Qmd.{s}: bit {d}, header says {d}", .{
                    name, @bitOffsetOf(Qmd, name), want,
                }));
            }
        }
    }
};

/// Shared memory sizes of compute capability 8.6 are 0, 8, 16, 32, 64 or 100 KiB
/// per SM, encoded as size / 4 + 1. These values select 0 for the minimum and
/// target and 100 KiB for the maximum, for kernels that use no shared memory.
const smem_config_none: u6 = 1;
const smem_config_max: u6 = 26;

/// Constant bank 0 values read by PTX special registers (%ntid, %nctaid).
const CbankLayout = struct {
    const ntid: usize = 0x00;
    const nctaid: usize = 0x0c;
    const size: usize = 1024;
};

/// Grid of thread blocks, and threads per block.
pub const Size = struct {
    grid: [3]u32 = .{ 1, 1, 1 },
    block: [3]u16 = .{ 1, 1, 1 },
};

/// A kernel in an ELF embedded at compile time, with parameters of type `P`. A
/// missing kernel or a parameter mismatch is a compile error.
pub fn KernelDef(comptime elf: []const u8, comptime name: []const u8, comptime P: type) type {
    const info = comptime blk: {
        @setEvalBranchQuota(1_000_000);
        const parsed = kernel_elf.parse(elf, name) catch |err| {
            @compileError("kernel " ++ name ++ ": " ++ @errorName(err));
        };
        const expected = std.mem.alignForward(usize, parsed.param_size, @alignOf(P));
        if (@sizeOf(P) != expected) {
            @compileError(std.fmt.comptimePrint("kernel {s}: parameters are {d} bytes, {s} has {d}", .{
                name, parsed.param_size, @typeName(P), @sizeOf(P),
            }));
        }
        if (parsed.param_offset + parsed.param_size > CbankLayout.size) {
            @compileError("kernel " ++ name ++ ": parameters do not fit constant bank 0");
        }
        break :blk parsed;
    };
    return struct {
        pub const Params = P;
        pub const kernel_info = info;
    };
}

/// Each dispatch uses one slot, holding its QMD and constant bank 0. The queue
/// allows fewer submissions in flight than there are slots, and fences are
/// consecutive, so a slot is free by the time its index comes round again.
const slot_count = 64;
const slot_size: usize = 2048;
const slot_cbank_offset: usize = 1024;

comptime {
    std.debug.assert(@sizeOf(Qmd) <= slot_cbank_offset);
    std.debug.assert(CbankLayout.size <= slot_size - slot_cbank_offset);
}

pub const Compute = struct {
    queue: Queue,
    slots: Buffer,

    pub fn create(client: *Client, h_vaspace: abi.Handle) Error!Compute {
        var queue = try Queue.create(client, h_vaspace, abi.engine.gr0);
        errdefer queue.destroy(client);
        try queue.bind(client, abi.class.compute, null, "alloc compute object");
        try queue.start(client);

        var slots = try Buffer.create(client, h_vaspace, .sysmem, slot_count * slot_size);
        errdefer slots.destroy(client);

        var self = Compute{ .queue = queue, .slots = slots };

        const recording = try self.queue.begin(32);
        var s = methods.Stream(Engine){ .words = recording.words };
        s.put(Engine.set_object, abi.class.compute);
        s.put(Engine.set_shared_memory_window, shared_window);
        s.put(Engine.set_local_memory_window, local_window);
        s.put(Engine.invalidate_sked_caches, 0);
        release(&s, recording);
        try self.queue.wait(try self.queue.submit(recording, s.len));
        return self;
    }

    pub fn destroy(self: *Compute, client: *Client) void {
        self.queue.destroy(client);
        self.slots.destroy(client);
    }

    /// Runs a kernel loaded into `code`. The returned fence passes when every
    /// thread has finished and its writes are visible to the CPU and other
    /// engines.
    pub fn dispatch(self: *Compute, comptime Def: type, code: *const Buffer, params: Def.Params, size: Size) Error!queue_mod.Fence {
        const info = Def.kernel_info;
        const recording = try self.queue.begin(32);

        const slot_index = recording.fence_value % slot_count;
        const slot_va = self.slots.gpu_va + slot_index * slot_size;
        const slot_bytes = self.slots.slice(u8)[slot_index * slot_size..][0..slot_size];

        const cbank = slot_bytes[slot_cbank_offset..][0..CbankLayout.size];
        @memset(cbank, 0);
        writeU32s(cbank[CbankLayout.ntid..], &.{ size.block[0], size.block[1], size.block[2] });
        writeU32s(cbank[CbankLayout.nctaid..], &.{ size.grid[0], size.grid[1], size.grid[2] });
        const param_bytes = std.mem.asBytes(&params);
        @memcpy(cbank[info.param_offset..][0..info.param_size], param_bytes[0..info.param_size]);

        const cbank_va = slot_va + slot_cbank_offset;
        const qmd = Qmd{
            .cta_raster_width = size.grid[0],
            .cta_raster_height = @intCast(size.grid[1]),
            .cta_raster_depth = @intCast(size.grid[2]),
            .cta_thread_dimension0 = size.block[0],
            .cta_thread_dimension1 = size.block[1],
            .cta_thread_dimension2 = size.block[2],
            .min_sm_config_shared_mem_size = smem_config_none,
            .target_sm_config_shared_mem_size = smem_config_none,
            .max_sm_config_shared_mem_size = smem_config_max,
            .register_count = info.register_count,
            .constant_buffer_valid = 1,
            .constant_buffer0 = .{
                .addr_lower = @truncate(cbank_va),
                .addr_upper = @intCast(cbank_va >> 32),
                .size_shifted4 = @intCast(CbankLayout.size >> 4),
            },
            .program_address_lower = @truncate(code.gpu_va),
            .program_address_upper = @intCast(code.gpu_va >> 32),
        };
        const qmd_words = std.mem.bytesAsSlice(u32, slot_bytes[0..@sizeOf(Qmd)]);
        @memcpy(qmd_words, &qmd.toWords());

        var s = methods.Stream(Engine){ .words = recording.words };
        s.put(Engine.send_pcas_a, @intCast(slot_va >> 8));
        s.put(Engine.send_signaling_pcas2_b, pcas_action_invalidate_copy_schedule);
        // This invalidate waits for idle, so the release below follows the kernel.
        s.put(Engine.invalidate_shader_caches, .{ .data = true, .flush_data = true });
        s.put(Engine.mem_op_a, 0);
        s.put(Engine.mem_op_b, 0);
        s.put(Engine.mem_op_c, mem_op_c_sys_membar);
        s.put(Engine.mem_op_d, mem_op_d_membar);
        release(&s, recording);
        return self.queue.submit(recording, s.len);
    }

    fn release(s: *methods.Stream(Engine), recording: queue_mod.Recording) void {
        s.put(Engine.set_report_semaphore, recording.fence_va);
        s.put(Engine.set_report_semaphore_c, recording.fence_value);
        s.put(Engine.set_report_semaphore_d, .{ .one_word = true });
    }
};

/// The instruction cache prefetches past the end of the code, so that memory must
/// be mapped. NVK allocates 2 KiB more on NVIDIA's advice. Without this, a kernel
/// ending near a page boundary faults (Xid 31) only when execution nears its end,
/// such as in a division slow path, and the fault arrives after its fence passes.
const code_padding: usize = 4096;

/// Copies kernel code into a GPU-visible buffer.
pub fn loadCode(client: *Client, h_vaspace: abi.Handle, comptime Def: type) sys.Error!Buffer {
    var code = try Buffer.create(client, h_vaspace, .sysmem, Def.kernel_info.code.len + code_padding);
    @memcpy(code.slice(u8)[0..Def.kernel_info.code.len], Def.kernel_info.code);
    return code;
}

fn writeU32s(out: []u8, values: []const u32) void {
    for (values, 0..) |value, i| std.mem.writeInt(u32, out[i * 4..][0..4], value, .little);
}

test "QMD words place fields where the header says" {
    const qmd = Qmd{
        .register_count = 8,
        .program_address_lower = 0x2000_0100,
        .program_address_upper = 0x1,
    };
    const words = qmd.toWords();
    try std.testing.expectEqual(@as(usize, 64), words.len);
    // Bit 580: major version 3 in word 18, bits 4-7.
    try std.testing.expectEqual(@as(u32, 3 << 4), words[18] & 0xff);
    try std.testing.expectEqual(@as(u32, 8 << 8), words[20] & (0x1ff << 8));
    try std.testing.expectEqual(@as(u32, 0x2000_0100), words[48]);
    try std.testing.expectEqual(@as(u32, 1), words[49]);
}

test "the semaphore release encodes as a one-word flushed release" {
    try std.testing.expectEqual(@as(u32, 0x1000_0000), @as(u32, @bitCast(ReportSemaphoreD{ .one_word = true })));
}
