//! zig-gpu-copy: experiments in training a neural network on an NVIDIA GPU from
//! Zig, without libcuda.
//!
//!   probe        Show the GPU, its engines and the classes this program needs.
//!   copy-test    Round trip 64 MiB through VRAM and verify it.
//!   fault-test   Copy to an unmapped address and confirm the fault is reported.
//!   dispatch-test  Run a compute kernel on the GPU and verify its output.
//!   gemm-bench   Check matrix multiplication against the CPU and time it.
//!   train [n]    Train on MNIST for n epochs, on the GPU and then on the CPU.

const std = @import("std");
const gpu = @import("gpu");
const kernels = @import("kernels");
const abi = gpu.abi;

const gemm_bench = @import("gemm_bench.zig");
const train = @import("train.zig");

const Command = enum { probe, @"copy-test", @"fault-test", @"dispatch-test", @"gemm-bench", train };

fn probe(client: *gpu.Client) !void {
    const classes = try client.classList();
    const engines = try client.engines();

    std.debug.print("{d} classes reported. Needed:\n", .{classes.numClasses});
    const needed = [_]struct { class: u32, name: []const u8 }{
        .{ .class = abi.class.usermode, .name = "VOLTA_USERMODE_A" },
        .{ .class = abi.class.channel_gpfifo, .name = "AMPERE_CHANNEL_GPFIFO_A" },
        .{ .class = abi.class.copy, .name = "AMPERE_DMA_COPY_B" },
        .{ .class = abi.class.compute, .name = "AMPERE_COMPUTE_B" },
    };
    for (needed) |n| {
        std.debug.print("  0x{x:0>4} {s}: {s}\n", .{ n.class, n.name, if (classes.has(n.class)) "yes" else "NO" });
    }

    std.debug.print("compute classes reported:", .{});
    for (classes.slice()) |class| {
        if (class & 0xff == 0xc0) std.debug.print(" 0x{x:0>4}", .{class});
    }
    std.debug.print("\nengines:", .{});
    for (engines.slice()) |engine| std.debug.print(" 0x{x}", .{engine});
    std.debug.print("\n", .{});
}

/// Everything a copy-engine experiment needs.
const CopyContext = struct {
    h_vaspace: abi.Handle,
    queue: gpu.copy.Queue,

    fn create(client: *gpu.Client) !CopyContext {
        const h_vaspace = try gpu.memory.allocVaSpace(client);
        errdefer client.free(h_vaspace, client.h_device);
        return .{ .h_vaspace = h_vaspace, .queue = try gpu.copy.Queue.create(client, h_vaspace) };
    }

    fn destroy(self: *CopyContext, client: *gpu.Client) void {
        self.queue.destroy(client);
        client.free(self.h_vaspace, client.h_device);
    }

    fn copy(self: *CopyContext, src: u64, dst: u64, length: u32) !void {
        try self.queue.copy(src, dst, length);
    }
};

fn copyTest(client: *gpu.Client) !void {
    const length: u32 = 64 * 1024 * 1024;

    var ctx = try CopyContext.create(client);
    defer ctx.destroy(client);

    var source = try gpu.Buffer.create(client, ctx.h_vaspace, .sysmem, length);
    defer source.destroy(client);
    var vram = try gpu.Buffer.create(client, ctx.h_vaspace, .vidmem, length);
    defer vram.destroy(client);
    var destination = try gpu.Buffer.create(client, ctx.h_vaspace, .sysmem, length);
    defer destination.destroy(client);

    const src_words = source.slice(u64);
    const dst_words = destination.slice(u64);
    var state = gpu.sys.monotonicNs() | 1;
    for (src_words) |*word| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        word.* = state;
    }
    @memset(dst_words, 0xA5A5_A5A5_A5A5_A5A5);

    // The destination can receive the pattern only by way of VRAM.
    const t0 = gpu.sys.monotonicNs();
    try ctx.copy(source.gpu_va, vram.gpu_va, length);
    const t1 = gpu.sys.monotonicNs();
    try ctx.copy(vram.gpu_va, destination.gpu_va, length);
    const t2 = gpu.sys.monotonicNs();

    printRate("sysmem -> VRAM", length, t1 - t0);
    printRate("VRAM -> sysmem", length, t2 - t1);

    if (std.mem.indexOfDiff(u64, src_words, dst_words)) |at| {
        std.debug.print("error: mismatch at word {d}: 0x{x} != 0x{x}\n", .{ at, src_words[at], dst_words[at] });
        return error.DataMismatch;
    }
    std.debug.print("verified: all {d} bytes match\n", .{length});

    // More submissions than the GPFIFO has entries, to exercise ring reuse.
    const small: u32 = 4096;
    for (0..1000) |_| try ctx.copy(source.gpu_va, destination.gpu_va, small);
    std.debug.print("verified: 1000 submissions through the 64-entry ring\n", .{});
}

/// Deliberately faults this program's own channel. The kernel log gains an Xid.
fn faultTest(client: *gpu.Client) !void {
    var ctx = try CopyContext.create(client);
    defer ctx.destroy(client);

    var source = try gpu.Buffer.create(client, ctx.h_vaspace, .sysmem, 4096);
    defer source.destroy(client);

    const unmapped_va: u64 = 0x7f_0000_0000;
    ctx.copy(source.gpu_va, unmapped_va, 4096) catch |err| switch (err) {
        error.ChannelFault => {
            std.debug.print("verified: the fault was reported through the error notifier\n", .{});
            return;
        },
        else => return err,
    };
    return error.FaultNotReported;
}

const Fill3d = gpu.compute.KernelDef(@embedFile("kernel_fill3d"), "fill3d", kernels.abi.Fill3d);

/// The largest accepted difference between GPU and CPU results. The arithmetic is
/// IEEE on both sides, so anything above rounding differences means a real error.
const max_allowed_ulp: u32 = 2;

/// Distance between two floats in units in the last place.
fn ulpDistance(a: f32, b: f32) u32 {
    const ai: i64 = @as(i32, @bitCast(a));
    const bi: i64 = @as(i32, @bitCast(b));
    return @intCast(@abs(ai - bi));
}

/// Runs a kernel written in Zig on several grid shapes, and compares every element
/// with the same Zig function compiled for the CPU.
fn dispatchTest(client: *gpu.Client) !void {
    const h_vaspace = try gpu.memory.allocVaSpace(client);
    defer client.free(h_vaspace, client.h_device);

    var compute = try gpu.compute.Compute.create(client, h_vaspace);
    defer compute.destroy(client);
    std.debug.print("compute queue on GR0, token 0x{x}\n", .{compute.queue.token});
    std.debug.print("kernel: {d} bytes, {d} registers, parameters at 0x{x}, {d} bytes\n", .{
        Fill3d.kernel_info.code.len,
        Fill3d.kernel_info.register_count,
        Fill3d.kernel_info.param_offset,
        Fill3d.kernel_info.param_size,
    });

    var code = try gpu.compute.loadCode(client, h_vaspace, Fill3d);
    defer code.destroy(client);

    const sizes = [_]gpu.compute.Size{
        .{ .grid = .{ 1, 1, 1 }, .block = .{ 1, 1, 1 } },
        .{ .grid = .{ 5, 3, 7 }, .block = .{ 1, 1, 1 } },
        .{ .grid = .{ 3, 2, 2 }, .block = .{ 4, 2, 2 } },
        .{ .grid = .{ 8, 4, 2 }, .block = .{ 32, 1, 1 } },
        .{ .grid = .{ 64, 64, 8 }, .block = .{ 32, 1, 1 } },
    };
    const poison: u32 = 0xA5A5_A5A5;

    for (sizes) |size| {
        const width = size.grid[0] * size.block[0];
        const height = size.grid[1] * size.block[1];
        const depth = size.grid[2] * size.block[2];
        const count: usize = @as(usize, width) * height * depth;

        // One extra element, so that a write past the end is detected.
        var out = try gpu.Buffer.create(client, h_vaspace, .sysmem, (count + 1) * @sizeOf(f32));
        defer out.destroy(client);
        const raw = out.slice(u32);
        @memset(raw, poison);

        const params = kernels.abi.Fill3d{
            .out = out.gpu_va,
            .width = width,
            .height = height,
            .scale = 0.37,
            .offset = 1.25,
        };
        const t0 = gpu.sys.monotonicNs();
        try compute.queue.wait(try compute.dispatch(Fill3d, &code, params, size));
        const elapsed = gpu.sys.monotonicNs() - t0;

        const values = out.slice(f32);
        var max_ulp: u32 = 0;
        var exact: usize = 0;
        for (0..depth) |z| for (0..height) |y| for (0..width) |x| {
            const i = x + width * (y + height * z);
            if (raw[i] == poison) {
                std.debug.print("error: element ({d},{d},{d}) was not written\n", .{ x, y, z });
                return error.DataMismatch;
            }
            const expected = kernels.fill3d.element(params, @intCast(x), @intCast(y), @intCast(z));
            const distance = ulpDistance(values[i], expected);
            if (distance == 0) exact += 1;
            max_ulp = @max(max_ulp, distance);
        };
        if (raw[count] != poison) return error.WrotePastEnd;
        if (max_ulp > max_allowed_ulp) {
            std.debug.print("error: {d}x{d}x{d}: GPU and CPU differ by {d} ULP\n", .{ width, height, depth, max_ulp });
            return error.DataMismatch;
        }
        std.debug.print("verified: {d}x{d}x{d} = {d} threads, {d:.3} ms, {d} exact, max {d} ULP\n", .{
            width, height, depth, count, @as(f64, @floatFromInt(elapsed)) / 1e6, exact, max_ulp,
        });
    }
    try compute.queue.checkFault();
    std.debug.print("verified: no fault reported on the channel\n", .{});
}

fn printRate(label: []const u8, bytes: u64, ns: u64) void {
    const seconds = @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
    const mib = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
    std.debug.print("  {s}: {d:.3} ms, {d:.0} MiB/s\n", .{ label, seconds * 1000.0, mib / seconds });
}

fn usage() void {
    std.debug.print("usage: zig-gpu-copy <probe|copy-test|fault-test|dispatch-test|gemm-bench|train [epochs]>\n", .{});
}

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.next();

    // Commands are parsed before the device is opened, so that a mistyped
    // command reports usage on any machine.
    const name = args.next() orelse {
        usage();
        return error.MissingCommand;
    };
    const command = std.meta.stringToEnum(Command, name) orelse {
        usage();
        return error.InvalidCommand;
    };

    var client = try gpu.Client.open(null);
    defer client.close();
    std.debug.print("driver {s}, /dev/nvidia{d}, PCI device 0x{x:0>4}\n", .{
        abi.driver_version, client.minor, client.pci_device_id,
    });

    switch (command) {
        .probe => try probe(&client),
        .@"copy-test" => try copyTest(&client),
        .@"fault-test" => try faultTest(&client),
        .@"dispatch-test" => try dispatchTest(&client),
        .@"gemm-bench" => try gemm_bench.run(&client),
        .train => {
            const epochs = if (args.next()) |text| try std.fmt.parseInt(u32, text, 10) else null;
            try train.run(&client, epochs);
        },
    }
}
