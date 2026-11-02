//! Matrix multiplication on the GPU, checked against the CPU and timed.

const std = @import("std");
const gpu = @import("gpu");
const kernels = @import("kernels");
const ml = @import("ml");
const abi = gpu.abi;

const Gemm = gpu.compute.KernelDef(@embedFile("kernel_gemm"), "gemm", kernels.abi.Gemm);

const Shape = struct { rows: u32, cols: u32, inner: u32 };

/// Shapes for correctness, including ones that do not fill the last thread block.
const check_shapes = [_]Shape{
    .{ .rows = 1, .cols = 1, .inner = 1 },
    .{ .rows = 3, .cols = 5, .inner = 7 },
    .{ .rows = 33, .cols = 33, .inner = 33 },
    .{ .rows = 100, .cols = 31, .inner = 65 },
    .{ .rows = 257, .cols = 129, .inner = 513 },
};
const bench_sizes = [_]u32{ 64, 256, 512, 1024, 2048 };

/// The result is the same sum in the same order on both sides, so any difference
/// beyond rounding would be a real error.
const relative_tolerance: f32 = 1e-5;

fn fillRandom(values: []f32, seed: u64) void {
    var state = seed | 1;
    for (values) |*value| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const unit: f32 = @floatFromInt(state >> 40); // 24 bits, exact in f32.
        value.* = unit / (1 << 23) - 1.0;
    }
}

fn msSince(start: u64) f64 {
    return @as(f64, @floatFromInt(gpu.sys.monotonicNs() - start)) / 1e6;
}

const Run = struct { copy_ms: f64, kernel_ms: f64 };

const Bench = struct {
    client: *gpu.Client,
    h_vaspace: abi.Handle,
    compute: *gpu.compute.Compute,
    copier: *gpu.copy.Queue,
    code: *const gpu.Buffer,

    /// Computes staging_c = staging_a * staging_b, with all matrices in VRAM.
    fn run(self: *Bench, s: Shape, a: *const gpu.Buffer, b: *const gpu.Buffer, c: *const gpu.Buffer) !Run {
        const a_len = s.rows * s.inner * @sizeOf(f32);
        const b_len = s.inner * s.cols * @sizeOf(f32);
        const c_len = s.rows * s.cols * @sizeOf(f32);

        var vram_a = try gpu.Buffer.create(self.client, self.h_vaspace, .vidmem, a_len);
        defer vram_a.destroy(self.client);
        var vram_b = try gpu.Buffer.create(self.client, self.h_vaspace, .vidmem, b_len);
        defer vram_b.destroy(self.client);
        var vram_c = try gpu.Buffer.create(self.client, self.h_vaspace, .vidmem, c_len);
        defer vram_c.destroy(self.client);

        const t0 = gpu.sys.monotonicNs();
        try self.copier.copy(a.gpu_va, vram_a.gpu_va, a_len);
        try self.copier.copy(b.gpu_va, vram_b.gpu_va, b_len);
        const upload_ms = msSince(t0);

        const t1 = gpu.sys.monotonicNs();
        const fence = try self.compute.dispatch(Gemm, self.code, .{
            .a = vram_a.gpu_va,
            .b = vram_b.gpu_va,
            .c = vram_c.gpu_va,
            .rows = s.rows,
            .cols = s.cols,
            .inner = s.inner,
        }, .{
            .grid = .{ std.mem.alignForward(u32, s.cols, kernels.gemm.block_x) / kernels.gemm.block_x, s.rows, 1 },
            .block = .{ kernels.gemm.block_x, 1, 1 },
        });
        try self.compute.queue.wait(fence);
        const kernel_ms = msSince(t1);

        const t2 = gpu.sys.monotonicNs();
        try self.copier.copy(vram_c.gpu_va, c.gpu_va, c_len);
        return .{ .copy_ms = upload_ms + msSince(t2), .kernel_ms = kernel_ms };
    }
};

pub fn run(client: *gpu.Client) !void {
    const h_vaspace = try gpu.memory.allocVaSpace(client);
    defer client.free(h_vaspace, client.h_device);

    var compute = try gpu.compute.Compute.create(client, h_vaspace);
    defer compute.destroy(client);
    var copier = try gpu.copy.Queue.create(client, h_vaspace);
    defer copier.destroy(client);
    var code = try gpu.compute.loadCode(client, h_vaspace, Gemm);
    defer code.destroy(client);

    var bench = Bench{ .client = client, .h_vaspace = h_vaspace, .compute = &compute, .copier = &copier, .code = &code };

    std.debug.print("kernel: {d} bytes, {d} registers\n", .{ Gemm.kernel_info.code.len, Gemm.kernel_info.register_count });
    std.debug.print("correctness against the same function on the CPU:\n", .{});
    for (check_shapes) |s| {
        var a = try gpu.Buffer.create(client, h_vaspace, .sysmem, s.rows * s.inner * @sizeOf(f32));
        defer a.destroy(client);
        var b = try gpu.Buffer.create(client, h_vaspace, .sysmem, s.inner * s.cols * @sizeOf(f32));
        defer b.destroy(client);
        var c = try gpu.Buffer.create(client, h_vaspace, .sysmem, s.rows * s.cols * @sizeOf(f32));
        defer c.destroy(client);

        fillRandom(a.slice(f32), 11);
        fillRandom(b.slice(f32), 22);
        @memset(c.slice(u32), 0xA5A5_A5A5);
        _ = try bench.run(s, &a, &b, &c);

        const params = kernels.abi.Gemm{ .a = 0, .b = 0, .c = 0, .rows = s.rows, .cols = s.cols, .inner = s.inner };
        const out = c.slice(f32);
        var exact: usize = 0;
        for (0..s.rows) |r| for (0..s.cols) |col| {
            const expected = kernels.gemm.element(params, a.slice(f32).ptr, b.slice(f32).ptr, @intCast(r), @intCast(col));
            const got = out[r * s.cols + col];
            if (got == expected) exact += 1;
            if (@abs(got - expected) > relative_tolerance * @max(@abs(expected), 1.0)) {
                std.debug.print("error: {d}x{d}x{d}: ({d},{d}) is {d}, expected {d}\n", .{ s.rows, s.cols, s.inner, r, col, got, expected });
                return error.DataMismatch;
            }
        };
        std.debug.print("  {d} x {d} (inner {d}): {d}/{d} exact\n", .{ s.rows, s.cols, s.inner, exact, s.rows * s.cols });
    }

    std.debug.print("square matrices (GPU kernel excludes copies; CPU is one thread):\n", .{});
    std.debug.print("  {s:>5} {s:>12} {s:>12} {s:>12} {s:>8} {s:>12}\n", .{ "n", "GPU ms", "copies ms", "CPU ms", "speedup", "max rel diff" });
    for (bench_sizes) |n| {
        const s = Shape{ .rows = n, .cols = n, .inner = n };
        const bytes = @as(u64, n) * n * @sizeOf(f32);
        var a = try gpu.Buffer.create(client, h_vaspace, .sysmem, bytes);
        defer a.destroy(client);
        var b = try gpu.Buffer.create(client, h_vaspace, .sysmem, bytes);
        defer b.destroy(client);
        var c = try gpu.Buffer.create(client, h_vaspace, .sysmem, bytes);
        defer c.destroy(client);
        var cpu_c = try gpu.Buffer.create(client, h_vaspace, .sysmem, bytes);
        defer cpu_c.destroy(client);

        fillRandom(a.slice(f32), 33);
        fillRandom(b.slice(f32), 44);

        const gpu_run = try bench.run(s, &a, &b, &c);
        const t0 = gpu.sys.monotonicNs();
        ml.cpu.gemm(a.slice(f32), b.slice(f32), cpu_c.slice(f32), n, n, n);
        const cpu_ms = msSince(t0);

        var max_diff: f32 = 0;
        for (c.slice(f32)[0 .. n * n], cpu_c.slice(f32)[0 .. n * n]) |x, y| {
            max_diff = @max(max_diff, @abs(x - y) / @max(@abs(y), 1.0));
        }
        if (max_diff > relative_tolerance) return error.DataMismatch;
        std.debug.print("  {d:>5} {d:>12.3} {d:>12.3} {d:>12.3} {d:>7.1}x {e:>12.2}\n", .{
            n, gpu_run.kernel_ms, gpu_run.copy_ms, cpu_ms, cpu_ms / gpu_run.kernel_ms, max_diff,
        });
    }
    try compute.queue.checkFault();
    std.debug.print("verified: no fault reported on the channel\n", .{});
}
