//! Matrix multiplication. One thread computes one element of the result.

const abi = @import("abi.zig");
const rt = @import("rt.zig");

/// Threads per block in x. A grid covers the result in whole blocks, so threads
/// beyond the last column do nothing.
pub const block_x: u16 = 32;

/// One element of the result. The host uses this same function to check the GPU.
pub fn element(p: abi.Gemm, a: [*]const f32, b: [*]const f32, row: u32, col: u32) f32 {
    const a_row = a + @as(usize, row) * p.inner;
    var sum: f32 = 0;
    var k: u32 = 0;
    while (k < p.inner) : (k += 1) {
        sum += a_row[k] * b[@as(usize, k) * p.cols + col];
    }
    return sum;
}

pub const gpu_entry = if (rt.is_gpu) struct {
    export fn gemm(p: abi.Gemm) callconv(.kernel) void {
        const id = rt.globalId();
        const col = id[0];
        const row = id[1];
        if (col >= p.cols or row >= p.rows) return;
        const a: [*]const f32 = @ptrFromInt(p.a);
        const b: [*]const f32 = @ptrFromInt(p.b);
        const c: [*]f32 = @ptrFromInt(p.c);
        c[@as(usize, row) * p.cols + col] = element(p, a, b, row, col);
    }
} else struct {};

comptime {
    _ = gpu_entry;
}
