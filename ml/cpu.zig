//! CPU implementations used as a baseline. They use `@Vector`, so the comparison
//! is against reasonable single-threaded SIMD code, not a naive loop.

const std = @import("std");

const lanes = std.simd.suggestVectorLength(f32) orelse 4;
const V = @Vector(lanes, f32);

/// y[i] += a * x[i]
pub inline fn axpy(y: []f32, a: f32, x: []const f32) void {
    const splat_a: V = @splat(a);
    var i: usize = 0;
    while (i + lanes <= y.len) : (i += lanes) {
        const vx: V = x[i..][0..lanes].*;
        const vy: V = y[i..][0..lanes].*;
        y[i..][0..lanes].* = vy + splat_a * vx;
    }
    while (i < y.len) : (i += 1) y[i] += a * x[i];
}

/// c = a * b, for row-major matrices of rows x inner, inner x cols and rows x cols.
pub fn gemm(a: []const f32, b: []const f32, c: []f32, rows: usize, cols: usize, inner: usize) void {
    std.debug.assert(a.len >= rows * inner and b.len >= inner * cols and c.len >= rows * cols);
    for (0..rows) |r| {
        const c_row = c[r * cols..][0..cols];
        @memset(c_row, 0);
        const a_row = a[r * inner..][0..inner];
        for (a_row, 0..) |value, k| axpy(c_row, value, b[k * cols..][0..cols]);
    }
}

test "gemm matches a hand-computed product" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2 x 3
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3 x 2
    var c: [4]f32 = undefined;
    gemm(&a, &b, &c, 2, 2, 3);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &c);
}

test "gemm handles a width that does not fill a vector" {
    const n = lanes + 3;
    var a: [n * n]f32 = @splat(0);
    var b: [n * n]f32 = undefined;
    var c: [n * n]f32 = undefined;
    for (0..n) |i| a[i * n + i] = 1; // Identity.
    for (&b, 0..) |*value, i| value.* = @floatFromInt(i);
    gemm(&a, &b, &c, n, n, n);
    try std.testing.expectEqualSlices(f32, &b, &c);
}
