//! `exp` and `log` for f32. Kernels have no math library, because the nvptx64 target
//! has no libcalls. The host uses these too, so both sides compute the same values.

const ln2_hi: f32 = 6.9314575195e-1;
const ln2_lo: f32 = 1.4286067653e-6;
const inv_ln2: f32 = 1.4426950216;

/// Inputs are clamped to the range where the result is a normal f32.
pub inline fn exp(input: f32) f32 {
    const x = if (input > 88.0) @as(f32, 88.0) else if (input < -87.0) @as(f32, -87.0) else input;
    // x = n * ln2 + r, with |r| <= ln2 / 2.
    const rounded = x * inv_ln2 + (if (x >= 0) @as(f32, 0.5) else @as(f32, -0.5));
    const n: i32 = @intFromFloat(rounded);
    const fn_: f32 = @floatFromInt(n);
    const r = (x - fn_ * ln2_hi) - fn_ * ln2_lo;
    // Taylor polynomial of exp(r), adequate for |r| <= ln2 / 2.
    const p = 1.0 + r * (1.0 + r * (0.5 + r * (1.6666667e-1 + r * (4.1666668e-2 + r * (8.3333338e-3 + r * 1.3888889e-3)))));
    const scale: f32 = @bitCast(@as(u32, @intCast(n + 127)) << 23);
    return p * scale;
}

/// Natural logarithm of a positive, normal f32.
pub inline fn log(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    var e: i32 = @as(i32, @intCast(bits >> 23)) - 127;
    var m: f32 = @bitCast((bits & 0x007f_ffff) | 0x3f80_0000); // In [1, 2).
    if (m > 1.41421356) {
        m *= 0.5;
        e += 1;
    }
    // log(m) = 2 atanh(s), with s = (m - 1) / (m + 1), |s| < 0.172.
    const s = (m - 1.0) / (m + 1.0);
    const z = s * s;
    const series = 1.0 + z * (3.3333334e-1 + z * (2.0e-1 + z * (1.4285715e-1 + z * 1.1111111e-1)));
    const fe: f32 = @floatFromInt(e);
    return 2.0 * s * series + fe * ln2_hi + fe * ln2_lo;
}

const std = @import("std");

test "exp agrees with the standard library" {
    var x: f32 = -80.0;
    while (x < 80.0) : (x += 0.137) {
        const want = @exp(x);
        try std.testing.expectApproxEqRel(want, exp(x), 2e-6);
    }
}

test "exp clamps instead of overflowing" {
    try std.testing.expect(std.math.isFinite(exp(1000.0)));
    try std.testing.expect(exp(-1000.0) > 0.0);
}

test "log agrees with the standard library" {
    var x: f32 = 1e-30;
    while (x < 1e30) : (x *= 1.37) {
        const want = @log(x);
        try std.testing.expectApproxEqAbs(want, log(x), 1e-5 * @max(@abs(want), 1.0));
    }
}
