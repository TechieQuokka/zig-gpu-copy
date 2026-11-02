//! The computations of a multilayer perceptron with hand-derived backpropagation.
//! Each pure function computes one element, and is used by the GPU entry points
//! below and by the host to check them. Matrices are row-major.

const abi = @import("abi.zig");
const math = @import("math.zig");
const rt = @import("rt.zig");

/// Threads per block in x. Grids cover their range in whole blocks, so entry
/// points ignore threads beyond the end.
pub const block_x: u16 = 32;

pub fn forwardElement(p: abi.DenseForward, input: [*]const f32, weight: [*]const f32, bias: [*]const f32, row: u32, col: u32) f32 {
    const in_row = input + @as(usize, row) * p.inner;
    var sum: f32 = 0;
    var k: u32 = 0;
    while (k < p.inner) : (k += 1) {
        sum += in_row[k] * weight[@as(usize, k) * p.cols + col];
    }
    sum += bias[col];
    return if (p.relu != 0 and sum <= 0) 0 else sum;
}

/// The gradient before the activation function, which is zero where relu is off.
inline fn preActivation(relu: u32, output: f32, grad: f32) f32 {
    return if (relu != 0 and output <= 0) 0 else grad;
}

/// Writes softmax - one_hot(label) for one row, and returns that row's metrics.
/// The maximum is subtracted before `exp` so that large logits cannot overflow.
pub fn softmaxRow(classes: u32, logits: [*]const f32, label: u32, grad: [*]f32) abi.RowMetrics {
    var max = logits[0];
    var best: u32 = 0;
    var k: u32 = 1;
    while (k < classes) : (k += 1) {
        if (logits[k] > max) {
            max = logits[k];
            best = k;
        }
    }
    var sum: f32 = 0;
    k = 0;
    while (k < classes) : (k += 1) sum += math.exp(logits[k] - max);
    k = 0;
    while (k < classes) : (k += 1) {
        const probability = math.exp(logits[k] - max) / sum;
        grad[k] = if (k == label) probability - 1.0 else probability;
    }
    return .{
        .loss = math.log(sum) - (logits[label] - max),
        .correct = if (best == label) 1 else 0,
    };
}

/// d(mean loss) / d(weight[i][j]).
pub fn gradWeightElement(p: abi.DenseBackward, input: [*]const f32, output: [*]const f32, grad_output: [*]const f32, i: u32, j: u32) f32 {
    var sum: f32 = 0;
    var r: u32 = 0;
    while (r < p.rows) : (r += 1) {
        const at = @as(usize, r) * p.cols + j;
        sum += input[@as(usize, r) * p.inner + i] * preActivation(p.relu, output[at], grad_output[at]);
    }
    return sum / @as(f32, @floatFromInt(p.rows));
}

/// d(mean loss) / d(bias[j]).
pub fn gradBiasElement(p: abi.DenseBackward, output: [*]const f32, grad_output: [*]const f32, j: u32) f32 {
    var sum: f32 = 0;
    var r: u32 = 0;
    while (r < p.rows) : (r += 1) {
        const at = @as(usize, r) * p.cols + j;
        sum += preActivation(p.relu, output[at], grad_output[at]);
    }
    return sum / @as(f32, @floatFromInt(p.rows));
}

/// The gradient at element i of the input, for one row.
pub fn gradInputElement(p: abi.DenseBackward, output: [*]const f32, grad_output: [*]const f32, weight: [*]const f32, row: u32, i: u32) f32 {
    var sum: f32 = 0;
    var j: u32 = 0;
    while (j < p.cols) : (j += 1) {
        const at = @as(usize, row) * p.cols + j;
        sum += preActivation(p.relu, output[at], grad_output[at]) * weight[@as(usize, i) * p.cols + j];
    }
    return sum;
}

pub const gpu_entry = if (rt.is_gpu) struct {
    export fn dense_forward(p: abi.DenseForward) callconv(.kernel) void {
        const id = rt.globalId();
        const col = id[0];
        const row = id[1];
        if (col >= p.cols or row >= p.rows) return;
        const out: [*]f32 = @ptrFromInt(p.output);
        out[@as(usize, row) * p.cols + col] = forwardElement(p, @ptrFromInt(p.input), @ptrFromInt(p.weight), @ptrFromInt(p.bias), row, col);
    }

    export fn softmax_grad(p: abi.SoftmaxGrad) callconv(.kernel) void {
        const row = rt.globalId()[0];
        if (row >= p.rows) return;
        const logits: [*]const f32 = @ptrFromInt(p.logits);
        const labels: [*]const u32 = @ptrFromInt(p.labels);
        const grad: [*]f32 = @ptrFromInt(p.grad);
        const metrics: [*]abi.RowMetrics = @ptrFromInt(p.metrics);
        const at = @as(usize, row) * p.classes;
        metrics[row] = softmaxRow(p.classes, logits + at, labels[row], grad + at);
    }

    export fn grad_weight(p: abi.DenseBackward) callconv(.kernel) void {
        const id = rt.globalId();
        const j = id[0];
        const i = id[1];
        if (j >= p.cols or i >= p.inner) return;
        const out: [*]f32 = @ptrFromInt(p.result);
        out[@as(usize, i) * p.cols + j] = gradWeightElement(p, @ptrFromInt(p.input), @ptrFromInt(p.output), @ptrFromInt(p.grad_output), i, j);
    }

    export fn grad_bias(p: abi.DenseBackward) callconv(.kernel) void {
        const j = rt.globalId()[0];
        if (j >= p.cols) return;
        const out: [*]f32 = @ptrFromInt(p.result);
        out[j] = gradBiasElement(p, @ptrFromInt(p.output), @ptrFromInt(p.grad_output), j);
    }

    export fn grad_input(p: abi.DenseBackward) callconv(.kernel) void {
        const id = rt.globalId();
        const i = id[0];
        const row = id[1];
        if (i >= p.inner or row >= p.rows) return;
        const out: [*]f32 = @ptrFromInt(p.result);
        out[@as(usize, row) * p.inner + i] = gradInputElement(p, @ptrFromInt(p.output), @ptrFromInt(p.grad_output), @ptrFromInt(p.weight), row, i);
    }

    export fn sgd_step(p: abi.SgdStep) callconv(.kernel) void {
        const i = rt.globalId()[0];
        if (i >= p.count) return;
        const param: [*]f32 = @ptrFromInt(p.param);
        const grad: [*]const f32 = @ptrFromInt(p.grad);
        const velocity: [*]f32 = @ptrFromInt(p.velocity);
        const v = p.momentum * velocity[i] + grad[i];
        velocity[i] = v;
        param[i] -= p.rate * v;
    }
} else struct {};

comptime {
    _ = gpu_entry;
}

const std = @import("std");

test "softmax gradient sums to zero and metrics are right" {
    const logits = [_]f32{ 1, 3, 2 };
    var grad: [3]f32 = undefined;
    const m = softmaxRow(3, &logits, 1, &grad);
    try std.testing.expectEqual(@as(u32, 1), m.correct);
    try std.testing.expectApproxEqAbs(@as(f32, 0), grad[0] + grad[1] + grad[2], 1e-6);
    // -log(exp(3) / (exp(1) + exp(3) + exp(2))) = 0.40761
    try std.testing.expectApproxEqAbs(@as(f32, 0.40761), m.loss, 1e-4);
    try std.testing.expect(grad[1] < 0);
}

test "softmax does not overflow on large logits" {
    const logits = [_]f32{ 5000, 0, -5000 };
    var grad: [3]f32 = undefined;
    const m = softmaxRow(3, &logits, 1, &grad);
    try std.testing.expectEqual(@as(u32, 0), m.correct);
    try std.testing.expect(std.math.isFinite(m.loss));
    try std.testing.expect(std.math.isFinite(grad[0]));
}

test "backward gradient matches a numerical gradient" {
    // One dense layer into softmax, with 2 rows, 3 inputs and 2 classes.
    const input = [_]f32{ 0.5, -1.0, 2.0, 1.5, 0.25, -0.5 };
    var weight = [_]f32{ 0.1, -0.2, 0.3, 0.4, -0.5, 0.6 };
    const bias = [_]f32{ 0.05, -0.05 };
    const labels = [_]u32{ 1, 0 };
    const f = abi.DenseForward{ .input = 0, .weight = 0, .bias = 0, .output = 0, .rows = 2, .cols = 2, .inner = 3, .relu = 0 };
    const b = abi.DenseBackward{ .input = 0, .output = 0, .grad_output = 0, .weight = 0, .result = 0, .rows = 2, .cols = 2, .inner = 3, .relu = 0 };

    const Loss = struct {
        fn of(fw: abi.DenseForward, in: []const f32, w: []const f32, bi: []const f32, l: []const u32) f64 {
            var out: [4]f32 = undefined;
            var g: [4]f32 = undefined;
            var total: f64 = 0;
            for (0..2) |r| {
                for (0..2) |c| out[r * 2 + c] = forwardElement(fw, in.ptr, w.ptr, bi.ptr, @intCast(r), @intCast(c));
                total += softmaxRow(2, out[r * 2..].ptr, l[r], g[r * 2..].ptr).loss;
            }
            return total / 2.0;
        }
    };

    var out: [4]f32 = undefined;
    var grad_out: [4]f32 = undefined;
    for (0..2) |r| {
        for (0..2) |c| out[r * 2 + c] = forwardElement(f, &input, &weight, &bias, @intCast(r), @intCast(c));
        _ = softmaxRow(2, out[r * 2..].ptr, labels[r], grad_out[r * 2..].ptr);
    }

    const h: f32 = 1e-2;
    for (0..3) |i| for (0..2) |j| {
        const analytic = gradWeightElement(b, &input, &out, &grad_out, @intCast(i), @intCast(j));
        const at = i * 2 + j;
        const saved = weight[at];
        weight[at] = saved + h;
        const plus = Loss.of(f, &input, &weight, &bias, &labels);
        weight[at] = saved - h;
        const minus = Loss.of(f, &input, &weight, &bias, &labels);
        weight[at] = saved;
        const numeric: f32 = @floatCast((plus - minus) / (2.0 * h));
        try std.testing.expectApproxEqAbs(numeric, analytic, 1e-3);
    };
}
