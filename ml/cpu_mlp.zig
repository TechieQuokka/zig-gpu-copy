//! The CPU model: the same network, seed and update rule as the GPU model, using
//! `@Vector` and one thread. Sums run in the same order as the GPU's, so results
//! can be compared directly.

const std = @import("std");
const kernels = @import("kernels");
const cpu = @import("cpu.zig");
const model = @import("model.zig");
const mnist = @import("mnist.zig");
const abi = kernels.abi;

pub fn CpuMlp(comptime config: model.Config) type {
    comptime config.validate();
    const dense_count = config.denseCount();

    return struct {
        const Self = @This();

        pub const Layer = struct {
            inputs: u32,
            outputs: u32,
            relu: bool,
            weight: []f32,
            bias: []f32,
            d_weight: []f32,
            d_bias: []f32,
            v_weight: []f32,
            v_bias: []f32,
            /// The weight matrix transposed, refreshed for each backward pass.
            transposed: []f32,
            /// Activations and gradient at this layer's output.
            output: []f32,
            grad: []f32,
        };

        allocator: std.mem.Allocator,
        layers: [dense_count]Layer,
        /// The gradient at the input, which backpropagation computes and nothing uses.
        input_grad: []f32,
        metrics: []abi.RowMetrics,

        pub fn init(allocator: std.mem.Allocator, max_rows: u32) !Self {
            var self = Self{
                .allocator = allocator,
                .layers = undefined,
                .input_grad = try allocator.alloc(f32, max_rows * @as(usize, config.layers[0])),
                .metrics = try allocator.alloc(abi.RowMetrics, max_rows),
            };
            var rng = model.Xorshift.init(config.seed);
            for (&self.layers, 0..) |*layer, l| {
                const inputs = config.layers[l];
                const outputs = config.layers[l + 1];
                const params = @as(usize, inputs) * outputs;
                const activations = @as(usize, max_rows) * outputs;
                layer.* = .{
                    .inputs = inputs,
                    .outputs = outputs,
                    .relu = l + 1 < dense_count,
                    .weight = try allocator.alloc(f32, params),
                    .bias = try allocator.alloc(f32, outputs),
                    .d_weight = try allocator.alloc(f32, params),
                    .d_bias = try allocator.alloc(f32, outputs),
                    .v_weight = try allocator.alloc(f32, params),
                    .v_bias = try allocator.alloc(f32, outputs),
                    .transposed = try allocator.alloc(f32, params),
                    .output = try allocator.alloc(f32, activations),
                    .grad = try allocator.alloc(f32, activations),
                };
                model.initWeights(&rng, layer.weight, inputs);
                @memset(layer.bias, 0);
                @memset(layer.v_weight, 0);
                @memset(layer.v_bias, 0);
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.layers) |*layer| {
                for ([_][]f32{
                    layer.weight, layer.bias,       layer.d_weight, layer.d_bias, layer.v_weight,
                    layer.v_bias, layer.transposed, layer.output,   layer.grad,
                }) |slice| self.allocator.free(slice);
            }
            self.allocator.free(self.input_grad);
            self.allocator.free(self.metrics);
        }

        pub fn forward(self: *Self, images: []const f32, rows: u32) void {
            var input = images;
            for (&self.layers) |*layer| {
                for (0..rows) |r| {
                    const out = layer.output[r * layer.outputs..][0..layer.outputs];
                    @memset(out, 0);
                    const in = input[r * layer.inputs..][0..layer.inputs];
                    for (in, 0..) |value, k| cpu.axpy(out, value, layer.weight[k * layer.outputs..][0..layer.outputs]);
                    for (out, layer.bias) |*o, b| {
                        o.* += b;
                        if (layer.relu and o.* <= 0) o.* = 0;
                    }
                }
                input = layer.output;
            }
        }

        fn softmax(self: *Self, labels: []const u32, rows: u32) void {
            const last = &self.layers[dense_count - 1];
            const n = last.outputs;
            for (0..rows) |r| {
                self.metrics[r] = kernels.mlp.softmaxRow(n, last.output[r * n..].ptr, labels[r], last.grad[r * n..].ptr);
            }
        }

        /// Computes gradients of all parameters for the batch.
        pub fn computeGrads(self: *Self, images: []const f32, labels: []const u32, rows: u32) void {
            self.forward(images, rows);
            self.softmax(labels, rows);

            var l: usize = dense_count;
            while (l > 0) {
                l -= 1;
                const layer = &self.layers[l];
                const input: []const f32 = if (l == 0) images else self.layers[l - 1].output;
                const grad_in: []f32 = if (l == 0) self.input_grad else self.layers[l - 1].grad;
                const n = layer.outputs;
                const m = layer.inputs;

                // Turn the gradient at the output into the one before the activation.
                if (layer.relu) {
                    for (layer.grad[0 .. rows * n], layer.output[0 .. rows * n]) |*g, o| {
                        if (o <= 0) g.* = 0;
                    }
                }

                @memset(layer.d_weight, 0);
                @memset(layer.d_bias, 0);
                for (0..m) |i| for (0..n) |j| {
                    layer.transposed[j * m + i] = layer.weight[i * n + j];
                };
                @memset(grad_in[0 .. rows * m], 0);
                for (0..rows) |r| {
                    const g = layer.grad[r * n..][0..n];
                    const in = input[r * m..][0..m];
                    for (in, 0..) |value, i| cpu.axpy(layer.d_weight[i * n..][0..n], value, g);
                    cpu.axpy(layer.d_bias, 1.0, g);
                    const out = grad_in[r * m..][0..m];
                    for (g, 0..) |value, j| cpu.axpy(out, value, layer.transposed[j * m..][0..m]);
                }
                for (layer.d_weight) |*d| d.* /= @as(f32, @floatFromInt(rows));
                for (layer.d_bias) |*d| d.* /= @as(f32, @floatFromInt(rows));
            }
        }

        pub fn applyUpdate(self: *Self) void {
            for (&self.layers) |*layer| {
                sgd(layer.weight, layer.d_weight, layer.v_weight);
                sgd(layer.bias, layer.d_bias, layer.v_bias);
            }
        }

        fn sgd(param: []f32, grad: []const f32, velocity: []f32) void {
            for (param, grad, velocity) |*p, g, *v| {
                v.* = config.momentum * v.* + g;
                p.* -= config.rate * v.*;
            }
        }

        /// Returns the mean loss and the number correct over the rows.
        pub fn summarize(self: *const Self, rows: u32) struct { loss: f64, correct: u64 } {
            var loss: f64 = 0;
            var correct: u64 = 0;
            for (self.metrics[0..rows]) |m| {
                loss += m.loss;
                correct += m.correct;
            }
            return .{ .loss = loss / @as(f64, @floatFromInt(rows)), .correct = correct };
        }

        /// Accuracy on a whole set, with metrics computed in `self.metrics`.
        pub fn evaluate(self: *Self, images: []const f32, labels: []const u32) f64 {
            const rows: u32 = @intCast(labels.len);
            self.forward(images, rows);
            self.softmax(labels, rows);
            const result = self.summarize(rows);
            return @as(f64, @floatFromInt(result.correct)) / @as(f64, @floatFromInt(rows));
        }
    };
}

test "one small training run reduces the loss" {
    const config = model.Config{
        .layers = &.{ 784, 16, 10 },
        .batch = 8,
        .rate = 0.05,
        .momentum = 0.9,
        .seed = 7,
    };
    var net = try CpuMlp(config).init(std.testing.allocator, 8);
    defer net.deinit();

    var images: [8 * 784]f32 = undefined;
    var labels: [8]u32 = undefined;
    var rng = model.Xorshift.init(99);
    for (&images) |*x| x.* = rng.unit();
    for (&labels, 0..) |*y, i| y.* = @intCast(i % 10);

    net.computeGrads(&images, &labels, 8);
    const first = net.summarize(8).loss;
    for (0..30) |_| {
        net.applyUpdate();
        net.computeGrads(&images, &labels, 8);
    }
    try std.testing.expect(net.summarize(8).loss < first * 0.5);
}
