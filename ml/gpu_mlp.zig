//! The GPU model. Parameters and activations live in VRAM, and shapes come from the
//! compile-time configuration. One batch is a sequence of dispatches, each of which
//! waits for the one before, so the CPU waits only at the end of the batch.

const std = @import("std");
const gpu = @import("gpu");
const kernels = @import("kernels");
const model = @import("model.zig");
const abi = kernels.abi;
const Buffer = gpu.Buffer;

const elf = @embedFile("kernel_mlp");
const DenseForward = gpu.compute.KernelDef(elf, "dense_forward", abi.DenseForward);
const SoftmaxGrad = gpu.compute.KernelDef(elf, "softmax_grad", abi.SoftmaxGrad);
const GradWeight = gpu.compute.KernelDef(elf, "grad_weight", abi.DenseBackward);
const GradBias = gpu.compute.KernelDef(elf, "grad_bias", abi.DenseBackward);
const GradInput = gpu.compute.KernelDef(elf, "grad_input", abi.DenseBackward);
const SgdStep = gpu.compute.KernelDef(elf, "sgd_step", abi.SgdStep);

/// A grid of whole thread blocks covering `x` columns and `y` rows.
fn cover(x: u32, y: u32) gpu.compute.Size {
    const block = kernels.mlp.block_x;
    return .{
        .grid = .{ std.mem.alignForward(u32, x, block) / block, y, 1 },
        .block = .{ block, 1, 1 },
    };
}

pub fn GpuMlp(comptime config: model.Config) type {
    comptime config.validate();
    const dense_count = config.denseCount();

    return struct {
        const Self = @This();

        pub const Layer = struct {
            inputs: u32,
            outputs: u32,
            relu: bool,
            weight: Buffer,
            bias: Buffer,
            d_weight: Buffer,
            d_bias: Buffer,
            v_weight: Buffer,
            v_bias: Buffer,
            output: Buffer,
            grad: Buffer,

            fn buffers(self: *Layer) [8]*Buffer {
                return .{
                    &self.weight, &self.bias,   &self.d_weight, &self.d_bias,
                    &self.v_weight, &self.v_bias, &self.output,   &self.grad,
                };
            }
        };

        const Code = struct {
            forward: Buffer,
            softmax: Buffer,
            grad_weight: Buffer,
            grad_bias: Buffer,
            grad_input: Buffer,
            sgd: Buffer,
        };

        client: *gpu.Client,
        h_vaspace: gpu.abi.Handle,
        compute: *gpu.compute.Compute,
        copier: *gpu.copy.Queue,
        code: Code,
        layers: [dense_count]Layer,
        /// The gradient at the input, which backpropagation computes and nothing uses.
        input_grad: Buffer,
        /// In system memory, so that the CPU can read it directly.
        metrics: Buffer,

        pub fn init(
            client: *gpu.Client,
            h_vaspace: gpu.abi.Handle,
            compute: *gpu.compute.Compute,
            copier: *gpu.copy.Queue,
            max_rows: u32,
        ) !Self {
            var self: Self = undefined;
            self.client = client;
            self.h_vaspace = h_vaspace;
            self.compute = compute;
            self.copier = copier;
            self.code = .{
                .forward = try gpu.compute.loadCode(client, h_vaspace, DenseForward),
                .softmax = try gpu.compute.loadCode(client, h_vaspace, SoftmaxGrad),
                .grad_weight = try gpu.compute.loadCode(client, h_vaspace, GradWeight),
                .grad_bias = try gpu.compute.loadCode(client, h_vaspace, GradBias),
                .grad_input = try gpu.compute.loadCode(client, h_vaspace, GradInput),
                .sgd = try gpu.compute.loadCode(client, h_vaspace, SgdStep),
            };
            self.input_grad = try Buffer.create(client, h_vaspace, .vidmem, @as(u64, max_rows) * config.layers[0] * @sizeOf(f32));
            self.metrics = try Buffer.create(client, h_vaspace, .sysmem, @as(u64, max_rows) * @sizeOf(abi.RowMetrics));

            var rng = model.Xorshift.init(config.seed);
            for (&self.layers, 0..) |*layer, l| {
                const inputs = config.layers[l];
                const outputs = config.layers[l + 1];
                const param_bytes = @as(u64, inputs) * outputs * @sizeOf(f32);
                const bias_bytes = @as(u64, outputs) * @sizeOf(f32);
                const activation_bytes = @as(u64, max_rows) * outputs * @sizeOf(f32);
                layer.* = .{
                    .inputs = inputs,
                    .outputs = outputs,
                    .relu = l + 1 < dense_count,
                    .weight = try Buffer.create(client, h_vaspace, .vidmem, param_bytes),
                    .bias = try Buffer.create(client, h_vaspace, .vidmem, bias_bytes),
                    .d_weight = try Buffer.create(client, h_vaspace, .vidmem, param_bytes),
                    .d_bias = try Buffer.create(client, h_vaspace, .vidmem, bias_bytes),
                    .v_weight = try Buffer.create(client, h_vaspace, .vidmem, param_bytes),
                    .v_bias = try Buffer.create(client, h_vaspace, .vidmem, bias_bytes),
                    .output = try Buffer.create(client, h_vaspace, .vidmem, activation_bytes),
                    .grad = try Buffer.create(client, h_vaspace, .vidmem, activation_bytes),
                };

                // The same generator and order as the CPU model.
                var staging = try Buffer.create(client, h_vaspace, .sysmem, param_bytes);
                defer staging.destroy(client);
                model.initWeights(&rng, staging.slice(f32)[0 .. inputs * outputs], inputs);
                try copier.copy(staging.gpu_va, layer.weight.gpu_va, @intCast(param_bytes));

                @memset(staging.slice(f32), 0);
                try copier.copy(staging.gpu_va, layer.v_weight.gpu_va, @intCast(param_bytes));
                var zero = try Buffer.create(client, h_vaspace, .sysmem, bias_bytes);
                defer zero.destroy(client);
                @memset(zero.slice(f32), 0);
                try copier.copy(zero.gpu_va, layer.bias.gpu_va, @intCast(bias_bytes));
                try copier.copy(zero.gpu_va, layer.v_bias.gpu_va, @intCast(bias_bytes));
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.layers) |*layer| {
                for (layer.buffers()) |buffer| buffer.destroy(self.client);
            }
            self.metrics.destroy(self.client);
            self.input_grad.destroy(self.client);
            for ([_]*Buffer{
                &self.code.forward,    &self.code.softmax,    &self.code.grad_weight,
                &self.code.grad_bias, &self.code.grad_input, &self.code.sgd,
            }) |buffer| buffer.destroy(self.client);
        }

        fn forward(self: *Self, images_va: u64, rows: u32) !void {
            var input = images_va;
            for (&self.layers) |*layer| {
                _ = try self.compute.dispatch(DenseForward, &self.code.forward, .{
                    .input = input,
                    .weight = layer.weight.gpu_va,
                    .bias = layer.bias.gpu_va,
                    .output = layer.output.gpu_va,
                    .rows = rows,
                    .cols = layer.outputs,
                    .inner = layer.inputs,
                    .relu = @intFromBool(layer.relu),
                }, cover(layer.outputs, rows));
                input = layer.output.gpu_va;
            }
        }

        fn softmax(self: *Self, labels_va: u64, rows: u32) !gpu.queue.Fence {
            const last = &self.layers[dense_count - 1];
            return self.compute.dispatch(SoftmaxGrad, &self.code.softmax, .{
                .logits = last.output.gpu_va,
                .labels = labels_va,
                .grad = last.grad.gpu_va,
                .metrics = self.metrics.gpu_va,
                .rows = rows,
                .classes = last.outputs,
            }, cover(rows, 1));
        }

        /// Records the gradients of all parameters, then the update if asked. The
        /// returned fence passes when everything has finished.
        fn recordStep(self: *Self, images_va: u64, labels_va: u64, rows: u32, update: bool) !gpu.queue.Fence {
            try self.forward(images_va, rows);
            var fence = try self.softmax(labels_va, rows);

            var l: usize = dense_count;
            while (l > 0) {
                l -= 1;
                const layer = &self.layers[l];
                const input = if (l == 0) images_va else self.layers[l - 1].output.gpu_va;
                const grad_in = if (l == 0) self.input_grad.gpu_va else self.layers[l - 1].grad.gpu_va;
                const params = abi.DenseBackward{
                    .input = input,
                    .output = layer.output.gpu_va,
                    .grad_output = layer.grad.gpu_va,
                    .weight = layer.weight.gpu_va,
                    .result = undefined,
                    .rows = rows,
                    .cols = layer.outputs,
                    .inner = layer.inputs,
                    .relu = @intFromBool(layer.relu),
                };
                var p = params;
                p.result = layer.d_weight.gpu_va;
                fence = try self.compute.dispatch(GradWeight, &self.code.grad_weight, p, cover(layer.outputs, layer.inputs));
                p.result = layer.d_bias.gpu_va;
                fence = try self.compute.dispatch(GradBias, &self.code.grad_bias, p, cover(layer.outputs, 1));
                p.result = grad_in;
                fence = try self.compute.dispatch(GradInput, &self.code.grad_input, p, cover(layer.inputs, rows));
            }

            if (update) {
                for (&self.layers) |*layer| {
                    fence = try self.sgd(&layer.weight, &layer.d_weight, &layer.v_weight, layer.inputs * layer.outputs);
                    fence = try self.sgd(&layer.bias, &layer.d_bias, &layer.v_bias, layer.outputs);
                }
            }
            return fence;
        }

        fn sgd(self: *Self, param: *const Buffer, grad: *const Buffer, velocity: *const Buffer, count: u32) !gpu.queue.Fence {
            return self.compute.dispatch(SgdStep, &self.code.sgd, .{
                .param = param.gpu_va,
                .grad = grad.gpu_va,
                .velocity = velocity.gpu_va,
                .count = count,
                .rate = config.rate,
                .momentum = config.momentum,
            }, cover(count, 1));
        }

        /// Computes gradients without updating, for comparison with the CPU model.
        pub fn computeGrads(self: *Self, images_va: u64, labels_va: u64, rows: u32) !void {
            try self.compute.queue.wait(try self.recordStep(images_va, labels_va, rows, false));
        }

        /// One training step. Metrics for the rows are readable afterwards.
        pub fn step(self: *Self, images_va: u64, labels_va: u64, rows: u32) !void {
            try self.compute.queue.wait(try self.recordStep(images_va, labels_va, rows, true));
        }

        pub fn summarize(self: *const Self, rows: u32) struct { loss: f64, correct: u64 } {
            var loss: f64 = 0;
            var correct: u64 = 0;
            for (self.metrics.slice(abi.RowMetrics)[0..rows]) |m| {
                loss += m.loss;
                correct += m.correct;
            }
            return .{ .loss = loss / @as(f64, @floatFromInt(rows)), .correct = correct };
        }

        /// Accuracy on a whole set in VRAM.
        pub fn evaluate(self: *Self, images_va: u64, labels_va: u64, rows: u32) !f64 {
            try self.forward(images_va, rows);
            try self.compute.queue.wait(try self.softmax(labels_va, rows));
            const result = self.summarize(rows);
            return @as(f64, @floatFromInt(result.correct)) / @as(f64, @floatFromInt(rows));
        }

        /// Copies a VRAM buffer to a new system memory buffer, for the CPU to read.
        pub fn readBack(self: *Self, source: *const Buffer, bytes: u64) !Buffer {
            var out = try Buffer.create(self.client, self.h_vaspace, .sysmem, bytes);
            errdefer out.destroy(self.client);
            try self.copier.copy(source.gpu_va, out.gpu_va, @intCast(bytes));
            return out;
        }
    };
}
