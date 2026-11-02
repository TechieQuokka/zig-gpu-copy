//! Trains the same network on the GPU and the CPU, after checking on one batch that
//! both compute the same gradients.

const std = @import("std");
const gpu = @import("gpu");
const ml = @import("ml");
const mnist = ml.mnist;
const model = ml.model;

const config = model.mnist_mlp;
const Gpu = ml.GpuMlp(config);
const Cpu = ml.CpuMlp(config);

const default_epochs: u32 = 10;
const target_accuracy: f64 = 0.97;
/// The largest accepted difference between GPU and CPU gradients.
const gradient_tolerance: f32 = 1e-4;

fn secondsSince(start: u64) f64 {
    return @as(f64, @floatFromInt(gpu.sys.monotonicNs() - start)) / 1e9;
}

/// A data set in system memory and in VRAM, in the current epoch's order.
const DataOnGpu = struct {
    images_staging: gpu.Buffer,
    labels_staging: gpu.Buffer,
    images: gpu.Buffer,
    labels: gpu.Buffer,
    rows: u32,

    fn create(client: *gpu.Client, h_vaspace: gpu.abi.Handle, rows: u32) !DataOnGpu {
        const image_bytes = @as(u64, rows) * mnist.image_pixels * @sizeOf(f32);
        const label_bytes = @as(u64, rows) * @sizeOf(u32);
        return .{
            .images_staging = try gpu.Buffer.create(client, h_vaspace, .sysmem, image_bytes),
            .labels_staging = try gpu.Buffer.create(client, h_vaspace, .sysmem, label_bytes),
            .images = try gpu.Buffer.create(client, h_vaspace, .vidmem, image_bytes),
            .labels = try gpu.Buffer.create(client, h_vaspace, .vidmem, label_bytes),
            .rows = rows,
        };
    }

    fn destroy(self: *DataOnGpu, client: *gpu.Client) void {
        self.images.destroy(client);
        self.labels.destroy(client);
        self.images_staging.destroy(client);
        self.labels_staging.destroy(client);
    }

    fn fill(self: *DataOnGpu, set: mnist.Set, order: []const u32) void {
        model.gather(set, order, self.images_staging.slice(f32), self.labels_staging.slice(u32));
    }

    fn upload(self: *DataOnGpu, copier: *gpu.copy.Queue) !void {
        const rows: u64 = self.rows;
        try copier.copy(self.images_staging.gpu_va, self.images.gpu_va, @intCast(rows * mnist.image_pixels * @sizeOf(f32)));
        try copier.copy(self.labels_staging.gpu_va, self.labels.gpu_va, @intCast(rows * @sizeOf(u32)));
    }

    fn imagesVa(self: *const DataOnGpu, row: u32) u64 {
        return self.images.gpu_va + @as(u64, row) * mnist.image_pixels * @sizeOf(f32);
    }

    fn labelsVa(self: *const DataOnGpu, row: u32) u64 {
        return self.labels.gpu_va + @as(u64, row) * @sizeOf(u32);
    }
};

fn maxDifference(a: []const f32, b: []const f32) f32 {
    var worst: f32 = 0;
    for (a, b) |x, y| worst = @max(worst, @abs(x - y));
    return worst;
}

/// Compares the gradient of every parameter on one batch. Returns an error on a
/// real disagreement, before any time is spent on training.
fn checkGradients(net: *Gpu, cpu_net: *Cpu, data: *const DataOnGpu) !void {
    const rows = config.batch;
    net.computeGrads(data.imagesVa(0), data.labelsVa(0), rows) catch |err| return err;
    cpu_net.computeGrads(data.images_staging.slice(f32), data.labels_staging.slice(u32), rows);

    std.debug.print("gradients on one batch, GPU against CPU:\n", .{});
    for (&net.layers, &cpu_net.layers, 0..) |*g, *c, l| {
        var d_weight = try net.readBack(&g.d_weight, c.d_weight.len * @sizeOf(f32));
        defer d_weight.destroy(net.client);
        var d_bias = try net.readBack(&g.d_bias, c.d_bias.len * @sizeOf(f32));
        defer d_bias.destroy(net.client);

        const weight_diff = maxDifference(d_weight.slice(f32)[0..c.d_weight.len], c.d_weight);
        const bias_diff = maxDifference(d_bias.slice(f32)[0..c.d_bias.len], c.d_bias);
        std.debug.print("  layer {d}: weight max diff {e:.2}, bias max diff {e:.2}\n", .{ l, weight_diff, bias_diff });
        if (weight_diff > gradient_tolerance or bias_diff > gradient_tolerance) return error.GradientMismatch;
    }
    const gpu_loss = net.summarize(rows).loss;
    const cpu_loss = cpu_net.summarize(rows).loss;
    std.debug.print("  loss: GPU {d:.6}, CPU {d:.6}\n", .{ gpu_loss, cpu_loss });
    if (@abs(gpu_loss - cpu_loss) > 1e-3) return error.LossMismatch;
}

const Totals = struct { seconds: f64 = 0, accuracy: f64 = 0 };

fn trainGpu(net: *Gpu, copier: *gpu.copy.Queue, train: *DataOnGpu, test_data: *DataOnGpu, set: mnist.Set, order: []u32, epochs: u32) !Totals {
    var totals = Totals{};
    var rng = model.Xorshift.init(config.seed ^ 0xABCD);
    const batches: u32 = train.rows / config.batch;
    std.debug.print("GPU training, {d} batches per epoch:\n", .{batches});

    for (0..epochs) |epoch| {
        model.shuffle(&rng, order);
        train.fill(set, order);
        const t_copy = gpu.sys.monotonicNs();
        try train.upload(copier);
        const copy_seconds = secondsSince(t_copy);

        var loss: f64 = 0;
        var correct: u64 = 0;
        const t0 = gpu.sys.monotonicNs();
        for (0..batches) |b| {
            const row: u32 = @intCast(b * config.batch);
            try net.step(train.imagesVa(row), train.labelsVa(row), config.batch);
            const summary = net.summarize(config.batch);
            loss += summary.loss;
            correct += summary.correct;
        }
        const seconds = secondsSince(t0);
        totals.seconds += seconds;

        totals.accuracy = try net.evaluate(test_data.imagesVa(0), test_data.labelsVa(0), test_data.rows);
        std.debug.print("  epoch {d:>2}: loss {d:.4}, train acc {d:.4}, test acc {d:.4}, {d:.3} s (+{d:.3} s upload)\n", .{
            epoch + 1,
            loss / @as(f64, @floatFromInt(batches)),
            @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(batches * config.batch)),
            totals.accuracy,
            seconds,
            copy_seconds,
        });
    }
    return totals;
}

fn trainCpu(net: *Cpu, images: []f32, labels: []u32, data: mnist.Mnist, order: []u32, epochs: u32) !Totals {
    var totals = Totals{};
    var rng = model.Xorshift.init(config.seed ^ 0xABCD);
    const rows: u32 = @intCast(data.train.count());
    const batches: u32 = rows / config.batch;
    std.debug.print("CPU training, one thread:\n", .{});

    for (0..epochs) |epoch| {
        model.shuffle(&rng, order);
        model.gather(data.train, order, images, labels);

        var loss: f64 = 0;
        var correct: u64 = 0;
        const t0 = gpu.sys.monotonicNs();
        for (0..batches) |b| {
            const row = b * config.batch;
            net.computeGrads(images[row * mnist.image_pixels..], labels[row..], config.batch);
            net.applyUpdate();
            const summary = net.summarize(config.batch);
            loss += summary.loss;
            correct += summary.correct;
        }
        const seconds = secondsSince(t0);
        totals.seconds += seconds;

        totals.accuracy = net.evaluate(data.test_set.images, data.test_set.labels);
        std.debug.print("  epoch {d:>2}: loss {d:.4}, train acc {d:.4}, test acc {d:.4}, {d:.3} s\n", .{
            epoch + 1,
            loss / @as(f64, @floatFromInt(batches)),
            @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(batches * config.batch)),
            totals.accuracy,
            seconds,
        });
    }
    return totals;
}

pub fn run(client: *gpu.Client, epochs_arg: ?u32) !void {
    const allocator = std.heap.page_allocator;
    const epochs = epochs_arg orelse default_epochs;

    const data = try mnist.Mnist.load(allocator);
    defer data.deinit(allocator);
    const train_rows: u32 = @intCast(data.train.count());
    const test_rows: u32 = @intCast(data.test_set.count());
    std.debug.print("MNIST: {d} training and {d} test images\n", .{ train_rows, test_rows });

    const order = try allocator.alloc(u32, train_rows);
    defer allocator.free(order);
    const identity = try allocator.alloc(u32, test_rows);
    defer allocator.free(identity);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    for (identity, 0..) |*o, i| o.* = @intCast(i);

    const h_vaspace = try gpu.memory.allocVaSpace(client);
    defer client.free(h_vaspace, client.h_device);
    var compute = try gpu.compute.Compute.create(client, h_vaspace);
    defer compute.destroy(client);
    var copier = try gpu.copy.Queue.create(client, h_vaspace);
    defer copier.destroy(client);

    var train = try DataOnGpu.create(client, h_vaspace, train_rows);
    defer train.destroy(client);
    var test_data = try DataOnGpu.create(client, h_vaspace, test_rows);
    defer test_data.destroy(client);
    test_data.fill(data.test_set, identity);
    try test_data.upload(&copier);

    // Evaluation runs all test rows at once, so activations must hold them.
    const max_rows = @max(config.batch, test_rows);

    // The first batch, for the gradient check.
    train.fill(data.train, order);
    try train.upload(&copier);
    {
        var check_gpu = try Gpu.init(client, h_vaspace, &compute, &copier, max_rows);
        defer check_gpu.deinit();
        var check_cpu = try Cpu.init(allocator, max_rows);
        defer check_cpu.deinit();
        try checkGradients(&check_gpu, &check_cpu, &train);
    }

    // Both runs start from the same weights and see the same order of images.
    var net = try Gpu.init(client, h_vaspace, &compute, &copier, max_rows);
    defer net.deinit();
    const gpu_totals = try trainGpu(&net, &copier, &train, &test_data, data.train, order, epochs);
    try compute.queue.checkFault();

    for (order, 0..) |*o, i| o.* = @intCast(i);
    var cpu_net = try Cpu.init(allocator, max_rows);
    defer cpu_net.deinit();
    const images = try allocator.alloc(f32, data.train.images.len);
    defer allocator.free(images);
    const labels = try allocator.alloc(u32, data.train.labels.len);
    defer allocator.free(labels);
    const cpu_totals = try trainCpu(&cpu_net, images, labels, data, order, epochs);

    const e: f64 = @floatFromInt(epochs);
    std.debug.print("summary over {d} epochs:\n", .{epochs});
    std.debug.print("  GPU: {d:.3} s per epoch, final test accuracy {d:.4}\n", .{ gpu_totals.seconds / e, gpu_totals.accuracy });
    std.debug.print("  CPU: {d:.3} s per epoch, final test accuracy {d:.4}\n", .{ cpu_totals.seconds / e, cpu_totals.accuracy });
    std.debug.print("  GPU is {d:.1}x the speed of one CPU thread\n", .{cpu_totals.seconds / gpu_totals.seconds});
    if (gpu_totals.accuracy < target_accuracy) {
        std.debug.print("error: GPU test accuracy is below the {d:.2} target\n", .{target_accuracy});
        return error.AccuracyBelowTarget;
    }
    std.debug.print("verified: GPU test accuracy reached the {d:.2} target\n", .{target_accuracy});
}
