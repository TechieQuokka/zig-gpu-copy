//! What the GPU and CPU models share: the configuration, its compile-time checks,
//! initial weights and data shuffling.

const std = @import("std");
const mnist = @import("mnist.zig");

pub const Config = struct {
    /// Layer widths, from the input to the classes. All shapes derive from this.
    layers: []const u32,
    batch: u32,
    rate: f32,
    momentum: f32,
    seed: u64,

    pub fn validate(comptime self: Config) void {
        if (self.layers.len < 2) @compileError("a model needs at least an input and an output");
        if (self.layers[0] != mnist.image_pixels) @compileError("the first layer must match the image size");
        if (self.layers[self.layers.len - 1] != mnist.classes) @compileError("the last layer must match the class count");
        for (self.layers) |width| if (width == 0) @compileError("a layer cannot be empty");
        if (self.batch == 0) @compileError("the batch size cannot be zero");
    }

    pub fn denseCount(comptime self: Config) usize {
        return self.layers.len - 1;
    }
};

pub const mnist_mlp = Config{
    .layers = &.{ 784, 256, 128, 10 },
    .batch = 128,
    .rate = 0.05,
    .momentum = 0.9,
    .seed = 0x5EED_2026,
};

comptime {
    mnist_mlp.validate();
}

/// A small generator, so that both models see exactly the same numbers.
pub const Xorshift = struct {
    state: u64,

    pub fn init(seed: u64) Xorshift {
        return .{ .state = seed | 1 };
    }

    pub fn next(self: *Xorshift) u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        return self.state;
    }

    /// Uniform in [-1, 1).
    pub fn unit(self: *Xorshift) f32 {
        const bits: f32 = @floatFromInt(self.next() >> 40); // 24 bits, exact in f32.
        return bits / (1 << 23) - 1.0;
    }

    pub fn below(self: *Xorshift, limit: usize) usize {
        return @intCast(self.next() % limit);
    }
};

/// Kaiming uniform initialization for relu layers. The bound is sqrt(6 / inputs).
pub fn initWeights(rng: *Xorshift, weights: []f32, inputs: u32) void {
    const bound = @sqrt(6.0 / @as(f32, @floatFromInt(inputs)));
    for (weights) |*w| w.* = rng.unit() * bound;
}

/// Fisher-Yates shuffle of an index list.
pub fn shuffle(rng: *Xorshift, order: []u32) void {
    var i = order.len;
    while (i > 1) : (i -= 1) {
        const j = rng.below(i);
        std.mem.swap(u32, &order[i - 1], &order[j]);
    }
}

/// Writes images and labels into `images_out` and `labels_out` in the given order.
pub fn gather(set: mnist.Set, order: []const u32, images_out: []f32, labels_out: []u32) void {
    for (order, 0..) |source, i| {
        const from = set.images[source * mnist.image_pixels..][0..mnist.image_pixels];
        @memcpy(images_out[i * mnist.image_pixels..][0..mnist.image_pixels], from);
        labels_out[i] = set.labels[source];
    }
}

test "shuffle keeps every index exactly once" {
    var order: [100]u32 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    var rng = Xorshift.init(42);
    shuffle(&rng, &order);

    var seen: [100]bool = @splat(false);
    var moved = false;
    for (order, 0..) |o, i| {
        try std.testing.expect(!seen[o]);
        seen[o] = true;
        if (o != i) moved = true;
    }
    try std.testing.expect(moved);
}

test "initial weights stay within the bound" {
    var weights: [1000]f32 = undefined;
    var rng = Xorshift.init(1);
    initWeights(&rng, &weights, 6);
    for (weights) |w| try std.testing.expect(@abs(w) <= 1.0);
}
