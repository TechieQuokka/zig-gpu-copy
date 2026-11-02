//! Loads MNIST from the uncompressed IDX files in `data/`.

const std = @import("std");
const linux = std.os.linux;

pub const image_pixels = 28 * 28;
pub const classes = 10;

// Standard normalization constants for MNIST.
const mean: f32 = 0.1307;
const std_dev: f32 = 0.3081;

pub const Set = struct {
    images: []f32,
    labels: []u32,

    pub fn count(self: Set) usize {
        return self.labels.len;
    }

    pub fn deinit(self: Set, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
        allocator.free(self.labels);
    }
};

pub const Mnist = struct {
    train: Set,
    test_set: Set,

    pub fn load(allocator: std.mem.Allocator) !Mnist {
        const train = try loadSet(allocator, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        errdefer train.deinit(allocator);
        const test_set = try loadSet(allocator, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
        return .{ .train = train, .test_set = test_set };
    }

    pub fn deinit(self: Mnist, allocator: std.mem.Allocator) void {
        self.train.deinit(allocator);
        self.test_set.deinit(allocator);
    }
};

fn readFile(allocator: std.mem.Allocator, path: [*:0]const u8) ![]u8 {
    const opened = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(opened) != .SUCCESS) {
        std.debug.print("error: cannot open {s}. See README.md for downloading MNIST.\n", .{path});
        return error.FileNotFound;
    }
    const fd: i32 = @intCast(opened);
    defer _ = linux.close(fd);

    const end = linux.lseek(fd, 0, linux.SEEK.END);
    if (linux.errno(end) != .SUCCESS) return error.ReadFailed;
    _ = linux.lseek(fd, 0, linux.SEEK.SET);

    const bytes = try allocator.alloc(u8, end);
    errdefer allocator.free(bytes);
    var done: usize = 0;
    while (done < bytes.len) {
        const got = linux.read(fd, bytes[done..].ptr, bytes.len - done);
        if (linux.errno(got) != .SUCCESS or got == 0) return error.ReadFailed;
        done += got;
    }
    return bytes;
}

/// Parses IDX files: a big-endian magic number, dimension sizes, then u8 values.
fn parseSet(allocator: std.mem.Allocator, image_file: []const u8, label_file: []const u8) !Set {
    if (image_file.len < 16 or std.mem.readInt(u32, image_file[0..4], .big) != 0x0803) return error.BadImageFile;
    if (label_file.len < 8 or std.mem.readInt(u32, label_file[0..4], .big) != 0x0801) return error.BadLabelFile;

    const count = std.mem.readInt(u32, image_file[4..8], .big);
    const rows = std.mem.readInt(u32, image_file[8..12], .big);
    const cols = std.mem.readInt(u32, image_file[12..16], .big);
    if (rows * cols != image_pixels) return error.BadImageFile;
    if (std.mem.readInt(u32, label_file[4..8], .big) != count) return error.LabelCountMismatch;
    if (image_file.len != 16 + count * image_pixels or label_file.len != 8 + count) return error.TruncatedFile;

    const images = try allocator.alloc(f32, count * image_pixels);
    errdefer allocator.free(images);
    const labels = try allocator.alloc(u32, count);
    errdefer allocator.free(labels);

    for (images, image_file[16..]) |*out, pixel| {
        const unit = @as(f32, @floatFromInt(pixel)) / 255.0;
        out.* = (unit - mean) / std_dev;
    }
    for (labels, label_file[8..]) |*out, label| {
        if (label >= classes) return error.BadLabelFile;
        out.* = label;
    }
    return .{ .images = images, .labels = labels };
}

fn loadSet(allocator: std.mem.Allocator, image_path: [*:0]const u8, label_path: [*:0]const u8) !Set {
    const image_file = try readFile(allocator, image_path);
    defer allocator.free(image_file);
    const label_file = try readFile(allocator, label_path);
    defer allocator.free(label_file);
    return parseSet(allocator, image_file, label_file);
}

test "parses a minimal IDX pair" {
    const allocator = std.testing.allocator;
    const header = [_]u8{ 0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 28, 0, 0, 0, 28 };
    const image_file = try allocator.alloc(u8, header.len + image_pixels);
    defer allocator.free(image_file);
    @memcpy(image_file[0..header.len], &header);
    @memset(image_file[header.len..], 255);
    const label_file = [_]u8{ 0, 0, 8, 1, 0, 0, 0, 1, 7 };

    const set = try parseSet(allocator, image_file, &label_file);
    defer set.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), set.count());
    try std.testing.expectEqual(@as(u32, 7), set.labels[0]);
    try std.testing.expectApproxEqAbs((1.0 - mean) / std_dev, set.images[0], 1e-6);
}

test "a truncated image file is rejected" {
    const allocator = std.testing.allocator;
    const image_file = [_]u8{ 0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 28, 0, 0, 0, 28, 1, 2, 3 };
    const label_file = [_]u8{ 0, 0, 8, 1, 0, 0, 0, 1, 7 };
    try std.testing.expectError(error.TruncatedFile, parseSet(allocator, &image_file, &label_file));
}
