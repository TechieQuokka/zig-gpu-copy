//! Methods of the AMPERE_DMA_COPY_B (0xC7B5) copy engine, from clc7b5.h.

const std = @import("std");
const methods = @import("methods.zig");
const Method = methods.Method;
const abi = @import("rm/abi.zig");
const Client = @import("rm/client.zig").Client;
const queue_mod = @import("queue.zig");

pub const Engine = struct {
    /// Hardware recommends subchannel 4 for copy engines, and GRCE requires it.
    pub const subchannel = 4;

    pub const set_object = Method(Engine, 0x0000, u32);
    pub const set_semaphore = Method(Engine, 0x0240, u64);
    pub const set_semaphore_payload = Method(Engine, 0x0248, u32);
    pub const launch_dma = Method(Engine, 0x0300, LaunchDma);
    pub const offset_in = Method(Engine, 0x0400, u64);
    pub const offset_out = Method(Engine, 0x0408, u64);
    pub const line_length_in = Method(Engine, 0x0418, u32);
};

pub const LaunchDma = packed struct(u32) {
    transfer: enum(u2) { none = 0, pipelined = 1, non_pipelined = 2 } = .none,
    flush: bool = false,
    semaphore: enum(u2) { none = 0, release_one_word = 1 } = .none,
    interrupt: u2 = 0,
    src_layout: Layout = .blocklinear,
    dst_layout: Layout = .blocklinear,
    multi_line: bool = false,
    remap: bool = false,
    reserved0: u15 = 0,
    disable_plc: bool = false,
    reserved1: u5 = 0,

    pub const Layout = enum(u1) { blocklinear = 0, pitch = 1 };
};

/// The largest length one LAUNCH_DMA can copy without multi-line mode.
pub const max_length: u64 = std.math.maxInt(u32);

/// Words used by `encode`.
pub const encoded_words = 2 + 3 + 3 + 2 + 2 + 3 + 2 + 2;

/// Encodes a copy followed by a flushed semaphore release, so that seeing
/// `fence_value` at `fence_va` means the copied data is visible to the CPU.
pub fn encode(
    words: []u32,
    src: u64,
    dst: u64,
    length: u32,
    fence_va: u64,
    fence_value: u32,
) usize {
    var s = methods.Stream(Engine){ .words = words };
    s.put(Engine.set_object, abi.class.copy);
    s.put(Engine.offset_in, src);
    s.put(Engine.offset_out, dst);
    s.put(Engine.line_length_in, length);
    s.put(Engine.launch_dma, .{
        .transfer = .non_pipelined,
        .src_layout = .pitch,
        .dst_layout = .pitch,
        .disable_plc = true,
    });
    s.put(Engine.set_semaphore, fence_va);
    s.put(Engine.set_semaphore_payload, fence_value);
    // A flush only works on a LAUNCH_DMA paired with a semaphore release
    // (driver comment, bug 1709888).
    s.put(Engine.launch_dma, .{ .flush = true, .semaphore = .release_one_word, .disable_plc = true });
    return s.len;
}

/// A channel on the first copy engine the GPU reports.
pub const Queue = struct {
    queue: queue_mod.Queue,

    pub fn create(client: *Client, h_vaspace: abi.Handle) !Queue {
        const engines = try client.engines();
        const engine = for (engines.slice()) |e| {
            if (e >= abi.engine.copy0 and e <= abi.engine.copy9) break e;
        } else return error.NoCopyEngine;

        var queue = try queue_mod.Queue.create(client, h_vaspace, engine);
        errdefer queue.destroy(client);
        var params = abi.CopyAlloc{ .version = abi.copy_alloc_version_1, .engineType = engine };
        try queue.bind(client, abi.class.copy, &params, "alloc copy object");
        try queue.start(client);
        return .{ .queue = queue };
    }

    pub fn destroy(self: *Queue, client: *Client) void {
        self.queue.destroy(client);
    }

    /// Copies and waits for completion.
    pub fn copy(self: *Queue, src: u64, dst: u64, length: u32) queue_mod.Error!void {
        const recording = try self.queue.begin(encoded_words);
        const used = encode(recording.words, src, dst, length, recording.fence_va, recording.fence_value);
        try self.queue.wait(try self.queue.submit(recording, used));
    }
};

test "LAUNCH_DMA fields match the expected register values" {
    try std.testing.expectEqual(@as(u32, 0x0400_0182), @as(u32, @bitCast(LaunchDma{
        .transfer = .non_pipelined,
        .src_layout = .pitch,
        .dst_layout = .pitch,
        .disable_plc = true,
    })));
    try std.testing.expectEqual(@as(u32, 0x0400_000C), @as(u32, @bitCast(LaunchDma{
        .flush = true,
        .semaphore = .release_one_word,
        .disable_plc = true,
    })));
}

test "encode uses exactly the declared number of words" {
    var words: [encoded_words]u32 = undefined;
    const used = encode(&words, 0x1_0000_0000, 0x2_0000_0000, 4096, 0x3_0000_0000, 9);
    try std.testing.expectEqual(@as(usize, encoded_words), used);
    try std.testing.expectEqual(abi.class.copy, words[1]);
}
