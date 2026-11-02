//! What a kernel can ask the GPU about itself. PTX special registers have no Zig
//! builtin here, so each is read through inline assembly. Only code compiled for
//! nvptx64 may call these functions.

const builtin = @import("builtin");

pub const is_gpu = builtin.cpu.arch == .nvptx64;

fn register(comptime name: []const u8) u32 {
    return asm ("mov.u32 %[r], %" ++ name ++ ";"
        : [r] "=r" (-> u32),
    );
}

/// The index of the calling thread across the whole grid, in each dimension.
pub inline fn globalId() [3]u32 {
    return .{
        register("ctaid.x") * register("ntid.x") + register("tid.x"),
        register("ctaid.y") * register("ntid.y") + register("tid.y"),
        register("ctaid.z") * register("ntid.z") + register("tid.z"),
    };
}
