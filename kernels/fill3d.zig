//! Writes a value computed from each element's position. It exercises several
//! parameters of different types, three-dimensional dispatch and f32 arithmetic.

const abi = @import("abi.zig");
const rt = @import("rt.zig");

/// The computation itself. It is pure, so the host compiles this same function
/// to check what the GPU produced.
pub fn element(p: abi.Fill3d, x: u32, y: u32, z: u32) f32 {
    const fx: f32 = @floatFromInt(x);
    const fy: f32 = @floatFromInt(y);
    const fz: f32 = @floatFromInt(z);
    const a = fx * p.scale + p.offset;
    const b = fy * p.scale - p.offset;
    return @sqrt(a * a + b * b) + fz;
}

/// Exists only when compiling for the GPU.
pub const gpu_entry = if (rt.is_gpu) struct {
    export fn fill3d(p: abi.Fill3d) callconv(.kernel) void {
        const id = rt.globalId();
        const out: [*]f32 = @ptrFromInt(p.out);
        const index = id[0] + p.width * (id[1] + p.height * id[2]);
        out[index] = element(p, id[0], id[1], id[2]);
    }
} else struct {};

comptime {
    _ = gpu_entry;
}
