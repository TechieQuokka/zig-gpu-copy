//! The host's view of the kernels: their parameters and their pure computations.
//! Each kernel file is also compiled on its own for nvptx64 by build.zig.

pub const abi = @import("abi.zig");
pub const math = @import("math.zig");
pub const fill3d = @import("fill3d.zig");
pub const gemm = @import("gemm.zig");
pub const mlp = @import("mlp.zig");

test {
    _ = math;
    _ = mlp;
}
