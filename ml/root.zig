//! Deep learning on top of the `gpu` module.

pub const cpu = @import("cpu.zig");
pub const mnist = @import("mnist.zig");
pub const model = @import("model.zig");
pub const CpuMlp = @import("cpu_mlp.zig").CpuMlp;
pub const GpuMlp = @import("gpu_mlp.zig").GpuMlp;

test {
    _ = cpu;
    _ = mnist;
    _ = model;
    _ = @import("cpu_mlp.zig");
}
