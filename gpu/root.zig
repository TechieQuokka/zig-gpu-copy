//! Direct control of an NVIDIA GPU through the kernel driver, without libcuda or
//! libc. This module knows nothing about deep learning.

pub const abi = @import("rm/abi.zig");
pub const sys = @import("rm/sys.zig");
pub const Client = @import("rm/client.zig").Client;
pub const ClientError = @import("rm/client.zig").Error;
pub const memory = @import("memory.zig");
pub const Buffer = memory.Buffer;
pub const queue = @import("queue.zig");
pub const Queue = queue.Queue;
pub const copy = @import("copy.zig");
pub const compute = @import("compute.zig");
pub const kernel_elf = @import("kernel_elf.zig");

test {
    _ = @import("rm/status.zig");
    _ = @import("methods.zig");
    _ = compute;
    _ = kernel_elf;
    _ = queue;
    _ = copy;
    _ = abi;
}
