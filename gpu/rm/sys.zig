//! Raw Linux syscalls used to talk to the NVIDIA kernel driver.

const std = @import("std");
const linux = std.os.linux;
const abi = @import("abi.zig");
const status_names = @import("status.zig");

pub const Error = error{ SyscallFailed, RmFailed };

fn check(rc: usize, what: []const u8) Error!usize {
    const e = linux.errno(rc);
    if (e == .SUCCESS) return rc;
    std.debug.print("error: {s}: errno {s}\n", .{ what, @tagName(e) });
    return error.SyscallFailed;
}

pub fn openDevice(path: [*:0]const u8) Error!i32 {
    return @intCast(try check(linux.open(path, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0), "open"));
}

pub fn close(fd: i32) void {
    _ = linux.close(fd);
}

/// Issues `_IOWR('F', esc, T)`, where `arg` is a `*T` in/out parameter block.
pub fn ioctl(fd: i32, esc: abi.Esc, arg: anytype, what: []const u8) Error!void {
    const T = @typeInfo(@TypeOf(arg)).pointer.child;
    const request = linux.IOCTL.IOWR(abi.ioctl_magic, @intFromEnum(esc), T);
    _ = try check(linux.ioctl(fd, request, @intFromPtr(arg)), what);
}

/// RM reports failures in a status field while the ioctl itself succeeds.
pub fn checkStatus(code: u32, what: []const u8) Error!void {
    if (code == 0) return;
    std.debug.print("error: {s}: RM status 0x{x} {s}\n", .{ what, code, status_names.name(code) });
    return error.RmFailed;
}

pub fn mmapShared(fd: i32, length: usize, readable: bool) Error![*]u8 {
    const rc = try check(linux.mmap(
        null,
        length,
        .{ .READ = readable, .WRITE = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    ), "mmap");
    return @ptrFromInt(rc);
}

pub fn munmap(ptr: [*]u8, length: usize) void {
    _ = linux.munmap(ptr, length);
}

pub fn monotonicNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
