//! Physical allocation, GPU virtual mapping and CPU mapping of RM memory.

const std = @import("std");
const abi = @import("rm/abi.zig");
const sys = @import("rm/sys.zig");
const client_mod = @import("rm/client.zig");
const Client = client_mod.Client;
const Handle = abi.Handle;

const page_size: u64 = 4096;
const va_alignment: u64 = 2 * 1024 * 1024;
const owner: u32 = 0x5a47_4350; // "ZGCP"

pub const Location = enum { sysmem, vidmem };

pub fn allocVaSpace(client: *Client) sys.Error!Handle {
    var params = std.mem.zeroes(abi.VaSpaceAlloc);
    return client.alloc(client.h_device, abi.class.vaspace, &params, "alloc VA space");
}

/// An RM object mapped into this process.
pub const CpuMapping = struct {
    fd: i32,
    ptr: [*]u8,
    length: usize,
    h_memory: Handle,
    /// RM's identifier for this mapping, needed to unmap it. It is not the CPU
    /// pointer that mmap returns.
    rm_address: u64,

    pub fn create(
        client: *Client,
        target: client_mod.MappingTarget,
        h_memory: Handle,
        length: usize,
        access: u32,
    ) sys.Error!CpuMapping {
        const fd = try client.newMappingFd(target);
        errdefer sys.close(fd);

        var req = std.mem.zeroes(abi.RmMapMemoryWithFd);
        req.params.hClient = client.h_client;
        req.params.hDevice = client.h_subdevice;
        req.params.hMemory = h_memory;
        req.params.length = length;
        req.params.flags = access;
        req.fd = fd;
        try sys.ioctl(client.ctl_fd, .rm_map_memory, &req, "map memory to CPU");
        try sys.checkStatus(req.params.status, "map memory to CPU");
        errdefer rmUnmap(client, h_memory, req.params.pLinearAddress);

        const ptr = try sys.mmapShared(fd, length, access != abi.map_access_write_only);
        return .{
            .fd = fd,
            .ptr = ptr,
            .length = length,
            .h_memory = h_memory,
            .rm_address = req.params.pLinearAddress,
        };
    }

    pub fn destroy(self: *CpuMapping, client: *Client) void {
        sys.munmap(self.ptr, self.length);
        rmUnmap(client, self.h_memory, self.rm_address);
        sys.close(self.fd);
    }

    fn rmUnmap(client: *Client, h_memory: Handle, rm_address: u64) void {
        var req = std.mem.zeroes(abi.RmUnmapMemory);
        req.hClient = client.h_client;
        req.hDevice = client.h_subdevice;
        req.hMemory = h_memory;
        req.pLinearAddress = rm_address;
        sys.ioctl(client.ctl_fd, .rm_unmap_memory, &req, "unmap memory from CPU") catch return;
        sys.checkStatus(req.status, "unmap memory from CPU") catch {};
    }
};

/// A physical allocation mapped into a GPU VA space, and into this process when
/// it is system memory.
pub const Buffer = struct {
    location: Location,
    size: u64,
    h_physical: Handle,
    h_virtual: Handle,
    gpu_va: u64,
    cpu: ?CpuMapping,

    pub fn create(client: *Client, h_vaspace: Handle, location: Location, requested: u64) sys.Error!Buffer {
        const size = std.mem.alignForward(u64, requested, page_size);

        var phys = std.mem.zeroes(abi.MemoryAlloc);
        phys.owner = owner;
        phys.size = size;
        phys.flags = abi.alloc_flags.memory_handle_provided;
        const class: u32 = switch (location) {
            .sysmem => blk: {
                phys.attr = .{ .location = .pci, .physicality = .noncontiguous, .coherency = .cached };
                break :blk abi.class.memory_system;
            },
            .vidmem => blk: {
                phys.flags |= abi.alloc_flags.map_not_required;
                phys.attr = .{ .location = .vidmem };
                break :blk abi.class.memory_local_user;
            },
        };
        const h_physical = try client.alloc(client.h_device, class, &phys, "alloc physical memory");
        errdefer client.free(h_physical, client.h_device);

        // RM may place memory elsewhere than requested, so confirm the result.
        const expected: abi.MemoryAttr.Location = if (location == .sysmem) .pci else .vidmem;
        if (phys.attr.location != expected) {
            std.debug.print("error: requested {s}, RM returned location {d}\n", .{
                @tagName(location), @intFromEnum(phys.attr.location),
            });
            return error.RmFailed;
        }

        var virt = std.mem.zeroes(abi.MemoryAlloc);
        virt.owner = owner;
        virt.size = size;
        virt.alignment = va_alignment;
        virt.hVASpace = h_vaspace;
        virt.flags = abi.alloc_flags.virtual | abi.alloc_flags.alignment_force | abi.alloc_flags.memory_handle_provided;
        const h_virtual = try client.alloc(client.h_device, abi.class.memory_virtual, &virt, "alloc GPU virtual range");
        errdefer client.free(h_virtual, client.h_device);

        var map = std.mem.zeroes(abi.RmMapMemoryDma);
        map.hClient = client.h_client;
        map.hDevice = client.h_device;
        map.hDma = h_virtual;
        map.hMemory = h_physical;
        map.length = size;
        map.flags = abi.dma_flags_cache_snoop_enable; // Ignored for vidmem.
        try sys.ioctl(client.ctl_fd, .rm_map_memory_dma, &map, "map memory to GPU");
        try sys.checkStatus(map.status, "map memory to GPU");

        var self = Buffer{
            .location = location,
            .size = size,
            .h_physical = h_physical,
            .h_virtual = h_virtual,
            .gpu_va = map.dmaOffset,
            .cpu = null,
        };
        errdefer self.unmapGpu(client);

        if (location == .sysmem) {
            self.cpu = try CpuMapping.create(client, .sysmem, h_physical, size, abi.map_access_read_write);
        }
        return self;
    }

    pub fn destroy(self: *Buffer, client: *Client) void {
        if (self.cpu) |*cpu| cpu.destroy(client);
        self.unmapGpu(client);
        client.free(self.h_virtual, client.h_device);
        client.free(self.h_physical, client.h_device);
    }

    fn unmapGpu(self: *Buffer, client: *Client) void {
        var req = std.mem.zeroes(abi.RmUnmapMemoryDma);
        req.hClient = client.h_client;
        req.hDevice = client.h_device;
        req.hDma = self.h_virtual;
        req.hMemory = self.h_physical;
        req.dmaOffset = self.gpu_va;
        sys.ioctl(client.ctl_fd, .rm_unmap_memory_dma, &req, "unmap memory from GPU") catch return;
        sys.checkStatus(req.status, "unmap memory from GPU") catch {};
    }

    /// CPU view of a system memory buffer as elements of `T`. Asserts that the
    /// buffer is mapped, which is true of all sysmem buffers.
    pub fn slice(self: *const Buffer, comptime T: type) []T {
        const bytes = self.cpu.?.ptr[0..self.size];
        return @alignCast(std.mem.bytesAsSlice(T, bytes));
    }
};
