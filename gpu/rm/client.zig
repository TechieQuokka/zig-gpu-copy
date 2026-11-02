//! RM client lifetime: device files, client, device and subdevice objects.

const std = @import("std");
const abi = @import("abi.zig");
const sys = @import("sys.zig");

const Handle = abi.Handle;

pub const Error = sys.Error || error{ DriverVersionMismatch, NoGpu };

const first_handle: u32 = 0x0a11_0001;

/// Sysmem mappings need a control-device fd, and BAR mappings need a registered
/// GPU-device fd (nv_add_mapping_context_to_file).
pub const MappingTarget = enum { sysmem, gpu_bar };

pub const Client = struct {
    ctl_fd: i32,
    dev_fd: i32,
    minor: u32,
    gpu_id: u32,
    pci_device_id: u16,
    h_client: Handle,
    h_device: Handle,
    h_subdevice: Handle,
    next_handle: u32,

    /// Opens the GPU with the given device minor, or the first GPU if null.
    pub fn open(minor: ?u32) Error!Client {
        const ctl_fd = try sys.openDevice("/dev/nvidiactl");
        errdefer sys.close(ctl_fd);

        try checkDriverVersion(ctl_fd);

        var cards = std.mem.zeroes([abi.max_devices]abi.CardInfo);
        try sys.ioctl(ctl_fd, .card_info, &cards, "NV_ESC_CARD_INFO");
        const card = for (&cards) |*c| {
            if (c.valid != 0 and (minor == null or minor == c.minor_number)) break c;
        } else {
            std.debug.print("error: no matching GPU\n", .{});
            return error.NoGpu;
        };

        const dev_fd = try openGpuFd(ctl_fd, card.minor_number);
        errdefer sys.close(dev_fd);

        var gpu_id: u32 = card.gpu_id;
        try sys.ioctl(ctl_fd, .attach_gpus_to_fd, &gpu_id, "NV_ESC_ATTACH_GPUS_TO_FD");

        var self = Client{
            .ctl_fd = ctl_fd,
            .dev_fd = dev_fd,
            .minor = card.minor_number,
            .gpu_id = card.gpu_id,
            .pci_device_id = card.pci_info.device_id,
            .h_client = .none,
            .h_device = .none,
            .h_subdevice = .none,
            .next_handle = first_handle,
        };

        // A client object is its own root. RM generates its handle.
        var root = std.mem.zeroes(abi.RmAlloc);
        root.hClass = abi.class.root_client;
        try sys.ioctl(ctl_fd, .rm_alloc, &root, "alloc client");
        try sys.checkStatus(root.status, "alloc client");
        self.h_client = root.hObjectNew;
        errdefer self.free(self.h_client, .none);

        var id_info = std.mem.zeroes(abi.GpuIdInfoV2);
        id_info.gpuId = card.gpu_id;
        try self.control(self.h_client, abi.ctrl.gpu_get_id_info_v2, &id_info, "get GPU id info");

        var device = std.mem.zeroes(abi.DeviceAlloc);
        device.deviceId = id_info.deviceInstance;
        device.hClientShare = self.h_client;
        device.vaMode = abi.vamode_multiple_vaspaces;
        self.h_device = try self.alloc(self.h_client, abi.class.device, &device, "alloc device");

        var subdevice = abi.SubdeviceAlloc{ .subDeviceId = id_info.subDeviceInstance };
        self.h_subdevice = try self.alloc(self.h_device, abi.class.subdevice, &subdevice, "alloc subdevice");

        return self;
    }

    pub fn close(self: *Client) void {
        // Freeing the client releases every object beneath it.
        self.free(self.h_client, .none);
        sys.close(self.dev_fd);
        sys.close(self.ctl_fd);
    }

    /// Allocates an object under `parent`. `params` is a pointer to the
    /// class-specific parameters, or null for classes that take none.
    pub fn alloc(self: *Client, parent: Handle, class: u32, params: anytype, what: []const u8) sys.Error!Handle {
        const handle: Handle = @enumFromInt(self.next_handle);
        self.next_handle += 1;

        var req = std.mem.zeroes(abi.RmAlloc);
        req.hRoot = self.h_client;
        req.hObjectParent = parent;
        req.hObjectNew = handle;
        req.hClass = class;
        if (@typeInfo(@TypeOf(params)) == .pointer) {
            req.pAllocParms = @intFromPtr(params);
            req.paramsSize = @sizeOf(@typeInfo(@TypeOf(params)).pointer.child);
        }
        try sys.ioctl(self.ctl_fd, .rm_alloc, &req, what);
        try sys.checkStatus(req.status, what);
        return handle;
    }

    /// Frees an object. Teardown failures are reported but cannot be acted on.
    pub fn free(self: *Client, handle: Handle, parent: Handle) void {
        var req = abi.RmFree{
            .hRoot = self.h_client,
            .hObjectParent = parent,
            .hObjectOld = handle,
            .status = 0,
        };
        sys.ioctl(self.ctl_fd, .rm_free, &req, "free object") catch return;
        sys.checkStatus(req.status, "free object") catch {};
    }

    pub fn control(self: *Client, object: Handle, cmd: u32, params: anytype, what: []const u8) sys.Error!void {
        var req = std.mem.zeroes(abi.RmControl);
        req.hClient = self.h_client;
        req.hObject = object;
        req.cmd = cmd;
        req.params = @intFromPtr(params);
        req.paramsSize = @sizeOf(@typeInfo(@TypeOf(params)).pointer.child);
        try sys.ioctl(self.ctl_fd, .rm_control, &req, what);
        try sys.checkStatus(req.status, what);
    }

    pub fn classList(self: *Client) sys.Error!abi.ClassListV2 {
        var out = std.mem.zeroes(abi.ClassListV2);
        try self.control(self.h_device, abi.ctrl.gpu_get_classlist_v2, &out, "get class list");
        return out;
    }

    pub fn engines(self: *Client) sys.Error!abi.EnginesV2 {
        var out = std.mem.zeroes(abi.EnginesV2);
        try self.control(self.h_subdevice, abi.ctrl.gpu_get_engines_v2, &out, "get engines");
        return out;
    }

    /// Opens a new fd for one CPU mapping.
    pub fn newMappingFd(self: *Client, target: MappingTarget) sys.Error!i32 {
        return switch (target) {
            .sysmem => sys.openDevice("/dev/nvidiactl"),
            .gpu_bar => openGpuFd(self.ctl_fd, self.minor),
        };
    }
};

fn openGpuFd(ctl_fd: i32, minor: u32) sys.Error!i32 {
    var path_buf: [32]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&path_buf, "/dev/nvidia{d}", .{minor}, 0) catch unreachable;
    const fd = try sys.openDevice(path.ptr);
    errdefer sys.close(fd);
    var reg = abi.RegisterFd{ .ctl_fd = ctl_fd };
    try sys.ioctl(fd, .register_fd, &reg, "NV_ESC_REGISTER_FD");
    return fd;
}

/// The ABI is valid only for the exact driver it was taken from.
fn checkDriverVersion(ctl_fd: i32) Error!void {
    var req = std.mem.zeroes(abi.RmApiVersion);
    @memcpy(req.version_string[0..abi.driver_version.len], abi.driver_version);
    try sys.ioctl(ctl_fd, .check_version_str, &req, "NV_ESC_CHECK_VERSION_STR");
    if (req.reply != abi.version_reply_recognized) {
        const loaded = std.mem.sliceTo(&req.version_string, 0);
        std.debug.print("error: built for driver {s}, but driver {s} is loaded\n", .{ abi.driver_version, loaded });
        return error.DriverVersionMismatch;
    }
}
