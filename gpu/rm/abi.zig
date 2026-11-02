//! NVIDIA Resource Manager kernel ABI for driver 595.99.02.
//!
//! The ABI is not stable across driver versions. Definitions come from the
//! open-gpu-kernel-modules tag named by `driver_version`. Sizes and offsets in
//! the `comptime` block at the end were produced by compiling the C headers, so
//! a layout that drifts from the driver does not build.

const std = @import("std");

pub const driver_version = "595.99.02";

pub const ioctl_magic: u8 = 'F';

/// An RM object handle. Distinct from u32 so that a handle cannot be confused
/// with a class, a size or a status.
pub const Handle = enum(u32) { none = 0, _ };

/// Ioctl numbers, from nv-ioctl-numbers.h and nv_escape.h.
pub const Esc = enum(u8) {
    card_info = 200,
    register_fd = 201,
    check_version_str = 210,
    attach_gpus_to_fd = 212,
    rm_free = 0x29,
    rm_control = 0x2A,
    rm_alloc = 0x2B,
    rm_map_memory = 0x4E,
    rm_unmap_memory = 0x4F,
    rm_map_memory_dma = 0x57,
    rm_unmap_memory_dma = 0x58,
};

pub const max_devices = 32;
pub const max_subdevices = 8;
pub const version_reply_recognized: u32 = 1;

pub const class = struct {
    pub const root_client: u32 = 0x41;
    pub const device: u32 = 0x80;
    pub const subdevice: u32 = 0x2080;
    pub const memory_system: u32 = 0x3E;
    pub const memory_local_user: u32 = 0x40;
    pub const memory_virtual: u32 = 0x50A0;
    pub const vaspace: u32 = 0x90F1;
    pub const usermode: u32 = 0xC361;
    pub const channel_gpfifo: u32 = 0xC56F;
    pub const copy: u32 = 0xC7B5;
    pub const compute: u32 = 0xC7C0;
};

pub const ctrl = struct {
    pub const gpu_get_id_info_v2: u32 = 0x205;
    pub const gpu_get_classlist_v2: u32 = 0x800292;
    pub const gpu_get_engines_v2: u32 = 0x20800170;
    pub const gpfifo_schedule: u32 = 0xa06f0103;
    pub const gpfifo_get_work_submit_token: u32 = 0xc36f0108;
};

/// NV2080_ENGINE_TYPE values, from cl2080_notification.h.
pub const engine = struct {
    pub const gr0: u32 = 0x1;
    pub const copy0: u32 = 0x9;
    pub const copy9: u32 = 0x12;
};

pub const vamode_multiple_vaspaces: u32 = 2;
pub const copy_alloc_version_1: u32 = 1;

pub const alloc_flags = struct {
    pub const alignment_force: u32 = 0x0000_0100;
    pub const memory_handle_provided: u32 = 0x0000_4000;
    pub const map_not_required: u32 = 0x0000_8000;
    pub const virtual: u32 = 0x0008_0000;
};

/// NVOS32_ATTR, the location, physicality and coherency fields only.
pub const MemoryAttr = packed struct(u32) {
    reserved0: u25 = 0,
    location: Location = .vidmem,
    physicality: Physicality = .default,
    coherency: Coherency = .default,

    pub const Location = enum(u2) { vidmem = 0, pci = 1, _ };
    pub const Physicality = enum(u2) { default = 0, noncontiguous = 1, _ };
    pub const Coherency = enum(u3) { default = 0, cached = 1, _ };
};

pub const map_access_read_write: u32 = 0;
pub const map_access_write_only: u32 = 2;
pub const dma_flags_cache_snoop_enable: u32 = 1 << 4;

pub const usermode_size: usize = 0x10000;
pub const usermode_notify_channel_pending: usize = 0x90;
pub const userd_size: usize = 0x200;
pub const userd_gp_get: usize = 0x88;
pub const userd_gp_put: usize = 0x8C;
pub const notifier_slots: usize = 3;

pub const PciInfo = extern struct {
    domain: u32,
    bus: u8,
    slot: u8,
    function: u8,
    vendor_id: u16,
    device_id: u16,
};

/// nv_ioctl_card_info_t
pub const CardInfo = extern struct {
    valid: u8,
    pci_info: PciInfo,
    gpu_id: u32,
    interrupt_line: u16,
    reg_address: u64,
    reg_size: u64,
    fb_address: u64,
    fb_size: u64,
    minor_number: u32,
    dev_name: [10]u8,
};

/// nv_ioctl_rm_api_version_t
pub const RmApiVersion = extern struct {
    cmd: u32,
    reply: u32,
    version_string: [64]u8,
};

/// nv_ioctl_register_fd_t
pub const RegisterFd = extern struct {
    ctl_fd: i32,
};

/// NVOS00_PARAMETERS
pub const RmFree = extern struct {
    hRoot: Handle,
    hObjectParent: Handle,
    hObjectOld: Handle,
    status: u32,
};

/// NVOS21_PARAMETERS
pub const RmAlloc = extern struct {
    hRoot: Handle,
    hObjectParent: Handle,
    hObjectNew: Handle,
    hClass: u32,
    pAllocParms: u64,
    paramsSize: u32,
    status: u32,
};

/// NVOS54_PARAMETERS
pub const RmControl = extern struct {
    hClient: Handle,
    hObject: Handle,
    cmd: u32,
    flags: u32,
    params: u64,
    paramsSize: u32,
    status: u32,
};

/// NVOS33_PARAMETERS
pub const RmMapMemory = extern struct {
    hClient: Handle,
    hDevice: Handle,
    hMemory: Handle,
    offset: u64,
    length: u64,
    pLinearAddress: u64,
    status: u32,
    flags: u32,
};

/// nv_ioctl_nvos33_parameters_with_fd
pub const RmMapMemoryWithFd = extern struct {
    params: RmMapMemory,
    fd: i32,
};

/// NVOS34_PARAMETERS
pub const RmUnmapMemory = extern struct {
    hClient: Handle,
    hDevice: Handle,
    hMemory: Handle,
    pLinearAddress: u64,
    status: u32,
    flags: u32,
};

/// NVOS46_PARAMETERS
pub const RmMapMemoryDma = extern struct {
    hClient: Handle,
    hDevice: Handle,
    hDma: Handle,
    hMemory: Handle,
    offset: u64,
    length: u64,
    flags: u32,
    flags2: u32,
    kindOverride: u32,
    dmaOffset: u64,
    status: u32,
};

/// NVOS47_PARAMETERS
pub const RmUnmapMemoryDma = extern struct {
    hClient: Handle,
    hDevice: Handle,
    hDma: Handle,
    hMemory: Handle,
    flags: u32,
    dmaOffset: u64,
    size: u64,
    status: u32,
};

/// NV0080_ALLOC_PARAMETERS
pub const DeviceAlloc = extern struct {
    deviceId: u32,
    hClientShare: Handle,
    hTargetClient: Handle,
    hTargetDevice: Handle,
    flags: u32,
    vaSpaceSize: u64,
    vaStartInternal: u64,
    vaLimitInternal: u64,
    vaMode: u32,
};

/// NV2080_ALLOC_PARAMETERS
pub const SubdeviceAlloc = extern struct {
    subDeviceId: u32,
};

/// NV_VASPACE_ALLOCATION_PARAMETERS
pub const VaSpaceAlloc = extern struct {
    index: u32,
    flags: u32,
    vaSize: u64,
    vaStartInternal: u64,
    vaLimitInternal: u64,
    bigPageSize: u32,
    vaBase: u64,
    pasid: u32,
};

/// NV_MEMORY_ALLOCATION_PARAMS
pub const MemoryAlloc = extern struct {
    owner: u32,
    type: u32,
    flags: u32,
    width: u32,
    height: u32,
    pitch: i32,
    attr: MemoryAttr,
    attr2: u32,
    format: u32,
    comprCovg: u32,
    zcullCovg: u32,
    rangeLo: u64,
    rangeHi: u64,
    size: u64,
    alignment: u64,
    offset: u64,
    limit: u64,
    address: u64,
    ctagOffset: u32,
    hVASpace: Handle,
    internalflags: u32,
    tag: u32,
    numaNode: i32,
};

/// NV_MEMORY_DESC_PARAMS
pub const MemoryDesc = extern struct {
    base: u64,
    size: u64,
    addressSpace: u32,
    cacheAttrib: u32,
};

/// NV_CHANNEL_ALLOC_PARAMS
pub const ChannelAlloc = extern struct {
    hObjectError: Handle,
    hObjectBuffer: Handle,
    gpFifoOffset: u64,
    gpFifoEntries: u32,
    flags: u32,
    hContextShare: Handle,
    hVASpace: Handle,
    hUserdMemory: [max_subdevices]Handle,
    userdOffset: [max_subdevices]u64,
    engineType: u32,
    cid: u32,
    subDeviceId: u32,
    hObjectEccError: Handle,
    instanceMem: MemoryDesc,
    userdMem: MemoryDesc,
    ramfcMem: MemoryDesc,
    mthdbufMem: MemoryDesc,
    hPhysChannelGroup: Handle,
    internalFlags: u32,
    errorNotifierMem: MemoryDesc,
    eccErrorNotifierMem: MemoryDesc,
    ProcessID: u32,
    SubProcessID: u32,
    encryptIv: [3]u32,
    decryptIv: [3]u32,
    hmacNonce: [8]u32,
    tpcConfigID: u32,
};

/// NVB0B5_ALLOCATION_PARAMETERS
pub const CopyAlloc = extern struct {
    version: u32,
    engineType: u32,
};

/// NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS
pub const GpuIdInfoV2 = extern struct {
    gpuId: u32,
    gpuFlags: u32,
    deviceInstance: u32,
    subDeviceInstance: u32,
    sliStatus: u32,
    boardId: u32,
    gpuInstance: u32,
    numaId: i32,
};

/// NV0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS
pub const ClassListV2 = extern struct {
    numClasses: u32,
    classList: [200]u32,

    pub fn slice(self: *const ClassListV2) []const u32 {
        return self.classList[0..self.numClasses];
    }

    pub fn has(self: *const ClassListV2, wanted: u32) bool {
        return std.mem.indexOfScalar(u32, self.slice(), wanted) != null;
    }
};

/// NV2080_CTRL_GPU_GET_ENGINES_V2_PARAMS
pub const EnginesV2 = extern struct {
    engineCount: u32,
    engineList: [0x54]u32,

    pub fn slice(self: *const EnginesV2) []const u32 {
        return self.engineList[0..self.engineCount];
    }
};

/// NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS
pub const WorkSubmitToken = extern struct {
    workSubmitToken: u32,
};

/// NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS
pub const GpfifoSchedule = extern struct {
    bEnable: u8,
    bSkipSubmit: u8,
    bSkipEnable: u8,
};

/// NvNotification, written by RM to the error notifier when a channel faults.
pub const Notification = extern struct {
    timeStamp: [2]u32,
    info32: u32,
    info16: u16,
    status: u16,
};

/// Fails the build if `T` does not have `size` bytes and the given field offsets.
fn pin(comptime T: type, comptime size: usize, comptime offsets: anytype) void {
    if (@sizeOf(T) != size) {
        @compileError(std.fmt.comptimePrint("{s}: size is {d}, driver has {d}", .{ @typeName(T), @sizeOf(T), size }));
    }
    for (std.meta.fieldNames(@TypeOf(offsets))) |name| {
        const expected: usize = @field(offsets, name);
        if (@offsetOf(T, name) != expected) {
            @compileError(std.fmt.comptimePrint("{s}.{s}: offset is {d}, driver has {d}", .{
                @typeName(T), name, @offsetOf(T, name), expected,
            }));
        }
    }
}

comptime {
    pin(CardInfo, 72, .{ .gpu_id = 16, .minor_number = 56 });
    pin(RmApiVersion, 72, .{});
    pin(RmFree, 16, .{});
    pin(RmAlloc, 32, .{ .pAllocParms = 16 });
    pin(RmControl, 32, .{});
    pin(RmMapMemory, 48, .{});
    pin(RmMapMemoryWithFd, 56, .{ .fd = 48 });
    pin(RmUnmapMemory, 32, .{});
    pin(RmMapMemoryDma, 64, .{ .dmaOffset = 48 });
    pin(RmUnmapMemoryDma, 48, .{});
    pin(DeviceAlloc, 56, .{});
    pin(VaSpaceAlloc, 56, .{});
    pin(MemoryAlloc, 128, .{ .size = 64, .hVASpace = 108 });
    pin(ChannelAlloc, 368, .{ .hUserdMemory = 32, .engineType = 128, .errorNotifierMem = 248, .tpcConfigID = 360 });
    pin(ClassListV2, 804, .{});
    pin(EnginesV2, 340, .{});
    pin(Notification, 16, .{ .info32 = 8, .status = 14 });
}
