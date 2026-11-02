//! Reads the kernel binaries that ptxas produces. Parsing needs no allocation and
//! runs at compile time, so a missing kernel or an unsupported feature fails the
//! build, not a run on the GPU.

const std = @import("std");

pub const Error = error{ BadElf, KernelNotFound, MissingParamInfo, UsesStack, UsesSharedMemory };

pub const KernelInfo = struct {
    code: []const u8,
    register_count: u9,
    /// Offset of the kernel's parameters in constant bank 0.
    param_offset: u32,
    param_size: u32,
};

// Attribute IDs in .nv.info sections. NVIDIA does not document them; they were
// read from `cuobjdump -elf` output next to the raw section bytes.
const attr_param_cbank: u8 = 0x0a;
const attr_max_stack_size: u8 = 0x23;

const Section = struct { name: []const u8, offset: u64, size: u64, info: u32 };

fn read(comptime T: type, elf: []const u8, offset: usize) Error!T {
    if (offset + @sizeOf(T) > elf.len) return error.BadElf;
    return std.mem.readInt(T, elf[offset..][0..@sizeOf(T)], .little);
}

fn section(elf: []const u8, index: usize) Error!Section {
    const table = try read(u64, elf, 0x28);
    const entry_size = try read(u16, elf, 0x3A);
    const strings_index = try read(u16, elf, 0x3E);
    const entry = table + index * entry_size;
    const strings_offset = try read(u64, elf, table + strings_index * entry_size + 24);

    const name_offset = strings_offset + try read(u32, elf, entry);
    if (name_offset >= elf.len) return error.BadElf;
    return .{
        .name = std.mem.sliceTo(elf[name_offset..], 0),
        .offset = try read(u64, elf, entry + 24),
        .size = try read(u64, elf, entry + 32),
        .info = try read(u32, elf, entry + 44),
    };
}

fn findSection(elf: []const u8, name: []const u8) Error!?Section {
    const count = try read(u16, elf, 0x3C);
    for (0..count) |i| {
        const s = try section(elf, i);
        if (std.mem.eql(u8, s.name, name)) return s;
    }
    return null;
}

fn contents(elf: []const u8, s: Section) Error![]const u8 {
    if (s.offset + s.size > elf.len) return error.BadElf;
    return elf[s.offset..][0..s.size];
}

/// Walks attributes of an .nv.info section, returning the payload of the next
/// format-4 (sized) attribute with ID `wanted`, starting at `cursor.*`.
fn nextAttribute(info: []const u8, cursor: *usize, wanted: u8) Error!?[]const u8 {
    while (cursor.* + 4 <= info.len) {
        const format = info[cursor.*];
        const id = info[cursor.* + 1];
        const start = cursor.* + 4;
        switch (format) {
            1, 2, 3 => cursor.* = start,
            4 => {
                const size = std.mem.readInt(u16, info[cursor.* + 2..][0..2], .little);
                if (start + size > info.len) return error.BadElf;
                cursor.* = start + size;
                if (id == wanted) return info[start..][0..size];
            },
            else => return error.BadElf,
        }
    }
    return null;
}

pub fn parse(elf: []const u8, comptime name: []const u8) Error!KernelInfo {
    if (elf.len < 0x40 or !std.mem.eql(u8, elf[0..4], "\x7fELF")) return error.BadElf;

    const text = try findSection(elf, ".text." ++ name) orelse return error.KernelNotFound;
    const info_section = try findSection(elf, ".nv.info." ++ name) orelse return error.MissingParamInfo;
    if (try findSection(elf, ".nv.shared." ++ name) != null) return error.UsesSharedMemory;

    // The global .nv.info section lists stack use per function. Kernels that
    // use a stack need shader local memory, which is not set up yet.
    if (try findSection(elf, ".nv.info")) |global_info| {
        const bytes = try contents(elf, global_info);
        var cursor: usize = 0;
        while (try nextAttribute(bytes, &cursor, attr_max_stack_size)) |value| {
            if (value.len != 8) return error.BadElf;
            if (std.mem.readInt(u32, value[4..8], .little) != 0) return error.UsesStack;
        }
    }

    var cursor: usize = 0;
    const param = try nextAttribute(try contents(elf, info_section), &cursor, attr_param_cbank) orelse
        return error.MissingParamInfo;
    if (param.len != 8) return error.BadElf;
    const packed_location = std.mem.readInt(u32, param[4..8], .little);

    return .{
        .code = try contents(elf, text),
        .register_count = @intCast(text.info >> 24),
        .param_offset = packed_location & 0xffff,
        .param_size = packed_location >> 16,
    };
}

test "attribute walk skips other formats and finds sized attributes" {
    const info = [_]u8{
        0x01, 0x35, 0x00, 0x00, // format 1, no payload
        0x03, 0x19, 0x0c, 0x00, // format 3, value in the header
        0x04, 0x0a, 0x08, 0x00, 3, 0, 0, 0, 0x60, 0x01, 0x0c, 0x00,
    };
    var cursor: usize = 0;
    const value = (try nextAttribute(&info, &cursor, attr_param_cbank)).?;
    try std.testing.expectEqual(@as(usize, 8), value.len);
    try std.testing.expectEqual(@as(u32, 0x000c_0160), std.mem.readInt(u32, value[4..8], .little));
    try std.testing.expectEqual(@as(?[]const u8, null), try nextAttribute(&info, &cursor, attr_param_cbank));
}

test "a non-ELF input is rejected" {
    try std.testing.expectError(error.BadElf, parse("not an elf file, but long enough to pass the length check, surely...", "fill"));
}
