//! Typed GPU methods. A method is a write to a register of the engine bound to a
//! subchannel. Each method here is a type that names its engine, address and
//! payload, so sending one to the wrong engine or with the wrong payload does
//! not compile.

const std = @import("std");

/// Declares a method of `Engine` at byte offset `offset`. The payload is a u32,
/// a `packed struct(u32)`, or a u64, which is sent upper word first to two
/// consecutive registers, as the GPU's address registers expect.
pub fn Method(comptime Engine: type, comptime offset: u16, comptime P: type) type {
    comptime {
        std.debug.assert(offset % 4 == 0);
        if (@bitSizeOf(P) != 32 and @bitSizeOf(P) != 64) @compileError("payload must be 32 or 64 bits");
    }
    return struct {
        pub const engine = Engine;
        pub const address = offset;
        pub const Payload = P;
    };
}

/// clc56f.h: NVC56F_DMA_* fields of a method header.
const Header = packed struct(u32) {
    address: u12,
    reserved: u1 = 0,
    subchannel: u3,
    count: u13,
    sec_op: u3 = 1, // INC_METHOD
};

/// Writes methods of one engine into pushbuffer words.
pub fn Stream(comptime Engine: type) type {
    return struct {
        words: []u32,
        len: usize = 0,

        const Self = @This();

        pub fn put(self: *Self, comptime M: type, payload: M.Payload) void {
            comptime if (M.engine != Engine) @compileError("method belongs to another engine");

            const count = @bitSizeOf(M.Payload) / 32;
            self.push(@bitCast(Header{
                .address = M.address >> 2,
                .subchannel = Engine.subchannel,
                .count = count,
            }));
            if (count == 2) {
                const value: u64 = @bitCast(payload);
                self.push(@truncate(value >> 32));
                self.push(@truncate(value));
            } else {
                self.push(@bitCast(payload));
            }
        }

        fn push(self: *Self, word: u32) void {
            self.words[self.len] = word;
            self.len += 1;
        }
    };
}

const TestEngine = struct {
    pub const subchannel = 4;
    pub const reg = Method(TestEngine, 0x400, u32);
    pub const addr = Method(TestEngine, 0x408, u64);
};

test "a method header has the documented layout" {
    var words: [8]u32 = undefined;
    var stream = Stream(TestEngine){ .words = &words };
    stream.put(TestEngine.reg, 7);
    // SEC_OP=1, COUNT=1, SUBCHANNEL=4, ADDRESS=0x400>>2.
    try std.testing.expectEqual(@as(u32, 0x2001_8100), words[0]);
    try std.testing.expectEqual(@as(u32, 7), words[1]);
}

test "a u64 payload is sent upper word first" {
    var words: [8]u32 = undefined;
    var stream = Stream(TestEngine){ .words = &words };
    stream.put(TestEngine.addr, 0x12_3456_7000);
    try std.testing.expectEqual(@as(u32, 0x2002_8102), words[0]);
    try std.testing.expectEqual(@as(u32, 0x12), words[1]);
    try std.testing.expectEqual(@as(u32, 0x3456_7000), words[2]);
    try std.testing.expectEqual(@as(usize, 3), stream.len);
}
