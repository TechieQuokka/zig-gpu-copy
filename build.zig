const std = @import("std");

/// Kernels in `kernels/`. Each is compiled for the GPU and embedded in the program.
const kernels = [_][]const u8{ "fill3d", "gemm", "mlp" };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The dependency direction app -> ml -> gpu is enforced by these imports:
    // `gpu` and `kernels` import no other module.
    const gpu = b.addModule("gpu", .{
        .root_source_file = b.path("gpu/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const kernels_module = b.addModule("kernels", .{
        .root_source_file = b.path("kernels/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ml = b.addModule("ml", .{
        .root_source_file = b.path("ml/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
            .{ .name = "kernels", .module = kernels_module },
        },
    });
    const app = b.createModule(.{
        .root_source_file = b.path("app/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
            .{ .name = "kernels", .module = kernels_module },
            .{ .name = "ml", .module = ml },
        },
    });

    for (kernels) |name| {
        const elf = addKernel(b, name);
        const import_name = b.fmt("kernel_{s}", .{name});
        app.addAnonymousImport(import_name, .{ .root_source_file = elf });
        ml.addAnonymousImport(import_name, .{ .root_source_file = elf });
    }

    const exe = b.addExecutable(.{ .name = "zig-gpu-copy", .root_module = app });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run zig-gpu-copy").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run tests that need no GPU");
    for ([_]*std.Build.Module{ gpu, kernels_module, ml, app }) |module| {
        const tests = b.addTest(.{ .root_module = module });
        test_step.dependOn(&b.addRunArtifact(tests).step);
    }
}

/// Zig -> PTX -> ptxas -> ELF. ptxas is a build tool only: the program loads
/// nothing from the CUDA toolkit.
fn addKernel(b: *std.Build, name: []const u8) std.Build.LazyPath {
    const object = b.addObject(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("kernels/{s}.zig", .{name})),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_80 },
            }),
            // Safety checks would call a panic handler, which a kernel cannot have.
            .optimize = .ReleaseFast,
        }),
    });

    const ptxas = b.addSystemCommand(&.{ "ptxas", "-arch=sm_86", "-m64" });
    ptxas.addFileArg(object.getEmittedAsm());
    ptxas.addArg("-o");
    return ptxas.addOutputFileArg(b.fmt("{s}.cubin", .{name}));
}
