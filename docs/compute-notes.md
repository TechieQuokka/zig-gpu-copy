# Compute notes

Findings for driver 595.99.02, Zig 0.17.0 and an RTX 3060 (GA106).

## Milestone 0 results

Nothing here has run on the GPU yet. These results cover compilation and reference
material only.

### Kernel toolchain

```sh
zig build-obj kernel.zig -target nvptx64-cuda -mcpu=sm_80 -O ReleaseFast \
    -fno-emit-bin -femit-asm=kernel.ptx
ptxas -arch=sm_86 -m64 kernel.ptx -o kernel.cubin
cuobjdump -sass kernel.cubin
```

| Check | Result |
|---|---|
| `export fn ... callconv(.kernel)` | Compiles. The crash in Zig issue #35905 did not reproduce |
| Pointer, `u32` and `f32` parameters | Compile and assemble |
| Pointer to an `extern struct` parameter | Compiles and assembles |
| Thread ID through inline asm (`mov.u32 %[r], %tid.x;`) | Compiles and assembles |
| `@workItemId` | Not tested in isolation, because the test kernel also used `@exp` |
| `@exp` | Fails: `no libcall available for fexp` |

Math functions that need a libcall are not available in kernels. Softmax and
cross-entropy need `exp` and `log`, so kernels must implement them.

### Kernel binary layout

For the `fill(ptr, u32, u32)` test kernel, `ptxas` output contains:

- `.text.fill`: the machine code.
- `.nv.constant0.fill`: constant bank 0, 0x170 bytes.
- `.nv.info.fill`: metadata, including `EIATTR_PARAM_CBANK` and
  `EIATTR_CBANK_PARAM_SIZE`.

The disassembly reads parameters from `c[0x0][0x160]` onward. It reads `c[0x0][0x28]`
and `c[0x0][0x118]` too, so the driver-defined part of bank 0 must also be filled in.

### QMD and compute methods

- Layouts: `refs/open-gpu-doc/classes/compute/clc7c0qmd.h` defines `QMDV02_03`,
  `QMDV02_04` and `QMDV03_00`.
- Methods: `refs/open-gpu-doc/classes/compute/clc7c0.h`, including `SEND_PCAS_A` (0x02b4),
  `SEND_SIGNALING_PCAS2_B` (0x02c0), `SET_SHADER_LOCAL_MEMORY_A` (0x0790) and
  `INVALIDATE_SHADER_CACHES_NO_WFI` (0x1698).
- Working example: `refs/mesa/src/nouveau/vulkan/nvk_cmd_dispatch.c` and
  `refs/mesa/src/nouveau/compiler/nak/qmd.rs`. Mesa initializes a v3.0 QMD through
  `clc6c0`.

### Still open

- Whether `sm_86` is right for this GPU in practice. `ptxas` accepts it, but only a run
  on the hardware confirms it.

## Milestone 1 results

Run on the RTX 3060 with a ReleaseSafe build.

| Check | Result |
|---|---|
| Compute class | `0xC7C0` reported, and it is the only compute class |
| Engines | `0x1` (GR0), `0x9`-`0xc` (copy), and others |
| 64 MiB sysmem -> VRAM -> sysmem | All bytes match. 3058 and 2637 MiB/s in one run |
| 1000 submissions of 19 words | Completed, passing both the 64-entry GPFIFO and the 64 KiB pushbuffer limit. Only completion was checked, not the copied data |
| Dependencies | `ldd`: not a dynamic executable |

Layout checks: `gpu/rm/abi.zig` pins sizes and offsets at compile time, with values
from compiling the driver's C headers. `GpuIdInfoV2`, `CopyAlloc`, `WorkSubmitToken`
and `GpfifoSchedule` are not pinned, because those headers were not part of that C
program. They work in the runs above.

### Error notifier

`hObjectError` takes the handle of a sysmem allocation. RM treats it as a memory
notifier (`kchannelGetNotifierInfo`) and writes `NvNotification` on a fault
(`krcErrorWriteNotifier_CPU`), with the error code in `info32`.

Both paths are verified. `fault-test` copies to an unmapped address, and the program
reports `error code 31, status 0xffff` about 50 ms later instead of waiting for the
5-second timeout. Copies still work in the next process.

The kernel log is not readable without elevated rights here (`dmesg` is denied). Fault
checks rely on the notifier, not on the log.

## Milestone 2 results

`dispatch-test` runs `kernels/fill.ptx` on GR0. It worked on the first attempt.

| Threads | Grid x block | Time | Output |
|---|---|---|---|
| 1 | 1 x 1 | 0.024 ms | All values match, no write past the end |
| 128 | 4 x 32 | 0.023 ms | All values match, no write past the end |
| 1048576 | 32768 x 32 | 1.276 ms | All values match, no write past the end |

### What a dispatch needs

Channel setup, once: allocate the channel on engine `0x1` (GR0), bind class `0xC7C0` with
no allocation parameters, then send `SET_OBJECT`, the shared and local memory windows
and `INVALIDATE_SKED_CACHES`.

Per dispatch:

1. Fill constant bank 0 (see below) and a v3.0 QMD, each at a 256-byte aligned address.
2. `SEND_PCAS_A` = QMD address >> 8, then `SEND_SIGNALING_PCAS2_B` = 3.
3. `INVALIDATE_SHADER_CACHES` with data and flush_data. This also waits for idle.
4. `MEM_OP_A`-`D` for a system memory barrier.
5. `SET_REPORT_SEMAPHORE` A-D, one word, for the fence.

### Constant bank 0

| Offset | Content |
|---|---|
| `0x00` | `%ntid.x` (`.y` and `.z` assumed to follow; only `.x` seen in disassembly) |
| `0x0c` | `%nctaid.x` (`.y` and `.z` assumed to follow) |
| `0x28` | Stack pointer. Read by every kernel, used only by kernels with a stack |
| `0x118` | 64-bit value read by every kernel, unused by `fill`. Left at 0, meaning unknown |
| `0x160` | Kernel parameters, located through `.nv.info.<name>` attribute `0x0a` |

### Kernel binary facts

- Register count: top byte of `sh_info` of `.text.<name>`.
- `.nv.info` attribute formats: 1 and 3 have 4 bytes in total, 4 has a u16 size and then
  the payload.
- Kernels with a stack or shared memory are rejected at compile time. Neither is set up.

### Limitations

- `EIATTR_SW2861232_WAR` is present in the metadata and ignored. Its meaning is unknown.
- Code, QMDs and constant banks live in sysmem.

## Milestone 3 results

Kernels are now Zig. `kernels/fill.ptx` was removed. `zig build` compiles each kernel
for nvptx64, runs `ptxas`, and embeds the result.

| Size (x, y, z) | Grid and block | Time | Elements equal to the CPU's result |
|---|---|---|---|
| 1 x 1 x 1 | grid 1,1,1 block 1,1,1 | 0.021 ms | All |
| 5 x 3 x 7 | grid 5,3,7 block 1,1,1 | 0.020 ms | All |
| 12 x 4 x 4 | grid 3,2,2 block 4,2,2 | 0.020 ms | All |
| 256 x 4 x 2 | grid 8,4,2 block 32,1,1 | 0.021 ms | All |
| 2048 x 64 x 8 | grid 64,64,8 block 32,1,1 | 2.103 ms | All (1048576) |

`max_allowed_ulp` permits 2 ULP, but these runs were bit-exact, including `@sqrt`.

### How a kernel is written

- `kernels/<name>.zig` has a pure `pub fn` with the computation, and an `export fn` with
  `callconv(.kernel)` that exists only when compiling for nvptx64.
- The host imports the module `kernels` (`kernels/root.zig`) and calls the same pure
  function to check GPU output. A file can belong to only one module in a compilation,
  so `kernels/abi.zig` is reached through this module, not as a module of its own.
- Thread IDs come from inline assembly in `kernels/rt.zig`.
- Parameters are one `extern struct` passed by value. It appears in bank 0 at `0x160`.

### Facts confirmed from the machine code

- Bank 0 holds `%ntid` at `0x00`, `0x04` and `0x08`. Grids with y and z greater than 1
  produced correct positions, which also relies on these values.
- `%nctaid` at `0x0c` is confirmed for `.x` only. Its `.y` and `.z` positions are still
  assumed, because `fill3d` does not read them.
- `@sqrt` becomes a fast path and a relative call to a slow path in the same `.text`
  section. There are no relocations against the code, so it needs no linking.

## Reference versions

| Directory under `refs/` | Version |
|---|---|
| `open-gpu-kernel-modules` | tag `595.99.02`, commit `e394a40` |
| `open-gpu-doc` | commit `9fdf5c4` (2026-06-09) |
| `mesa` (`src/nouveau` only) | commit `1618a1a` (2026-11-02) |

## Milestone 4 results

`gemm-bench` checks shapes up to 257 x 129 (inner 513), including ones that do not fill
the last thread block. All elements equal the same function run on the CPU. Square
matrices up to 2048 match the `@Vector` CPU version with a difference of exactly 0. See the
README for timings.

Results are exact, not only close, because both sides add in the same order, and neither
contracts `a * b + c` into a fused operation in this code.

## Milestone 5 results

`train` reaches 97.89% test accuracy in 10 epochs. See the README.

### Design

- One batch is 19 dispatches: 3 forward, 1 softmax, 3 x 3 backward, 3 x 2 updates. Each
  dispatch ends with an invalidate that waits for idle, so they run in order. The CPU waits
  only at the end of the batch.
- Each epoch's shuffled data is uploaded whole by the copy engine, and batches are address
  offsets into it. This differs slightly from the plan's "batches are uploaded".
- Metrics are written to system memory, so that the CPU can read them directly.
- `exp` and `log` are in `kernels/math.zig`, shared with the host.

### Pitfall: instruction prefetch past the end of the code

Symptom: Xid 31 during training, reported about one dispatch after the kernel responsible.
It occurred only when a sum was exactly 0.

Cause: the instruction cache prefetches past the end of a kernel. `grad_weight` was `0x1f00`
bytes in an 8 KiB buffer, leaving `0x100` bytes mapped after it. Its division slow path sits at
the end of the code, and a zero numerator runs that path. Prefetch then read an unmapped page.

`loadCode` now allocates 4 KiB more than the code. NVK does the same (`nvk_device.c` allocates
2 KiB more, on NVIDIA's advice).

Evidence, in the order it was gathered:

| Test | Result | Conclusion |
|---|---|---|
| Inspect sections and calls | Calls stay inside each `.text` | Code was not lost in copying |
| QMD register count raised to 64 | Still faulted | Not a register count problem |
| Division probe with 0 and denormals | Correct, no fault | Slow paths work in themselves |
| `grad_bias` alone, all conditions | Correct, no fault | That kernel was not at fault |
| Skip the `grad_bias` dispatch | The next dispatch failed instead | The fault came from an earlier kernel |
| Wait 50 ms after each dispatch | Fault attributed to `grad_weight` | Faults arrive after the fence passes |
| `grad_weight` alone with zeros | Faulted at every size | Depends on data, not on shape |
| Same, with nonzero data | No fault | A zero sum is the trigger |
| Same zeros, 4 KiB padding | No fault | Confirms the cause |

### Consequence for fault checks

A fault can arrive after the fence of the work that caused it. `Queue.wait` checks the notifier
on every return, but that sees only faults reported so far. A kernel that faults at the very end
of a run may need a further `checkFault` after a short delay to be caught.

## Milestone 6: verification

- `strace` of `train 1`: the only files opened are `/dev/nvidiactl`, `/dev/nvidia0`, the four MNIST
  files and the executable. Nothing with "cuda" in its name.
- `ldd`: not a dynamic executable.
- `nvidia-smi` during training: the process is listed with 276 MiB, at 76-100% GPU utilization.
- A build from an empty cache succeeds, and 26 tests pass.

Not verified: the kernel log, which needs elevated rights to read.

### Still unknown

- Bank 0 at `0x118`, and `EIATTR_SW2861232_WAR`. Everything has worked with them ignored.
- `%nctaid.y` and `.z` positions are assumed. No kernel reads them.
- Why `gemm` at 1024 is out of line with its neighbors.
