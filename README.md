# zig-gpu-copy

An experiment: training a neural network on an NVIDIA GPU from Zig, talking directly
to the kernel driver, with no libcuda and no libc.

```text
Usual path:    PyTorch -> libcuda -> NVIDIA driver -> GPU
This project:  Zig ---------------> NVIDIA driver -> GPU
```

It trains a `784-256-128-10` MLP on MNIST to 97.89% test accuracy on an RTX 3060, at
about 5.6 times the speed of the same model on one CPU thread. Both kernels and host
are Zig.

This is a prototype. The driver ABI it depends on is not stable, and it runs only on
the exact driver version it was written for.

## Requirements

- Linux x86_64 with NVIDIA driver **595.99.02** (open kernel module)
- A GPU with classes `0xC7C0`, `0xC7B5`, `0xC56F` and `0xC361`. Tested on an RTX 3060 (GA106).
- Zig 0.17.0
- `ptxas` from the CUDA toolkit, at build time only

## Usage

MNIST is not included. To download it from the PyTorch mirror and verify its checksums:

```sh
mkdir -p data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -fsSL -O "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
done
md5sum -c ../docs/mnist.md5 && gunzip *.gz
cd ..
```

Then, from the project root, because the data path is relative:

```sh
zig build test                    # Tests that need no GPU
zig build -Doptimize=ReleaseFast

./zig-out/bin/zig-gpu-copy probe          # GPU, engines and needed classes
./zig-out/bin/zig-gpu-copy copy-test      # Copy engine round trip through VRAM
./zig-out/bin/zig-gpu-copy fault-test     # Deliberately faults its own channel
./zig-out/bin/zig-gpu-copy dispatch-test  # Zig kernel, compared with the CPU
./zig-out/bin/zig-gpu-copy gemm-bench     # Matrix multiplication
./zig-out/bin/zig-gpu-copy train 10       # MNIST on the GPU, then on the CPU
```

`fault-test` adds an Xid 31 message to the kernel log. That is expected.

## Results

RTX 3060 12 GiB, Intel Core i3-14100F, ReleaseFast. Timings vary between runs, especially
when other programs are using the GPU or CPU.

### MNIST training

SGD with momentum 0.9, learning rate 0.05, batch size 128, 10 epochs, fixed seed.

| | GPU | CPU (one thread, SIMD) |
|---|---|---|
| Time per epoch | 0.513 s | 2.892 s |
| Final test accuracy | 97.89% | 97.89% |
| Final training loss | 0.0110 | 0.0110 |

- On one batch, all gradients differed by exactly 0 between GPU and CPU.
- Loss and accuracy printed for each epoch are identical to the printed four decimals, on
  both devices and across separate runs. Final weights were not compared bit by bit.
- The first GPU epoch took 0.712 s, and later ones about 0.488 s. The cause of the slower
  first epoch has not been investigated. The time per epoch above is the 10-epoch average.
- An earlier run measured 6.1 times, with a slower CPU time of 3.326 s per epoch. Its CPU
  times were less steady, so the figure of 5.6 times from this run is the one to use.
- The GPU time excludes uploading each epoch's shuffled data, about 0.016 s. The CPU figure
  excludes shuffling as well. Both exclude evaluation.
- The CPU baseline uses one thread. A multithreaded baseline would narrow the gap: this
  CPU has 4 cores, so the 5.6 times figure is not GPU against the whole CPU.

### Matrix multiplication

One run of `gemm-bench`. The kernel has one thread per result element and is not tuned.

| n x n | GPU kernel | CPU, one thread | Speedup |
|---|---|---|---|
| 64 | 0.291 ms | 0.038 ms | 0.1x |
| 256 | 0.744 ms | 2.472 ms | 3.3x |
| 512 | 3.833 ms | 15.608 ms | 4.1x |
| 1024 | 50.029 ms | 108.893 ms | 2.2x |
| 2048 | 245.594 ms | 1996.694 ms | 8.1x |

Small matrices are slower on the GPU because of dispatch overhead. The 1024 result is
out of line with its neighbors in every run so far, and this has not been investigated.

## Layout

| Path | Purpose |
|---|---|
| `gpu/rm/` | Kernel ABI, ioctls, client objects. Knows nothing of deep learning |
| `gpu/memory.zig` | Allocation, GPU and CPU mapping |
| `gpu/queue.zig` | GPFIFO channel, fences, fault reporting |
| `gpu/methods.zig` | Typed GPU methods |
| `gpu/copy.zig` | Copy engine |
| `gpu/compute.zig` | QMD, dispatch, kernel binding |
| `gpu/kernel_elf.zig` | Reads `ptxas` output at compile time |
| `kernels/` | Kernels, compiled for nvptx64. Their pure functions also run on the host |
| `ml/` | MNIST, the shared model definition, GPU and CPU models |
| `app/` | Command line |
| `docs/compute-notes.md` | Findings and pitfalls from each milestone |

## Where Zig helped

- **One language.** A kernel's pure function runs on the GPU and on the host, so checking
  the GPU means calling the same code. This is why gradients could be compared exactly.
- **Compile-time checks of hardware facts.** Struct layouts, QMD bit positions, kernel names
  and parameter sizes fail the build when wrong, rather than faulting on the GPU.
- **Typed methods.** Sending a method to the wrong engine does not compile.
- **Compile-time model shape.** All shapes derive from one table, checked against the data.
- **`build.zig` as the pipeline.** One command goes from Zig to PTX to GPU binary to executable.

## Limitations

- One driver version, one GPU family, one GPU at a time.
- Kernels with a stack or shared memory are rejected at compile time, because neither is set
  up. No kernel here has needed them.
- Thread IDs use inline assembly, and `exp` and `log` are written by hand, because the
  nvptx64 target has no libcalls.
- Submission is synchronous at the batch level.
- Some constant bank fields are unknown or assumed. See `docs/compute-notes.md`.

## Third-party sources

`refs/` holds shallow clones for reference and is not part of the package:
[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules),
[open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc) and
[Mesa](https://gitlab.freedesktop.org/mesa/mesa). This project contains no code from them.
Structure layouts, register positions and the generated `gpu/rm/status.zig` were derived from
NVIDIA's headers. Mesa's NVK driver was read as an example of the dispatch sequence.

MNIST is by LeCun, Cortes and Burges.
