# CUDA C++ with CMake — Flox-Managed Toolchain

**Duration**: 6-8 minutes
**GPU required**: Any NVIDIA GPU (compiles without one)

## What This Demo Shows

`nvcc` (CUDA 12.x) requires `gcc <= 13`, but modern Linux
ships gcc 14-15. Developers hit `"unsupported GNU version"`
errors and spend 20-30 minutes per machine juggling
`update-alternatives`, `CUDA_HOST_COMPILER`, and CMake
toolchain files. This demo declares the entire CUDA C++
toolchain — `nvcc`, `gcc13`, `cmake` — in a Flox manifest.
`flox activate` detects GPU hardware, builds a tiled SGEMM
benchmark automatically, and caches the result. No sudo, no
system changes, no Docker.

**One-line pitch**: Eliminate `nvcc`'s "unsupported GNU
version" error — pin gcc 13 in six lines of TOML, build CUDA
C++ anywhere.

## The Pain Points

### nvcc rejects the system gcc (The Core Problem)

CUDA 12.x requires gcc <= 13. Ubuntu 24.04 ships gcc 14.
Fedora 41 ships gcc 14. The result:

```
/usr/include/c++/14/bits/std_function.h:
    error: identifier "is_invocable_r_v"
nvcc fatal: Host compiler targets unsupported OS
```

This is the single most common CUDA build failure on modern
Linux.

**What developers do today**:
- `sudo apt install gcc-13 g++-13`
- `sudo update-alternatives --install /usr/bin/gcc gcc \`
  `/usr/bin/gcc-13 100`
- Set `CUDA_HOST_COMPILER=/usr/bin/gcc-13` in every build
- Write custom CMake toolchain files
- Maintain per-distro setup scripts
- Or give up and use Docker

**What Flox does**: One line in the manifest:
`gcc13.pkg-path = "gcc13"`. Done.

### CMake can't find nvcc or the right gcc

Even after installing a compatible gcc, CMake needs hardcoded
paths that change between machines:

```cmake
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
```

The on-activate hook passes the Flox-provided paths to CMake
automatically.

### "Works on my machine" for CUDA builds

Developer A has gcc 13.2, CUDA 12.4, CMake 3.28. Developer B
has gcc 13.4, CUDA 12.8, CMake 3.30. The manifest pins exact
versions — everyone builds with the same toolchain.

### CI/CD CUDA build environments

Setting up CUDA builds in CI requires custom Docker images or
complex setup scripts. With Flox:
`flox activate -- cmake --build build` on any runner with Flox
installed.

## Demo Flow

### Act 1: The Pain — nvcc vs. System gcc (2 min)

```bash
gcc --version
# gcc (Ubuntu 14.2.0) 14.2.0

nvcc src/gpu_gemm.cu -o gpu-gemm
# /usr/include/c++/14/bits/std_function.h:
#     error: identifier "is_invocable_r_v"
# nvcc fatal: Host compiler targets unsupported OS
```

The fix requires installing gcc-13 alongside gcc-14, running
`update-alternatives`, setting `CUDA_HOST_COMPILER`,
configuring CMake... 20-30 minutes of fighting the toolchain
on every new machine.

### Act 2: The Flox Solution (2-3 min)

```bash
cat .flox/env/manifest.toml
```

Six packages from two catalogs. Not the entire CUDA toolkit —
just `cuda_nvcc` and `cuda_cudart` (compiler and runtime).
From the `flox-cuda` catalog — pre-built, so they download in
seconds. `gcc13` pins a host compiler that nvcc supports.

```bash
flox activate
```

The hook prints toolchain versions, detects the GPU, and
builds the project automatically:

```
============================================================
  CUDA C++ Toolchain (Flox-managed)
============================================================
  gcc          : 13.4.0
  nvcc (CUDA)  : 12.4
  cmake        : 3.30.5
------------------------------------------------------------
  GPU          : NVIDIA A100-SXM4-80GB
  Driver       : 570.86.15
  Status       : GPU ready
============================================================

Building gpu-gemm ...
  [100%] Built target gpu-gemm
```

```bash
gpu-gemm
```

```
SGEMM Benchmark  (2048 x 2048)
------------------------------------------------------------
  Kernel time    : 1.234 ms  (avg of 10 runs)
  Performance    : 1394.2 GFLOPS
  Verification   : PASS
============================================================
```

### Act 3: The Proof — It's Flox, Not the System (1-2 min)

```bash
which gcc        # Flox path, not /usr/bin/gcc
gcc --version    # 13.4.0, not system gcc 14
which nvcc       # Flox-provided
```

System gcc 14 is still there, untouched. No sudo, no
`update-alternatives`. When you deactivate, your system is
exactly as it was.

### Act 4: The Portability Story (1 min)

This manifest is checked into git. New developer clones the
repo, runs `flox activate`, gets the exact same toolchain —
gcc 13.4, CUDA 12.4, cmake 3.30. On Ubuntu, Fedora, NixOS,
bare metal, cloud instances.

Six lines of TOML replaced three pages of setup instructions.
And unlike those instructions, the manifest can't get out of
date — it IS the environment.

## Technical Details

- **CUDA compiler**: `cuda_nvcc` from the `flox-cuda` catalog
  (pre-built, cached)
- **CUDA runtime**: `cuda_cudart` (headers + static libs for
  linking)
- **Host compiler**: `gcc13` — pinned to a version nvcc
  supports
- **Build system**: `cmake` + `gnumake`
- **Kernel**: 16x16 shared-memory tiled SGEMM,
  multi-architecture PTX (sm_75 through sm_90)
- **Performance**: ~500-2000+ GFLOPS depending on GPU

## Talking Points

- **"nvcc requires gcc <= 13, but your distro ships gcc 14.
  Flox pins gcc 13 without touching your system."**
- **"You don't have to install the entire CUDA toolkit. Just
  install what you need."** — This demo uses only `cuda_nvcc`
  and `cuda_cudart`. Need cuBLAS? Add `cudaPackages.cublas`.
- **"The driver is on the host. Everything above the driver is
  in the manifest."**
- **"Five packages, ten lines of TOML, zero pages of setup
  instructions."**
- **"`flox activate` builds the project automatically.
  Subsequent activations are instant."**
- **"Same manifest on every machine — same compiler, same CUDA
  version, same PTX."**
- **"No sudo, no update-alternatives, no Docker."**

## Q&A

**Q: Does this work with gcc 14?**
A: Not with CUDA 12.x — NVIDIA hasn't added gcc 14 support
yet. That's exactly the problem Flox solves. When NVIDIA adds
gcc 14 support, you update one line in the manifest.

**Q: What about clang as a host compiler?**
A: nvcc supports clang on some platforms. The manifest approach
works the same — you'd pin the clang version instead.

**Q: Can I use this in CI/CD?**
A: Yes. `flox activate -- cmake --build build` works in any CI
runner with Flox installed.

**Q: What about cuBLAS / cuDNN / other CUDA libraries?**
A: The `flox-cuda` catalog provides split CUDA packages —
`cudaPackages.cublas`, `cudaPackages.cudnn`,
`cudaPackages.nccl`, etc. Add them to the manifest the same
way.

**Q: Why `flox-cuda` packages instead of the base catalog?**
A: The same CUDA packages are available in both catalogs, but
`flox-cuda` packages are pre-built and cached. Without it,
Flox would build CUDA packages from source locally, which is
slow.

**Q: What CUDA architectures does the binary support?**
A: sm_75 through sm_90 (Turing through Hopper). Configured in
`src/CMakeLists.txt`.

**Q: How is this different from Docker?**
A: Docker gives you an isolated filesystem. Flox gives you an
isolated toolchain in your existing workflow — same shell, same
editor, same git. No Dockerfile, no volume mounts. And Flox
can produce Docker images via `flox containerize` if needed.

**Q: Do I need the NVIDIA driver installed separately?**
A: Yes — the driver is a kernel module, so it has to be on the
host. Everything above the driver — toolkit, compiler, runtime,
libraries — is what Flox manages.

**Q: What if NVIDIA adds gcc 14 support?**
A: Update one line: `gcc13` becomes `gcc14`. That's it.

## Setup Requirements

### Hardware
- Any x86_64 Linux machine with NVIDIA GPU (for running the
  benchmark)
- Without GPU: compiles successfully but cannot execute

### Software
- Flox installed
- NVIDIA driver installed on the host

### Preparation
- Run `flox activate` once to pre-cache the build
- Test `gpu-gemm` runs and prints PASS

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvcc` not found after activation | Check `cuda_nvcc` is in manifest; verify `which nvcc` |
| CMake can't find CUDA compiler | Hook passes `-DCMAKE_CUDA_COMPILER=$(which nvcc)` — check hook output |
| Build fails with "unsupported GNU version" | Verify gcc13 is in manifest and `gcc --version` shows 13.x |
| `gpu-gemm` segfaults or wrong results | Check GPU memory; try smaller N: `gpu-gemm 512` |
| "No CUDA devices found" | NVIDIA driver not installed or GPU not present; binary still compiles |
| Build takes too long | First build caches in `$FLOX_ENV_CACHE/build`; subsequent activations skip |
