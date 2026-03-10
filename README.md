# GTC 2026 — Flox GPU/CUDA Demos

Demo materials from the Flox booth at NVIDIA GTC 2026, showing how [Flox](https://flox.dev) eliminates GPU/CUDA environment pain.

These demos illustrate a core idea: **the NVIDIA driver lives on the host (it's a kernel module), but everything above the driver — toolkit, compiler, runtime, libraries — is declared in a Flox manifest.** One manifest, any machine, any GPU.

## Demos

### [Introduction to Flox](intro-to-flox/)
**3-5 minutes** | No GPU required

What Flox is and why declarative, portable development environments matter. Establishes the foundation before the GPU-specific demos.

### [Multi-GPU Workflow](multi-gpu-workflow/)
**8-10 minutes** | Laptop + cloud GPU instances (A100, H100)

Develop locally on your laptop, activate on any cloud GPU — same manifest, same command. The on-activate hook detects GPU hardware and installs the correct PyTorch wheels automatically. Includes real LLM inference with Qwen2.5-0.5B-Instruct: ~4 tok/s on CPU, ~70+ on A100, ~100+ on H100.

### [CUDA C++ with CMake](cuda-cpp-cmake/)
**6-8 minutes** | Any NVIDIA GPU

Eliminate `nvcc`'s "unsupported GNU version" error. Pin gcc 13 in six lines of TOML, build CUDA C++ anywhere. Compiles and runs a tiled SGEMM benchmark — no sudo, no `update-alternatives`, no Docker.

## Background: GPU/CUDA Pain Points

See [GPU_CUDA_PAIN_POINTS.md](GPU_CUDA_PAIN_POINTS.md) for a comprehensive inventory of the 51 real-world pain points developers face with NVIDIA GPUs and CUDA. The demos above address several of the most common ones.

## Getting Started

1. **Install Flox**: https://flox.dev/docs/install-flox/
2. **Pick a demo** and follow its README
3. **Run `flox activate`** — that's it

Each demo directory contains a `.flox/` environment. Clone the repo, `cd` into a demo, and `flox activate`. The manifest and hooks handle the rest.

## Requirements

- **Flox** installed ([install guide](https://flox.dev/docs/install-flox/))
- **NVIDIA driver** installed on the host (for GPU demos)
- **No other CUDA installation needed** — Flox provides the toolkit, compiler, and libraries
