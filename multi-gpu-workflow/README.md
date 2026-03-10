# Multi-GPU Workflow with Driver Compatibility Validation

**Duration**: 8-10 minutes | **GPU required**: Laptop (no GPU) + cloud GPU instances (A100, H100)

## What This Demo Shows

A Flox environment that eliminates the CUDA compatibility gauntlet — driver version, toolkit version, cuDNN, PyTorch wheel variant, and libstdc++ runtime — by declaring the entire stack in a single manifest. The on-activate hook detects GPU hardware, validates driver/toolkit compatibility, and selects the correct PyTorch wheels (or falls back to CPU with a clear explanation). Includes real LLM inference with Qwen2.5-0.5B-Instruct across all hardware targets.

**One-line pitch**: Develop locally on your laptop, activate on any cloud GPU — same manifest, same command, it adapts to what's there.

## The Pain Points

### pip silently installs CPU-only PyTorch
`pip install torch` on a GPU machine pulls CPU-only wheels by default. No warning. `torch.cuda.is_available()` returns `False`. Developers don't discover this until training runs 10-100x slower than expected. The hook solves this by detecting the GPU and selecting the correct pip index URL automatically.

### The four-way version lock
Driver, toolkit, cuDNN, and framework must all be compatible. Get any one wrong and you get cryptic errors or silent failures. The manifest pins toolkit + Python version. The hook pins the matching PyTorch wheel. One place to manage, not four separate compatibility lookups.

### System library mismatches
PyTorch needs `libstdc++.so.6` with recent GLIBCXX symbols, but bare cloud instances vary. `gcc-unwrapped` in the manifest pins the exact C++ runtime. Without it, PyTorch imports fail with `GLIBCXX_3.4.30 not found`.

### Cloud GPU idle time
At $2-4/hr for an A100, 30 minutes debugging CUDA compatibility is real money. `flox activate` + the idempotent hook cuts repeat setup to seconds.

### "Works on my machine"
Share the manifest, everyone gets the same stack. No more "which CUDA version are you on?" The manifest is the single source of truth.

### conda's CUDA confusion
conda has three different `cudatoolkit` packages, `LD_LIBRARY_PATH` conflicts with system CUDA, and the conda-forge version doesn't include `nvcc`. Flox provides a single coherent `cudatoolkit` from Nixpkgs with `nvcc` included, isolated from system libraries.

See [GPU_CUDA_PAIN_POINTS.md](../GPU_CUDA_PAIN_POINTS.md) for the full inventory.

## How It Works

**Level 1 — Declarative manifest packages** (Flox/Nix resolver): The manifest declares `python3`, `uv`, `cudatoolkit`, `gcc-unwrapped`, and `pciutils`. These resolve to identical packages on every `x86_64-linux` machine — laptop, A100, H100, doesn't matter. The resolver does not see GPU hardware.

**Level 2 — On-activate hook** (bash, runs during `flox activate`): The hook detects GPU presence via `nvidia-smi` (preferred) or `lspci` (fallback), validates driver compatibility, then uses `uv` to install the correct PyTorch wheel — CUDA 12.8 wheels on a GPU machine, CPU-only wheels on a laptop. A persistent venv in `$FLOX_ENV_CACHE/venv` with a marker file makes this idempotent across activations.

A100 and H100 get the **same packages and the same PyTorch wheels**. The CUDA wheels contain kernels for multiple GPU architectures. PyTorch dispatches the right kernels at runtime based on compute capability. Nothing needs to change when you move between GPUs — and that's the point.

## Demo Flow

### Act 1: The Pain — Silent CPU Fallback (2 min)

On a cloud GPU instance (outside Flox):
```bash
nvidia-smi                    # A100 detected, driver looks good
pip install torch             # Installs successfully, no warnings
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# False — no warning, no error
```

pip doesn't know about your GPU. It gives you CPU wheels and says nothing.

### Act 2: Introduce Flox — Local Development (2 min)

On your laptop (no GPU):
```bash
cat .flox/env/manifest.toml   # Declarative: python3, uv, cudatoolkit, gcc
flox activate                  # Hook detects no GPU -> CPU-only wheels
python test_gpu.py --quick     # CPU-only | PyTorch 2.x.x
python test_gpu.py --infer     # ~4 tok/s on CPU. Remember that number.
```

The driver is on the host. Everything above the driver is in the manifest.

### Act 3: Same Environment on A100 (2-3 min)

```bash
flox activate                  # Hook detects A100 -> CUDA 12.8 wheels
python test_gpu.py             # Full report: A100, Ampere, 80 GB
python test_gpu.py --infer     # ~70+ tok/s. Same model, same manifest. 17x faster.
```

### Act 4: Same Environment on H100 — The Punchline (2-3 min)

```bash
flox activate                  # Same hook, same wheels
python test_gpu.py             # H100, Hopper, compute capability 9.0
python test_gpu.py --infer     # ~100+ tok/s
```

"I changed nothing because nothing needed to change."

### Visual Summary

| Instance | GPU | What Happens | Inference Speed |
|----------|-----|-------------|-----------------|
| Local laptop | None | Hook detects no GPU -> CPU-only PyTorch wheels | ~4 tok/s |
| Cloud A100 | A100 (Ampere, sm_80) | Hook detects GPU -> CUDA 12.8 PyTorch wheels | ~70+ tok/s |
| Cloud H100 | H100 (Hopper, sm_90) | Hook detects GPU -> CUDA 12.8 PyTorch wheels (same!) | ~100+ tok/s |

**Same manifest.toml. Same hook. Same wheels on GPU machines. It just works.**

## test_gpu.py

The included diagnostic tool has four modes:

```bash
python test_gpu.py           # Full environment report
python test_gpu.py --quick   # One-line summary
python test_gpu.py --bench   # Matrix multiply benchmark (CPU vs GPU)
python test_gpu.py --infer   # LLM inference with Qwen2.5-0.5B-Instruct
python test_gpu.py --infer --prompt "Your question here"  # Custom prompt
```

## Talking Points

- **"Develop locally, deploy to any GPU, zero reconfiguration"** — manifest is unchanged across machines
- **"Same manifest, same activation — it adapts to what's there"** — hook detects hardware, installs correct wheels
- **"No more pip-installing the wrong PyTorch wheel"** — hook selects the right index URL automatically
- **"Your CUDA stack travels with you — toolkit, libraries, Python, all pinned"** — manifest pins them declaratively
- **"The driver is on the host. Everything above the driver is in the manifest."** — Flox manages the full userspace CUDA stack
- **"You don't have to install the entire CUDA toolkit — install only what you need."** — The C++ demo shows the granular approach: just nvcc + cudart
- **"Setup time on cloud GPUs goes from hours to seconds"** — idempotent hook, cached venv
- **"I changed nothing because nothing needed to change"** — A100 -> H100 punchline

## Q&A

**Q: What's different between the A100 and H100 install?**
A: Nothing — and that's the point. The CUDA PyTorch wheels contain kernels for both Ampere and Hopper architectures. PyTorch dispatches the right ones at runtime based on compute capability.

**Q: Does Flox serve different packages for different GPUs?**
A: The Nix resolver provides the same packages on every x86_64-linux machine. The on-activate hook handles the GPU-vs-CPU distinction — that's the real binary choice. Within GPU machines, the same CUDA wheels work across architectures.

**Q: Does this work with other cloud providers?**
A: Yes. Flox works anywhere — AWS, GCP, Azure, on-prem, bare metal. The Flox environment is provider-agnostic.

**Q: What if I need a specific CUDA version?**
A: Pin it in the manifest. Currently we use `cudatoolkit ^12.8` and the hook targets `cu128` PyTorch wheels. Both are configurable.

**Q: Do I need anything installed on the host?**
A: Just the NVIDIA driver — that's a kernel module, it has to be on the machine. Flox takes over from there.

**Q: Can I install just pieces of CUDA?**
A: Yes. For Python/ML workflows, the full cudatoolkit is convenient. For C++ builds, you can install just `cuda_nvcc` and `cuda_cudart`. See the [CUDA C++ demo](../cuda-cpp-cmake/).

**Q: Does this redownload everything each time?**
A: No. The hook checks a marker file. If the correct PyTorch variant is already installed, it skips entirely. First activation installs; subsequent activations are instant.

**Q: What about TensorFlow?**
A: Same approach. Adjust the hook to install TensorFlow with the right CUDA variant.

**Q: Is the GPU detection done by the Flox resolver?**
A: No. The Nix resolver provides identical packages on all x86_64-linux machines. GPU detection happens in the on-activate hook (a bash script). Future Flox versions may bring hardware-aware resolution to the resolver itself.

## Setup Requirements

### Hardware
- Local development machine (laptop, no GPU required)
- Cloud GPU instances with A100 and H100 (e.g., Brev, AWS, GCP)

### Software
- Flox installed locally and on GPU instances
- NVIDIA driver installed on GPU instances

### Preparation
- Run `flox activate` once on each machine to populate the cached venv
- Pre-download the Qwen2.5-0.5B-Instruct model (caches in `~/.cache/huggingface/hub/`)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Hook shows "No NVIDIA GPU" on GPU instance | Check that nvidia-smi is installed and working; verify driver |
| PyTorch import fails with libstdc++ error | Verify gcc-unwrapped is in manifest; check LD_LIBRARY_PATH |
| `torch.cuda.is_available()` returns False after hook | Check hook output — was cu128 or cpu variant installed? |
| First activation slow | Expected on first run (uv downloads wheels). Subsequent activations are instant. |
| `ImportError: No module named 'transformers'` | Delete cached venv: `rm -rf $FLOX_ENV_CACHE/venv` then re-activate |
