# GPU/CUDA Developer Pain Points — Comprehensive Reference

A categorized inventory of real-world pain points developers face when working with NVIDIA GPUs and CUDA. Each entry describes the concrete problem, the current workaround, whether it's a setup-time or runtime issue, and (where applicable) a representative error message or developer quote.

This document serves as a reference for demo narratives, feature planning, and marketing positioning. It is intentionally exhaustive — not every entry needs to appear in a demo.

---

## 1. CUDA Setup & Installation

### 1.1 nvidia-smi vs nvcc version confusion

**Problem**: `nvidia-smi` reports the driver's maximum supported CUDA version, while `nvcc --version` reports the installed toolkit version. Developers routinely confuse these, believing they have a CUDA version they don't actually have installed.

**Example**:
```
$ nvidia-smi
CUDA Version: 12.4    ← driver supports UP TO 12.4

$ nvcc --version
Cuda compilation tools, release 11.8   ← actually installed toolkit
```
A developer sees "CUDA 12.4" in nvidia-smi and installs PyTorch for CUDA 12.4. The toolkit is 11.8. Nothing works.

**Current workaround**: Learn the distinction (usually after hours of debugging). Check both commands. Cross-reference manually.

**Classification**: Setup-time

---

### 1.2 pip silently installs CPU-only PyTorch

**Problem**: Running `pip install torch` on a machine with a GPU pulls CPU-only wheels by default. There is no warning. `torch.cuda.is_available()` silently returns `False`. Developers believe they have GPU support and don't discover the problem until training runs 10-100x slower than expected — or until they explicitly check.

**Example**:
```python
$ pip install torch
# ... installs successfully, zero warnings ...

$ python -c "import torch; print(torch.cuda.is_available())"
False
# Developer doesn't check this. Starts training. Wonders why it's slow.
```

**Developer quote**: "I mass-deployed to our GPU cluster and everything was silently running on CPU for two weeks. We burned $40k in GPU hours doing nothing."

**Current workaround**: Memorize the correct pip index URL for your CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
But which CUDA version? See pain point 1.1.

**Classification**: Setup-time (but manifests as runtime performance issue)

---

### 1.3 cuDNN manual install ritual

**Problem**: NVIDIA requires an NVIDIA Developer account to download cuDNN. The download page requires accepting a license, selecting the exact version matching your CUDA toolkit, downloading a tarball, and manually copying files to the right system directories. Package managers don't reliably provide it.

**Example**:
```bash
# The cuDNN install ritual:
# 1. Go to developer.nvidia.com
# 2. Create account / log in
# 3. Accept license agreement
# 4. Find the right version for your CUDA
# 5. Download .tar.xz
# 6. Extract and copy:
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
# 7. Verify:
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

**Current workaround**: Follow the ritual. Some teams bake it into Docker images and never touch it again.

**Classification**: Setup-time

---

### 1.4 Multi-day CUDA debugging is normalized

**Problem**: The CUDA ecosystem has so many interacting components (driver, toolkit, cuDNN, framework, Python version, OS libraries) that multi-day debugging sessions for initial setup are considered normal. Teams budget days, not hours, for CUDA environment setup.

**Developer quote**: "Our onboarding doc has a section called 'CUDA Setup (expect 1-2 days).' Nobody questions this."

**Current workaround**: Senior developers maintain tribal knowledge. Setup documentation is constantly outdated. Some teams mandate Docker images to avoid the problem entirely.

**Classification**: Setup-time

---

### 1.5 WSL2 driver overwrite trap

**Problem**: On Windows Subsystem for Linux 2, installing CUDA toolkit inside WSL can overwrite the Windows-provided GPU driver with a Linux one, breaking GPU passthrough entirely. The standard Linux CUDA install instructions are wrong for WSL2.

**Example**:
```bash
# Standard install (BREAKS WSL2):
sudo apt-get install cuda

# What you actually need:
sudo apt-get install cuda-toolkit-12-4
# Notice: cuda-toolkit, NOT cuda (which pulls in the driver)
```

**Current workaround**: Know to use `cuda-toolkit` instead of `cuda`. Learn this the hard way.

**Classification**: Setup-time

---

### 1.6 CUDA toolkit not on PATH after install

**Problem**: The NVIDIA CUDA installer places binaries in `/usr/local/cuda/bin` and libraries in `/usr/local/cuda/lib64`, but doesn't update `PATH` or `LD_LIBRARY_PATH`. Every new shell needs manual configuration.

**Example**:
```bash
$ nvcc --version
Command 'nvcc' not found

# Need to add to .bashrc:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Current workaround**: Add to shell profile. Forget when switching to a new machine or container.

**Classification**: Setup-time

---

## 2. The Compatibility Matrix

### 2.1 The four-way version lock

**Problem**: Driver, toolkit, cuDNN, and framework must all be mutually compatible. There are no clear error messages when they're not — just cryptic failures. The compatibility matrix is documented in scattered tables across NVIDIA's docs, PyTorch's docs, and TensorFlow's docs.

**Example**: To run PyTorch 2.3 with CUDA, you need:
- NVIDIA driver >= 525.60
- CUDA toolkit 12.1 (not 12.0, not 12.2 without checking)
- cuDNN 8.9.x (matching CUDA 12.1)
- Python 3.8-3.12 (not 3.13)

Get any one wrong and you see:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
or simply `torch.cuda.is_available()` returning False with no explanation.

**Developer quote**: "I keep a spreadsheet of which CUDA/cuDNN/PyTorch versions work together. It has 47 rows."

**Current workaround**: Spreadsheets, bookmarked compatibility tables, trial and error.

**Classification**: Setup-time

---

### 2.2 PyTorch and TensorFlow need different CUDA versions

**Problem**: PyTorch 2.3 ships wheels for CUDA 12.1. TensorFlow 2.16 requires CUDA 12.3. If you need both frameworks (common in research), you can't have one system CUDA that satisfies both. System-wide CUDA installs become impossible.

**Current workaround**: Separate virtual environments with framework-specific CUDA. Or give up and use one framework. Or use Docker containers per framework.

**Classification**: Setup-time

---

### 2.3 pip ignores GPU driver version

**Problem**: pip has no concept of hardware. `pip install torch` does not check what GPU you have, what driver is installed, or what CUDA version is available. It just installs whatever wheel matches the platform and Python version. This means pip will happily install CUDA 12.1 wheels on a machine with a driver that only supports up to CUDA 11.8.

**Current workaround**: Manually check driver version, look up CUDA compatibility, select the right pip index URL.

**Classification**: Setup-time

---

### 2.4 New GPU architectures break existing framework builds

**Problem**: When NVIDIA releases a new GPU architecture (e.g., Hopper, Blackwell), existing framework builds don't include kernels for the new compute capability. PyTorch wheels compiled for sm_80 (Ampere) may work on sm_90 (Hopper) via forward compatibility, but performance may suffer, and some operations may fail.

**Example**:
```
CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call
```

**Current workaround**: Wait for updated framework releases. Build from source with new compute capability flags. Use NVIDIA's containers which tend to update faster.

**Classification**: Setup-time / Runtime

---

### 2.5 CUDA minor version incompatibilities

**Problem**: Even within a major CUDA version, minor versions can introduce breaking changes. CUDA 12.0, 12.1, 12.2, and 12.3 are not fully interchangeable. Libraries compiled against CUDA 12.1 may crash with CUDA 12.3 runtime.

**Current workaround**: Pin exact CUDA versions everywhere. Rebuild everything when upgrading.

**Classification**: Setup-time / Runtime

---

## 3. Cloud GPU Instances

### 3.1 Setup complexity on every new instance

**Problem**: Cloud GPU instances come with base OS images. CUDA is sometimes pre-installed, sometimes not, and the pre-installed version may not match what your project needs. Every new instance requires environment setup. Spot instances make this worse — you lose your setup when the instance is reclaimed.

**Developer quote**: "I spend the first 45 minutes of every GPU session installing the same stuff. At $3.50/hr for an A100, that's $2.60 just to get started."

**Current workaround**: Custom AMIs/images. Docker. Ansible scripts. Or just accept the waste.

**Classification**: Setup-time

---

### 3.2 Cloud GPU pricing and hidden costs

**Problem**: GPU instances are expensive ($2-4/hr for A100, $8-12/hr for H100). But the hidden cost is setup time: if 30-60 minutes of every session is environment setup, that's $1-4 per session wasted. For teams running dozens of sessions per day, this is thousands of dollars monthly.

**Current workaround**: Pre-built images, persistent storage, accepting the cost.

**Classification**: Setup-time (cost dimension)

---

### 3.3 Spot/preemptible instance preemption

**Problem**: The cheapest GPU instances are spot/preemptible, but they can be reclaimed at any time. If your environment setup takes 30+ minutes and your spot instance is reclaimed after 20 minutes, you've lost both time and money with zero useful work done.

**Current workaround**: Use on-demand (more expensive). Checkpoint frequently. Docker images with pre-baked environments. Accept the risk.

**Classification**: Setup-time / Runtime

---

### 3.4 GPU availability waitlists

**Problem**: High-demand GPU types (A100, H100) often have waitlists or are unavailable in preferred regions. Teams can't always get the GPU they designed their workflow for, and switching GPU types means re-doing environment setup (see category 6).

**Current workaround**: Multi-region strategies. Accept whatever GPU is available. Over-provision.

**Classification**: Setup-time (logistics dimension)

---

### 3.5 Proprietary silicon fragmentation

**Problem**: As Google TPUs, AWS Trainium/Inferentia, and other accelerators emerge alongside NVIDIA GPUs, teams need environments that work across hardware vendors. Current CUDA tooling is NVIDIA-specific by design.

**Current workaround**: Separate codepaths per hardware vendor. Framework abstraction layers (JAX, etc.). Vendor lock-in.

**Classification**: Setup-time / Runtime

---

## 4. Multi-GPU & Multi-Node

### 4.1 NCCL timeout errors

**Problem**: NVIDIA Collective Communications Library (NCCL) is used for multi-GPU and multi-node communication. Timeout errors are common and opaque. They can be caused by network configuration, firewall rules, NCCL version mismatches, or subtle GPU topology issues.

**Example**:
```
RuntimeError: NCCL communicator was aborted on rank 1.
  Original reason for abortion was: NCCL timeout.
```

**Current workaround**: Set `NCCL_DEBUG=INFO`, increase `NCCL_TIMEOUT`, check network configuration, try different NCCL backends. Often requires trial and error across multiple environment variables.

**Classification**: Runtime

---

### 4.2 DDP constructor hangs

**Problem**: PyTorch DistributedDataParallel (DDP) hangs silently during construction if any rank fails to connect. No error message — just an infinite hang. The root cause is usually environment mismatch (different NCCL versions, different CUDA versions) across nodes.

**Example**:
```python
model = DistributedDataParallel(model, device_ids=[local_rank])
# ... hangs forever, no output, no error ...
```

**Developer quote**: "It just sits there. No error. No timeout for 30 minutes. Then 'NCCL timeout.' I've lost entire afternoons to this."

**Current workaround**: Verify all nodes have identical CUDA/NCCL versions (manually). Set aggressive timeouts. Add debug logging.

**Classification**: Runtime (caused by setup-time issues)

---

### 4.3 Multi-node scaling slower than expected

**Problem**: Developers expect linear scaling with multi-node GPU training but see sublinear or even negative scaling due to communication overhead, bandwidth bottlenecks, or incorrect topology configuration.

**Current workaround**: Profile with `NCCL_DEBUG`, tune batch sizes, use gradient compression, accept sublinear scaling.

**Classification**: Runtime

---

### 4.4 P2P GPU communication silent failures

**Problem**: Peer-to-peer (P2P) GPU communication (GPUDirect) can silently fall back to slower paths (through host memory) when P2P is not available between GPU pairs. No error is raised — performance just degrades.

**Current workaround**: Check P2P connectivity with `nvidia-smi topo -m`. Configure GPU affinity manually.

**Classification**: Runtime

---

### 4.5 Container shared memory limits

**Problem**: Docker's default shared memory (`/dev/shm`) is 64MB. PyTorch DataLoader with `num_workers > 0` uses shared memory for IPC. Multi-GPU training multiplies this. Containers crash with "bus error" — a completely opaque error message.

**Example**:
```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```

**Current workaround**: `docker run --shm-size=8g` or `--ipc=host`. But you have to know this in advance.

**Classification**: Setup-time (container configuration)

---

### 4.6 IOMMU interference with GPU communication

**Problem**: IOMMU (Input-Output Memory Management Unit) can interfere with GPU P2P communication and NVLink transfers, causing performance degradation. Cloud instances may have IOMMU enabled by default.

**Current workaround**: Disable IOMMU in BIOS/GRUB (not possible on cloud). Set `iommu=off` kernel parameter. Accept the performance hit.

**Classification**: Setup-time / Runtime

---

## 5. Docker/Container GPU Issues

### 5.1 GPU invisible inside container

**Problem**: By default, Docker containers cannot see host GPUs. Requires NVIDIA Container Toolkit (nvidia-docker2/nvidia-container-toolkit), the `--gpus all` flag, and compatible driver versions. Missing any piece results in no GPU access with no helpful error.

**Example**:
```bash
$ docker run -it pytorch/pytorch python -c "import torch; print(torch.cuda.is_available())"
False
# Forgot --gpus all

$ docker run --gpus all -it pytorch/pytorch python -c "import torch; print(torch.cuda.is_available())"
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
# NVIDIA Container Toolkit not installed
```

**Current workaround**: Install NVIDIA Container Toolkit. Always use `--gpus all`. Know the incantation.

**Classification**: Setup-time

---

### 5.2 OS/driver upgrades break container GPU passthrough

**Problem**: Upgrading the host OS kernel or NVIDIA driver can break GPU passthrough to containers. The NVIDIA Container Toolkit depends on specific driver interfaces. An `apt upgrade` can silently break all GPU containers.

**Current workaround**: Pin driver versions. Test upgrades in staging. Hold NVIDIA packages during upgrades.

**Classification**: Setup-time

---

### 5.3 SELinux/AppArmor blocks GPU access

**Problem**: Security modules like SELinux (RHEL/CentOS) or AppArmor (Ubuntu) can block GPU device access from containers. Error messages reference security contexts, not GPUs, making them hard to diagnose.

**Example**:
```
OCI runtime create failed: [...] permission denied
```

**Current workaround**: Add SELinux exceptions. Use `--security-opt` flags. Disable security modules (bad practice).

**Classification**: Setup-time

---

### 5.4 NVIDIA_VISIBLE_DEVICES misconfiguration

**Problem**: The `NVIDIA_VISIBLE_DEVICES` environment variable controls which GPUs a container sees. Misconfiguration (wrong UUID, wrong index, not set) results in no GPU access. In orchestrated environments (Kubernetes), this is especially error-prone.

**Current workaround**: Always use `all` or carefully manage device UUIDs. Debug with `nvidia-smi -L` inside container.

**Classification**: Setup-time

---

### 5.5 Container/host CUDA driver version mismatch

**Problem**: The CUDA runtime inside the container must be compatible with the driver on the host. If the container's CUDA version is newer than the host driver supports, you get cryptic errors or silent failures — but there's no check at container start.

**Example**:
```
CUDA driver version is insufficient for CUDA runtime version
```

**Current workaround**: Match container CUDA version to host driver capability. Check NVIDIA's compatibility matrix.

**Classification**: Setup-time

---

## 6. Switching Between GPU Types

### 6.1 Compute capability mismatch

**Problem**: CUDA code compiled for one compute capability (e.g., sm_80 for A100) may not run on a different GPU (e.g., sm_70 for V100). Pre-compiled wheels include kernels for common capabilities, but edge cases exist. Custom CUDA extensions must be recompiled.

**Example**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Current workaround**: Compile with multiple `-gencode` flags. Use JIT compilation. Accept that switching GPUs means recompiling.

**Classification**: Setup-time

---

### 6.2 Performance tuning is GPU-specific

**Problem**: Optimal batch sizes, learning rates, memory configurations, and data loading strategies differ between GPU types. Code tuned for A100 (80GB HBM2e) will OOM on a T4 (16GB GDDR6) or underutilize an H100 (80GB HBM3).

**Current workaround**: Per-GPU configuration profiles. Auto-tuning scripts. Accept suboptimal performance when switching.

**Classification**: Runtime

---

### 6.3 Framework support lags new GPU architectures

**Problem**: When NVIDIA releases new GPUs (e.g., Blackwell), it takes weeks to months for PyTorch, TensorFlow, and other frameworks to release builds with optimized kernels. Early adopters are stuck building from source or accepting forward-compatibility mode with degraded performance.

**Current workaround**: Wait. Build from source. Use NVIDIA's NGC containers (which update faster than pip packages).

**Classification**: Setup-time

---

### 6.4 Consumer vs datacenter GPU feature gaps

**Problem**: Consumer GPUs (RTX 4090) and datacenter GPUs (A100, H100) have different capabilities: different memory sizes, NVLink availability, MIG support, ECC memory. Code developed on consumer hardware may not transfer cleanly to datacenter, and vice versa.

**Current workaround**: Test on target hardware. Accept that local development on consumer GPU is approximate.

**Classification**: Setup-time / Runtime

---

## 7. conda/mamba CUDA Issues

### 7.1 conda silently installs CPU PyTorch

**Problem**: `conda install pytorch` defaults to CPU builds unless you explicitly specify the cuda channel/variant. Same silent failure as pip (pain point 1.2), but in conda's ecosystem.

**Example**:
```bash
$ conda install pytorch torchvision torchaudio -c pytorch
# Installs CPU version. No warning.
# Need:
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Current workaround**: Always specify `pytorch-cuda=XX.X` and both channels. Memorize the incantation.

**Classification**: Setup-time

---

### 7.2 Three different cudatoolkit packages

**Problem**: conda has (at least) three packages that sound like they provide CUDA:
- `cudatoolkit` (conda-forge) — runtime libraries only, **no nvcc**
- `cuda-toolkit` (nvidia channel) — full toolkit with nvcc
- `cuda` (nvidia channel) — metapackage

They are not interchangeable. Using the wrong one breaks compilation workflows. The naming similarity is a trap.

**Developer quote**: "I've installed `cudatoolkit` three times in different environments before learning it doesn't include the compiler."

**Current workaround**: Know which package provides what. Use the `nvidia` channel for compilation work.

**Classification**: Setup-time

---

### 7.3 conda CUDA conflicts with system CUDA

**Problem**: When conda installs its own CUDA libraries, they can conflict with the system CUDA installation via `LD_LIBRARY_PATH`. The conda libraries may be a different version than system libraries, causing symbol conflicts, incorrect library loading, or crashes.

**Example**:
```
libcublas.so.12: version `libcublas.so.12' not found (required by /usr/lib/x86_64-linux-gnu/libcublas.so.12)
```

**Current workaround**: Isolate environments. Set `LD_LIBRARY_PATH` carefully. Purge system CUDA when using conda (or vice versa).

**Classification**: Setup-time / Runtime

---

### 7.4 `__cuda` virtual package conflicts

**Problem**: conda uses a virtual package `__cuda` to detect the system CUDA driver version. This can conflict with explicitly specified CUDA versions, causing confusing solver failures like "package requires __cuda >=12 but __cuda==11.8 is installed."

**Current workaround**: Override with environment variables. Use `--override-channels`. Manual package pinning.

**Classification**: Setup-time

---

### 7.5 conda-forge cudatoolkit has no nvcc

**Problem**: The most commonly installed CUDA package in conda (`cudatoolkit` from conda-forge) does not include the CUDA compiler (`nvcc`). Developers who need to compile CUDA code discover this only after installation when `nvcc` is not found.

**Example**:
```bash
$ conda install cudatoolkit
# ... installs successfully ...
$ nvcc --version
nvcc: command not found
```

**Current workaround**: Install from the `nvidia` channel. Or install `cuda-toolkit` instead. Know the difference.

**Classification**: Setup-time

---

### 7.6 conda version pinning failures

**Problem**: conda's solver can produce environments where CUDA package versions are technically compatible according to metadata but practically incompatible at runtime. The solver doesn't understand GPU driver requirements.

**Current workaround**: Explicit version pinning. Manual testing. Lock files.

**Classification**: Setup-time

---

## 8. NGC (NVIDIA GPU Cloud) Container Issues

### 8.1 Massive image sizes

**Problem**: NGC containers are 10-15+ GB in size. Pulling them takes significant time and storage, especially on cloud instances with limited bandwidth or disk.

**Developer quote**: "Our CI pipeline spends 20 minutes just pulling the NGC container before any tests run."

**Current workaround**: Pre-pull images. Use docker layer caching. Accept the wait.

**Classification**: Setup-time

---

### 8.2 Confusing versioning scheme

**Problem**: NGC container tags follow a scheme like `23.05-py3` (year.month-python) that doesn't directly indicate CUDA version, framework version, or driver requirements. Finding which tag matches your requirements requires consulting a compatibility matrix.

**Current workaround**: Consult release notes per tag. Trial and error.

**Classification**: Setup-time

---

### 8.3 Outdated framework versions

**Problem**: NGC containers bundle specific framework versions that may lag behind the latest release. You can't easily upgrade PyTorch inside an NGC container without risking CUDA/cuDNN incompatibilities.

**Current workaround**: Accept the lag. Build custom images. Mix NGC base with custom framework install (fragile).

**Classification**: Setup-time

---

### 8.4 Host driver requirements

**Problem**: Each NGC container release requires a minimum host driver version. If your host driver is too old (common on cloud instances with older images), the container won't work. The error message is often unhelpful.

**Current workaround**: Check NGC support matrix. Upgrade host driver (requires admin access, may break other things).

**Classification**: Setup-time

---

### 8.5 Not general-purpose environments

**Problem**: NGC containers are optimized for specific ML frameworks but are poor general-purpose development environments. They lack common tools, use unfamiliar base images, and make it hard to install additional system packages.

**Current workaround**: Build custom layers on top. Use multi-stage builds. Accept the limitations.

**Classification**: Setup-time

---

### 8.6 Inherited CVEs

**Problem**: NGC containers inherit vulnerabilities from their base images and bundled libraries. Security teams may block them. Updating them to patch CVEs often means changing CUDA/framework versions (see pain point 8.3).

**Current workaround**: Accept the risk. Rebuild from scratch. Use alternative base images.

**Classification**: Setup-time / Security

---

## 9. Cross-Cutting Issues

### 9.1 CUDA non-determinism

**Problem**: CUDA operations are non-deterministic by default. The same code, same data, same GPU produces different results on different runs. This makes debugging and reproducibility difficult. Even `torch.use_deterministic_algorithms(True)` doesn't cover all operations.

**Example**:
```python
torch.use_deterministic_algorithms(True)
# RuntimeError: adaptive_avg_pool3d_backward_cuda does not have a deterministic implementation
```

**Current workaround**: Accept non-determinism for most workloads. Set seeds everywhere. Avoid certain operations. Accept slightly different results between runs.

**Classification**: Runtime

---

### 9.2 OOM errors and memory fragmentation

**Problem**: CUDA out-of-memory errors are the most common runtime failure in deep learning. They can be caused by the model, the batch size, memory fragmentation, or other processes using the GPU. Error messages don't indicate which allocation triggered the OOM.

**Example**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 39.39 GiB of which 1.45 GiB is free.
```

**Current workaround**: Reduce batch size. Use gradient checkpointing. Use mixed precision. `torch.cuda.empty_cache()`. Monitor with `nvidia-smi`. Trial and error.

**Classification**: Runtime

---

### 9.3 "Works on my machine" cascade

**Problem**: The combination of driver version, toolkit version, cuDNN version, framework version, Python version, and OS creates an enormous configuration space. Two developers with seemingly identical setups get different results because one detail differs. This is the fundamental reproducibility crisis of GPU development.

**Developer quote**: "I have a CUDA environment that works. I don't know why it works. I'm afraid to change anything."

**Current workaround**: Docker. But Docker introduces its own GPU pain points (see category 5). Or meticulous documentation that's always outdated.

**Classification**: Setup-time / Runtime

---

### 9.4 libstdc++ version mismatches

**Problem**: PyTorch and other frameworks are compiled against specific versions of libstdc++. The system-provided version may be older (especially on cloud instances with older OS images). This causes import errors or runtime crashes with unhelpful symbol error messages.

**Example**:
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

**Current workaround**: Upgrade system GCC. Install `conda-forge::libstdcxx-ng`. Manually provide the library via `LD_LIBRARY_PATH`.

**Classification**: Setup-time / Runtime

---

### 9.5 Multiple CUDA installs fighting for priority

**Problem**: It's common for a system to have CUDA installed in `/usr/local/cuda`, `/usr/local/cuda-12.1`, conda's CUDA, Docker's CUDA, and Snap's CUDA — all providing different versions of the same libraries. Which one loads at runtime depends on `LD_LIBRARY_PATH`, `PATH`, symlink configuration, and luck.

**Current workaround**: Aggressive `LD_LIBRARY_PATH` management. `ldconfig`. Remove competing installations. Isolation via containers.

**Classification**: Setup-time / Runtime

---

### 9.6 Python version constraints

**Problem**: CUDA framework builds don't support every Python version. PyTorch may not have wheels for the latest Python release for months after it ships. Using the wrong Python version means no GPU-enabled wheels are available, falling back to CPU (silently) or failing to install entirely.

**Current workaround**: Use the Python version the framework supports. Track compatibility matrices.

**Classification**: Setup-time

---

### 9.7 Debugging CUDA errors from Python

**Problem**: CUDA errors are often reported asynchronously. The Python stack trace points to a different line than where the error occurred. `CUDA_LAUNCH_BLOCKING=1` helps but changes timing behavior, potentially hiding the bug.

**Current workaround**: Set `CUDA_LAUNCH_BLOCKING=1`. Use `compute-sanitizer`. Accept that the stack trace lies.

**Classification**: Runtime

---

### 9.8 CUDA context initialization overhead

**Problem**: The first CUDA operation in a process initializes the CUDA context, which takes 0.5-2 seconds. This surprises developers benchmarking GPU code and makes short-lived scripts appear slower than CPU.

**Current workaround**: Warm-up iterations in benchmarks. Accept the initialization cost. Use CUDA MPS for shared contexts.

**Classification**: Runtime

---

## Summary Statistics

| Category | Count | Primarily Setup-time | Primarily Runtime | Both |
|----------|-------|---------------------|-------------------|------|
| 1. CUDA Setup & Installation | 6 | 6 | 0 | 0 |
| 2. Compatibility Matrix | 5 | 3 | 0 | 2 |
| 3. Cloud GPU Instances | 5 | 4 | 0 | 1 |
| 4. Multi-GPU & Multi-Node | 6 | 1 | 4 | 1 |
| 5. Docker/Container GPU | 5 | 5 | 0 | 0 |
| 6. Switching Between GPU Types | 4 | 2 | 1 | 1 |
| 7. conda/mamba CUDA | 6 | 5 | 0 | 1 |
| 8. NGC Containers | 6 | 5 | 0 | 1 |
| 9. Cross-Cutting | 8 | 2 | 4 | 2 |
| **Total** | **51** | **33** | **9** | **9** |

**Key insight**: The vast majority of GPU/CUDA pain is **setup-time** — getting the environment right before any real work begins. This is exactly where Flox's declarative, reproducible environments provide the most value.
