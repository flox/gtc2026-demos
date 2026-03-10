#!/usr/bin/env python3
"""GPU environment diagnostic tool for the Brev Multi-GPU Workflow demo.

Four modes:
  python test_gpu.py           # Full environment report
  python test_gpu.py --quick   # One-line summary
  python test_gpu.py --bench   # Matrix-multiply benchmark (CPU vs GPU)
  python test_gpu.py --infer   # LLM inference with Qwen2.5-0.5B-Instruct
"""

import argparse
import sys
import time

import torch

# ---------------------------------------------------------------------------
# GPU architecture lookup (compute capability major version -> arch name)
# ---------------------------------------------------------------------------
ARCH_NAMES = {
    3: "Kepler",
    5: "Maxwell",
    6: "Pascal",
    7: "Volta/Turing",
    8: "Ampere",
    9: "Hopper",
    10: "Blackwell",
}

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def gpu_arch_name(major: int) -> str:
    return ARCH_NAMES.get(major, f"Unknown (SM {major}x)")


# ---------------------------------------------------------------------------
# Mode: full report (default)
# ---------------------------------------------------------------------------
def full_report() -> None:
    print("=" * 60)
    print("  Brev Multi-GPU Workflow — Environment Report")
    print("=" * 60)
    print()

    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA built  : {torch.version.cuda or 'N/A'}")
    print(f"cuDNN       : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"GPUs found  : {count}")
        print()
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            arch = gpu_arch_name(props.major)
            mem_gb = props.total_memory / (1024 ** 3)
            print(f"  [{i}] {props.name}")
            print(f"      Compute capability : {props.major}.{props.minor}")
            print(f"      Architecture       : {arch}")
            print(f"      Memory             : {mem_gb:.1f} GB")
            print(f"      Multi-processors   : {props.multi_processor_count}")
            print()
    else:
        print("GPUs found  : 0  (CPU-only mode)")
        print()
        print("  No NVIDIA GPU detected.  PyTorch is using CPU-only wheels.")
        print("  On a Brev GPU instance this would show GPU details.")
        print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Mode: --quick
# ---------------------------------------------------------------------------
def quick_summary() -> None:
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
        gpu_str = ", ".join(names)
        print(f"CUDA {torch.version.cuda} | PyTorch {torch.__version__} | {gpu_str}")
    else:
        print(f"CPU-only | PyTorch {torch.__version__}")


# ---------------------------------------------------------------------------
# Mode: --bench
# ---------------------------------------------------------------------------
def benchmark(size: int = 4096, iterations: int = 50) -> None:
    print(f"Matrix multiply benchmark  ({size}x{size}, {iterations} iterations)")
    print("-" * 60)

    # --- CPU ---
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # Warm-up
    _ = torch.mm(a_cpu, b_cpu)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.mm(a_cpu, b_cpu)
    cpu_time = time.perf_counter() - start
    cpu_avg = cpu_time / iterations * 1000  # ms

    print(f"  CPU  : {cpu_avg:8.2f} ms / iteration  (total {cpu_time:.2f}s)")

    # --- GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # Warm-up (includes kernel compilation)
        for _ in range(5):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start
        gpu_avg = gpu_time / iterations * 1000  # ms

        speedup = cpu_avg / gpu_avg
        print(f"  GPU  : {gpu_avg:8.2f} ms / iteration  (total {gpu_time:.2f}s)")
        print(f"  Speedup : {speedup:.1f}x")
    else:
        print("  GPU  : skipped (no CUDA device)")

    print("-" * 60)


# ---------------------------------------------------------------------------
# Mode: --infer
# ---------------------------------------------------------------------------
def run_inference(prompt: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Auto-detect device and precision
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        precision = "float16"
    else:
        device = "cpu"
        dtype = torch.float32
        precision = "float32"

    print("=" * 60)
    print("  Brev Multi-GPU Workflow — LLM Inference")
    print("=" * 60)
    print()
    print(f"Model       : {MODEL_ID}")
    print(f"Device      : {device}")
    print(f"Precision   : {precision}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU         : {props.name}")
    print()

    # Load model and tokenizer
    print("Loading model ...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Loaded in {load_time:.1f}s")
    print()

    # Build chat prompt
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    print("-" * 60)
    print(f"  Prompt: {prompt}")
    print("-" * 60)
    gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    gen_time = time.perf_counter() - gen_start

    new_tokens = outputs[0][input_len:]
    n_tokens = len(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0

    print()
    print(f"  Response: {response}")
    print()
    print("-" * 60)
    print(f"  Tokens generated : {n_tokens}")
    print(f"  Generation time  : {gen_time:.2f}s")
    print(f"  Throughput       : {tok_per_sec:.1f} tok/s")
    print("-" * 60)
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GPU environment diagnostic tool")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="One-line summary")
    group.add_argument("--bench", action="store_true", help="Matrix multiply benchmark")
    group.add_argument("--infer", action="store_true", help="LLM inference with Qwen2.5-0.5B-Instruct")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size for benchmark (default: 4096)")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations (default: 50)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in two sentences why GPU acceleration matters for machine learning.",
        help="Custom prompt for --infer mode",
    )
    args = parser.parse_args()

    if args.quick:
        quick_summary()
    elif args.bench:
        benchmark(size=args.size, iterations=args.iterations)
    elif args.infer:
        run_inference(prompt=args.prompt)
    else:
        full_report()


if __name__ == "__main__":
    main()
