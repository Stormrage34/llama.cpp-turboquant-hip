#!/usr/bin/env python3
"""llama-cli benchmark: KV cache compression on Gemma 12B (dense) + 26B (MoE)."""

import json, os, re, subprocess, sys, time
from collections import defaultdict

LLAMA_CLI = "/home/stormrage/llama.cpp/build-rocm/bin/llama-cli"
MODELS = {
    "gemma-12b-dense": {
        "path": "/home/stormrage/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "moe": False,
    },
    "gemma-26b-moe": {
        "path": "/home/stormrage/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "moe": True,
    },
}
CACHE_CONFIGS = [
    # (name, cache_k, cache_v, symmetry, needs_fa)
    ("sym_f16",        "f16",       "f16",       "symmetric",  False),
    ("sym_turbo3",     "turbo3_0",  "turbo3_0",  "symmetric",  True),
]
N_PREDICT = 5000
TEMP = 0.0

# Gemma 4 uses <|im_start|> / <|im_end|> format
REASONING_PROMPT = (
    "<|im_start|>user\n"
    "Problem: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Show each step clearly.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n"
)

MATH_PROMPT = (
    "<|im_start|>user\n"
    "Solve these and provide the final answers as an array:\n"
    "1. Find the roots of x^2 - 5x + 6 = 0\n"
    "2. Derivative of f(x) = 3x^3 - 2x^2 + x - 7 at x = 2\n"
    "3. Sum of first 20 terms: a_1 = 3, d = 4\n"
    "Format: [root1, root2, derivative, sum]\n"
    "<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n"
)

CODING_PROMPT = (
    "<|im_start|>user\n"
    "Write a C function: int binary_search(int arr[], int size, int target);\n"
    "Trace n = [2,5,8,12,16,23,38,56,72,91], target = 23.\n"
    "Final answer: returns index [INDEX]\n"
    "<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n"
)

PROMPTS = [("reasoning", REASONING_PROMPT), ("math", MATH_PROMPT), ("coding", CODING_PROMPT)]

def validate_reasoning(text):
    ball = "0.05" in text or "5 cents" in text.lower() or "$0.05" in text
    return {"ball_cost_correct": bool(ball), "all_correct": bool(ball)}

def validate_math(text):
    m = re.search(r'\[([^\]]+)\]', text)
    parts = re.findall(r'-?\d+\.?\d*', m.group(1)) if m else []
    if len(parts) < 4:
        parts = re.findall(r'-?\d+\.?\d*', text)
    if len(parts) < 3:
        return {"all_correct": False, "detail": "need 3+ numbers"}
    nums = [float(p) for p in parts[:4]]
    roots_ok = (abs(nums[0]-2) < 0.5 and abs(nums[1]-3) < 0.5) or (abs(nums[0]-3) < 0.5 and abs(nums[1]-2) < 0.5)
    deriv_ok = abs(nums[2] - 29) < 1.0 if len(nums) > 2 else False
    sum_ok = abs(nums[3] - 820) < 1.0 if len(nums) > 3 else False
    return {"roots_correct": bool(roots_ok), "derivative_correct": bool(deriv_ok),
            "sum_correct": bool(sum_ok), "all_correct": bool(roots_ok and deriv_ok and sum_ok)}

def validate_coding(text):
    m = re.search(r'(?:index|returns?)\D*(\d+)', text, re.IGNORECASE)
    idx_ok = m and int(m.group(1)) == 5
    return {"index_correct": bool(idx_ok), "all_correct": bool(idx_ok)}

VALIDATORS = {"reasoning": validate_reasoning, "math": validate_math, "coding": validate_coding}

def try_ngl(model_path, cache_k, cache_v, target_ngl=99):
    """Find the max GPU layers that fits in VRAM for this config."""
    for ngl in [target_ngl, 60, 40, 20, 10]:
        cmd = [LLAMA_CLI, "--model", model_path, "--gpu-layers", str(ngl),
               "--n-predict", "5", "--temp", "0",
               "--cache-type-k", cache_k, "--cache-type-v", cache_v,
               "--threads", "8", "--batch-size", "2048", "--no-mmap",
               "-fa", "on",
               "--prompt", "test"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = r.stdout + r.stderr
            if "out of memory" not in out.lower() and "failed to allocate" not in out.lower():
                return ngl
        except (subprocess.TimeoutExpired, Exception):
            continue
    return 10  # last resort

def run_test(model_name, model_info, cache_name, cache_k, cache_v, symmetry, needs_fa, prompt_name, prompt_text):
    print(f"  [{model_name}] [{cache_name}] [{prompt_name}] ...", end=" ", flush=True)
    try:
        ngl = try_ngl(model_info["path"], cache_k, cache_v)
        cmd = [LLAMA_CLI, "--model", model_info["path"], "--gpu-layers", str(ngl),
               "--n-predict", str(N_PREDICT), "--temp", str(TEMP),
               "--cache-type-k", cache_k, "--cache-type-v", cache_v,
               "--threads", "8", "--batch-size", "2048", "--no-mmap",
               "--prompt", prompt_text]
        if needs_fa:
            cmd.append("-fa"); cmd.append("on")
        if model_info["moe"]:
            cmd += ["--n-cpu-moe-range", "0-21"]

        start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start
        output = r.stdout + r.stderr

        # Parse tokens/sec
        tps = 0.0
        for line in output.split("\n"):
            m = re.search(r'([\d.]+)\s*tokens/s', line, re.IGNORECASE)
            if m: tps = float(m.group(1))
            m = re.search(r'predicted_per_token_ms[^0-9]*([\d.]+)', line)
            if m: tps = 1000.0 / float(m.group(1)) if float(m.group(1)) > 0 else 0

        # VRAM
        vram = 0.0
        for m in re.finditer(r'ROCm0\s.*?buffer\s.*?=\s+([\d.]+)\s+MiB', output):
            vram += float(m.group(1))

        # Validation
        valid = VALIDATORS[prompt_name](r.stdout)
        status = "PASS" if valid.get("all_correct") else "FAIL"
        print(f"{status} ({tps:.1f} t/s, ngl={ngl})")
        return {"model": model_name, "cache_type": cache_name, "symmetry": symmetry,
                "prompt_type": prompt_name, "tokens_per_second": round(tps, 2),
                "gpu_layers": ngl, "validation": valid, "error": None}
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"model": model_name, "cache_type": cache_name, "error": "timeout"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"model": model_name, "cache_type": cache_name, "error": str(e)}

def main():
    results = []
    total = len(MODELS) * len(CACHE_CONFIGS) * len(PROMPTS)
    i = 0
    print(f"\nRunning {total} benchmark tests...\n")
    for model_name, model_info in MODELS.items():
        for cache_name, cache_k, cache_v, symmetry, needs_fa in CACHE_CONFIGS:
            for prompt_name, prompt_text in PROMPTS:
                i += 1
                print(f"[{i}/{total}] ", end="")
                r = run_test(model_name, model_info, cache_name, cache_k, cache_v,
                           symmetry, needs_fa, prompt_name, prompt_text)
                results.append(r)

    # Report
    print("\n" + "=" * 120)
    print("BENCHMARK REPORT: KV Cache Compression")
    print("=" * 120)

    for model_name in MODELS:
        print(f"\n{'─' * 100}")
        print(f"Model: {model_name}")
        print(f"{'─' * 100}")
        mr = [r for r in results if r.get("model") == model_name and not r.get("error")]
        print(f"{'Cache':<22} {'Sym':<12} {'Prompt':<12} {'t/s':>8} {'NGL':>5} {'Valid':>6}")
        print("-" * 100)
        for r in sorted(mr, key=lambda x: (x["cache_type"], x["prompt_type"])):
            v = "PASS" if r.get("validation", {}).get("all_correct") else "FAIL"
            print(f"{r['cache_type']:<22} {r['symmetry']:<12} {r['prompt_type']:<12} "
                  f"{r['tokens_per_second']:>8.1f} {r['gpu_layers']:>5} {v:>6}")
        for r in results:
            if r.get("model") == model_name and r.get("error"):
                print(f"  ERROR {r['cache_type']}/{r.get('prompt_type','?')}: {r['error']}")

    # Correctness matrix
    print(f"\n{'─' * 80}")
    print("Correctness Matrix")
    print(f"{'─' * 80}")
    print(f"{'Model':<22} {'Cache':<22} {'R':>6} {'M':>6} {'C':>6} {'Score':>7}")
    print("-" * 80)
    for model_name in MODELS:
        for ct_name, _, _, _, _ in CACHE_CONFIGS:
            rows = [r for r in results if r.get("model") == model_name and r.get("cache_type") == ct_name and not r.get("error")]
            sc = {r["prompt_type"]: r.get("validation", {}).get("all_correct", False) for r in rows}
            ratio = sum(sc.values())
            print(f"{model_name:<22} {ct_name:<22} {str(sc.get('reasoning', '-')):>6} "
                  f"{str(sc.get('math', '-')):>6} {str(sc.get('coding', '-')):>6} {ratio:>5}/3")

    # Speed summary
    print(f"\n{'─' * 70}")
    print("Speed vs f16 Baseline (avg across prompts)")
    print(f"{'─' * 70}")
    for model_name in MODELS:
        mr = [r for r in results if r.get("model") == model_name and not r.get("error")]
        f16 = [r for r in mr if r["cache_type"] == "sym_f16"]
        base_tps = sum(r["tokens_per_second"] for r in f16) / max(len(f16), 1)
        print(f"\n{model_name} (baseline f16: {base_tps:.1f} t/s):")
        for ct_name, _, _, _, _ in CACHE_CONFIGS:
            if ct_name == "sym_f16": continue
            group = [r for r in mr if r["cache_type"] == ct_name]
            if group:
                avg = sum(r["tokens_per_second"] for r in group) / len(group)
                print(f"  {ct_name:<22}: {avg:.1f} t/s  ({avg/base_tps:.2f}x)")

    # Save
    with open("/home/stormrage/llama.cpp/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] benchmark_results.json")

if __name__ == "__main__":
    main()
