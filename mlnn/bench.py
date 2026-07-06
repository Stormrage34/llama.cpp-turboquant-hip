"""
mlnn/bench.py - KV Cache Benchmark & Integration Tests
========================================================
Merged from:
  - benchmark_kv_cache.py (llama-cli benchmark for KV cache compression)
  - test_turbo_kv_cache.py (server integration test with HTTP requests)
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict


# ============================================================================
# Constants
# ============================================================================

LLAMA_CLI = "/home/stormrage/llama.cpp/build-rocm/bin/llama-cli"

MODELS = {
    "gemma-26b-moe": {
        "path": "/home/stormrage/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "moe": True,
    },
}

CACHE_CONFIGS = [
    # (name, cache_k, cache_v, symmetry, needs_fa)
    ("sym_f16",        "f16",       "f16",       "symmetric",  False),
    ("sym_q8_0",       "q8_0",      "q8_0",      "symmetric",  True),
    ("sym_turbo3",     "turbo3_0",  "turbo3_0",  "symmetric",  True),
    ("asym_q8_f16",    "q8_0",      "f16",       "asymmetric", False),
    ("asym_q8_turbo3", "q8_0",      "turbo3_0",  "asymmetric", True),
]

N_PREDICT = 5000
TEMP = 0.0

# Gemma 4 uses </s> / think format
REASONING_PROMPT = (
    "</s>\n"
    "Problem: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Show each step clearly.\n"
    "</s>\n"
)

MATH_PROMPT = (
    "</s>\n"
    "Solve these and provide the final answers as an array:\n"
    "1. Find the roots of x^2 - 5x + 6 = 0\n"
    "2. Derivative of f(x) = 3x^3 - 2x^2 + x - 7 at x = 2\n"
    "3. Sum of first 20 terms: a_1 = 3, d = 4\n"
    "Format: [root1, root2, derivative, sum]\n"
    "</s>\n"
)

CODING_PROMPT = (
    "</s>\n"
    "Write a C function: int binary_search(int arr[], int size, int target);\n"
    "Trace n = [2,5,8,12,16,23,38,56,72,91], target = 23.\n"
    "Final answer: returns index [INDEX]\n"
    "</s>\n"
)

PROMPTS = [("reasoning", REASONING_PROMPT), ("math", MATH_PROMPT), ("coding", CODING_PROMPT)]

# Server integration test defaults
BASE_ARGS = [
    "-ngl", "99",
    "-c", "4096",
    "-b", "1024",
    "-ub", "1024",
    "-fa", "on",
    "--temp", "0.0",
    "--top-p", "1.0",
    "-t", "4",
    "-tb", "4",
    "-ncmoe", "18",
    "--no-mmap",
    "--numa", "isolate",
    "--jinja",
]

TEST_CASES = [
    # Symmetric (same type K/V) — should always work
    ("sym-f16-f16",     "f16",     "f16",     False),
    ("sym-q8_0-q8_0",   "q8_0",    "q8_0",    False),
    ("sym-turbo3",      "turbo3_0","turbo3_0",False),
    ("sym-turbo4",      "turbo4_0","turbo4_0",False),
    ("sym-planar3",     "planar3_0","planar3_0",False),
    ("sym-iso3",        "iso3_0",  "iso3_0",  False),
    ("sym-rq_mse",      "rq_mse",  "rq_mse",  False),
    ("sym-rq_prod",     "rq_prod", "rq_prod", False),

    # Asymmetric (different K/V types) — may fall back to CPU FA
    ("asym-q8_0-turbo3","q8_0",    "turbo3_0",False),
    ("asym-q8_0-turbo4","q8_0",    "turbo4_0",False),
    ("asym-q8_0-planar3","q8_0",   "planar3_0",False),
    ("asym-q8_0-iso3",  "q8_0",    "iso3_0",  False),
    ("asym-q8_0-rq_mse","q8_0",    "rq_mse",  False),
    ("asym-q8_0-rq_prod","q8_0",   "rq_prod", False),
    ("asym-turbo3-q8_0","turbo3_0","q8_0",    False),
]


# ============================================================================
# Validation helpers (from benchmark_kv_cache.py)
# ============================================================================

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


# ============================================================================
# llama-cli benchmark (from benchmark_kv_cache.py)
# ============================================================================

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


def run_benchmark_test(model_name, model_info, cache_name, cache_k, cache_v, symmetry, needs_fa, prompt_name, prompt_text):
    """Run a single llama-cli benchmark test and return results dict."""
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


def benchmark_kv_cache(model: str = None, cache_configs: list = None, prompts: list = None, ncpus: int = 1):
    """Run llama-cli benchmarks for KV cache compression.

    Args:
        model: Optional specific model name to test (default: all MODELS).
        cache_configs: Override CACHE_CONFIGS if provided.
        prompts: Override PROMPTS if provided.
        ncpus: Number of parallel subprocess workers (default: os.cpu_count() or 1).

    Returns:
        List of result dicts.
    """
    if cache_configs is None:
        cache_configs = CACHE_CONFIGS
    if prompts is None:
        prompts = PROMPTS

    models_to_run = {k: v for k, v in MODELS.items() if model is None or k == model}

    total = len(models_to_run) * len(cache_configs) * len(prompts)
    print(f"\nRunning {total} benchmark tests with {ncpus} parallel workers...\n")

    # Build flat task list: (name_key, args_tuple) for each test
    tasks = []
    for model_name, model_info in models_to_run.items():
        for cache_name, cache_k, cache_v, symmetry, needs_fa in cache_configs:
            for prompt_name, prompt_text in prompts:
                tasks.append((f"{model_name}:{cache_name}:{prompt_name}",
                              (model_name, model_info, cache_name, cache_k, cache_v,
                               symmetry, needs_fa, prompt_name, prompt_text)))

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpus) as pool:
        fut_to_key = {pool.submit(run_benchmark_test, *args): key for key, args in tasks}
        for fut in concurrent.futures.as_completed(fut_to_key):
            key = fut_to_key[fut]
            try:
                r = fut.result()
                results[key] = r
            except Exception as e:
                print(f"[ERROR] {key} failed: {e}")

    # Return in original order for deterministic reporting
    task_keys = [k for k, _ in tasks]
    ordered = [results[k] for k in task_keys if k in results]
    return ordered


def print_benchmark_report(results, models=None):
    """Print the benchmark report (from original main)."""
    if models is None:
        models = MODELS

    print("\n" + "=" * 120)
    print("BENCHMARK REPORT: KV Cache Compression")
    print("=" * 120)

    for model_name in models:
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
    for model_name in models:
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
    for model_name in models:
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
    output_path = "/home/stormrage/llama.cpp/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {output_path}")


# ============================================================================
# Server integration test (from test_turbo_kv_cache.py)
# ============================================================================

def check_server_ready(url, timeout=30):
    """Poll server health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    return False


def send_completion(url, prompt="The capital of France is"):
    """Send a completion request and return response text."""
    data = json.dumps({
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result.get("content", "")


def run_integration_test(test_name, ctk, ctv, expect_crash, server_exe, port, model_path):
    """Run a single integration test case and return (passed, output)."""
    server_args = [server_exe, *BASE_ARGS,
                   "-m", model_path,
                   "-ctk", ctk,
                   "-ctv", ctv,
                   "--host", "127.0.0.1",
                   "--port", str(port),
                   ]

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  ctk={ctk}  ctv={ctv}")
    print(f"  Command: {' '.join(server_args)}")
    print(f"{'='*60}")

    proc = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    url = f"http://127.0.0.1:{port}"
    ready = check_server_ready(url, timeout=60)

    if not ready:
        # Check if process crashed
        ret = proc.poll()
        if ret is not None:
            stderr = proc.stderr.read()
            print(f"  FAIL: Server exited with code {ret}")
            print(f"  STDERR (last 20 lines):")
            for line in stderr.strip().splitlines()[-20:]:
                print(f"    {line}")
            proc.stdout.close()
            proc.stderr.close()
            if expect_crash:
                print(f"  => EXPECTED crash (test passed)")
                return True, stderr
            print(f"  => UNEXPECTED crash (test FAILED)")
            return False, stderr
        print(f"  FAIL: Server not ready within timeout")
        proc.terminate()
        proc.wait()
        return False, ""

    print(f"  Server ready on {url}")

    try:
        output = send_completion(url)
        print(f"  Output: {output[:80]}...")

        # Shut down server gracefully
        try:
            urllib.request.urlopen(f"{url}/shutdown", timeout=3)
        except Exception:
            pass

    except Exception as e:
        print(f"  FAIL: Request error: {e}")
        output = ""

    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    proc.stdout.close()
    proc.stderr.close()

    if output and len(output) > 10:
        print(f"  PASS: got valid output ({len(output)} chars)")
        return True, output
    else:
        print(f"  FAIL: empty or too short output ({len(output)} chars)")
        return False, output


def test_integration(model: str = None, cache_combos: list = None, port: int = 8888, build_dir: str = "build", test_names: list = None, ncpus: int = 1):
    """Run turbo KV cache integration tests against llama-server.

    Args:
        model: Path to GGUF model (required).
        cache_combos: Override TEST_CASES if provided.
        port: Server port.
        build_dir: Build directory for finding llama-server.
        test_names: Specific test names to run (default: all).
        ncpus: Number of parallel subprocess workers (default: os.cpu_count() or 1).

    Returns:
        Dict with pass/fail/skip counts.
    """
    if cache_combos is None:
        cache_combos = TEST_CASES

    server_exe = os.path.join(build_dir, "bin", "llama-server")
    if not os.path.exists(server_exe):
        print(f"ERROR: llama-server not found in {build_dir}/bin/")
        sys.exit(1)

    # Filter out skipped tests
    active_combos = [(name, ctk, ctv, expect_crash) for name, ctk, ctv, expect_crash in cache_combos
                     if not test_names or name in test_names]

    print(f"\nRunning {len(active_combos)} integration tests with {ncpus} parallel workers...\n")

    results = {"pass": 0, "fail": 0, "skip": len(cache_combos) - len(active_combos)}
    failures = []

    def _run_one(name, ctk, ctv, expect_crash):
        return (name, run_integration_test(name, ctk, ctv, expect_crash, server_exe, port, model))

    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpus) as pool:
        futs = {pool.submit(_run_one, name, ctk, ctv, expect_crash): (name, ctk, ctv)
                for name, ctk, ctv, expect_crash in active_combos}
        for fut in concurrent.futures.as_completed(futs):
            try:
                name, (ok, output) = fut.result()
                if ok:
                    results["pass"] += 1
                else:
                    results["fail"] += 1
                    failures.append((name, output))
            except Exception as e:
                print(f"[ERROR] integration test failed: {e}")
                results["fail"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped")
    if failures:
        print(f"\nFAILURES:")
        for name, out in failures:
            print(f"  {name}")
    print(f"{'='*60}")

    return results


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    """CLI: --mode bench | integration"""
    parser = argparse.ArgumentParser(description="KV Cache Benchmark & Integration Tests")
    parser.add_argument("--mode", choices=["bench", "integration"], required=True,
                        help="Run mode: 'bench' for llama-cli benchmarks, 'integration' for server tests")
    parser.add_argument("--model", default=None,
                        help="Model path (required for both modes). For bench mode, also used as model name filter.")
    parser.add_argument("--bench-model", default=None,
                        help="Specific model name to benchmark (gemma-12b-dense or gemma-26b-moe)")
    parser.add_argument("--port", type=int, default=8888,
                        help="Server port for integration tests")
    parser.add_argument("--build-dir", default="build-rocm",
                        help="Build directory for finding llama-server")
    parser.add_argument("--test", nargs="*",
                        help="Integration mode: specific test names to run")
    parser.add_argument("--ncpus", type=int, default=1,
                        help="Number of parallel subprocess workers (default: 1; GPU benchmarks share VRAM, use --ncpus 16 for CPU-only)")
    args = parser.parse_args()

    # Control thread counts for child processes to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = str(args.ncpus)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpus)

    if args.mode == "bench":
        results = benchmark_kv_cache(model=args.bench_model, ncpus=args.ncpus)
        print_benchmark_report(results)
    elif args.mode == "integration":
        if not args.model:
            print("ERROR: --model is required for integration mode")
            sys.exit(1)
        result_counts = test_integration(
            model=args.model,
            port=args.port,
            build_dir=args.build_dir,
            test_names=args.test,
            ncpus=args.ncpus,
        )
        sys.exit(1 if result_counts["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
