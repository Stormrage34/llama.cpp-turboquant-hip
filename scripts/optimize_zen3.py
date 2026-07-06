#!/usr/bin/env python3
"""
Zen 3 (Ryzen 7 5700X) Architecture Optimizer

Probes the CPU, tests speculative decoding parameters, KV cache types,
and thread configurations to find optimal settings for performance,
precision, and task completion on AMD Zen 3 hardware.

Usage:
    # Probe CPU + recommend optimal speculative params
    python3 scripts/optimize_zen3.py --probe

    # Full benchmark sweep (requires running server)
    python3 scripts/optimize_zen3.py --bench --server http://localhost:8080

    # Probe + benchmark in one run
    python3 scripts/optimize_zen3.py --probe --bench --server http://localhost:8080
"""

import argparse
import json
import math
import multiprocessing
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CPU Probe
# ============================================================================

@dataclass
class CpuInfo:
    """Detected CPU capabilities and cache topology."""
    model: str = "unknown"
    cores_physical: int = 0
    cores_logical: int = 0
    l1d_kb: int = 0
    l1i_kb: int = 0
    l2_kb: int = 0
    l3_kb: int = 0
    has_avx2: bool = False
    has_bmi2: bool = False
    has_fma3: bool = False
    has_avx512: bool = False
    # Ryzen CCX topology
    ccx_count: int = 0
    cores_per_ccx: int = 0


def probe_cpu() -> CpuInfo:
    """Probe CPU capabilities and cache topology from the system."""
    info = CpuInfo()

    # CPU model
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        for line in cpuinfo.split("\n"):
            if line.startswith("model name"):
                info.model = line.split(":")[1].strip()
                break

    # Core counts
    info.cores_physical = multiprocessing.cpu_count() // 2 if os.path.exists("/sys/devices/system/cpu/cpu0/topology/core_id") else multiprocessing.cpu_count()
    info.cores_logical = multiprocessing.cpu_count()

    # On Ryzen, try to detect logical vs physical
    try:
        phys = set()
        for i in range(info.cores_logical):
            with open(f"/sys/devices/system/cpu/cpu{i}/topology/core_id") as f:
                phys.add(int(f.read().strip()))
        info.cores_physical = len(phys)
    except (FileNotFoundError, PermissionError):
        info.cores_physical = info.cores_logical // 2 if info.cores_logical <= 16 else info.cores_logical

    # Cache topology from sysfs
    cache_types = {"Data": "l1d", "Instruction": "l1i", "Unified": "l2"}
    for cache_type, attr in cache_types.items():
        try:
            with open(f"/sys/devices/system/cpu/cpu0/cache/index0/type") as f:
                pass  # Just checking it exists
        except FileNotFoundError:
            pass

    for i in range(10):  # Check up to 10 cache indices
        try:
            base = f"/sys/devices/system/cpu/cpu0/cache/index{i}"
            with open(f"{base}/type") as f:
                ctype = f.read().strip()
            with open(f"{base}/size") as f:
                size_str = f.read().strip()
            # Parse size (e.g., "32K", "512K", "32768K")
            if size_str.endswith("K"):
                size_kb = int(size_str[:-1])
            elif size_str.endswith("M"):
                size_kb = int(size_str[:-1]) * 1024
            else:
                size_kb = int(size_str) // 1024

            if ctype == "Data":
                info.l1d_kb = size_kb
            elif ctype == "Instruction":
                info.l1i_kb = size_kb
            elif ctype == "Unified":
                if info.l2_kb == 0:
                    info.l2_kb = size_kb
                else:
                    info.l3_kb = size_kb
        except (FileNotFoundError, PermissionError, ValueError):
            continue

    # ISA feature detection using Python's struct/ctypes
    # Fallback to /proc/cpuinfo flags
    if os.path.exists("/proc/cpuinfo"):
        flags = ""
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                flags = line.lower()
                break
            if line.startswith("Features"):
                flags = line.lower()
                break
        info.has_avx2 = "avx2" in flags
        info.has_bmi2 = "bmi2" in flags
        info.has_fma3 = "fma3" in flags or "fma" in flags
        info.has_avx512 = "avx512f" in flags

    # CCX detection for Ryzen
    if "ryzen" in info.model.lower():
        # L3 slice size on Zen 3 is 16 MB per CCX. If we detected L3 < 32 MB, it's a partial probe
        # 5700X has single CCX with 8 cores and 32 MB L3
        # Other Zen 3 chips have dual CCX (8+8 cores, 16+16 MB L3)
        if info.l3_kb >= 32768:
            info.ccx_count = 1
            info.cores_per_ccx = info.cores_physical
        elif info.l3_kb >= 16384:
            info.ccx_count = info.cores_physical // (info.cores_physical // 2)  # heuristic
            info.cores_per_ccx = info.cores_physical // info.ccx_count
        else:
            info.ccx_count = 1
            info.cores_per_ccx = info.cores_physical
    else:
        info.ccx_count = 1
        info.cores_per_ccx = info.cores_physical

    return info


# ============================================================================
# Recommendations
# ============================================================================

@dataclass
class Zen3Recommendations:
    """Optimal parameters for the detected CPU."""

    # Threading
    n_threads: int = 0
    n_batch_threads: int = 0
    cpu_mask: str = ""  # e.g., "0-7"
    cpu_strict: int = 0
    numa: str = "isolate"

    # Batch sizes
    batch_size: int = 512
    ubatch_size: int = 512

    # KV Cache
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: str = "on"

    # Speculative decoding (ngram-mod)
    spec_n_min: int = 4
    spec_n_max: int = 16
    ngram_mod_n_min: int = 16
    ngram_mod_n_max: int = 32
    ngram_mod_n_match: int = 24

    # Speculative decoding (ngram-map-k4v)
    map_k4v_size_n: int = 12
    map_k4v_size_m: int = 4
    map_k4v_n_min: int = 4
    map_k4v_n_max: int = 12

    # Model offloading
    n_gpu_layers: int = 99  # max offload
    no_kv_offload: bool = True  # Keep KV on CPU for L3 access

    # Sampling (for precision)
    temp: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.00
    repeat_penalty: float = 1.08
    repeat_last_n: int = 4096

    # MoE
    n_cpu_moe: int = 0
    cpu_range: str = ""

    # Context
    ctx_size: int = 16384
    ctxcp: int = 64


def recommend_settings(cpu: CpuInfo) -> Zen3Recommendations:
    """Compute optimal settings for the detected Zen 3 CPU."""
    rec = Zen3Recommendations()

    # Thread configuration
    rec.n_threads = cpu.cores_physical  # Use physical cores, not SMT
    rec.n_batch_threads = min(12, cpu.cores_logical)  # SMT threads for batch work
    rec.cpu_mask = f"0-{cpu.cores_physical - 1}"
    rec.cpu_strict = 1  # Pin threads to cores

    # NUMA: single CCX = isolate (no cross-CCX penalty)
    # Dual CCX: prefer "interleave" for balanced L3 usage
    if cpu.ccx_count <= 1:
        rec.numa = "isolate"
    else:
        rec.numa = "interleave"

    # Batch size: larger fits in L3 better with quantized KV
    # Q8_0 KV cache: each K/V pair = 8 bytes per layer per head
    # For 35B model with ~60 layers, 8 kv heads at Q8_0:
    #   ~60 * 8 * 8 = ~3.8 KB per token
    # 512 tokens = ~2 MB — fits in L3
    rec.batch_size = 4096   # Large batch for efficient prefill
    rec.ubatch_size = 4096  # Match batch for HQQ models

    # KV Cache: Q8_0 provides good precision at half the memory of F16
    # For 166K context with 35B model at Q8_0:
    #   K: 166K * 60 layers * 8 kv_heads * 1 byte = ~80 MB (DDR4)
    #   V: same. Total: ~160 MB — fits in system RAM easily
    rec.cache_type_k = "q8_0"
    rec.cache_type_v = "q8_0"
    rec.flash_attn = "on"  # FA helps even on CPU for memory-bound attention

    # Speculative decoding tuning
    # Baseline: n_min=48, n_max=64 for ngram-mod
    # With EMA adaptation, we can use shorter minima and let the EMA
    # extend dynamically when acceptance is strong.
    rec.spec_n_min = 4
    rec.spec_n_max = 16

    # ngram-mod: n_match=24 (standard), n_min=16, n_max=32
    # The 16 MB hash table fits in L3. With freq guard (now reverted),
    # we rely on blind overwrite which is proven stable.
    rec.ngram_mod_n_min = 16
    rec.ngram_mod_n_max = 32
    rec.ngram_mod_n_match = 24

    # ngram-map-k4v: 12-gram key, 4-gram value
    # 1 MB key_map + 4 value maps = ~5 MB total — all fits in L3
    rec.map_k4v_size_n = 12
    rec.map_k4v_size_m = 4
    rec.map_k4v_n_min = 4
    rec.map_k4v_n_max = 12

    # MoE tuning
    # 5700X: 8 cores, best with range covering all cores
    # Reduce expert count by a few to leave headroom for speculative decoding
    rec.n_cpu_moe = max(0, cpu.cores_physical * 2 + 2)  # e.g., 18 for 8 cores
    rec.cpu_range = rec.cpu_mask

    return rec


# ============================================================================
# Benchmark (work in progress — requires running server)
# ============================================================================

def run_llama_bench(
    server_url: str,
    prompt: str,
    params: dict,
    temperature: float = 0.6
) -> Optional[dict]:
    """Run a single inference request against running llama-server."""
    import requests
    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 256,
                **params,
            },
            timeout=300,
        )
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            return {
                "tokens_prompt": usage.get("prompt_tokens", 0),
                "tokens_generated": usage.get("completion_tokens", 0),
                "total_time_s": data.get("metrics", {}).get("time_to_first_token", 0) / 1000,
                "tokens_per_sec": usage.get("completion_tokens", 0) / max(
                    data.get("metrics", {}).get("time_per_output_token", 1) / 1000, 0.001
                ),
                "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            }
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def benchmark_variants(
    server_url: str,
    cpu: CpuInfo,
    spec_params: List[Dict[str, int]],
    cache_types: List[str],
    prompts: List[str],
) -> List[dict]:
    """Benchmark multiple parameter combinations."""
    results = []
    for spec in spec_params:
        for ctype in cache_types:
            params = {
                "speculative.ngram_mod.n_match": spec.get("n_match", 24),
                "speculative.ngram_mod.n_min": spec.get("mod_n_min", 16),
                "speculative.ngram_mod.n_max": spec.get("mod_n_max", 32),
                "speculative.ngram_map.n": spec.get("map_n", 12),
                "speculative.ngram_map.m": spec.get("map_m", 4),
                "speculative.cache_type_k": ctype,
                "speculative.cache_type_v": ctype,
            }
            variant_name = f"nmod={spec.get('mod_n_min',16)}-{spec.get('mod_n_max',32)}_cache={ctype}"
            print(f"  Testing: {variant_name}")

            for prompt in prompts:
                result = run_llama_bench(server_url, prompt, params)
                if result:
                    result["variant"] = variant_name
                    result["spec_params"] = spec
                    result["cache_type"] = ctype
                    results.append(result)
                    print(f"    tokens/s: {result.get('tokens_per_sec', 0):.1f}")
                time.sleep(1)  # Cooldown between requests

    return results


# ============================================================================
# Main
# ============================================================================

def print_probe_results(cpu: CpuInfo, rec: Zen3Recommendations):
    """Print probe results and recommendations."""
    print("=" * 60)
    print("AMD Ryzen / Zen 3 Architecture Probe")
    print("=" * 60)
    print(f"  CPU Model:       {cpu.model}")
    print(f"  Physical Cores:  {cpu.cores_physical}")
    print(f"  Logical Cores:   {cpu.cores_logical}")
    print(f"  L1d Cache:       {cpu.l1d_kb} KB per core")
    print(f"  L1i Cache:       {cpu.l1i_kb} KB per core")
    print(f"  L2 Cache:        {cpu.l2_kb} KB per core")
    print(f"  L3 Cache:        {cpu.l3_kb} KB (shared)")
    print(f"  CCX Count:       {cpu.ccx_count}")
    if cpu.l3_kb > 0:
        print(f"  L3 / core:       {cpu.l3_kb // cpu.cores_physical} KB")
    print(f"  AVX2:            {'YES' if cpu.has_avx2 else 'NO'}")
    print(f"  BMI2:            {'YES' if cpu.has_bmi2 else 'NO'}")
    print(f"  FMA3:            {'YES' if cpu.has_fma3 else 'NO'}")

    print()
    print("=" * 60)
    print("Recommended Settings for Zen 3")
    print("=" * 60)

    print()
    print("--- Threading ---")
    print(f"  --threads / -t:       {rec.n_threads}")
    print(f"  --batch-threads / -tb:{rec.n_batch_threads}")
    print(f"  --cpu-mask:           {rec.cpu_mask}")
    print(f"  --cpu-strict:         {rec.cpu_strict}")
    print(f"  --numa:               {rec.numa}")

    print()
    print("--- Batch / Context ---")
    print(f"  --batch-size / -b:    {rec.batch_size}")
    print(f"  --ubatch-size / -ub:  {rec.ubatch_size}")
    print(f"  --ctx-size / -c:      {rec.ctx_size}")

    print()
    print("--- KV Cache (Memory vs Precision) ---")
    print(f"  Recommended:          -ctk {rec.cache_type_k} -ctv {rec.cache_type_v}")
    print(f"  Compact:              -ctk q4_0 -ctv q4_0   (needs --flash-attn)")
    print(f"  Balanced:             -ctk q8_0 -ctv q8_0   (default recommendation)")
    print(f"  Precision:            -ctk f16  -ctv f16    (highest quality)")
    print(f"  FA Required:          --flash-attn on       (for quantized V)")

    print()
    print("--- Speculative Decoding (ngram-mod) ---")
    print(f"  --spec-draft-n-min:   {rec.spec_n_min}")
    print(f"  --spec-draft-n-max:   {rec.spec_n_max}")
    print(f"  --spec-ngram-mod-n-min: {rec.ngram_mod_n_min}")
    print(f"  --spec-ngram-mod-n-max: {rec.ngram_mod_n_max}")

    print()
    print("--- Speculative Decoding (ngram-map-k4v) ---")
    print(f"  --spec-ngram-map-k4v-size-n: {rec.map_k4v_size_n}")
    print(f"  --spec-ngram-map-k4v-size-m: {rec.map_k4v_size_m}")
    print(f"  --spec-ngram-map-k4v-n-min:  {rec.map_k4v_n_min}")
    print(f"  --spec-ngram-map-k4v-n-max:  {rec.map_k4v_n_max}")

    print()
    print("--- Model Offloading ---")
    print(f"  --n-gpu-layers / -ngl:{rec.n_gpu_layers}")
    print(f"  --no-kv-offload:      {'recommended' if rec.no_kv_offload else 'optional'}")
    print(f"  (Keep KV on CPU for direct L3 cache access)")

    print()
    print("--- Sampling (Precision) ---")
    print(f"  --temp:               {rec.temp}")
    print(f"  --top-p:              {rec.top_p}")
    print(f"  --top-k:              {rec.top_k}")
    print(f"  --min-p:              {rec.min_p}")
    print(f"  --repeat-penalty:     {rec.repeat_penalty}")
    print(f"  --repeat-last-n:      {rec.repeat_last_n}")

    print()
    print("--- MoE ---")
    print(f"  --n-cpu-moe:          {rec.n_cpu_moe}")
    print(f"  --cpu-range:          {rec.cpu_range}")

    print()
    print("--- Full Command (example) ---")
    print(f"  -t {rec.n_threads} -tb {rec.n_batch_threads} "
          f"{' '.join([''] + ['--cpu-mask', rec.cpu_mask, '--cpu-strict', str(rec.cpu_strict)])}"
          f" ")
    print()


def print_cache_analysis(cpu: CpuInfo):
    """Print cache utilization analysis for the workload."""
    table_size_mb = 16  # ngram-mod hash table
    table_entries = 4 * 1024 * 1024
    entry_size_bytes = 4
    map_k_size_mb = 1    # ngram-map key_map
    map_v_size_mb = 4    # ngram-map value maps (k4v)

    print()
    print("--- Cache Utilization Analysis ---")
    print(f"  ngram-mod table:   {table_size_mb} MB ({table_size_mb * 1024 / cpu.l3_kb * 100:.0f}% of L3)")
    print(f"  ngram-map tables:  {map_k_size_mb + map_v_size_mb} MB ({(map_k_size_mb + map_v_size_mb) * 1024 / cpu.l3_kb * 100:.0f}% of L3)")
    print(f"  Total speculative: {table_size_mb + map_k_size_mb + map_v_size_mb} MB ({(table_size_mb + map_k_size_mb + map_v_size_mb) * 1024 / cpu.l3_kb * 100:.0f}% of L3)")
    print(f"  Headroom:          {cpu.l3_kb / 1024 - table_size_mb - map_k_size_mb - map_v_size_mb:.0f} MB")
    if cpu.l3_kb / 1024 - table_size_mb - map_k_size_mb - map_v_size_mb < 0:
        print(f"  WARNING: Speculative tables exceed L3 capacity! "
              f"Consider reducing ngram-mod table size or disabling one speculator.")
    else:
        print(f"  OK: All speculative tables fit in L3 with headroom.")

    # Per-core working set
    l2_entries = cpu.l2_kb * 1024 // entry_size_bytes
    l1d_entries = cpu.l1d_kb * 1024 // entry_size_bytes
    print(f"  Per-core L2 capacity: {l2_entries:,} entries ({cpu.l2_kb} KB)")
    print(f"  Per-core L1d capacity:{l1d_entries:,} entries ({cpu.l1d_kb} KB)")
    print(f"  Draft working set:    ~16 cache lines ({16 * 64} bytes) — fits in L1d")


def main():
    parser = argparse.ArgumentParser(description="Zen 3 (R7 5700X) Optimizer")
    parser.add_argument("--probe", action="store_true", help="Probe CPU + recommend settings")
    parser.add_argument("--bench", action="store_true", help="Benchmark variants (needs server)")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL for bench")
    parser.add_argument("--spec-variants", nargs="+", type=int,
                        default=[4, 8, 12, 16, 24, 32],
                        help="Speculative n_min values to test")
    parser.add_argument("--cache-types", nargs="+",
                        default=["f16", "q8_0", "q4_0", "turbo3"],
                        help="KV cache types to test")
    args = parser.parse_args()

    # Probe CPU
    cpu = probe_cpu()
    rec = recommend_settings(cpu)

    if args.probe:
        print_probe_results(cpu, rec)
        print_cache_analysis(cpu)

    if args.bench:
        print()
        print("=" * 60)
        print("Benchmark Mode (requires running server)")
        print("=" * 60)
        print(f"  Server: {args.server}")
        print(f"  Testing spec n_min:    {args.spec_variants}")
        print(f"  Testing cache types:   {args.cache_types}")

        prompts = [
            "Explain the cache hierarchy of a modern CPU and how it affects inference performance.",
            "Write a Python function that computes Fibonacci numbers using dynamic programming.",
            "What are the key differences between speculative decoding and beam search?",
        ]

        spec_params = []
        for n_min in args.spec_variants:
            spec_params.append({
                "mod_n_min": n_min,
                "mod_n_max": min(n_min * 2, 64),
                "n_match": 24,
                "map_n": 12,
                "map_m": 4,
            })

        results = benchmark_variants(
            args.server, cpu, spec_params, args.cache_types, prompts
        )

        if results:
            print()
            print("--- Results Summary ---")
            print(f"{'Variant':<40} {'Tokens/s':<10} {'Generated':<10}")
            print("-" * 60)
            for r in results:
                print(f"{r['variant']:<40} {r.get('tokens_per_sec', 0):<10.1f} "
                      f"{r.get('tokens_generated', 0):<10}")
        else:
            print("  No benchmark results (server may not be running)")

    if not args.probe and not args.bench:
        parser.print_help()


if __name__ == "__main__":
    main()
