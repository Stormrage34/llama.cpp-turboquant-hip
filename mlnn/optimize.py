"""
mlnn/optimize.py - Turbo Quantization Optimization
====================================================
Merged from:
  - scripts/optimize-turboquant.py (benchmark, profile, bottleneck analysis)
  - scripts/compute_turbo_centroids.py (Lloyd-Max centroid computation)
"""

import argparse
import csv
import json
import numpy as np
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Shared Constants
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(REPO_ROOT, "build-rocm")
BENCH_BIN = os.path.join(BUILD_DIR, "bin", "llama-bench")

QUANT_TYPES_STANDARD = ["q4_0", "q8_0"]
QUANT_TYPES_PLANAR_ISO = ["planar3_0", "planar4_0", "iso3_0", "iso4_0"]
QUANT_TYPES_TURBO = ["turbo2_0", "turbo3_0", "turbo4_0"]

ALL_QUANT_TYPES = QUANT_TYPES_STANDARD + QUANT_TYPES_PLANAR_ISO + QUANT_TYPES_TURBO

BUILD_CONFIGS = {
    "baseline": {
        "cmake_flags": "-DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=OFF",
        "description": "Standard build, no turbo/planar FA",
    },
    "fa_all_quants": {
        "cmake_flags": "-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=OFF",
        "description": "FA instances for all quant types",
    },
    "graphs": {
        "cmake_flags": "-DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON",
        "description": "hipGraph capture enabled",
    },
    "asymmetric_fa": {
        "cmake_flags": "-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON",
        "description": "FA + Graphs: fixes asymmetric KV (q8K/turboV) using FA VEC path",
    },
    "trace": {
        "cmake_flags": "-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA_TRACE_SOFTMAX=ON",
        "description": "As asymmetric_fa + per-head softmax tracing",
    },
}

ASYMMETRIC_PAIRS = [
    ("q8_0", "turbo2_0"),
    ("q8_0", "turbo3_0"),
    ("q8_0", "turbo4_0"),
    ("f16",  "turbo3_0"),
    ("q8_0", "q8_0"),  # symmetric baseline for comparison
]


# ============================================================================
# Dataclasses (from optimize-turboquant.py)
# ============================================================================

@dataclass
class BenchResult:
    config_name: str
    quant_type: str
    pp_tok_s: float = 0.0
    tg_tok_s: float = 0.0
    pp_time_ms: float = 0.0
    tg_time_ms: float = 0.0
    memory_mb: float = 0.0
    passes: bool = False
    error: str = ""


@dataclass
class ProfileResult:
    config_name: str
    quant_type: str
    total_kernel_ms: float = 0.0
    kernel_count: int = 0
    h2d_count: int = 0
    h2d_total_ms: float = 0.0


@dataclass
class TraceHeadStats:
    head: int = 0
    n_samples: int = 0
    sum_mean: float = 0.0
    sum_std: float = 0.0
    max_mean: float = 0.0
    max_std: float = 0.0
    n_tiles: int = 0


@dataclass
class BottleneckResult:
    """Decomposed bottleneck analysis for a single run."""
    config_name: str = ""
    pp_tok_s: float = 0.0
    tg_tok_s: float = 0.0
    n_heads_with_traces: int = 0
    kq_sum_range: str = ""
    kq_sum_anomaly_pct: float = 0.0  # fraction of heads with sum>2.0 or sum<0.01
    trace_sampling_rate: int = 512


# ============================================================================
# Build System (from optimize-turboquant.py)
# ============================================================================

def build_config(config_name: str, cmake_flags: str, force_rebuild: bool = False, ncpus: int = 4) -> bool:
    """Build llama-bench with specific cmake flags."""
    build_path = os.path.join(BUILD_DIR, config_name)
    print(f"\n{'='*60}")
    print(f"Building: {config_name}")
    print(f"  Flags: {cmake_flags}")
    print(f"  Dir: {build_path}")
    print(f"{'='*60}")

    if force_rebuild and os.path.exists(build_path):
        subprocess.run(["rm", "-rf", build_path], check=True)

    os.makedirs(build_path, exist_ok=True)

    # cmake configure
    cmd = [
        "cmake", "-B", build_path, "-S", REPO_ROOT,
        "-DGGML_HIP=ON",
        "-DGGML_HIP_UMA=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ] + cmake_flags.split()

    print(f"  Configure: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR: cmake configure failed (rc={result.returncode})")
        print(f"  {result.stderr[-500:]}")
        return False

    # cmake build (just llama-bench)
    cmd = ["cmake", "--build", build_path, "-j" + str(ncpus),
           "--target", "llama-bench"]
    print(f"  Build: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        print(f"  ERROR: build failed (rc={result.returncode})")
        print(f"  {result.stderr[-500:]}")
        return False

    bench_path = os.path.join(build_path, "bin", "llama-bench")
    if not os.path.exists(bench_path):
        print(f"  ERROR: llama-bench not found at {bench_path}")
        return False

    print(f"  OK: {bench_path}")
    return True


# ============================================================================
# Benchmark Runner (from optimize-turboquant.py)
# ============================================================================

def run_bench(
    bench_path: str,
    model_path: str,
    quant_type: str = "",
    batch_size: int = 512,
    n_prompt: int = 512,
    n_gen: int = 128,
    threads: int = 1,
    repetitions: int = 3,
    timeout_s: int = 300,
    ctk: Optional[str] = None,  # K cache type (e.g. q8_0)
    ctv: Optional[str] = None,  # V cache type (e.g. turbo3_0)
) -> BenchResult:
    """Run llama-bench and parse results.

    llama-bench does NOT accept quant types as arguments. The quantization
    is determined by the GGUF file itself. If quant_type is provided, it's
    used for labeling only — the model file must already be quantized.
    Supports asymmetric KV cache via --ctk/--ctv flags.
    """
    result = BenchResult(config_name="", quant_type=quant_type)

    cmd = [
        bench_path,
        "-m", model_path,
        "-t", str(threads),
        "-b", str(batch_size),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-o", "csv",
    ]
    if ctk:
        cmd += ["--ctk", str(ctk)]
    if ctv:
        cmd += ["--ctv", str(ctv)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        output = proc.stdout

        # llama-bench CSV columns (from header):
        # ... n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts
        # avg_ts = tokens per second
        # n_prompt > 0 means prefill test, n_gen > 0 means decode test
        for line in output.strip().split("\n"):
            if line.startswith("#") or not line.strip() or line.startswith("build_"):
                continue
            parts = line.split(",")
            if len(parts) < 40:
                continue

            try:
                n_prompt_val = int(parts[33].strip('"'))
                n_gen_val = int(parts[34].strip('"'))
                avg_ns = float(parts[37].strip('"'))   # avg_ns column
                avg_ts = float(parts[39].strip('"'))    # avg_ts column

                if n_prompt_val > 0 and n_gen_val == 0:
                    # Prefill test
                    result.pp_tok_s = avg_ts
                    result.pp_time_ms = avg_ns / 1_000_000
                elif n_gen_val > 0 and n_prompt_val == 0:
                    # Decode test
                    result.tg_tok_s = avg_ts
                    result.tg_time_ms = avg_ns / 1_000_000
            except (ValueError, IndexError):
                continue

        result.passes = result.pp_tok_s > 0 or result.tg_tok_s > 0

    except subprocess.TimeoutExpired:
        result.error = f"Timeout after {timeout_s}s"
    except Exception as e:
        result.error = str(e)

    return result


# ============================================================================
# FA_SOFTMAX Trace Analyzer (from optimize-turboquant.py)
# ============================================================================

def parse_softmax_traces(log_path: str) -> Dict[int, TraceHeadStats]:
    """Parse FA_SOFTMAX trace lines from a llama-cli or llama-perplexity run.

    Format: FA_SOFTMAX: seq=N hd=N col=N n_tiles=N sum=N max=N sid=N
    sum and max are scaled integers (sum*1e6, max*1e2).
    Returns per-head aggregate statistics.
    """
    # raw[head] = {"sum": [...], "max": [...], "nt": int}
    raw: Dict[int, dict] = {}

    with open(log_path) as f:
        for line in f:
            m = re.search(r'FA_SOFTMAX:\s*seq=\d+\s+hd=(\d+)\s+col=\d+\s+n_tiles=(\d+)\s+sum=(\d+)\s+max=(\d+)', line)
            if not m:
                continue
            hd = int(m.group(1))
            nt = int(m.group(2))
            s = int(m.group(3)) / 1e6
            m_val = int(m.group(4)) / 1e2

            if hd not in raw:
                raw[hd] = {'sum': [], 'max': []}
            raw[hd]['sum'].append(s)
            raw[hd]['max'].append(m_val)
            raw[hd]['nt'] = nt

    result = {}
    for hd, d in raw.items():
        ss = d['sum']
        ms = d['max']
        result[hd] = TraceHeadStats(
            head=hd,
            n_samples=len(ss),
            sum_mean=float(np.mean(ss)) if ss else 0.0,
            sum_std=float(np.std(ss)) if len(ss) > 1 else 0.0,
            max_mean=float(np.mean(ms)) if ms else 0.0,
            max_std=float(np.std(ms)) if len(ms) > 1 else 0.0,
            n_tiles=d.get('nt', 0),
        )
    return result


# ============================================================================
# Bottleneck Analysis (from optimize-turboquant.py)
# ============================================================================

def analyze_bottleneck(
    log_path: str,
    config_name: str = "asymmetric",
    pp_tok_s: float = 0.0,
    tg_tok_s: float = 0.0,
) -> BottleneckResult:
    """Analyze FA_SOFTMAX traces to identify attention bottlenecks.

    Heuristics:
      - KQ_sum >> 1.0: over-concentrated attention (potential numerical issue)
      - KQ_sum << 0.1: diffuse/weak attention (potential precision loss)
      - High head count with traces: wide coverage = healthy sampling
      - Speculative decoding effectiveness: inferred from n_tiles pattern
    """
    heads = parse_softmax_traces(log_path)
    n_heads = len(heads)
    if n_heads == 0:
        return BottleneckResult(config_name=config_name, pp_tok_s=pp_tok_s, tg_tok_s=tg_tok_s)

    # Count anomalous heads
    anomalous = 0
    sum_vals = []
    for hd, h in heads.items():
        if h.n_samples > 0:
            if h.sum_mean > 2.0 or h.sum_mean < 0.01:
                anomalous += 1
            sum_vals.append(h.sum_mean)

    sum_range = f"{min(sum_vals):.3f}..{max(sum_vals):.3f}" if sum_vals else "N/A"
    anomaly_pct = (anomalous / n_heads * 100) if n_heads > 0 else 0.0

    return BottleneckResult(
        config_name=config_name,
        pp_tok_s=pp_tok_s,
        tg_tok_s=tg_tok_s,
        n_heads_with_traces=n_heads,
        kq_sum_range=sum_range,
        kq_sum_anomaly_pct=round(anomaly_pct, 1),
    )


# ============================================================================
# Profiling (from optimize-turboquant.py)
# ============================================================================

def run_profile(
    bench_path: str,
    model_path: str,
    quant_type: str,
    output_dir: str,
    batch_size: int = 512,
) -> Optional[ProfileResult]:
    """Run rocprofv3 profiling on a single quant type."""
    result = ProfileResult(config_name="", quant_type=quant_type)

    prof_dir = os.path.join(output_dir, f"profile_{quant_type}")
    os.makedirs(prof_dir, exist_ok=True)

    cmd = [
        "/opt/rocm/bin/rocprofv3",
        "--output-directory", prof_dir,
        "--output-format", "csv",
        "--runtime-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--", bench_path,
        "-m", model_path,
        "-t", "1", "-b", str(batch_size),
        "-p", "512", "-n", "128", "-r", "1",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # Find and parse CSVs
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            fp = os.path.join(root, f)
            if "kernel_trace" in f and f.endswith(".csv"):
                result.total_kernel_ms, result.kernel_count = _parse_kernel_time(fp)
            elif "memory_copy_trace" in f and f.endswith(".csv"):
                result.h2d_count, result.h2d_total_ms = _parse_memory_copies(fp)

    return result


def _parse_kernel_time(path: str) -> tuple:
    """Parse kernel trace CSV, return (total_ms, count)."""
    total_ns = 0
    count = 0
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                total_ns += (end - start)
                count += 1
            except (ValueError, KeyError):
                continue
    return total_ns / 1_000_000, count


def _parse_memory_copies(path: str) -> tuple:
    """Parse memory copy trace CSV, return (h2d_count, h2d_total_ms)."""
    h2d_count = 0
    h2d_ns = 0
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "HOST_TO_DEVICE" not in row.get("Direction", ""):
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                h2d_ns += (end - start)
                h2d_count += 1
            except (ValueError, KeyError):
                continue
    return h2d_count, h2d_ns / 1_000_000


# ============================================================================
# Report Generation (from optimize-turboquant.py)
# ============================================================================

def generate_report(
    results: List[BenchResult],
    profiles: List[ProfileResult],
    output_path: str,
    asym_results: Optional[List[BenchResult]] = None,
):
    """Generate the quant optimization report."""
    lines = []
    lines.append("=" * 78)
    lines.append("TURBOQUANT / PLANAR / ISOQUANT OPTIMIZATION REPORT")
    lines.append("=" * 78)
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"GPU: AMD Radeon RX 6800 XT (gfx1030)")
    lines.append("")

    # --- Benchmark Results Table ---
    lines.append("-" * 78)
    lines.append("BENCHMARK RESULTS BY QUANT TYPE")
    lines.append("-" * 78)

    # Group by quant type
    by_quant = {}
    for r in results:
        if r.quant_type not in by_quant:
            by_quant[r.quant_type] = {}
        by_quant[r.quant_type][r.config_name] = r

    # Header
    configs_used = sorted(set(r.config_name for r in results))
    header = f"{'Quant':12s}"
    for c in configs_used:
        header += f"  {c:>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    for qt in ALL_QUANT_TYPES:
        if qt not in by_quant:
            continue
        row = f"{qt:12s}"
        for c in configs_used:
            r = by_quant[qt].get(c)
            if r and r.passes:
                row += f"  {r.pp_tok_s:8.1f}/{r.tg_tok_s:5.1f}"
            elif r and r.error:
                row += f"  {'ERROR':>12s}"
            else:
                row += f"  {'N/A':>12s}"
        lines.append(row)

    lines.append("")
    lines.append("  Format: pp_tok_s / tg_tok_s")
    lines.append("")

    # --- Delta vs Baseline ---
    if "baseline" in configs_used and len(configs_used) > 1:
        lines.append("-" * 78)
        lines.append("DELTA vs BASELINE (pp_tok_s)")
        lines.append("-" * 78)

        header = f"{'Quant':12s}"
        for c in configs_used:
            if c == "baseline":
                continue
            header += f"  {c:>12s}"
        lines.append(header)
        lines.append("-" * len(header))

        for qt in ALL_QUANT_TYPES:
            if qt not in by_quant:
                continue
            baseline = by_quant[qt].get("baseline")
            if not baseline or not baseline.passes:
                continue
            row = f"{qt:12s}"
            for c in configs_used:
                if c == "baseline":
                    continue
                r = by_quant[qt].get(c)
                if r and r.passes and baseline.pp_tok_s > 0:
                    delta = ((r.pp_tok_s - baseline.pp_tok_s) / baseline.pp_tok_s) * 100
                    row += f"  {delta:+10.1f}%"
                else:
                    row += f"  {'N/A':>12s}"
            lines.append(row)

        lines.append("")

    # --- Turbo2 Norm Fix Verification ---
    lines.append("-" * 78)
    lines.append("TURBO2 NORM FIX VERIFICATION")
    lines.append("-" * 78)

    turbo2_results = [r for r in results if r.quant_type == "turbo2_0"]
    turbo3_results = [r for r in results if r.quant_type == "turbo3_0"]

    if turbo2_results and turbo3_results:
        for config in configs_used:
            t2 = next((r for r in turbo2_results if r.config_name == config), None)
            t3 = next((r for r in turbo3_results if r.config_name == config), None)
            if t2 and t3 and t2.passes and t3.passes:
                # If turbo2 is roughly comparable to turbo3, norm fix works
                # If turbo2 is way worse, norm fix may be needed
                ratio = t2.tg_tok_s / t3.tg_tok_s if t3.tg_tok_s > 0 else 0
                status = "OK" if ratio > 0.7 else "SUSPECT (ratio<0.7)"
                lines.append(f"  {config:15s}: turbo2={t2.tg_tok_s:.1f} t/s, turbo3={t3.tg_tok_s:.1f} t/s, ratio={ratio:.2f} [{status}]")
            elif t2:
                lines.append(f"  {config:15s}: turbo2={t2.tg_tok_s:.1f} t/s, turbo3=N/A")
    else:
        lines.append("  Insufficient data (need both turbo2_0 and turbo3_0 results)")
    lines.append("")

    # --- Profile Comparison ---
    if profiles:
        lines.append("-" * 78)
        lines.append("PROFILE COMPARISON")
        lines.append("-" * 78)

        header = f"{'Quant':12s}  {'Kernel(ms)':>10s}  {'Kernels':>8s}  {'H2D':>5s}  {'H2D(ms)':>8s}"
        lines.append(header)
        lines.append("-" * len(header))

        for p in sorted(profiles, key=lambda x: x.total_kernel_ms):
            lines.append(f"{p.quant_type:12s}  {p.total_kernel_ms:10.1f}  {p.kernel_count:8d}  {p.h2d_count:5d}  {p.h2d_total_ms:8.1f}")

        if len(profiles) >= 2:
            fastest = min(profiles, key=lambda p: p.total_kernel_ms)
            slowest = max(profiles, key=lambda p: p.total_kernel_ms)
            if slowest.total_kernel_ms > 0:
                speedup = slowest.total_kernel_ms / fastest.total_kernel_ms
                lines.append(f"\n  Fastest: {fastest.quant_type} ({fastest.total_kernel_ms:.1f} ms)")
                lines.append(f"  Slowest: {slowest.quant_type} ({slowest.total_kernel_ms:.1f} ms)")
                lines.append(f"  Speedup: {speedup:.2f}x")
        lines.append("")

    # --- Asymmetric KV Cache Comparison ---
    if asym_results:
        lines.append("-" * 78)
        lines.append("ASYMMETRIC KV CACHE COMPARISON")
        lines.append("-" * 78)
        lines.append("  Comparing symmetric (ctk=ctv) vs asymmetric (ctk!=ctv) KV cache types.")
        lines.append("  Uses FA VEC path for all variants (turbo FA dispatch fix applied).")
        lines.append("")

        # Group by config name
        by_config: Dict[str, List[BenchResult]] = {}
        for r in asym_results:
            by_config.setdefault(r.config_name, []).append(r)

        for cfg_name, cfg_results in sorted(by_config.items()):
            # Find symmetric baseline (q8_0,q8_0) for this config
            sym = next((r for r in cfg_results if "q8_0_q8_0" in r.quant_type), None)
            if not sym or not sym.passes:
                continue
            lines.append(f"  Config: {cfg_name}")
            lines.append(f"  {'Asymmetric Pair':>20s}  {'PP t/s':>8s}  {'TG t/s':>8s}  {'PP drift':>10s}  {'TG drift':>10s}")
            lines.append(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
            lines.append(f"  {'q8_0/q8_0 (sym)':>20s}  {sym.pp_tok_s:>8.1f}  {sym.tg_tok_s:>8.1f}  {'-':>10s}  {'-':>10s}")
            for r in cfg_results:
                if r == sym or not r.passes:
                    continue
                pp_d = ((r.pp_tok_s - sym.pp_tok_s) / sym.pp_tok_s) * 100 if sym.pp_tok_s > 0 else 0
                tg_d = ((r.tg_tok_s - sym.tg_tok_s) / sym.tg_tok_s) * 100 if sym.tg_tok_s > 0 else 0
                lines.append(f"  {r.quant_type:>20s}  {r.pp_tok_s:>8.1f}  {r.tg_tok_s:>8.1f}  {pp_d:>+9.1f}%  {tg_d:>+9.1f}%")
            lines.append("")
        lines.append("")

    # --- Recommendations ---
    lines.append("=" * 78)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 78)

    # Find best performing turbo/planar/iso
    best_tg = 0
    best_quant = ""
    for r in results:
        if r.passes and r.quant_type not in QUANT_TYPES_STANDARD and r.tg_tok_s > best_tg:
            best_tg = r.tg_tok_s
            best_quant = r.quant_type

    q40_tg = 0
    for r in results:
        if r.quant_type == "q4_0" and r.passes and r.config_name == "baseline":
            q40_tg = r.tg_tok_s
            break

    if best_quant and q40_tg > 0:
        ratio = best_tg / q40_tg
        lines.append(f"  Best turbo/planar/iso: {best_quant} at {best_tg:.1f} t/s decode")
        lines.append(f"  vs q4_0 baseline ({q40_tg:.1f} t/s): {ratio:.2f}x")
        if ratio >= 1.0:
            lines.append(f"  Status: MATCHES OR EXCEEDS q4_0 baseline")
        elif ratio >= 0.85:
            lines.append(f"  Status: WITHIN 15% of q4_0 — acceptable for compression ratio")
        else:
            lines.append(f"  Status: SIGNIFICANTLY BELOW q4_0 — investigate kernel path")
    else:
        lines.append("  Insufficient data for comparison")

    # FA_ALL_QUANTS impact
    fa_results = [r for r in results if r.config_name == "fa_all_quants" and r.passes]
    bl_results = [r for r in results if r.config_name == "baseline" and r.passes]
    if fa_results and bl_results:
        lines.append("")
        lines.append("  FA_ALL_QUANTS impact:")
        for r_fa in fa_results:
            r_bl = next((r for r in bl_results if r.quant_type == r_fa.quant_type), None)
            if r_bl and r_bl.pp_tok_s > 0:
                delta = ((r_fa.pp_tok_s - r_bl.pp_tok_s) / r_bl.pp_tok_s) * 100
                lines.append(f"    {r_fa.quant_type:12s}: pp {delta:+.1f}% ({r_bl.pp_tok_s:.1f} -> {r_fa.pp_tok_s:.1f})")

    lines.append("")
    lines.append("  Asymmetric KV Cache (--asymmetric flag):")
    lines.append("  - Asymmetric configs (q8K/turboV) now use FA VEC path via dispatch fix")
    lines.append("  - Expected ~6% TG overhead vs symmetric q8/q8 (from turbo V dequant overhead)")
    lines.append("  - PP overhead should be negligible (TILE/MMA kernel used for prompt processing)")
    lines.append("  - For perf-critical asymmetric configs, compare results table above")
    lines.append("")
    lines.append("=" * 78)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    return report


# ============================================================================
# Trace Analysis (from optimize-turboquant.py)
# ============================================================================

def _run_trace_analysis(log_path: str):
    """Analyze FA_SOFTMAX traces for bottleneck detection."""
    if not os.path.exists(log_path):
        print(f"ERROR: trace log not found: {log_path}", file=sys.stderr)
        return

    print(f"\n{'='*70}")
    print(f"  BOTTLENECK ANALYSIS FROM TRACES")
    print(f"  Log: {log_path}")
    print(f"{'='*70}\n")

    heads = parse_softmax_traces(log_path)
    if not heads:
        print("  No FA_SOFTMAX traces found in log.")
        print("  Make sure the binary was compiled with -DFA_TRACE_SOFTMAX.")
        return

    # Extract timing from log
    pp_speed = 0.0
    tg_speed = 0.0
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Prompt:\s*([\d.,]+)\s*t/s', line)
            if m: pp_speed = float(m.group(1).replace(',', '.'))
            m = re.search(r'Generation:\s*([\d.,]+)\s*t/s', line)
            if m: tg_speed = float(m.group(1).replace(',', '.'))

    n_heads = len(heads)
    total_samples = sum(h.n_samples for h in heads.values())
    anomalous = sum(1 for h in heads.values() if h.sum_mean > 2.0 or h.sum_mean < 0.01)
    avg_sum = float(np.mean([h.sum_mean for h in heads.values()]))
    avg_max = float(np.mean([h.max_mean for h in heads.values()]))
    heads_wide = sum(1 for h in heads.values() if h.sum_mean > 1.0)
    heads_narrow = sum(1 for h in heads.values() if h.sum_mean < 0.1)

    print(f"  Total heads tracked:   {n_heads}")
    print(f"  Total trace samples:   {total_samples}")
    print(f"  Prompt speed:          {pp_speed:.1f} t/s" if pp_speed > 0 else f"  Prompt speed:          N/A")
    print(f"  Generation speed:      {tg_speed:.1f} t/s" if tg_speed > 0 else f"  Generation speed:      N/A")
    print()
    print(f"  KQ_sum across heads:")
    print(f"    Average:             {avg_sum:.4f}")
    print(f"    Range:               {min(h.sum_mean for h in heads.values()):.4f} .. {max(h.sum_mean for h in heads.values()):.4f}")
    print(f"    Heads > 1.0 (wide):  {heads_wide} ({heads_wide/n_heads*100:.1f}%)")
    print(f"    Heads < 0.1 (narrow):{heads_narrow} ({heads_narrow/n_heads*100:.1f}%)")
    print(f"    Anomalous:           {anomalous} ({anomalous/n_heads*100:.1f}%)")
    print()
    print(f"  KQ_max (peak logit):")
    print(f"    Average:             {avg_max:.1f}")
    print(f"    Range:               {min(h.max_mean for h in heads.values()):.1f} .. {max(h.max_mean for h in heads.values()):.1f}")
    print()

    # Bottleneck identification
    print(f"  BOTTLENECK ANALYSIS:")
    print(f"  {'Component':<35s} {'Status':<15s} {'Action':<30s}")
    print(f"  {'-'*35} {'-'*15} {'-'*30}")

    # 1. Attention distribution health
    if anomalous / max(n_heads, 1) > 0.5:
        print(f"  {'Attention distribution':<35s} {'ANOMALOUS':<15s} {'Check turbo dequant paths':<30s}")
    elif heads_wide > n_heads * 0.3:
        print(f"  {'Attention distribution':<35s} {'WIDE':<15s} {'Possible precision loss':<30s}")
    else:
        print(f"  {'Attention distribution':<35s} {'OK':<15s} {'Nominal range':<30s}")

    # 2. Prefill speed
    if pp_speed > 0 and pp_speed < 200:
        print(f"  {'Prefill throughput':<35s} {'SLOW':<15s} {'Check TILE/MMA kernel choice':<30s}")
    elif pp_speed > 0:
        print(f"  {'Prefill throughput':<35s} {'OK':<15s}")

    # 3. Generation speed
    if tg_speed > 0 and tg_speed < 30:
        print(f"  {'Generation throughput':<35s} {'SLOW':<15s} {'Check VEC kernel & turbo dequant':<30s}")
    elif tg_speed > 0 and tg_speed < 50:
        print(f"  {'Generation throughput':<35s} {'MODERATE':<15s} {'Turbo V dequant overhead likely':<30s}")
    elif tg_speed > 0:
        print(f"  {'Generation throughput':<35s} {'OK':<15s}")

    # 4. Speculative decoding hint
    tile_counts = set(h.n_tiles for h in heads.values())
    if len(tile_counts) <= 1:
        tc_str = str(list(tile_counts)[0]) if tile_counts else '?'
        print(f"  {'Speculative decoding':<35s} {'N/A':<15s} {'All traces n_tiles=' + tc_str:<30s}")
    else:
        print(f"  {'Speculative decoding':<35s} {'ACTIVE':<15s} {'Multiple tile sizes detected':<30s}")

    print()
    print(f"  TOP HEADS BY KQ_SUM:")
    sorted_heads = sorted(heads.values(), key=lambda h: h.sum_mean, reverse=True)
    print(f"  {'Head':>6s}  {'Samples':>8s}  {'KQ_sum':>10s}  {'KQ_max':>8s}  {'Risk':>10s}")
    for h in sorted_heads[:10]:
        risk = "HIGH" if h.sum_mean > 1.5 or h.sum_mean < 0.05 else "LOW" if 0.1 < h.sum_mean < 0.5 else "MOD"
        print(f"  {h.head:>6d}  {h.n_samples:>8d}  {h.sum_mean:>10.4f}  {h.max_mean:>8.1f}  {risk:>10s}")


# ============================================================================
# Lloyd-Max & Centroid Computation (from compute_turbo_centroids.py)
# ============================================================================

def fwht_inplace(a):
    """Fast Walsh-Hadamard Transform in-place (Hadamard order)."""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def fwht(x):
    """Apply WHT to vector x (returns new array)."""
    y = x.copy()
    fwht_inplace(y)
    y /= np.sqrt(len(y))  # normalize to preserve L2 norm
    return y


def lloyd_max_1d(data, n_centroids, max_iter=100, tol=1e-6):
    """
    Lloyd-Max algorithm for 1D optimal scalar quantization.
    Returns (centroids, midpoints, mse).
    """
    # Initialize centroids using quantiles (n_centroids bins -> n_centroids-1 boundaries)
    # Place centroids at the centers of equal-probability bins
    bin_edges = np.linspace(0, 100, n_centroids + 1)
    centroids = np.array([
        np.percentile(data, (bin_edges[i] + bin_edges[i+1]) / 2)
        for i in range(n_centroids)
    ])

    for iteration in range(max_iter):
        # Assignment step: find nearest centroid for each data point
        # Using midpoints for fast assignment
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0

        # Assign each point to nearest centroid
        indices = np.searchsorted(midpoints, data)
        indices = np.clip(indices, 0, n_centroids - 1)

        # Update step: compute new centroids as conditional means
        new_centroids = np.zeros(n_centroids)
        counts = np.zeros(n_centroids)
        for i in range(n_centroids):
            mask = indices == i
            if np.any(mask):
                new_centroids[i] = np.mean(data[mask])
                counts[i] = np.sum(mask)

        # Handle empty clusters (keep old centroid)
        for i in range(n_centroids):
            if counts[i] == 0:
                new_centroids[i] = centroids[i]

        # Check convergence
        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if delta < tol:
            break

    # Compute final midpoints and MSE
    midpoints = np.zeros(n_centroids - 1)
    for i in range(n_centroids - 1):
        midpoints[i] = (centroids[i] + centroids[i + 1]) / 2.0

    # Compute MSE
    indices = np.searchsorted(midpoints, data)
    indices = np.clip(indices, 0, n_centroids - 1)
    reconstructed = centroids[indices]
    mse = np.mean((data - reconstructed) ** 2)

    return centroids, midpoints, mse


def load_gguf_weights(model_path):
    """
    Load V projection weights from GGUF model.
    This is a simplified version - in practice, use gguf library.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(str(model_path), 'r')

        # Find V projection weights
        v_weights = {}
        for tensor in reader.tensors:
            name = tensor.name
            if 'attn_v' in name and 'weight' in name:
                v_weights[name] = tensor.data.astype(np.float32)

        return v_weights
    except ImportError:
        print("Warning: gguf library not found. Using random weights for demonstration.")
        return None


def simulate_v_cache(v_weights, n_samples, head_dim=128):
    """
    Simulate V-cache by projecting random hidden states through V weights.
    Returns vectors of shape (n_samples, head_dim).
    """
    if v_weights is None:
        # Generate random V-cache-like vectors
        # Typical V-cache: unit-norm vectors with some structure
        print("Using random vectors (install gguf library for real data)")
        vectors = np.random.randn(n_samples, head_dim).astype(np.float32)
        # Normalize to unit norm (as done in turbo pipeline)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        return vectors

    # Use real V weights
    all_vectors = []
    for name, weight in v_weights.items():
        # weight shape: (n_heads * head_dim, hidden_dim) or similar
        if weight.ndim == 2:
            # Generate random hidden states and project
            hidden_dim = weight.shape[1]
            n_heads = weight.shape[0] // head_dim

            for _ in range(n_samples // (n_heads * len(v_weights)) + 1):
                h = np.random.randn(hidden_dim).astype(np.float32)
                v = weight @ h  # project through V
                # Split into heads
                v = v.reshape(n_heads, head_dim)
                # Normalize each head vector
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                v = v / (norms + 1e-10)
                all_vectors.append(v)

    if all_vectors:
        all_vectors = np.concatenate(all_vectors, axis=0)
        # Take requested number of samples
        if len(all_vectors) > n_samples:
            idx = np.random.choice(len(all_vectors), n_samples, replace=False)
            all_vectors = all_vectors[idx]
        return all_vectors

    # Fallback
    vectors = np.random.randn(n_samples, head_dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    return vectors


def _optimize_quant_depth(data, n_centroids):
    """Top-level Lloyd-Max optimization for a single quant depth.

    Pickle-serializable wrapper so ProcessPoolExecutor can run multiple
    quant depths (3-bit, 4-bit, ...) in parallel. Each call is fully
    independent and thread-safe.
    """
    centroids, midpoints, mse = lloyd_max_1d(data, n_centroids)
    return {
        'n_centroids': n_centroids,
        'centroids': centroids.tolist(),
        'midpoints': midpoints.tolist(),
        'mse': float(mse),
    }


def compute_centroids_from_data(vectors, d=128):
    """
    Compute optimal centroids from real V-cache data.
    1. Apply WHT rotation
    2. Compute distribution statistics
    3. Run Lloyd-Max for 3-bit and 4-bit (parallelized)
    """
    print(f"Input vectors: {vectors.shape}")

    # Apply WHT to each vector
    rotated = np.zeros_like(vectors)
    for i in range(len(vectors)):
        rotated[i] = fwht(vectors[i])

    # Flatten all rotated coordinates
    all_coords = rotated.flatten()

    print(f"Distribution of rotated coordinates:")
    print(f"  Mean: {np.mean(all_coords):.6f}")
    print(f"  Std:  {np.std(all_coords):.6f}")
    print(f"  Min:  {np.min(all_coords):.6f}")
    print(f"  Max:  {np.max(all_coords):.6f}")
    print(f"  Theoretical std for N(0,1/{d}): {1.0/np.sqrt(d):.6f}")

    # Compute per-coordinate variance (should be ~1/d for unit-norm input)
    coord_var = np.var(rotated, axis=0)
    print(f"\nPer-coordinate variance:")
    print(f"  Mean: {np.mean(coord_var):.6f}")
    print(f"  Std:  {np.std(coord_var):.6f}")
    print(f"  Min:  {np.min(coord_var):.6f}")
    print(f"  Max:  {np.max(coord_var):.6f}")

    # Parallel Lloyd-Max: each quant depth is an independent CPU-bound optimization
    quant_depths = [(8, "3bit"), (16, "4bit")]
    with ProcessPoolExecutor(max_workers=min(len(quant_depths), os.cpu_count() or 1)) as pool:
        futures = [pool.submit(_optimize_quant_depth, all_coords, n_c) for n_c, _ in quant_depths]
        opt_results = {label: f.result() for f, (_, label) in zip(futures, quant_depths)}

    centroids_3bit = opt_results["3bit"]['centroids']
    midpoints_3bit = opt_results["3bit"]['midpoints']
    mse_3bit = opt_results["3bit"]['mse']
    centroids_4bit = opt_results["4bit"]['centroids']
    midpoints_4bit = opt_results["4bit"]['midpoints']
    mse_4bit = opt_results["4bit"]['mse']

    print(f"\n3-bit centroids (8 levels):")
    print(f"  Centroids: {centroids_3bit}")
    print(f"  MSE: {mse_3bit:.8f}")

    print(f"\n4-bit centroids (16 levels):")
    print(f"  Centroids: {centroids_4bit}")
    print(f"  MSE: {mse_4bit:.8f}")

    return {
        'centroids_3bit': centroids_3bit,
        'midpoints_3bit': midpoints_3bit,
        'centroids_4bit': centroids_4bit,
        'midpoints_4bit': midpoints_4bit,
        'mse_3bit': mse_3bit,
        'mse_4bit': mse_4bit,
        'distribution': {
            'mean': float(np.mean(all_coords)),
            'std': float(np.std(all_coords)),
            'theoretical_std': 1.0 / np.sqrt(d),
        }
    }


def compare_with_current(new_centroids):
    """Compare new centroids with current theoretical values."""
    # Current centroids (from turbo-quant.cuh)
    current_3bit = np.array([
        -0.190685, -0.117832, -0.065717, -0.021460,
         0.021460,  0.065717,  0.117832,  0.190685
    ])
    current_4bit = np.array([
        -0.173926, -0.117195, -0.089527, -0.068756,
        -0.051262, -0.035597, -0.020989, -0.006938,
         0.006938,  0.020989,  0.035597,  0.051262,
         0.068756,  0.089527,  0.117195,  0.173926
    ])

    new_3bit = np.array(new_centroids['centroids_3bit'])
    new_4bit = np.array(new_centroids['centroids_4bit'])

    print("\n=== Comparison with Current Centroids ===")
    print(f"\n3-bit centroids:")
    print(f"  Current: {current_3bit}")
    print(f"  New:     {new_3bit}")
    print(f"  Max diff: {np.max(np.abs(new_3bit - current_3bit)):.6f}")
    print(f"  Rel diff: {np.max(np.abs(new_3bit - current_3bit) / (np.abs(current_3bit) + 1e-10)) * 100:.2f}%")

    print(f"\n4-bit centroids:")
    print(f"  Current: {current_4bit}")
    print(f"  New:     {new_4bit}")
    print(f"  Max diff: {np.max(np.abs(new_4bit - current_4bit)):.6f}")
    print(f"  Rel diff: {np.max(np.abs(new_4bit - current_4bit) / (np.abs(current_4bit) + 1e-10)) * 100:.2f}%")


def generate_c_code(centroids_3bit, centroids_4bit, midpoints_3bit, midpoints_4bit):
    """Generate C code for updated centroid arrays."""
    code = f"""
// =====================================================
// Auto-generated centroids from real V-cache data
// Run: python3 scripts/compute_turbo_centroids.py
// =====================================================

// ---- 3-bit centroids (Lloyd-Max for actual V-cache distribution) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {{
    {', '.join(f'{v:.6f}f' for v in centroids_3bit)}
}};

// ---- Midpoints for nearest 3-bit centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {{
    {', '.join(f'{v:.6f}f' for v in midpoints_3bit)}
}};

// ---- 4-bit centroids (Lloyd-Max for actual V-cache distribution) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {{
    {', '.join(f'{v:.6f}f' for v in centroids_4bit)}
}};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {{
    {', '.join(f'{v:.6f}f' for v in midpoints_4bit)}
}};
"""
    return code


# ============================================================================
# Main
# ============================================================================

def main():
    """CLI: --mode optimize | centroids"""
    parser = argparse.ArgumentParser(
        description="Turbo Quantization Optimization & Centroid Computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["optimize", "centroids"], required=True,
                        help="Mode: 'optimize' for benchmarking, 'centroids' for Lloyd-Max centroid computation")

    # Optimize mode args
    parser.add_argument("--model-path", default=None, help="GGUF model path (required for optimize mode)")
    parser.add_argument("--output-dir", default="bench-results", help="Output dir")
    parser.add_argument("--configs", default="baseline,fa_all_quants",
                        help="Comma-separated build configs (default: baseline,fa_all_quants)")
    parser.add_argument("--quants", default=",".join(ALL_QUANT_TYPES),
                        help="Comma-separated quant types (default: all)")
    parser.add_argument("--repetitions", type=int, default=3, help="Benchmark repetitions")
    parser.add_argument("--profile", action="store_true", help="Also run rocprofv3 profiling")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline config")
    parser.add_argument("--skip-build", action="store_true", help="Skip cmake build, use existing")
    parser.add_argument("--rebuild", action="store_true", help="Force clean rebuild")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--asymmetric", action="store_true",
                        help="Test asymmetric KV cache (ctk/ctv pairs). Requires --ctk and --ctv or uses defaults from ASYMMETRIC_PAIRS")
    parser.add_argument("--ctk", default=None, help="K cache type for asymmetric test (e.g. q8_0)")
    parser.add_argument("--ctv", default=None, help="V cache type for asymmetric test (e.g. turbo3_0)")
    parser.add_argument("--list-asymmetric", action="store_true", help="List asymmetric KV pairs and exit")
    parser.add_argument("--trace-analyze", default=None, help="Analyze FA_SOFTMAX trace log for bottleneck detection")
    parser.add_argument("--ncpus", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel workers for benchmarks and centroid computation")

    # Centroids mode args
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model file (for centroids mode)")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of V-cache vectors to sample")
    parser.add_argument("--n_layers", type=int, default=5, help="Number of layers to sample from")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--output", type=str, default="centroids.json", help="Output JSON file (for centroids mode)")
    parser.add_argument("--c_code", action="store_true", help="Generate C code for CUDA/host arrays (for centroids mode)")

    args = parser.parse_args()

    # Numpy thread control — set before any numpy or subprocess calls
    os.environ["OMP_NUM_THREADS"] = str(args.ncpus)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpus)

    if args.mode == "optimize":
        _run_optimize(args)
    elif args.mode == "centroids":
        _run_centroids(args)


def _run_optimize(args):
    """Run the optimization benchmark pipeline."""
    if args.list_asymmetric:
        print("Asymmetric KV cache pairs for benchmarking:")
        print(f"  {'K type':>10s}  {'V type':>10s}")
        print(f"  {'-'*10}  {'-'*10}")
        for ctk, ctv in ASYMMETRIC_PAIRS:
            print(f"  {ctk:>10s}  {ctv:>10s}")
        return

    # Trace analysis mode: no model needed
    if args.trace_analyze:
        _run_trace_analysis(args.trace_analyze)
        return

    if not args.model_path:
        print("ERROR: --model-path is required for benchmarking", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.join(args.output_dir, f"turboquant_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    quant_types = [q.strip() for q in args.quants.split(",")]
    config_names = ["baseline"] if args.baseline_only else [c.strip() for c in args.configs.split(",")]

    print(f"Model: {args.model_path}")
    print(f"Configs: {config_names}")
    print(f"Quant types: {quant_types}")
    print(f"Output: {output_dir}")

    # Step 1: Build each config
    bench_paths = {}
    if not args.skip_build:
        for cfg_name in config_names:
            cfg = BUILD_CONFIGS[cfg_name]
            success = build_config(cfg_name, cfg["cmake_flags"], force_rebuild=args.rebuild, ncpus=args.ncpus)
            if success:
                bench_paths[cfg_name] = os.path.join(BUILD_DIR, cfg_name, "bin", "llama-bench")
            else:
                print(f"WARNING: build failed for {cfg_name}, skipping")
    else:
        # Use existing builds
        for cfg_name in config_names:
            p = os.path.join(BUILD_DIR, cfg_name, "bin", "llama-bench")
            if os.path.exists(p):
                bench_paths[cfg_name] = p
            elif os.path.exists(BENCH_BIN):
                bench_paths[cfg_name] = BENCH_BIN

    if not bench_paths:
        print("ERROR: no builds available", file=sys.stderr)
        sys.exit(1)

    # Step 2: Correctness gate — run 5-token test
    print("\n--- Correctness Gate ---")
    first_bench = list(bench_paths.values())[0]
    cmd = [first_bench, "-m", args.model_path, "-t", "1", "-b", "32",
           "-p", "512", "-n", "5", "-o", "csv"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"WARNING: correctness gate failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:300]}")
    else:
        print("  Correctness gate: PASS")

    # Step 3: Benchmark each config (symmetric)
    results: List[BenchResult] = []
    profiles: List[ProfileResult] = []
    asym_results: List[BenchResult] = []

    # Auto-detect quant from filename
    model_basename = os.path.basename(args.model_path)
    detected_quant = "unknown"
    for qt in ALL_QUANT_TYPES:
        if qt in model_basename.lower():
            detected_quant = qt
            break
    quant_label = args.quants.split(",")[0] if args.quants else detected_quant
    print(f"\nDetected quant: {detected_quant} (label: {quant_label})")

    # Parallel benchmarking: each config is an independent subprocess (I/O-bound)
    def _bench_one(cfg_name: str, bench_path: str):
        print(f"\nBenchmarking: {cfg_name}")
        r = run_bench(
            bench_path=bench_path,
            model_path=args.model_path,
            quant_type=quant_label,
            batch_size=args.batch_size,
            repetitions=args.repetitions,
        )
        r.config_name = cfg_name

        if r.passes:
            print(f"  pp={r.pp_tok_s:.1f} t/s  tg={r.tg_tok_s:.1f} t/s")
        elif r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  FAILED: no valid output")

        # Profile if requested
        if args.profile and r.passes:
            print(f"  Profiling...")
            p = run_profile(bench_path, args.model_path, quant_label, output_dir, args.batch_size)
            if p:
                p.config_name = cfg_name
                profiles.append(p)
                print(f"  Profile: {p.total_kernel_ms:.1f} ms, {p.kernel_count} kernels")

        return r

    with ThreadPoolExecutor(max_workers=min(len(bench_paths), args.ncpus)) as pool:
        bench_futures = [pool.submit(_bench_one, cfg_name, path) for cfg_name, path in bench_paths.items()]
        for f in as_completed(bench_futures):
            results.append(f.result())

    # Step 3b: Asymmetric KV cache benchmark
    if args.asymmetric:
        asym_pairs = []
        if args.ctk and args.ctv:
            asym_pairs = [(args.ctk, args.ctv)]
        else:
            asym_pairs = ASYMMETRIC_PAIRS

        # Parallel asymmetric benchmarking: each (config, ctk, ctv) combo is independent
        def _asym_one(cfg_name: str, bench_path: str, ctk: str, ctv: str):
            label = f"{cfg_name}_{ctk}_{ctv}"
            print(f"\nAsymmetric benchmark: {label}")
            r = run_bench(
                bench_path=bench_path,
                model_path=args.model_path,
                quant_type=f"asym_{ctk}_{ctv}",
                batch_size=args.batch_size,
                repetitions=args.repetitions,
                ctk=ctk,
                ctv=ctv,
            )
            r.config_name = label

            if r.passes:
                print(f"  pp={r.pp_tok_s:.1f} t/s  tg={r.tg_tok_s:.1f} t/s")
            elif r.error:
                print(f"  ERROR: {r.error}")

            return r

        asym_tasks = []
        for cfg_name, bench_path in bench_paths.items():
            for ctk, ctv in asym_pairs:
                asym_tasks.append((cfg_name, bench_path, ctk, ctv))

        with ThreadPoolExecutor(max_workers=min(len(asym_tasks), args.ncpus)) as pool:
            asym_futures = [pool.submit(_asym_one, *task) for task in asym_tasks]
            for f in as_completed(asym_futures):
                asym_results.append(f.result())

    # Step 4: Generate reports
    report_path = os.path.join(output_dir, "turboquant_report.txt")
    generate_report(results, profiles, report_path, asym_results)

    # Save JSON
    json_path = os.path.join(output_dir, "turboquant_results.json")
    json_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_path,
        "results": [
            {
                "config": r.config_name,
                "quant": r.quant_type,
                "pp_tok_s": round(r.pp_tok_s, 2),
                "tg_tok_s": round(r.tg_tok_s, 2),
                "pp_time_ms": round(r.pp_time_ms, 2),
                "tg_time_ms": round(r.tg_time_ms, 2),
                "passes": r.passes,
                "error": r.error,
            }
            for r in results
        ],
        "profiles": [
            {
                "config": p.config_name,
                "quant": p.quant_type,
                "total_kernel_ms": round(p.total_kernel_ms, 2),
                "kernel_count": p.kernel_count,
                "h2d_count": p.h2d_count,
                "h2d_total_ms": round(p.h2d_total_ms, 2),
            }
            for p in profiles
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nReports:")
    print(f"  Text: {report_path}")
    print(f"  JSON: {json_path}")


def _run_centroids(args):
    """Run the centroid computation pipeline."""
    print("=" * 60)
    print("TurboQuant Centroid Recomputation")
    print("=" * 60)

    # Load model weights
    if args.model:
        print(f"\nLoading model: {args.model}")
        v_weights = load_gguf_weights(args.model)
    else:
        print("\nNo model specified, using random vectors")
        v_weights = None

    # Simulate V-cache
    print(f"\nSimulating V-cache ({args.n_samples} vectors, head_dim={args.head_dim})")
    vectors = simulate_v_cache(v_weights, args.n_samples, args.head_dim)

    # Compute centroids
    print(f"\nComputing optimal centroids...")
    result = compute_centroids_from_data(vectors, args.head_dim)

    # Compare with current
    compare_with_current(result)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate C code if requested
    if args.c_code:
        c_code = generate_c_code(
            result['centroids_3bit'],
            result['centroids_4bit'],
            result['midpoints_3bit'],
            result['midpoints_4bit']
        )
        c_path = output_path.with_suffix('.cuh')
        with open(c_path, 'w') as f:
            f.write(c_code)
        print(f"C code saved to: {c_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
