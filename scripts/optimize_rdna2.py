#!/usr/bin/env python3
"""
RDNA2 Optimization Script — Targets validated levers from hipfire #298-#304

Tests optimizations against two benchmark models:
  - Dense:  gemma-4-12B Q4_K_XL (6.3 GB)
  - MoE:    gemma-4-26B-A4B Q4_K_XL (14 GB, 4B active)

Each "optimization" is a cmake build variant or runtime flag change,
rebuilding from source, benchmarking both models, and reporting deltas.

Usage:
    python3 scripts/optimize_rdna2.py --model-dense /path/to/12b.gguf --model-moe /path/to/26b.gguf
    python3 scripts/optimize_rdna2.py --model-dense /path/to/12b.gguf --variants baseline,hipgraph
    python3 scripts/optimize_rdna2.py --model-dense /path/to/12b.gguf --variants all --skip-build
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_BASE = os.path.join(REPO_ROOT, "build-opt")

# Benchmark params
DEFAULT_BATCH = 512
DEFAULT_NPROMPT = 512
DEFAULT_NGEN = 128
DEFAULT_THREADS = 1
DEFAULT_REPS = 3

# ============================================================================
# Build Variants — each targets a specific hipfire-validated lever
# ============================================================================

VARIANTS = {
    "baseline": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
        ],
        "description": "Standard build, no turbo FA instances. Baseline.",
        "hipfire_ref": "Phase 0 baseline",
    },
    "hipgraph": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
            "-DGGML_CUDA_GRAPHS=ON",
        ],
        "description": "hipGraph capture enabled. hipfire #300 F1: +10-20%.",
        "hipfire_ref": "#300 F1: hipGraph for prefill (+10-20%)",
    },
    "fa_turbo": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        ],
        "description": "FA instances for all quant types (turbo/planar/iso).",
        "hipfire_ref": "N/A — llama.cpp specific",
    },
    "fa_graphs": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            "-DGGML_CUDA_GRAPHS=ON",
        ],
        "description": "FA + hipGraphs: enables turbo FA with launch overhead reduction.",
        "hipfire_ref": "N/A — llama.cpp specific",
    },
    "asymmetric_fa": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            "-DGGML_CUDA_GRAPHS=ON",
            "-DGGML_CUDA_FA_TRACE_SOFTMAX=ON",
        ],
        "description": "FA + Graphs + traces: enables asymmetric KV FA + diagnostic tracing.",
        "hipfire_ref": "N/A — llama.cpp specific",
    },
    "debug_build": {
        "cmake_flags": [
            "-DGGML_HIP=ON",
            "-DGGML_HIP_UMA=OFF",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
        ],
        "description": "Debug build — checks for assertions, UB, etc.",
        "hipfire_ref": "Diagnostics only",
    },
}

# Variants to test with different batch sizes (wave-32 alignment)
BATCH_SWEEP_VARIANTS = [32, 64, 128, 256, 512]

# ============================================================================
# Data types
# ============================================================================

@dataclass
class BenchResult:
    variant: str
    model_label: str
    batch_size: int
    pp_tok_s: float = 0.0
    tg_tok_s: float = 0.0
    pp_ns: int = 0
    tg_ns: int = 0
    passes: bool = False
    error: str = ""


@dataclass
class VariantReport:
    variant: str
    description: str
    hipfire_ref: str
    dense_results: Dict[int, BenchResult] = field(default_factory=dict)  # batch -> result
    moe_results: Dict[int, BenchResult] = field(default_factory=dict)


# ============================================================================
# Build
# ============================================================================

def build_variant(variant_name: str, cmake_flags: List[str], force: bool = False) -> Optional[str]:
    """Build a variant. Returns path to llama-bench or None on failure."""
    build_dir = os.path.join(BUILD_BASE, variant_name)

    if force and os.path.exists(build_dir):
        subprocess.run(["rm", "-rf", build_dir], check=True)

    os.makedirs(build_dir, exist_ok=True)

    # Check if already built
    bench_bin = os.path.join(build_dir, "bin", "llama-bench")
    if os.path.exists(bench_bin):
        print(f"  [{variant_name}] Already built: {bench_bin}")
        return bench_bin

    print(f"  [{variant_name}] Configuring...")
    cmd = ["cmake", "-B", build_dir, "-S", REPO_ROOT] + cmake_flags
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [{variant_name}] cmake FAILED: {result.stderr[-300:]}")
        return None

    print(f"  [{variant_name}] Building...")
    cmd = ["cmake", "--build", build_dir, "-j" + str(os.cpu_count() or 4),
           "--target", "llama-bench"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  [{variant_name}] build FAILED: {result.stderr[-300:]}")
        return None

    if not os.path.exists(bench_bin):
        print(f"  [{variant_name}] llama-bench not found after build")
        return None

    print(f"  [{variant_name}] Build OK")
    return bench_bin


# ============================================================================
# Benchmark
# ============================================================================

def run_bench(
    bench_path: str,
    model_path: str,
    batch_size: int = DEFAULT_BATCH,
    n_prompt: int = DEFAULT_NPROMPT,
    n_gen: int = DEFAULT_NGEN,
    repetitions: int = DEFAULT_REPS,
    timeout_s: int = 600,
) -> BenchResult:
    """Run llama-bench and parse CSV output."""
    result = BenchResult(variant="", model_label="", batch_size=batch_size)

    cmd = [
        bench_path,
        "-m", model_path,
        "-t", str(DEFAULT_THREADS),
        "-b", str(batch_size),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-o", "csv",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("#") or not line.strip() or line.startswith("build_"):
                continue
            parts = line.split(",")
            if len(parts) < 40:
                continue
            try:
                n_p = int(parts[33].strip('"'))
                n_g = int(parts[34].strip('"'))
                avg_ns = float(parts[37].strip('"'))
                avg_ts = float(parts[39].strip('"'))
                if n_p > 0 and n_g == 0:
                    result.pp_tok_s = avg_ts
                    result.pp_ns = int(avg_ns)
                elif n_g > 0 and n_p == 0:
                    result.tg_tok_s = avg_ts
                    result.tg_ns = int(avg_ns)
            except (ValueError, IndexError):
                continue
        result.passes = result.pp_tok_s > 0 or result.tg_tok_s > 0
    except subprocess.TimeoutExpired:
        result.error = f"Timeout {timeout_s}s"
    except Exception as e:
        result.error = str(e)

    return result


# ============================================================================
# Correctness gate
# ============================================================================

def correctness_gate(bench_path: str, model_path: str) -> bool:
    """5-token sanity check."""
    cmd = [
        bench_path, "-m", model_path,
        "-t", "1", "-b", "32", "-p", "512", "-n", "5", "-o", "csv",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return proc.returncode == 0 and "512" in proc.stdout
    except Exception:
        return False


# ============================================================================
# Report
# ============================================================================

def generate_report(
    reports: List[VariantReport],
    baseline_report: Optional[VariantReport],
    output_path: str,
):
    """Generate markdown + CSV report."""
    lines = []
    lines.append("# RDNA2 Optimization Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**GPU:** AMD Radeon RX 6800 XT (gfx1030, RDNA2)")
    lines.append(f"**Source:** hipfire #298-#304 RDNA2 research")
    lines.append("")

    # Baselines
    lines.append("## Baselines")
    lines.append("")
    if baseline_report:
        for label, results in [("Dense (12B)", baseline_report.dense_results),
                               ("MoE (26B-A4B)", baseline_report.moe_results)]:
            if results:
                lines.append(f"### {label}")
                lines.append("| Batch | pp512 (t/s) | tg128 (t/s) |")
                lines.append("|-------|-------------|-------------|")
                for bs in sorted(results.keys()):
                    r = results[bs]
                    lines.append(f"| {bs} | {r.pp_tok_s:.1f} | {r.tg_tok_s:.1f} |")
                lines.append("")

    # Variant comparisons
    lines.append("## Optimization Results")
    lines.append("")
    lines.append("| Variant | Description | Dense pp512 | Dense tg128 | MoE pp512 | MoE tg128 | Dense pp delta | Dense tg delta |")
    lines.append("|---------|-------------|-------------|-------------|-----------|-----------|----------------|----------------|")

    bl_dense = baseline_report.dense_results.get(DEFAULT_BATCH) if baseline_report else None
    bl_moe = baseline_report.moe_results.get(DEFAULT_BATCH) if baseline_report else None

    for report in reports:
        d = report.dense_results.get(DEFAULT_BATCH)
        m = report.moe_results.get(DEFAULT_BATCH)

        d_pp = f"{d.pp_tok_s:.1f}" if d and d.passes else "N/A"
        d_tg = f"{d.tg_tok_s:.1f}" if d and d.passes else "N/A"
        m_pp = f"{m.pp_tok_s:.1f}" if m and m.passes else "N/A"
        m_tg = f"{m.tg_tok_s:.1f}" if m and m.passes else "N/A"

        # Delta
        if d and d.passes and bl_dense and bl_dense.passes and bl_dense.pp_tok_s > 0:
            d_pp_d = f"{((d.pp_tok_s - bl_dense.pp_tok_s) / bl_dense.pp_tok_s * 100):+.1f}%"
        else:
            d_pp_d = "-"
        if d and d.passes and bl_dense and bl_dense.passes and bl_dense.tg_tok_s > 0:
            d_tg_d = f"{((d.tg_tok_s - bl_dense.tg_tok_s) / bl_dense.tg_tok_s * 100):+.1f}%"
        else:
            d_tg_d = "-"

        lines.append(f"| {report.variant} | {report.description[:40]} | {d_pp} | {d_tg} | {m_pp} | {m_tg} | {d_pp_d} | {d_tg_d} |")

    lines.append("")

    # Batch sweep
    lines.append("## Batch Size Sweep (Dense 12B)")
    lines.append("")
    lines.append("| Batch | pp512 (t/s) | tg128 (t/s) | Wave alignment |")
    lines.append("|-------|-------------|-------------|----------------|")
    for report in reports:
        for bs in sorted(report.dense_results.keys()):
            r = report.dense_results[bs]
            if r.passes and report.variant == "baseline":
                aligned = "OK" if bs % 32 == 0 else f"NOT ({bs} mod 32 = {bs % 32})"
                lines.append(f"| {bs} | {r.pp_tok_s:.1f} | {r.tg_tok_s:.1f} | {aligned} |")
    lines.append("")

    # hipfire cross-reference
    lines.append("## hipfire Cross-Reference")
    lines.append("")
    lines.append("| hipfire Lever | hipfire Result | Our Result | Notes |")
    lines.append("|---------------|----------------|------------|-------|")
    for report in reports:
        if report.hipfire_ref and report.hipfire_ref.startswith("#"):
            d = report.dense_results.get(DEFAULT_BATCH)
            if d and d.passes and bl_dense and bl_dense.passes:
                delta = ((d.pp_tok_s - bl_dense.pp_tok_s) / bl_dense.pp_tok_s * 100)
                lines.append(f"| {report.variant} | {report.hipfire_ref} | {delta:+.1f}% pp | |")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RDNA2 optimization script targeting hipfire-validated levers",
    )
    parser.add_argument("--model-dense", required=True, help="Dense model GGUF path")
    parser.add_argument("--model-moe", default="", help="MoE model GGUF path")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names, or 'all'")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPS)
    parser.add_argument("--output-dir", default="bench-results")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--batch-sweep", action="store_true",
                        help="Also test different batch sizes on baseline")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"rdna2_opt_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # Select variants
    if args.variants == "all":
        variant_names = list(VARIANTS.keys())
    else:
        variant_names = [v.strip() for v in args.variants.split(",")]

    print(f"Variants: {variant_names}")
    print(f"Dense model: {args.model_dense}")
    if args.model_moe:
        print(f"MoE model: {args.model_moe}")

    # Step 1: Build all variants
    bench_paths = {}
    if not args.skip_build:
        print("\n=== BUILDING VARIANTS ===")
        for vname in variant_names:
            v = VARIANTS[vname]
            path = build_variant(vname, v["cmake_flags"], force=args.rebuild)
            if path:
                bench_paths[vname] = path
    else:
        for vname in variant_names:
            p = os.path.join(BUILD_BASE, vname, "bin", "llama-bench")
            if os.path.exists(p):
                bench_paths[vname] = p

    if not bench_paths:
        print("ERROR: no builds available")
        sys.exit(1)

    # Step 2: Correctness gate
    print("\n=== CORRECTNESS GATE ===")
    first_bench = list(bench_paths.values())[0]
    ok = correctness_gate(first_bench, args.model_dense)
    print(f"  Dense: {'PASS' if ok else 'FAIL'}")
    if args.model_moe:
        ok2 = correctness_gate(first_bench, args.model_moe)
        print(f"  MoE: {'PASS' if ok2 else 'FAIL'}")

    # Step 3: Benchmark
    print("\n=== BENCHMARKING ===")
    reports: List[VariantReport] = []

    for vname in variant_names:
        if vname not in bench_paths:
            continue

        v = VARIANTS[vname]
        bench = bench_paths[vname]
        report = VariantReport(
            variant=vname,
            description=v["description"],
            hipfire_ref=v.get("hipfire_ref", ""),
        )

        # Dense model
        print(f"\n  [{vname}] Dense 12B @ batch={args.batch_size}")
        r = run_bench(bench, args.model_dense, args.batch_size, repetitions=args.repetitions)
        r.variant = vname
        r.model_label = "dense"
        report.dense_results[args.batch_size] = r
        if r.passes:
            print(f"    pp={r.pp_tok_s:.1f} tg={r.tg_tok_s:.1f}")
        else:
            print(f"    FAILED: {r.error}")

        # MoE model
        if args.model_moe:
            print(f"  [{vname}] MoE 26B-A4B @ batch={args.batch_size}")
            r2 = run_bench(bench, args.model_moe, args.batch_size, repetitions=args.repetitions)
            r2.variant = vname
            r2.model_label = "moe"
            report.moe_results[args.batch_size] = r2
            if r2.passes:
                print(f"    pp={r2.pp_tok_s:.1f} tg={r2.tg_tok_s:.1f}")
            else:
                print(f"    FAILED: {r2.error}")

        reports.append(report)

    # Step 4: Batch sweep on baseline
    if args.batch_sweep and "baseline" in bench_paths:
        print("\n=== BATCH SWEEP (baseline) ===")
        baseline = reports[0] if reports else None
        if baseline and baseline.variant == "baseline":
            for bs in BATCH_SWEEP_VARIANTS:
                print(f"  batch={bs}")
                r = run_bench(bench_paths["baseline"], args.model_dense, bs, repetitions=args.repetitions)
                r.variant = "baseline"
                r.model_label = "dense"
                baseline.dense_results[bs] = r
                if r.passes:
                    print(f"    pp={r.pp_tok_s:.1f} tg={r.tg_tok_s:.1f}")

    # Step 5: Report
    bl = reports[0] if reports and reports[0].variant == "baseline" else None
    report_path = os.path.join(output_dir, "rdna2_optimization_report.md")
    generate_report(reports, bl, report_path)

    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    json_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": [
            {
                "name": r.variant,
                "description": r.description,
                "hipfire_ref": r.hipfire_ref,
                "dense": {str(k): {"pp": v.pp_tok_s, "tg": v.tg_tok_s, "passes": v.passes}
                          for k, v in r.dense_results.items()},
                "moe": {str(k): {"pp": v.pp_tok_s, "tg": v.tg_tok_s, "passes": v.passes}
                        for k, v in r.moe_results.items()},
            }
            for r in reports
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nReports: {report_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
