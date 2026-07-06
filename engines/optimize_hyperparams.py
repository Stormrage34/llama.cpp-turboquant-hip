#!/usr/bin/env python3
"""
Hyperparameter Optimization for llama.cpp-turboquant-hip

Uses Bayesian Optimization (bayes-opt library by wangronin) to find optimal
runtime parameters for inference throughput on RDNA 2 (gfx1030).

Wraps the existing bash benchmark scripts (run_benchmark.sh, run_rdna2_bench.sh)
as the objective function — no duplication of the execution logic.

Usage:
    # Optimize over all tunable params for a dense model
    python3 scripts/optimize_hyperparams.py --model /path/to/model.gguf

    # Optimize only cache type + batch for a MoE model
    python3 scripts/optimize_hyperparams.py --model /path/to/moe.gguf \
        --search-space cache,batch --moe

    # Run 50 evaluations with 5 initial random starts
    python3 scripts/optimize_hyperparams.py --model /path/to/model.gguf \
        --max-evals 50 --doe 5

    # Compare against current defaults
    python3 scripts/optimize_hyperparams.py --model /path/to/model.gguf \
        --baseline-first

    # Dry run: print proposed configs without running
    python3 scripts/optimize_hyperparams.py --model /path/to/model.gguf \
        --dry-run --max-evals 5
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

# ===========================================================================
# Search space definitions
# ===========================================================================
# Each hyperparameter belongs to one of three types:
#   real:    continuous, rounded at evaluation time
#   integer: discrete within bounds
#   choice:  categorical from a fixed list

SEARCH_SPACES = {
    "cache": {
        "ctk": {
            "type": "choice",
            "values": ["turbo2", "turbo3", "turbo4", "q8_0", "q4_0", "f16", "bf16"],
            "default": "q8_0",
            "description": "Key cache quantization type",
        },
        "ctv": {
            "type": "choice",
            "values": ["turbo2", "turbo3", "turbo4", "q8_0", "q4_0", "f16", "bf16"],
            "default": "turbo3",
            "description": "Value cache quantization type",
        },
    },
    "batch": {
        "batch_size": {
            "type": "integer",
            "low": 32,
            "high": 1024,
            "default": 512,
            "description": "Batch size (-b). CR-008: symmetrical with ubatch.",
        },
        "ubatch_size": {
            "type": "integer",
            "low": 32,
            "high": 1024,
            "default": 512,
            "description": "Ubatch size (-ub). Always set equal to batch.",
        },
    },
    "compute": {
        "fa": {
            "type": "choice",
            "values": [0, 1],
            "default": 1,
            "description": "Flash attention (-fa)",
        },
        "ngl": {
            "type": "integer",
            "low": 1,
            "high": 99,
            "default": 99,
            "description": "GPU layers (-ngl)",
        },
        "threads": {
            "type": "integer",
            "low": 1,
            "high": 16,
            "default": 8,
            "description": "Thread count (-t)",
        },
    },
    "moe": {
        "n_cpu_moe_start": {
            "type": "integer",
            "low": 0,
            "high": 64,
            "default": 0,
            "description": "MoE CPU offload range start (--n-cpu-moe-range)",
        },
        "n_cpu_moe_end": {
            "type": "integer",
            "low": 0,
            "high": 64,
            "default": 0,
            "description": "MoE CPU offload range end (--n-cpu-moe-range)",
        },
    },
    "prompt": {
        "n_prompt": {
            "type": "choice",
            "values": [128, 512, 2048, 4096],
            "default": 512,
            "description": "Prompt length for benchmarking",
        },
        "n_gen": {
            "type": "integer",
            "low": 32,
            "high": 512,
            "default": 128,
            "description": "Generation length for benchmarking",
        },
    },
}


def build_param_dict(search_groups: List[str], params: Dict[str, any],
                     default_overrides: Optional[Dict[str, any]] = None) -> Dict[str, any]:
    """Convert BO-proposed values to discrete runtime parameters.

    Accepts both raw floats from BO and pre-discretized values from DOE sampling.
    """
    result = {}
    for group in search_groups:
        for name, spec in SEARCH_SPACES[group].items():
            if default_overrides and name in default_overrides:
                result[name] = default_overrides[name]
                continue
            raw = params.get(name, spec["default"])
            if spec["type"] == "choice":
                # Already discretized (from DOE) or continuous (from BO)
                if isinstance(raw, str) or isinstance(raw, int):
                    result[name] = raw if isinstance(raw, str) else spec["values"][raw]
                else:
                    idx = max(0, min(len(spec["values"]) - 1, int(round(raw))))
                    result[name] = spec["values"][idx]
            elif spec["type"] == "integer":
                result[name] = int(round(max(float(spec["low"]), min(float(spec["high"]), float(raw)))))
            elif spec["type"] == "real":
                result[name] = float(max(spec["low"], min(spec["high"], raw)))
            else:
                result[name] = spec["default"]
    return result


# ===========================================================================
# Objective function: run benchmark and return decode throughput
# ===========================================================================

@dataclass
class BenchResult:
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    vram_mib: float = 0.0
    error: str = ""


def run_llama_bench(
    binary: str,
    model: str,
    params: Dict[str, any],
    prompt: str = "Write a function to compute prime numbers in Python.",
    runs: int = 2,
    timeout_s: int = 600,
) -> BenchResult:
    """Run llama-cli benchmark directly, parse output. Core objective function."""
    result = BenchResult()

    cmd = [
        binary, "-m", model,
        "-ngl", str(params.get("ngl", 99)),
        "-c", str(params.get("ctx", 32768)),
        "-b", str(params.get("batch_size", 512)),
        "-ub", str(params.get("ubatch_size", 512)),
        "-t", str(params.get("threads", 8)),
        "-ctk", str(params.get("ctk", "q8_0")),
        "-ctv", str(params.get("ctv", "turbo3")),
        "-fa", str(params.get("fa", 1)),
        "-n", str(params.get("n_gen", 128)),
        "-p", prompt,
        "--no-display-prompt",
    ]

    moe_start = params.get("n_cpu_moe_start", 0)
    moe_end = params.get("n_cpu_moe_end", 0)
    if moe_end > moe_start:
        cmd += ["--n-cpu-moe-range", f"{moe_start}-{moe_end}"]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                          timeout=timeout_s).decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        result.error = "timeout"
        return result
    except subprocess.CalledProcessError as e:
        result.error = f"exit={e.returncode}: {e.output.decode('utf-8', errors='replace')[-200:]}"
        return result
    except FileNotFoundError:
        result.error = f"binary not found: {binary}"
        return result

    # Parse: llama-cli prints "Prompt:  XX.XX t/s" and "Generation: YY.YY t/s"
    for line in output.split("\n"):
        m = re.search(r"Prompt.*?([\d.]+)\s*t/s", line)
        if m and not result.prefill_tps:
            result.prefill_tps = float(m.group(1))
        m = re.search(r"Generation.*?([\d.]+)\s*t/s", line)
        if m and not result.decode_tps:
            result.decode_tps = float(m.group(1))

    if result.decode_tps == 0 and not result.error:
        result.error = "could not parse throughput from output"

    return result


# ===========================================================================
# Bayesian Optimization wrapper
# ===========================================================================


def create_bo_search_space(search_groups: List[str],
                           params: Dict[str, float]) -> Tuple:
    """Build bayes-optim search space from configured groups.

    Returns a tuple of (space, param_order) where space is a product
    space compatible with bayes_optim.BO and param_order preserves the
    order of dimensions for mapping results back to parameters.
    """
    # Collect dimensions
    dims = []
    param_order = []
    for group in search_groups:
        for name, spec in SEARCH_SPACES[group].items():
            if name in params:
                continue  # fixed
            if spec["type"] == "choice":
                lo, hi = 0.0, float(len(spec["values"]) - 1)
            else:
                lo, hi = float(spec["low"]), float(spec["high"])
            dims.append((lo, hi))
            param_order.append((group, name))

    try:
        from bayes_optim import RealSpace
        return RealSpace(dims), param_order
    except ImportError:
        return None, param_order


def sample_doe_from_params(
    search_groups: List[str],
    fixed_params: Dict[str, float],
    n_samples: int = 5,
    seed: int = 42,
) -> List[Dict[str, any]]:
    """Generate initial design-of-experiment samples without bayes-optim."""
    import random
    rng = random.Random(seed)
    configs = []
    for _ in range(n_samples):
        pt = {}
        for g in search_groups:
            for name, spec in SEARCH_SPACES[g].items():
                if name in fixed_params:
                    pt[name] = fixed_params[name]
                elif spec["type"] == "choice":
                    pt[name] = rng.choice(spec["values"])
                elif spec["type"] == "integer":
                    pt[name] = rng.randint(spec["low"], spec["high"])
                else:
                    pt[name] = rng.uniform(spec["low"], spec["high"])
        configs.append(pt)
    return configs


def run_bo_optimization(
    search_groups: List[str],
    fixed_params: Dict[str, float],
    obj_func: Callable,
    max_evals: int = 40,
    doe_size: int = 5,
    verbose: bool = True,
) -> Tuple[List[Dict], List[float]]:
    """Run Bayesian Optimization loop.

    Falls back to random search if bayes-optim is not installed.
    """
    space, param_order = create_bo_search_space(search_groups, fixed_params)

    # ===================================================================
    # Phase 1: Design of Experiments (initial random samples)
    # ===================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 1: Design of Experiments ({doe_size} samples)")
        print(f"{'='*60}")

    doe_configs = sample_doe_from_params(search_groups, fixed_params, doe_size)
    all_configs, all_scores = [], []

    for i, cfg in enumerate(doe_configs[:doe_size]):
        concrete = build_param_dict(search_groups, cfg, fixed_params)
        if verbose:
            print(f"\n  [{i+1}/{doe_size}] {summarize_config(concrete)}")
        result = obj_func(concrete)
        score = -result.decode_tps if result.decode_tps > 0 else -1e6
        all_configs.append(concrete)
        all_scores.append(result.decode_tps)
        if verbose:
            print(f"    decode={result.decode_tps:.1f} t/s  "
                  f"prefill={result.prefill_tps:.1f} t/s" +
                  (f"  error={result.error}" if result.error else ""))

    if max_evals <= doe_size:
        return all_configs, all_scores

    # ===================================================================
    # Phase 2: Bayesian Optimization loop
    # ===================================================================
    if space is not None and param_order:
        _run_bo_bayesoptim(
            space, param_order, search_groups, fixed_params,
            obj_func, all_configs, all_scores,
            max_evals - doe_size, doe_size, verbose,
        )
    else:
        _run_bo_random(
            search_groups, fixed_params, obj_func,
            all_configs, all_scores,
            max_evals - len(all_configs), verbose,
        )

    return all_configs, all_scores


def _run_bo_bayesoptim(
    space, param_order, search_groups, fixed_params,
    obj_func, all_configs, all_scores,
    n_iter, doe_size, verbose,
):
    """Core BO loop using bayes-optim library (wangronin)."""
    try:
        from bayes_optim import BO
        from bayes_optim.Surrogate import GaussianProcess
        import numpy as np
    except ImportError as e:
        if verbose:
            print(f"\n  bayes-optim import failed: {e}")
            print("  Falling back to random search")
        return _run_bo_random(
            search_groups, fixed_params, obj_func,
            all_configs, all_scores, n_iter, verbose,
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 2: Bayesian Optimization ({n_iter} evaluations)")
        print(f"{'='*60}")

    # Prepare initial training data from DOE
    # Map DOE configs to vector space
    X_init = []
    y_init = []
    for i, cfg in enumerate(all_configs):
        vec = []
        for (group, name) in param_order:
            concrete_cfg = build_param_dict(search_groups, cfg, fixed_params)
            spec = SEARCH_SPACES[group][name]
            val = concrete_cfg.get(name, spec["default"])
            if spec["type"] == "choice":
                # Map choice to normalized index
                idx = spec["values"].index(val) if val in spec["values"] else 0
                vec.append(idx / max(1.0, len(spec["values"]) - 1))
            elif spec["type"] == "integer":
                vec.append((val - spec["low"]) / max(1.0, spec["high"] - spec["low"]))
            else:
                vec.append(max(0.0, min(1.0, (val - spec["low"]) / max(1e-9, spec["high"] - spec["low"]))))
        X_init.append(vec)
        y_init.append(-all_scores[i])  # negative = minimize latency

    X_init = np.array(X_init)
    y_init = np.array(y_init)

    # Build GP model
    dim = len(param_order)
    bounds = np.array([[0.0, 1.0]] * dim)
    thetaL = 1e-5 * np.ones(dim)
    thetaU = 10.0 * np.ones(dim)

    try:
        model = GaussianProcess(thetaL=thetaL, thetaU=thetaU)
        opt = BO(
            search_space=space,
            obj_fun=None,  # we evaluate manually
            model=model,
            DoE_size=0,    # we provide initial data
            max_FEs=n_iter,
            verbose=False,
        )
        # Manually set initial data
        opt.X = X_init
        opt.y = y_init
        opt.n_eval = len(X_init)
        opt.model.fit(opt.X, opt.y)
    except Exception as e:
        if verbose:
            print(f"  BO model init failed: {e}")
            print("  Falling back to random search")
        return _run_bo_random(
            search_groups, fixed_params, obj_func,
            all_configs, all_scores, n_iter, verbose,
        )

    # BO loop
    for it in range(n_iter):
        try:
            x_next = opt._BO__suggest()
            x_next = x_next.flatten().tolist()
        except Exception as e:
            if verbose:
                print(f"  BO suggest failed at iter {it}: {e}")
                # fallback to random
                import random
                rng = random.Random(42 + it)
                x_next = [rng.random() for _ in range(dim)]
            else:
                import random as _r
                rng = _r.Random(42 + it)
                x_next = [rng.random() for _ in range(dim)]

        # Map normalized vector back to parameters
        cfg = {}
        for j, (group, name) in enumerate(param_order):
            spec = SEARCH_SPACES[group][name]
            norm_val = max(0.0, min(1.0, x_next[j]))
            if spec["type"] == "choice":
                idx = int(round(norm_val * (len(spec["values"]) - 1)))
                idx = max(0, min(len(spec["values"]) - 1, idx))
                cfg[name] = spec["values"][idx]
            elif spec["type"] == "integer":
                val = spec["low"] + norm_val * (spec["high"] - spec["low"])
                cfg[name] = int(round(val))
            else:
                val = spec["low"] + norm_val * (spec["high"] - spec["low"])
                cfg[name] = float(val)

        concrete = build_param_dict(search_groups, cfg, fixed_params)
        if verbose:
            t0 = time.time()
            print(f"  [BO {it+1}/{n_iter}] {summarize_config(concrete)}")

        result = obj_func(concrete)

        # Record
        all_configs.append(concrete)
        all_scores.append(result.decode_tps)
        y_new = -result.decode_tps if result.decode_tps > 0 else -1e6

        if verbose:
            elapsed = time.time() - t0
            print(f"    decode={result.decode_tps:.1f} t/s  "
                  f"({elapsed:.0f}s)" +
                  (f"  error={result.error}" if result.error else ""))

        # Update model
        try:
            vec_new = np.array([[max(0.0, min(1.0, v)) for v in x_next]])
            opt.X = np.vstack([opt.X, vec_new])
            opt.y = np.hstack([opt.y, [y_new]])
            opt.n_eval += 1
            opt.model.fit(opt.X, opt.y)
        except Exception as e:
            if verbose:
                print(f"    model update skipped: {e}")
    return


def _run_bo_random(
    search_groups, fixed_params, obj_func,
    all_configs, all_scores, n_iter, verbose,
):
    """Fallback: random search when bayes-optim is unavailable."""
    import random
    rng = random.Random(42)

    for i in range(n_iter):
        pt = {}
        for g in search_groups:
            for name, spec in SEARCH_SPACES[g].items():
                if name in fixed_params:
                    pt[name] = fixed_params[name]
                elif spec["type"] == "choice":
                    pt[name] = rng.choice(spec["values"])
                elif spec["type"] == "integer":
                    pt[name] = rng.randint(spec["low"], spec["high"])
                else:
                    pt[name] = rng.uniform(spec["low"], spec["high"])

        concrete = build_param_dict(search_groups, pt, fixed_params)
        if verbose:
            t0 = time.time()
            print(f"  [RS {i+1}/{n_iter}] {summarize_config(concrete)}")

        result = obj_func(concrete)
        all_configs.append(concrete)
        all_scores.append(result.decode_tps)

        if verbose:
            elapsed = time.time() - t0
            print(f"    decode={result.decode_tps:.1f} t/s  "
                  f"({elapsed:.0f}s)" +
                  (f"  error={result.error}" if result.error else ""))


def summarize_config(cfg: Dict[str, any]) -> str:
    """Brief one-line summary of a parameter config."""
    parts = [
        f"ctk={cfg.get('ctk','?')}",
        f"ctv={cfg.get('ctv','?')}",
        f"b={cfg.get('batch_size','?')}",
        f"fa={cfg.get('fa','?')}",
    ]
    if cfg.get("n_cpu_moe_end", 0) > cfg.get("n_cpu_moe_start", 0):
        parts.append(f"moe={cfg['n_cpu_moe_start']}-{cfg['n_cpu_moe_end']}")
    return "  ".join(parts)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for llama.cpp-turboquant-hip hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--binary", default="",
                        help="llama-cli binary path (auto-detect if empty)")
    parser.add_argument("--moe", action="store_true",
                        help="Include MoE-specific parameters in search")
    parser.add_argument("--search-space", default="cache,batch,compute",
                        help="Comma-separated groups to optimize: "
                             "cache,batch,compute,prompt,moe")
    parser.add_argument("--baseline-first", action="store_true",
                        help="Always run the default config first for comparison")
    parser.add_argument("--max-evals", type=int, default=30,
                        help="Maximum objective evaluations")
    parser.add_argument("--doe", type=int, default=5,
                        help="Initial design-of-experiment samples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print proposed configs without running benchmarks")
    parser.add_argument("--output-dir", default="bench-results",
                        help="Directory for result files")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx", type=int, default=32768,
                        help="Context size for benchmark")
    parser.add_argument("--debug", action="store_true",
                        help="Print full command output on error")

    args = parser.parse_args()
    args.search_space = args.search_space.split(",")

    # Validate search groups
    for g in args.search_space:
        if g not in SEARCH_SPACES:
            print(f"error: unknown search space group '{g}'. "
                  f"Valid: {list(SEARCH_SPACES.keys())}", file=sys.stderr)
            sys.exit(1)

    if args.moe and "moe" not in args.search_space:
        args.search_space.append("moe")

    # Auto-detect binary
    binary = args.binary
    if not binary:
        candidates = [
            os.path.expanduser("~/llama.cpp/build/bin/llama-cli"),
            os.path.expanduser("~/llama.cpp-turboquant-hip/build/bin/llama-cli"),
            "./build/bin/llama-cli",
            "../build/bin/llama-cli",
        ]
        for c in candidates:
            if os.path.exists(c):
                binary = c
                break

    if not binary or not os.path.exists(binary):
        print(f"error: llama-cli not found. Specify --binary or build first.",
              file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        model_path = args.model
    else:
        model_path = os.path.abspath(args.model)
        if not os.path.exists(model_path):
            print(f"error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    print(f"{'='*60}")
    print(f"llama.cpp Hyperparameter Optimization (Bayesian)")
    print(f"{'='*60}")
    print(f"  Model:     {model_path}")
    print(f"  Binary:    {binary}")
    print(f"  Space:     {args.search_space}")
    print(f"  Max evals: {args.max_evals}")
    print(f"  DOE:       {args.doe}")
    print(f"  MoE:       {args.moe}")
    print(f"  CTX:       {args.ctx}")

    # Fixed parameters (optimization boundaries)
    fixed = {
        "ctx": args.ctx,
        "n_prompt": 512,
        "n_gen": 128,
    }
    if args.moe:
        fixed["n_cpu_moe_start"] = 0
        fixed["n_cpu_moe_end"] = 0

    # Baseline run
    if args.baseline_first:
        print(f"\n  --- Baseline (default config) ---")
        default_cfg = build_param_dict(args.search_space, {}, fixed)
        default_cfg["ctx"] = args.ctx
        print(f"    {summarize_config(default_cfg)}")
        if not args.dry_run:
            result = run_llama_bench(binary, model_path, default_cfg)
            print(f"    decode={result.decode_tps:.1f} t/s  "
                  f"prefill={result.prefill_tps:.1f} t/s")
        print()

    # Dry-run: just sample the space, no benchmarks
    if args.dry_run:
        print(f"\n  --- Dry run: sampling {args.doe} configs ---")
        for i, cfg in enumerate(
            sample_doe_from_params(args.search_space, fixed, args.doe, args.seed)
        ):
            concrete = build_param_dict(args.search_space, cfg, fixed)
            print(f"  [{i+1}/{args.doe}] {summarize_config(concrete)}")
        print("\n  Dry run complete. Pass --dry-run to actually run.")
        return

    # ===================================================================
    # Main optimization
    # ===================================================================

    def objective(params: Dict[str, any]) -> BenchResult:
        p = dict(params)
        p["ctx"] = args.ctx
        return run_llama_bench(binary, model_path, p)

    t_start = time.time()
    all_configs, all_scores = run_bo_optimization(
        search_groups=args.search_space,
        fixed_params=fixed,
        obj_func=objective,
        max_evals=args.max_evals,
        doe_size=args.doe,
        verbose=True,
    )
    elapsed = time.time() - t_start

    # ===================================================================
    # Results
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Optimization Complete ({elapsed:.0f}s, {len(all_configs)} evals)")
    print(f"{'='*60}")

    if not all_scores:
        print("  No valid results.")
        return

    # Best config
    best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
    best_cfg = all_configs[best_idx]
    best_score = all_scores[best_idx]

    print(f"\n  Best decode throughput: {best_score:.1f} t/s")
    print(f"  Best config:")
    for k, v in sorted(best_cfg.items()):
        print(f"    {k}: {v}")

    # Top 3
    top3 = sorted(zip(all_scores, all_configs), key=lambda x: -x[0])[:3]
    print(f"\n  Top 3 configurations:")
    for rank, (score, cfg) in enumerate(top3, 1):
        if score <= 0:
            continue
        print(f"  #{rank}: {summarize_config(cfg)}  decode={score:.1f} t/s")

    # Save if requested
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"hyperopt_{timestamp}.json"
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_path,
            "binary": binary,
            "search_space": args.search_space,
            "max_evals": args.max_evals,
            "doe": args.doe,
            "ctx": args.ctx,
            "total_evals": len(all_scores),
            "best_score": best_score,
            "best_config": best_cfg,
            "top3": [
                {"score": s, "config": c}
                for s, c in top3 if s > 0
            ],
            "all_scores": [s for s in all_scores],
            "all_configs": all_configs,
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  Results saved: {out_path}")

    # Return exit code
    if best_score <= 0:
        print("\n  All evaluations failed. Check model path, binary, and VRAM.")
        sys.exit(1)


if __name__ == "__main__":
    main()
