#!/usr/bin/env python3
"""
llama.cpp RDNA 2 Master Diagnostic Script

Unifies the three simulation engines (occupancy solver, memory simulator,
telemetry bridge) with the quantization validation and long-context pipelines
into a single entry point. Subsumes mlnn_v40_runner.py and master_debug_turbo.py.

Usage:
    # Quick diagnostic (no GPU/model needed)
    python3 engines/run_diagnostics.py --mode quick

    # Turbo3 quantization validation (12-test suite)
    python3 engines/run_diagnostics.py --mode turbo-validate

    # Long-context attention simulation
    python3 engines/run_diagnostics.py --mode long-context

    # Compare against upstream reference
    python3 engines/run_diagnostics.py --mode compare-upstream

    # Run everything
    python3 engines/run_diagnostics.py --mode all

    # With rocprofv3 telemetry
    python3 engines/run_diagnostics.py --mode all --profile-dir rocprof-output
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from _roctx import mark, roctx as _roctx

from rdna2_occupancy_solver import RDNA2OccupancySolver
from rdna2_memory_simulator import RDNA2MemoryHardwareSimulator
from compiler_telemetry_bridge import CompilerTelemetryBridge


class DiagnosticReport:
    """Container for structured diagnostic output across all modes."""

    def __init__(self, mode: str):
        self.mode = mode
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.results: Dict[str, Any] = {
            "mode": mode,
            "timestamp": self.timestamp,
            "status": "running",
        }

    def add_block(self, name: str, data: dict) -> None:
        self.results[name] = data

    def set_status(self, status: str) -> None:
        self.results["status"] = status

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.results, indent=indent, default=str)

    def print_summary(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"DIAGNOSTIC REPORT — mode={self.mode}")
        print(f"{'=' * 70}")
        for key, value in self.results.items():
            if key in ("mode", "timestamp", "status"):
                continue
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        print(f"    {k}:")
                        for kk, vv in v.items():
                            print(f"      {kk}: {vv}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print(f"\n  Status: {self.results.get('status', 'unknown')}")
        print(f"{'=' * 70}\n")


def run_quick_diagnostic(report: DiagnosticReport) -> DiagnosticReport:
    """Quick diagnostic — no GPU or model required (subsumes mlnn_v40_runner.py)."""
    print("\n[Quick Diagnostic] Analyzing RDNA 2 hardware configuration...\n")

    solver = RDNA2OccupancySolver()
    simulator = RDNA2MemoryHardwareSimulator()
    telemetry = CompilerTelemetryBridge()

    # 1. Flash attention kernel occupancy
    with mark("quick_fattn"):
        print("[1/4] Flash Attention kernel occupancy...")
        fattn = solver.analyze_flash_attention_kernels()
        occ_block = {}
        if fattn.get("head_sizes"):
            for key, cfg in list(fattn["head_sizes"].items())[:5]:
                occ_block[key] = {
                    "occupancy_pct": cfg.get("calculated_occupancy_pct", 0),
                    "limiting_factor": cfg.get("limiting_factor", "unknown"),
                }
            occ_block["hardware"] = {
                "simds_per_cu": solver.SIMDS_PER_CU,
                "max_waves_per_cu": solver.MAX_WAVES_PER_CU,
                "lds_capacity_bytes": solver.LDS_CAPACITY_BYTES,
            }
        report.add_block("occupancy_analysis", occ_block)

    # 2. Memory working set (262K context, D=128, 3.5-bit)
    with mark("quick_memory"):
        print("[2/4] Memory working set (262K context)...")
        memory = simulator.analyze_attention_working_set(
            head_dim=128, context_size=262144, quantization_bits=3.5
        )
        report.add_block("memory_working_set", memory)

    # 3. VRAM throughput
    with mark("quick_throughput"):
        print("[3/4] VRAM throughput estimate...")
        throughput = simulator.calculate_real_vram_throughput(
            head_dim=128, quantization_bits=3.5, token_throughput=666.0
        )
        report.add_block("vram_throughput", throughput)

    # 4. Simulation self-check
    with mark("quick_validation"):
        print("[4/4] Simulation self-check...")
        validation = telemetry.validate_simulation_against_telemetry(
            simulated_occupancy=100.0,
            measured_occupancy=12.5,
            tolerance_pct=5.0,
        )
        report.add_block("simulation_self_check", validation)

    return report


def run_turbo_validate(report: DiagnosticReport) -> DiagnosticReport:
    """Run the 12-test turbo3 quantization validation suite.

    Imports the full test suite from master_debug_turbo.py and runs all tests,
    collecting pass/fail results into the diagnostic report.
    """
    print("\n[Turbo Validate] Running turbo3 quantization validation suite...\n")

    try:
        # Import the full module — it has its own test functions
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "master_debug_turbo",
            Path(__file__).parent / "master_debug_turbo.py",
        )
        _mdt = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mdt)

        # Re-export the important symbols locally to match the module's expectations
        test_names = [
            ("Block Structure Integrity", _mdt.test_block_structure),
            ("WHT Rotation Roundtrip", _mdt.test_rotation_mismatch),
            ("Quant Pipeline Consistency", _mdt.test_cpu_vs_gpu_quant),
            ("Attention Pipeline Quality", _mdt.test_attention_collapse),
            ("Norm Blowup Detection", _mdt.test_norm_blowup),
            ("InnerQ Calibration Interference", _mdt.test_innerq_interference),
            ("Centroid Distribution Analysis", _mdt.test_centroid_distribution),
            ("Full Pipeline Simulation", _mdt.test_full_pipeline_simulation),
            ("Per-centroid Error Analysis", _mdt.test_per_centroid_errors),
            ("Structured KV Pattern Test", _mdt.test_structured_kv_patterns),
            ("Wrong-Pipeline Detection", _mdt.test_wrong_pipeline_detection),
            ("Precision Audit", _mdt.precision_audit),
        ]

        # Override global D and QK_TURBO3 from the module
        D = getattr(_mdt, "D", 128)
        QK_TURBO3 = getattr(_mdt, "QK_TURBO3", 32)

    except ImportError as e:
        report.add_block("turbo_tests", {
            "error": f"Could not load master_debug_turbo.py: {e}",
        })
        return report

    results = []
    all_pass = True
    for name, func in test_names:
        with mark(f"debug_{name.lower().replace(' ', '_')}"):
            try:
                passed = func()
                if not passed:
                    all_pass = False
                results.append({"test": name, "passed": passed})
                status = "PASS" if passed else "FAIL"
                print(f"  {name:<50} {status:<8}")
            except Exception as e:
                all_pass = False
                results.append({"test": name, "passed": False, "error": str(e)})
                print(f"  {name:<50} FAIL (exception: {e})")

    report.add_block("turbo_tests", {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "all_pass": all_pass,
        "results": results,
    })

    return report


def run_long_context(report: DiagnosticReport) -> DiagnosticReport:
    """Run long-context attention simulation from mlnn.py's pipeline."""
    print("\n[Long Context] Running attention simulation pipeline...\n")

    try:
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "mlnn",
            Path(__file__).parent / "mlnn.py",
        )
        _mlnn = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mlnn)
    except ImportError as e:
        report.add_block("long_context", {"error": f"Could not load mlnn.py: {e}"})
        return report

    # Run the long-context simulation with default parameters
    with mark("lc_simulation"):
        try:
            d = getattr(_mlnn, "D", 128)
            context_size = getattr(_mlnn, "DEFAULT_CONTEXT_SIZE", 16384)
            n_iterations = 1
            model = _mlnn.ModelConfig(head_dim=d, n_heads=32, n_kv_heads=8, n_layers=60)

            # Simulate upstream behavior (baseline, no turbo)
            print(f"  Simulating upstream (FP16) attention for D={d}...")
            upstream = _mlnn.simulate_upstream_behavior(
                context_size, n_iterations, model
            )
            report.add_block("upstream_attention", {
                "max_logit_diff": upstream.get("max_logit_diff", "N/A"),
                "cosine_similarity": upstream.get("cosine_similarity", "N/A"),
                "outlier_ratio": upstream.get("outlier_ratio", "N/A"),
            })

            # Simulate long-context with turbo quant
            print(f"  Simulating long-context attention with turbo quant...")
            lc = _mlnn.simulate_long_context_attention(
                context_size, n_iterations, model
            )
            report.add_block("long_context_attention", {
                "context_size": lc.get("context_size", "N/A"),
                "head_dim": lc.get("head_dim", d),
                "blocks": lc.get("blocks", []),
            })

        except AttributeError as e:
            report.add_block("long_context", {
                "error": f"mlnn.py simulation functions unavailable: {e}",
            })

        except Exception as e:
            report.add_block("long_context", {
                "error": f"mlnn.py simulation failed: {e}",
            })

    return report


def run_compare_upstream(report: DiagnosticReport) -> DiagnosticReport:
    """Run both turbo and upstream reference, compare results."""
    print("\n[Compare Upstream] Running turbo vs upstream comparison...\n")

    try:
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "mlnn",
            Path(__file__).parent / "mlnn.py",
        )
        _mlnn = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mlnn)
    except ImportError as e:
        report.add_block("comparison", {"error": f"Could not load mlnn.py: {e}"})
        return report

    try:
        with mark("compare_quant"):
            d = getattr(_mlnn, "D", 128)
            model = _mlnn.ModelConfig(head_dim=d, n_heads=32, n_kv_heads=8, n_layers=60)

            context_size = getattr(_mlnn, "DEFAULT_CONTEXT_SIZE", 16384)
            n_iterations = 1

            print(f"  [1/2] Turbo quant comparison (D={d})...")
            quant_results = _mlnn.simulate_all_quant_comparison(context_size, n_iterations, model)
            report.add_block("quant_comparison", quant_results)

            print(f"  [2/2] RDNA 2 memory access simulation...")
            mem = _mlnn.simulate_rdna2_memory_access(
                bytes_accessed=context_size * d,
                data_size_bytes=d
            )
            report.add_block("memory_access_simulation", mem)
    except Exception as e:
        report.add_block("comparison_error", {"message": str(e)})

    return report


def run_all(report: DiagnosticReport, profile_dir: Optional[str] = None) -> DiagnosticReport:
    """Run every diagnostic mode, optionally with telemetry."""
    report = run_quick_diagnostic(report)
    report = run_turbo_validate(report)
    report = run_long_context(report)
    report = run_compare_upstream(report)

    if profile_dir:
        try:
            telemetry = CompilerTelemetryBridge()
            trace_file = Path(profile_dir) / "kernel_dispatch.csv"
            if trace_file.exists():
                print(f"\n[Telemetry] Parsing {trace_file}...")
                data = telemetry.parse_rocprofv3_telemetry(str(trace_file))
                report.add_block("telemetry", data)
        except Exception as e:
            report.add_block("telemetry", {"error": str(e)})

    return report


def save_report(report: DiagnosticReport) -> Path:
    """Save report to bench-results/ with timestamp."""
    out_dir = Path(__file__).parent.parent / "bench-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"diagnostic_{timestamp}.json"
    with open(out_path, "w") as f:
        f.write(report.to_json())
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp RDNA 2 Master Diagnostic Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["quick", "turbo-validate", "long-context",
                           "compare-upstream", "all"],
        default="quick",
        help="Diagnostic mode (default: quick)",
    )
    parser.add_argument(
        "--profile-dir", type=str,
        help="Path to rocprofv3 output directory with kernel_dispatch.csv",
    )
    parser.add_argument(
        "--save-json", action="store_true", default=True,
        help="Save results to bench-results/ (default: true)",
    )

    args = parser.parse_args()

    report = DiagnosticReport(mode=args.mode)

    try:
        with mark(f"run_{args.mode}"):
            if args.mode == "quick":
                report = run_quick_diagnostic(report)
            elif args.mode == "turbo-validate":
                report = run_turbo_validate(report)
            elif args.mode == "long-context":
                report = run_long_context(report)
            elif args.mode == "compare-upstream":
                report = run_compare_upstream(report)
            elif args.mode == "all":
                report = run_all(report, profile_dir=args.profile_dir)

        report.set_status("completed")
        report.print_summary()

        if args.save_json:
            out_path = save_report(report)
            print(f"Results saved to: {out_path}")

        # Return exit code based on test pass/fail for CI
        turbo_tests = report.results.get("turbo_tests", {})
        if turbo_tests.get("failed", 0) > 0:
            return 1
        return 0

    except Exception as e:
        report.set_status("failed")
        report.add_block("error", {"message": str(e)})
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
