# Benchmark Data Structure

> Raw telemetry and benchmark data for the RDNA2 optimization project.
> All claims must link to raw data. No undocumented features.

## Directory Layout

```
benchmarks/
├── forensic_audit/          # Version comparison evidence
│   ├── FINAL_REPORT.md      # v0.1→v0.3.0→v0.3.1 regression analysis
│   └── diffs/              # (pending) Git diffs between versions
├── std_bench/              # Standardized benchmark runs
│   └── <timestamp>_<config>/  # e.g., 20260514_moe-99/
│       ├── bench.log        # Raw llama-bench output
│       ├── summary.txt      # Parsed summary
│       └── rocprof/         # (optional) rocprofv3 counter data
├── ab_telemetry/            # A/B comparison results
│   └── <flag_name>/         # e.g., RDNA2_BFE_DISPATCHER/
│       ├── A_off/           # Flag disabled runs
│       └── B_on/            # Flag enabled runs
└── README.md                # This file
```

## How to Add a New Benchmark

1. **Run with `-r 10`** and capture stdout:
   ```bash
   ./scripts/run_std_bench.sh <model.gguf> moe-99
   ```

2. **Parse to JSON** (optional):
   ```bash
   python3 scripts/parse_bench.py benchmarks/std_bench/<dir>/bench.log > raw/<name>.json
   ```

3. **Commit JSON + config footnote**: Include `-ngl`, model, quant, context, flags

4. **Update `docs/BENCHMARKS.md`** with summary table

## Telemetry Runs

- Always use `rocprofv3 --kernel-trace` for kernel verification
- Use `--dispatch-filter <kernel_name>` for kernel-isolated metrics
- Commit `results.csv` + `trace.hip` to `benchmarks/ab_telemetry/<experiment>/`
- Include `rocm-smi` pre/post VRAM logs

## Validation Gates

Before attributing counter deltas to any optimization:

| Gate | Requirement | Tool |
|------|------------|------|
| Kernel invoked | Target kernel appears in trace | `verify_kernel_dispatch.sh` |
| A/B comparison | Same build, only flag toggled | `run_ab_telemetry.sh` |
| Variance | ≥3 runs, median ± std dev | `run_std_bench.sh -r 10` |
| Parity | Zero NaN/garbage at `temp=0.0` | Manual token comparison |

## Naming Convention

- `forensic_audit/` — Cross-version comparisons
- `std_bench/YYYYMMDD_HHMMSS_<config>/` — Standardized runs
- `ab_telemetry/<FLAG_NAME>/` — A/B flag comparisons
- `kernel_dispatch/` — Kernel verification results