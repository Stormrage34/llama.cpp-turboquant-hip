#!/usr/bin/env bash
# v0.3.1.1 — rocprofv3 baseline telemetry run
# Generates L2, LDS, ALU counters for RDNA2 IQ4_XS inference.
#
# Usage:
#   ./scripts/run_rocprof_baseline.sh <model.gguf>
#
# Output:
#   benchmarks/phase4/baseline_v0.3.1/  — CSV + JSON dumps

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf>"
    exit 1
fi

BENCH="${SCRIPT_DIR}/build/bin/llama-bench"
OUTDIR="${SCRIPT_DIR}/benchmarks/phase4/baseline_v0.3.1"
mkdir -p "$OUTDIR"

# Determine if rocprofv3 is available
ROCPROF=""
if command -v rocprofv3 &>/dev/null; then
    ROCPROF="rocprofv3"
elif [ -x /opt/rocm/bin/rocprofv3 ]; then
    ROCPROF="/opt/rocm/bin/rocprofv3"
else
    echo "rocprofv3 not found — running without counters"
fi

# Common bench args for 27B IQ4_XS @ 4k context
BENCH_ARGS=(
    -m "$MODEL"
    -c 4096 -p 512 -n 128 -b 256 -ub 256
    -ctk turbo4 -ctv turbo2
    --flash-attn on
    --no-mmap
    -r 10
)

# Run 1: RDNA2 optimized (BFE dequant + async)
echo "=== Run 1: RDNA2 OPT (RDNA2_OPT_V1=1) ==="
if [ -n "$ROCPROF" ]; then
    RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 \
        "$ROCPROF" --stats --hip-trace --kernel-trace \
            -i "${SCRIPT_DIR}/scripts/counters_inf_cache.txt" \
            --output-dir "${OUTDIR}/rdna2_opt" \
            "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/rdna2_opt/bench.log"
else
    RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 \
        "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/rdna2_opt/bench.log"
fi

# Run 2: RDNA2 baseline (no opt, standard dequant)
echo "=== Run 2: RDNA2 baseline (no RDNA2 flags) ==="
if [ -n "$ROCPROF" ]; then
    "$ROCPROF" --stats --hip-trace --kernel-trace \
        -i "${SCRIPT_DIR}/scripts/counters_inf_cache.txt" \
        --output-dir "${OUTDIR}/baseline" \
        "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/baseline/bench.log"
else
    "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/baseline/bench.log"
fi

echo "=== Done ==="
echo "Results: ${OUTDIR}/"
echo "  rdna2_opt/  — BFE dequant + async pipeline"
echo "  baseline/   — standard dequant"
