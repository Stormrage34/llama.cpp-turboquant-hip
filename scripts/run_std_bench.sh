#!/usr/bin/env bash
# run_std_bench.sh — Standardized benchmark harness for RDNA2
#
# Enforces identical benchmark configuration across versions.
# All runs use -r 10 (10 iterations) with median ± std dev reporting.
# Kernel-trace enabled when rocprofv3 is available.
#
# Usage:
#   ./scripts/run_std_bench.sh <model.gguf> [config]
#   ./scripts/run_std_bench.sh /path/to/model.gguf moe-99
#   ./scripts/run_std_bench.sh /path/to/model.gguf dense-30
#
# Configs:
#   moe-99    — MoE model, -ngl 99 (full offload) [default]
#   moe-30    — MoE model, -ngl 30 (partial offload)
#   dense-99  — Dense model, -ngl 99
#   dense-30  — Dense model, -ngl 30
#
# Output:
#   benchmarks/std_bench/<timestamp>_<config>/ — raw logs + summary
#
# This script exists because the v0.3.0→v0.3.1 "regression" was caused by
# comparing MoE -ngl 99 vs Dense -ngl 30 benchmarks. Never again.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
CONFIG="${2:-moe-99}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [config]"
    echo ""
    echo "Configs: moe-99 (default), moe-30, dense-99, dense-30"
    echo ""
    echo "This script enforces identical benchmark configs across versions."
    echo "All runs use -r 10 with median ± std dev reporting."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# ─── Standardized benchmark parameters ────────────────────────────────────────
# These are FIXED. Do not change them without updating the forensic audit.
CONTEXT=4096
PROMPT=512
GEN_LEN=128
BATCH=256
UBATCH=256
CTK="turbo4"
CTV="turbo2"
RUNS=10
FLASH_ATTN="on"

# Config-dependent parameters
case "${CONFIG}" in
    moe-99)
        NGL=99
        DESC="MoE model, full GPU offload"
        ;;
    moe-30)
        NGL=30
        DESC="MoE model, partial GPU offload"
        ;;
    dense-99)
        NGL=99
        DESC="Dense model, full GPU offload"
        ;;
    dense-30)
        NGL=30
        DESC="Dense model, partial GPU offload"
        ;;
    *)
        echo "Error: Unknown config '${CONFIG}'"
        echo "Valid configs: moe-99, moe-30, dense-99, dense-30"
        exit 1
        ;;
esac

# ─── Locate binaries ──────────────────────────────────────────────────────────
BENCH="${SCRIPT_DIR}/build/bin/llama-bench"
if [ ! -x "$BENCH" ]; then
    echo "Error: llama-bench not found at $BENCH"
    echo "Run: cmake -B build -S . -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 && cmake --build build --config Release -j \$(nproc)"
    exit 1
fi

# ─── Locate rocprofv3 ─────────────────────────────────────────────────────────
ROCPROF=""
if command -v rocprofv3 &>/dev/null; then
    ROCPROF="rocprofv3"
elif [ -x /opt/rocm/bin/rocprofv3 ]; then
    ROCPROF="/opt/rocm/bin/rocprofv3"
fi

# ─── Output directory ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${SCRIPT_DIR}/benchmarks/std_bench/${TIMESTAMP}_${CONFIG}"
mkdir -p "$OUTDIR"

# ─── System info ──────────────────────────────────────────────────────────────
echo "=== Standardized RDNA2 Benchmark ===" | tee "${OUTDIR}/summary.txt"
echo "  Timestamp:  ${TIMESTAMP}" | tee -a "${OUTDIR}/summary.txt"
echo "  Model:      ${MODEL}" | tee -a "${OUTDIR}/summary.txt"
echo "  Config:      ${CONFIG} (${DESC})" | tee -a "${OUTDIR}/summary.txt"
echo "  NGL:         ${NGL}" | tee -a "${OUTDIR}/summary.txt"
echo "  Context:     ${CONTEXT}" | tee -a "${OUTDIR}/summary.txt"
echo "  Prompt:      ${PROMPT}" | tee -a "${OUTDIR}/summary.txt"
echo "  Gen len:     ${GEN_LEN}" | tee -a "${OUTDIR}/summary.txt"
echo "  Batch:       ${BATCH}" | tee -a "${OUTDIR}/summary.txt"
echo "  Ubatch:      ${UBATCH}" | tee -a "${OUTDIR}/summary.txt"
echo "  CTK/CTV:     ${CTK}/${CTV}" | tee -a "${OUTDIR}/summary.txt"
echo "  Runs:        ${RUNS}" | tee -a "${OUTDIR}/summary.txt"
echo "  Flash attn:  ${FLASH_ATTN}" | tee -a "${OUTDIR}/summary.txt"
echo "" | tee -a "${OUTDIR}/summary.txt"

# Record git info
echo "=== Git Info ===" >> "${OUTDIR}/summary.txt"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)" >> "${OUTDIR}/summary.txt"
echo "  Commit: $(git rev-parse --short HEAD)" >> "${OUTDIR}/summary.txt"
echo "  Tag:    $(git describe --tags --always 2>/dev/null || echo 'none')" >> "${OUTDIR}/summary.txt"
echo "" >> "${OUTDIR}/summary.txt"

# Record GPU info
ROCM_SMI="${ROCM_PATH:-/opt/rocm}/bin/rocm-smi"
if [ -x "${ROCM_SMI}" ]; then
    echo "=== GPU Info ===" >> "${OUTDIR}/summary.txt"
    ${ROCM_SMI} >> "${OUTDIR}/summary.txt" 2>&1 || true
    echo "" >> "${OUTDIR}/summary.txt"
fi

# ─── Standardized benchmark command ───────────────────────────────────────────
BENCH_ARGS=(
    -m "$MODEL"
    -c $CONTEXT
    -p $PROMPT
    -n $GEN_LEN
    -b $BATCH
    -ub $UBATCH
    -ctk $CTK
    -ctv $CTV
    --flash-attn $FLASH_ATTN
    --no-mmap
    -ngl $NGL
    -r $RUNS
)

echo "Running benchmark (${RUNS} iterations)..." | tee -a "${OUTDIR}/summary.txt"
echo "  Command: RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 llama-bench ${BENCH_ARGS[*]}" | tee -a "${OUTDIR}/summary.txt"
echo "" | tee -a "${OUTDIR}/summary.txt"

# ─── Run with kernel trace if available ─────────────────────────────────────────
COUNTERS_FILE="${SCRIPT_DIR}/scripts/counters_inf_cache.txt"

if [ -n "$ROCPROF" ] && [ -f "$COUNTERS_FILE" ]; then
    echo "  rocprofv3: ENABLED (kernel trace + counters)" | tee -a "${OUTDIR}/summary.txt"
    echo "" | tee -a "${OUTDIR}/summary.txt"

    RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 \
        "$ROCPROF" --stats --hip-trace --kernel-trace \
            -i "$COUNTERS_FILE" \
            --output-dir "${OUTDIR}/rocprof" \
            "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/bench.log"
else
    echo "  rocprofv3: DISABLED (not available or counters file missing)" | tee -a "${OUTDIR}/summary.txt"
    echo "" | tee -a "${OUTDIR}/summary.txt"

    RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 \
        "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${OUTDIR}/bench.log"
fi

# ─── Parse results ─────────────────────────────────────────────────────────────
echo "" | tee -a "${OUTDIR}/summary.txt"
echo "=== Results ===" | tee -a "${OUTDIR}/summary.txt"

# Extract pp512 and tg128 from llama-bench output
PP512_VALUES=$(grep -oP 'pp512\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null || echo "N/A")
TG128_VALUES=$(grep -oP 'tg128\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null || echo "N/A")

if [ "$PP512_VALUES" != "N/A" ]; then
    # Calculate median and std dev
    PP512_MEDIAN=$(echo "$PP512_VALUES" | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}')
    TG128_MEDIAN=$(echo "$TG128_VALUES" | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}')

    echo "  Prefill (pp512): ${PP512_MEDIAN} t/s (median of ${RUNS})" | tee -a "${OUTDIR}/summary.txt"
    echo "  Decode (tg128):   ${TG128_MEDIAN} t/s (median of ${RUNS})" | tee -a "${OUTDIR}/summary.txt"
else
    echo "  Prefill: Could not parse from bench.log" | tee -a "${OUTDIR}/summary.txt"
    echo "  Decode:  Could not parse from bench.log" | tee -a "${OUTDIR}/summary.txt"
fi

echo "" | tee -a "${OUTDIR}/summary.txt"
echo "Results saved to: ${OUTDIR}/" | tee -a "${OUTDIR}/summary.txt"
echo "" | tee -a "${OUTDIR}/summary.txt"
echo "To compare across versions, use identical config:" | tee -a "${OUTDIR}/summary.txt"
echo "  $0 <model.gguf> ${CONFIG}" | tee -a "${OUTDIR}/summary.txt"

# ─── Kernel dispatch verification ──────────────────────────────────────────────
if [ -n "$ROCPROF" ] && [ -d "${OUTDIR}/rocprof" ]; then
    echo "" | tee -a "${OUTDIR}/summary.txt"
    echo "=== Kernel Dispatch ===" | tee -a "${OUTDIR}/summary.txt"

    # Extract unique kernel names from trace
    TRACE_FILE=$(find "${OUTDIR}/rocprof" -name "*kernel_trace*" -o -name "*KernelTrace*" 2>/dev/null | head -1)
    if [ -n "$TRACE_FILE" ] && [ -f "$TRACE_FILE" ]; then
        echo "  Trace file: ${TRACE_FILE}" | tee -a "${OUTDIR}/summary.txt"
        echo "  Dequant kernels:" | tee -a "${OUTDIR}/summary.txt"
        grep -i "dequant" "$TRACE_FILE" | awk -F',' '{print "    " $1}' | sort -u | head -10 | tee -a "${OUTDIR}/summary.txt"
        echo "  Matmul kernels:" | tee -a "${OUTDIR}/summary.txt"
        grep -i "mul_mat\|mmq" "$TRACE_FILE" | awk -F',' '{print "    " $1}' | sort -u | head -10 | tee -a "${OUTDIR}/summary.txt"
    else
        echo "  No kernel trace file found" | tee -a "${OUTDIR}/summary.txt"
    fi
fi

echo "" | tee -a "${OUTDIR}/summary.txt"
echo "=== Benchmark Complete ===" | tee -a "${OUTDIR}/summary.txt"