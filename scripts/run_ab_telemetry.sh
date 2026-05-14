#!/usr/bin/env bash
# run_ab_telemetry.sh — A/B telemetry harness for RDNA2 kernel optimizations
#
# Runs same-build A/B comparison with only the target flag toggled.
# This eliminates the cross-build counter noise that invalidated the P3+DPP
# results (where SQ_INSTS_VALU showed +165% from build-state differences,
# not from the optimization).
#
# Methodology:
#   1. Build ONCE with both paths compiled (flag-gated)
#   2. Run A (flag OFF) with rocprofv3 counters
#   3. Run B (flag ON) with rocprofv3 counters
#   4. Compare kernel-filtered counter deltas
#   5. Report median ± std dev across N runs
#
# Usage:
#   ./scripts/run_ab_telemetry.sh <model.gguf> <flag_name> [runs]
#
# Examples:
#   ./scripts/run_ab_telemetry.sh model.gguf RDNA2_BFE_DISPATCHER 5
#   ./scripts/run_ab_telemetry.sh model.gguf RDNA2_OPT_V1 3
#
# Output:
#   benchmarks/ab_telemetry/<flag_name>/ — CSV + summary

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
FLAG_NAME="${2:-}"
NUM_RUNS="${3:-3}"

# ─── Validate args ────────────────────────────────────────────────────────────
if [ -z "$MODEL" ] || [ -z "$FLAG_NAME" ]; then
    echo "Usage: $0 <model.gguf> <flag_name> [runs]"
    echo ""
    echo "Supported flags:"
    echo "  RDNA2_OPT_V1          — BFE dequant kernel"
    echo "  RDNA2_BFE_DISPATCHER  — BFE v_bfe_u32 for Q4_K_M/Q5_K_M"
    echo "  RDNA2_ASYNC_PIPELINE  — Async HIP pipeline"
    echo "  RDNA2_MATMUL_OPT_V1  — LDS double-buffer matmul for MoE"
    echo ""
    echo "The flag must be compiled into the binary (check CMakeLists.txt)."
    echo "This script toggles the RUNTIME env var only."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# ─── Locate binaries ──────────────────────────────────────────────────────────
BENCH="${SCRIPT_DIR}/build/bin/llama-bench"
if [ ! -x "$BENCH" ]; then
    echo "Error: llama-bench not found at $BENCH"
    echo "Run: cmake -B build -S . -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 && cmake --build build --config Release -j \$(nproc)"
    exit 1
fi

ROCPROF=""
if command -v rocprofv3 &>/dev/null; then
    ROCPROF="rocprofv3"
elif [ -x /opt/rocm/bin/rocprofv3 ]; then
    ROCPROF="/opt/rocm/bin/rocprofv3"
else
    echo "WARNING: rocprofv3 not found — running without counters"
    echo "  Counter comparison will not be available."
fi

COUNTERS_FILE="${SCRIPT_DIR}/scripts/counters_inf_cache.txt"
if [ ! -f "$COUNTERS_FILE" ]; then
    echo "WARNING: counters_inf_cache.txt not found — running without counters"
    ROCPROF=""
fi

# ─── Output directory ─────────────────────────────────────────────────────────
OUTDIR="${SCRIPT_DIR}/benchmarks/ab_telemetry/${FLAG_NAME}"
mkdir -p "${OUTDIR}/A_off" "${OUTDIR}/B_on"

# ─── Benchmark parameters ─────────────────────────────────────────────────────
# Use consistent parameters for reproducibility
CONTEXT=4096
PROMPT=512
GEN_LEN=128
BATCH=256
UBATCH=128
RUNS_PER_VARIANT="$NUM_RUNS"

BENCH_ARGS=(
    -m "$MODEL"
    -c $CONTEXT -p $PROMPT -n $GEN_LEN -b $BATCH -ub $UBATCH
    --flash-attn on
    --no-mmap
    -r 1
)

# ─── Helper: run one variant ─────────────────────────────────────────────────
run_variant() {
    local label="$1"
    local env_flag="$2"
    local outdir="$3"
    local extra_env="$4"

    echo ""
    echo "=== Running variant: $label ==="
    echo "  Flag: ${FLAG_NAME}=${env_flag}"
    echo "  Runs: $RUNS_PER_VARIANT"
    echo ""

    local all_pp512=()
    local all_tg128=()

    for run in $(seq 1 "$RUNS_PER_VARIANT"); do
        echo "  Run $run/$RUNS_PER_VARIANT..."

        local run_outdir="${outdir}/run_${run}"
        mkdir -p "$run_outdir"

        # Build env vars: always enable base RDNA2 flags, toggle target flag
        local env_vars="RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ${FLAG_NAME}=${env_flag}"
        if [ -n "$extra_env" ]; then
            env_vars="${env_vars} ${extra_env}"
        fi

        if [ -n "$ROCPROF" ]; then
            eval "${env_vars}" \
                "$ROCPROF" --stats --hip-trace --kernel-trace \
                    -i "$COUNTERS_FILE" \
                    -d "$run_outdir" \
                    "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${run_outdir}/bench.log" || true
        else
            eval "${env_vars}" \
                "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "${run_outdir}/bench.log" || true
        fi

        # Parse results
        local pp512 tg128
        pp512=$(grep -oP 'pp512\s+\K[\d.]+' "${run_outdir}/bench.log" 2>/dev/null || \
                grep -E "prompt.*tokens" "${run_outdir}/bench.log" | tail -1 | grep -oP '[\d.]+(?=\s*t/s)' || echo "N/A")
        tg128=$(grep -oP 'tg128\s+\K[\d.]+' "${run_outdir}/bench.log" 2>/dev/null || \
                grep -E "gen.*tokens" "${run_outdir}/bench.log" | tail -1 | grep -oP '[\d.]+(?=\s*t/s)' || echo "N/A")

        all_pp512+=("$pp512")
        all_tg128+=("$tg128")

        sleep 2  # Cool down between runs
    done

    # Compute median and std dev
    echo ""
    echo "=== $label Results ==="
    echo "  pp512: ${all_pp512[*]}"
    echo "  tg128: ${all_tg128[*]}"

    # Save raw values for post-processing
    printf '%s\n' "${all_pp512[@]}" > "${outdir}/pp512_values.txt"
    printf '%s\n' "${all_tg128[@]}" > "${outdir}/tg128_values.txt"
}

# ─── Run A: Flag OFF ──────────────────────────────────────────────────────────
# For compile-time flags that default to OFF, just don't set them.
# For flags that default to ON, set to 0.
run_variant "A (flag OFF)" "0" "${OUTDIR}/A_off" ""

# ─── Run B: Flag ON ──────────────────────────────────────────────────────────
run_variant "B (flag ON)" "1" "${OUTDIR}/B_on" ""

# ─── Compare results ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          A/B Telemetry Comparison                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Flag: $FLAG_NAME"
echo "Runs per variant: $RUNS_PER_VARIANT"
echo ""

# Simple comparison (Python would be better for stats, but keep it portable)
echo "A (OFF) values:"
cat "${OUTDIR}/A_off/pp512_values.txt" 2>/dev/null || echo "  N/A"
echo "B (ON) values:"
cat "${OUTDIR}/B_on/pp512_values.txt" 2>/dev/null || echo "  N/A"
echo ""

# ─── Counter comparison ───────────────────────────────────────────────────────
if [ -n "$ROCPROF" ]; then
    echo "=== Counter Comparison ==="
    echo "A (OFF) counters: ${OUTDIR}/A_off/"
    echo "B (ON)  counters: ${OUTDIR}/B_on/"
    echo ""
    echo "To compare specific counters, use:"
    echo "  python3 -c \""
    echo "    import csv, statistics"
    echo "    # Parse and compare counter CSVs"
    echo "    # Report median delta, CV%"
    echo "  \""
    echo ""
    echo "Gate criteria:"
    echo "  SQ_INSTS_VALU ↓ ≥10% (kernel-filtered)"
    echo "  LDSBankConflict ≤3%"
    echo "  GL2C_HIT ≥60% for target tiles"
    echo "  Variance ≤±1.5 t/s across runs"
fi

echo ""
echo "Results saved to: $OUTDIR"
echo ""
echo "Next steps:"
echo "  1. Verify kernel dispatch: ./scripts/verify_kernel_dispatch.sh $MODEL"
echo "  2. Check counter CSVs for kernel-filtered deltas"
echo "  3. If gates pass, promote flag to ON-by-default"