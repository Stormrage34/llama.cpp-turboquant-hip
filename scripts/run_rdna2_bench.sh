#!/bin/bash
# RDNA2 Benchmark Harness for llama.cpp-turboquant-hip
#
# Runs end-to-end inference benchmarks across context lengths,
# captures performance metrics, VRAM usage, and compares against baseline.
#
# Usage:
#   ./scripts/run_rdna2_bench.sh                    # Run with RDNA2 optimizations
#   ./scripts/run_rdna2_bench.sh baseline           # Run without optimizations
#   ./scripts/run_rdna2_bench.sh <model.gguf>       # Run with custom model
#   ./scripts/run_rdna2_bench.sh <model.gguf> baseline

set -e

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/build/bin"
LLAMA_BENCH="${BIN_DIR}/llama-bench-rdna2"

# Default model — override via argument or MODEL_PATH env var
MODEL_PATH="${MODEL_PATH:-}"

# ROCm SMI for VRAM monitoring (optional)
ROCM_SMI="${ROCM_PATH:-/opt/rocm}/bin/rocm-smi"
[ -x "${ROCM_SMI}" ] || ROCM_SMI=""

# VRAM hard limit (bytes) — 13.5 GB
VRAM_LIMIT=$((13500 * 1024 * 1024))

# Context lengths to test
PROMPT_SIZES=(512 2048 4096)

# Benchmark parameters
GEN_LEN=128
BATCH_SIZE=256
UBATCH_SIZE=128
CTK="turbo4"
CTV="turbo2"
FA="1"
THREADS=8
NGPL=99
FIT_TARGET=2048
FITC=4096
RUNS=3

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Parse Arguments ─────────────────────────────────────────────────────────
MODE="optimized"

# Check for model path argument
if [ -n "${1:-}" ]; then
    if [ "${1}" = "baseline" ]; then
        MODE="baseline"
    elif [ -f "${1}" ]; then
        MODEL_PATH="${1}"
        # Check second arg for mode
        if [ -n "${2:-}" ] && [ "${2}" = "baseline" ]; then
            MODE="baseline"
        fi
    else
        echo -e "${RED}Model not found: ${1}${NC}"
        exit 1
    fi
fi

# ─── Header ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   llama.cpp-turboquant-hip — RDNA2 Benchmark       ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Model Selection ─────────────────────────────────────────────────────────
if [ -z "${MODEL_PATH}" ]; then
    # Try common model locations
    for candidate in \
        "${PROJECT_ROOT}/models/"*.gguf \
        "$HOME/models/"*.gguf \
        "/home/stormrage/models/Qwen3_35BMTPIQ4.gguf"; do
        if [ -f "${candidate}" ]; then
            MODEL_PATH="${candidate}"
            break
        fi
    done
fi

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}✗ No model found. Specify path:${NC}"
    echo "  $0 /path/to/model.gguf"
    echo ""
    echo "  Or set MODEL_PATH environment variable."
    exit 1
fi

echo -e "${BOLD}Model:${NC} ${MODEL_PATH}"
echo -e "${BOLD}Mode:${NC}  ${MODE}"
echo ""

# ─── Prerequisites ───────────────────────────────────────────────────────────
if [ ! -x "${LLAMA_BENCH}" ]; then
    echo -e "${RED}✗ llama-bench-rdna2 not found.${NC}"
    echo "  Run: ./scripts/build_rdna2_llama.sh"
    exit 1
fi

# ─── Pre-flight VRAM Check ───────────────────────────────────────────────────
if [ -n "${ROCM_SMI}" ]; then
    echo -e "${CYAN}=== Pre-flight VRAM Check ===${NC}"
    VRAM_BEFORE=$(${ROCM_SMI} --showmeminfo 0 2>/dev/null | grep -i "gpu memory" | awk '{print $NF}' || echo "0")
    echo "  VRAM before: ${VRAM_BEFORE} MB"
    echo ""
fi

# ─── Results Storage ─────────────────────────────────────────────────────────
declare -A RESULTS_PREFILL
declare -A RESULTS_DECODE
declare -A RESULTS_VRAM

# ─── Run Benchmarks ──────────────────────────────────────────────────────────
for prompt_len in "${PROMPT_SIZES[@]}"; do
    echo -e "${CYAN}=== Context: ${prompt_len} tokens ===${NC}"

    # Set environment variables
    if [ "${MODE}" = "optimized" ]; then
        export RDNA2_OPT_V1=1
        export RDNA2_ASYNC_PIPELINE=1
        export RDNA2_MATMUL_OPT_V1=1
        echo "  Env: RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1"
    else
        unset RDNA2_OPT_V1
        unset RDNA2_ASYNC_PIPELINE
        unset RDNA2_MATMUL_OPT_V1
        echo "  Env: baseline (no RDNA2 optimizations)"
    fi

    echo "  Running (${RUNS} iterations)..."
    BENCH_OUTPUT=$( ${LLAMA_BENCH} \
        -m "${MODEL_PATH}" \
        -p ${prompt_len} \
        -n ${GEN_LEN} \
        -b ${BATCH_SIZE} \
        -ub ${UBATCH_SIZE} \
        -ctk ${CTK} \
        -ctv ${CTV} \
        -fa ${FA} \
        -mmp 0 \
        -t ${THREADS} \
        -ngl ${NGPL} \
        --fit-target ${FIT_TARGET} \
        -fitc ${FITC} \
        -r ${RUNS} \
        -o md \
        2>&1 ) || true

    # Parse results
    PREFILL_TPS=$(echo "${BENCH_OUTPUT}" | grep -E "^llama" | awk '{print $5}' | head -1)
    DECODE_TPS=$(echo "${BENCH_OUTPUT}" | grep -E "^llama" | awk '{print $7}' | head -1)

    RESULTS_PREFILL[${prompt_len}]="${PREFILL_TPS:-N/A}"
    RESULTS_DECODE[${prompt_len}]="${DECODE_TPS:-N/A}"

    echo "  Prefill: ${PREFILL_TPS:-N/A} t/s"
    echo "  Decode:  ${DECODE_TPS:-N/A} t/s"

    # VRAM check
    if [ -n "${ROCM_SMI}" ]; then
        VRAM_AFTER=$(${ROCM_SMI} --showmeminfo 0 2>/dev/null | grep -i "gpu memory" | awk '{print $NF}' || echo "0")
        RESULTS_VRAM[${prompt_len}]="${VRAM_AFTER}"
    fi

    echo ""
    sleep 2
done

# ─── Summary Table ───────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              Benchmark Results                       ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
printf "  %-12s | %-18s | %-18s | %-10s\n" "Context" "Prefill (t/s)" "Decode (t/s)" "VRAM (MB)"
printf "  %-12s-+-%-18s-+-%-18s-+-%-10s\n" "------------" "------------------" "------------------" "----------"

for prompt_len in "${PROMPT_SIZES[@]}"; do
    printf "  %-12s | %-18s | %-18s | %-10s\n" \
        "${prompt_len}" \
        "${RESULTS_PREFILL[${prompt_len}]}" \
        "${RESULTS_DECODE[${prompt_len}]}" \
        "${RESULTS_VRAM[${prompt_len}]:-N/A}"
done

echo ""

# ─── VRAM Validation ─────────────────────────────────────────────────────────
echo -e "${CYAN}=== VRAM Validation ===${NC}"
VRAM_EXCEEDED=0
for prompt_len in "${PROMPT_SIZES[@]}"; do
    VRAM_MB=${RESULTS_VRAM[${prompt_len}]:-0}
    VRAM_BYTES=$((VRAM_MB * 1024 * 1024))
    if [ "${VRAM_BYTES}" -gt "${VRAM_LIMIT}" ] 2>/dev/null; then
        echo -e "  ${RED}✗ Context ${prompt_len}: ${VRAM_MB} MB exceeds 13.5 GB limit${NC}"
        VRAM_EXCEEDED=1
    else
        echo -e "  ${GREEN}✓ Context ${prompt_len}: ${VRAM_MB} MB within limit${NC}"
    fi
done

if [ "${VRAM_EXCEEDED}" -eq 1 ]; then
    echo -e "${RED}VRAM LIMIT EXCEEDED${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo ""
echo "Compare modes:"
echo "  Optimized:  $0"
echo "  Baseline:   $0 baseline"
