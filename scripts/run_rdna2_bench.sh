#!/bin/bash
# Phase 2E: RDNA2 Benchmark Harness
# Runs end-to-end inference benchmarks across context lengths
# Captures performance metrics, VRAM usage, and compares against baseline

set -e

# Configuration
MODEL_PATH="/home/stormrage/models/Qwen3_35BMTPIQ4.gguf"
PROJECT_ROOT="/home/stormrage/llama.cpp-turboquant-hip"
BIN_DIR="${PROJECT_ROOT}/build/bin"
LLAMA_BENCH="${BIN_DIR}/llama-bench-rdna2"
ROCM_SMI="/home/stormrage/rocm-7.13-nightly/bin/rocm-smi"

# VRAM hard limit (bytes) - 13.5 GB
VRAM_LIMIT=$((13500 * 1024 * 1024))

# Prompt lengths to test (simulates different context usage)
PROMPT_SIZES=(512 2048 4096)

# Benchmark parameters
PROMPT_LEN=512
GEN_LEN=128
BATCH_SIZE=256
UBATCH_SIZE=128
CTK="turbo4"
CTV="turbo2"
FA="1"
NO_MMAP=0
FIT_TARGET=2048  # Leave 2GB margin for safety

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Mode
MODE="${1:-optimized}"

echo -e "${GREEN}=== Phase 2E: RDNA2 Benchmark Harness ===${NC}"
echo "Model: ${MODEL_PATH}"
echo "Mode: ${MODE}"
echo ""

# Verify prerequisites
if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}ERROR: Model not found at ${MODEL_PATH}${NC}"
    exit 1
fi

if [ ! -x "${LLAMA_BENCH}" ]; then
    echo -e "${RED}ERROR: llama-bench-rdna2 not found. Run build_rdna2_llama.sh first.${NC}"
    exit 1
fi

if [ ! -x "${ROCM_SMI}" ]; then
    echo -e "${YELLOW}WARNING: rocm-smi not found at ${ROCM_SMI}. VRAM monitoring disabled.${NC}"
    ROCM_SMI=""
fi

# Kill any existing llama-server processes
echo "Cleaning up existing processes..."
pkill -f llama-server 2>/dev/null || true
sleep 2

# Pre-flight VRAM check
echo ""
echo -e "${CYAN}=== Pre-flight VRAM Check ===${NC}"
if [ -n "${ROCM_SMI}" ]; then
    VRAM_BEFORE=$(${ROCM_SMI} --showmeminfo 0 2>/dev/null | grep -i "gpu memory" | awk '{print $NF}' || echo "0")
    echo "VRAM before: ${VRAM_BEFORE} MB"
else
    echo "VRAM monitoring: disabled (rocm-smi not available)"
    VRAM_BEFORE=0
fi

# Results storage
declare -A RESULTS_PREFILL
declare -A RESULTS_DECODE
declare -A RESULTS_VRAM

# Run benchmarks
for prompt_len in "${PROMPT_SIZES[@]}"; do
    echo ""
    echo -e "${CYAN}=== Prompt Length: ${prompt_len} ===${NC}"

    # Set environment variables based on mode
    if [ "${MODE}" = "optimized" ]; then
        export RDNA2_OPT_V1=1
        export RDNA2_ASYNC_PIPELINE=1
        export RDNA2_MATMUL_OPT_V1=1
        echo "Environment: RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1"
    else
        unset RDNA2_OPT_V1
        unset RDNA2_ASYNC_PIPELINE
        unset RDNA2_MATMUL_OPT_V1
        echo "Environment: baseline (no RDNA2 optimizations)"
    fi

    # Run llama-bench
    echo "Running llama-bench..."
    BENCH_OUTPUT=$( ${LLAMA_BENCH} \
        -m "${MODEL_PATH}" \
        -p ${prompt_len} \
        -n ${GEN_LEN} \
        -b ${BATCH_SIZE} \
        -ub ${UBATCH_SIZE} \
        -ctk ${CTK} \
        -ctv ${CTV} \
        -fa ${FA} \
        -mmp ${NO_MMAP} \
        -t 8 \
        -ngl 99 \
        --fit-target ${FIT_TARGET} \
        -fitc 4096 \
        -r 3 \
        -o md \
        2>&1 ) || true

    echo "${BENCH_OUTPUT}"

    # Parse results (llama-bench outputs tab-separated values)
    PREFILL_TPS=$(echo "${BENCH_OUTPUT}" | grep -E "^llama" | awk '{print $5}' | head -1)
    DECODE_TPS=$(echo "${BENCH_OUTPUT}" | grep -E "^llama" | awk '{print $7}' | head -1)

    RESULTS_PREFILL[${prompt_len}]="${PREFILL_TPS:-N/A}"
    RESULTS_DECODE[${prompt_len}]="${DECODE_TPS:-N/A}"

    # Post-benchmark VRAM check
    if [ -n "${ROCM_SMI}" ]; then
        VRAM_AFTER=$(${ROCM_SMI} --showmeminfo 0 2>/dev/null | grep -i "gpu memory" | awk '{print $NF}' || echo "0")
        echo "VRAM after: ${VRAM_AFTER} MB"
        RESULTS_VRAM[${prompt_len}]="${VRAM_AFTER}"
    fi

    sleep 2
done

# Summary
echo ""
echo -e "${GREEN}=== Phase 2E Benchmark Summary ===${NC}"
echo ""
printf "%-12s | %-15s | %-15s | %-10s\n" "Context" "Prefill (t/s)" "Decode (t/s)" "VRAM (MB)"
printf "%-12s-+-%-15s-+-%-15s-+-%-10s\n" "------------" "---------------" "---------------" "----------"

for prompt_len in "${PROMPT_SIZES[@]}"; do
    printf "%-12s | %-15s | %-15s | %-10s\n" \
        "${prompt_len}" \
        "${RESULTS_PREFILL[${prompt_len}]}" \
        "${RESULTS_DECODE[${prompt_len}]}" \
        "${RESULTS_VRAM[${prompt_len}]:-N/A}"
done

echo ""

# VRAM validation
echo -e "${CYAN}=== VRAM Validation ===${NC}"
VRAM_EXCEEDED=0
for prompt_len in "${PROMPT_SIZES[@]}"; do
    VRAM_MB=${RESULTS_VRAM[${prompt_len}]:-0}
    VRAM_BYTES=$((VRAM_MB * 1024 * 1024))
    if [ "${VRAM_BYTES}" -gt "${VRAM_LIMIT}" ] 2>/dev/null; then
        echo -e "${RED}✗ Prompt ${prompt_len}: VRAM ${VRAM_MB} MB exceeds limit (13.5 GB)${NC}"
        VRAM_EXCEEDED=1
    else
        echo -e "${GREEN}✓ Prompt ${prompt_len}: VRAM ${VRAM_MB} MB within limit${NC}"
    fi
done

if [ "${VRAM_EXCEEDED}" -eq 1 ]; then
    echo -e "${RED}VRAM LIMIT EXCEEDED - ABORTING${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Phase 2E Complete ===${NC}"
echo "Results saved. Compare with baseline using: $0 baseline"
