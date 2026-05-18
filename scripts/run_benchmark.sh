#!/bin/bash
# run_benchmark.sh — Standardized RDNA2 cache comparison benchmark
# Usage: ./scripts/run_benchmark.sh [model.gguf] [cache_k,cache_v]...
#
# Compares our fork (turbo cache) vs original (standard cache) performance
# across 4 prompt types: coding, creative, thinking, solving
#
# Default: compares q8_0/turbo3 (our), q8_0/q8_0 (original symmetric),
#          q8_0/q4_0 (original asymmetric), q4_0/q4_0 (original aggressive)
#
# Prompts:
#   coding:   "Write a Python function to check if a number is prime."
#   creative: "Write a short story about a robot learning to paint."
#   thinking: "Explain quantum entanglement to a high school student."
#   solving:  "Solve: A train leaves Station A at 60 mph..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT_ROOT}/build/bin/llama-cli"
MODEL="${1:-${PROJECT_ROOT}/../models/Qwen3_35BMTPIQ4.gguf}"
shift || true

# Default cache configs to test
CACHE_CONFIGS=("$@")
if [ ${#CACHE_CONFIGS[@]} -eq 0 ]; then
    CACHE_CONFIGS=(
        "q8_0,turbo3"   # Our fork - asymmetric turbo
        "turbo3,turbo3" # Our fork - symmetric turbo
        "q8_0,q8_0"     # Original - symmetric q8
        "q8_0,q4_0"     # Original - asymmetric
        "q4_0,q4_0"     # Original - aggressive
    )
fi

PROMPTS=(
    "Write a Python function that takes a list of integers and returns only the prime numbers. Include type hints and docstring."
    "Write a short story about a robot learning to paint, focusing on its internal struggle between precision and artistic expression."
    "Explain quantum entanglement to a high school student using analogies. Avoid technical jargon."
    "Solve step by step: A train leaves Station A at 60 mph. Another train leaves Station B at 90 mph. Station B is 300 miles away. When and where do they meet?"
)

PROMPT_NAMES=("coding" "creative" "thinking" "solving")

echo "================================================================"
echo "  RDNA2 Cache Comparison Benchmark"
echo "  Model: $(basename "$MODEL")"
echo "  GPU: AMD Radeon RX 6800 XT (gfx1030)"
echo "  Config: -ngl 99 --n-cpu-moe 41 -c 32768 -fa 1 -st -n 1000"
echo "================================================================"
echo ""

# Check binary
if [ ! -x "$BINARY" ]; then
    echo "❌ Binary not found: $BINARY"
    echo "   Build first: cmake --build build -- -j 16"
    exit 1
fi

# Check model
if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

# Source GPU failback
source "${SCRIPT_DIR}/gpu_failback.sh"

RESULTS_FILE="${PROJECT_ROOT}/benchmarks/raw/benchmark_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$RESULTS_FILE")"

header="Cache Config | Prompt Type | Prompt t/s | Decode t/s | Draft Accept | VRAM MiB"
echo "$header" | tee -a "$RESULTS_FILE"
echo "----------- | ----------- | ---------- | ---------- | ------------ | --------" | tee -a "$RESULTS_FILE"

for cache_config in "${CACHE_CONFIGS[@]}"; do
    CTK=$(echo "$cache_config" | cut -d, -f1)
    CTV=$(echo "$cache_config" | cut -d, -f2)
    
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        pname="${PROMPT_NAMES[$i]}"
        
        echo -n "  [$cache_config] [$pname] ... "
        
        gpu_acquire > /dev/null 2>&1
        sleep 4
        
        output=$(timeout 300 "$BINARY" \
            -m "$MODEL" \
            -ngl 99 --n-cpu-moe 41 \
            -c 32768 --cache-type-k "$CTK" --cache-type-v "$CTV" \
            -fa 1 --spec-type mtp --spec-draft-n-max 2 \
            -st -n 1000 -p "$prompt" 2>&1)
        
        prompt_t=$(echo "$output" | grep 'Prompt:' | sed 's/.*Prompt: \([0-9.]*\) t\/s.*/\1/' || echo "N/A")
        gen_t=$(echo "$output" | grep 'Generation:' | sed 's/.*Generation: \([0-9.]*\) t\/s.*/\1/' || echo "N/A")
        draft=$(echo "$output" | grep 'draft' | head -1 || echo "")
        vram=$(echo "$output" | grep 'ROCm0' | sed 's/.*self[[:space:]]*\([0-9]*\).*/\1/' || echo "N/A")
        
        echo "${cache_config} | ${pname} | ${prompt_t} | ${gen_t} | ${draft:0:30} | ${vram}" | tee -a "$RESULTS_FILE"
        sleep 2
    done
    echo "" | tee -a "$RESULTS_FILE"
done

echo ""
echo "✅ Benchmark complete. Results saved to: $RESULTS_FILE"
echo ""
echo "=== Summary ==="
echo "(avg decode)"
tail -n +3 "$RESULTS_FILE" | grep -v "^-" | sed 's/|/:/g' | while IFS=: read -r config pname pt gen rest; do
    echo "  $config / $pname: prompt=${pt}t/s decode=${gen}t/s"
done
