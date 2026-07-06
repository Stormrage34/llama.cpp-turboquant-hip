#!/usr/bin/env bash
# Long-context throughput benchmark: q8_0 vs turbo3 V vs turbo4 V
# Tests progressively longer prompts to find scaling gaps.
set -euo pipefail

MODEL="/home/stormrage/models/Qwen-AgentWorld-35B-A3B-UD-IQ4_NL.gguf"
BIN="./build/bin/llama-bench"
COMMON="-m $MODEL -ngl 99 -ncmoe 15 -b 4096 -ub 2048 -fa 1 -r 1"

# Long context sizes: 32K, 48K, 64K, 80K, 96K
# Push until VRAM OOM or throughput collapses.
PROMPTS="16384"

echo "============================================"
echo " Long Context Benchmark"
echo " GPU: $(rocm-smi --showproductname 2>/dev/null || echo 'unknown')"
echo " Model: $(basename $MODEL)"
echo " Prompts: $PROMPTS"
echo "============================================"
echo ""

# --- Baseline: q8_0 / q8_0 ---
echo "--- [1/4] q8_0 / q8_0 (baseline) ---"
$BIN $COMMON -ctk q8_0 -ctv q8_0 -p $PROMPTS 2>&1

echo ""
echo "--- [2/4] q8_0 K / turbo3 V ---"
$BIN $COMMON -ctk q8_0 -ctv turbo3_0 -p $PROMPTS 2>&1

echo ""
echo "--- [3/4] q8_0 K / turbo4 V ---"
$BIN $COMMON -ctk q8_0 -ctv turbo4_0 -p $PROMPTS 2>&1

echo ""
echo "--- [4/4] turbo3 / turbo3 (symmetric) ---"
$BIN $COMMON -ctk turbo3_0 -ctv turbo3_0 -p $PROMPTS 2>&1

echo ""
echo "Done. Compare pp columns across the 4 runs."
