#!/usr/bin/env bash
# Test script for llama.cpp RDNA2 build — runs inference test and reports corruption
# Usage: ./scripts/test_inference.sh [model_path] [prompt]
#
# Auto-handles: killing existing llama-server, waiting for VRAM, restoring after test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/gpu_failback.sh"

MODEL="${1:-/home/stormrage/models/allura-org_Qwen3.6-35B-A3B-Anko-IQ4_NL.gguf}"
PROMPT="${2:-Hello, please respond in English only. Say 'test passed' if you understand this message.}"
LLAMA_CLI="./build/bin/llama-cli"

# Check binary exists
if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: $LLAMA_CLI not found. Run cmake build first."
    exit 1
fi

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

# Acquire GPU (kill existing llama-server, save state)
gpu_acquire
gpu_failback_trap

echo "=== Running inference test ==="
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo ""

# Run test with timeout (300s max)
OUTPUT=$({ timeout 300 "$LLAMA_CLI" \
    -m "$MODEL" \
    -ngl 99 \
    -c 4096 \
    -b 512 \
    -ub 1024 \
    -ctk turbo4 \
    -ctv turbo2 \
    -fa on \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --repeat-penalty 1.1 \
    --presence-penalty 0.0 \
    --no-mmap \
    --mlock \
    -t 8 \
    -tb 12 \
    --parallel 1 \
    --jinja \
    --no-context-shift \
    -ncmoe 33 \
    --cpu-range 0-7 \
    --cpu-strict 1 \
    -fitt 256 \
    --kv-unified \
    --cache-ram 4096 \
    --cache-reuse 256 \
    --reasoning auto \
    -n 50 \
    -p "$PROMPT" 2>&1; } | tee /tmp/llama-test-output.log)

echo ""
echo "=== Test Results ==="

# Check for corruption indicators
CORRUPTION=0

# Check for mixed language characters (CJK, Arabic, Cyrillic, etc)
if echo "$OUTPUT" | grep -qP '[\x{4e00}-\x{9fff}\x{3040}-\x{309f}\x{30a0}-\x{30ff}\x{0600}-\x{06ff}\x{0400}-\x{04ff}\x{ac00}-\x{d7af}]'; then
    echo "FAIL: Output contains mixed language characters (corruption detected)"
    CORRUPTION=1
fi

# Check for random symbol sequences
if echo "$OUTPUT" | grep -qP '[\x{2000}-\x{206f}\x{2100}-\x{214f}]{10,}'; then
    echo "FAIL: Output contains random symbol sequences (corruption detected)"
    CORRUPTION=1
fi

# Check for "test passed" or coherent English response
if echo "$OUTPUT" | grep -qiP '(test passed|hello|hi|good|understand|english)'; then
    if [[ $CORRUPTION -eq 0 ]]; then
        echo "PASS: Output is coherent English, no corruption detected"
    else
        echo "PARTIAL: Contains some English but also corruption"
        CORRUPTION=1
    fi
else
    if [[ $CORRUPTION -eq 0 ]]; then
        echo "PASS: No corruption detected (output may be short)"
    else
        echo "FAIL: No coherent output, likely memory corruption"
        CORRUPTION=1
    fi
fi

echo ""
echo "=== Output Log ==="
tail -50 /tmp/llama-test-output.log

echo ""
echo "=== VRAM Usage After Test ==="
_vram_file=$(_gpu_vram_file)
if [[ -n "$_vram_file" ]]; then
    _used=$(_gpu_vram_used)
    echo "$((_used / 1024 / 1024)) MB used"
fi

exit $CORRUPTION
