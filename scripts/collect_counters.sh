#!/usr/bin/env bash
# scripts/collect_counters.sh
# Runs rocprofv3 on llama-cli and saves SQLite to benchmarks/raw/
# Usage: ./scripts/collect_counters.sh [binary] [prompt]

set -e

ROCMPATH="${ROCM_PATH:-/opt/rocm}"
BINARY="${1:-build/bin/llama-cli}"
PROMPT="${2:-test prompt for inference}"

COUNTERS="SQ_INSTS_VALU,VALUBusy,MeanOccupancyPerCU,MemUnitBusy,WAVE_ISSUE_WAIT"
OUTPUT_DIR="benchmarks/raw/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running rocprofv3 on $BINARY..."
echo "Prompt: $PROMPT"
echo "Counters: $COUNTERS"
echo "Output: $OUTPUT_DIR"

$ROCMPATH/bin/rocprof --counters "$COUNTERS" \
    "$BINARY" -p "$PROMPT" -n 128 -ngl 99 \
    --stats --stats-output "$OUTPUT_DIR/counters.h5"

echo "Results saved to $OUTPUT_DIR/"
