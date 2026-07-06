#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  MODEL=$(find /home/stormrage/models -type f -name "*IQ4_XS*.gguf" | head -n1)
fi
if [[ -z "$MODEL" ]]; then
  echo "No IQ4_XS model found. Provide a model path."
  exit 1
fi
echo "Using model: $MODEL"

N_BATCH=512
N_UBATCH=2048  # Will be clamped to <=1024 by fix
PROMPT="Write a short story about a robot learning to paint."

OUTDIR="benchmarks/raw/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/moe_longgen_$(basename "$MODEL" .gguf).log"

echo "Running long-generation test: n_tokens=2000, n_ubatch=$N_UBATCH, n_batch=$N_BATCH"
build/bin/llama-cli -m "$MODEL" -ngl 99 -ncmoe 41 -ub "$N_UBATCH" -b "$N_BATCH" \
  -c 4096 --single-turn -n 2000 -p "$PROMPT" 2>&1 | tee "$LOGFILE"

# Check for corruption
if grep -q "<unused" "$LOGFILE"; then
  echo "FAIL: found <unused> tokens"
  exit 1
fi
echo "PASS: 2000-token MoE generation completed without corruption."
