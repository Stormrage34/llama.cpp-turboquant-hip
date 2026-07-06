#!/usr/bin/env bash
# Revised swizzle profiling script – sequential per‑model execution, filters tiny models
# Usage: ./scripts/run_swizzle_test.sh [options] [model1 model2 ...]
# Options:
#   --model PATH         Path to a gguf model (can be repeated)
#   --ngl N              Number of GPU layers (default: 99)
#   --cpu-moe N          CPU MoE experts (default auto‑detect: 20 for MOE, 0 otherwise)
#   --cache-k TYPE       Cache type for K (default: q8_0)
#   --cache-v TYPE       Cache type for V (default: turbo3)
#   --tokens N           Number of tokens to generate (default: 64)
#   --prompt TEXT        Prompt string (default: "test")
#   --output-dir DIR     Directory to store profiling DB and JSON (default: ./swizzle_results)
#   --batch-size N       Batch size (default: 4096)
#   --ubatch N           Unified batch size (default: 512)
#   --help               Show this help message
#
# The script runs rocprofv3 with --stats --hip‑trace on llama‑cli, stores the
# SQLite DB, extracts copy‑buffer kernel launches and per‑kernel statistics,
# and prints a machine‑readable JSON summary for each model.
# It processes models sequentially, skips files smaller than 1 GiB, and
# automatically selects a CPU‑MoE expert count for MOE models unless
# overridden by the user.

set -euo pipefail

usage() {
  grep -E '^#' "$0" | sed -e 's/^# //'
  exit 0
}

# Default parameters
NGPU_LAYERS=99
CPU_MOE=0          # will be overridden by auto‑detect if not explicitly set
CPU_MOE_SET=0
CACHE_K="q8_0"
CACHE_V="turbo3"
FA=1
C=128               # context size (tokens)
TOKEN_COUNT=64
PROMPT="test"
OUTPUT_DIR="./swizzle_results"
BATCH_SIZE=4096
UBATCH=512

# Heuristic: lower NGPU_LAYERS for very large models (size > 8 GiB)
auto_adjust_ngl() {
  local model_path="$1"
  if [[ -f "$model_path" ]]; then
    local sz=$(stat -c%s "$model_path")
    # Heuristic thresholds (bytes)
    #   >16 GiB  → GPU offload disabled (ngl=0)
    #   >12 GiB  → ngl=20
    #   >8 GiB   → ngl=30
    if (( sz > 16000000000 )); then
      NGPU_LAYERS=0
    elif (( sz > 12000000000 )); then
      NGPU_LAYERS=20
    elif (( sz > 8000000000 )); then
      NGPU_LAYERS=30
    fi
  fi
}

# Heuristic: detect MOE by filename or by inspecting model metadata for "moe"/"expert"
is_moe_model() {
  local model_path="$1"
  # Filename clues
  if [[ "$model_path" =~ [Mm][Oo][Ee] || "$model_path" =~ [Mm][Tt][Pp] ]]; then
    return 0
  fi
  # Scan binary strings for common MOE markers
  if command -v strings >/dev/null; then
    if strings "$model_path" | grep -i -E "moe|expert|router" >/dev/null; then
      return 0
    fi
  fi
  return 1
}

# Collect model paths (positional arguments or via --model)
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2;;
    --ngl) NGPU_LAYERS="$2"; shift 2;;
    --cpu-moe) CPU_MOE="$2"; CPU_MOE_SET=1; shift 2;;
    --cache-k) CACHE_K="$2"; shift 2;;
    --cache-v) CACHE_V="$2"; shift 2;;
    --tokens) TOKEN_COUNT="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --ubatch) UBATCH="$2"; shift 2;;
    --help) usage;;
    *) MODELS+=("$1"); shift;;
  esac
done

# If no models supplied, fall back to first IQ4_XS model in $HOME/models
if [[ ${#MODELS[@]} -eq 0 ]]; then
  DEFAULT_MODEL=$(find "$HOME/models" -type f -name "*IQ4_XS*.gguf" | head -n1)
  if [[ -n "$DEFAULT_MODEL" ]]; then
    MODELS=("$DEFAULT_MODEL")
  else
    echo "Error: No model provided and no IQ4_XS model found in $HOME/models" >&2
    exit 1
  fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load GPU acquisition helpers (no‑op if server not running)
source "$(dirname "$0")/gpu_failback.sh" && gpu_acquire

for MODEL_PATH in "${MODELS[@]}"; do
  # Skip models smaller than 1 GiB (1_073_741_824 bytes)
  if [[ -f "$MODEL_PATH" ]]; then
    FILE_SIZE=$(stat -c%s "$MODEL_PATH")
    if (( FILE_SIZE < 1073741824 )); then
      echo "Skipping $MODEL_PATH (size $(($FILE_SIZE/1024/1024)) MiB < 1 GiB)"
      continue
    fi
  else
    echo "Warning: Model path $MODEL_PATH does not exist – skipping"
    continue
  fi

  echo "=== Profiling $MODEL_PATH ==="

auto_adjust_ngl "$MODEL_PATH"  # re‑enable automatic GPU‑layer reduction

# Determine CPU‑MoE count if not forced by user (detect MOE via filename or metadata)
if [[ $CPU_MOE_SET -eq 0 ]]; then
    if is_moe_model "$MODEL_PATH"; then
        CUR_CPU_MOE=20
    else
        CUR_CPU_MOE=0
    fi
else
    CUR_CPU_MOE=$CPU_MOE
fi

  DB_PATH="$OUTPUT_DIR/$(basename "$MODEL_PATH" .gguf)_rocprof.db"
  JSON_PATH="$OUTPUT_DIR/$(basename "$MODEL_PATH" .gguf)_summary.json"

  ROC_PROF_CMD=(rocprofv3 --stats --hip-trace -o "$DB_PATH")
  LLAMA_CMD=(./build/bin/llama-bench -m "$MODEL_PATH" -ngl $NGPU_LAYERS --n-cpu-moe $CUR_CPU_MOE \
    --cache-type-k $CACHE_K --cache-type-v $CACHE_V -fa $FA  -b $BATCH_SIZE -ub $UBATCH -n $TOKEN_COUNT -p 0)

  # Run profiling – allow non‑zero exit (e.g., OOM) but capture status
  set +e
  "${ROC_PROF_CMD[@]}" -- "${LLAMA_CMD[@]}"
  RUN_STATUS=$?
  set -e

  # Locate the actual ROCprof DB (use the known path we created)
  DB_ACTUAL="$DB_PATH"
  if [[ -z "$DB_ACTUAL" ]]; then
    COPY_COUNT=null; TOTAL_DISPATCH=null; KERNEL_CSV=""
  else
    if command -v sqlite3 >/dev/null; then
      TABLE_EXISTS=$(sqlite3 "$DB_ACTUAL" "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='rocpd_kernel_dispatch';")
      if [[ "$TABLE_EXISTS" -eq 1 ]]; then
        COPY_COUNT=$(sqlite3 "$DB_ACTUAL" "SELECT COUNT(*) FROM rocpd_kernel_dispatch WHERE kernel_name LIKE '%copyBuffer%';")
        TOTAL_DISPATCH=$(sqlite3 "$DB_ACTUAL" "SELECT COUNT(*) FROM rocpd_kernel_dispatch;")
        KERNEL_CSV=$(sqlite3 -header -csv "$DB_ACTUAL" "SELECT kernel_name, COUNT(*) AS launches, SUM(duration) AS total_us FROM rocpd_kernel_dispatch GROUP BY kernel_name ORDER BY launches DESC;")
      else
        COPY_COUNT=0; TOTAL_DISPATCH=0; KERNEL_CSV=""
      fi
    else
      COPY_COUNT=null; TOTAL_DISPATCH=null; KERNEL_CSV=""
    fi
  fi

  # Produce machine‑readable JSON
  python3 - <<PY >"$JSON_PATH"
import json, os, csv
summary = {
  "model": os.path.basename("$MODEL_PATH"),
  "parameters": {
    "ngl": $NGPU_LAYERS,
    "cpu_moe": $CUR_CPU_MOE,
    "cache_k": "$CACHE_K",
    "cache_v": "$CACHE_V",
    "fa": $FA,
    "c": $C,
    "batch_size": $BATCH_SIZE,
    "ubatch": $UBATCH,
    "token_count": $TOKEN_COUNT,
    "prompt": "$PROMPT",
    "fallback_to_cpu": $(( RUN_STATUS != 0 && $NGPU_LAYERS != 0 ? 1 : 0 ))
  },
  "profiling_db": "$DB_ACTUAL",
  "copy_buffer_launches": $COPY_COUNT,
  "total_kernel_dispatches": $TOTAL_DISPATCH,
  "kernel_stats": []
}
csv_data = """$KERNEL_CSV"""
if csv_data.strip():
    for row in csv.DictReader(csv_data.splitlines()):
        try:
            row["launches"] = int(row["launches"])
        except Exception:
            row["launches"] = 0
        try:
            row["total_us"] = float(row["total_us"]) if row["total_us"] else 0.0
        except Exception:
            row["total_us"] = 0.0
        summary["kernel_stats"].append(row)
print(json.dumps(summary, indent=2))
PY

  cat "$JSON_PATH"

  if [[ $RUN_STATUS -ne 0 && $NGPU_LAYERS -ne 0 ]]; then
    echo "Model $MODEL_PATH failed with exit $RUN_STATUS. Consider retrying with a different configuration (e.g., lower ngl)" >&2
  fi
done

exit $RUN_STATUS
