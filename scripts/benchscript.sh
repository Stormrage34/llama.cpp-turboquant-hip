#!/bin/bash
set -euo pipefail

# =============================================================================
# LLAMA-SERVER BENCHMARK LAUNCHER (Dynamic & Size-Aware)
# RX 6800 XT (16GB VRAM) Optimized | May 2026 Mainline Compliant
# =============================================================================

# --- Environment Setup ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROCM_PATH="${ROCM_PATH:-/home/stormrage/rocm-7.13-nightly}"
export HIP_PATH="${ROCM_PATH}"
MODEL_DIR="${MODEL_DIR:-/home/stormrage/models}"
BINARY_DIR="${PROJECT_ROOT}/build/bin"

# --- Build Environment Detection ---
echo "========================================="
echo "   Select Llama-Server Build Engine"
echo "========================================="
echo "1) Original Main Branch (/home/stormrage/llama.cpp/build/bin/)"
echo "2) TurboQuant-HIP Custom (${BINARY_DIR})"
read -rp "Enter choice [1-2, Default: 1]: " BUILD_CHOICE

if [ "$BUILD_CHOICE" = "2" ]; then
    ALLOWED_CACHES=("f32" "f16" "bf16" "q8_0" "q4_0" "q4_1" "iq4_nl" "q5_0" "q5_1" "turbo2" "turbo3" "turbo4")
    echo "-> TurboQuant-HIP Build Detected. PolarQuant cache types enabled."
else
    BINARY_DIR="/home/stormrage/llama.cpp/build/bin"
    ALLOWED_CACHES=("f32" "f16" "bf16" "q8_0" "q4_0" "q4_1" "iq4_nl" "q5_0" "q5_1")
    echo "-> Original Main Build Detected. Standard cache formats enabled."
fi

# Validate binary early
if [[ ! -x "$BINARY_DIR/llama-server" ]]; then
    echo -e "\n❌ Error: llama-server not found or not executable in $BINARY_DIR"
    exit 1
fi

# --- Cache Type Selection ---
echo -e "\nAvailable Cache Formats (0-based index):"
for i in "${!ALLOWED_CACHES[@]}"; do
    echo "$i) ${ALLOWED_CACHES[$i]}"
done

read -rp "Select Key Cache Type (ctk) index [Default 1/f16]: " K_INDEX
CACHE_K="${ALLOWED_CACHES[${K_INDEX:-1}]}" 2>/dev/null || CACHE_K="f16"

read -rp "Select Value Cache Type (ctv) index [Default 1/f16]: " V_INDEX
CACHE_V="${ALLOWED_CACHES[${V_INDEX:-1}]}" 2>/dev/null || CACHE_V="f16"

# --- Task Profile Selection ---
echo -e "\n========================================="
echo "   Select Task Optimization Profile"
echo "========================================="
echo "1) Coding Research & Debugging (Balanced)"
echo "2) Hard Technical Tasks & Scripts (Strict)"
echo "3) Creative Generation & Brainstorming (High Entropy)"
echo "4) General Conversation / Default"
read -rp "Enter choice [1-4]: " TASK_CHOICE

case $TASK_CHOICE in
    1) TEMP=0.6; TOP_P=0.95; TOP_K=20; MIN_P=0.05; DESC="Coding Research (Balanced)" ;;
    2) TEMP=0.0; TOP_P=0.90; TOP_K=10; MIN_P=0.01; DESC="Hard Coding (Strict)" ;;
    3) TEMP=0.85; TOP_P=0.98; TOP_K=50; MIN_P=0.08; DESC="Creative Logic (High Entropy)" ;;
    *) TEMP=0.7; TOP_P=0.95; TOP_K=40; MIN_P=0.05; DESC="General Workflow" ;;
esac

# --- DYNAMIC MODEL DISCOVERY ---
echo -e "\n📂 Scanning Models Directory: $MODEL_DIR"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found."; exit 1
fi

# Find .gguf files safely, sorted alphabetically
mapfile -t MODEL_PATHS < <(find "$MODEL_DIR" -maxdepth 1 -type f -name "*.gguf" 2>/dev/null | sort)

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "❌ No .gguf models found in $MODEL_DIR"; exit 1
fi

echo -e "\n📦 Available Automation Models:"
for i in "${!MODEL_PATHS[@]}"; do
    FNAME=$(basename "${MODEL_PATHS[$i]}")
    SIZE=$(du -h "${MODEL_PATHS[$i]}" | cut -f1)
    echo "$i) $FNAME [$SIZE]"
done

read -rp "Select model index [0-$(( ${#MODEL_PATHS[@]} - 1 ))]: " MODEL_INDEX

if [[ ! "$MODEL_INDEX" =~ ^[0-9]+$ ]] || [ "$MODEL_INDEX" -ge "${#MODEL_PATHS[@]}" ]; then
    echo "Error: Invalid model index selection."
    exit 1
fi

SELECTED_MODEL="${MODEL_PATHS[$MODEL_INDEX]}"
MODEL_NAME=$(basename "$SELECTED_MODEL")

# --- CONTEXT SIZE CONFIGURATION (MOVED UP BEFORE NCMOE CALC) ---
echo -e "\n========================================="
echo "   Configure Target Evaluation Context   "
echo "========================================="
echo "Presets: 1) 32k (32768)  2) 64k (65536)  3) 128k (131072)  4) Custom Max"
read -rp "Enter choice [1-4]: " CTX_CHOICE

case $CTX_CHOICE in
    1) CTX=32768 ;;
    2) CTX=65536 ;;
    3) CTX=131072 ;;
    4)
        read -rp "Enter custom max context size (e.g. 200000): " CTX_INPUT
        if [[ "$CTX_INPUT" =~ ^[0-9]+$ ]] && [ "$CTX_INPUT" -gt 0 ]; then
            CTX=$CTX_INPUT
        else
            echo "Invalid input. Falling back to 131072."
            CTX=131072
        fi
        ;;
    *) CTX=32768 ;;
esac

# --- SIZE-AWARE NCMOE CALCULATION ---
# Now that CTX is defined, we can safely calculate NCMOE
MODEL_BYTES=$(stat -c %s "$SELECTED_MODEL")
SIZE_GB=$(( MODEL_BYTES / 1073741824 ))

LOWER_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
IS_MTP=0; IS_MOE=0; IS_GEMMA=0

[[ "$LOWER_NAME" =~ "mtp" ]] && IS_MTP=1
[[ "$LOWER_NAME" =~ "gemma" ]] && IS_GEMMA=1
if [[ "$LOWER_NAME" =~ a[0-9]+b ]] || [[ "$LOWER_NAME" =~ (_moe|-moe|mixtral|deepseek) ]]; then
    IS_MOE=1
fi

# Dynamic NCMOE: Scales with model size & context pressure
# Formula: Base (2 per GB) + Context Load (2 per 32k tokens). Capped at 32 for 16GB VRAM safety.
if [ $IS_MOE -eq 1 ]; then
    CTX_LOAD=$(( CTX / 32768 ))
    NCMOE=$(( (SIZE_GB * 2) + (CTX_LOAD * 2) ))
    # Safety caps for RX 6800 XT (16GB VRAM)
    [ "$NCMOE" -lt 4 ]  && NCMOE=4
    [ "$NCMOE" -gt 32 ] && NCMOE=32
    echo "-> MoE Detected (${SIZE_GB}GB). Auto-calculated N-CPU-MOE: $NCMOE"
else
    NCMOE=0
    echo "-> Dense Model (${SIZE_GB}GB). N-CPU-MOE: Disabled"
fi

# --- BATCH STRATEGY ---
if [ "$CTX" -gt 100000 ]; then
    BATCH=1024; UBATCH=1024
else
    BATCH=2048; UBATCH=512
fi

# --- PERFORMANCE REPORT ---
clear
echo "================================================================"
echo "          LLAMA-SERVER AUTOMATION PROFILED REPORT               "
echo "================================================================"
echo "Build Path:     $BINARY_DIR/"
echo "Target Model:   $MODEL_NAME [$SIZE_GB GB]"
echo "Target Engine:  $DESC"
echo "Context Config: $CTX tokens [K: $CACHE_K | V: $CACHE_V]"
echo "Architecture:   MTP=$IS_MTP | MoE=$IS_MOE | Gemma=$IS_GEMMA"
echo "Memory Layout:  N-CPU-MOE=$NCMOE | Batch=$BATCH | U-Batch=$UBATCH"
echo "----------------------------------------------------------------"
read -rp "Press [Enter] to spin up the production llama-server engine..."

# --- COMMAND CONSTRUCTION ---
CMD=("$BINARY_DIR/llama-server" "-m" "$SELECTED_MODEL" "-ngl" "99")

[ "$NCMOE" -gt 0 ] && CMD+=("-ncmoe" "$NCMOE")
CMD+=("-c" "$CTX" "-b" "$BATCH" "-ub" "$UBATCH")
CMD+=("--cache-type-k" "$CACHE_K" "--cache-type-v" "$CACHE_V")
CMD+=("-fa" "on" "--temp" "$TEMP" "--top-p" "$TOP_P" "--top-k" "$TOP_K" "--min-p" "$MIN_P")
CMD+=("--threads" "8" "--threads-batch" "12")
CMD+=("--cpu-range" "0-7" "--cpu-strict" "1")
CMD+=("--cpu-range-batch" "8-19" "--cpu-strict-batch" "1")
CMD+=("--numa" "isolate" "--prio" "2" "--parallel" "1" "--jinja" "--metrics" "--cache-reuse" "256")

# FIXED: Explicitly pass 'auto' to --reasoning, separate --no-context-shift
CMD+=("--reasoning" "auto" "--no-context-shift")

# MTP + Benchmark Marker
if [ $IS_MTP -eq 1 ]; then
    CMD+=("--spec-type" "draft-mtp" "--spec-draft-n-max" "2" "--spec-draft-p-min" "0.75")
fi

# ROCm/Resource Safety
FREE_RAM_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
if [ "$FREE_RAM_MB" -lt 48000 ]; then
    CMD+=("--mmap")
    echo -e "\n⚠️  System RAM < 48GB. Using --mmap for stability."
else
    CMD+=("--no-mmap" "--mlock")
fi

# --- BOOT ---
echo -e "\n🚀 Launching Executable Vector Payload:\n${CMD[*]}\n"
exec "${CMD[@]}"
