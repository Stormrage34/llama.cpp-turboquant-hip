#!/bin/bash
set -euo pipefail

# =============================================================================
# LLAMA-SERVER DUAL-BUILD AUTOMATION LAUNCHER (Dynamic VRAM-Aware)
# RX 6800 XT (16GB VRAM) Optimized | May 2026 Mainline Compliant
# =============================================================================

# --- Environment Setup ---
export ROCM_PATH="/home/stormrage/rocm-7.13-nightly"
export HIP_PATH="/home/stormrage/rocm-7.13-nightly"
MODEL_DIR="/home/stormrage/models"

# --- Build Environment Detection ---
echo "========================================="
echo "   Select Llama-Server Build Engine"
echo "========================================="
echo "1) Original Main Branch (/home/stormrage/llama.cpp/build/bin/)"
echo "2) TurboQuant-HIP Custom (/home/stormrage/llama.cpp-turboquant-hip/build/bin/)"
read -rp "Enter choice [1-2, Default: 1]: " BUILD_CHOICE

if [ "$BUILD_CHOICE" = "2" ]; then
    BINARY_DIR="/home/stormrage/llama.cpp-turboquant-hip/build/bin"
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

# --- ARCHITECTURE DISCOVERY (GGUF KV Parse) ---
echo -e "\n🔍 Discovering model architecture..."

ARCH_DISCOVERY_OUTPUT=$(timeout 15 "$BINARY_DIR/llama-cli" \
    -m "$SELECTED_MODEL" \
    -ngl 0 \
    -c 64 \
    -n 0 \
    -f /dev/null \
    --verbose 2>&1 | head -200)

# Parse architecture
LOWER_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
IS_MOE=0
IS_MTP=0
IS_GEMMA=0
IS_SLIDING_WINDOW=0
BLOCK_COUNT=0
FULL_ATTENTION_INTERVAL=0
EXPERT_COUNT=0
EXPERT_USED=0

# Detect architecture from KV dump
if echo "$ARCH_DISCOVERY_OUTPUT" | grep -q "qwen35moe"; then
    IS_MOE=1
    echo "-> Detected: Qwen3.6-MoE architecture"
    
    # Extract block_count
    BLOCK_COUNT=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "qwen35moe.block_count" | grep -oP '\d+$' || echo "0")
    
    # Extract full_attention_interval
    FULL_ATTENTION_INTERVAL=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "qwen35moe.full_attention_interval" | grep -oP '\d+$' || echo "0")
    
    # Extract expert counts
    EXPERT_COUNT=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "qwen35moe.expert_count" | grep -oP '\d+$' || echo "0")
    EXPERT_USED=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "qwen35moe.expert_used_count" | grep -oP '\d+$' || echo "0")
    
    echo "   block_count=$BLOCK_COUNT full_attention_interval=$FULL_ATTENTION_INTERVAL"
    echo "   expert_count=$EXPERT_COUNT expert_used=$EXPERT_USED"
    
elif echo "$ARCH_DISCOVERY_OUTPUT" | grep -q "gemma4"; then
    IS_MOE=1
    IS_GEMMA=1
    IS_SLIDING_WINDOW=1
    echo "-> Detected: Gemma 4 architecture (hybrid dense+MoE)"
    
    # Gemma 4: 6-layer cycle (5 MoE + 1 dense)
    BLOCK_COUNT=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "gemma4.block_count" | grep -oP '\d+$' || echo "0")
    FULL_ATTENTION_INTERVAL=6
    
    EXPERT_COUNT=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "gemma4.expert_count" | grep -oP '\d+$' || echo "0")
    EXPERT_USED=$(echo "$ARCH_DISCOVERY_OUTPUT" | grep "gemma4.expert_used_count" | grep -oP '\d+$' || echo "0")
    
    echo "   block_count=$BLOCK_COUNT cycle=6 (5 MoE + 1 dense)"
    echo "   expert_count=$EXPERT_COUNT expert_used=$EXPERT_USED"
    
elif echo "$LOWER_NAME" | grep -qi "mixtral\|dbrx\|qwen3.*moe\|deepseek.*moe"; then
    IS_MOE=1
    echo "-> Detected: MoE architecture (name-based)"
    BLOCK_COUNT=32  # Default assumption
elif echo "$LOWER_NAME" | grep -qi "mtp\|multistep"; then
    IS_MTP=1
    echo "-> Detected: MTP (Multi-Token Prediction) architecture"
elif echo "$LOWER_NAME" | grep -qi "gemma"; then
    IS_GEMMA=1
    if echo "$LOWER_NAME" | grep -qi "gemma-4\|gemma-3"; then
        IS_SLIDING_WINDOW=1
        echo "-> Detected: Gemma 3/4 IT architecture with Sliding Window"
    else
        IS_SLIDING_WINDOW=1
        echo "-> Detected: Gemma architecture with Sliding Window"
    fi
fi

# --- NCMOE CALCULATION (Architecture-Aware) ---
# For MoE models: ncmoe = number of MoE layers to offload to CPU
# Formula depends on architecture:
#   Qwen3.6-MoE: ncmoe = block_count - (block_count / full_attention_interval)
#   Standard MoE (Mixtral): ncmoe = block_count (all layers are MoE)

NCMOE=0
if [ "$IS_MOE" -eq 1 ]; then
    if [ "$FULL_ATTENTION_INTERVAL" -gt 0 ] && [ "$BLOCK_COUNT" -gt 0 ]; then
        # Hybrid dense+MoE: dense layers = block_count / cycle_interval
        DENSE_LAYERS=$(( BLOCK_COUNT / FULL_ATTENTION_INTERVAL ))
        NCMOE=$(( BLOCK_COUNT - DENSE_LAYERS ))
        echo "-> Hybrid MoE: $BLOCK_COUNT layers, $DENSE_LAYERS dense, $NCMOE MoE → ncmoe=$NCMOE"
    else
        # Standard MoE: all layers are MoE
        NCMOE=$BLOCK_COUNT
        echo "-> Standard MoE: all $NCMOE layers offloaded"
    fi
    
    # Safety caps
    [ "$NCMOE" -lt 4 ] && NCMOE=4
    [ "$NCMOE" -gt 64 ] && NCMOE=64
else
    echo "-> Dense model: ncmoe=0 (disabled)"
fi

# --- SIZE-AWARE NGL CALCULATION ---
MODEL_BYTES=$(stat -c %s "$SELECTED_MODEL")
SIZE_GB=$(( MODEL_BYTES / 1073741824 ))

# VRAM budget: ~14.5GB for MoE (experts on CPU), ~13GB for dense
if [ "$IS_MOE" -eq 1 ]; then
    VRAM_SAFE_GB=14.5
else
    VRAM_SAFE_GB=13.0
fi

# Estimate total layers from architecture
EST_LAYERS=$BLOCK_COUNT
if [ "$EST_LAYERS" -eq 0 ]; then
    # Fallback: estimate from model size
    [[ "$LOWER_NAME" =~ (8b|7b) ]] && EST_LAYERS=32
    [[ "$LOWER_NAME" =~ (4b|3b|2b) ]] && EST_LAYERS=24
    [[ "$LOWER_NAME" =~ (26b|27b) ]] && EST_LAYERS=44
    [ "$EST_LAYERS" -eq 0 ] && EST_LAYERS=48  # Default for 35B+
fi

# Calculate safe GPU layer offload
NGL=$(awk "BEGIN {
    val = int($VRAM_SAFE_GB / $SIZE_GB * $EST_LAYERS);
    if (val > $EST_LAYERS) val = $EST_LAYERS;
    if (val < 1) val = 1;
    print val
}")
echo "-> Calculated safe GPU layers (-ngl $NGL) for ${SIZE_GB}GB model on 16GB VRAM"

# --- CONTEXT SIZE CONFIGURATION ---
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
echo "Architecture:   MTP=$IS_MTP | MoE=$IS_MOE | Gemma=$IS_GEMMA | SW=$IS_SLIDING_WINDOW"
if [ "$IS_MOE" -eq 1 ]; then
    echo "MoE Layout:     ncmoe=$NCMOE (offload $NCMOE expert layers to CPU)"
    [ "$EXPERT_COUNT" -gt 0 ] && echo "                expert_count=$EXPERT_COUNT expert_used=$EXPERT_USED"
fi
echo "Memory Layout:  GPU Layers=$NGL | Batch=$BATCH | U-Batch=$UBATCH"
echo "----------------------------------------------------------------"
read -rp "Press [Enter] to spin up the production llama-server engine..."

# --- COMMAND CONSTRUCTION ---
CMD=("$BINARY_DIR/llama-server" "-m" "$SELECTED_MODEL")

# Dynamic GPU offload
CMD+=("-ngl" "$NGL")

[ "$NCMOE" -gt 0 ] && CMD+=("-ncmoe" "$NCMOE")
CMD+=("-c" "$CTX" "-b" "$BATCH" "-ub" "$UBATCH")
CMD+=("--cache-type-k" "$CACHE_K" "--cache-type-v" "$CACHE_V")
CMD+=("-fa" "on" "--temp" "$TEMP" "--top-p" "$TOP_P" "--top-k" "$TOP_K" "--min-p" "$MIN_P")
CMD+=("--threads" "8" "--threads-batch" "12")
CMD+=("--cpu-range" "0-7" "--cpu-strict" "1")
CMD+=("--cpu-range-batch" "8-19" "--cpu-strict-batch" "1")
CMD+=("--numa" "isolate" "--prio" "2" "--parallel" "1" "--jinja" "--metrics" "--cache-reuse" "256")

# Reasoning & context management
CMD+=("--reasoning" "auto" "--no-context-shift")

# MTP support
if [ "$IS_MTP" -eq 1 ]; then
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
