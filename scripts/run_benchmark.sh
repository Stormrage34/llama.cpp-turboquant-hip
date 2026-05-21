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

# ─── CR-008 Benchmark Template ────────────────────────────────────
# SYMMETRICAL BATCH: Always set -b equal to -ub (CR-008 Council Verdict)
#   Dense: -b 128 -ub 128 -r 2  (Llama 8B peak at pp256: 1190 t/s)
#   MoE:   -b 64  -ub 64  -r 2  (Qwen35 peak at pp256: 146 t/s)
#   ROCm:  7.2.3 Stable (/opt/rocm) — not 7.13 nightly
#   rocprofv3: Pure PMC only, NO --hip-trace, NO --kernel-trace

set -euo pipefail

# ─── VRAM Safety Gate ────────────────────────────────────────────
# Enforces the 15.5GB redline (Chief Engineer §17). Called before
# any GPU inference to prevent OOM crashes.
check_vram_budget() {
    local model_vram_mib=$1
    local context_mib=$2
    local overhead_mib=${3:-512}

    local total_vram_mib=$(rocm-smi --showmeminfo vram | grep -i 'Total:' | awk '{print $2}')
    local used_vram_mib=$(rocm-smi --showmeminfo vram | grep -i 'Used:' | awk '{print $2}')
    local free_vram_mib=$((total_vram_mib - used_vram_mib))
    local estimated_required=$((model_vram_mib + context_mib + overhead_mib))
    local redline_limit=15872

    echo "--- VRAM PRE-FLIGHT AUDIT ---"
    echo "Hardware Total VRAM : ${total_vram_mib} MiB"
    echo "Currently Free VRAM : ${free_vram_mib} MiB"
    echo "Estimated Workload   : ${estimated_required} MiB"

    if [ "$estimated_required" -gt "$redline_limit" ]; then
        echo "ERROR: Estimated VRAM workload exceeds 15.5GB Safety Redline." >&2
        exit 1
    fi

    if [ "$estimated_required" -gt "$free_vram_mib" ]; then
        echo "ERROR: Insufficient free VRAM ($free_vram_mib < $estimated_required)." >&2
        exit 1
    fi
    echo "VRAM Budget Cleared."
}

# ─── Thermal & Power Audit ───────────────────────────────────────
# Captures GPU thermal state before/after inference runs.
# Required by Chief Engineer §13 for diagnosing speed variance.
log_thermal_state() {
    local label=$1
    local logfile=${2:-telemetry.log}
    echo "=== THERMAL & POWER AUDIT (${label}) ===" >> "$logfile"
    rocm-smi --showtemp --showpower >> "$logfile" 2>/dev/null
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT_ROOT}/build/bin/llama-cli"

# ─── Parse Arguments ───────────────────────────────────────────────────────────
MODEL=""
RUNS=5
TOKEN_LIMIT="${TOKEN_COUNT:-1000}"
CACHE_CONFIGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [model.gguf] [-r RUNS] [cache_k,cache_v...]"
            echo ""
            echo "  -r, --runs RUNS   Number of benchmark runs per config (default: 5)"
            echo "  cache_k,cache_v   Cache configs to test, e.g. q8_0,turbo3"
            echo ""
            echo "Default cache configs:"
            echo "  q8_0,turbo3  (our asymmetric turbo)"
            echo "  turbo3,turbo3 (symmetric turbo)"
            echo "  q8_0,q8_0    (original symmetric)"
            echo "  q8_0,q4_0    (original asymmetric)"
            echo "  q4_0,q4_0    (original aggressive)"
            exit 0
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            else
                CACHE_CONFIGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    MODEL="${PROJECT_ROOT}/../models/Qwen3_35BMTPIQ4.gguf"
fi

# Default cache configs to test
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
  echo "  Config: -ngl 99 --n-cpu-moe 41 -c 32768 -fa 1 -st -n ${TOKEN_LIMIT}"
echo "================================================================"
echo ""

# ─── GPU Gate ────────────────────────────────────────────────────
source "${SCRIPT_DIR}/gpu_failback.sh"
if ! gpu_ensure_free; then
    exit 1
fi
echo "GPU Gate: CLEAR"

# Thermal snapshot before benchmark
log_thermal_state "pre-benchmark" "${RESULTS_FILE}"

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

trap 'gpu_release' EXIT

# ─── Stale Binary Detection ─────────────────────────────────────────────────────
# If a shared lib is newer than the binary that links it, a partial rebuild
# was done (common cause of SIGSEGV on HIP backend init).
check_stale_binaries() {
    local stale_found=0
    local bins=("llama-server" "llama-cli" "llama-bench")
    local libs=("libggml-hip.so.0" "libggml-cpu.so.0" "libggml-base.so.0" "libllama.so.0" "libllama-common.so.0")
    local bin_dir="$(dirname "$BINARY")"
    for bin_name in "${bins[@]}"; do
        local bin_path="${bin_dir}/${bin_name}"
        [[ -f "$bin_path" ]] || continue
        for lib_name in "${libs[@]}"; do
            local lib_path="${bin_dir}/${lib_name}"
            [[ -f "$lib_path" ]] || continue
            if [[ "$lib_path" -nt "$bin_path" ]]; then
                echo "⚠ STALE BINARY: ${bin_name} is older than ${lib_name}"
                echo "  → Partial rebuild detected. Full rebuild required."
                stale_found=1
            fi
        done
    done
    return $stale_found
}

# Stale check for alternate build directories (build-swizzle etc.)
check_alt_build_stale() {
    local alt_dir="${1}"
    local stale_found=0
    [[ -d "$alt_dir/bin" ]] || return 0
    local bins=("llama-server" "llama-cli" "llama-bench")
    local libs=("libggml-hip.so.0" "libggml-cpu.so.0" "libggml-base.so.0" "libllama.so.0" "libllama-common.so.0")
    for bin_name in "${bins[@]}"; do
        local bin_path="${alt_dir}/bin/${bin_name}"
        [[ -f "$bin_path" ]] || continue
        for lib_name in "${libs[@]}"; do
            local lib_path="${alt_dir}/bin/${lib_name}"
            [[ -f "$lib_path" ]] || continue
            if [[ "$lib_path" -nt "$bin_path" ]]; then
                echo "⚠ STALE BINARY in $(basename "$alt_dir"): ${bin_name} is older than ${lib_name}"
                echo "  → Partial rebuild detected. Full rebuild required."
                stale_found=1
            fi
        done
    done
    return $stale_found
}
# Note: build-swizzle was deleted as part of HARDWARE_TARGET.md cleanup
# check_alt_build_stale "${PROJECT_ROOT}/build-swizzle" || true
check_stale_binaries && {
    echo "  [stale binary check: clean]"
} || {
    echo "⚠ Stale binaries detected — SIGSEGV risk during inference."
}

RESULTS_FILE="${PROJECT_ROOT}/benchmarks/raw/benchmark_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$RESULTS_FILE")"

RAW_DIR="${PROJECT_ROOT}/benchmarks/raw/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RAW_DIR"

# Statistical helpers (use awk via stdin for robust math with sqrt)
mean() {
    [ $# -eq 0 ] && echo "0" && return
    printf '%s\n' "$@" | awk '{s+=$1} END {printf "%.2f", s/NR}' 2>/dev/null || echo "0"
}

stddev() {
    [ $# -le 1 ] && echo "0" && return
    printf '%s\n' "$@" | awk '{
        s+=$1; sq+=$1*$1
    } END {
        m=s/NR;
        var=sq/NR - m*m;
        if (var < 0) var = 0;
        printf "%.2f", sqrt(var * NR / (NR - 1))
    }' 2>/dev/null || echo "0"
}

header="Config | Prompt | n | Prompt t/s (mean±std) | Decode t/s (mean±std) | VRAM MiB (mean)"
echo "$header" | tee -a "$RESULTS_FILE"
echo "------ | ------ | - | --------------------- | ---------------------- | ----------------" | tee -a "$RESULTS_FILE"

for cache_config in "${CACHE_CONFIGS[@]}"; do
    CTK=$(echo "$cache_config" | cut -d, -f1)
    CTV=$(echo "$cache_config" | cut -d, -f2)
    
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        pname="${PROMPT_NAMES[$i]}"
        
        echo -n "  [$cache_config] [$pname] (${RUNS}x) ... "
        
        # Collect results across runs
        prompt_ts=()
        gen_ts=()
        vrams=()
        run_log=""
        
        for run in $(seq 1 "$RUNS"); do
            gpu_acquire > /dev/null 2>&1 || true
            sleep 4
            
            output=$(timeout "${TIMEOUT_SEC:-600}" "$BINARY" \
                -m "$MODEL" \
                -ngl 99 --n-cpu-moe 41 \
                -c 32768 --cache-type-k "$CTK" --cache-type-v "$CTV" \
                -fa 1 --spec-type mtp --spec-draft-n-max 2 \
                -st -n "${TOKEN_LIMIT}" -p "$prompt" 2>&1 || true)
            
            # Normalize European locale (comma decimal → dot)
            output_lc=$(echo "$output" | sed 's/,/./g')
            
            pt=$(echo "$output_lc" | grep 'Prompt:' | sed 's/.*Prompt:[[:space:]]*\([0-9.]*\)[[:space:]]*t\/s.*/\1/' || echo "N/A")
            gt=$(echo "$output_lc" | grep 'Generation:' | sed 's/.*Generation:[[:space:]]*\([0-9.]*\)[[:space:]]*t\/s.*/\1/' || echo "N/A")
            vr=$(echo "$output" | grep 'common_memory_breakdown_print' | sed 's/.*( \([0-9]*\) =.*/\1/' || echo "N/A")
            
            # Store numeric values
            if [ "$pt" != "N/A" ]; then
                prompt_ts+=("$pt")
            fi
            if [ "$gt" != "N/A" ]; then
                gen_ts+=("$gt")
            fi
            if [ "$vr" != "N/A" ]; then
                vrams+=("$vr")
            fi
            
            # Save raw output
            echo "=== Run ${run}: ${cache_config} / ${pname} ===" >> "${RAW_DIR}/run_${cache_config//,/_}_${pname}.txt"
            echo "Prompt t/s: ${pt}" >> "${RAW_DIR}/run_${cache_config//,/_}_${pname}.txt"
            echo "Gen t/s: ${gt}" >> "${RAW_DIR}/run_${cache_config//,/_}_${pname}.txt"
            echo "VRAM: ${vr} MiB" >> "${RAW_DIR}/run_${cache_config//,/_}_${pname}.txt"
            echo "" >> "${RAW_DIR}/run_${cache_config//,/_}_${pname}.txt"
            
            sleep 2
        done
        # Release GPU between prompts/configs
        gpu_release > /dev/null 2>&1 || true
        
        # Compute statistics
        n_prompt=${#prompt_ts[@]}
        n_gen=${#gen_ts[@]}
        
        if [ "$n_prompt" -gt 0 ]; then
            prompt_mean=$(mean "${prompt_ts[@]}")
            prompt_sd=$(stddev "${prompt_ts[@]}")
        else
            prompt_mean="N/A"
            prompt_sd=""
        fi
        
        if [ "$n_gen" -gt 0 ]; then
            gen_mean=$(mean "${gen_ts[@]}")
            gen_sd=$(stddev "${gen_ts[@]}")
        else
            gen_mean="N/A"
            gen_sd=""
        fi
        
        if [ ${#vrams[@]} -gt 0 ]; then
            vram_mean=$(mean "${vrams[@]}")
        else
            vram_mean="N/A"
        fi
        
        # Report
        if [ "$prompt_mean" != "N/A" ] && [ "$gen_mean" != "N/A" ]; then
            echo "${cache_config} | ${pname} | ${RUNS} | ${prompt_mean}±${prompt_sd} | ${gen_mean}±${gen_sd} | ${vram_mean}" | tee -a "$RESULTS_FILE"
        else
            echo "${cache_config} | ${pname} | ${RUNS} | ${prompt_mean} | ${gen_mean} | ${vram_mean}" | tee -a "$RESULTS_FILE"
        fi
    done
    echo "" | tee -a "$RESULTS_FILE"
done

# Thermal snapshot after benchmark
log_thermal_state "post-benchmark" "$RESULTS_FILE"

echo ""
echo "✅ Benchmark complete. Results saved to: $RESULTS_FILE"
echo "   Raw run data saved to: $RAW_DIR"
echo ""
echo "=== Summary (mean decode t/s) ==="
tail -n +3 "$RESULTS_FILE" | grep -v "^-" | sed 's/|/:/g' | while IFS=: read -r config pname n pt gen rest; do
    echo "  $config / $pname: decode=${gen}t/s"
done
