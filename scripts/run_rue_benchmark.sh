#!/bin/bash
# run_rue_benchmark.sh — Resource Utilization Efficiency (RUE) Benchmark
#
# Compares 3 MoE offloading configurations for throughput-per-VRAM efficiency:
#   A: --n-cpu-moe 41            (all-CPU expert floor)
#   B: --n-cpu-moe-range 10-20   (middle-layer structural offload)
#   C: --n-cpu-moe 41            (standard layer-wise shift, reference)
#
# Outputs: pp512 (t/s), tg128 (t/s), VRAM (GiB), RUE (t/s per GiB)
#
# Usage: ./scripts/run_rue_benchmark.sh [model.gguf]
#
# Requires:
#   - root for cache dropping (auto-detected, skipped if unavailable)
#   - chrt + taskset for process pinning (auto-detected, skipped if unavailable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT_ROOT}/build/bin/llama-bench"

# ─── Parse Arguments ───────────────────────────────────────────────────────────
MODEL="${1:-${HOME}/models/Qwen3_35BMTPIQ4.gguf}"

# ─── Configurations ────────────────────────────────────────────────────────────
# Each config: "name|extra_flags"
CONFIGS=(
    "A|--n-cpu-moe 41"
    "B|--n-cpu-moe-range 10-20"
    "C|--n-cpu-moe 41"
)
COMMON_FLAGS=(
    -m "$MODEL"
    -t 8
    -ngl 99
    -mmp 0       # --no-mmap equivalent for llama-bench
    -p 512
    -n 128
    -b 64
    -r 5
    -o md
)

TIMEOUT_SEC=120

# ─── Utility: VRAM measurement ─────────────────────────────────────────────────
source "${SCRIPT_DIR}/gpu_failback.sh"

_vram_gib() {
    # Try rocm-smi first (more reliable than sysfs)
    local vram_pct
    vram_pct=$(rocm-smi --showmemuse 2>/dev/null | grep "GPU Memory Allocated (VRAM%)" | awk '{print $NF}' | head -1)
    if [[ -n "$vram_pct" && "$vram_pct" =~ ^[0-9]+$ ]]; then
        # RX 6800 XT has 16384 MiB VRAM
        local vram_mib=$(( 16384 * vram_pct / 100 ))
        awk "BEGIN { printf \"%.1f\", $vram_mib / 1024 }"
        return 0
    fi
    
    # Fallback to sysfs
    local used
    used=$(_gpu_vram_used) 2>/dev/null || { echo "0"; return 1; }
    if [[ -z "$used" || "$used" -lt 1 ]]; then
        echo "0"
        return 1
    fi
    # Convert bytes to GiB with 1 decimal
    awk "BEGIN { printf \"%.1f\", $used / (1024*1024*1024) }"
}

# ─── Utility: check if running as root ─────────────────────────────────────────
_is_root() {
    [[ "$(id -u)" -eq 0 ]]
}

# ─── Utility: parse llama-bench markdown output ────────────────────────────────
# Extracts the numeric value from a test row like:
#   | ... | pp512 | 302.95 ± 0.69 |
# Usage: _parse_bench_output "$output" "pp512"
_parse_bench_output() {
    local output="$1" testname="$2"
    # Match: | ... | pp512 | VALUE ± STDDEV |
    echo "$output" | grep -oP '\|\s+'"${testname}"'\s+\|\s+\K[\d.]+' 2>/dev/null | tail -1
}

# ─── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo "❌ Binary not found: $BINARY"
    echo "   Build first: cmake --build build --target llama-bench -- -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "   Usage: $0 [path/to/model.gguf]"
    exit 1
fi

# Detect available capabilities
CAN_PIN=false
if command -v chrt &>/dev/null && command -v taskset &>/dev/null; then
    CAN_PIN=true
fi

CAN_DROP_CACHE=false
if _is_root; then
    CAN_DROP_CACHE=true
fi

# ─── Setup ─────────────────────────────────────────────────────────────────────
gpu_failback_trap

echo "================================================================"
echo "  RUE (Resource Utilization Efficiency) Benchmark"
echo "  Model: $(basename "$MODEL")"
echo "  GPU:   AMD Radeon RX 6800 XT (gfx1030)"
echo "  Flags: -t 8 -ngl 99 -p 512 -n 128 -b 64 -r 5 (-mmp 0)"
echo "================================================================"
echo ""
echo "Capabilities:"
echo "  Process pinning (chrt+taskset): $([ "$CAN_PIN" = true ] && echo "yes" || echo "no")"
echo "  Cache dropping (root):          $([ "$CAN_DROP_CACHE" = true ] && echo "yes" || echo "no")"
echo ""

# ─── Main Benchmark Loop ───────────────────────────────────────────────────────
RESULTS=()

echo "Running 3 configurations (sequential, n=1 GPU)..."
echo ""

for config_entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name extra_flags <<< "$config_entry"

    echo "----------------------------------------------------------------"
    echo "  Config $name: $extra_flags"
    echo "----------------------------------------------------------------"

    # Build argument array for this config
    CMD_ARGS=("${COMMON_FLAGS[@]}")
    # shellcheck disable=SC2206
    CMD_ARGS+=($extra_flags)

    # Drop caches (if root)
    if [ "$CAN_DROP_CACHE" = true ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        echo "  [caches dropped]"
    fi

    # Acquire GPU
    echo -n "  Acquiring GPU... "
    gpu_acquire > /dev/null 2>&1 || true
    sleep 2
    echo "done"

    # Measure VRAM before
    VRAM_BEFORE=$(_vram_gib) || VRAM_BEFORE="0"
    echo "  VRAM before: ${VRAM_BEFORE} GiB"

    # Run benchmark with timeout
    echo -n "  Benchmarking (timeout ${TIMEOUT_SEC}s)... "

    # Build command wrapper for pinning
    RUN_CMD=()
    if [ "$CAN_PIN" = true ]; then
        RUN_CMD=(chrt -f 99 taskset -c 0-7)
    fi
    RUN_CMD+=("$BINARY" "${CMD_ARGS[@]}")

    # Execute with timeout
    set +e
    OUTPUT=$(timeout "$TIMEOUT_SEC" "${RUN_CMD[@]}" 2>&1) || {
        exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            echo "TIMEOUT (>${TIMEOUT_SEC}s)"
        else
            echo "FAILED (exit=$exit_code)"
        fi
        echo "  ⚠ Skipping config $name"
        RESULTS+=("$name|FAILED|FAILED|0|0")
        gpu_release > /dev/null 2>&1 || true
        set -e
        continue
    }
    set -euo pipefail
    echo "done"

    # Measure VRAM after
    VRAM_AFTER=$(_vram_gib) || VRAM_AFTER="0"
    echo "  VRAM after:  ${VRAM_AFTER} GiB"

    # Extract VRAM used (use max of before/after as the steady-state)
    VRAM_USED=$(awk "BEGIN { v=$VRAM_AFTER; if (v < $VRAM_BEFORE) v=$VRAM_BEFORE; printf \"%.1f\", v }")

    # Parse results
    PP512=$(_parse_bench_output "$OUTPUT" "pp512")
    TG128=$(_parse_bench_output "$OUTPUT" "tg128")

    # Validate parsing
    if [[ -z "$PP512" || "$PP512" == "0" ]]; then
        echo "  ⚠ Failed to parse pp512 from output"
        echo "  --- raw output snippet ---"
        echo "$OUTPUT" | head -20
        echo "  ---"
        PP512="N/A"
    fi
    if [[ -z "$TG128" || "$TG128" == "0" ]]; then
        echo "  ⚠ Failed to parse tg128 from output"
        TG128="N/A"
    fi

    # Calculate RUE (t/s per GiB) using tg128 / vram
    if [[ "$TG128" != "N/A" && "$VRAM_USED" != "0" ]]; then
        RUE=$(awk "BEGIN { printf \"%.2f\", $TG128 / $VRAM_USED }")
    else
        RUE="N/A"
    fi

    echo "  pp512:      ${PP512} t/s"
    echo "  tg128:      ${TG128} t/s"
    echo "  VRAM used:  ${VRAM_USED} GiB"
    echo "  RUE:        ${RUE} t/s/GiB"
    echo ""

    RESULTS+=("$name|$PP512|$TG128|$VRAM_USED|$RUE")

    # Release GPU between configs
    gpu_release > /dev/null 2>&1 || true
    sleep 2
done

# ─── Results Table ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  RUE Benchmark Results"
echo "================================================================"
printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "Config" "pp512 (t/s)" "tg128 (t/s)" "VRAM (GiB)" "RUE (t/s/GiB)"
printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "------" "-----------" "-----------" "----------" "-------------"
for result in "${RESULTS[@]}"; do
    IFS='|' read -r name pp tg vram rue <<< "$result"
    printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "$name" "$pp" "$tg" "$vram" "$rue"
done
echo "================================================================"

# Also save to file
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/rue"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/rue_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "RUE Benchmark Results"
    echo "Date: $(date)"
    echo "Model: $(basename "$MODEL")"
    echo "Build: $("${BINARY}" --help 2>&1 | grep "build:" | head -1 || echo "unknown")"
    echo ""
    printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "Config" "pp512 (t/s)" "tg128 (t/s)" "VRAM (GiB)" "RUE (t/s/GiB)"
    printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "------" "-----------" "-----------" "----------" "-------------"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name pp tg vram rue <<< "$result"
        printf "| %-6s | %-11s | %-11s | %-10s | %-13s |\n" "$name" "$pp" "$tg" "$vram" "$rue"
    done
} > "$RESULTS_FILE"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Restore server if needed
gpu_release > /dev/null 2>&1 || true
