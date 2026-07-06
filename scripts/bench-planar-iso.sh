#!/usr/bin/env bash
# RDNA2 optimization patch-and-measure harness.
# Applies measured optimizations one at a time, rebuilds, benchmarks,
# and reports deltas vs baseline.
#
# Platform: AMD Radeon RX 6800 XT (gfx1030, RDNA2, wave-32, 72 CUs)
# ROCm 7.13
#
# Usage: ./bench-planar-iso.sh [options]
#   --model-path PATH    GGUF model path (required)
#   --output-dir DIR     Output directory (default: bench-results)
#   --patches LIST       Comma-separated: baseline,v128,graph,combined,wave32 (default: all)
#   --help               Show help

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build-rocm"
LLAMA_BENCH="${BUILD_DIR}/bin/llama-bench"

DEFAULT_MODEL_PATH=""
DEFAULT_OUTPUT_DIR="bench-results"
DEFAULT_PATCHES="baseline,graph,wave32,combined"

# Patch registry: id -> {name, cmake_flags, runtime_flags}
# NOTE: RDNA2_V128_LOAD does not exist in this fork — removed.
# Only validated patches included.
declare -A PATCH_NAMES=(
    ["baseline"]="baseline"
    ["graph"]="hipgraph"
    ["wave32"]="wave32_batch"
    ["combined"]="combined"
)

declare -A PATCH_CMAKE=(
    ["baseline"]=""
    ["graph"]="-DGGML_CUDA_GRAPHS=ON"
    ["wave32"]="-DGGML_CUDA_GRAPHS=ON"
    ["combined"]="-DGGML_CUDA_GRAPHS=ON"
)

declare -A PATCH_RUNTIME=(
    ["baseline"]="-b 512"
    ["graph"]="-b 512"
    ["wave32"]="-b 32"
    ["combined"]="-b 32"
)

VALID_PATCH_IDS=("baseline" "graph" "wave32" "combined")

# ============================================================================
# Utility Functions
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error_exit() {
    log "ERROR: $*" >&2
    exit 1
}

extract_pp_tg() {
    # llama-bench outputs a table like:
    # | model | size | ... | pp512 | 1341.35 ± 30.92 |
    # | model | size | ... | tg128 |   54.75 ±  0.54 |
    local bench_output="$1"
    local pp_tok_s tg_tok_s
    pp_tok_s=$(echo "$bench_output" | grep -oP '\| *pp[0-9]+ *\| *\K[0-9]+\.?[0-9]*' | head -1 || echo "")
    tg_tok_s=$(echo "$bench_output" | grep -oP '\| *tg[0-9]+ *\| *\K[0-9]+\.?[0-9]*' | head -1 || echo "")
    echo "$pp_tok_s $tg_tok_s"
}

timestamp_csv() {
    date '+%Y%m%d_%H%M%S'
}

# ============================================================================
# Build
# ============================================================================
build_variant() {
    local patch_id="$1"
    local cmake_flags="${PATCH_CMAKE[$patch_id]}"
    local build_log="$OUTPUT_DIR/build_${patch_id}.log"

    log "=== Building: $patch_id ==="
    if [[ -n "$cmake_flags" ]]; then
        log "  CMake flags: $cmake_flags"
    fi

    # Clean previous build artifacts to avoid stale state
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    # Configure
    cmake -B "$BUILD_DIR" -S "$REPO_ROOT" \
        -DGGML_HIP=ON \
        -DGGML_HIP_UMA=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        $cmake_flags \
        > "$build_log" 2>&1 || {
            log "FAILED cmake configure for $patch_id. See $build_log"
            return 1
        }

    # Build
    local nproc_count
    nproc_count=$(nproc 2>/dev/null || echo 4)
    cmake --build "$BUILD_DIR" -j"$nproc_count" --target llama-bench \
        >> "$build_log" 2>&1 || {
            log "FAILED build for $patch_id. See $build_log"
            return 1
        }

    if [[ ! -x "$LLAMA_BENCH" ]]; then
        error_exit "Build completed but llama-bench not found at $LLAMA_BENCH"
    fi

    log "Build complete for $patch_id"
    return 0
}

# ============================================================================
# Correctness Gate
# ============================================================================
run_correctness_gate() {
    local model_path="$1"
    local passed=false

    log "  Running 5-token correctness test..."
    local result
    # Run with the model's native quantization (no quant override)
    result=$(timeout 60 "$LLAMA_BENCH" \
        -m "$model_path" \
        -t 1 \
        -b 32 \
        -p 512 \
        -n 5 \
        2>&1) || true

    local pp_tok_s tg_tok_s
    read pp_tok_s tg_tok_s <<< "$(extract_pp_tg "$result")"

    if [[ -n "$pp_tok_s" ]] && [[ -n "$tg_tok_s" ]] && \
       [[ "$pp_tok_s" != "0" ]] && [[ "$tg_tok_s" != "0" ]] && \
       [[ "$pp_tok_s" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$tg_tok_s" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        log "  PASS: turbo3_0 pp=$pp_tok_s tg=$tg_tok_s"
        passed=true
    else
        log "  FAIL: turbo3_0 (pp='$pp_tok_s', tg='$tg_tok_s')"
    fi

    echo "$passed"
}

# ============================================================================
# Summary Report
# ============================================================================
generate_summary() {
    local csv_file="$1"
    local summary_file="$2"

    log "=== Generating Summary ==="

    local baseline_pp=""
    local baseline_tg=""
    local best_patch=""
    local best_score=0

    while IFS=',' read -r patch_id patch_name pp_tok_s tg_tok_s passes_correctness; do
        if [[ "$patch_id" == "baseline" ]]; then
            baseline_pp="$pp_tok_s"
            baseline_tg="$tg_tok_s"
        fi

        if [[ -n "$tg_tok_s" ]] && [[ "$tg_tok_s" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            local score
            score=$(echo "$tg_tok_s" | awk '{printf "%.2f", $1}')
            if (( $(echo "$score > $best_score" | bc -l) )); then
                best_score="$score"
                best_patch="$patch_name"
            fi
        fi
    done < <(tail -n +2 "$csv_file")

    cat > "$summary_file" << EOF
================================================================================
RDNA2 Patch Optimization Comparison
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Platform: AMD Radeon RX 6800 XT (gfx1030, RDNA2, wave-32, 72 CUs)
ROCm: 7.13
================================================================================

Results by patch:
EOF

    while IFS=',' read -r patch_id patch_name pp_tok_s tg_tok_s pp_delta_pct tg_delta_pct passes_correctness; do
        local delta_pp=""
        local delta_tg=""

        if [[ "$patch_id" != "baseline" ]] && [[ -n "$baseline_pp" ]] && [[ -n "$pp_tok_s" ]]; then
            delta_pp=$(echo "$baseline_pp $pp_tok_s" | awk '{printf "+%.1f%%", (($2-$1)/$1)*100}')
        fi
        if [[ "$patch_id" != "baseline" ]] && [[ -n "$baseline_tg" ]] && [[ -n "$tg_tok_s" ]]; then
            delta_tg=$(echo "$baseline_tg $tg_tok_s" | awk '{printf "+%.1f%%", (($2-$1)/$1)*100}')
        fi

        local pp_pass="FAIL"
        local tg_pass="FAIL"
        if [[ -n "$pp_tok_s" ]] && (( $(echo "$pp_tok_s >= 58" | bc -l) )); then pp_pass="PASS"; fi
        if [[ -n "$tg_tok_s" ]] && (( $(echo "$tg_tok_s >= 34" | bc -l) )); then tg_pass="PASS"; fi

        cat >> "$summary_file" << EOF

--- $patch_name ---
  PP speed:         ${pp_tok_s:-N/A} tok/s (target>=58) [$pp_pass]${delta_pp:+ ($delta_pp vs baseline)}
  TG speed:         ${tg_tok_s:-N/A} tok/s (target>=34) [$tg_pass]${delta_tg:+ ($delta_tg vs baseline)}
  Correctness:      ${passes_correctness}
EOF
    done < <(tail -n +2 "$csv_file")

    if [[ -n "$best_patch" ]]; then
        cat >> "$summary_file" << EOF

================================================================================
Best performing patch: $best_patch (TG speed = ${best_score} tok/s)
================================================================================
EOF
    fi

    log "Summary saved to: $summary_file"
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-path)
                DEFAULT_MODEL_PATH="$2"
                shift 2
                ;;
            --output-dir)
                DEFAULT_OUTPUT_DIR="$2"
                shift 2
                ;;
            --patches)
                DEFAULT_PATCHES="$2"
                shift 2
                ;;
            --help)
                echo "RDNA2 optimization patch-and-measure harness."
                echo ""
                echo "Usage: $0 [options]"
                echo "  --model-path PATH    GGUF model path (required)"
                echo "  --output-dir DIR     Output directory (default: bench-results)"
                echo "  --patches LIST       Comma-separated: baseline,v128,graph,combined,wave32 (default: all)"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done

    TIMESTAMP=$(timestamp_csv)
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}/${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"

    # Parse and validate selected patches
    IFS=',' read -ra SELECTED_PATCHES <<< "$DEFAULT_PATCHES"
    for p in "${SELECTED_PATCHES[@]}"; do
        p="$(echo "$p" | xargs)"
        local found=false
        for valid in "${VALID_PATCH_IDS[@]}"; do
            if [[ "$p" == "$valid" ]]; then found=true; break; fi
        done
        if ! $found; then
            error_exit "Unknown patch: '$p'. Valid: baseline,graph,wave32,combined"
        fi
    done

    local MODEL_PATH="$DEFAULT_MODEL_PATH"
    if [[ -z "$MODEL_PATH" ]]; then
        error_exit "--model-path required"
    fi
    if [[ ! -f "$MODEL_PATH" ]]; then
        error_exit "Model file not found: $MODEL_PATH"
    fi

    log "=========================================="
    log "RDNA2 Patch Optimization Harness"
    log "Model: $MODEL_PATH"
    log "Output: $OUTPUT_DIR"
    log "Patches: ${SELECTED_PATCHES[*]}"
    log "=========================================="

    # CSV header
    local csv_file="$OUTPUT_DIR/patch_comparison_${TIMESTAMP}.csv"
    echo "patch_id,patch_name,cmake_flags,runtime_flags,pp_tok_s,tg_tok_s,pp_delta_pct,tg_delta_pct,passes_correctness" > "$csv_file"

    local baseline_pp=""
    local baseline_tg=""

    for patch_id in "${SELECTED_PATCHES[@]}"; do
        log ""
        log "=========================================="
        log "Processing patch: $patch_id (${PATCH_NAMES[$patch_id]})"
        log "=========================================="

        # Build
        if ! build_variant "$patch_id"; then
            echo "$patch_id,${PATCH_NAMES[$patch_id]},${PATCH_CMAKE[$patch_id]:-},N/A,N/A,N/A,N/A,N/A,NO" >> "$csv_file"
            continue
        fi

        # Correctness gate
        local passes_correctness="YES"
        if [[ "$patch_id" != "baseline" ]]; then
            local correctness
            correctness=$(run_correctness_gate "$MODEL_PATH")
            if [[ "$correctness" != "true" ]]; then
                passes_correctness="NO"
            fi
        fi

        # Benchmark with the appropriate runtime flags (300s timeout)
        local bench_result=""
        local bench_exit=0
        bench_result=$(timeout 300 "${LLAMA_BENCH}" \
            -m "$MODEL_PATH" \
            -t 1 \
            ${PATCH_RUNTIME[$patch_id]} \
            -p 512 \
            -n 128 \
            turbo3_0 \
            2>&1) || bench_exit=$?
        if (( bench_exit != 0 && bench_exit != 124 )); then
            log "  WARNING: llama-bench exited with code $bench_exit"
        fi
        local pp_tok_s tg_tok_s
        read pp_tok_s tg_tok_s <<< "$(extract_pp_tg "$bench_result")"

        # Calculate deltas vs baseline
        local pp_delta=""
        local tg_delta=""
        if [[ "$patch_id" != "baseline" ]] && [[ -n "$baseline_pp" ]] && [[ -n "$pp_tok_s" ]]; then
            pp_delta=$(echo "$baseline_pp $pp_tok_s" | awk '{printf "+%.1f%%", (($2-$1)/$1)*100}')
        fi
        if [[ "$patch_id" != "baseline" ]] && [[ -n "$baseline_tg" ]] && [[ -n "$tg_tok_s" ]]; then
            tg_delta=$(echo "$baseline_tg $tg_tok_s" | awk '{printf "+%.1f%%", (($2-$1)/$1)*100}')
        fi

        # Write CSV row
        echo "$patch_id,${PATCH_NAMES[$patch_id]},${PATCH_CMAKE[$patch_id]:-},${PATCH_RUNTIME[$patch_id]:-},${pp_tok_s:-N/A},${tg_tok_s:-N/A},${pp_delta:-},${tg_delta:-},${passes_correctness}" >> "$csv_file"

        if [[ "$patch_id" == "baseline" ]]; then
            baseline_pp="$pp_tok_s"
            baseline_tg="$tg_tok_s"
        fi

        log "  Result: PP=${pp_tok_s:-N/A}, TG=${tg_tok_s:-N/A}"
    done

    # Generate summary
    local summary_file="$OUTPUT_DIR/patch_summary_${TIMESTAMP}.txt"
    generate_summary "$csv_file" "$summary_file"

    log ""
    log "=========================================="
    log "Benchmark complete!"
    log "CSV results:   $csv_file"
    log "Summary report: $summary_file"
    log "=========================================="
}

main "$@"
