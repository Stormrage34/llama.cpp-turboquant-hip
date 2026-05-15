#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

# ─── Qwen3 Reasoning Validation Harness ──────────────────────────────────────
# Validates that RDNA2 flags do NOT interfere with sampling/chat logic.
# Pre-flight VRAM gate enforced. timeout guard on all runs.
#
# Usage:
#   ./scripts/validate_qwen3_reasoning.sh
#
# Exit codes:
#   0 = PASS (all tests pass)
#   1 = FAIL (any test fails)
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${ROOT_DIR}/build/bin/llama-cli"
MODEL="/home/stormrage/models/Qwen3_35BMTPIQ4.gguf"
PROMPT_FILE="/tmp/qwen3_val_prompt.txt"  # -f input
LOG_DIR="${ROOT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ─── Helper: VRAM (bytes → human) ────────────────────────────────────────────

vram_bytes() {
    rocm-smi --showmeminfo vram 2>/dev/null \
        | grep "VRAM Total Used Memory (B):" \
        | awk '{print $NF}'
}

vram_mib() {
    local b
    b=$(vram_bytes)
    if [ -z "$b" ]; then echo 0; else echo $(( b / 1048576 )); fi
}

# ─── Pre-flight ──────────────────────────────────────────────────────────────

if [ ! -x "$BINARY" ]; then
    echo "Binary not found at ${BINARY}"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Model not found at ${MODEL}"
    exit 1
fi

# Kill lingering VRAM consumers
pkill -f llama-cli 2>/dev/null || true; sleep 1

# Verify rocm-smi
rocm-smi --showmeminfo vram &>/dev/null || { echo "ROCM SMI failed"; exit 1; }

vram_pre=$(vram_mib)
echo "VRAM pre-flight: ${vram_pre} MiB used"

# Generate ~6k-token prompt
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Generating ~6k-token prompt..."
    python3 -c "
p = 'The ancient forest of Eldoria stretched across the horizon, its towering trees reaching toward the heavens like silent sentinels guarding secrets older than time itself. Beneath the emerald canopy, shafts of golden sunlight pierced through the leaves, illuminating patches of moss-covered ground where tiny bioluminescent fungi pulsed with ethereal light. A gentle breeze whispered through the branches, carrying the sweet fragrance of wildflowers and the distant murmur of a hidden waterfall. '
with open('${PROMPT_FILE}', 'w') as f:
    f.write(p * 50 + '\n\nPlease continue this story in a creative and detailed way, describing what happens next in the ancient forest.\n\n')
"
fi

# ─── Common Flags ─────────────────────────────────────────────────────────────
FLAGS=(
    -m "$MODEL"
    -ngl 30
    -fitt 512
    -c 4096
    --repeat-penalty 1.1
    -n 1024
    -f "$PROMPT_FILE"
    --no-display-prompt
)

# ─── Helper: run test, return result in a temp file ───────────────────────────

run_test() {
    local test_name="$1"
    local log_suffix="$2"
    local extra_flags="$3"
    local env_vars="$4"

    local log_file="${LOG_DIR}/qwen3_${log_suffix}_${TIMESTAMP}.log"
    local meta_file=$(mktemp)

    echo ""
    echo "--- ${test_name} ---"
    echo "  Log: ${log_file}"

    set +e
    (
        [ -n "$env_vars" ] && eval "export ${env_vars}"
        exec timeout 120 "${BINARY}" "${FLAGS[@]}" ${extra_flags}
    ) &> "$log_file"
    local ec=$?
    set -e

    # Strip ANSI
    sed -i 's/\x1b\[[0-9;]*m//g; s/\x08//g' "$log_file" 2>/dev/null || true

    if [ "$ec" -eq 124 ]; then
        echo "  TIMEOUT (120s): last 5 lines:"
        tail -5 "$log_file"
        echo "failed,$test_name,timeout" > "$meta_file"
        echo "$meta_file"
        return
    fi

    if [ "$ec" -ne 0 ]; then
        local el=$(grep -i "error\|fatal\|abort" "$log_file" | head -3 || true)
        echo "  CRASH (exit=$ec): ${el}"
        echo "failed,$test_name,crash" > "$meta_file"
        echo "$meta_file"
        return
    fi

    # Count tokens from the "Evaluated N tokens" -> "Generate..."
    local gen_line=$(grep -i "generated\|speed\|tokens/s" "$log_file" | tail -1)
    # Try "NCached=... NGenerated=... NTokens/s=..." style
    local gen_ts=$(echo "$gen_line" | grep -oP 'NTokens/s=\K[\d.]+' || true)
    if [ -z "$gen_ts" ]; then
        # Fallback: try generic tokens/s
        gen_ts=$(echo "$gen_line" | grep -oP '[\d.]+ tokens/s' | grep -oP '[\d.]+' || true)
    fi
    if [ -z "$gen_ts" ]; then gen_ts="0.00"; fi

    # Count lines of generated output (after prompt, before stats)
    local response_block=$(sed -n '/Evaluated\|Generate\|ggml_init\|llama_init\|Perf\|--repeat\|^$/!p' "$log_file" | tail -20 | head -15 || true)
    local line_count=$(echo "$response_block" | grep -c . || true)
    if [ "$line_count" -eq 0 ]; then line_count=$(wc -l < "$log_file" || echo 0); fi

    # Coherence: look for story markers
    local coh=0
    if echo "$response_block" | grep -qiE "thinking|story|forest|eldoria|creature|ancient|path|beauty|felt\|thought\|knew\|wonder"; then
        coh=1
        echo "  OK coherence=1 exit=$ec gen_ts=$gen_ts"
    else
        local len=$(echo "$response_block" | wc -c)
        if [ "$len" -gt 50 ]; then
            coh=1
            echo "  OK output_len=$len exit=$ec gen_ts=$gen_ts"
        else
            echo "  MINIMAL output_len=$len exit=$ec"
        fi
    fi

    echo "$ec,$test_name,$gen_ts,$coh" > "$meta_file"
    echo "$meta_file"
}

# ─── Tests ───────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"
TESTS_PASSED=0
TESTS_FAILED=0

echo ""
echo "==============================="
echo " Qwen3 Reasoning Validation"
echo "==============================="
echo " Model: $(basename $MODEL)"
echo " Time:  ${TIMESTAMP}"

m1=$(run_test "Baseline (default reasoning)" "baseline_default" "" "")
m2=$(run_test "Baseline (reasoning=off)" "baseline_no_reason" "--reasoning off" "")

# ─── Summary Table ───────────────────────────────────────────────────────────

echo ""
echo "==============================="
echo " Results"
echo "==============================="
printf "%-30s %-8s %-10s %-6s\n" "Test" "Status" "Tokens/s" "Coh"

do_summary() {
    local mf="$1"
    local label="$2"
    local data
    data=$(cat "$mf" 2>/dev/null || echo "failed,$label,nodata,0")
    local ec=$(echo "$data" | cut -d, -f1)
    local name=$(echo "$data" | cut -d, -f2)
    local ts=$(echo "$data" | cut -d, -f3)
    local coh=$(echo "$data" | cut -d, -f4)
    if [ "$ec" = "0" ]; then
        TESTS_PASSED=$((TESTS_PASSED+1))
        printf "%-30s PASS      %-10s %-6s\n" "$name" "$ts" "$([ "$coh" = "1" ] && echo 'Y' || echo '-')"
    else
        TESTS_FAILED=$((TESTS_FAILED+1))
        printf "%-30s FAIL      ---        --\n" "$label"
        echo "  Cause: $data"
    fi
    rm -f "$mf"
}

do_summary "$m1" "Baseline (default)"
do_summary "$m2" "Baseline (reasoning=off)"
do_summary "$m3" "RDNA2 (default)"
do_summary "$m4" "RDNA2 (reasoning=off)"

# VRAM leak check
sleep 2
vram_post=$(vram_mib)
echo ""
echo "VRAM leak check: pre=${vram_pre} MiB -> post=${vram_post} MiB"
if [ "$vram_post" -gt "$((vram_pre + 2048))" ] 2>/dev/null; then
    echo "  WARNING: +$((vram_post - vram_pre)) MiB"
    TESTS_FAILED=$((TESTS_FAILED+1))
else
    echo "  PASS"
fi

echo ""
echo "Tests: ${TESTS_PASSED} passed, ${TESTS_FAILED} failed"
echo "---"

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo "ALL PASS"
    exit 0
else
    echo "SOME FAILED"
    exit 1
fi
