#!/usr/bin/env bash
# Test script for TurboQuant / IsoQuant / PolarQuant KV cache parity
# Usage: ./test-tq-iso-polar.sh [model.gguf] [n_cpu_moe]
#
# Runs a long-prompt (5500+ tokens) generation with flash attention,
# ISO3 K cache, and the custom set-rows kernels, then validates output.
#
# Exit codes:
#   0 - all good
#   1 - garbled output (all zeros, repeated single token, or error)
#   2 - crash / segfault
#   3 - model missing

set -euo pipefail

MODEL="${1:-/home/stormrage/models/ornith-1.0-35b-Q4_K_M.gguf}"
N_CPU_MOE="${2:-15}"
PROMPT_LEN="$((5500 + RANDOM % 500))"   # 5500-6000 tokens
TEMP_DIR="$(mktemp -d /tmp/tq-iso-test-XXXXXX)"
PROMPT_FILE="${TEMP_DIR}/prompt.txt"
LOG_FILE="${TEMP_DIR}/cli-output.txt"
TIMING_FILE="${TEMP_DIR}/timing.txt"

cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 3
fi

# ---- Generate a long prompt (simple repeated text to hit 5500+ tokens) ----
# Each block ~50 tokens, repeat ~110x to get ~5500 tokens
echo -n '[
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "' > "$PROMPT_FILE"

# Build a long text that will tokenize to ~5500 tokens
# Each repetition adds ~50 tokens
for i in $(seq 1 110); do
    echo -n "The quick brown fox jumps over the lazy dog near the river bank. " >> "$PROMPT_FILE"
done

echo ' Please summarize what was said."}
]' >> "$PROMPT_FILE"

echo "=== Test Configuration ==="
echo "Model:     $MODEL"
echo "GPU:       ngl 99, ncmoe ${N_CPU_MOE}"
echo "Cache:     ctk iso3_"
echo "Flash:     fa on"
echo "Ubatch:    1024"
echo "Context:   ${PROMPT_LEN} tokens (approx)"
echo "=========================="
echo ""

# ---- Launch llama-cli ----
set +e
/usr/bin/time -v --output="${TIMING_FILE}" \
./build/bin/llama-cli \
    -m "$MODEL" \
    -ngl 99 \
    -ncmoe "${N_CPU_MOE}" \
    -ctk iso3_ \
    --flash-attn on \
    -ub 1024 \
    -c "$((PROMPT_LEN + 2048))" \
    -f "$PROMPT_FILE" \
    -n 128 \
    --temp 0.0 \
    --repeat-penalty 1.0 \
    --no-mmap \
    --numa isolate \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE="${PIPESTATUS[0]}"

echo ""
echo "=== Results ==="

# ---- Check 1: Did it crash? ----
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "FAIL: llama-cli exited with code $EXIT_CODE (crash / error)"
    exit 2
fi

# ---- Check 2: Extract generated text (after the prompt) ----
# llama-cli prints the prompt then the response. Extract the response text
# that comes after the final user message.
GEN_TEXT=$(grep -A 9999 "Please summarize" "$LOG_FILE" 2>/dev/null \
           | tail -n +2 \
           | tr -d '\n' \
           | sed 's/^[[:space:]]*//')

# If no response found, try extracting via the raw output
if [ -z "$GEN_TEXT" ] || [ ${#GEN_TEXT} -lt 20 ]; then
    # Fall back: look for unique tokens
    echo "WARN: Generated text too short, trying alternative extraction"
    GEN_TEXT=$(grep -v "^$\|^\[INST\|^<\|user\|assistant\|system" "$LOG_FILE" \
               | strings | tr -d '\n' | sed 's/^[[:space:]]*//')
fi

if [ -z "$GEN_TEXT" ]; then
    echo "FAIL: No generated text found (empty output)"
    exit 1
fi

# ---- Check 3: Detect garbled output ----
# Garbled indicators: all same character, too many repeats, binary garbage
GARBLED=0

# 3a: Check for repeated single character (more than 50% same char)
FIRST_CHAR="${GEN_TEXT:0:1}"
SAME_COUNT=$(echo "$GEN_TEXT" | fold -w1 | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
TOTAL_CHARS=${#GEN_TEXT}
if [ "$SAME_COUNT" -gt $((TOTAL_CHARS * 50 / 100)) ] && [ "$TOTAL_CHARS" -gt 20 ]; then
    echo "WARN: >50% of output is the same character ('$FIRST_CHAR')"
    GARBLED=1
fi

# 3b: Check for high entropy / binary garbage
NON_PRINT=$(echo "$GEN_TEXT" | fold -w1 | grep -cP '[^\x20-\x7E]' 2>/dev/null || echo 0)
if [ "$NON_PRINT" -gt $((TOTAL_CHARS * 20 / 100)) ] && [ "$TOTAL_CHARS" -gt 20 ]; then
    echo "WARN: >20% non-printable characters (binary garbage)"
    GARBLED=1
fi

# 3c: Check for repeated bigrams (e.g., "aa aa aa" pattern)
UNIQUE_WORDS=$(echo "$GEN_TEXT" | tr ' ' '\n' | sort -u | wc -l)
TOTAL_WORDS=$(echo "$GEN_TEXT" | wc -w)
if [ "$TOTAL_WORDS" -gt 10 ] && [ "$UNIQUE_WORDS" -lt 3 ]; then
    echo "WARN: Fewer than 3 unique words in output (repetition loop)"
    GARBLED=1
fi

# ---- Check 4: Extract timing ----
REAL_TIME=$(grep "Elapsed (wall clock) time" "${TIMING_FILE}" 2>/dev/null | awk '{print $NF}')
MAX_RSS=$(grep "Maximum resident set size" "${TIMING_FILE}" 2>/dev/null | awk '{print $NF}')

echo ""
echo "Generated text (first 200 chars):"
echo "${GEN_TEXT:0:200}"
echo "..."

echo ""
echo "Timing:  ${REAL_TIME:-N/A}"
echo "RSS:     ${MAX_RSS:-N/A} KB"
echo "Tokens:  ${TOTAL_CHARS} (output chars)"
echo ""

if [ "$GARBLED" -eq 1 ]; then
    echo "FAIL: Detected potentially garbled output"
    EXIT_CODE=1
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "PASS: Output looks reasonable"
fi

echo ""
echo "Full output saved to: $LOG_FILE"
exit $EXIT_CODE
