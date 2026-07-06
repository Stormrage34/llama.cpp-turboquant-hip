#!/usr/bin/env bash
#
# debug_turbo.sh - Master debug script for KV cache quantization garbling
#
# Tests each -ctk/-ctv combination with a fixed prompt and checks for:
# 1. Output quality (coherent text vs garbled/random bytes)
# 2. Token generation consistency across runs
# 3. Reasoning content integrity
# 4. Cache hit behavior
#
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
SERVER_BIN="./build/bin/llama-server"
MODEL="/home/stormrage/models/deepreinforce-ai_Ornith-1.0-35B-IQ4_XS.gguf"
PORT=8099
BASE_ARGS="-ngl 99 -c 166000 -fa on -t 8 -tb 12 --numa isolate -np 1 --parallel 1"
PROMPT="Write a haiku about the ocean."
MAX_TOKENS=200
TEMP=0.0

# KV cache combos to test: "label ctk ctv"
COMBOS=(
    "f16_f16       f16     f16"
    "q8_q8         q8_0    q8_0"
    "q8_turbo3     q8_0    turbo3_0"
    "turbo3_q8     turbo3_0 q8_0"
    "turbo3_turbo3  turbo3_0 turbo3_0"
)

RESULTS_DIR="/tmp/debug_turbo_results"
mkdir -p "$RESULTS_DIR"

# ── Functions ──────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_server() {
    local pid
    pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
    fi
}

wait_server() {
    local max_wait=120
    local elapsed=0
    while ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $max_wait ]]; then
            log "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    log "Server ready after ${elapsed}s"
}

test_prompt() {
    local run_label="$1"
    local out_file="$RESULTS_DIR/${run_label}.json"

    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': 'test',
    'messages': [{'role': 'user', 'content': $(python3 -c "import json; print(json.dumps('$PROMPT'))")}],
    'max_tokens': $MAX_TOKENS,
    'temperature': $TEMP
}))
")

    curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$payload" > "$out_file" 2>&1

    python3 -c "
import json, sys, re

try:
    with open('$out_file') as f:
        d = json.load(f)
except Exception as e:
    print(f'PARSE ERROR: {e}')
    sys.exit(1)

msg = d.get('choices', [{}])[0].get('message', {})
content = msg.get('content', '')
reasoning = msg.get('reasoning_content', '')
finish = d['choices'][0].get('finish_reason', 'unknown')
usage = d.get('usage', {})

print(f'  finish_reason: {finish}')
print(f'  prompt_tokens: {usage.get(\"prompt_tokens\", 0)}')
print(f'  completion_tokens: {usage.get(\"completion_tokens\", 0)}')
print(f'  cached_tokens: {usage.get(\"prompt_tokens_details\", {}).get(\"cached_tokens\", 0)}')
print()

# Check for garbling
issues = []

# 1. Empty content (possible garble - model got confused)
if not content.strip() and finish == 'stop':
    issues.append('EMPTY_CONTENT: content is empty but finish=stop')

# 2. Non-UTF8 or replacement characters
if '\ufffd' in content or '\ufffd' in reasoning:
    issues.append('REPLACEMENT_CHARS: contains U+FFFD (replacement character)')

# 3. Excessive repetition
if len(content) > 50:
    for patlen in [3, 5, 10]:
        for i in range(len(content) - patlen * 5):
            pat = content[i:i+patlen]
            if pat * 5 in content:
                issues.append(f'REPETITION: {patlen}-char pattern repeated 5+ times')
                break

# 4. Gibberish detection: ratio of special/punctuation chars to alphanum
alphanum = len(re.findall(r'[a-zA-Z0-9]', content))
special = len(content) - alphanum
if len(content) > 50 and alphanum > 0:
    ratio = special / alphanum
    if ratio > 3.0:
        issues.append(f'HIGH_SPECIAL_RATIO: {ratio:.1f} (special/alpha)')

# 5. Random byte patterns
if re.search(r'[\x00-\x08\x0e-\x1f]', content):
    issues.append('CONTROL_CHARS: contains control characters')

# 6. Excessive newlines (possible formatting garble)
if content.count('\n') > len(content) / 10 and len(content) > 100:
    issues.append('NEWLINE_FLOOD: too many newlines')

# 7. Unicode garble: mixed scripts or CJK in English output
cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', content))
if cjk > 0 and alphanum > 10:
    issues.append(f'UNEXPECTED_CJK: {cjk} CJK characters in English output')

print(f'  content length: {len(content)} chars')
if reasoning:
    print(f'  reasoning length: {len(reasoning)} chars')

# Show content preview
preview = content[:300] if content else '(empty)'
print(f'  --- content preview ---')
print(f'  {preview}')
if len(content) > 300:
    print(f'  ... ({len(content) - 300} more chars)')
print(f'  --- end preview ---')

if issues:
    print()
    for i in issues:
        print(f'  !! ISSUE: {i}')
else:
    print()
    print(f'  OK: no garbling detected')
"
}

# ── Main ───────────────────────────────────────────────────────────
log "=== KV Cache Quantization Debug Script ==="
log "Model: $MODEL"
log "Prompt: $PROMPT"
log "Results: $RESULTS_DIR"
echo

# First kill any existing server on the test port
kill_server

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
> "$SUMMARY_FILE"

for combo in "${COMBOS[@]}"; do
    read -r label ctk ctv <<< "$combo"
    log "────────────────────────────────────────────"
    log "TEST: -ctk $ctk -ctv $ctv ($label)"
    log "────────────────────────────────────────────"

    kill_server 2>/dev/null || true
    sleep 1

    # Start server
    log "Starting server..."
    $SERVER_BIN \
        -m "$MODEL" \
        $BASE_ARGS \
        -c "$PORT" \
        -ctk "$ctk" \
        -ctv "$ctv" \
        --temp $TEMP \
        -n 16384 \
        --ctxcp 64 \
        -b 4096 -ub 4096 \
        --no-mmap --mlock \
        2>"$RESULTS_DIR/${label}_server.log" &

    SERVER_PID=$!
    sleep 3

    if ! wait_server; then
        log "FAILED: Server didn't start for $label"
        echo "$label FAILED server_start" >> "$SUMMARY_FILE"
        kill "$SERVER_PID" 2>/dev/null || true
        continue
    fi

    # Test 1: Fresh prompt (no cache)
    log "Test 1: Fresh prompt..."
    echo "  $label" >> "$SUMMARY_FILE"
    test_prompt "${label}_fresh" 2>&1 | tee -a "$SUMMARY_FILE"

    # Test 2: Same prompt again (cache hit)
    log "Test 2: Cache hit..."
    test_prompt "${label}_cached" 2>&1 | tee -a "$SUMMARY_FILE"

    # Test 3: Slightly different prompt (partial cache)
    log "Test 3: Partial cache..."
    PROMPT_HOLD="$PROMPT"
    PROMPT="Write a haiku about the mountains."
    test_prompt "${label}_partial" 2>&1 | tee -a "$SUMMARY_FILE"
    PROMPT="$PROMPT_HOLD"

    # Test 4: Longer generation
    log "Test 4: Long generation (500 tokens)..."
    MAX_TOKENS_HOLD="$MAX_TOKENS"
    MAX_TOKENS=500
    test_prompt "${label}_long" 2>&1 | tee -a "$SUMMARY_FILE"
    MAX_TOKENS="$MAX_TOKENS_HOLD"

    # Dump server log tail
    log "Server log (last 20 lines):"
    tail -20 "$RESULTS_DIR/${label}_server.log" 2>/dev/null | sed 's/^/  /'

    echo >> "$SUMMARY_FILE"

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 2
done

# Restore original server
kill_server 2>/dev/null || true

log "════════════════════════════════════════════"
log "SUMMARY"
log "════════════════════════════════════════════"
cat "$SUMMARY_FILE"
log "════════════════════════════════════════════"
log "Full results in: $RESULTS_DIR"
