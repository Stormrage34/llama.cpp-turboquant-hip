#!/bin/bash
set -e

SERVER_URL="http://localhost:8080"
cd /home/stormrage/llama.cpp-turboquant-hip

bench_one() {
    local name="$1"
    local file="$2"
    local prompt=$(cat "$file" | tr '\n' ' ' | sed 's/"/\\"/g')
    
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  PROMPT: $name"
    echo "╚══════════════════════════════════════════╝"
    
    # VRAM + RAM before
    echo "--- Before ---"
    rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk '{printf "VRAM Used: %.0f MB\n", $NF/1048576}'
    ps -o rss= -p $(pgrep llama-server) | awk '{printf "RAM Used: %.0f MB\n", $1/1024}'
    
    # Send prompt
    local json=$(jq -n --arg p "$prompt" '{
        prompt: $p,
        n_predict: 512,
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.05,
        cache_prompt: true,
        stream: false
    }')
    
    local result=$(curl -s "$SERVER_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$json" 2>&1)
    
    # Parse timings
    echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
usage = data.get('usage', {})
timings = data.get('timings', {})

print(f'Prompt tokens: {usage.get(\"prompt_tokens\",\"?\")}')
print(f'Generated tokens: {usage.get(\"completion_tokens\",\"?\")}')
if timings:
    ppms = timings.get('prompt_per_token_ms', 0)
    pps = timings.get('prompt_per_second', 0)
    dpms = timings.get('predicted_per_token_ms', 0)
    dps = timings.get('predicted_per_second', 0)
    print(f'Prompt: {ppms:.2f} ms/tok → {pps:.2f} tok/s')
    print(f'Generate: {dpms:.2f} ms/tok → {dps:.2f} tok/s')
    
    dn = timings.get('predicted_n', 0)
    dm = timings.get('predicted_ms', 0)
    if dn and dm:
        print(f'Total gen time: {dm/1000:.1f}s for {dn} tokens')
" 2>&1
    
    # VRAM + RAM after
    echo "--- After ---"
    rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk '{printf "VRAM Used: %.0f MB\n", $NF/1048576}'
    ps -o rss= -p $(pgrep llama-server) | awk '{printf "RAM Used: %.0f MB\n", $1/1024}'
}

# Run all 4
bench_one "Creative Writing"   creativeMTP.txt
bench_one "Code Review"        codingmtp.txt
bench_one "Problem Solving"    problemsolvingmtp.txt
bench_one "Research Analysis"  researchmtp.txt

echo ""
echo "============================================"
echo "  BENCHMARK COMPLETE"
echo "============================================"
