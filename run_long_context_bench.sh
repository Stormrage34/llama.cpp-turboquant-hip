#!/bin/bash
set -e

# VRAM budget calculator for Qwen3.6-35B MTP + turbo4 KV cache
# Model: Qwen3_35BMTPIQ4.gguf (19 GB, 41 layers, 2 KV heads, dim=128)
# KV cache density (turbo4): ~10.25 KB/token
#
# Per-MoE-layer VRAM: ~520 MB at IQ4
# Dense-only VRAM: ~2,030 MB
# -ncmoe 16 → VRAM: ~15,520 MB, free: ~864 MB → fits ~84K tokens
# -ncmoe 20 → VRAM: ~13,440 MB, free: ~2,944 MB → fits ~287K tokens
# -ncmoe 24 → VRAM: ~11,360 MB, free: ~5,024 MB → fits ~490K tokens

SERVER_URL="http://localhost:8080"
cd /home/stormrage/llama.cpp-turboquant-hip

# Generate long prompt by repeating a template to fill ~X tokens
gen_prompt() {
    local target_tokens=$1
    local template="The fundamental principles of quantum computing and machine learning intersect at the frontier of computational science. "
    local template_tokens=20  # approximate
    local repeats=$(( target_tokens / template_tokens ))
    local result=""
    for ((i=0; i<repeats; i++)); do
        result+="[SECTION $i] $template "
    done
    # Add a unique instruction at the end
    result+="Based on all the above sections, provide a comprehensive summary of the key themes, identify any contradictions between sections, and propose a unified framework. Be thorough and detailed."
    echo "$result"
}

bench_context() {
    local ctx_name="$1"
    local ctx_len="$2"
    local ncmoe="$3"
    local prompt_tokens="$4"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  CONTEXT: $ctx_name ($ctx_len tokens)"
    echo "║  -ncmoe: $ncmoe  |  Prompt tokens: ~$prompt_tokens"
    echo "╚══════════════════════════════════════════════════════════╝"

    # Start server
    ./build/bin/llama-server \
        -m /home/stormrage/models/Qwen3_35BMTPIQ4.gguf \
        -ngl 99 -ncmoe $ncmoe \
        -c $ctx_len \
        -b 1024 -ub 6400 \
        --cache-type-k turbo4 --cache-type-v turbo4 \
        -fa on \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 \
        --threads 8 --threads-batch 12 \
        --cpu-range 0-7 --cpu-strict 1 \
        --cpu-range-batch 0-11 --cpu-strict-batch 1 \
        --numa isolate --prio 2 \
        --no-mmap --mlock --parallel 1 --jinja \
        --cache-reuse 256 --ctx-checkpoints 8 --metrics \
        -fitt 256 --reasoning auto \
        --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 --kv-unified \
        > /tmp/llama-server-${ctx_name}.log 2>&1 &
    local PID=$!

    # Wait for server
    for i in $(seq 1 60); do
        sleep 5
        if curl -s $SERVER_URL/health 2>/dev/null | grep -q "ok"; then
            echo "  Server ready after $((i*5))s"
            break
        fi
    done

    sleep 2

    # VRAM + RAM before inference
    local VRAM_BEFORE=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk '{printf "%.0f", $NF/1048576}')
    local RAM_BEFORE=$(ps -o rss= -p $PID 2>/dev/null | awk '{printf "%.0f", $1/1024}')
    echo "  VRAM before: ${VRAM_BEFORE} MB | RAM before: ${RAM_BEFORE} MB"

    # Generate long prompt
    local prompt=$(gen_prompt $prompt_tokens)
    local json=$(jq -n --arg p "$prompt" '{
        prompt: $p,
        n_predict: 128,
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.05,
        cache_prompt: true,
        stream: false
    }')

    echo "  Sending prompt..."
    local result=$(curl -s "$SERVER_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$json" 2>&1)

    # VRAM + RAM after inference
    local VRAM_AFTER=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used Memory" | awk '{printf "%.0f", $NF/1048576}')
    local RAM_AFTER=$(ps -o rss= -p $PID 2>/dev/null | awk '{printf "%.0f", $1/1024}')

    # Extract timings
    echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
usage = data.get('usage', {})
timings = data.get('timings', {})
pt = usage.get('prompt_tokens', 0)
ct = usage.get('completion_tokens', 0)
print(f'  Prompt tokens: {pt}')
print(f'  Generated: {ct}')
print(f'  Total tokens: {usage.get(\"total_tokens\",0)}')
if timings:
    print(f'  Prompt: {timings.get(\"prompt_per_token_ms\",0):.2f} ms/tok → {timings.get(\"prompt_per_second\",0):.2f} tok/s')
    print(f'  Gen: {timings.get(\"predicted_per_token_ms\",0):.2f} ms/tok → {timings.get(\"predicted_per_second\",0):.2f} tok/s')
" 2>&1

    echo "  VRAM after: ${VRAM_AFTER} MB (Δ=$((VRAM_AFTER - VRAM_BEFORE)) MB) | RAM after: ${RAM_AFTER} MB (Δ=$((RAM_AFTER - RAM_BEFORE)) MB)"
    echo "  VRAM total used: ${VRAM_AFTER} MB / 16384 MB ($(( (VRAM_AFTER * 100) / 16384 ))%)"

    # Stop server
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    sleep 3
}

# === RUN BENCHMARKS ===

# 64K context: -ncmoe 16, ~60K prompt tokens
bench_context "64K" 65536 16 60000

# 128K context: -ncmoe 18, ~120K prompt tokens  
bench_context "128K" 131072 18 120000

# 256K context: -ncmoe 22, ~240K prompt tokens
bench_context "256K" 262144 22 240000

echo ""
echo "============================================"
echo "  DONE: 64K / 128K / 256K benchmarks"
echo "============================================"
