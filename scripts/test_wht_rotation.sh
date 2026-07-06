#!/bin/bash
# Diagnostic script to isolate WHT rotation as root cause of KV cache drift
#
# Strategy: test turbo3_0 with InnerQ calibration at different strengths
# and compare against q8_0 baseline to see if calibration reduces drift.
#
# Also tests whether the drift is from:
# 1. WHT rotation (both turbo types share it)
# 2. Quantization in rotated domain
# 3. Dequantization without inverse rotation in VEC FA path

set -euo pipefail

MODEL="${MODEL:-/mnt/new2tb/llm-models/qwen3.5-35B-A3B/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf}"
PORT="${PORT:-8080}"
HOST="0.0.0.0"
COMMON_ARGS="-c 4900 -fa on -ts 10.5,5.0 --n-cpu-moe-range 0-13 --n-cpu-moe 14 --host $HOST --port $PORT"

echo "============================================"
echo "TurboQuant KV Cache Drift Diagnostic"
echo "============================================"
echo ""
echo "Model: $MODEL"
echo "Port:  $PORT"
echo ""
echo "The drift occurs identically in turbo3_0 (3-bit) and turbo4_0 (4-bit),"
echo "ruling out quantization precision as the cause."
echo ""
echo "The only shared component between turbo types that q8_0 lacks is WHT rotation."
echo ""
echo "Test plan:"
echo "  1. q8_0 baseline (known correct)"
echo "  2. turbo3_0 default (known drift)"
echo "  3. turbo3_0 + InnerQ (calibration may correct systematic offset)"
echo "  4. turbo3_0 + strong InnerQ (max calibration strength)"
echo ""

run_test() {
    local name="$1"
    local extra_args="$2"
    local logfile="/tmp/turbo_diag_${name}.log"
    
    echo "-------------------------------------------"
    echo "TEST: $name"
    echo "Extra args: $extra_args"
    echo "Log: $logfile"
    echo ""
    
    # Start server
    eval "echo \"Starting server with: $extra_args\" > $logfile"
    eval "timeout 300 ./bin/llama-server -m $MODEL $COMMON_ARGS $extra_args >> $logfile 2>&1" &
    SERVER_PID=$!
    
    # Wait for server to be ready
    echo "Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s http://$HOST:$PORT/health | grep -q '"status":"ok"'; then
            echo "Server ready after ${i}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Server died. Check $logfile"
            return 1
        fi
        sleep 5
    done
    
    echo "Server running. Run your Variable Tracking Narrative prompt now."
    echo "Press Enter when done (or 'skip' to skip)..."
    read -r response
    
    # Kill server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    
    echo "TEST $name complete. Log: $logfile"
    echo ""
}

echo "Starting diagnostic tests..."
echo ""

# Test 1: turbo3_0 default (baseline drift)
run_test "turbo3_default" "-ctv turbo3_0 -ctk q8_0"

# Test 2: turbo3_0 + InnerQ moderate
run_test "turbo3_innerq_500" "TURBO_INNERQ=500 TURBO_INNERQ_STRENGTH=0.5 -ctv turbo3_0 -ctk q8_0"

# Test 3: turbo3_0 + InnerQ strong
run_test "turbo3_innerq_800" "TURBO_INNERQ=500 TURBO_INNERQ_STRENGTH=0.8 -ctv turbo3_0 -ctk q8_0"

echo "============================================"
echo "All tests complete."
echo ""
echo "Compare results across tests. If drift is identical in all cases,"
echo "the root cause is WHT rotation itself (not calibration or centroids)."
echo ""
echo "Next step if drift persists: test with rotation disabled."
echo "============================================"
