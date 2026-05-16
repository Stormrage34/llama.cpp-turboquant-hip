#!/bin/bash
# TurboQuant ROCm Test Script
# Run this after llama-server has finished serving / shut down
# Usage: ./test-turboquant.sh <path-to-model.gguf> [port]

set -e

MODEL_PATH="${1:?Usage: $0 <path-to-model.gguf> [port]}"
PORT="${2:-8080}"
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "=== TurboQuant ROCm Test ==="
echo "Model: $MODEL_PATH"
echo "Server port: $PORT"
echo ""

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# Start llama-server with TurboQuant settings
echo "Starting llama-server with TurboQuant KV cache..."
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --n-gpu-layers 999 \
    --flash-attn \
    --turbocomp \
    --log-disable \
    > /tmp/llama-server-turboquant.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Server failed to start within 60 seconds"
        cat /tmp/llama-server-turboquant.log
        exit 1
    fi
    sleep 1
done

# Run inference test
echo ""
echo "Running inference test..."
RESPONSE=$(curl -s http://localhost:$PORT/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of France is",
        "n_predict": 32,
        "temperature": 0.7,
        "top_p": 0.9
    }' 2>/dev/null)

echo "Response: $(echo "$RESPONSE" | grep -o '"completion":"[^"]*"' | head -1)"
echo ""

# Check memory usage
echo "Checking ROCm memory usage..."
rocm-smi 2>/dev/null || echo "rocm-smi not available"
echo ""

# Get server stats
echo "Server stats:"
curl -s http://localhost:$PORT/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/stats 2>/dev/null

echo ""
echo "=== Test Complete ==="
echo "Server logs saved to /tmp/llama-server-turboquant.log"
echo "Server will shut down automatically in 30 seconds..."
sleep 30
