#!/usr/bin/env bash
set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

# Model selection
MODEL_DIR="${HOME}/models"
MODEL_PATH=""
ADDITIONAL_ARGS=()

if [[ $# -ge 1 ]]; then
    MODEL_PATH="$1"
    shift
    ADDITIONAL_ARGS=("$@")
fi

if [[ -z "$MODEL_PATH" ]]; then
    mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f -name "*.gguf" | sort)
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo "❌ No .gguf models found in $MODEL_DIR"
        exit 1
    fi
    MODEL_PATH="${MODELS[0]}"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

# Check required binaries
if [[ ! -x "$BIN_DIR/llama-gguf-meta" ]]; then
    echo "❌ Missing helper binary: $BIN_DIR/llama-gguf-meta"
    echo "   Build it first with: ./scripts/build_rdna2.sh"
    exit 1
fi
if [[ ! -x "$BIN_DIR/llama-server" ]]; then
    echo "❌ Missing server binary: $BIN_DIR/llama-server"
    echo "   Build it first with: ./scripts/build_rdna2.sh"
    exit 1
fi

# Extract metadata
META_OUTPUT=$("$BIN_DIR/llama-gguf-meta" "$MODEL_PATH")
declare -A META
while IFS='=' read -r key value; do
    case "$key" in
        architecture|n_embd|n_ff|n_layer|n_head|n_head_kv|n_expert|n_embd_head_k|n_embd_head_v|n_split|n_split_tensors)
            META["$key"]="$value"
            ;;
    esac
done <<< "$META_OUTPUT"

# Build auto flags
FLAGS=("-m" "$MODEL_PATH")
FLAGS+=("-ngl" "99")
if [[ ${META[n_expert]:-0} -gt 0 ]]; then
    OFFLOAD=$(( ${META[n_expert]} / 2 ))
    if [[ $OFFLOAD -gt 0 ]]; then
        FLAGS+=("-ncmoe" "$OFFLOAD")
    fi
fi
FLAGS+=("-c" "8192")
FLAGS+=("-b" "4096")
FLAGS+=("-ub" "512")
FLAGS+=("-ctk" "q8_0" "-ctv" "turbo3")
FLAGS+=("${ADDITIONAL_ARGS[@]}")

# Lifecycle files
PIDFILE="${PROJECT_ROOT}/llama_server.pid"
LOGFILE="${PROJECT_ROOT}/llama_server.log"
TELEMETRY="${PROJECT_ROOT}/llama_server_telemetry.log"

# Single-instance guard
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Server already running (PID $(cat "$PIDFILE"))."
    exit 0
fi

# Log rotation
rotate_log() {
    if [[ -f "$LOGFILE" ]]; then
        local size
        size=$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)
        if (( size > 10*1024*1024 )); then
            mv "$LOGFILE" "${LOGFILE}.$(date +%Y%m%d%H%M%S)"
        fi
    fi
}

start_server() {
    rotate_log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting llama-server with flags: ${FLAGS[*]}" | tee -a "$LOGFILE"
    "$BIN_DIR/llama-server" "${FLAGS[@]}" >> "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE"
    echo "$(date +%s) START $SERVER_PID ${FLAGS[*]}" >> "$TELEMETRY"
}

monitor() {
    while true; do
        if [[ -f "$PIDFILE" ]]; then
            SERVER_PID=$(cat "$PIDFILE")
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                EXIT_CODE=$?
                echo "$(date +%s) EXIT $SERVER_PID $EXIT_CODE" >> "$TELEMETRY"
                echo "Server exited (code $EXIT_CODE), restarting..."
                start_server
            fi
        else
            start_server
        fi
        sleep 5
    done
}

start_server
monitor