#!/usr/bin/env bash
# server_check.sh — Shared utility for llama-server awareness
#
# Source this from any script that needs GPU access:
#   source "$(dirname "$0")/server_check.sh"
#
# Then use:
#   if ! check_server_available; then
#       echo "Cannot proceed — server is running"
#       exit 1
#   fi
#
# Functions provided:
#   check_server_available    — Returns 0 if GPU is free, 1 if server is running
#   get_server_pid            — Prints llama-server PID if running, empty if not
#   get_server_stats          — Fetches /stats endpoint, prints JSON
#   get_server_metrics        — Fetches /metrics endpoint, prints raw prometheus
#   server_status             — Prints human-readable status line

# Prevent double-sourcing
if [[ -n "${SERVER_CHECK_LOADED:-}" ]]; then
    return 0 2>/dev/null || true
fi
SERVER_CHECK_LOADED=1

# ─── Core Detection ───────────────────────────────────────────────────────────

# Returns 0 if server is NOT running (GPU free), 1 if server IS running
check_server_available() {
    if pgrep -x llama-server >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Returns the PID of llama-server, or empty string
get_server_pid() {
    pgrep -x llama-server 2>/dev/null | head -1
}

# Prints human-readable server status
server_status() {
    local pid
    pid=$(get_server_pid)
    if [[ -n "$pid" ]]; then
        local cmd
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || echo "unknown")
        echo "SERVER_BUSY: PID=$pid — $cmd"
        return 1
    else
        echo "SERVER_FREE: No llama-server running"
        return 0
    fi
}

# ─── Server Metrics (when server IS running) ──────────────────────────────────

# Fetches /stats endpoint from running server
get_server_stats() {
    local stats_url="${BENCH_STATS_URL:-http://localhost:8080/stats}"
    curl -s --max-time 5 "$stats_url" 2>/dev/null | jq '.' 2>/dev/null || \
        echo "Could not fetch stats from $stats_url"
}

# Fetches /metrics endpoint from running server
get_server_metrics() {
    local metrics_url="${BENCH_METRICS_URL:-http://localhost:8080/metrics}"
    curl -s --max-time 5 "$metrics_url" 2>/dev/null || \
        echo "Could not fetch metrics from $metrics_url"
}

# ─── User-Friendly Warning ────────────────────────────────────────────────────

# Prints a formatted warning when server is running and benchmark is blocked
server_blocked_warning() {
    local script_name="${1:-$(basename "$0")}"
    local pid
    pid=$(get_server_pid)

    echo "================================================================"
    echo "  ⚠ BENCHMARK BLOCKED — llama-server is running"
    echo "================================================================"
    echo ""
    echo "  Server PID:  $pid"
    echo "  Server args: $(ps -p "$pid" -o args= 2>/dev/null | head -1)"
    echo ""
    echo "  Cannot benchmark while GPU is in use. Results would be"
    echo "  contaminated by server workload."
    echo ""
    echo "  Options:"
    echo "    1) Stop server:  kill $pid"
    echo "       (or use: scripts/gpu_failback.sh to save state first)"
    echo ""
    echo "    2) Check server stats (if --metrics enabled):"
    echo "       curl http://localhost:8080/stats | jq '.'"
    echo ""
    echo "    3) Run cloud benchmark with identical config:"
    echo "       ./scripts/run_std_bench.sh <model> <config>"
    echo "       (on remote GPU instance)"
    echo ""
    echo "    4) Quick local check (no GPU, CPU-only):"
    echo "       ./build/bin/llama-bench -m <model> -p 512 -n 32 -t \$(nproc)"
    echo "       (CPU benchmark — no GPU results, but validates binary)"
    echo ""
    echo "================================================================"
}
