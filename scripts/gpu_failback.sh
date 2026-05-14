# GPU Acquire/Release Failback for RDNA2 build & benchmark pipeline
#
# Source this from scripts that need exclusive GPU access:
#   source "$(dirname "$0")/gpu_failback.sh"
#   gpu_failback_trap
#   gpu_acquire
#
# Environment:
#   GPU_FAILBACK_STATE  — state file path (default: /tmp/llama-server-state.sh)
#   GPU_FAILBACK_NORESTORE — set to 1 to skip restore on exit

GPU_FAILBACK_STATE="${GPU_FAILBACK_STATE:-/tmp/llama-server-state.sh}"
GPU_FAILBACK_VRAM_MIN=$((1 * 1024 * 1024 * 1024))  # 1 GB threshold
GPU_FAILBACK_KILL_TIMEOUT=10
GPU_FAILBACK_VRAM_TIMEOUT=30

_gpu_vram_file() {
    for card in /sys/class/drm/card*/device/mem_info_vram_total; do
        [[ -f "$card" ]] || continue
        local total=0
        total=$(<"$card") 2>/dev/null || continue
        [[ "$total" -gt $((1 * 1024 * 1024 * 1024)) ]] || continue
        echo "${card/\/mem_info_vram_total/\/mem_info_vram_used}"
        return 0
    done
    return 1
}

_gpu_vram_used() {
    local vram_file
    vram_file=$(_gpu_vram_file) || return 1
    local used
    used=$(<"$vram_file") 2>/dev/null || return 1
    echo "$used"
}

_gpu_wait_vram_below() {
    local threshold=$1 timeout=${2:-$GPU_FAILBACK_VRAM_TIMEOUT} elapsed=0
    while (( elapsed < timeout )); do
        local used
        used=$(_gpu_vram_used) || { sleep 1; ((elapsed++)); continue; }
        if [[ -n "$used" && "$used" -lt "$threshold" ]]; then
            return 0
        fi
        sleep 1
        ((elapsed++))
    done
    return 1
}

_gpu_find_llama_pid() {
    pgrep -f '[l]lama-server' | head -1
}

# Save llama-server state to file. Returns PID if saved, empty string if none.
_gpu_save_state() {
    local pid
    pid=$(_gpu_find_llama_pid) || return 1
    [[ -z "$pid" ]] && return 1

    local binary cwd env_vars
    binary=$(readlink "/proc/$pid/exe" 2>/dev/null) || return 1
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || return 1

    # Extract args (skip argv[0])
    local args=()
    while IFS= read -r -d '' arg; do
        args+=("$arg")
    done < "/proc/$pid/cmdline" 2>/dev/null
    binary="${args[0]}"

    # Extract relevant env vars
    env_vars=""
    while IFS= read -r -d '' envvar; do
        if [[ "$envvar" =~ ^(RDNA2_|LD_LIBRARY_PATH) ]]; then
            env_vars+="export ${envvar@Q}; "
        fi
    done < "/proc/$pid/environ" 2>/dev/null

    # Build restore command
    local cmd
    cmd="("
    cmd+="$env_vars"
    cmd+="cd ${cwd@Q}; "
    cmd+="exec ${binary@Q}"
    for arg in "${args[@]:1}"; do
        cmd+=" ${arg@Q}"
    done
    cmd+=" &>/tmp/llama-server-restore.log &)"

    {
        echo "# llama-server state — saved by gpu_failback.sh"
        echo "RESTORE_CMD=${cmd@Q}"
        echo "LLAMA_SERVER_PID=$pid"
    } > "$GPU_FAILBACK_STATE"

    echo "$pid"
}

gpu_acquire() {
    # Check if state file exists with a still-alive PID
    if [[ -f "$GPU_FAILBACK_STATE" ]]; then
        local old_pid
        old_pid=$(grep '^LLAMA_SERVER_PID=' "$GPU_FAILBACK_STATE" | cut -d= -f2)
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "gpu_acquire: already acquired (llama-server PID $old_pid)"
            return 0
        fi
    fi

    local pid
    pid=$(_gpu_save_state) || {
        echo "gpu_acquire: no llama-server running, nothing to do"
        return 0
    }

    echo "gpu_acquire: stopping llama-server (PID $pid)..."

    # SIGTERM — graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    local waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < GPU_FAILBACK_KILL_TIMEOUT )); do
        sleep 1
        ((waited++))
    done

    # SIGKILL if still alive
    if kill -0 "$pid" 2>/dev/null; then
        echo "gpu_acquire: force killing llama-server (PID $pid)..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    # Wait for VRAM to free
    echo "gpu_acquire: waiting for VRAM to free..."
    if _gpu_wait_vram_below "$GPU_FAILBACK_VRAM_MIN"; then
        local used
        used=$(_gpu_vram_used) || used=0
        echo "gpu_acquire: VRAM freed ($(( used / 1024 / 1024 )) MB used)"
    else
        echo "gpu_acquire: WARNING — VRAM still elevated after ${GPU_FAILBACK_VRAM_TIMEOUT}s"
    fi

    return 0
}

gpu_release() {
    [[ -f "$GPU_FAILBACK_STATE" ]] || return 0  # nothing to restore
    [[ "${GPU_FAILBACK_NORESTORE:-0}" == "1" ]] && { rm -f "$GPU_FAILBACK_STATE"; return 0; }

    local restore_cmd
    restore_cmd=$(grep '^RESTORE_CMD=' "$GPU_FAILBACK_STATE" | cut '-d=' -f2-)
    restore_cmd="${restore_cmd#\'}"
    restore_cmd="${restore_cmd%\'}"
    [[ -z "$restore_cmd" ]] && { rm -f "$GPU_FAILBACK_STATE"; return 0; }

    # Check if llama-server somehow already running
    local new_pid
    new_pid=$(_gpu_find_llama_pid)
    if [[ -n "$new_pid" ]]; then
        echo "gpu_release: llama-server already running (PID $new_pid), skipping restore"
        rm -f "$GPU_FAILBACK_STATE"
        return 0
    fi

    echo "gpu_release: restoring llama-server..."

    # Execute saved restore command
    eval "$restore_cmd"

    sleep 2

    new_pid=$(_gpu_find_llama_pid)
    if [[ -n "$new_pid" ]]; then
        echo "gpu_release: llama-server restored (PID $new_pid)"
    else
        echo "gpu_release: WARNING — llama-server may not have started (check /tmp/llama-server-restore.log)"
    fi

    local used
    used=$(_gpu_vram_used) || used=0
    echo "gpu_release: VRAM after restore: $(( used / 1024 / 1024 )) MB"

    rm -f "$GPU_FAILBACK_STATE"
}

gpu_is_busy() {
    # Check if any llama-server is running
    if _gpu_find_llama_pid >/dev/null 2>&1; then
        return 0
    fi
    # Check VRAM usage
    local used
    used=$(_gpu_vram_used) 2>/dev/null || return 1
    [[ -n "$used" && "$used" -gt "$GPU_FAILBACK_VRAM_MIN" ]] && return 0
    return 1
}

gpu_failback_trap() {
    trap gpu_release EXIT
}
