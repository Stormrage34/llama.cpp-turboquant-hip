#!/usr/bin/env bash
# Exclusive-access guard for gfx1030 measurements.
#
# Enforces the two local benchmarking rules recorded in AGENTS.md:
#   1. llama-server must never be active.
#   2. Exactly one llama-bench process at a time.
#
# Usage:
#   scripts/bench-exclusive.sh <path-to-llama-bench> [llama-bench args...]
#
# Exits non-zero (and runs nothing) if either rule would be violated, so a
# blocked run can never be mistaken for a slow one.

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <path-to-llama-bench> [args...]" >&2
    exit 2
fi

BIN="$1"
shift

if [ ! -x "$BIN" ]; then
    echo "bench-exclusive: not executable: $BIN" >&2
    exit 2
fi

# Rule 1: no llama-server, ever.
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "bench-exclusive: REFUSING - llama-server is running (pid $(pgrep -x llama-server | tr '\n' ' '))." >&2
    echo "bench-exclusive: stop it before measuring; measurements taken with it alive are invalid." >&2
    exit 3
fi

# Rule 2: at most one llama-bench. Match the binary name only, so this
# script's own command line cannot trip the check.
if pgrep -x llama-bench >/dev/null 2>&1; then
    echo "bench-exclusive: REFUSING - another llama-bench is running (pid $(pgrep -x llama-bench | tr '\n' ' '))." >&2
    echo "bench-exclusive: concurrent instances exhaust VRAM and corrupt the numbers." >&2
    exit 4
fi

# Advisory: report VRAM so a leaking previous run is visible in the log.
VRAM="$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): *[0-9]+' | grep -oE '[0-9]+$' | head -1)"
if [ -n "$VRAM" ] && [ "$VRAM" -gt 25 ]; then
    echo "bench-exclusive: WARNING - ${VRAM}% VRAM already allocated; a stale process may be holding the GPU." >&2
fi

exec "$BIN" "$@"
