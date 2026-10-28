#!/usr/bin/env bash
# Compare two llama-bench binaries (usually this fork vs vanilla upstream) on
# the same GPU, the same model and the same flags, in interleaved pairs.
#
# WHAT THIS IS AND IS NOT
#   This is a *decode-speed parity* harness, not a code comparison. It answers
#   one narrow question: "on this model and these flags, does the fork's
#   llama-bench reach the same tokens/second as a clean upstream build?"
#   Anything about quality, graph contents or kernels belongs to
#   `gfx1030.py bench --ab` / `trace`, not here.
#
# WHY PAIRS ARE INTERLEAVED
#   Decode speed on this host drifts by ~2% run to run (clock and thermal
#   wander). Two unpaired runs therefore prove nothing. Each pair runs fork
#   then upstream back to back, so the drift applies to both sides and largely
#   cancels. The paired statistics (mean delta, sd of per-pair differences,
#   required pair count) are computed by `gfx1030.py bench --ab-stats`; feed it
#   the per-pair deltas printed here.
#
# RULES HONOURED
#   llama-server is never started, stopped or queried. Every llama-bench run
#   goes through scripts/bench-exclusive.sh, which refuses to run if a server
#   or a second llama-bench is alive. Runs are strictly sequential.
#
# UPSTREAM BUILD
#   By default the first run clones upstream into --upstream-src and builds
#   llama-bench there. The build is an experiment-setup step, not part of the
#   measurement; pass --no-build to forbid it, and the script will fail loudly
#   if the binary is missing.
#
# ENVIRONMENT PARITY
#   The only deviations from upstream defaults are environment choices:
#     AMDGPU_TARGETS=gfx1030   this card
#     CMAKE_BUILD_TYPE=Release matching the fork build
#     GGML_HIP_NO_VMM=ON       upstream's default VMM pool aborts on this ROCm
#                              under prompt processing, so the arm cannot run
#                              at all without it
#   Everything else is upstream as shipped. Upstream's default
#   GGML_CUDA_FA_QUANTS does not contain q8_0-q5_0 (a local addition), so
#   upstream converts K/V to f16 for a -ctk q8_0 -ctv q5_0 run. Noted, not
#   corrected: the point is to measure the fork as deployed.
#
# USAGE
#   scripts/bench-fork-vs-upstream.sh --help
#
#   scripts/bench-fork-vs-upstream.sh -m dense.gguf -a "-dev ROCm0"
#   scripts/bench-fork-vs-upstream.sh -m moe.gguf -a "-dev ROCm0 -ncmoe 9" -n 5
#
#   -m, --model FILE        model to compare (repeatable, at least one; required)
#   -f, --fork-bin FILE     fork llama-bench (default: build-hip-gfx1030/bin/llama-bench)
#   -u, --upstream-bin FILE upstream llama-bench (default: <upstream-src>/build/bin/llama-bench)
#       --upstream-src DIR  upstream checkout (default: /tmp/opencode/llama-upstream)
#   -n, --pairs N           interleaved pairs per model (default: 5)
#   -r, --reps N            llama-bench -r repetitions (default: 3)
#   -t, --tokens N          llama-bench -n generated tokens (default: 128)
#   -a, --bench-args STR    extra llama-bench flags applied to BOTH sides
#   -o, --outdir DIR        logs and pair data (default: /tmp/opencode/upstream)
#       --no-build          never clone or build; fail if a binary is missing
#   -h, --help              this text
#
# Exit status is 1 if any single run failed, else 0. A failed run is printed as
# FAIL and excluded from the summary rather than silently counted as 0 t/s.
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$ROOT/scripts/common.sh"

usage() { sed -n '2,/^set -u$/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

MODELS=()
FORK="$ROOT/build-hip-gfx1030/bin/llama-bench"
UP_SRC=/tmp/opencode/llama-upstream
UP_BIN=""
PAIRS=5
REPS=3
TOKENS=128
BENCH_ARGS=""
OUT=/tmp/opencode/upstream
DO_BUILD=1

die() { echo "ABORT: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        -m|--model)       MODELS+=("${2:?-m needs a path}"); shift 2 ;;
        -f|--fork-bin)    FORK="${2:?-f needs a path}"; shift 2 ;;
        -u|--upstream-bin) UP_BIN="${2:?-u needs a path}"; shift 2 ;;
        --upstream-src)   UP_SRC="${2:?--upstream-src needs a path}"; shift 2 ;;
        -n|--pairs)       PAIRS="${2:?-n needs a number}"; shift 2 ;;
        -r|--reps)        REPS="${2:?-r needs a number}"; shift 2 ;;
        -t|--tokens)      TOKENS="${2:?-t needs a number}"; shift 2 ;;
        -a|--bench-args)  BENCH_ARGS="${2:?-a needs a string}"; shift 2 ;;
        -o|--outdir)      OUT="${2:?-o needs a path}"; shift 2 ;;
        --no-build)       DO_BUILD=0; shift ;;
        *)                die "unknown argument: $1 (try --help)" ;;
    esac
done

[ "${#MODELS[@]}" -gt 0 ] || die "-m/--model is required (repeat for several); try --help"
for m in "${MODELS[@]}"; do [ -f "$m" ] || die "model not found: $m"; done
[ -x "$BENCH_EXCLUSIVE" ] || die "guard not executable: $BENCH_EXCLUSIVE"

UP_BIN="${UP_BIN:-$UP_SRC/build/bin/llama-bench}"
mkdir -p "$OUT" || die "cannot create outdir: $OUT"
rm -f "$OUT/pairs.tsv"
EXTRA=()
if [ -n "$BENCH_ARGS" ]; then
    # The caller passes llama-bench flags as one shell-quoted string on purpose.
    read -r -a EXTRA <<< "$BENCH_ARGS"
fi

echo "=== [1/2] binaries ==="
if [ "$DO_BUILD" = 1 ] && [ ! -x "$UP_BIN" ]; then
    if [ ! -d "$UP_SRC/.git" ]; then
        echo "  cloning upstream into $UP_SRC ..."
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$UP_SRC" > "$OUT/clone.log" 2>&1
        echo "  clone rc=$?"
    fi
    echo "  configuring upstream (HIP, gfx1030, no VMM) ..."
    cmake -S "$UP_SRC" -B "$UP_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGPU_TARGETS=gfx1030 \
        -DGGML_HIP_NO_VMM=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF > "$OUT/configure.log" 2>&1
    echo "  configure rc=$?"
    ok=0
    for i in 1 2 3 4 5 6; do
        if cmake --build "$UP_SRC/build" --target llama-bench -- -j4 > "$OUT/build.log" 2>&1; then
            echo "  build OK (attempt $i)"; ok=1; break
        fi
        echo "  build attempt $i failed:"
        grep -E "error:" "$OUT/build.log" | head -3
    done
    [ "$ok" = 1 ] || die "upstream build failed after 6 attempts (see $OUT/build.log)"
elif [ "$DO_BUILD" = 0 ] && [ ! -x "$UP_BIN" ]; then
    die "--no-build given but upstream llama-bench is missing: $UP_BIN"
fi
for b in "$FORK" "$UP_BIN"; do
    [ -x "$b" ] || die "not executable: $b"
    echo "  $b"
done
git -C "$UP_SRC" log -1 --format='  upstream HEAD %h %ad %s' --date=short 2>/dev/null \
    | tee "$OUT/upstream-head.txt"

# bench <label> <binary> <model> [extra...] -> prints avg_ts, or FAIL
FAILED=0
bench() {
    local label="$1" bin="$2" model="$3"; shift 3
    local ts
    ts=$("$BENCH_EXCLUSIVE" "$bin" -m "$model" \
            -p 0 -n "$TOKENS" -r "$REPS" -d 0 \
            -ngl 99 -fa $FA -ctk $CTK -ctv $CTV "$@" -o jsonl 2>/dev/null \
         | grep -o '"avg_ts": *[0-9.]*' | head -1 | sed 's/.*: *//')
    case "$ts" in
        ''|*[!0-9.]*)
            printf '%-10s %s\n' "$label" FAIL >&2
            echo ""
            ;;
        *)  printf '%-10s %s\n' "$label" "$ts" >&2
            echo "$ts"
            ;;
    esac
}

echo "=== [2/2] interleaved pairs ==="
for model in "${MODELS[@]}"; do
    label="$(basename "$model" .gguf)"
    echo "--- $label ---"
    for p in $(seq 1 "$PAIRS"); do
        a=$(bench fork     "$FORK"   "$model" "${EXTRA[@]}")
        b=$(bench upstream "$UP_BIN" "$model" "${EXTRA[@]}")
        printf '  pair %-3s fork=%s  upstream=%s\n' "$p" "${a:-FAIL}" "${b:-FAIL}"
        if [ -n "$a" ] && [ -n "$b" ]; then
            printf '%s\t%s\t%s\n' "$label" "$a" "$b" >> "$OUT/pairs.tsv"
        else
            FAILED=1
        fi
    done
done

echo "=== summary (mean decode speed, positive delta = fork slower) ==="
if [ -s "$OUT/pairs.tsv" ]; then
    awk -F'\t' '
        { n[$1]++; f[$1] += 1000.0/$2; u[$1] += 1000.0/$3 }
        END {
            printf "  %-28s %6s %12s %12s %9s\n", "model", "pairs", "fork ms/tok", "up ms/tok", "delta%"
            for (m in n)
                printf "  %-28s %6d %12.3f %12.3f %+8.2f%%\n",
                    m, n[m], f[m]/n[m], u[m]/n[m], 100.0*(f[m]/n[m] - u[m]/n[m])/(u[m]/n[m])
        }' "$OUT/pairs.tsv"
    echo "  pair data: $OUT/pairs.tsv  (feed per-pair deltas to: gfx1030.py bench --ab-stats)"
else
    echo "  no complete pairs"
    FAILED=1
fi

echo "UPSTREAM_COMPARE_DONE rc=$FAILED"
exit "$FAILED"
