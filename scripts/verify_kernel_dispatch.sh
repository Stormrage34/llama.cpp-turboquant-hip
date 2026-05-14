#!/usr/bin/env bash
# verify_kernel_dispatch.sh — Verify which dequant kernels are actually invoked
# during inference. Uses rocprofv3 --kernel-trace to capture kernel names,
# then greps for target kernels.
#
# This script exists because the DPP optimization targeted iq4_xs_rdn2 but
# the benchmark model used TurboQuant, which dispatches through a different
# kernel path. Kernel-path verification is now a mandatory gate before
# attributing counter deltas to any optimization.
#
# Usage:
#   ./scripts/verify_kernel_dispatch.sh <model.gguf> [quant_type]
#   ./scripts/verify_kernel_dispatch.sh /path/to/model.gguf Q4_K_M
#
# Output:
#   benchmarks/kernel_dispatch/ — kernel trace CSV + summary
#
# Exit codes:
#   0 — target kernel found in dispatch trace
#   1 — target kernel NOT found (optimization path mismatch)
#   2 — rocprofv3 not available

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
QUANT_TYPE="${2:-}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [quant_type]"
    echo ""
    echo "Quant types to check (comma-separated or 'all'):"
    echo "  Q4_K_M, Q5_K_M, IQ4_XS, turbo4, turbo2, turbo3"
    echo ""
    echo "If quant_type is omitted, checks all known RDNA2 dequant kernels."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# ─── Locate rocprofv3 ────────────────────────────────────────────────────────
ROCPROF=""
if command -v rocprofv3 &>/dev/null; then
    ROCPROF="rocprofv3"
elif [ -x /opt/rocm/bin/rocprofv3 ]; then
    ROCPROF="/opt/rocm/bin/rocprofv3"
else
    echo "ERROR: rocprofv3 not found. Cannot verify kernel dispatch."
    echo "Install ROCm profiler tools or run manually with:"
    echo "  rocprofv3 --kernel-trace ./build/bin/llama-bench [args]"
    exit 2
fi

BENCH="${SCRIPT_DIR}/build/bin/llama-bench"
if [ ! -x "$BENCH" ]; then
    echo "Error: llama-bench not found at $BENCH"
    echo "Run: cmake --build build --config Release"
    exit 1
fi

OUTDIR="${SCRIPT_DIR}/benchmarks/kernel_dispatch"
mkdir -p "$OUTDIR"

# ─── Known RDNA2 dequant kernel names ─────────────────────────────────────────
# These are the kernel function names that appear in rocprofv3 kernel traces.
# The exact mangled names may vary; we grep for substrings.
declare -A KERNEL_PATTERNS
KERNEL_PATTERNS[Q4_K_M]="dequantize_row_q4_K"
KERNEL_PATTERNS[Q5_K_M]="dequantize_row_q5_K"
KERNEL_PATTERNS[IQ4_XS]="dequantize_block_iq4_xs_rdn2"
KERNEL_PATTERNS[turbo4]="dequantize_turbo4"
KERNEL_PATTERNS[turbo2]="dequantize_turbo2"
KERNEL_PATTERNS[turbo3]="dequantize_turbo3"
KERNEL_PATTERNS[BFE]="rdn2_bfe"
KERNEL_PATTERNS[mul_mat_vec]="mul_mat_vec"
KERNEL_PATTERNS[mul_mat]="mul_mat_q\|mul_mat"

# ─── Determine which kernels to check ─────────────────────────────────────────
if [ -n "$QUANT_TYPE" ] && [ "$QUANT_TYPE" != "all" ]; then
    # Check specific quant type(s)
    IFS=',' read -ra TYPES <<< "$QUANT_TYPE"
    CHECK_TYPES=()
    for t in "${TYPES[@]}"; do
        t=$(echo "$t" | xargs)  # trim whitespace
        if [ -n "${KERNEL_PATTERNS[$t]+x}" ]; then
            CHECK_TYPES+=("$t")
        else
            echo "Warning: Unknown quant type '$t', checking all"
            CHECK_TYPES=("${!KERNEL_PATTERNS[@]}")
            break
        fi
    done
else
    CHECK_TYPES=("${!KERNEL_PATTERNS[@]}")
fi

# ─── Run benchmark with kernel trace ──────────────────────────────────────────
echo "=== Kernel Dispatch Verification ==="
echo "Model: $MODEL"
echo "Checking: ${CHECK_TYPES[*]}"
echo ""

# Use a short prompt to minimize runtime while still triggering dequant
BENCH_ARGS=(
    -m "$MODEL"
    -c 2048 -p 128 -n 32 -b 128 -ub 128
    --flash-attn on
    --no-mmap
    -r 1
)

echo "Running llama-bench with kernel tracing..."
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 \
    "$ROCPROF" --kernel-trace \
        -d "$OUTDIR" \
        "$BENCH" "${BENCH_ARGS[@]}" 2>&1 | tee "$OUTDIR/bench.log" || true

# ─── Parse kernel trace ───────────────────────────────────────────────────────
# rocprofv3 outputs kernel names in CSV files
# Output directory may be under OUTDIR or in a subdirectory
TRACE_FILE=$(find "$OUTDIR" -name "*.csv" -path "*kernel*" 2>/dev/null | head -1)
if [ -z "$TRACE_FILE" ]; then
    # Try broader search for any CSV with kernel dispatch data
    TRACE_FILE=$(find "$OUTDIR" -name "*.csv" 2>/dev/null | head -1)
fi

if [ -z "$TRACE_FILE" ] || [ ! -f "$TRACE_FILE" ]; then
    echo ""
    echo "ERROR: No kernel trace file found in $OUTDIR"
    echo "Available files:"
    ls -la "$OUTDIR/"
    exit 1
fi

echo ""
echo "=== Kernel Dispatch Results ==="
echo "Trace file: $TRACE_FILE"
echo ""

# Extract unique kernel names from trace
KERNEL_NAMES=$(awk -F',' 'NR>1 {print $1}' "$TRACE_FILE" 2>/dev/null | sort -u || \
               awk '{print $1}' "$TRACE_FILE" | sort -u)

echo "All dispatched kernels:"
echo "$KERNEL_NAMES" | head -30
echo ""

# ─── Check each target kernel ─────────────────────────────────────────────────
FOUND_ANY=0
MISSING_ANY=0

for qt in "${CHECK_TYPES[@]}"; do
    pattern="${KERNEL_PATTERNS[$qt]}"
    if echo "$KERNEL_NAMES" | grep -qi "$pattern"; then
        count=$(echo "$KERNEL_NAMES" | grep -ci "$pattern")
        echo "✓ $qt: kernel '$pattern' FOUND ($count dispatch(es))"
        FOUND_ANY=1
    else
        echo "✗ $qt: kernel '$pattern' NOT FOUND in dispatch trace"
        MISSING_ANY=1
    fi
done

echo ""
echo "=== Summary ==="
if [ $MISSING_ANY -eq 1 ]; then
    echo "⚠ Some target kernels were not dispatched."
    echo "  This means the optimization path does not match the model/quant."
    echo "  Do NOT attribute counter deltas to optimizations targeting missing kernels."
    echo ""
    echo "  Next steps:"
    echo "  1. Use a model with the target quant type (e.g., Q4_K_M for BFE)"
    echo "  2. Verify dispatch with: $0 <model.gguf> <quant_type>"
    exit 1
else
    echo "✓ All target kernels found in dispatch trace."
    echo "  Counter deltas can be attributed to these kernels."
fi

exit 0