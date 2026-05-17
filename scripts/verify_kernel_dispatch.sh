#!/usr/bin/env bash
# verify_kernel_dispatch.sh — Verify which dequant kernels are actually invoked
# during inference. Uses rocprofv3 database + sqlite3 to capture kernel names.
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
#   benchmarks/kernel_dispatch/ — kernel trace summary
#
# Exit codes:
#   0 — target kernel found in dispatch trace
#   1 — target kernel NOT found (optimization path mismatch)
#   2 — sqlite3 not available or no database found

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

# ─── Locate sqlite3 ──────────────────────────────────────────────────────────
if ! command -v sqlite3 &>/dev/null; then
    echo "ERROR: sqlite3 not found. Cannot query rocprofv3 database."
    echo "Install sqlite3 or run rocprofv3 manually:"
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

# ─── Known ggml_type values for quantization types ───────────────────────────
# These are the ggml_type enum values used in mul_mat_vec_q kernels.
# Based on ggml.h enum ggml_type:
#   GGML_TYPE_F16        = 0
#   GGML_TYPE_Q4_0       = 1
#   GGML_TYPE_Q4_1       = 2
#   GGML_TYPE_Q5_0       = 8
#   GGML_TYPE_Q5_1       = 9
#   GGML_TYPE_Q8_0       = 10
#   GGML_TYPE_Q8_1       = 11
#   GGML_TYPE_Q2_K       = 12
#   GGML_TYPE_Q3_K       = 13
#   GGML_TYPE_Q4_K       = 14
#   GGML_TYPE_Q5_K       = 15
#   GGML_TYPE_Q6_K       = 16
#   GGML_TYPE_Q8_K       = 17
#   GGML_TYPE_IQ2_XS     = 18
#   GGML_TYPE_IQ2_RS     = 19
#   GGML_TYPE_IQ3_S      = 20
#   GGML_TYPE_IQ3_XS     = 21
#   GGML_TYPE_IQ4_NL     = 22
#   GGML_TYPE_IQ4_XS     = 23
#   GGML_TYPE_IQ1_S      = 24
#   GGML_TYPE_IQ1_M      = 25
#   GGML_TYPE_IQ2_S      = 26
#   GGML_TYPE_TURBO2     = 27
#   GGML_TYPE_TURBO3     = 28
#   GGML_TYPE_TURBO4     = 29
declare -A QUANT_TYPE_MAP
QUANT_TYPE_MAP[Q4_K_M]="14"      # GGML_TYPE_Q4_K
QUANT_TYPE_MAP[Q5_K_M]="15"      # GGML_TYPE_Q5_K
QUANT_TYPE_MAP[IQ4_XS]="23"      # GGML_TYPE_IQ4_XS
QUANT_TYPE_MAP[turbo4]="29"      # GGML_TYPE_TURBO4
QUANT_TYPE_MAP[turbo2]="27"      # GGML_TYPE_TURBO2
QUANT_TYPE_MAP[turbo3]="28"      # GGML_TYPE_TURBO3

# ─── Determine which kernels to check ─────────────────────────────────────────
if [ -n "$QUANT_TYPE" ] && [ "$QUANT_TYPE" != "all" ]; then
    IFS=',' read -ra TYPES <<< "$QUANT_TYPE"
    CHECK_TYPES=()
    for t in "${TYPES[@]}"; do
        t=$(echo "$t" | xargs)
        if [ -n "${QUANT_TYPE_MAP[$t]+x}" ]; then
            CHECK_TYPES+=("$t")
        else
            echo "Warning: Unknown quant type '$t', checking all"
            CHECK_TYPES=("${!QUANT_TYPE_MAP[@]}")
            break
        fi
    done
else
    CHECK_TYPES=("${!QUANT_TYPE_MAP[@]}")
fi

# ─── Find existing rocprofv3 database ─────────────────────────────────────────
echo "=== Kernel Dispatch Verification ==="
echo "Model: $MODEL"
echo "Checking: ${CHECK_TYPES[*]}"
echo ""

DB_FILE=$(find "$OUTDIR" -name "*.db" 2>/dev/null | sort -r | head -1)
if [ -z "$DB_FILE" ]; then
    echo "ERROR: No rocprofv3 database file (.db) found in $OUTDIR"
    echo "Run llama-bench with rocprofv3 first:"
    echo "  rocprofv3 --kernel-trace -d benchmarks/kernel_dispatch/ ./build/bin/llama-bench [args]"
    exit 2
fi

echo "Using database: $DB_FILE"
echo ""

# ─── Extract kernel names from database ───────────────────────────────────────
echo "Extracting kernel trace from database..."

# Try to find the right kernel_dispatch table name
KERNEL_NAMES=""
for table in $(sqlite3 "$DB_FILE" ".tables" 2>/dev/null | grep -E "rocpd_kernel_dispatch" | head -1); do
    KERNEL_NAMES=$(sqlite3 "$DB_FILE" "
        SELECT DISTINCT ks.formatted_kernel_name
        FROM $table k
        JOIN kernel_symbols ks ON k.kernel_id = ks.id
        WHERE ks.formatted_kernel_name IS NOT NULL
        AND ks.formatted_kernel_name != ''
        ORDER BY ks.formatted_kernel_name;
    " 2>/dev/null || echo "")
    if [ -n "$KERNEL_NAMES" ]; then
        break
    fi
done

if [ -z "$KERNEL_NAMES" ]; then
    echo "ERROR: Could not extract kernel names from database"
    exit 1
fi

# ─── Check each target kernel ─────────────────────────────────────────────────
echo ""
echo "=== Kernel Dispatch Results ==="
echo ""

# Show all mul_mat_q and mul_mat_vec_q kernels
echo "All dequant/mul_mat kernels dispatched:"
echo "$KERNEL_NAMES" | grep -E "mul_mat_q|mul_mat_vec_q" | sort -u | head -30
echo ""

FOUND_ANY=0
MISSING_ANY=0

for qt in "${CHECK_TYPES[@]}"; do
    type_id="${QUANT_TYPE_MAP[$qt]}"
    # Check if mul_mat_vec_q with this ggml_type was dispatched
    if echo "$KERNEL_NAMES" | grep -q "mul_mat_vec_q<(ggml_type)${type_id}"; then
        count=$(echo "$KERNEL_NAMES" | grep -c "mul_mat_vec_q<(ggml_type)${type_id}" || true)
        echo "✓ $qt (type $type_id): kernel FOUND ($count dispatch(es))"
        FOUND_ANY=1
    elif echo "$KERNEL_NAMES" | grep -q "mul_mat_q<(ggml_type)${type_id}"; then
        count=$(echo "$KERNEL_NAMES" | grep -c "mul_mat_q<(ggml_type)${type_id}" || true)
        echo "✓ $qt (type $type_id): kernel FOUND via mul_mat_q ($count dispatch(es))"
        FOUND_ANY=1
    else
        echo "✗ $qt (type $type_id): kernel NOT FOUND in dispatch trace"
        MISSING_ANY=1
    fi
done

echo ""
echo "=== Summary ==="
if [ $MISSING_ANY -eq 1 ]; then
    echo "Some target kernels were not dispatched."
    echo "  This means the optimization path does not match the model/quant."
    echo "  Do NOT attribute counter deltas to optimizations targeting missing kernels."
    echo ""
    echo "  Next steps:"
    echo "  1. Use a model with the target quant type"
    echo "  2. Verify dispatch with: $0 <model.gguf> <quant_type>"
    exit 1
else
    echo "All target kernels found in dispatch trace."
    echo "  Counter deltas can be attributed to these kernels."
fi

exit 0
