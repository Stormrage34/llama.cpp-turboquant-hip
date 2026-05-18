#!/usr/bin/env bash
# run_longctx_bench.sh — Long-context benchmark suite for RDNA2
#
# Tests model performance at 64k, 128k, 256k context sizes with
# different KV cache configurations to find optimal settings
# for (16 GB VRAM + 48 GB RAM) setups.
#
# Uses --fit-ctx to control context size via the auto-fitting
# memory manager (context is auto-calculated from model+VRAM).
#
# Usage:
#   ./scripts/run_longctx_bench.sh <model.gguf> [config]
#   ./scripts/run_longctx_bench.sh /path/to/model.gguf moe-99
#
# Configs:
#   moe-99    — MoE model, full GPU offload [default]
#   dense-99  — Dense model, full GPU offload
#
# Scenarios: 3 contexts × 3 cache configs = 9 total
#   Contexts: 64k, 128k, 256k
#   Cache:    balanced (q8_0/turbo3), decode (turbo3/turbo3), aggressive (turbo3/turbo2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Server awareness
source "$(cd "$(dirname "$0")" && pwd)/server_check.sh"
if ! check_server_available; then
    server_blocked_warning "run_longctx_bench.sh"
    exit 1
fi

MODEL="${1:-}"
CONFIG="${2:-moe-99}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [config]"
    echo ""
    echo "Tests long-context performance at 64k, 128k, 256k"
    echo "with multiple KV cache configurations."
    echo ""
    echo "Configs: moe-99 (default), dense-99"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

# Config-dependent parameters
case "${CONFIG}" in
    moe-99)
        NGL=99
        NCMOE=32
        DESC="MoE model, full GPU offload"
        ;;
    dense-99)
        NGL=99
        NCMOE=0
        DESC="Dense model, full GPU offload"
        ;;
    *)
        echo "Error: Unknown config '${CONFIG}'"
        exit 1
        ;;
esac

# ─── System info ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${SCRIPT_DIR}/benchmarks/longctx/${TIMESTAMP}_${CONFIG}"
mkdir -p "$OUTDIR"

echo "=== Long Context Benchmark Suite ===" | tee "${OUTDIR}/summary.txt"
echo "  Timestamp:  ${TIMESTAMP}" | tee -a "${OUTDIR}/summary.txt"
echo "  Model:      ${MODEL}" | tee -a "${OUTDIR}/summary.txt"
echo "  Config:     ${CONFIG} (${DESC})" | tee -a "${OUTDIR}/summary.txt"
echo "  NGL:        ${NGL}" | tee -a "${OUTDIR}/summary.txt"
echo "  NCMOE:      ${NCMOE}" | tee -a "${OUTDIR}/summary.txt"
echo "  RAM:        48 GB | VRAM: 16 GB (RX 6800 XT)" | tee -a "${OUTDIR}/summary.txt"
echo "" | tee -a "${OUTDIR}/summary.txt"

# Record git info
echo "=== Git Info ===" >> "${OUTDIR}/summary.txt"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)" >> "${OUTDIR}/summary.txt"
echo "  Commit: $(git rev-parse --short HEAD)" >> "${OUTDIR}/summary.txt"
echo "  Tag:    $(git describe --tags --always 2>/dev/null || echo 'none')" >> "${OUTDIR}/summary.txt"
echo "" >> "${OUTDIR}/summary.txt"

# Record GPU info
ROCM_SMI="${ROCM_PATH:-/opt/rocm}/bin/rocm-smi"
if [ -x "${ROCM_SMI}" ]; then
    echo "=== GPU Info ===" >> "${OUTDIR}/summary.txt"
    ${ROCM_SMI} >> "${OUTDIR}/summary.txt" 2>&1 || true
    echo "" >> "${OUTDIR}/summary.txt"
fi

# ─── Benchmark locations ──────────────────────────────────────────────────────
BENCH="${SCRIPT_DIR}/build/bin/llama-bench"
if [ ! -x "$BENCH" ]; then
    echo "Error: llama-bench not found at $BENCH"
    exit 1
fi

# ─── Scenarios ─────────────────────────────────────────────────────────────────
# Format: ctx ctk ctv description
# Contexts: 65536 (64k), 131072 (128k), 262144 (256k)
# Cache: balanced (q8_0/turbo3), decode (turbo3/turbo3), aggressive (turbo3/turbo2)
SCENARIOS=(
    "65536  q8_0   turbo3 64k_balanced"
    "65536  turbo3 turbo3 64k_decode"
    "65536  turbo3 turbo2 64k_aggressive"
    "131072 q8_0   turbo3 128k_balanced"
    "131072 turbo3 turbo3 128k_decode"
    "131072 turbo3 turbo2 128k_aggressive"
    "262144 q8_0   turbo3 256k_balanced"
    "262144 turbo3 turbo3 256k_decode"
    "262144 turbo3 turbo2 256k_aggressive"
)

echo "Scenarios to run: ${#SCENARIOS[@]}" | tee -a "${OUTDIR}/summary.txt"
echo "" | tee -a "${OUTDIR}/summary.txt"

# ─── Run benchmarks ────────────────────────────────────────────────────────────
for scenario in "${SCENARIOS[@]}"; do
    read -r ctx ctk ctv desc <<< "$scenario"
    
    echo "--- [$desc] ctx=${ctx} ctk=${ctk} ctv=${ctv} ---" | tee -a "${OUTDIR}/summary.txt"
    
    # Each scenario is a separate llama-bench invocation because
    # --fit-ctx is a global setting (not per-test parameter)
    "$BENCH" \
        -m "$MODEL" \
        -p 512 -n 128 \
        -b 128 -ub 128 \
        -ctk "$ctk" -ctv "$ctv" \
        -fa 1 -mmp 0 \
        -ngl "$NGL" \
        $( [ "${NCMOE:-0}" -gt 0 ] && echo "-ncmoe $NCMOE" ) \
        --fit-target 2000 \
        --fit-ctx "$ctx" \
        -r 1 \
        -o md 2>&1 | tee -a "${OUTDIR}/bench.log"
    
    # Parse results
    pp=$(grep -oP '\|\s+pp512\s+\|\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null | tail -1)
    tg=$(grep -oP '\|\s+tg128\s+\|\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null | tail -1)
    
    echo "  => pp512=${pp:-N/A} t/s  tg128=${tg:-N/A} t/s" | tee -a "${OUTDIR}/summary.txt"
    echo "" | tee -a "${OUTDIR}/summary.txt"
done

# ─── Build result table ────────────────────────────────────────────────────────
echo "" | tee -a "${OUTDIR}/summary.txt"
echo "=== Results Table ===" | tee -a "${OUTDIR}/summary.txt"
echo "  Context  | Cache           | Prefill (pp512) | Decode (tg128)" | tee -a "${OUTDIR}/summary.txt"
echo "  ---------|-----------------|-----------------|---------------" | tee -a "${OUTDIR}/summary.txt"

for scenario in "${SCENARIOS[@]}"; do
    read -r ctx ctk ctv desc <<< "$scenario"
    pp=$(grep -oP '\|\s+pp512\s+\|\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null | tail -1)
    tg=$(grep -oP '\|\s+tg128\s+\|\s+\K[\d.]+' "${OUTDIR}/bench.log" 2>/dev/null | tail -1)
    printf "  %-7s | %-15s | %-15s | %s\n" "$((ctx/1024))k" "${ctk}/${ctv}" "${pp:-N/A}" "${tg:-N/A}" | tee -a "${OUTDIR}/summary.txt"
done

echo "" | tee -a "${OUTDIR}/summary.txt"
echo "=== Benchmark Complete ===" | tee -a "${OUTDIR}/summary.txt"
echo "Results saved to: ${OUTDIR}/" | tee -a "${OUTDIR}/summary.txt"
