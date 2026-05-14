#!/usr/bin/env bash
# download_bfe_models.sh — Download Q4_K_M and Q5_K_M models for BFE validation
#
# BFE (bit-field extract) optimization targets Q4_K_M and Q5_K_M dequant kernels.
# This script downloads the models needed for kernel-path verification and A/B telemetry.
#
# Usage:
#   ./scripts/download_bfe_models.sh           # Download all models
#   ./scripts/download_bfe_models.sh q4_km      # Download Q4_K_M only
#   ./scripts/download_bfe_models.sh q5_km      # Download Q5_K_M only
#
# Models:
#   Q4_K_M: gemma-4-26B-A4B-it-UD-Q4_K_M.gguf (16 GB) — ALREADY PRESENT
#   Q5_K_M: gemma-4-26B-A4B-it-UD-Q5_K_M.gguf (~18 GB) — needs download
#
# These are Gemma 4 26B models with UD (Ultimate-Dither) quantization,
# which exercises the Q4_K_M and Q5_K_M dequant paths that BFE targets.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/stormrage/models}"
mkdir -p "$MODEL_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

MODE="${1:-all}"

# ─── Model definitions ───────────────────────────────────────────────────────
# Gemma 4 26B A4B it — UD quantization by Bartowski
# These exercise the Q4_K_M and Q5_K_M dequant paths that BFE targets.
# UD (Ultimate-Dither) quantizations are high-quality and well-tested in llama.cpp.

declare -A MODELS
MODELS[q4_km]="https://huggingface.co/bartowski/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
MODELS[q5_km]="https://huggingface.co/bartowski/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf"

declare -A SIZES
SIZES[q4_km]="16 GB"
SIZES[q5_km]="18 GB"

declare -A SHA256_EXPECTED
# SHA256 will be filled after first download verification
SHA256_EXPECTED[q4_km]=""
SHA256_EXPECTED[q5_km]=""

# ─── Download function ────────────────────────────────────────────────────────
download_model() {
    local key="$1"
    local url="${MODELS[$key]}"
    local filename=$(basename "$url")
    local target="${MODEL_DIR}/${filename}"
    local size="${SIZES[$key]}"

    if [ -f "$target" ]; then
        echo -e "${GREEN}✓ $key: $filename already present (${size})${NC}"
        echo "  Path: $target"
        return 0
    fi

    echo -e "${YELLOW}⬇ $key: Downloading $filename (${size})...${NC}"
    echo "  URL: $url"
    echo "  Target: $target"
    echo ""

    # Use the project's hf.sh if available, otherwise wget
    if [ -x "${SCRIPT_DIR}/scripts/hf.sh" ]; then
        "${SCRIPT_DIR}/scripts/hf.sh" "$url" "$target"
    elif command -v wget &>/dev/null; then
        wget -c -O "$target" "$url"
    elif command -v curl &>/dev/null; then
        curl -L -o "$target" "$url"
    else
        echo -e "${RED}✗ No download tool found (wget, curl, hf.sh)${NC}"
        return 1
    fi

    if [ -f "$target" ]; then
        echo -e "${GREEN}✓ $key: Download complete${NC}"
        echo "  Path: $target"
        echo "  Size: $(du -h "$target" | cut -f1)"
    else
        echo -e "${RED}✗ $key: Download failed${NC}"
        return 1
    fi
}

# ─── Select models to download ────────────────────────────────────────────────
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   BFE Validation Model Downloader                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Model directory: $MODEL_DIR"
echo "Mode: $MODE"
echo ""

case "${MODE}" in
    all)
        for key in q4_km q5_km; do
            download_model "$key"
        done
        ;;
    q4_km)
        download_model "q4_km"
        ;;
    q5_km)
        download_model "q5_km"
        ;;
    *)
        echo "Usage: $0 [all|q4_km|q5_km]"
        exit 1
        ;;
esac

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}=== Model Summary ===${NC}"
echo ""
echo "Available models for BFE validation:"
echo ""

for key in q4_km q5_km; do
    url="${MODELS[$key]}"
    filename=$(basename "$url")
    target="${MODEL_DIR}/${filename}"
    size="${SIZES[$key]}"

    if [ -f "$target" ]; then
        actual_size=$(du -h "$target" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $key: $filename ($actual_size)"
        echo "    Path: $target"
        echo "    BFE target: dequantize_row_q4_K_cuda / dequantize_row_q5_K_cuda"
    else
        echo -e "  ${RED}✗${NC} $key: NOT DOWNLOADED ($size needed)"
    fi
    echo ""
done

echo "=== Next Steps ==="
echo ""
echo "1. Verify kernel dispatch:"
echo "   ./scripts/verify_kernel_dispatch.sh /home/stormrage/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf Q4_K_M"
echo ""
echo "2. Run A/B telemetry:"
echo "   ./scripts/run_ab_telemetry.sh /home/stormrage/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf RDNA2_BFE_DISPATCHER 5"
echo ""
echo "3. Run standardized benchmark:"
echo "   ./scripts/run_std_bench.sh /home/stormrage/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf dense-99"