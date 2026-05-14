#!/bin/bash
# RDNA2 Build Script for llama.cpp-turboquant-hip
# Compiles with RDNA2 optimizations for AMD GPUs (gfx1030/gfx1100)
#
# Usage:
#   ./scripts/build_rdna2_llama.sh              # Build with all optimizations
#   ./scripts/build_rdna2_llama.sh stable       # Build stable features only
#   ./scripts/build_rdna2_llama.sh baseline     # Build without RDNA2 optimizations

set -e

# GPU failback — free VRAM before compilation
source "$(cd "$(dirname "$0")" && pwd)/gpu_failback.sh"
gpu_failback_trap
gpu_acquire

# ─── Configuration ───────────────────────────────────────────────────────────
# ROCm path — adjust if your installation differs
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"

# Project paths — works from repo root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

# Offload architecture (default: gfx1030 = RDNA2)
OFFLOAD_ARCH="${OFFLOAD_ARCH:---offload-arch=gfx1030}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Header ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   llama.cpp-turboquant-hip — RDNA2 Build Script    ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── Parse Mode ──────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "${MODE}" in
    all|optimized)
        echo -e "${GREEN}Mode: ${BOLD}All optimizations${NC}"
        echo "  - BFE dequantization kernel (stable)"
        echo "  - Async pipeline (stable)"
        echo "  - LDS double-buffer matmul (experimental)"
        RDNA2_FLAGS="-DRDNA2_OPT_V1=1 -DRDNA2_ASYNC_PIPELINE=1 -DRDNA2_MATMUL_OPT_V1=1"
        ;;
    stable)
        echo -e "${GREEN}Mode: ${BOLD}Stable only${NC}"
        echo "  - BFE dequantization kernel"
        echo "  - Async pipeline"
        RDNA2_FLAGS="-DRDNA2_OPT_V1=1 -DRDNA2_ASYNC_PIPELINE=1"
        ;;
    baseline)
        echo -e "${YELLOW}Mode: ${BOLD}Baseline (no RDNA2 optimizations)${NC}"
        RDNA2_FLAGS=""
        ;;
    *)
        echo "Usage: $0 [all|stable|baseline]"
        echo ""
        echo "  all       - Build with all RDNA2 optimizations (default)"
        echo "  stable    - Build stable features only (production-ready)"
        echo "  baseline  - Build without RDNA2 optimizations"
        exit 1
        ;;
esac

echo ""

# ─── Prerequisites ───────────────────────────────────────────────────────────
echo -e "${CYAN}Checking prerequisites...${NC}"

if [ ! -x "${HIPCC}" ]; then
    echo -e "${RED}✗ hipcc not found at ${HIPCC}${NC}"
    echo "  Set ROCM_PATH environment variable or install ROCm."
    exit 1
fi
echo -e "${GREEN}✓ hipcc: ${HIPCC}${NC}"

if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}✗ Build directory not found: ${BUILD_DIR}${NC}"
    echo "  Run CMake build first: cmake -S . -B build && cmake --build build"
    exit 1
fi
echo -e "${GREEN}✓ Build dir: ${BUILD_DIR}${NC}"

MISSING_LIBS=0
for lib in libggml-hip.so libggml-base.so libggml-cpu.so libggml.so libllama.so; do
    if [ ! -f "${BIN_DIR}/${lib}" ] && [ ! -L "${BIN_DIR}/${lib}" ]; then
        echo -e "${YELLOW}⚠ ${lib} not found in ${BIN_DIR}${NC}"
        MISSING_LIBS=1
    fi
done
if [ "${MISSING_LIBS}" -eq 0 ]; then
    echo -e "${GREEN}✓ All baseline libraries present${NC}"
fi

echo ""

# ─── Build ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}Compiling llama-bench-rdna2...${NC}"
echo "  Flags: ${RDNA2_FLAGS:-none}"
echo "  Arch:  ${OFFLOAD_ARCH}"
echo ""

# Include paths
INCLUDES="-I${PROJECT_ROOT}/ggml/src/../include \
-I${PROJECT_ROOT}/src/../include \
-I${PROJECT_ROOT}/common/. \
-I${PROJECT_ROOT}/common/../vendor \
-I${PROJECT_ROOT}/tools/server \
-I${PROJECT_ROOT}/tools/server/../mtmd \
-I${PROJECT_ROOT}/tools/mtmd/. \
-I${PROJECT_ROOT} \
-I${ROCM_PATH}/include"

LIBS="-L${BIN_DIR} -lggml-hip -lggml-base -lggml-cpu -lggml -lllama -lllama-common -lamdhip64 -lpthread"

${HIPCC} -O3 -DNDEBUG ${OFFLOAD_ARCH} ${RDNA2_FLAGS} \
    ${INCLUDES} \
    -o "${BIN_DIR}/llama-bench-rdna2" \
    "${PROJECT_ROOT}/tools/llama-bench/llama-bench.cpp" \
    ${LIBS} \
    -Wl,-rpath,"${BIN_DIR}" \
    2>&1

echo -e "${GREEN}✓ llama-bench-rdna2 compiled${NC}"

# Check for llama-server
if [ -f "${BIN_DIR}/llama-server" ]; then
    echo -e "${GREEN}✓ llama-server found (CMake-built)${NC}"
else
    echo -e "${YELLOW}⚠ llama-server not found — run full CMake build:${NC}"
    echo "  cmake --build build --target llama-server"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              Build Complete                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Binaries:${NC}"
echo "  ${BIN_DIR}/llama-bench-rdna2"
[ -f "${BIN_DIR}/llama-server" ] && echo "  ${BIN_DIR}/llama-server"
echo ""

if [ "${MODE}" != "baseline" ]; then
    echo -e "${BOLD}Run with:${NC}"
    if [ "${MODE}" = "stable" ]; then
        echo "  RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ${BIN_DIR}/llama-server -m model.gguf -ngl 99"
    else
            echo "  # Stable (production):"
            echo "  RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ${BIN_DIR}/llama-server -m model.gguf -ngl 99"
            echo ""
            echo "  # + Experimental MoE prefill (benchmark first):"
            echo "  RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 ${BIN_DIR}/llama-server -m model.gguf -ngl 99"
    fi
    echo ""
    echo -e "${YELLOW}See docs/rdna2-experimental.md for details.${NC}"
fi
