#!/bin/bash
# RDNA2 CMake Build Wrapper — llama.cpp-turboquant-hip
# Usage: ./scripts/build_rdna2.sh [all|stable|baseline] [--clean] [--verbose]
# Env: OFFLOAD_ARCH (gfx1030), ROCM_PATH, LLAMA_BUILD_TARGETS (llama-cli)
# Coexists with the hipcc-based build_rdna2_llama.sh.

set -euo pipefail

# ─── GPU Failback ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/gpu_failback.sh"
gpu_failback_trap
gpu_acquire

# ─── ROCm Path Detection ───────────────────────────────────────────────────
# Set ROCM_PATH before running if your install is non-standard, e.g.:
#   export ROCM_PATH=/home/stormrage/rocm-7.13-nightly
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1030}"
LLAMA_BUILD_TARGETS="${LLAMA_BUILD_TARGETS:-llama-cli}"

# ─── Colors ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── Argument Parsing ──────────────────────────────────────────────────────
MODE="all"; CLEAN_BUILD=0; VERBOSE=0
for arg in "$@"; do
    case "${arg}" in
        --clean)   CLEAN_BUILD=1 ;;
        --verbose) VERBOSE=1     ;;
        --help|-h) echo "Usage: $0 [all|stable|baseline] [--clean] [--verbose]"; exit 0 ;;
        all|optimized|stable|baseline) MODE="${arg}" ;;
        *) echo -e "${RED}Unknown: ${arg}${NC}" >&2; exit 1 ;;
    esac
done

# ─── Header ────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   llama.cpp-turboquant-hip — RDNA2 CMake Build     ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}" && echo ""

# ─── ROCm Version Check ────────────────────────────────────────────────────
# Best-effort version display. CMake validates the minimum (6.1) at configure time.
echo -e "${CYAN}Checking ROCm...${NC}"
if [ -x "${ROCM_PATH}/bin/hipcc" ]; then
    echo -e "${GREEN}✓ hipcc: ${ROCM_PATH}/bin/hipcc${NC}"
else
    echo -e "${RED}✗ hipcc not found at ${ROCM_PATH}/bin/hipcc${NC}"
    echo "  Set ROCM_PATH to your ROCm installation, e.g.:"
    echo "    export ROCM_PATH=/opt/rocm"
    echo "    export ROCM_PATH=/home/stormrage/rocm-7.13-nightly"
    exit 1
fi
if [ -r "${ROCM_PATH}/.info/version" ]; then
    echo -e "${GREEN}✓ ROCm $(head -1 "${ROCM_PATH}/.info/version")${NC}"
else
    echo -e "${YELLOW}⚠ ROCm version unknown (CMake will validate ≥6.1)${NC}"
fi
echo ""

# ─── Prerequisites ─────────────────────────────────────────────────────────
echo -e "${CYAN}Checking prerequisites...${NC}"
command -v cmake &>/dev/null || { echo -e "${RED}✗ cmake not found${NC}"; exit 1; }
[ -d "${ROCM_PATH}" ]         || { echo -e "${RED}✗ ${ROCM_PATH} not found${NC}"; exit 1; }
echo -e "${GREEN}✓ cmake: $(cmake --version | head -1)${NC}"
echo -e "${GREEN}✓ ROCm path: ${ROCM_PATH}${NC}" && echo ""

# ─── Mode → Runtime Environment ─────────────────────────────────────────────
# Phantom gates removed per v0.3.2-alpha "Ghost Protocol":
#   RDNA2_OPT_V1 (BFE dequant) — targets cold standalone dequant path, ~0% impact
#   RDNA2_ASYNC_PIPELINE       — had zero kernel implementation
# RDNA2_MATMUL_OPT_V1 remains as runtime-only dispatch (no compile gate).
case "${MODE}" in
    all|optimized)
        echo -e "${GREEN}Mode: ${BOLD}All optimizations${NC}"
        echo "  RDNA2_MATMUL_OPT_V1 runtime gate (LDS double-buffer matmul)"
        RUN_ENV="RDNA2_MATMUL_OPT_V1=1"
        RUN_STABLE="RDNA2_MATMUL_OPT_V1=1" ;;
    stable)
        echo -e "${GREEN}Mode: ${BOLD}Stable only${NC}"
        echo "  No RDNA2 env gates enabled by default"
        RUN_ENV=""
        RUN_STABLE="" ;;
    baseline)
        echo -e "${YELLOW}Mode: ${BOLD}Baseline (no RDNA2 optimizations)${NC}"
        RUN_ENV=""
        RUN_STABLE="" ;;
esac
echo ""

# ─── Clean Build ───────────────────────────────────────────────────────────
[ "${CLEAN_BUILD}" -eq 1 ] && { echo -e "${YELLOW}Cleaning: ${BUILD_DIR}${NC}"; rm -rf "${BUILD_DIR}"; echo ""; }

# ─── CMake Configure ───────────────────────────────────────────────────────
echo -e "${CYAN}Configuring CMake...${NC}"
echo "  Build dir: ${BUILD_DIR}"
echo "  Arch:      ${OFFLOAD_ARCH}"
echo "  Targets:   ${LLAMA_BUILD_TARGETS}"
echo ""

cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="${OFFLOAD_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release
echo ""

# ─── CMake Build ───────────────────────────────────────────────────────────
echo -e "${CYAN}Building targets: ${LLAMA_BUILD_TARGETS}...${NC}" && echo ""
BUILD_OPTS=(--parallel "$(nproc)")
[ "${VERBOSE}" -eq 1 ] && BUILD_OPTS+=(--verbose)
for target in ${LLAMA_BUILD_TARGETS}; do
    echo -e "  ${GREEN}→ ${target}${NC}"
    cmake --build "${BUILD_DIR}" --target "${target}" "${BUILD_OPTS[@]}"
done
echo "" && echo -e "${GREEN}✓ Build complete${NC}" && echo ""

# ─── Summary ───────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              Build Complete                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}Binaries:${NC}"
count=0
for target in ${LLAMA_BUILD_TARGETS}; do
    if [ -f "${BIN_DIR}/${target}" ]; then
        echo "  ${BIN_DIR}/${target}"; count=$((count + 1))
    fi
done
[ "${count}" -eq 0 ] && echo "  (check build output above)"
echo ""

if [ -n "${RUN_STABLE}" ] || [ -n "${RUN_ENV}" ]; then
    EXAMPLE_BIN="${BIN_DIR}/llama-cli"
    [ -f "${BIN_DIR}/llama-server" ] && EXAMPLE_BIN="${BIN_DIR}/llama-server"
    [ -f "${BIN_DIR}/llama-bench" ]  && EXAMPLE_BIN="${BIN_DIR}/llama-bench"

    echo -e "${BOLD}Run with:${NC}"
    [ -n "${RUN_STABLE}" ] && echo "  ${RUN_STABLE} ${EXAMPLE_BIN} -m model.gguf -ngl 99"
    if [ -n "${RUN_STABLE}" ] && [ -n "${RUN_ENV}" ]; then
        echo ""; echo "  # + Experimental LDS matmul (benchmark first):"
    fi
    [ -n "${RUN_ENV}" ] && echo "  ${RUN_ENV} ${EXAMPLE_BIN} -m model.gguf -ngl 99"
    echo ""; echo -e "${YELLOW}See docs/rdna2-experimental.md for details.${NC}"
fi

echo "" && echo -e "${GREEN}Done.${NC}"
