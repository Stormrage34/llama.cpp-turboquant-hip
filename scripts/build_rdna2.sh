#!/bin/bash
# RDNA2 Build Script — llama.cpp-turboquant-hip
# Unified replacement for build_rdna2.sh + build_rdna2_llama.sh
#
# Usage:
#   ./scripts/build_rdna2.sh [mode] [options]
#
# Modes:
#   all        All optimizations (default, RDNA2_MATMUL_OPT_V1 runtime-gated)
#   stable     Production-safe, no experimental features
#   baseline   No RDNA2 optimizations
#
# Options:
#   --clean          Remove build dir before building
#   --verbose        Verbose cmake output
#   --benchmark      Also build llama-bench-rdna2 (hipcc, needs cmake first)
#   --no-interactive Skip ROCm selection prompt, use ROCM_PATH or default
#   --help           Show this message
#
# Environment:
#   ROCM_PATH        Path to ROCm installation (skip prompt if set)
#   LLAMA_BUILD_TARGETS  Space-separated cmake targets (default: llama-cli llama-server llama-bench)
#   OFFLOAD_ARCH     GPU architecture (default: gfx1030)
#
# ROCm versions available:
#   /opt/rocm                   = ROCm 7.2.1 (stable) — build with this
#   /home/stormrage/rocm-7.13-nightly = ROCm 7.13 (nightly) — newer kernels
#
# Difference:
#   Stable 7.2.1: cmake works cleanly, no .dll pollution. Use for BUILDING.
#   Nightly 7.13: newer runtime libs (hipblas 3.4, rocblas 5.4). Use for RUNNING at runtime.
#   Both use the same LLVM/clang 23.0.0 — generated GPU code is identical.
#   Build with stable, optionally LD_LIBRARY_PATH to nightly at runtime.

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

# ─── Colors ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── Defaults ──────────────────────────────────────────────────────────────
MODE="all"
CLEAN_BUILD=0
VERBOSE=0
BUILD_BENCHMARK=0
NO_INTERACTIVE=0
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1030}"
LLAMA_BUILD_TARGETS="${LLAMA_BUILD_TARGETS:-llama-cli llama-server llama-bench}"
ROCM_STABLE="/opt/rocm"
ROCM_NIGHTLY="/home/stormrage/rocm-7.13-nightly"
ROCM_PATH="${ROCM_PATH:-}"

# ─── Arg Parse ────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "${arg}" in
        --clean)        CLEAN_BUILD=1 ;;
        --verbose)      VERBOSE=1 ;;
        --benchmark)    BUILD_BENCHMARK=1 ;;
        --no-interactive) NO_INTERACTIVE=1 ;;
        --help|-h)
            sed -n '3,26p' "$0" | sed 's/^#//'; exit 0 ;;
        all|optimized|stable|baseline) MODE="${arg}" ;;
        *) echo -e "${RED}Unknown: ${arg}${NC}" >&2; exit 1 ;;
    esac
done

# ─── GPU Failback ─────────────────────────────────────────────────────────
source "${SCRIPT_DIR}/gpu_failback.sh"
gpu_failback_trap
gpu_acquire

# ─── Header ───────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   llama.cpp-turboquant-hip — RDNA2 Build Script    ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── ROCm Selection ──────────────────────────────────────────────────────
detect_rocm() {
    local path="$1"
    if [ -x "${path}/bin/hipcc" ] && [ -r "${path}/.info/version" ]; then
        echo "$(head -1 "${path}/.info/version")"
    elif [ -x "${path}/bin/hipcc" ]; then
        echo "detected (version unknown)"
    else
        echo ""
    fi
}

if [ -z "${ROCM_PATH}" ]; then
    STABLE_VER="$(detect_rocm "${ROCM_STABLE}")"
    NIGHTLY_VER="$(detect_rocm "${ROCM_NIGHTLY}")"

    if [ "${NO_INTERACTIVE}" -eq 1 ]; then
        # Auto-select: prefer stable for building
        if [ -n "${STABLE_VER}" ]; then
            ROCM_PATH="${ROCM_STABLE}"
        elif [ -n "${NIGHTLY_VER}" ]; then
            ROCM_PATH="${ROCM_NIGHTLY}"
        else
            echo -e "${RED}No ROCm found. Set ROCM_PATH.${NC}"; exit 1
        fi
    else
        echo -e "${CYAN}ROCm Installation Selection${NC}"
        echo ""
        echo "  Found these ROCm versions:"
        [ -n "${STABLE_VER}" ] && echo "    1) ${ROCM_STABLE}  (stable ${STABLE_VER})"
        [ -n "${NIGHTLY_VER}" ] && echo "    2) ${ROCM_NIGHTLY}  (nightly ${NIGHTLY_VER})"
        echo ""
        echo "  Differences:"
        echo "    Stable 7.2.1 — cmake builds cleanly, no .dll pollution."
        echo "                   Same LLVM/clang as nightly. Recommended for building."
        echo "    Nightly 7.13 — newer GPU runtime libs (hipblas/rocblas)."
        echo "                   bin/ has .dll files that confuse cmake."
        echo "                   Build with stable, then LD_LIBRARY_PATH to nightly at runtime."
        echo ""

        # Check if both are available
        if [ -n "${STABLE_VER}" ] && [ -n "${NIGHTLY_VER}" ]; then
            echo -e "  ${YELLOW}Both available — stable is preferred for building.${NC}"
            read -r -p "  Choose [1/2, default 1]: " choice
            case "${choice}" in
                2|2a|nightly) ROCM_PATH="${ROCM_NIGHTLY}" ;;
                *)            ROCM_PATH="${ROCM_STABLE}" ;;
            esac
        elif [ -n "${STABLE_VER}" ]; then
            echo "  Using: ${ROCM_STABLE} (${STABLE_VER})"
            ROCM_PATH="${ROCM_STABLE}"
        elif [ -n "${NIGHTLY_VER}" ]; then
            echo "  Using: ${ROCM_NIGHTLY} (${NIGHTLY_VER})"
            ROCM_PATH="${ROCM_NIGHTLY}"
        else
            echo -e "${RED}No ROCm found at ${ROCM_STABLE} or ${ROCM_NIGHTLY}${NC}"
            echo "  Set ROCM_PATH to your installation."
            exit 1
        fi
    fi
fi
export ROCM_PATH

HIPCC="${ROCM_PATH}/bin/hipcc"
echo -e "${GREEN}✓ ROCm: ${ROCM_PATH}${NC}"
if [ -r "${ROCM_PATH}/.info/version" ]; then
    echo -e "${GREEN}✓ Version: $(head -1 "${ROCM_PATH}/.info/version")${NC}"
fi
echo ""

# ─── Mode ────────────────────────────────────────────────────────────────
case "${MODE}" in
    all|optimized)
        echo -e "${GREEN}Mode: ${BOLD}All optimizations${NC}"
        echo "  LLVM compiler flags (-mllvm -amdgpu-* for gfx1030 decode)"
        RUN_ENV="" ;;
    stable)
        echo -e "${GREEN}Mode: ${BOLD}Stable only${NC}"
        echo "  LLVM compiler flags (-mllvm -amdgpu-*) always applied"
        RUN_ENV="" ;;
    baseline)
        echo -e "${YELLOW}Mode: ${BOLD}Baseline (no RDNA2 optimizations)${NC}"
        RUN_ENV="" ;;
esac
echo ""

# ─── Prerequisites ──────────────────────────────────────────────────────
echo -e "${CYAN}Checking prerequisites...${NC}"
command -v cmake &>/dev/null || { echo -e "${RED}✗ cmake not found${NC}"; exit 1; }
[ -x "${HIPCC}" ] || { echo -e "${RED}✗ hipcc not found at ${HIPCC}${NC}"; exit 1; }
echo -e "${GREEN}✓ cmake: $(cmake --version | head -1)${NC}"
echo -e "${GREEN}✓ hipcc: ${HIPCC}${NC}"
echo ""

# ─── Clean ──────────────────────────────────────────────────────────────
if [ "${CLEAN_BUILD}" -eq 1 ]; then
    echo -e "${YELLOW}Cleaning: ${BUILD_DIR}${NC}"
    rm -rf "${BUILD_DIR}"
    echo ""
fi

# ─── PATH setup for cmake ───────────────────────────────────────────────
# ROCm nightly's bin/ has .dll files that confuse cmake. Put /opt/rocm first
# in PATH if building with stable, or ensure the right path is first.
# We use the tool's HIPCC directly via -DCMAKE_HIP_COMPILER to be unambiguous.
echo -e "${CYAN}Configuring CMake...${NC}"
echo "  Build dir: ${BUILD_DIR}"
echo "  Arch:      ${OFFLOAD_ARCH}"
echo "  Targets:   ${LLAMA_BUILD_TARGETS}"
echo ""

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS:STRING="${OFFLOAD_ARCH}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_HIP_COMPILER="${HIPCC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
echo ""

# ─── CMake Build ────────────────────────────────────────────────────────
echo -e "${CYAN}Building targets: ${LLAMA_BUILD_TARGETS}...${NC}"
echo ""

BUILD_OPTS=(--config Release -- -j "$(nproc)")
[ "${VERBOSE}" -eq 1 ] && BUILD_OPTS+=(--verbose)

for target in ${LLAMA_BUILD_TARGETS}; do
    echo -e "  ${GREEN}→ ${target}${NC}"
    cmake --build "${BUILD_DIR}" --target "${target}" "${BUILD_OPTS[@]}"
done
echo ""

# ─── Benchmark Binary (hipcc) ───────────────────────────────────────────
if [ "${BUILD_BENCHMARK}" -eq 1 ]; then
    echo -e "${CYAN}Building llama-bench-rdna2 (hipcc)...${NC}"

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

    ${HIPCC} -O3 -DNDEBUG --offload-arch="${OFFLOAD_ARCH}" \
        ${INCLUDES} \
        -o "${BIN_DIR}/llama-bench-rdna2" \
        "${PROJECT_ROOT}/tools/llama-bench/llama-bench.cpp" \
        ${LIBS} \
        -Wl,-rpath,"${BIN_DIR}"

    echo -e "${GREEN}✓ llama-bench-rdna2 compiled${NC}"
    echo ""
fi

# ─── Summary ────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              Build Complete                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}ROCm:${NC} ${ROCM_PATH}"
echo -e "${BOLD}RPATH isolation:${NC} enabled (prevents LD_LIBRARY_PATH cross-contamination)"
echo ""
echo -e "${BOLD}Binaries:${NC}"

count=0
for target in ${LLAMA_BUILD_TARGETS}; do
    if [ -f "${BIN_DIR}/${target}" ]; then
        echo "  ${BIN_DIR}/${target}"; count=$((count + 1))
    fi
done
[ "${count}" -eq 0 ] && echo "  (check build output above)"
if [ "${BUILD_BENCHMARK}" -eq 1 ] && [ -f "${BIN_DIR}/llama-bench-rdna2" ]; then
    echo "  ${BIN_DIR}/llama-bench-rdna2"
fi
echo ""

echo -e "${BOLD}Verify library isolation:${NC}"
echo "  readelf -d ${BIN_DIR}/llama-cli | grep RPATH"
echo "  ldd ${BIN_DIR}/llama-cli | grep llama"
echo ""

if [ -n "${RUN_ENV}" ]; then
    echo -e "${BOLD}Run:${NC}"
    echo "  ${RUN_ENV} ${BIN_DIR}/llama-cli -m model.gguf -ngl 99"
    echo ""
    echo -e "${YELLOW}See docs/rdna2-experimental.md for details.${NC}"
fi

echo -e "${GREEN}Done.${NC}"
