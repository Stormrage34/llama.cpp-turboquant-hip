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
#   --verbose        Verbose cmake output
#   --benchmark      Also build llama-bench-rdna2 (hipcc, needs cmake first)
#   --no-interactive Skip ROCm selection prompt, use ROCM_PATH or default
#   --help           Show this message

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
        --verbose)       VERBOSE=1 ;;
        --benchmark)     BUILD_BENCHMARK=1 ;;
        --no-interactive) NO_INTERACTIVE=1 ;;
        --help|-h)
            sed -n '3,18p' "$0" | sed 's/^#//'; exit 0 ;;
        all|optimized|stable|baseline) MODE="${arg}" ;;
        *) echo -e "${RED}Unknown: ${arg}${NC}" >&2; exit 1 ;;
    esac
done

# ─── Header ───────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║    llama.cpp-turboquant-hip — RDNA2 Build Script     ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ─── ROCm Selection Logic ──────────────────────────────────────────────────
detect_rocm() {
    local path="$1"
    if [ -x "${path}/bin/hipcc" ]; then
        if [ -r "${path}/.info/version" ]; then
            read -r first_line < "${path}/.info/version"
            echo "${first_line}"
        else
            echo "detected (version unknown)"
        fi
    else
        echo ""
    fi
}

if [ -z "${ROCM_PATH}" ]; then
    STABLE_VER="$(detect_rocm "${ROCM_STABLE}")"
    NIGHTLY_VER="$(detect_rocm "${ROCM_NIGHTLY}")"

    if [ "${NO_INTERACTIVE}" -eq 1 ]; then
        if [ -n "${NIGHTLY_VER}" ]; then
            ROCM_PATH="${ROCM_NIGHTLY}"
        elif [ -n "${STABLE_VER}" ]; then
            ROCM_PATH="${ROCM_STABLE}"
        else
            echo -e "${RED}No ROCm environments found automatically. Fallback required.${NC}"
            NO_INTERACTIVE=0 # Force interactivity to prompt user for preferred path
        fi
    fi

    if [ "${NO_INTERACTIVE}" -eq 0 ]; then
        echo -e "${CYAN}ROCm Installation Selection${NC}"
        echo ""
        echo "  Found these ROCm versions:"
        [ -n "${STABLE_VER}" ] && echo "    1) ${ROCM_STABLE}  (stable ${STABLE_VER})"
        [ -n "${NIGHTLY_VER}" ] && echo "    2) ${ROCM_NIGHTLY}  (nightly ${NIGHTLY_VER})"

        # If neither path was auto-detected
        if [ -z "${STABLE_VER}" ] && [ -z "${NIGHTLY_VER}" ]; then
            echo -e "    ${YELLOW}No default installations detected at /opt/rocm or nightly path.${NC}"
            while true; do
                read -r -p "  Enter custom ROCm path directory: " custom_path
                if [ -x "${custom_path}/bin/hipcc" ]; then
                    ROCM_PATH="${custom_path}"
                    break
                else
                    echo -e "  ${RED}Invalid ROCm directory (bin/hipcc executable missing). Try again.${NC}"
                fi
            done
        elif [ -n "${STABLE_VER}" ] && [ -n "${NIGHTLY_VER}" ]; then
            echo ""
            echo -e "  ${YELLOW}Both available — stable is preferred for building unless tracking nightly features.${NC}"
            read -r -p "  Choose [1/2, default 2 (Nightly)]: " choice
            case "${choice}" in
                1|stable) ROCM_PATH="${ROCM_STABLE}" ;;
                *)        ROCM_PATH="${ROCM_NIGHTLY}" ;;
            esac
        elif [ -n "${STABLE_VER}" ]; then
            ROCM_PATH="${ROCM_STABLE}"
        else
            ROCM_PATH="${ROCM_NIGHTLY}"
        fi
    fi
fi
export ROCM_PATH

HIPCC="${ROCM_PATH}/bin/hipcc"
CLANG_HIP="${ROCM_PATH}/llvm/bin/clang++"

echo -e "${GREEN}✓ ROCm Path: ${ROCM_PATH}${NC}"
echo ""

# ─── Mode & Compiler Flags ───────────────────────────────────────────────
HIP_CXX_FLAGS=""
RUN_ENV=""

case "${MODE}" in
    all|optimized)
        echo -e "${GREEN}Mode: ${BOLD}All optimizations${NC}"
        HIP_CXX_FLAGS="-mllvm -amdgpu-early-inline-all=true"
        RUN_ENV="RDNA2_MATMUL_OPT_V1=1" ;;
    stable)
        echo -e "${GREEN}Mode: ${BOLD}Stable only${NC}"
        HIP_CXX_FLAGS="-mllvm -amdgpu-early-inline-all=true" ;;
    baseline)
        echo -e "${YELLOW}Mode: ${BOLD}Baseline (no RDNA2 optimizations)${NC}" ;;
esac
echo ""

# ─── Linker Isolation Setup ─────────────────────────────────────────────
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:-}"

# ─── Prerequisites ──────────────────────────────────────────────────────
echo -e "${CYAN}Checking prerequisites...${NC}"
command -v cmake &>/dev/null || { echo -e "${RED}✗ cmake not found${NC}"; exit 1; }
echo -e "${GREEN}✓ cmake verified${NC}"
echo ""

# ─── Clean Build Routine ─────────────────────────────────────────────────
echo -e "${YELLOW}Cleaning build tree: ${BUILD_DIR}${NC}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
echo ""

# ─── CMake Strategy Execution ───────────────────────────────────────────
echo -e "${CYAN}Configuring CMake...${NC}"

BUILD_OPTS=(--config Release)
[ "${VERBOSE}" -eq 1 ] && BUILD_OPTS+=(--verbose)
BUILD_OPTS+=(-- -j "$(nproc)")

echo -e "${YELLOW}Executing Configuration Strategy: Native Clang Execution...${NC}"

# Optimized using clean $ORIGIN rpaths to avoid runtime dependency failure
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS:STRING="${OFFLOAD_ARCH}" \
    -DAMDGPU_TARGETS:STRING="${OFFLOAD_ARCH}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH};${ROCM_PATH}/llvm" \
    -DCMAKE_LIBRARY_PATH="${ROCM_PATH}/lib;${ROCM_PATH}/lib64" \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_COMPILER="${CLANG_HIP}" \
    -DCMAKE_HIP_FLAGS="${HIP_CXX_FLAGS}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRDNA2_MOE_STREAM_V1=ON \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN;\$ORIGIN/../lib;\$ORIGIN/../bin;${ROCM_PATH}/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${ROCM_PATH}/lib -L${ROCM_PATH}/lib" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${ROCM_PATH}/lib -L${ROCM_PATH}/lib"
echo ""

# ─── CMake Build Execution (Efficiency Optimization) ─────────────────────
# Efficient: Rather than invoking cmake sequentially in a loop per target
# (which repeats initialization overhead), pass targets as a space-separated array
# to parse natively in parallel.
echo -e "${CYAN}Building targets parallelized: ${LLAMA_BUILD_TARGETS}...${NC}"
echo ""

# Convert space separated string targets directly into single matrix array command
IFS=' ' read -r -a TARGET_ARRAY <<< "${LLAMA_BUILD_TARGETS}"
cmake --build "${BUILD_DIR}" --target "${TARGET_ARRAY[@]}" "${BUILD_OPTS[@]}"
echo ""

# ─── Benchmark Binary (Raw hipcc compilation fallback) ──────────────────
if [ "${BUILD_BENCHMARK}" -eq 1 ]; then
    echo -e "${CYAN}Building llama-bench-rdna2 (via hipcc manual integration)...${NC}"

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

    export HIP_COMPILER=clang
    export HIP_DEVICE_COMPILER="${CLANG_HIP}"

    ${HIPCC} -O3 -DNDEBUG --offload-arch="${OFFLOAD_ARCH}" ${HIP_CXX_FLAGS} \
        ${INCLUDES} \
        -o "${BIN_DIR}/llama-bench-rdna2" \
        "${PROJECT_ROOT}/tools/llama-bench/llama-bench.cpp" \
        ${LIBS} \
        -Wl,-rpath,"\$ORIGIN" \
        -Wl,-rpath,"${ROCM_PATH}/lib"

    echo -e "${GREEN}✓ llama-bench-rdna2 compiled${NC}"
    echo ""
fi

# ─── Summary ────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                Build Complete                        ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Target Architecture:${NC}   ${OFFLOAD_ARCH}"
echo -e "${BOLD}RPATH isolation:${NC}       enabled (locked to chosen ROCm tree)"
echo ""
if [ -n "${RUN_ENV}" ]; then
    echo -e "${BOLD}Run configuration environment optimized for execution:${NC}"
    echo "  env ${RUN_ENV} ${BIN_DIR}/llama-cli -m model.gguf -ngl 99"
else
    echo "  ${BIN_DIR}/llama-cli -m model.gguf -ngl 99"
fi
echo ""
echo -e "${GREEN}Done.${NC}"
