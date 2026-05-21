#!/bin/bash
# RDNA2 Build Script — llama.cpp-turboquant-hip
# Unified replacement for build_rdna2.sh + build_rdna2_llama.sh
#
# Usage:
#   ./scripts/build_rdna2.sh [mode] [options]
#
# Modes:
#   all        All optimizations (default, lds_bank_pad trait-gated)
#   stable     Production-safe, no experimental features
#   baseline   No RDNA2 optimizations
#
# Options:
#   --verbose        Verbose cmake output
#   --benchmark      Also build llama-bench-rdna2 (hipcc, needs cmake first)
#   --no-interactive Skip ROCm selection prompt, use ROCM_PATH or default
#   --swizzle        Enable cache-aware SoA swizzle for IQ4_XS (experimental)
#   --swizzle-all    Enable SoA swizzle for ALL K-quants (Q4_K, Q5_K, IQ4_XS) — CR-020 experimental
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
FAST_MODE=0
SWIZZLE_MODE=0
SWIZZLE_ALL_MODE=0
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
        --fast)          FAST_MODE=1 ;;
        --swizzle)       SWIZZLE_MODE=1 ;;
        --swizzle-all)   SWIZZLE_MODE=1; SWIZZLE_ALL_MODE=1 ;;
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
# Defaults: MOE_STREAM_V1=ON, VGPR_OPT_V1=ON (baseline overrides VGPR_OPT to OFF)
CMAKE_EXTRA_FLAGS=""

case "${MODE}" in
    all|optimized)
        echo -e "${GREEN}Mode: ${BOLD}All optimizations${NC}"
        HIP_CXX_FLAGS="-mllvm -amdgpu-early-inline-all=true"
        ;;
    stable)
        echo -e "${GREEN}Mode: ${BOLD}Stable only${NC}"
        HIP_CXX_FLAGS="-mllvm -amdgpu-early-inline-all=true" ;;
    baseline)
        echo -e "${YELLOW}Mode: ${BOLD}Baseline (no RDNA2 optimizations)${NC}"
        CMAKE_EXTRA_FLAGS="-DRDNA2_VGPR_OPT_V1=OFF" ;;
esac

# Swizzle mode: add RDNA2_CACHE_SWIZZLE flag
if [ "${SWIZZLE_MODE}" -eq 1 ]; then
    if [ "${SWIZZLE_ALL_MODE}" -eq 1 ]; then
        echo -e "${YELLOW}Cache-aware SoA swizzle: ALL QUANTS (Q4_K, Q5_K, IQ4_XS) — CR-020 experimental${NC}"
        CMAKE_EXTRA_FLAGS="${CMAKE_EXTRA_FLAGS} -DRDNA2_CACHE_SWIZZLE=ON -DLLAMA_BUILD_SWIZZLE_DEV=ON"
    else
        echo -e "${YELLOW}Cache-aware SoA swizzle: IQ4_XS only (experimental)${NC}"
        CMAKE_EXTRA_FLAGS="${CMAKE_EXTRA_FLAGS} -DRDNA2_CACHE_SWIZZLE=ON"
    fi
fi
echo ""

# ─── Linker Isolation Setup ─────────────────────────────────────────────
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:-}"

# ─── Stale Binary Detection ───────────────────────────────────────────────
# Compares binary timestamps against their library dependencies.
# If a shared lib is newer than the binary that links it, a partial rebuild
# was done (common cause of SIGSEGV on HIP backend init).
# 
# --fast MODE WARNING: This check runs ONLY in --fast mode (incremental rebuilds).
# A partial rebuild (e.g., `cmake --build build --target libggml-hip`) creates version skew:
#   - llama-server still links old libggml-hip → SIGSEGV on startup (even --help)
#   - llama-cli may work (different memory layout, masks the issue)
# 
# Detection logic: For each binary (llama-server, llama-cli, llama-bench), check if any
# of its dependent libraries (libggml-hip.so.0, libggml-cpu.so.0, etc.) have a newer
# modification time. If found, warn user that full rebuild is required.
# 
# False positive tolerance: If stale binaries detected but user insists on --fast,
# the script continues with a warning (line 202-204).
check_stale_binaries() {
    local stale_found=0
    local bins=("llama-server" "llama-cli" "llama-bench")
    local libs=("libggml-hip.so.0" "libggml-cpu.so.0" "libggml-base.so.0" "libllama.so.0" "libllama-common.so.0")
    for bin_name in "${bins[@]}"; do
        local bin_path="${BIN_DIR}/${bin_name}"
        [[ -f "$bin_path" ]] || continue
        for lib_name in "${libs[@]}"; do
            local lib_path="${BIN_DIR}/${lib_name}"
            [[ -f "$lib_path" ]] || continue
            if [[ "$lib_path" -nt "$bin_path" ]]; then
                echo -e "${RED}⚠ STALE BINARY: ${bin_name} is older than ${lib_name}${NC}"
                echo -e "${YELLOW}  → Partial rebuild detected. Full rebuild required.${NC}"
                stale_found=1
            fi
        done
    done
    # Also check alternative build directories (build-swizzle etc.)
    for alt_build in "${PROJECT_ROOT}"/build-*; do
        [[ -d "$alt_build/bin" ]] || continue
        [[ "$alt_build" == "$BUILD_DIR" ]] && continue  # skip main build dir
        for bin_name in "${bins[@]}"; do
            local bin_path="${alt_build}/bin/${bin_name}"
            [[ -f "$bin_path" ]] || continue
            for lib_name in "${libs[@]}"; do
                local lib_path="${alt_build}/bin/${lib_name}"
                [[ -f "$lib_path" ]] || continue
                if [[ "$lib_path" -nt "$bin_path" ]]; then
                    echo -e "${RED}⚠ STALE BINARY in $(basename "$alt_build"): ${bin_name} is older than ${lib_name}${NC}"
                    echo -e "${YELLOW}  → Partial rebuild detected. Full rebuild required.${NC}"
                    stale_found=1
                fi
            done
        done
    done
    return $stale_found
}

# ─── Prerequisites ──────────────────────────────────────────────────────
echo -e "${CYAN}Checking prerequisites...${NC}"
command -v cmake &>/dev/null || { echo -e "${RED}✗ cmake not found${NC}"; exit 1; }
echo -e "${GREEN}✓ cmake verified${NC}"
echo ""

# ─── Build Parallelism ───────────────────────────────────────────────────
# Must be defined BEFORE fast-mode block (BUILD_OPTS used at line ~251)
BUILD_OPTS=(--config Release)
[ "${VERBOSE}" -eq 1 ] && BUILD_OPTS+=(--verbose)
BUILD_OPTS+=(-- -j "$(nproc)")

# Check for stale binaries (partial rebuilds that cause SIGSEGV)
if [ "$FAST_MODE" -eq 1 ]; then
    echo -e "${YELLOW}Fast mode: skipping clean build, rebuilding modified targets only${NC}"

    # CMake must be configured first
    if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
        echo -e "${RED}CMake not configured. Run without --fast first.${NC}"
        exit 1
    fi

    if check_stale_binaries; then
        echo -e "${YELLOW}No stale binaries detected. Proceeding with fast rebuild.${NC}"
    else
        echo -e "${RED}Stale binaries detected. Use full rebuild (remove --fast) to fix.${NC}"
        echo -e "${YELLOW}Continuing anyway — SIGSEGV risk during inference.${NC}"
    fi
    cmake --build "${BUILD_DIR}" --target ${LLAMA_BUILD_TARGETS} "${BUILD_OPTS[@]}"
    exit 0
fi

# ─── Clean Build Routine ─────────────────────────────────────────────────
echo -e "${YELLOW}Cleaning build tree: ${BUILD_DIR}${NC}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
echo ""

# ─── CMake Strategy Execution ───────────────────────────────────────────
echo -e "${CYAN}Configuring CMake...${NC}"

echo -e "${YELLOW}Executing Configuration Strategy: Native Clang Execution...${NC}"

# Optimized using clean $ORIGIN rpaths to avoid runtime dependency failure
# 
# RPATH ISOLATION FLAGS (CRITICAL for build isolation):
#   -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
#     → Embeds $ORIGIN in RPATH (not RUNPATH), ensuring binaries find their
#       bundled libraries first, not system-wide llama.cpp installations.
# 
#   -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags ..."
#   -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags ..."
#     → Forces RPATH over RUNPATH (ld.so behavior). RUNPATH is checked AFTER
#       LD_LIBRARY_PATH, which can pull in incompatible libraries from other
#       llama.cpp forks (e.g., upstream, koboldcpp, etc.) → ABI mismatch → segfault.
# 
#   -DCMAKE_INSTALL_RPATH="\$ORIGIN;\$ORIGIN/../lib;..."
#     → Runtime rpath baked into binaries. $ORIGIN = directory containing the binary.
# 
# Why this matters: Without --disable-new-dtags, a user with multiple llama.cpp
# builds can get libggml-hip.so from the wrong tree, causing GPU init crashes.
# This is a P0 fix for cross-fork coexistence on the same system.
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
    -DRDNA2_VGPR_OPT_V1=ON \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    ${CMAKE_EXTRA_FLAGS} \
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
echo "  ${BIN_DIR}/llama-cli -m model.gguf -ngl 99"
echo ""
echo -e "${GREEN}Done.${NC}"
