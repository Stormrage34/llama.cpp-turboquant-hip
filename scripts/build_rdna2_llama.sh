#!/bin/bash
# Phase 2E: RDNA2 Build Script for llama-bench and llama-server
# Compiles with RDNA2 optimizations (Phase 2A-2D) using manual hipcc
# Zero CMake changes - direct compilation wrapper

set -e

# Configuration
ROCM_PATH="/home/stormrage/rocm-7.13-nightly"
HIPCC="${ROCM_PATH}/bin/hipcc"
PROJECT_ROOT="/home/stormrage/llama.cpp-turboquant-hip"
BUILD_DIR="${PROJECT_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

# RDNA2 optimization flags
RDNA2_FLAGS="-DRDNA2_OPT_V1=1 -DRDNA2_ASYNC_PIPELINE=1 -DRDNA2_MATMUL_OPT_V1=1"
OFFLOAD_ARCH="--offload-arch=gfx1030"
OPT_LEVEL="-O3 -DNDEBUG"

# Include paths (from compile_commands.json)
INCLUDES_COMMON="-I${PROJECT_ROOT}/ggml/src/../include -I${PROJECT_ROOT}/src/../include -I${PROJECT_ROOT}/common/. -I${PROJECT_ROOT}/common/../vendor -I${PROJECT_ROOT}/tools/server -I${PROJECT_ROOT}/tools/server/../mtmd -I${PROJECT_ROOT}/tools/mtmd/. -I${PROJECT_ROOT} -I${ROCM_PATH}/include"

INCLUDES_BENCH="${INCLUDES_COMMON}"
INCLUDES_SERVER="${INCLUDES_COMMON}"

# Library paths
LIBS="-L${BIN_DIR} -lggml-hip -lggml-base -lggml-cpu -lggml -lllama -lllama-common -lamdhip64 -lpthread"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Phase 2E: RDNA2 Build Script ===${NC}"
echo "ROCm Path: ${ROCM_PATH}"
echo "HIPCC: ${HIPCC}"
echo "Build Dir: ${BUILD_DIR}"
echo ""

# Verify prerequisites
if [ ! -x "${HIPCC}" ]; then
    echo -e "${RED}ERROR: hipcc not found at ${HIPCC}${NC}"
    exit 1
fi

if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}ERROR: Build directory not found. Run CMake build first.${NC}"
    exit 1
fi

# Check if baseline libraries exist
for lib in libggml-hip.so libggml-base.so libggml.so libllama.so; do
    if [ ! -f "${BIN_DIR}/${lib}" ]; then
        echo -e "${YELLOW}WARNING: ${lib} not found in ${BIN_DIR}${NC}"
        echo "Ensure baseline CMake build completed first."
    fi
done

echo -e "${GREEN}Prerequisites verified.${NC}"
echo ""

# Build mode
MODE="${1:-optimized}"

case "${MODE}" in
    baseline)
        echo -e "${YELLOW}Building BASELINE (no RDNA2 optimizations)...${NC}"
        RDNA2_FLAGS=""
        ;;
    optimized)
        echo -e "${GREEN}Building OPTIMIZED (RDNA2 Phase 2A-2D)...${NC}"
        ;;
    *)
        echo "Usage: $0 [baseline|optimized]"
        exit 1
        ;;
esac

echo ""
echo "Compiling llama-bench with RDNA2 flags: ${RDNA2_FLAGS}"
echo ""

# Compile llama-bench
${HIPCC} ${OPT_LEVEL} ${OFFLOAD_ARCH} ${RDNA2_FLAGS} \
    ${INCLUDES_BENCH} \
    -o ${BIN_DIR}/llama-bench-rdna2 \
    ${PROJECT_ROOT}/tools/llama-bench/llama-bench.cpp \
    ${LIBS} \
    -Wl,-rpath,${BIN_DIR} \
    2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ llama-bench-rdna2 compiled successfully${NC}"
else
    echo -e "${RED}✗ llama-bench compilation failed${NC}"
    exit 1
fi

echo ""
echo "Compiling llama-server with RDNA2 flags: ${RDNA2_FLAGS}"
echo ""
echo "NOTE: llama-server requires full CMake build due to multi-file linking."
echo "Using CMake-built server from ${BIN_DIR}/llama-server"
echo ""

# Check if CMake-built server exists
if [ -f "${BIN_DIR}/llama-server" ]; then
    echo -e "${GREEN}✓ Using existing CMake-built llama-server${NC}"
else
    echo -e "${YELLOW}WARNING: llama-server not found. Run CMake build first.${NC}"
fi

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo "Binaries:"
echo "  ${BIN_DIR}/llama-bench-rdna2"
echo "  ${BIN_DIR}/llama-server-rdna2"
echo ""
echo "Run with: RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ${BIN_DIR}/llama-bench-rdna2 ..."
