#!/bin/bash
# Build baseline v0.3.0-stable for cross-fork comparison
# Usage: ./scripts/build_baseline.sh
#
# Builds the v0.3.0-stable tag with -DBUILD_TESTING=OFF to bypass
# the broken test_dequant_rdn2.cpp (references ggml_dequant_iq4_xs_rdn2
# which didn't exist in that tag).
#
# Output: llama-bench-v0.3.0 binary in project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-baseline"

echo "================================================"
echo "  Building v0.3.0-stable baseline"
echo "  Project: ${PROJECT_ROOT}"
echo "  Build:   ${BUILD_DIR}"
echo "================================================"

# Stash any uncommitted changes
if ! git -C "${PROJECT_ROOT}" diff --quiet; then
    echo "Stashing uncommitted changes..."
    git -C "${PROJECT_ROOT}" stash
    STASHED=true
else
    STASHED=false
fi

# Checkout baseline tag
echo "Checking out v0.3.0-stable..."
git -C "${PROJECT_ROOT}" checkout v0.3.0-stable

# Configure with tests disabled (v0.3.0-stable has broken test_dequant_rdn2)
echo "Configuring..."
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1030 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"

# Build (only llama-bench needed)
echo "Building..."
cmake --build "${BUILD_DIR}" --config Release --target llama-bench -- -j 16

# Copy binary
cp "${BUILD_DIR}/bin/llama-bench" "${PROJECT_ROOT}/llama-bench-v0.3.0"
echo "Baseline binary: ${PROJECT_ROOT}/llama-bench-v0.3.0"

# Restore current branch
echo "Restoring current branch..."
git -C "${PROJECT_ROOT}" checkout main
if [ "${STASHED}" = true ]; then
    echo "Restoring stashed changes..."
    git -C "${PROJECT_ROOT}" stash pop
fi

echo ""
echo "=== Done ==="
echo "Baseline binary: ${PROJECT_ROOT}/llama-bench-v0.3.0"
echo ""
echo "To run comparison:"
echo "  source scripts/gpu_failback.sh"
echo "  gpu_acquire"
echo "  # Baseline:"
echo "  ./llama-bench-v0.3.0 -m /home/stormrage/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\"
echo "      -ngl 99 -ncmoe 41 -fa 1 -b 128 -ub 512 -p 512 -n 128 -r 3"
echo "  # Current:"
echo "  build/bin/llama-bench -m /home/stormrage/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\"
echo "      -ngl 99 -ncmoe 41 -fa 1 -b 128 -ub 512 -p 512 -n 128 -r 3"
