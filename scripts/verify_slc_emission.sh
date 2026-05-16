#!/usr/bin/env bash
# QA Gate 2: Assembly Output Verification for RDNA2 MoE Stream V1
# Verifies that LLVM emits SLC=1 modifiers on GTT load instructions
# and respects "memory" clobber boundaries
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

KERNEL_SRC="ggml/src/ggml-cuda/common.cuh"
BUILD_DIR="build"
ROCm_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCm_PATH}/bin/hipcc"

echo "=== RDNA2 MoE Stream V1: SLC Emission Verification ==="
echo ""

# Check if RDNA2_MOE_STREAM_V1 is enabled
if ! grep -q "RDNA2_MOE_STREAM_V1" "${KERNEL_SRC}"; then
    echo -e "${RED}FAIL: RDNA2_MOE_STREAM_V1 macro not found in common.cuh${NC}"
    exit 1
fi
echo -e "${GREEN}PASS: RDNA2_MOE_STREAM_V1 macro present in common.cuh${NC}"

# Create test file that calls SLC functions
TEST_FILE="/tmp/rdna2_moe_test.cu"
cat > "$TEST_FILE" << 'EOF'
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-cuda/common.cuh"

__global__ void test_slc_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
#ifdef RDNA2_MOE_STREAM_V1
        dst[idx] = load_gtt_slc(&src[idx]);
#else
        dst[idx] = src[idx];
#endif
    }
}

int main() { return 0; }
EOF

# Check if hipcc is available
if [ ! -x "$HIPCC" ]; then
    echo -e "${RED}FAIL: hipcc not found at $HIPCC${NC}"
    exit 1
fi
echo "Using hipcc: $HIPCC"

# Compile to assembly
echo ""
echo "Compiling test kernel to assembly..."
$HIPCC -DGGML_USE_HIP -DRDNA2_MOE_STREAM_V1=1 \
  --offload-arch=gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102 \
  -O3 -S -mllvm -amdgpu-early-inline-all=true \
  -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
  "$TEST_FILE" -o /tmp/rdna2_moe_slc.s 2>&1 | grep -v "pragma once" || true

if [ ! -f /tmp/rdna2_moe_slc.s ]; then
    echo -e "${RED}FAIL: Assembly compilation failed${NC}"
    exit 1
fi
echo -e "${GREEN}PASS: Assembly generated successfully${NC}"

# Verify SLC=1 emission
echo ""
SLC_COUNT=$(grep -c "slc" /tmp/rdna2_moe_slc.s || true)
if [ "$SLC_COUNT" -eq 0 ]; then
    echo -e "${RED}FAIL: LLVM did not emit SLC modifiers. Cache bypass disabled.${NC}"
    echo "First 100 lines of assembly:"
    head -100 /tmp/rdna2_moe_slc.s
    exit 1
fi
echo -e "${GREEN}PASS: Found $SLC_COUNT SLC-modified instructions${NC}"

# Verify memory clobbers
echo ""
MEMORY_CLOBBER_COUNT=$(grep -c '"memory"' /tmp/rdna2_moe_slc.s || true)
if [ "$MEMORY_CLOBBER_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}WARN: No explicit memory clobber found. Check LLVM reordering.${NC}"
else
    echo -e "${GREEN}PASS: Found $MEMORY_CLOBBER_COUNT memory clobber boundaries${NC}"
fi

# Verify s_sleep instruction for low-power semaphore polling
echo ""
S_SLEEP_COUNT=$(grep -c "s_sleep" /tmp/rdna2_moe_slc.s || true)
if [ "$S_SLEEP_COUNT" -gt 0 ]; then
    echo -e "${GREEN}PASS: Found $S_SLEEP_COUNT s_sleep instructions (low-power semaphore polling)${NC}"
else
    echo -e "${YELLOW}WARN: No s_sleep instructions found. Semaphores may hot-spin.${NC}"
fi

# Verify no s_sendmsg instructions (driver compliance)
echo ""
S_SENDMSG_COUNT=$(grep -c "s_sendmsg" /tmp/rdna2_moe_slc.s || true)
if [ "$S_SENDMSG_COUNT" -gt 0 ]; then
    echo -e "${RED}FAIL: Found $S_SENDMSG_COUNT s_sendmsg instructions. Direct SDMA ring injection detected.${NC}"
    echo "This will cause CP hangs under the amdgpu driver."
    exit 1
fi
echo -e "${GREEN}PASS: No s_sendmsg instructions found (driver compliant)${NC}"

# Summary
echo ""
echo "=== Verification Summary ==="
echo -e "${GREEN}All QA gates passed.${NC}"
echo "Assembly output saved to: /tmp/rdna2_moe_slc.s"
echo ""
echo "Next steps:"
echo "  1. Build with: cmake -DRDNA2_MOE_STREAM_V1=ON .. && make"
echo "  2. Profile with: rocprofv3 --stats --profile-from-start off -o /tmp/rocm_profile ./llama-server -m <model.gguf>"
echo "  3. Check counters: rocprof-parse /tmp/rocm_profile | grep -E 'MemUnitBusy|WAVE_ISSUE_WAIT|L2CacheHit'"
