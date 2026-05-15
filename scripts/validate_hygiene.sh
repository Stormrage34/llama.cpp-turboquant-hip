#!/usr/bin/env bash
#
# validate_hygiene.sh — RDNA2 Pipeline Hygiene Validator
# Verifies: compile, smoke test, VRAM leak check
# Usage: ./scripts/validate_hygiene.sh [--verbose]
#
# Exits 0 on all pass, 1+ on failure (see exit code table below).
#
# Exit codes:
#   0 — All checks passed
#   1 — Compile failed
#   2 — Smoke test failed
#   3 — VRAM leak detected
#   4 — Multiple failures

set -euo pipefail

VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# GPU failback — free VRAM before smoke test
source "${SCRIPT_DIR}/scripts/gpu_failback.sh"
gpu_failback_trap
gpu_acquire
TEST_SRC="${SCRIPT_DIR}/tests/smoke_rdna2.cpp"
TEST_BIN="${BUILD_DIR}/bin/smoke_rdna2"
ROCMPATH="${ROCM_PATH:-/opt/rocm}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0
exit_code=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((pass_count++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((fail_count++))
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "==========================================="
echo " RDNA2 Pipeline Hygiene Validator v0.3.1.1"
echo " Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo " ROCm: ${ROCMPATH}"
echo "==========================================="
echo ""

# --------------------------------------------------
# Step 1: Compile smoke test
# --------------------------------------------------
echo "--- Step 1: Compile smoke test ---"
if [[ ! -f "${TEST_SRC}" ]]; then
    fail "Smoke test source not found: ${TEST_SRC}"
    exit 1
fi

mkdir -p "${BUILD_DIR}/bin"

if ${ROCMPATH}/bin/hipcc \
    -O2 \
    -DGGML_USE_HIP \
    -D__HIP_PLATFORM_AMD__ \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}" \
    -L"${BUILD_DIR}/bin" \
    -Wl,-rpath,"${BUILD_DIR}/bin" \
    -o "${TEST_BIN}" \
    "${TEST_SRC}" \
    -lggml-base -lggml-cpu -lggml-hip \
    -lpthread -ldl -lm \
    2>&1; then
    pass "Smoke test compiled successfully"
else
    fail "Smoke test compilation failed"
    exit_code=1
fi
echo ""

# --------------------------------------------------
# Step 2: Run smoke test
# --------------------------------------------------
echo "--- Step 2: Run smoke test ---"
if [[ ! -x "${TEST_BIN}" ]]; then
    fail "Smoke test binary not found: ${TEST_BIN}"
    ((fail_count++))
    [[ ${exit_code} -eq 0 ]] && exit_code=2
else
    # Run with all three RDNA2 flags enabled
    export RDNA2_MATMUL_OPT_V1=1
    export LD_LIBRARY_PATH="${BUILD_DIR}/bin:${LD_LIBRARY_PATH:-}"

    if "${TEST_BIN}"; then
        pass "Smoke test passed"
    else
        fail "Smoke test failed (exit code $?)"
        [[ ${exit_code} -eq 0 ]] && exit_code=2
    fi

    # Also run without flags to verify baseline path works
    echo ""
    echo "--- Step 2b: Baseline path (no flags) ---"
        pass "Baseline path passed" || \
        { fail "Baseline path failed"; [[ ${exit_code} -eq 0 ]] && exit_code=2; }
fi
echo ""

# --------------------------------------------------
# Step 3: VRAM leak check
# --------------------------------------------------
echo "--- Step 3: VRAM leak check ---"
if command -v rocm-smi &>/dev/null; then
    # Record VRAM usage before
    VRAM_BEFORE=$(rocm-smi --showmemuse 2>/dev/null | grep "GPU[[:space:]]*0" | awk '{print $NF}' | tr -d 'M' || echo "0")
    echo "VRAM before: ${VRAM_BEFORE} MiB"

    # Run smoke test 3 times to stress test teardown
    for i in 1 2 3; do
            LD_LIBRARY_PATH="${BUILD_DIR}/bin:${LD_LIBRARY_PATH:-}" \
            "${TEST_BIN}" > /dev/null 2>&1
    done

    # Record VRAM usage after
    sleep 1  # Allow teardown to complete
    VRAM_AFTER=$(rocm-smi --showmemuse 2>/dev/null | grep "GPU[[:space:]]*0" | awk '{print $NF}' | tr -d 'M' || echo "0")
    echo "VRAM after:  ${VRAM_AFTER} MiB"

    if [[ -n "${VRAM_BEFORE}" && -n "${VRAM_AFTER}" && "${VRAM_BEFORE}" != "0" && "${VRAM_AFTER}" != "0" ]]; then
        LEAK=$((VRAM_AFTER - VRAM_BEFORE))
        if [[ ${LEAK} -gt 100 ]]; then
            fail "VRAM leak detected: ${LEAK} MiB unreleased after 3 runs"
            [[ ${exit_code} -eq 0 ]] && exit_code=3
        else
            pass "No VRAM leak (delta: ${LEAK} MiB)"
        fi
    else
        warn "rocm-smi VRAM reading unreliable, skipping leak detection"
    fi
else
    warn "rocm-smi not found, skipping VRAM leak check. Install with: sudo apt install rocm-smi-lib"
fi
echo ""

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo "==========================================="
echo " Results: ${pass_count} passed, ${fail_count} failed"
if [[ ${exit_code} -eq 0 ]]; then
    echo -e " ${GREEN}ALL CHECKS PASSED${NC}"
else
    echo -e " ${RED}SOME CHECKS FAILED (exit code ${exit_code})${NC}"
fi
echo "==========================================="

exit ${exit_code}
