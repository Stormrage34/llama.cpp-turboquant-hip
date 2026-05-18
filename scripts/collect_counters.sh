#!/usr/bin/env bash
# scripts/collect_counters.sh
# Runs rocprofv3 on llama-cli/llama-bench and saves SQLite to benchmarks/raw/
#
# Usage:
#   ./scripts/collect_counters.sh [binary] [prompt] [counter_file] [ngl] [ncmoe]
#   ./scripts/collect_counters.sh build/bin/llama-bench "test" scripts/counters_p0_template.txt 60 8
#   ./scripts/collect_counters.sh build/bin/llama-cli "hello world" "" 60 8
#
# Counter files in scripts/:
#   counters_p0_template.txt  — P0 baseline (TXT format for rocprofv3)
#   counters_p0.json          — Legacy JSON (NOT recommended for ROCm 7.13)
#
# Output: benchmarks/raw/YYYYMMDD_HHMMSS/
#   - counters.sqlite         (rocprofv3 SQLite database)
#   - kernels.csv             (kernel dispatch data)
#   - run_info.txt            (command, ROCm version, GPU model, VRAM usage)
#
# ROCm 7.13 NOTES:
#   - Uses rocprofv3 (NOT legacy rocprof)
#   - gfx1030 has NO pre-defined counter sets — use TXT format with raw PMC names
#   - Counter names must match hardware PMC blocks (SQ, TA, TCC, TCP, LDS, etc.)

set -e

ROCMPATH="${ROCM_PATH:-/home/stormrage/rocm-7.13-nightly}"
BINARY="${1:-build/bin/llama-cli}"
PROMPT="${2:-test prompt for inference}"
COUNTER_FILE="${3:-scripts/counters_p0_template.txt}"
NGL="${4:-60}"      # Default: partial offload (avoid OOM)
NCMOE="${5:-8}"     # Default: reduced expert offload

OUTPUT_DIR="benchmarks/raw/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "ROCProf v3 Counter Collection (ROCm 7.13)"
echo "=============================================="
echo "Binary:       $BINARY"
echo "Prompt:       $PROMPT"
echo "Counter File: $COUNTER_FILE"
echo "Offload:      -ngl $NGL -ncmoe $NCMOE"
echo "Output:       $OUTPUT_DIR"
echo "ROCm Path:    $ROCMPATH"
echo "GPU:          gfx1030 (RX 6800 XT)"
echo "=============================================="

# Verify binary exists
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    exit 1
fi

# Verify rocprofv3 exists (ROCm 7.13 uses rocprofv3, NOT legacy rocprof)
if [[ ! -x "$ROCMPATH/bin/rocprofv3" ]]; then
    echo "ERROR: rocprofv3 not found at $ROCMPATH/bin/rocprofv3"
    echo "ROCm 7.13+ requires rocprofv3 (legacy rocprof removed)"
    exit 1
fi

# Verify counter file exists (JSON format required for ROCm 7.13)
if [[ ! -f "$COUNTER_FILE" ]]; then
    echo "ERROR: Counter file not found: $COUNTER_FILE"
    exit 1
fi

# Check if file is JSON (ROCm 7.13 requirement)
if ! grep -q '"jobs"' "$COUNTER_FILE" 2>/dev/null; then
    echo "WARNING: Counter file doesn't appear to be JSON format"
    echo "ROCm 7.13 rocprofv3 requires JSON format (TXT is deprecated)"
    echo "Converting TXT to JSON..."
    # Create temporary JSON from TXT
    JSON_TEMP=$(mktemp)
    echo '{"jobs":[{"pmc":[' > "$JSON_TEMP"
    grep -v '^#' "$COUNTER_FILE" | grep -v '^$' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//' >> "$JSON_TEMP"
    echo ']}]}' >> "$JSON_TEMP"
    COUNTER_FILE="$JSON_TEMP"
    trap "rm -f $JSON_TEMP" EXIT
fi

# Collect system info for the record
{
    echo "=== Run Information ==="
    echo "Date: $(date -Iseconds)"
    echo "ROCm Path: $ROCMPATH"
    echo "ROCm Version: $($ROCMPATH/rocm-smi --version 2>/dev/null || echo 'unknown')"
    echo "GPU Model: $($ROCMPATH/rocm-smi --showname 2>/dev/null | head -1 || echo 'unknown')"
    echo "Binary: $BINARY"
    echo "Prompt: $PROMPT"
    echo "Counter File: $COUNTER_FILE"
    echo "Offload Config: -ngl $NGL -ncmoe $NCMOE"
    echo "VRAM Usage (before):"
    $ROCMPATH/rocm-smi --showmemused 2>/dev/null || echo "  (rocm-smi not available)"
    echo "=============================================="
} > "$OUTPUT_DIR/run_info.txt"

echo ""
echo "Running rocprofv3 with kernel tracing..."
echo ""

# ROCm 7.13 rocprofv3 syntax:
#   --kernel-trace true    : Enable kernel dispatch tracing
#   -i <json>              : Input counter file (JSON format required)
#   -d <dir>               : Output directory (creates SQLite DB)
#   --                     : Separator before application command
#
# NOTE: ROCm 7.13 rocprofv3 requires JSON format for counter definitions.
# TXT format is deprecated and will be removed in future releases.
# gfx1030 (RDNA2) has NO pre-defined counter sets — use custom JSON with raw PMC names.
#
# Counter names in counters_p0.json are raw PMC counters that may not all be
# supported on gfx1030. rocprofv3 will warn about unsupported counters but continue.

$ROCMPATH/bin/rocprofv3 \
    --kernel-trace true \
    -i "$COUNTER_FILE" \
    -d "$OUTPUT_DIR" \
    -- \
    "$BINARY" -m model.gguf -ngl "$NGL" -ncmoe "$NCMOE" -c 256 -p "$PROMPT" -n 8 \
    2>&1 | tee "$OUTPUT_DIR/rocprofv3.log"

echo ""
echo "=============================================="
echo "Results saved to $OUTPUT_DIR/"
echo "  - counters.sqlite (or .rocpd) - SQLite database"
echo "  - kernels.csv                 - Kernel dispatch data"
echo "  - rocprofv3.log               - stdout/stderr"
echo "  - run_info.txt                - System info"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Check for errors: grep -i error $OUTPUT_DIR/rocprofv3.log"
echo "  2. Parse counters: python3 scripts/parse_telemetry.py $OUTPUT_DIR/counters.sqlite"
echo "  3. Analyze: ./scripts/analyze_counters.sh $OUTPUT_DIR"
