#!/usr/bin/env bash
# scripts/analyze_counters.sh
# Parses rocprofv3 SQLite/h5 output and produces human-readable tables
#
# Usage:
#   ./scripts/analyze_counters.sh <benchmarks/raw/DIR>
#   ./scripts/analyze_counters.sh benchmarks/raw/20260518_120000
#
# Output: Summary table of key metrics per kernel

set -e

OUTPUT_DIR="${1:-}"

if [[ -z "$OUTPUT_DIR" || ! -d "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <benchmarks/raw/DIR>"
    echo "Example: $0 benchmarks/raw/20260518_120000"
    exit 1
fi

if [[ ! -f "$OUTPUT_DIR/counters.h5" ]]; then
    echo "ERROR: counters.h5 not found in $OUTPUT_DIR"
    exit 1
fi

echo "=============================================="
echo "ROCProf Counter Analysis"
echo "=============================================="
echo "Directory: $OUTPUT_DIR"
echo ""

# Print run info if available
if [[ -f "$OUTPUT_DIR/run_info.txt" ]]; then
    echo "=== Run Information ==="
    cat "$OUTPUT_DIR/run_info.txt"
    echo ""
fi

# Try to extract stats using h5dump
if command -v h5dump &>/dev/null; then
    echo "=== Key Metrics ==="
    echo ""
    
    # Extract SQ_INSTS_VALU (ALU instructions)
    echo "--- ALU Instructions (SQ_INSTS_VALU) ---"
    h5dump -d /stats/SQ_INSTS_VALU "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # Extract MemUnitBusy
    echo "--- Memory Unit Busy (%) ---"
    h5dump -d /stats/MemUnitBusy "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # Extract MeanOccupancyPerCU
    echo "--- Mean Occupancy Per CU ---"
    h5dump -d /stats/MeanOccupancyPerCU "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # Extract WAVE_ISSUE_WAIT
    echo "--- Wave Issue Wait ---"
    h5dump -d /stats/WAVE_ISSUE_WAIT "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # Extract LDSBankConflict
    echo "--- LDS Bank Conflicts ---"
    h5dump -d /stats/LDSBankConflict "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # Extract VALUBusy
    echo "--- VALU Busy (%) ---"
    h5dump -d /stats/VALUBusy "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -E '^\s+[0-9]' | tail -5 || echo "  (not available or h5dump failed)"
    echo ""
    
    # List all available stats keys
    echo "=== Available Stat Keys ==="
    h5dump -H "$OUTPUT_DIR/counters.h5" 2>/dev/null | \
        grep -oP '/stats/\K[A-Z_]+(?=\s)' | sort -u || echo "  (h5dump not working)"
    echo ""

else
    echo "WARNING: h5dump not found. Cannot parse counters.h5."
    echo "Install h5tools or use Python with h5py to parse the stats file."
    echo ""
    echo "Available files:"
    ls -la "$OUTPUT_DIR/"
fi

# Try Python h5py as fallback
if command -v python3 &>/dev/null && python3 -c "import h5py" 2>/dev/null; then
    echo "=== Python h5py Summary ==="
    python3 -c "
import h5py
import sys

with h5py.File('$OUTPUT_DIR/counters.h5', 'r') as f:
    if 'stats' in f:
        stats = f['stats']
        print(f'  Total kernels traced: {len(stats)}')
        print()
        # Print first few kernel entries
        count = 0
        for key in sorted(stats.keys()):
            if count >= 5:
                break
            entry = stats[key]
            print(f'  Kernel: {key}')
            for subkey in sorted(entry.keys()):
                val = entry[subkey][()]
                if isinstance(val, (int, float)):
                    print(f'    {subkey}: {val:.4f}')
                else:
                    print(f'    {subkey}: {val}')
            count += 1
    else:
        print('  No /stats group found in h5 file.')
        print('  Available groups:', list(f.keys()))
" 2>/dev/null || echo "  (Python parsing failed)"
    echo ""
fi

echo "=============================================="
echo "Analysis complete. Raw data in $OUTPUT_DIR/"
echo "=============================================="
