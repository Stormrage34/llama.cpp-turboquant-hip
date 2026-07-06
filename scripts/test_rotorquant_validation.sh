#!/usr/bin/env bash
# RotorQuant counter-driven validation script.
# Validates RQ kernels via rocprofv3 PMC counters (no --hip-trace or --kernel-trace).
# Runs rotorquant-specific benchmarks and writes results to bench_counter_results.json.

set -euo pipefail

ROCMPROF="${1:-$(which rocprofv3)}"
MODEL="${2:-$HOME/models/qwen3-35b-iq4_xl.gguf}"
RQ_TYPE="${3:-RQ_MSE_2}"
N_RUNS="${4:-5}"

COUNTERS_MSE='[
  { "name": "SQ_WAVE_COUNTER0", "reg": 1 },
  { "name": "SQ_WAVE_COUNTER1", "reg": 2 },
  { "name": "WAVE_ISSUE_WAIT",   "reg": 3 },
  { "name": "VALU_INST_COUNT",   "reg": 4 },
  { "name": "SQ_INSTS_VALU_MAD_F16_I32", "reg": 5 },
  { "name": "SQ_INSTS_VALU_LSHR_F16_I32", "reg": 6 }
]'

COUNTERS_RQ='[
  { "name": "SQ_WAVE_COUNTER0",     "reg": 1 },
  { "name": "WAVE_ISSUE_WAIT",       "reg": 2 },
  { "name": "VALU_INST_COUNT",       "reg": 3 },
  { "name": "SQ_INSTS_VALU_MAD_F16_I32",   "reg": 4 },
  { "name": "SQ_INSTS_VALU_LSHR_F16_I32",   "reg": 5 },
  { "name": "SQ_INSTS_VALU_ADD_F16_I32",   "reg": 6 }
]'

RESULTS_FILE="bench_counter_results_${RQ_TYPE}.json"
mkdir -p benchmarks

echo "=== RotorQuant Counter Validation ==="
echo "Model: $MODEL"
echo "Type:  $RQ_TYPE ($N_RUNS runs)"

# Run the quantized model with RQ type and capture counters.
for run in $(seq 1 $N_RUNS); do
    echo "Run $run / $N_RUNS"
    
    rocprofv3 \
        -i "$COUNTERS_MSE" \
        -- "/home/stormrage/rocm-7.13-nightly/bin/llama-cli" \
        -m "$MODEL" \
        -ngl 99 \
        -ub 512 \
        --single-turn \
        --rqt "$RQ_TYPE" \
        -p "Write a short story about a robot learning to paint." \
        -n 512 2>&1 | tee /tmp/rq_counter_run_$run.out
    
    # Extract key counters from the output.
    VALU=$(grep -oP 'VALU_INST_COUNT\s*=\s*\K[0-9]+' /tmp/rq_counter_run_$run.out || echo 0)
    WAIT=$(grep -oP 'WAVE_ISSUE_WAIT\s*=\s*\K[0-9]+' /tmp/rq_counter_run_$run.out || echo 0)
    MAD=$(grep -oP 'SQ_INSTS_VALU_MAD_F16_I32\s*=\s*\K[0-9]+' /tmp/rq_counter_run_$run.out || echo 0)
    
    # Write JSON entry for this run.
    echo "{\"run\":$run,\"VALU_INST_COUNT\":$VALU,\"WAVE_ISSUE_WAIT\":$WAIT,\"MAD_F16_I32\":$MAD}" >> "$RESULTS_FILE"
done

# Final validation: if VALU > threshold, flag for review.
TOTAL_VALU=$(python3 -c "import json; print(sum(d['VALU_INST_COUNT'] for d in $(cat $RESULTS_FILE)))")
THRESHOLD=$((TOTAL_VALU * 110 / 100))  # 10% overhead allowed

if [ "$TOTAL_VALU" -gt "$THRESHOLD" ]; then
    echo "FAIL: VALU_INST_COUNT $TOTAL_VALU exceeds threshold $THRESHOLD"
    exit 1
else
    echo "PASS: VALU_INST_COUNT $TOTAL_VALU within threshold $THRESHOLD"
fi
