#!/bin/bash
# bench_turbo3.sh - Benchmark turbo3 KV cache via llama-bench
# Usage: ./bench_turbo3.sh /path/to/model.gguf [extra llama-bench args]
#
# Runs each (config, prompt_size) pair as a SEPARATE llama-bench invocation
# to avoid VRAM overflow from allocating max context upfront.

set -euo pipefail

MODEL="${1:?Usage: $0 /path/to/model.gguf [extra args...]}"
shift
EXTRA_ARGS=("$@")

OUTDIR="/tmp/bench_turbo3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

BENCH="./build/bin/llama-bench"

# Prompt sizes that fit in VRAM for all configs
# Q8_0 at 32K = 12.5 GB KV, turbo3 at 81K = 12.5 GB KV
# With model overhead (~3 GB), safe ceilings:
#   Q8_0/Q8_0:   max ~30K
#   Q8_0/turbo3: max ~24K (Q8_0 keys dominate)
#   turbo3/turbo3: max ~65K
# We test sizes that fit ALL configs, then extended for turbo3-only
PROMPTS="4096,8192,16384,24576,32768"
# Extended sizes for turbo3-only (Q8_0 won't fit)
PROMPTS_T3_ONLY="49152,65536,81920"

# Configs: label, -ctk, -ctv, extra prompt sizes
declare -a CONFIGS=(
    "q8q8|q8_0|q8_0|"
    "q8t3|q8_0|turbo3_0|"
    "t3t3|turbo3_0|turbo3_0|49152,65536,81920"
)

echo "================================================================"
echo "TURBO3 KV CACHE BENCHMARK (llama-bench)"
echo "================================================================"
echo "Model: $MODEL"
echo "Batch: -b 4096 -ub 4096 -fa on -ncmoe 15"
echo "Output: $OUTDIR"
echo ""

MASTER_CSV="$OUTDIR/all_results.csv"
echo "config,n_prompt,pp_tps,tg_tps" > "$MASTER_CSV"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r label ctk ctv extra_sizes <<< "$cfg"
    echo "--- Config: $label (-ctk $ctk -ctv $ctv) ---"

    ALL_SIZES="$PROMPTS"
    if [ -n "$extra_sizes" ]; then
        ALL_SIZES="${ALL_SIZES},${extra_sizes}"
    fi

    # Split comma-separated sizes into array
    IFS=',' read -ra SIZES <<< "$ALL_SIZES"

    for psize in "${SIZES[@]}"; do
        echo -n "  pp n=$psize ... "

        RUNOUT="$OUTDIR/${label}_${psize}.csv"
        RUNERR="$OUTDIR/${label}_${psize}.err"

        # Run ONE prompt size at a time
        if "$BENCH" \
            -m "$MODEL" \
            -ctk "$ctk" \
            -ctv "$ctv" \
            -fa on \
            -b 4096 \
            -ub 4096 \
            -ncmoe 15 \
            -p "$psize" \
            -n 0 \
            -r 1 \
            -o csv \
            "${EXTRA_ARGS[@]}" \
            > "$RUNOUT" 2>"$RUNERR"; then

            # Extract pp line
            pp_line=$(grep "^model,.*pp," "$RUNOUT" 2>/dev/null | head -1 || true)
            if [ -n "$pp_line" ]; then
                # CSV format: model,backend,threads,n_batch,n_ubatch,n_prompt,n_gen,t_pp,tg,pp,tg_per_token
                # Actually format varies. Just grab the pp t/s value
                tps=$(echo "$pp_line" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="pp") print $(i-2)}' || true)
                # Fallback: just print raw
                echo "$pp_line"
                echo "$label,$psize,$tps" >> "$MASTER_CSV"
            else
                echo "no pp data"
                # Show last lines of error for debugging
                tail -3 "$RUNERR" 2>/dev/null | sed 's/^/    /'
            fi
        else
            echo "FAILED (exit $?)"
            tail -5 "$RUNERR" 2>/dev/null | sed 's/^/    /'
        fi
    done
    echo ""
done

# Summary
echo "================================================================"
echo "RAW RESULTS"
echo "================================================================"
cat "$MASTER_CSV"
echo ""

# Parse with python for comparison table
python3 - "$MASTER_CSV" << 'PYEOF'
import csv, sys
from collections import defaultdict

csv_path = sys.argv[1]
results = defaultdict(dict)
with open(csv_path) as f:
    for row in csv.DictReader(f):
        try:
            n = int(row["n_prompt"])
            label = row["config"]
            # pp_tps might be empty
            tps_str = row.get("pp_tps", "").strip()
            if tps_str:
                tps = float(tps_str)
                results[label][n] = tps
        except:
            pass

if not results:
    print("No valid results to compare.")
    sys.exit(0)

all_n = sorted(set(n for r in results.values() for n in r.keys()))
labels = sorted(results.keys())

print(f"{'n_prompt':>10}", end="")
for l in labels:
    print(f" | {l:>12}", end="")
print()
print("-" * (10 + 15 * len(labels)))

for n in all_n:
    print(f"{n:>10}", end="")
    for l in labels:
        tps = results[l].get(n)
        if tps:
            print(f" | {tps:>9.1f} t/s", end="")
        else:
            print(f" | {'N/A':>12}", end="")
    print()

if len(labels) > 1:
    base = labels[0]
    print()
    print("Ratio vs %s:" % base)
    print(f"{'n_prompt':>10}", end="")
    for l in labels[1:]:
        print(f" | {l+'/'+base:>15}", end="")
    print()
    print("-" * (10 + 18 * (len(labels)-1)))
    for n in all_n:
        b = results[base].get(n, 0)
        if b <= 0:
            continue
        print(f"{n:>10}", end="")
        for l in labels[1:]:
            tps = results[l].get(n, 0)
            if tps > 0:
                print(f" | {tps/b:>12.2f}x", end="")
            else:
                print(f" | {'N/A':>15}", end="")
        print()
PYEOF

echo ""
echo "Logs in: $OUTDIR"
