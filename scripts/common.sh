# Shared measurement settings for the gfx1030 campaign scripts.
# Source this file after ROOT is defined (or let it define ROOT).
: "${ROOT:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

MODEL="${MODEL:-/home/stormrage/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
FA="${FA:-1}"
CTK="${CTK:-q8_0}"
CTV="${CTV:-q5_0}"
BENCH_EXCLUSIVE="$ROOT/scripts/bench-exclusive.sh"
