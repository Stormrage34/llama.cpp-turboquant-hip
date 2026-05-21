#!/bin/bash
set -euo pipefail
# =============================================================================
# LLAMA-SERVER CANONICAL LAUNCHER (v6 - Syntax-Verified & Production Hardened)
# RX 6800 XT (16GB VRAM) + Ryzen 5700X (8C/16T) + 48GB RAM
# CR-008 Verified: Symmetrical batch (-b == -ub) collapses variance to ±0.18 t/s
# MoE optimal: -b 512 -ub 512 (Qwen35: 357 t/s pp) [rocprof 2026-05-22]
# Gemma 4 tight: -b 64 -ub 64 (VRAM ceiling 15.7 GB)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${ROCM_PATH}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
source "${SCRIPT_DIR}/gpu_failback.sh"
MODEL_DIR="${MODEL_DIR:-/home/stormrage/models}"
BINARY_DIR="${PROJECT_ROOT}/build/bin"

# ─── CLI Argument Parsing ───────────────────────────────────────────────────
# Override interactive prompts for MoE offloading strategy
CLI_N_CPU_MOE=""
CLI_N_CPU_MOE_RANGE=""
CLI_NCMOE_VAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-cpu-moe)
            CLI_N_CPU_MOE="$2"
            shift 2
            ;;
        --n-cpu-moe-range)
            CLI_N_CPU_MOE_RANGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --n-cpu-moe N           Set --n-cpu-moe to N (fixed count of CPU experts)"
            echo "  --n-cpu-moe-range M-N   Set --n-cpu-moe-range (range of CPU experts, e.g. 10-20)"
            echo "  --help                  Show this message"
            echo ""
            echo "Without CLI args, the script runs interactively."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--n-cpu-moe N | --n-cpu-moe-range M-N]"
            exit 1
            ;;
    esac
done

# Colors
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Stale Binary Check ───────────────────────────────────────────────────
# If libraries were partially rebuilt, llama-server links old ABI → SIGSEGV
BINS=("llama-server" "llama-cli" "llama-bench")
LIBS=("libggml-hip.so.0" "libggml-cpu.so.0" "libggml-base.so.0" "libllama.so.0" "libllama-common.so.0")
for bin_name in "${BINS[@]}"; do
    bin_path="${BINARY_DIR}/${bin_name}"
    [[ -f "$bin_path" ]] || continue
    for lib_name in "${LIBS[@]}"; do
        lib_path="${BINARY_DIR}/${lib_name}"
        [[ -f "$lib_path" ]] || continue
        if [[ "$lib_path" -nt "$bin_path" ]]; then
            echo -e "${RED}⚠ STALE BINARY: ${bin_name} is older than ${lib_name}${NC}"
            echo -e "${YELLOW}  → Run './scripts/build_rdna2.sh' to rebuild ALL targets${NC}"
            echo -e "${YELLOW}  → Partial rebuilds cause SIGSEGV on startup${NC}"
            echo ""
            read -rp "Continue anyway? [y/N/rebuild]: " confirm
            if [[ "$confirm" =~ ^[rR] ]]; then
                echo -e "${GREEN}→ Rebuilding all targets...${NC}"
                "${SCRIPT_DIR}/build_rdna2.sh" --fast
                echo -e "${GREEN}→ Rebuild complete. Please re-run the launcher.${NC}"
                exit 0
            fi
            [[ "$confirm" =~ ^[yY] ]] || exit 1
            break
        fi
    done
done

# Cache types (2026-05 sweep validated)
CACHE_TYPES=("turbo3" "turbo4" "turbo2" "q8_0" "f16" "bf16" "f32" "q4_0" "q4_1" "iq4_nl" "q5_0" "q5_1")
DEFAULT_CTK="q8_0"
DEFAULT_CTV="turbo3"
ULTRA_CTK="turbo3"
ULTRA_CTV="turbo2"
ULTRA_CTX_THRESH=262144

# ============================================================================
# STEP 1: GPU failback acquire (handles server shutdown + VRAM release)
# ============================================================================
gpu_acquire
trap 'gpu_release' EXIT

# ============================================================================
# STEP 2: Build Environment Detection
# ============================================================================
echo -e "\n${BOLD}Select Build Engine:${NC} 1) TurboQuant-HIP [REC]  2) Main Branch"
read -rp "Choice [1-2]: " BUILD_CHOICE
if [ "${BUILD_CHOICE:-1}" = "2" ]; then
  BINARY_DIR="/home/stormrage/llama.cpp/build/bin"
  echo "-> Standard build detected."
else
  echo "-> TurboQuant-HIP detected. PolarQuant + RDNA2 matmul enabled."
  # RDNA2_MATMUL_OPT_V1 env var is deprecated — lds_bank_pad is now trait-gated
  # (see mmq_get_lds_bank_pad<type>() in mmq.cuh)
fi
[[ -x "$BINARY_DIR/llama-server" ]] || { echo -e "${RED}❌ llama-server missing in $BINARY_DIR${NC}"; exit 1; }

# ============================================================================
# STEP 3: Task Profile
# ============================================================================
echo -e "\n${BOLD}Task Profile:${NC} 1) Coding  2) Strict  3) Creative  4) General"
read -rp "Choice [1-4]: " TASK_CHOICE
case ${TASK_CHOICE:-4} in
  1) TEMP=0.6; TOP_P=0.95; TOP_K=20; MIN_P=0.05; DESC="Coding Research" ;;
  2) TEMP=0.0; TOP_P=0.90; TOP_K=10; MIN_P=0.01; DESC="Hard Technical" ;;
  3) TEMP=0.85; TOP_P=0.98; TOP_K=50; MIN_P=0.08; DESC="Creative Generation" ;;
  *) TEMP=0.7; TOP_P=0.95; TOP_K=40; MIN_P=0.05; DESC="General Workflow" ;;
esac

# ============================================================================
# STEP 4: Model Discovery & GGUF Parse
# ============================================================================
echo -e "\n📂 Scanning: $MODEL_DIR"
[[ -d "$MODEL_DIR" ]] || { echo -e "${RED}❌ Model dir missing.${NC}"; exit 1; }
mapfile -t MODEL_PATHS < <(find "$MODEL_DIR" -maxdepth 1 -type f -name "*.gguf" 2>/dev/null | sort)
[[ ${#MODEL_PATHS[@]} -gt 0 ]] || { echo -e "${RED}❌ No .gguf models found.${NC}"; exit 1; }

echo "📦 Models:"
for i in "${!MODEL_PATHS[@]}"; do
  FNAME=$(basename "${MODEL_PATHS[$i]}")
  SIZE=$(du -h "${MODEL_PATHS[$i]}" | cut -f1)
  echo "$i) $FNAME [$SIZE]"
done
read -rp "Index [0-$(( ${#MODEL_PATHS[@]} - 1 ))]: " MODEL_INDEX
MODEL_INDEX="${MODEL_INDEX//[[:space:]]/}"; MODEL_INDEX="${MODEL_INDEX:-0}"
[[ "$MODEL_INDEX" =~ ^[0-9]+$ && "$MODEL_INDEX" -lt "${#MODEL_PATHS[@]}" ]] || { echo "❌ Invalid index."; exit 1; }

SELECTED_MODEL="${MODEL_PATHS[$MODEL_INDEX]}"
MODEL_NAME=$(basename "$SELECTED_MODEL")
MODEL_BYTES=$(stat -c %s "$SELECTED_MODEL")
SIZE_GB=$(awk "BEGIN { v=$MODEL_BYTES/1073741824; if (v < 0.01) v=1; printf \"%.1f\", v }")

IS_MOE=0; IS_MTP=0; IS_GEMMA=0; IS_QWEN=0; IS_SSM=0; IS_SLIDING_WINDOW=0
BLOCK_COUNT=0; FULL_ATTENTION_INTERVAL=0; EXPERT_COUNT=0; EXPERT_USED=0; N_EMBD=0; N_KV_HEADS=0; HEAD_COUNT=0; KEY_LENGTH=0; ARCH=""; NEXTN_PREDICT=0

set +e
if command -v python3 &>/dev/null; then
  GGUF_META=$(python3 - "$SELECTED_MODEL" 2>/dev/null << 'PYEOF'
import struct, sys
def read_gguf(path):
  with open(path, 'rb') as f:
    if struct.unpack('<I', f.read(4))[0] != 0x46554747: return {}
    version = struct.unpack('<I', f.read(4))[0]
    tc = struct.unpack('<Q', f.read(8))[0]
    kc = struct.unpack('<Q', f.read(8))[0]
    meta = {}
    for _ in range(kc):
      kl = struct.unpack('<Q', f.read(8))[0]
      k = f.read(kl).decode('utf-8')
      vt = struct.unpack('<I', f.read(4))[0]
      if vt in (0,1,2,3,4,5,6,7,10,11,12):
        fmt,sz={0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}[vt]
        meta[k] = struct.unpack(f'<{fmt}', f.read(sz))[0]
      elif vt == 8:
        sl = struct.unpack('<Q', f.read(8))[0]
        meta[k] = f.read(sl).decode('utf-8')
      elif vt == 9:
        at = struct.unpack('<I', f.read(4))[0]
        al = struct.unpack('<Q', f.read(8))[0]
        esz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        for _ in range(al):
          if at == 8: f.read(struct.unpack('<Q', f.read(8))[0])
          else: f.read(esz.get(at, 4))
      else: continue
    return meta
m = read_gguf(sys.argv[1])
if m:
  arch = m.get('general.architecture', '')
  print(f'ARCH="{arch}"')
  for gk, vn in [('block_count','BLOCK_COUNT'),('full_attention_interval','FULL_ATTENTION_INTERVAL'),
                 ('expert_count','EXPERT_COUNT'),('expert_used_count','EXPERT_USED'),
                 ('nextn_predict_layers','NEXTN_PREDICT'),
                 ('embedding_length','N_EMBD')]:
    print(f'{vn.upper()}={int(m.get(f"{arch}.{gk}", 0))}')
  # Keys with nested subgroups (attention.*, rope.*, ssm.*)
  for sg, gk, vn in [('attention','head_count_kv','N_KV_HEADS'),('attention','head_count','HEAD_COUNT'),
                     ('attention','key_length','KEY_LENGTH')]:
    v = m.get(f"{arch}.{sg}.{gk}", m.get(f"{arch}.{gk}", 0))
    print(f'{vn.upper()}={int(v)}')
PYEOF
  )
  [[ -n "$GGUF_META" ]] && eval "$GGUF_META"
fi
set -e

LOWER_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
case "$ARCH" in
  qwen3*moe*|qwen3.6*moe*|qwen35moe*) IS_MOE=1; IS_QWEN=1; IS_SSM=1; echo -e "${GREEN}-> Qwen3.6-MoE + SSM${NC}" ;;
  gemma4moe*|gemma4*)      IS_MOE=1; IS_GEMMA=1; IS_SLIDING_WINDOW=1; [[ "$BLOCK_COUNT" -eq 0 ]] && BLOCK_COUNT=26; [[ "$EXPERT_COUNT" -eq 0 ]] && EXPERT_COUNT=30; echo -e "${GREEN}-> Gemma 4 26B A4B (sliding window)${NC}" ;;
  *) [[ "$LOWER_NAME" =~ (mixtral|dbrx|qwen.*moe|qwen.*a[0-9]+b|deepseek.*moe) ]] && IS_MOE=1; [[ "$LOWER_NAME" =~ mtp|multistep ]] && IS_MTP=1 ;;
esac
[[ "$EXPERT_COUNT" -gt 0 ]] && IS_MOE=1
[[ "$NEXTN_PREDICT" -gt 0 ]] && IS_MTP=1

if [[ "$IS_MTP" -eq 1 && "$NEXTN_PREDICT" -eq 0 ]]; then
  echo -e "${YELLOW}⚠ Warning: MTP enabled (via model name or explicit flag) but GGUF metadata '${ARCH}.nextn_predict_layers' is 0 or missing.${NC}"
  echo -e "${YELLOW}  → The server may crash if the model does not truly support MTP. Proceed with caution.${NC}"
  read -rp "Continue anyway? [y/N]: " confirm
  [[ "$confirm" =~ ^[yY] ]] || exit 1
fi

# ============================================================================
# STEP 5: VRAM Estimation Functions (NGL moved to after Step 7)
# ============================================================================
# Estimate VRAM consumed by KV cache, in GiB
# type_k/v: bpw = bits per element
kv_cache_gib() {
  local layers=$1 ctx=$2 k_bpw=$3 v_bpw=$4 head_dim=$5 n_kv_heads=$6
  local kv_dim=$(( head_dim * n_kv_heads ))
  awk "BEGIN { printf \"%.2f\", ($layers * $ctx * $kv_dim * ($k_bpw + $v_bpw)) / 8 / 1024 / 1024 / 1024 }"
}

# MoE offload range strategy (returns "start-end" range)
calc_ncmoe() {
  local bc=$1
  local range_start=$(( bc / 3 ))
  local range_end=$(( bc * 2 / 3 ))
  echo "${range_start}-${range_end}"
}

# ============================================================================
# STEP 6: Context & Launch Mode
# ============================================================================
echo -e "\n${BOLD}Launch Mode:${NC} 1) Quick  2) Long-Context  3) Custom"
read -rp "Choice [1-3]: " MODE_CHOICE
MODE="${MODE_CHOICE:-1}"
case "$MODE" in
  1) CTX=32768 ;;
  2) echo "Presets: 1)32k 2)64k 3)128k 4)256k 5)Custom"; read -rp "Choice [2]: " CC; case ${CC:-2} in 1)CTX=32768;;2)CTX=65536;;3)CTX=131072;;4)CTX=262144;;*) read -rp "Custom CTX: " CTX_INPUT; CTX=${CTX_INPUT:-65536}; esac ;;
  *) read -rp "CTX [32768]: " CTX_INPUT; CTX=${CTX_INPUT:-32768} ;;
esac
[[ "$CTX" =~ ^[0-9]+$ && "$CTX" -gt 0 ]] || { echo -e "${RED}❌ Invalid CTX. Must be positive integer.${NC}"; exit 1; }

if [ "$IS_MOE" -eq 1 ]; then
  NCMOE=0; MOE_RANGE=""

  # Auto-detect optimal offload range based on model size
  # Large models (Q5_K_XL ~25GB, SIZE_GB > 23) need more CPU experts than
  # standard quant (IQ4_XS ~21GB). Adjust range accordingly.
  if (( $(awk "BEGIN { print ($SIZE_GB > 23) }") )); then
    SUGGESTED_RANGE="$(( ${BLOCK_COUNT:-28} * 2 / 3 ))-${BLOCK_COUNT:-28}"
  else
    SUGGESTED_RANGE="$(calc_ncmoe "${BLOCK_COUNT:-48}")"
  fi

  if [ -n "$CLI_N_CPU_MOE" ]; then
    # CLI override: fixed --n-cpu-moe
    MOE_STRATEGY=1
    CLI_NCMOE_VAL="$CLI_N_CPU_MOE"
    echo -e "\n${BOLD}MoE Offload Strategy:${NC} CLI override --n-cpu-moe $CLI_NCMOE_VAL"
  elif [ -n "$CLI_N_CPU_MOE_RANGE" ]; then
    # CLI override: range-based --n-cpu-moe-range
    MOE_STRATEGY=2
    MOE_RANGE="$CLI_N_CPU_MOE_RANGE"
    echo -e "\n${BOLD}MoE Offload Strategy:${NC} CLI override --n-cpu-moe-range $MOE_RANGE"
  else
    echo -e "\n${BOLD}MoE Offload Strategy:${NC}"
    echo "  1) All experts CPU (recommended, most stable)"
    echo "  2) Range-based (offload middle layers, e.g. $SUGGESTED_RANGE)"
    read -rp "Choice [1-2]: " MOE_STRATEGY
    MOE_STRATEGY="${MOE_STRATEGY:-1}"
    if [ "$MOE_STRATEGY" = "2" ]; then
      MOE_RANGE="$SUGGESTED_RANGE"
      echo -e "  → Range: ${MOE_RANGE} CPU"
    else
      echo -e "  → All ${BLOCK_COUNT:-48} experts on CPU"
    fi
  fi
else
  NCMOE=0; MOE_RANGE=""
  MOE_STRATEGY=1
fi

# ============================================================================
# STEP 7: Cache, Batch & Memory
# ============================================================================
if [ "$MODE" = "3" ]; then
  echo "Cache Types: ${CACHE_TYPES[*]}"
  read -rp "K idx [3/q8_0]: " KI; CACHE_K="${CACHE_TYPES[${KI:-3}]:-$DEFAULT_CTK}"
  read -rp "V idx [0/turbo3]: " VI; CACHE_V="${CACHE_TYPES[${VI:-0}]:-$DEFAULT_CTV}"
elif [ "$CTX" -ge "$ULTRA_CTX_THRESH" ]; then
  CACHE_K="$ULTRA_CTK"; CACHE_V="$ULTRA_CTV"
else
  CACHE_K="$DEFAULT_CTK"; CACHE_V="$DEFAULT_CTV"
fi

# ============================================================================
# STEP 5b: VRAM Budget & NGL Calculation (after CTX + cache types known)
# ============================================================================
TOTAL_VRAM_GIB=16.0
VRAM_RESERVE_GIB=2.0

# KV cache bpw lookup
k_bpw() { case "$1" in q8_0) echo 8;; f16) echo 16;; f32) echo 32;; q4_0|turbo2|iq4_nl) echo 4;; *) echo 8;; esac; }
v_bpw() { case "$1" in turbo3|turbo4) echo 4;; q8_0) echo 8;; f16) echo 16;; f32) echo 32;; q4_0|turbo2) echo 4;; *) echo 4;; esac; }

EST_LAYERS=${BLOCK_COUNT:-48}; [[ "$EST_LAYERS" -eq 0 ]] && EST_LAYERS=48
KBW=$(k_bpw "$CACHE_K"); VBW=$(v_bpw "$CACHE_V")
# KV head dimension: use key_length from GGUF, else embedding_length/head_count, else 128
# N_KV_HEADS: if missing from GGUF (e.g. Gemma 4), defaults to HEAD_COUNT
N_KV_HEADS=${N_KV_HEADS:-0}
if [[ "$N_KV_HEADS" -lt 1 && "$HEAD_COUNT" -gt 0 ]]; then
  N_KV_HEADS=$HEAD_COUNT
elif [[ "$N_KV_HEADS" -lt 1 ]]; then
  N_KV_HEADS=8
fi
HEAD_DIM=${KEY_LENGTH:-0}
if [[ "$HEAD_DIM" -lt 1 && "$N_EMBD" -gt 0 && "$HEAD_COUNT" -gt 0 ]]; then
  HEAD_DIM=$(( N_EMBD / HEAD_COUNT ))
fi
[[ "$HEAD_DIM" -lt 1 ]] && HEAD_DIM=128
KV_GIB=$(kv_cache_gib "$EST_LAYERS" "$CTX" "$KBW" "$VBW" "$HEAD_DIM" "$N_KV_HEADS")
PER_LAYER_GIB=$(awk "BEGIN { printf \"%.4f\", $SIZE_GB / $EST_LAYERS }")
AVAIL_GIB=$(awk "BEGIN { printf \"%.1f\", $TOTAL_VRAM_GIB - $VRAM_RESERVE_GIB - $KV_GIB }")

if [ "$IS_MOE" -eq 1 ]; then
  # MoE: experts offloaded to CPU (strategy 1 all, strategy 2 middle range).
  # Non-expert weights (attention+router) are ~5% of model. NGL=99 loads all layers.
  NONGIB=$(awk "BEGIN { printf \"%.1f\", $SIZE_GB * 0.05 }")
  NONG_PER_LAYER=$(awk "BEGIN { printf \"%.4f\", $NONGIB / $EST_LAYERS }")
  NGL=99
  echo -e "\n🔹 MoE ($MODEL_NAME): ${SIZE_GB}G model, KV ~${KV_GIB}G, non-expert ~${NONGIB}G → NGL=$NGL (all layers GPU, experts CPU)"
else
  NGL=$(awk "BEGIN { v=int($AVAIL_GIB / $PER_LAYER_GIB); if(v<1)v=1; if(v>$EST_LAYERS)v=$EST_LAYERS; print v }")
  echo -e "\n🔹 Dense ($MODEL_NAME): ${SIZE_GB}G model, KV ~${KV_GIB}G, avail ~${AVAIL_GIB}G → NGL=$NGL"
fi

# Warn if model + KV exceeds VRAM
if [ "$IS_MOE" -eq 1 ]; then
  # MoE: non-expert weights only (experts CPU, ~5% of total)
  MODEL_ESTGIB=$(awk "BEGIN { printf \"%.1f\", $NGL * $NONG_PER_LAYER }")
  TOTAL_GIB=$(awk "BEGIN { printf \"%.1f\", $MODEL_ESTGIB + $KV_GIB + $VRAM_RESERVE_GIB }")
else
  MODEL_ESTGIB=$(awk "BEGIN { printf \"%.1f\", $NGL * $PER_LAYER_GIB }")
  TOTAL_GIB=$(awk "BEGIN { printf \"%.1f\", $MODEL_ESTGIB + $KV_GIB + $VRAM_RESERVE_GIB }")
fi
if (( $(awk "BEGIN { print ($TOTAL_GIB > 15.5) }") )); then
  echo -e "${YELLOW}⚠ Warning: Est. VRAM usage ${TOTAL_GIB}G > 15.5G redline. Reduce CTX or use --mmap.${NC}"
fi

# ─── CR-008 Symmetrical Batch Logic ──────────────────────────────
# Benchmarks proved symmetrical (-b == -ub) collapses variance
# from ±70 to ±0.18 t/s. Optimal values depend on VRAM budget:
#   MoE with room:  -b 512 -ub 512  (Qwen35: 357 t/s pp, +150%)
#   VRAM-tight:     -b 64  -ub 64   (Gemma 4: 471 t/s pp at limit)
#   Fallback:       -b 256 -ub 256  (safe for all models)
# ----------------------------------------------------------------
if [ "$CTX" -gt 100000 ]; then
    BATCH=1024; UBATCH=1024
elif [ "$IS_MOE" -eq 1 ]; then
    # MoE: experts offloaded to CPU → all models fit b=512
    # Gemma 4 26B OOMs at b=512 only WITHOUT expert offloading (-ngl 99)
    # With --n-cpu-moe-range, experts go to CPU → b=512 safe
    BATCH=512; UBATCH=512
else
    # Dense models — symmetrical, moderate batch
    BATCH=512; UBATCH=512
fi

FREE_RAM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
MMAP_MODE="--no-mmap --mlock"; (( FREE_RAM < 38000 )) && MMAP_MODE="--mmap"

# ============================================================================
# STEP 8: MTP Optimization (Modern Flags)
# ============================================================================
DRAFT_N=2; DRAFT_P_MIN=0.75; CACHE_REUSE=256
if [ "$IS_MTP" -eq 1 ]; then
  DRAFT_N=$(( CTX > 32768 ? 3 : 2 ))
  DRAFT_P_MIN=0.60; CACHE_REUSE=128; BATCH=128; UBATCH=128
  echo -e "\n${CYAN}→ MTP Optimized: draft=$DRAFT_N, p_min=$DRAFT_P_MIN, ubatch=$UBATCH${NC}"
  echo -e "${CYAN}  (also sets --spec-type mtp, --spec-draft-n-max, --spec-draft-p-min)${NC}"
fi

# ============================================================================
# STEP 9: Summary & Launch
# ============================================================================
clear
echo -e "${BOLD}${GREEN}================================================================"
echo "          LLAMA-SERVER CANONICAL CONFIGURATION (v6)"
echo -e "================================================================${NC}"
echo "Model: $MODEL_NAME (${SIZE_GB}G) | MoE=$IS_MOE | MTP=$IS_MTP"
if [ "$IS_MOE" -eq 1 ]; then
  if [ "$MOE_STRATEGY" = "2" ] && [ -n "$MOE_RANGE" ]; then
    MOE_SUMMARY="NCMOE-range=${MOE_RANGE}"
  elif [ -n "$CLI_NCMOE_VAL" ]; then
    MOE_SUMMARY="NCMOE=${CLI_NCMOE_VAL}"
  else
    MOE_SUMMARY="NCMOE=all CPU (${EST_LAYERS})"
  fi
else
  MOE_SUMMARY=""
fi
echo "CTX: $CTX | KV: $CACHE_K/$CACHE_V | NGL: $NGL | ${MOE_SUMMARY:+$MOE_SUMMARY}"
echo "Batch: $BATCH/$UBATCH | Cache-Reuse: $CACHE_REUSE | MMAP: $MMAP_MODE"
echo "Threads: 8 (0-7) | Batch CPU: 8-15 | Prio: High"
[[ "$IS_SLIDING_WINDOW" -eq 1 ]] && echo "⚠ Sliding Window: --no-context-shift enforced"
read -rp "Press Enter to launch..."

CMD=("$BINARY_DIR/llama-server" "-m" "$SELECTED_MODEL" "-ngl" "$NGL" "-c" "$CTX" "-b" "$BATCH" "-ub" "$UBATCH"
     "--cache-type-k" "$CACHE_K" "--cache-type-v" "$CACHE_V" "-fa" "1"
     "--temp" "$TEMP" "--top-p" "$TOP_P" "--top-k" "$TOP_K" "--min-p" "$MIN_P"
     "--threads" "8" "--threads-batch" "8" "--cpu-range" "0-7" "--cpu-strict" "1"
     "--cpu-range-batch" "8-15" "--cpu-strict-batch" "1" "--numa" "numactl" "--prio" "2"
     "--parallel" "1" "--jinja" "--metrics" "--cache-reuse" "$CACHE_REUSE"
     "--reasoning" "auto" "--no-context-shift")

if [ "$IS_MOE" -eq 1 ]; then
  if [ "$MOE_STRATEGY" = "2" ] && [ -n "$MOE_RANGE" ]; then
    CMD+=("--n-cpu-moe-range" "$MOE_RANGE")
  elif [ -n "$CLI_NCMOE_VAL" ]; then
    CMD+=("--n-cpu-moe" "$CLI_NCMOE_VAL")
  else
    CMD+=("--n-cpu-moe" "$EST_LAYERS")
  fi
fi
[[ "$MMAP_MODE" == "--mmap" ]] && CMD+=("--mmap") || CMD+=("--no-mmap" "--mlock")
if [ "$IS_MTP" -eq 1 ]; then
  CMD+=("--spec-type" "mtp" "--spec-draft-n-max" "$DRAFT_N" "--spec-draft-p-min" "$DRAFT_P_MIN")
fi

echo -e "\n🚀 Executing:"
echo "${CMD[*]}"
echo -e "${YELLOW}Stop: kill \$(pgrep -x llama-server)${NC}\n"
exec "${CMD[@]}"
