#!/bin/bash
set -euo pipefail
# =============================================================================
# LLAMA-SERVER CANONICAL LAUNCHER (v7 - MoE VRAM Math Calibrated)
# RX 6800 XT (16GB VRAM) + Ryzen 5700X (8C/16T) + 48GB RAM
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROCM_PATH="${ROCM_PATH:-/home/stormrage/rocm-7.13-nightly}"
export HIP_PATH="${ROCM_PATH}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
MODEL_DIR="${MODEL_DIR:-/home/stormrage/models}"
BINARY_DIR="${PROJECT_ROOT}/build/bin"

# Colors
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'

# Cache types
CACHE_TYPES=("turbo3" "turbo4" "turbo2" "q8_0" "f16" "bf16" "f32" "q4_0" "q4_1" "iq4_nl" "q5_0" "q5_1")
DEFAULT_CTK="q8_0"; DEFAULT_CTV="turbo3"
ULTRA_CTK="turbo3"; ULTRA_CTV="turbo2"; ULTRA_CTX_THRESH=262144

# ============================================================================
# STEP 1: Server Check (Self-contained)
# ============================================================================
SERVER_PID=$(pgrep -x llama-server 2>/dev/null | head -n1 || true)
if [[ -n "$SERVER_PID" ]]; then
  echo -e "\n${YELLOW}⚠ llama-server running (PID: $SERVER_PID)${NC}"
  read -rp "  Kill? [Y/n]: " KILL
  if [[ "${KILL:-Y}" =~ ^[Yy]$ ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    for i in {1..5}; do sleep 1; pgrep -x llama-server >/dev/null 2>&1 || break; done
    pgrep -x llama-server >/dev/null 2>&1 && { kill -9 "$SERVER_PID" 2>/dev/null; sleep 1; }
    echo "✅ Stopped."
  else echo "Aborted."; exit 0; fi
fi

# ============================================================================
# STEP 2: Build Detection
# ============================================================================
echo -e "\n${BOLD}Build Engine:${NC} 1) TurboQuant-HIP [REC]  2) Main"
read -rp "Choice [1-2]: " BC
[[ "${BC:-1}" = "2" ]] && BINARY_DIR="/home/stormrage/llama.cpp/build/bin" || true  # TurboQuant-HIP build selected (lds_bank_pad trait-gated, no env var needed)
[[ -x "$BINARY_DIR/llama-server" ]] || { echo -e "${RED}❌ Missing: $BINARY_DIR/llama-server${NC}"; exit 1; }

# ============================================================================
# STEP 3: Task Profile
# ============================================================================
echo -e "\n${BOLD}Task:${NC} 1)Coding 2)Strict 3)Creative 4)General"
read -rp "Choice [1-4]: " TC
case ${TC:-4} in
  1) TEMP=0.6; TOP_P=0.95; TOP_K=20; MIN_P=0.05 ;;
  2) TEMP=0.0; TOP_P=0.90; TOP_K=10; MIN_P=0.01 ;;
  3) TEMP=0.85; TOP_P=0.98; TOP_K=50; MIN_P=0.08 ;;
  *) TEMP=0.7; TOP_P=0.95; TOP_K=40; MIN_P=0.05 ;;
esac

# ============================================================================
# STEP 4: Model Discovery
# ============================================================================
echo -e "\n📂 Scanning: $MODEL_DIR"
[[ -d "$MODEL_DIR" ]] || { echo -e "${RED}❌ Dir missing${NC}"; exit 1; }
mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f -name "*.gguf" 2>/dev/null | sort)
[[ ${#MODELS[@]} -gt 0 ]] || { echo -e "${RED}❌ No .gguf${NC}"; exit 1; }
echo "📦 Models:"; for i in "${!MODELS[@]}"; do echo "$i) $(basename "${MODELS[$i]}") [$(du -h "${MODELS[$i]}"|cut -f1)]"; done
read -rp "Index [0-$(( ${#MODELS[@]}-1 ))]: " MI
MI="${MI//[[:space:]]/}"; MI="${MI:-0}"
[[ "$MI" =~ ^[0-9]+$ && "$MI" -lt "${#MODELS[@]}" ]] || { echo "❌ Invalid"; exit 1; }

MODEL="${MODELS[$MI]}"; MNAME=$(basename "$MODEL")
MBYTES=$(stat -c %s "$MODEL"); SIZE_GB=$(awk "BEGIN { v=$MBYTES / 1073741824; if (v < 1) v = 1; printf \"%.2f\", v }")

# GGUF Parse
IS_MOE=0; IS_MTP=0; IS_GEMMA=0; IS_QWEN=0; IS_SSM=0; IS_SW=0
BLOCK_COUNT=0; EXPERT_COUNT=0; EXPERT_USED=0; ARCH=""
set +e
if command -v python3 &>/dev/null; then
  META=$(python3 - "$MODEL" 2>/dev/null << 'PY'
import struct,sys
def parse(p):
  with open(p,'rb') as f:
    if struct.unpack('<I',f.read(4))[0]!=0x46554747: return {}
    struct.unpack('<II',f.read(8)); tc,kc=struct.unpack('<QQ',f.read(16)); m={}
    for _ in range(kc):
      kl=struct.unpack('<Q',f.read(8))[0]; k=f.read(kl).decode()
      vt=struct.unpack('<I',f.read(4))[0]
      if vt in(0,1,2,3,4,5,6,7,10,11,12):
        fmt,sz={0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}[vt]
        m[k]=struct.unpack(f'<{fmt}',f.read(sz))[0]
      elif vt==8: sl=struct.unpack('<Q',f.read(8))[0]; m[k]=f.read(sl).decode()
      elif vt==9:
        at=struct.unpack('<I',f.read(4))[0]; al=struct.unpack('<Q',f.read(8))[0]
        esz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        for _ in range(al): f.read(struct.unpack('<Q',f.read(8))[0] if at==8 else esz.get(at,4))
    return m
d=parse(sys.argv[1])
if d:
  a=d.get('general.architecture',''); print(f'ARCH="{a}"')
  for gk,vn in[('block_count','BLOCK_COUNT'),('expert_count','EXPERT_COUNT'),('expert_used_count','EXPERT_USED')]:
    print(f'{vn}={int(d.get(f"{a}.{gk}",0))}')
PY
  ); [[ -n "$META" ]] && eval "$META"
fi
set -e

LN=$(echo "$MNAME"|tr '[:upper:]' '[:lower:]')
case "$ARCH" in
  qwen35moe*|qwen3.6*moe*) IS_MOE=1; IS_QWEN=1; IS_SSM=1; echo -e "${GREEN}→ Qwen3.6-MoE+SSM${NC}" ;;
  gemma4*) IS_MOE=1; IS_GEMMA=1; IS_SW=1; BLOCK_COUNT=26; EXPERT_COUNT=30; echo -e "${GREEN}→ Gemma4 MoE (sliding window)${NC}" ;;
  *) [[ "$LN" =~ (mixtral|dbrx|qwen.*moe) ]] && IS_MOE=1; [[ "$LN" =~ mtp ]] && IS_MTP=1 ;;
esac

# ============================================================================
# STEP 5: MoE-Aware VRAM Math (FIXED)
# ============================================================================
# VRAM budget: 12.5 GB for 16 GB card (leaves 3.5 GB for KV+overhead+display)
VRAM_BUDGET_GB=12.5
ROCM_OVERHEAD_GB=1.5
KV_CACHE_PER_TOKEN_KB=20  # Dense attention; SSM reduces this ~70%

# Estimate KV cache size in GB
calc_kv_gb() {
  local ctx=$1 is_ssm=$2
  local kb_per_token=$KV_CACHE_PER_TOKEN_KB
  [[ "$is_ssm" -eq 1 ]] && kb_per_token=$(( kb_per_token * 3 / 10 ))  # SSM ~70% reduction
  awk "BEGIN { printf \"%.2f\", ($ctx * $kb_per_token) / (1024 * 1024) }"
}

# MoE base layer fraction: ~45% for Qwen3.6 35B A3B (empirical)
MOE_BASE_FRACTION=0.45

if [ "$IS_MOE" -eq 1 ]; then
  # For MoE: only base layers go to GPU; experts handled via -ncmoe
  BASE_MODEL_GB=$(awk "BEGIN{printf \"%.2f\", $SIZE_GB * $MOE_BASE_FRACTION}")
  KV_GB=$(calc_kv_gb "${CTX:-32768}" "$IS_SSM")
  # Available for layers: budget - overhead - KV cache
  LAYER_BUDGET_GB=$(awk "BEGIN{v=$VRAM_BUDGET_GB - $ROCM_OVERHEAD_GB - $KV_GB; if(v<1)v=1; print v}")
  # Layers per GB for base model
  LAYERS_PER_GB=$(awk "BEGIN{print $BLOCK_COUNT / $BASE_MODEL_GB}")
  NGL=$(awk "BEGIN{v=int($LAYER_BUDGET_GB * $LAYERS_PER_GB); if(v<1)v=1; if(v>$BLOCK_COUNT)v=$BLOCK_COUNT; print v}")
  NCMOE=$BLOCK_COUNT  # All expert layers on CPU
  echo -e "\n🔹 MoE VRAM Math:"
  echo "   Total: ${SIZE_GB}GB | Base fraction: ${MOE_BASE_FRACTION} → ${BASE_MODEL_GB}GB"
  echo "   KV cache @${CTX:-32768}: ${KV_GB}GB | ROCm overhead: ${ROCM_OVERHEAD_GB}GB"
  echo "   Layer budget: ${LAYER_BUDGET_GB}GB → NGL=$NGL / $BLOCK_COUNT"
else
  # Dense model: standard calculation
  KV_GB=$(calc_kv_gb "${CTX:-32768}" 0)
  LAYER_BUDGET_GB=$(awk "BEGIN{v=$VRAM_BUDGET_GB - $ROCM_OVERHEAD_GB - $KV_GB; if(v<1)v=1; print v}")
  NGL=$(awk "BEGIN{v=int($LAYER_BUDGET_GB / $SIZE_GB * ${BLOCK_COUNT:-48}); if(v<1)v=1; if(v>${BLOCK_COUNT:-48})v=${BLOCK_COUNT:-48}; print v}")
  NCMOE=0
  echo -e "\n🔹 Dense VRAM Math: NGL=$NGL (budget: ${LAYER_BUDGET_GB}GB after KV: ${KV_GB}GB)"
fi

# ============================================================================
# STEP 6: Context & Mode
# ============================================================================
echo -e "\n${BOLD}Mode:${NC} 1)Quick 2)Long 3)Custom"
read -rp "Choice [1-3]: " MC; MODE="${MC:-1}"
case "$MODE" in
  1) CTX=32768 ;;
  2) echo "1)32k 2)64k 3)128k 4)256k 5)Custom"; read -rp "Choice [2]: " CC
     case ${CC:-2} in 1)CTX=32768;;2)CTX=65536;;3)CTX=131072;;4)CTX=262144;;*) read -rp "CTX: " CI; CTX=${CI:-65536};;esac ;;
  *) read -rp "CTX [32768]: " CI; CTX=${CI:-32768} ;;
esac
# Re-calc KV/NGl if context changed in custom mode
if [ "$MODE" = "3" ] || [ "$MODE" = "2" ]; then
  KV_GB=$(calc_kv_gb "$CTX" "$IS_SSM")
  if [ "$IS_MOE" -eq 1 ]; then
    LAYER_BUDGET_GB=$(awk "BEGIN{v=$VRAM_BUDGET_GB - $ROCM_OVERHEAD_GB - $KV_GB; if(v<1)v=1; print v}")
    NGL=$(awk "BEGIN{v=int($LAYER_BUDGET_GB * $LAYERS_PER_GB); if(v<1)v=1; if(v>$BLOCK_COUNT)v=$BLOCK_COUNT; print v}")
  else
    LAYER_BUDGET_GB=$(awk "BEGIN{v=$VRAM_BUDGET_GB - $ROCM_OVERHEAD_GB - $KV_GB; if(v<1)v=1; print v}")
    NGL=$(awk "BEGIN{v=int($LAYER_BUDGET_GB / $SIZE_GB * ${BLOCK_COUNT:-48}); if(v<1)v=1; if(v>${BLOCK_COUNT:-48})v=${BLOCK_COUNT:-48}; print v}")
  fi
  echo "→ Adjusted NGL=$NGL for CTX=$CTX (KV: ${KV_GB}GB)"
fi

# ============================================================================
# STEP 7: Cache, Batch, Memory
# ============================================================================
if [ "$MODE" = "3" ]; then
  echo "Cache: ${CACHE_TYPES[*]}"; read -rp "K idx[3]: " KI; CACHE_K="${CACHE_TYPES[${KI:-3}]:-$DEFAULT_CTK}"
  read -rp "V idx[0]: " VI; CACHE_V="${CACHE_TYPES[${VI:-0}]:-$DEFAULT_CTV}"
elif [ "$CTX" -ge "$ULTRA_CTX_THRESH" ]; then CACHE_K="$ULTRA_CTK"; CACHE_V="$ULTRA_CTV"
else CACHE_K="$DEFAULT_CTK"; CACHE_V="$DEFAULT_CTV"; fi

if [ "$CTX" -gt 100000 ]; then
    BATCH=1024; UBATCH=1024
elif [ "$CTX" -gt 32768 ]; then
    BATCH=2048; UBATCH=512
else
    BATCH=2048; UBATCH=512
fi

FREE_RAM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
MMAP_MODE="--no-mmap --mlock"; (( FREE_RAM < 38000 )) && MMAP_MODE="--mmap"

# ============================================================================
# STEP 8: Launch Summary
# ============================================================================
clear
echo -e "${BOLD}${GREEN}================================================================"
echo "          LLAMA-SERVER CANONICAL CONFIGURATION (v7)"
echo -e "================================================================${NC}"
echo "Model: $MNAME ($SIZE_GB GB) | MoE=$IS_MOE | SSM=$IS_SSM"
echo "CTX: $CTX | KV: $CACHE_K/$CACHE_V | NGL: $NGL | NCMOE: $NCMOE"
echo "Batch: $BATCH/$UBATCH | Cache-Reuse: 256 | MMAP: $MMAP_MODE"
echo "Threads: 8 (0-7) | Batch CPU: 8-15 | Prio: High"
[[ "$IS_SW" -eq 1 ]] && echo "⚠ Sliding Window: --no-context-shift enforced"
read -rp "Press Enter to launch..."

# ============================================================================
# STEP 9: Command Assembly
# ============================================================================
CMD=("$BINARY_DIR/llama-server" "-m" "$MODEL" "-ngl" "$NGL" "-c" "$CTX" "-b" "$BATCH" "-ub" "$UBATCH"
     "--cache-type-k" "$CACHE_K" "--cache-type-v" "$CACHE_V" "-fa" "1"
     "--temp" "$TEMP" "--top-p" "$TOP_P" "--top-k" "$TOP_K" "--min-p" "$MIN_P"
     "--threads" "8" "--threads-batch" "8" "--cpu-range" "0-7" "--cpu-strict" "1"
     "--cpu-range-batch" "8-15" "--cpu-strict-batch" "1" "--numa" "numactl" "--prio" "2"
     "--parallel" "1" "--jinja" "--metrics" "--cache-reuse" "256"
     "--reasoning" "auto" "--no-context-shift")

[ "$NCMOE" -gt 0 ] && CMD+=("-ncmoe" "$NCMOE")
[[ "$MMAP_MODE" == "--mmap" ]] && CMD+=("--mmap") || CMD+=("--no-mmap" "--mlock")
# MTP flags removed - not applicable to Qwen3.6 MoE

echo -e "\n🚀 Executing:"; echo "${CMD[*]}"
echo -e "${YELLOW}Stop: kill \$(pgrep -x llama-server)${NC}\n"
exec "${CMD[@]}"
