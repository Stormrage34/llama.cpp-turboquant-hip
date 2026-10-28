#!/usr/bin/env bash
# Quality ladder: how much does each expert-tensor quantization cost?
#
# WHAT A KL-DIVERGENCE LADDER IS
#   llama-perplexity can compare a candidate model against a reference model
#   token by token. For every position it takes the next-token probability
#   distribution of the reference (saved once as base.dat) and the distribution
#   of the candidate, and reports the KL divergence: roughly "how many extra
#   bits does the candidate waste predicting text the reference found easy".
#   Zero means the two models are indistinguishable on this text; higher means
#   the quantization damaged the model more.
#
#   A ladder is the same measurement run over several candidate quantizations
#   ("arms"), cheapest-to-most-expensive, so you can see where quality
#   collapses and whether one extra bit per weight actually buys anything.
#   Here every arm differs ONLY in the expert-tensor type, so the ladder
#   answers exactly one question: expert bitrate vs damage.
#
# WHY EVERY ARM MUST COME FROM THE SAME SOURCE
#   Each arm is re-quantized from the same source GGUF with --allow-requantize.
#   Requantizing A -> B and A -> C compares B against C; quantizing B -> C and
#   A -> C would instead compare "B plus another generation of loss". The first
#   pass of this ladder made that mistake in a subtler form: IQ3_XXS got an
#   importance matrix and IQ3_S did not, so IQ3_XXS "won" on quality despite
#   fewer bits per weight. Every arm here gets the same treatment.
#
# WHY PLACEMENT AND -ncmoe MUST MATCH ACROSS ARMS
#   -ncmoe N keeps the expert weights of N layers on the CPU ("N cpu MoE").
#   That is not a cosmetic option: it moves hundreds of megabytes out of VRAM,
#   changes which kernels run, and slightly changes the floating-point
#   arithmetic. If arm A runs with -ncmoe 9 and arm B with -ncmoe 0, the KLD
#   difference contains quantization damage AND placement damage, and you
#   cannot separate them. So -ngl, -ncmoe, -fa, -ctk, -ctv, -c, -b, -ub, -dev
#   and --chunks are held fixed for every arm and for the reference run, and
#   the importance matrix is collected with that same placement.
#
# RULES HONOURED
#   llama-server is never started, stopped or queried. Exactly one GPU process
#   at a time (imatrix -> base.dat -> quantize -> KLD, strictly sequential).
#   The quantize step runs with every GPU device hidden so it cannot contend
#   for VRAM with a measurement.
#
# ARTIFACTS
#   <models-dir>/<src-stem>-exps-<arm>-im.gguf   one file per arm (15-19 GB each)
#   <work-dir>/imatrix.gguf                      importance matrix
#   <work-dir>/base.dat                          reference logits (~2 GB)
#   <work-dir>/*.log                             one log per step
#
# USAGE
#   scripts/quality_ladder.sh -s SRC.gguf -c corpus.txt
#   scripts/quality_ladder.sh -s SRC.gguf -c corpus.txt -a "IQ4_XS IQ3_S IQ3_XXS"
#   scripts/quality_ladder.sh --help
#
#   -s, --src FILE          source GGUF every arm is requantized from (required)
#   -c, --corpus FILE       plain-text corpus for imatrix and KLD (required)
#   -a, --arms "A B ..."    expert quant types to build (default: "IQ3_XXS IQ3_S")
#   -o, --models-dir DIR    where arm GGUFs live (default: ~/Downloads)
#   -O, --work-dir DIR      logs, imatrix, base.dat (default: /tmp/opencode/quality)
#   -q, --quantize-bin F    llama-quantize      (default: <root>/build/bin/...)
#   -p, --perplexity-bin F  llama-perplexity    (default: build-hip-gfx1030)
#   -i, --imatrix-bin F     llama-imatrix       (default: build-hip-gfx1030)
#   -n, --chunks N          KLD chunks          (default: 16)
#       --imatrix-chunks N  imatrix chunks      (default: 24)
#   -t, --threads N         CPU threads         (default: 6)
#       --ncmoe N           MoE layers kept on CPU, all arms (default: 9)
#   -g, --need-gb N         free space required next to the arm files (default: 18)
#       --source-type T     llama-quantize output type argument (default: Q4_K_M)
#       --force             rebuild imatrix and base.dat even if present
#   -h, --help              this text
#
# Re-running is cheap by design: an arm whose GGUF already exists with a valid
# magic number is not rebuilt. A half-written arm (killed mid-run) is detected
# by its magic bytes and removed before rebuilding.
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$ROOT/scripts/common.sh"

# The dead MTP/nextn block. llama-imatrix reports every blk.40 tensor as
# "unused ... ignoring", so it never gets an imatrix entry, and quantizers that
# require an entry refuse to convert it. Excluding blk.40 from the override
# leaves it at the source type, which is harmless: it does not execute in
# non-MTP decode. The regex therefore covers blk.0 .. blk.39 only.
OVERRIDE_RE='blk\.([0-9]|[1-3][0-9])\.ffn_(gate|up|down)_exps\.weight'

usage() { sed -n '2,/^set -u$/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

SRC=""
CORPUS=""
ARMS="IQ3_XXS IQ3_S"
MODELS_DIR="$HOME/Downloads"
WORK=/tmp/opencode/quality
QUANTIZE="$ROOT/build/bin/llama-quantize"
PERPLEXITY="$ROOT/build-hip-gfx1030/bin/llama-perplexity"
IMATRIX_BIN="$ROOT/build-hip-gfx1030/bin/llama-imatrix"
CHUNKS=16
IMATRIX_CHUNKS=24
THREADS=6
NCMOE=9
NEED_GB=18
SOURCE_TYPE=Q4_K_M
FORCE=0

die() { echo "ABORT: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        -s|--src)         SRC="${2:?-s needs a path}"; shift 2 ;;
        -c|--corpus)      CORPUS="${2:?-c needs a path}"; shift 2 ;;
        -a|--arms)        ARMS="${2:?-a needs a list}"; shift 2 ;;
        -o|--models-dir)  MODELS_DIR="${2:?-o needs a path}"; shift 2 ;;
        -O|--work-dir)    WORK="${2:?-O needs a path}"; shift 2 ;;
        -q|--quantize-bin)    QUANTIZE="${2:?-q needs a path}"; shift 2 ;;
        -p|--perplexity-bin)  PERPLEXITY="${2:?-p needs a path}"; shift 2 ;;
        -i|--imatrix-bin)     IMATRIX_BIN="${2:?-i needs a path}"; shift 2 ;;
        -n|--chunks)      CHUNKS="${2:?-n needs a number}"; shift 2 ;;
        --imatrix-chunks) IMATRIX_CHUNKS="${2:?needs a number}"; shift 2 ;;
        -t|--threads)     THREADS="${2:?-t needs a number}"; shift 2 ;;
        --ncmoe)          NCMOE="${2:?--ncmoe needs a number}"; shift 2 ;;
        -g|--need-gb)     NEED_GB="${2:?-g needs a number}"; shift 2 ;;
        --source-type)    SOURCE_TYPE="${2:?--source-type needs a type}"; shift 2 ;;
        --force)          FORCE=1; shift ;;
        *)                die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -n "$SRC" ]    || die "-s/--src is required (the reference GGUF); try --help"
[ -n "$CORPUS" ] || die "-c/--corpus is required (plain text); try --help"
[ -f "$SRC" ]    || die "source model not found: $SRC"
[ -f "$CORPUS" ] || die "corpus not found: $CORPUS"
for bin in "$QUANTIZE" "$PERPLEXITY" "$IMATRIX_BIN"; do
    [ -x "$bin" ] || die "not executable: $bin"
done
[ -d "$MODELS_DIR" ] || die "models dir does not exist: $MODELS_DIR"

mkdir -p "$WORK" || die "cannot create work dir: $WORK"
IMAT="$WORK/imatrix.gguf"
BASE="$WORK/base.dat"
SRC_STEM="$(basename "$SRC" .gguf)"

# /tmp is a 32 GB tmpfs on this host. A 16 GB GGUF written there dies mid-run
# with "basic_ios::clear: iostream error", which looks like a quantizer bug and
# is not one. Large artifacts go next to the source; only base.dat and logs
# stay in the work dir.
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
HAVE_GB="$(free_gb "$MODELS_DIR")"
HAVE_GB="${HAVE_GB:-0}"

echo "=== [1/5] preflight ==="
echo "  src        $SRC"
echo "  corpus     $CORPUS"
echo "  arms       $ARMS"
echo "  arm files  $MODELS_DIR/${SRC_STEM}-exps-<arm>-im.gguf"
echo "  work dir   $WORK"
echo "  placement  -ngl 99 -ncmoe $NCMOE -fa $FA -ctk $CTK -ctv $CTV -dev ROCm0"
if [ "$HAVE_GB" -lt "$NEED_GB" ]; then
    echo "ABORT: need ${NEED_GB} GB free on $MODELS_DIR, have ${HAVE_GB} GB" >&2
    echo "       (an arm is 15-19 GB; partial writes on a full disk look like quantizer bugs)" >&2
    exit 1
fi
echo "  disk ok: ${HAVE_GB} GB free on $MODELS_DIR"

# is_gguf: a finished GGUF file starts with the four ASCII bytes "GGUF". A
# killed quantize leaves a large file with garbage magic; that is the only
# reliable way to tell "complete" from "truncated" without re-hashing 16 GB.
is_gguf() { [ "$(head -c 4 "$1" 2>/dev/null)" = GGUF ]; }

echo "=== [2/5] importance matrix ==="
# The importance matrix is per-tensor activation statistics gathered by running
# the source model over text. It is GPU work, so it runs with the same
# placement as the KLD runs; collecting it elsewhere would bias the arms.
if [ "$FORCE" = 1 ]; then rm -f "$IMAT"; fi
if [ -s "$IMAT" ]; then
    echo "  present: $IMAT ($(stat -c%s "$IMAT") bytes)"
else
    echo "  collecting (GPU, $IMATRIX_CHUNKS chunks) ..."
    if ! "$IMATRIX_BIN" -m "$SRC" -f "$CORPUS" -o "$IMAT" \
            --chunks "$IMATRIX_CHUNKS" -t "$THREADS" -ngl 99 -ncmoe "$NCMOE" \
            -b 64 -ub 64 -dev ROCm0 > "$WORK/imatrix.log" 2>&1; then
        echo "IMATRIX FAILED - last lines:"; tail -5 "$WORK/imatrix.log"
        exit 1
    fi
    [ -s "$IMAT" ] || { echo "IMATRIX FAILED: no output at $IMAT"; exit 1; }
    echo "  imatrix rc=0 size=$(stat -c%s "$IMAT")"
fi

echo "=== [3/5] reference logits from the source (this is the KLD base) ==="
# base.dat holds the source model's next-token distributions. The control run
# below compares the source against itself and must print ~0, otherwise the
# base file is stale or was built with different flags and every arm is bogus.
if [ "$FORCE" = 1 ]; then rm -f "$BASE"; fi
if [ -s "$BASE" ]; then
    echo "  present: $BASE ($(stat -c%s "$BASE") bytes)"
else
    if ! "$PERPLEXITY" -m "$SRC" -f "$CORPUS" -c 512 -b 64 -ub 64 \
            --chunks "$CHUNKS" -t "$THREADS" -ngl 99 -ncmoe "$NCMOE" \
            -fa $FA -ctk $CTK -ctv $CTV -dev ROCm0 \
            --save-all-logits "$BASE" > "$WORK/ref.log" 2>&1; then
        echo "REFERENCE FAILED - last lines:"; tail -5 "$WORK/ref.log"
        exit 1
    fi
    [ -s "$BASE" ] || { echo "REFERENCE FAILED: no output at $BASE"; exit 1; }
    echo "  base.dat size=$(stat -c%s "$BASE")"
fi

report() { # report <tag> <log>
    local tag="$1" log="$2" line
    printf '  %-24s' "$tag"
    line="$(grep -a "Mean    KLD" "$log" | tail -1)"
    printf '%s' "${line:-mean KLD: n/a}"
    line="$(grep -a "RMS     KLD" "$log" | tail -1)"
    [ -n "$line" ] && printf '  |  %s' "$line"
    echo
}

kld_run() { # kld_run <tag> <model> <log>
    local tag="$1" model="$2" log="$3"
    if ! "$PERPLEXITY" -m "$model" -f "$CORPUS" -c 512 -b 64 -ub 64 \
            --chunks "$CHUNKS" -t "$THREADS" -ngl 99 -ncmoe "$NCMOE" \
            -fa $FA -ctk $CTK -ctv $CTV -dev ROCm0 \
            --kl-divergence --kl-divergence-base "$BASE" > "$log" 2>&1; then
        echo "  kld FAILED for $tag - last lines:"; tail -5 "$log"
        return 1
    fi
    report "$tag" "$log"
    return 0
}

FAILED=0
echo "  control: source against itself (must be ~0)"
kld_run "$(basename "$SRC") (control)" "$SRC" "$WORK/kld-control.log" || FAILED=1

echo "=== [4/5] arms ==="
DONE=()
# Deliberate word-split: ARMS is a space-separated list, one quant type per arm.
for arm in $ARMS; do
    out="$MODELS_DIR/${SRC_STEM}-exps-${arm}-im.gguf"
    log="$WORK/quant-exps-${arm}.log"
    klog="$WORK/kld-exps-${arm}.log"
    echo "--- $arm -> $(basename "$out") ---"

    if [ -s "$out" ] && ! is_gguf "$out"; then
        echo "  corrupt partial file ($(stat -c%s "$out") bytes, bad magic) - removing"
        rm -f "$out"
    fi

    if [ -s "$out" ]; then
        echo "  present and valid, skipping quantize"
    else
        # Every GPU device is hidden here: llama-quantize does not need the GPU,
        # and a quantizer holding VRAM would corrupt a concurrent measurement.
        HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= GGML_VK_VISIBLE_DEVICES= \
            nice -n 10 "$QUANTIZE" --allow-requantize --imatrix "$IMAT" \
            --tensor-type "$OVERRIDE_RE=$arm" \
            "$SRC" "$out" "$SOURCE_TYPE" "$THREADS" > "$log" 2>&1
        rc=$?
        ovr=$(grep -ac "applying manual override" "$log")
        echo "  quantize rc=$rc overrides=$ovr size=$(stat -c%s "$out" 2>/dev/null)"

        if [ "$rc" -ne 0 ] || ! is_gguf "$out"; then
            echo "  QUANTIZE FAILED - last lines:"; tail -5 "$log"
            rm -f "$out"
            FAILED=1
            continue
        fi
        # Zero overrides means the regex matched nothing and the "arm" is just
        # a copy of the source type. That is a silent no-op, not a result.
        if [ "$ovr" -le 0 ]; then
            echo "  QUANTIZE IGNORED THE OVERRIDE (0 matches) - the arm would not"
            echo "  differ from $SOURCE_TYPE; check --tensor-type support and the model"
            rm -f "$out"
            FAILED=1
            continue
        fi
    fi

    echo "  measuring KLD ..."
    if kld_run "exps-$arm" "$out" "$klog"; then
        DONE+=("$arm")
    else
        FAILED=1
    fi
done

echo "=== [5/5] summary ==="
for arm in "${DONE[@]:-}"; do
    [ -n "$arm" ] || continue
    report "exps-$arm" "$WORK/kld-exps-$arm.log"
done
[ "$FAILED" -eq 0 ] || echo "one or more steps failed; see $WORK/*.log"

echo "QUALITY_LADDER_DONE rc=$FAILED"
exit "$FAILED"
