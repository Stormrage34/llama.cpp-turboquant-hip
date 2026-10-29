# gfx1030 (RDNA 2) fork changes — summary

> **Purpose:** one place that says *what this fork changes*, *why*, *what it measured*, and *what is
> still broken*. Written for someone picking the fork up cold — including the parts that are known
> regressions rather than wins.
>
> **Target hardware:** AMD Radeon RX 6900 XT (gfx1030), RDNA 2, wave32 native, 80 CUs,
> 128 MB Infinity Cache. Measured DRAM read ceiling 500.6 GB/s (see `docs/gfx1030-directive.md`).
>
> **Scope:** HIP/ROCm backend only (`ggml/src/ggml-cuda/`), plus a small number of sampling/server
> correctness fixes. Nothing here changes CPU behaviour, other backends, or the public C API.

> **Provenance convention.** Every factual row carries one tag and nothing else may be asserted:
> - `[measured: path]` — measured on this host; path is the evidence file or the source anchor.
> - `[ISA §x p.y]` — stated in the RDNA 2 ISA manual (`docs/rdna2-optimization-guide.md` cites it).
> - `[code]` — evident from the diff itself, no measurement implied.
> - `[open]` — known gap, not fixed.
>
> A number without a tag is a number you should re-measure before quoting.

> **Companion documents:**
> - `docs/gfx1030-benchmarking.md` — **host rules of record**: one benchmark process at a time,
>   paired A/B statistics, and this box's actual resolving power. Read before trusting any delta.
> - `docs/rdna2-optimization-guide.md` — the playbook: how to find, apply and validate a kernel
>   change on this part.
> - `docs/gfx1030-directive.md` — campaign state: what is landed, what is closed, what is next.
> - `docs/gfx1030-isa-report.md` — why the 2-blocks/CU occupancy lane is closed.
> - `docs/gfx1030-optimization-plan.md`, `docs/rdna2-agent-tasks.md`, `docs/gfx1030-test-plan.md` —
>   work packages and queued experiments.

---

## 1. TL;DR

Two commits make up this batch: `d4a3d5dbb` (*"gfx1030: kernel optimizations and correctness fixes
for RX 6900 XT"*) is landed, and everything marked **staged** below sits in the working tree awaiting
the owner's commit.

| # | Change | Effect | Status |
|---|---|---|---|
| 1 | IQ3_S sign path: multiply-spread + `v_perm_b32` + fused xor-add | −55.4 % shifts+logic, −14.9 % instructions, pp +2.72 % | prefill site landed, decode port staged |
| 2 | FA K/V quant coverage extended (`GGML_CUDA_FA_QUANTS`) | pp512 +2.15 %, no f16 K/V cast per token | `q5_0-q5_0` landed, `q4_0-q5_0`/`q5_0-q4_0` staged |
| 3 | KLONG off by default, plus a per-graph runtime guard | MTP restored to 47.0 t/s; Q6_K bug avoided | default landed, guard staged |
| 4 | Seven audit fixes (W1–W4, I2, I5, I6) | correctness/tooling, no perf claim | mixed, see §5 |
| 5 | EOG guards in spec decoding (`common/sampling.cpp`, `server-context.cpp`) | stops post-EOG token pollution | landed |
| 6 | MMVQ compile-time switches: `SMALLK=1`, `WIDEN=1`, `LBMIN=1`, `2ROWS=0` | measured-optimal per switch | staged |
| 7 | `common.cuh` audit: atomic latches, arena latch, `strtoul` validation | removes data races, bounds a bad env var | staged |
| 8 | Build dir migrated to root `build/` | binaries md5-identical to `build-hip-gfx1030/` | landed |
| 9 | `AGENTS.md`: build, hardware, models of record, baselines | onboarding | staged |

Fork vs upstream, end to end `[measured: docs/gfx1030-directive.md, scripts/bench-fork-vs-upstream.sh]`:

| Model | Config | prefill (pp) | decode (tg) |
|---|---|---|---|
| MoE 35B A3B (IQ3_S + Q4_K_M) | f16 KV | **+13.98 %** | **+6.03 %** |
| MoE 35B A3B | q4_0/q5_0 KV | **+13.95 %** | **+1.78 %** |
| Dense 27B (IQ3_S) + MTP | f16 KV | **+3.52 %** | **+5.25 %** |

Benchmark shape: `-p 1000 -n 1000 -d 5000`. These are whole-run deltas covering *all* work landed in
the fork, not just this batch (see §2). Per `docs/gfx1030-benchmarking.md`, decode deltas under
~0.3 ms/token are not resolvable without hundreds of interleaved pairs — treat the single-digit
decode percentages as directionally solid but not as precise figures.

---

## 2. Context: what was already landed before this batch

The fork was not empty before `d4a3d5dbb`. Two earlier wins explain most of the fork-vs-upstream
prefill number above, and they are easy to mistake for the changes in §4:

- **q8_1 prequant arena** (`common.cuh`, `ggml-cuda.cu`, `unary.cu`, `mmvq.cu`) — fused RMS_NORM+MUL
  writes quantised blocks into a per-context side buffer once, and downstream matvecs skip their own
  `quantize_q8_1` launch. `quantize_q8_1` 127.65 → 24.0 launches/token; perplexity unchanged
  `[measured: docs/gfx1030-checkpoint.md]`. Size is set by `GGML_CUDA_PREQUANT_ARENA_MB` (default
  32 MB) and it transparently degrades to per-matvec quantisation when the arena is full or cannot
  be allocated.
- **gfx1030 GQA decode attention** (`fattn-gqa-gfx1030.cu`, admitted on gfx1030 only) — one K/V read
  per KV head instead of per Q head: 4.4–5× over the generic path at 32k context
  `[measured: docs/gfx1030-checkpoint.md]`.
- **Residual ADD fused into RMS_NORM** (`norm.cu`, `do_preadd`) — one launch removed per layer.
- **RDNA2 MMQ config table** (`mmq-config-rdna2.cuh`, `mmq.cuh`) — gfx1030 no longer silently inherits
  another arch's tiling.

Those are documented in `docs/gfx1030-directive.md` and `docs/gfx1030-checkpoint.md`; this file does
not re-derive them.


---

## 3. Build and run

The fork builds like upstream HIP, except the build directory is the repo-root `build/`.

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON
cmake --build build --config Release -j12      # NOT -j$(nproc)
```

Binaries land in `build/bin/`. Use `-j8`..`-j12`: at `-j28` this host OOMs and ROCm clang
segfaults on E-cores `[measured: AGENTS.md]`.

`build-hip-gfx1030/` is the previous owner build directory; its binaries are **md5-identical** to
`build/bin/` `[measured: md5sum]`. Nothing depends on the old path.

Native gfx1030 code paths are guarded with `defined(RDNA2)` at compile time and with the runtime
`GGML_CUDA_CC_IS_RDNA2(cc)` check, so a multi-arch build does not enable them on other cards.

---

## 4. Kernel changes

### 4.1 IQ3_S decode sign path — `ggml/src/ggml-cuda/vecdotq.cuh`

IQ3_S builds four signed grid values per iteration, in two places: the prefill tile loader
(`ggml_cuda_mmq_load_tiles_iq3_s`, `mmq-load-tiles.cuh`) and the decode dot product
(`vec_dot_iq3_s_q8_1`, `vecdotq.cuh`). Upstream derives the sign mask with `__vcmpne4`
(compare-and-produce-0xFFFF) and then negates with `__vsub4`. On RDNA 2 both are multi-instruction
sequences, and the surrounding shifts and masks are a large share of the loop's integer work.

The fork replaces the sign extraction with a multiply-spread, a byte permute and a fused xor-add.
The prefill site was converted first; the decode site (`vecdotq.cuh`) is brought to the same
sequence so the two paths cannot drift:

```cpp
// Spread the 4 sign bits of one nibble into bit 7 of the 4 bytes of a register.
const uint32_t sign_bits = signs_packed_8[l0/2];
const int      sign_lo   = (int) ((((sign_bits & 0x0Fu) * 0x10204080u) & 0x80808080u) >> 7);
const int      sign_hi   = (int) ((((sign_bits >> 4) * 0x10204080u) & 0x80808080u) >> 7);

// v_perm_b32 expands the 0x01/0x00 byte selectors into 0x00/0xFF byte masks.
const int signs0 = ggml_hip_perm(0x0000FF00, 0x0000FF00, sign_lo);
const int signs1 = ggml_hip_perm(0x0000FF00, 0x0000FF00, sign_hi);

// Grid values are odd and in [1, 15], so the bytes of (grid ^ 0xFF) + 1 do not
// carry into the next byte: that is the byte-wise negation of the set signs.
const int grid_l = (grid_pos.x ^ signs0) + sign_lo;
const int grid_h = (grid_pos.y ^ signs1) + sign_hi;
```

Why it is bit-identical: the `0x10204080` multiply scatters nibble bits 0..3 into bit 7 of bytes
0..3; the other partial products of the dot product only set bits below bit 7 of a byte and cannot
carry into it; and the grid entries are odd and ≤ 15, so `(g ^ 0xFF) + 1` per byte is exactly the
byte-wise negation `__vsub4` produced. `ggml_hip_perm` is the existing `v_perm_b32` wrapper used by
the MMQ load tiles; on non-HIP builds `__byte_perm` is used instead `[code]`.

Validation `[measured: mmq-load-tiles.cuh, docs/rdna2-optimization-guide.md §L-07]`:

| Gate | Result |
|---|---|
| Exhaustive CPU equivalence over all sign/grid combinations | 1,048,576 cases, bit-identical |
| GPU reference comparison (`test-backend-ops -b ROCm`, MUL_MAT/FLASH_ATTN) | 2,445/2,445 pass |
| ISA census: shifts + logic | 578 → 258 (−55.4 %) |
| ISA census: total instructions | 2,328 → 1,981 (−14.9 %) |
| Prefill A/B (paired) | 592.42 → 608.51 t/s (+2.72 %) |

> **Provenance caveat.** The instruction counts and the 1,048,576-case figure were measured on the
> prefill tile-loader instance of this sequence; `docs/rdna2-optimization-guide.md` still tags them
> `[unverified]` because the raw `gfx1030.py isa` output is not stored in the tree. The decode port
> emits the same sequence, so its instruction mix should match, but re-run
> `scripts/gfx1030.py isa` before quoting the deltas as decode numbers. The prefill A/B above is
> likewise the prefill site's own measurement — the decode-path delta is small enough to sit inside
> this host's noise floor and has not been resolved separately.

This is the model of a change this fork accepts: the *only* thing that makes it safe is exhaustive
equivalence, not a passing smoke test.

### 4.2 FlashAttention K/V quant coverage — `ggml/CMakeLists.txt`

FA falls back to a warning plus **converting K and V to f16 on every call** when the cache quant pair
has no compiled kernel (`ggml_cuda_flash_attn_ext_vec`). That is the "slow" branch in the code, and
for a KV-cache-heavy model it is the dominant prefill cost. Three pairs used by this host's operating
points were missing; all are now in the default list `[code]`:

```cmake
set(GGML_CUDA_FA_QUANTS "q4_0-q4_0;q4_0-q5_0;q8_0-q8_0;q8_0-q5_0;q5_0-q5_0;q5_0-q4_0;f16-f16;bf16-bf16" ...)
```

Added relative to upstream: `q5_0-q5_0` in `d4a3d5dbb`, then `q4_0-q5_0` and `q5_0-q4_0` in the
current batch. Paired A/B on the dense model: pp512 597.39 → 610.25 t/s, **+2.15 %**
`[measured: docs/gfx1030-optimization-plan.md]`.

Trade-off: each compiled pair is extra kernel code size and compile time. `GGML_CUDA_FA_ALL_QUANTS`
still exists if a deployment needs every pair.

### 4.3 MMVQ compile-time switches — `ggml/src/ggml-cuda/mmvq.cu`

Five RDNA2-only switches choose block width and rows-per-block for `mul_mat_vec_q`. The fork sets
the measured-optimal values as defaults `[measured: docs/gfx1030-optimization-plan.md]`:

| Macro | Default | Meaning |
|---|---|---|
| `GGML_RDNA2_MMVQ_LBMIN` | `1` | `__launch_bounds__` min blocks per CU |
| `GGML_RDNA2_MMVQ_SMALLK` | `1` | borrow the GENERIC width on the `ncols_dst == 1` path |
| `GGML_RDNA2_MMVQ_KLONG` | `0` | 2-warp K-loop reduction (see next section) |
| `GGML_RDNA2_MMVQ_WIDEN` | `1` | wider block for small-row single-token matvecs |
| `GGML_RDNA2_MMVQ_2ROWS` | `0` | two weight rows per block, sharing activation loads |

All of them are additionally gated on `defined(RDNA2)`; every arm is a no-op on other targets
`[code]`. The arms themselves are compile-time, so changing one requires a rebuild — there is no
runtime switch for the four width knobs.

### 4.4 KLONG — disabled by default, guarded per graph

KLONG widens the K loop to two warps so the reduction retires in half the trips; on plain decode it
is worth **+3.75 % tg**. It is **off by default** in this fork because it is wrong for at least one
quant: with KLONG on, a Q6_K model emits `333333333` where it should emit `Paris`
`[measured: docs/gfx1030-optimization-plan.md §L-01]`. `docs/gfx1030-optimization-plan.md` records
this as **a real correctness bug, not merely a floating-point reordering** — the changed reduction
tree is the suspected cause, but that is a hypothesis, not an established mechanism.

Two mechanisms exist, and only the second is load-bearing:

1. **Compile-time default** — `GGML_RDNA2_MMVQ_KLONG 0`. Set to `1` and rebuild to re-enable.
2. **Per-graph runtime guard** — `std::atomic<bool> ggml_cuda_klong_disabled` (defined in `mmvq.cu`,
   declared in `common.cuh`) plus an RAII guard in `llama_context::process_ubatch`
   (`src/llama-context.cpp`). While an `LLM_GRAPH_TYPE_DECODER_MTP` graph is being processed, the
   flag is raised and the KLONG path in `mul_mat_vec_q_switch_ncols_dst` refuses the job; the guard
   clears it on scope exit.

Why a guard at all, when the default is already off: MTP draft graphs must never take the KLONG
path even if a future change turns the default back on. The guard makes that a property of the code
rather than of a build flag, and it is what allows non-MTP decode to keep the KLONG gain if the
Q6_K bug is ever fixed.

Current behaviour with KLONG off `[measured: docs/gfx1030-optimization-plan.md §L-01, docs/rdna2-agent-tasks.md]`:

- MTP end-to-end: **47.0 t/s**.
- Plain decode: loses the +3.75 % KLONG gain (this is the price of correctness).

`[open]` The Q6_K reduction-order bug is **not fixed** — only avoided. A fix must either change the
Q6_K accumulation so the tree order does not matter, or restrict KLONG to quant/geometry
combinations proven equivalent.

---

## 5. Correctness and tooling fixes

### 5.1 EOG handling in speculative decoding — `common/sampling.cpp`, `tools/server/server-context.cpp`

Two EOG guards were added, in the two places that accept draft tokens:

- **Sampling** (`common_sampler_sample_and_accept_n_rejection`, `common/sampling.cpp`) — as soon as
  an accepted draft token is an EOG, the acceptance loop breaks. A *trailing* EOG still yields the
  final token; a *mid-draft* EOG ends acceptance, because every token after it would be generated
  from post-EOG context and pollute the retained sequence.
- **Server trim** (`tools/server/server-context.cpp`) — the accepted list is truncated with
  `accepted.resize(k + 1)` at the first EOG, so the EOG itself is kept (it becomes `slot.sampled`)
  while the prompt insert stops right before it. This runs *after* `n_rollback` is computed, so the
  trim can never re-enter the checkpoint/replay branch.

The existing `accepted.size() - 1` arithmetic is unchanged and stays correct: no acceptance path can
now grow past an EOG except on full acceptance, where `n_rollback` is already 0 `[code]`. Landed in
`d4a3d5dbb`; the failure mode and its reproducer are recorded in `docs/gfx1030-cjk-audit.md`.

> **Record correction:** `docs/gfx1030-optimization-plan.md` summarises this as
> "`accepted.size()` → `accepted.size() - 1`". The `- 1` was already upstream and is still there;
> what the fork adds is the mid-draft EOG break in two places. Read the diff, not the summary.

### 5.2 `common.cuh` audit

- **Atomic latches.** `logged_alloc`, `logged_full` (prequant arena) and `logged` (`mmq.cu` routed-J
  debug line) were plain `static bool` written from multiple backend threads. They are now
  `std::atomic<bool>` and set with `exchange(true)`, so a warn-once line prints **exactly** once and
  the unsynchronised read-modify-write is gone `[code]`.
- **`alloc_failed` latch.** A failed arena `cudaMalloc` used to be retried on every call, i.e. a
  failing allocation became a per-launch driver round-trip. It is now latched, and
  `ggml_cuda_prequant_add_impl` returns `nullptr` immediately when the arena is unusable. The
  pre-existing graceful degradation (fused norm skips quantisation, consumers run their own
  `quantize_q8_1`) is unchanged `[code]`.
- **`strtoul` trailing garbage.** `GGML_CUDA_PREQUANT_ARENA_MB` parsing accepted `"64junk"` as 64.
  The check now requires `*end == '\0'` in addition to the existing `end == env`, `ERANGE`, zero and
  absurd-value rejects. A typo'd value falls back to the 32 MB default instead of being honoured
  `[code]`.

### 5.3 FlashAttention launch bounds — `fattn-gqa-gfx1030.cu`

The gfx1030 GQA kernel's `__launch_bounds__` min-blocks argument moved from 1 to 2, with the
comment corrected to match: the register budget at this geometry allows the compiler to target
2-block occupancy without spilling the accumulators, and that measured faster on the verify path
`[measured: docs/gfx1030-optimization-plan.md]`. Note this is a *compiler hint*, not a promise of
two resident blocks — see `docs/gfx1030-isa-report.md` for why two blocks are architecturally
impossible for the MMQ prefill kernel at J=64.

### 5.4 Tooling fixes in `scripts/gfx1030.py` (audit items W1, W2, I6)

- **W1** — the A/B "swap" arm launched with the arms in the wrong order, so the interleaved pair
  was not actually swapped.
- **W2** — a failed arm produced `NaN`/`ZeroDivisionError` in the statistics instead of being
  reported as a failed arm.
- **I6** — `--dry-run` printed the wrong command because it did not use `_llama_bench_args`.

Also from the same audit: **I2** is a comment correction and **I5** fixes undefined behaviour in
`tolower` by passing `unsigned char`-cast input. These carry no performance claim.

---

## 6. Reproducing the numbers

The host rules in `docs/gfx1030-benchmarking.md` are binding, not advisory:

```bash
# Preflight: exactly one GPU process, nothing resident
pgrep -a -f "llama-server|llama-bench"     # must print nothing
rocm-smi --showmemuse                      # must show ~0

# Enforced A/B runner (one process at a time, interleaved pairs)
scripts/bench-exclusive.sh
```

- Fork-vs-upstream sweep: `scripts/bench-fork-vs-upstream.sh`.
- Paired A/B statistics: `scripts/gfx1030.py bench --ab`, `--ab-stats "d1,d2,..."`,
  `--min-detectable <ms>` for the pair count needed at a given effect size.
- Upstream reference build for comparison: `/tmp/opencode/llama-upstream/build/bin/llama-bench`.

`llama-server` must never be launched, stopped or health-checked for a measurement on this host; all
numbers come from `llama-bench` and `test-backend-ops`.

---

## 7. Known limitations

- **KLONG is off.** The Q6_K reduction-order bug is avoided, not fixed. Plain decode gives up
  +3.75 % tg until it is.
- **Decode deltas are near this box's noise floor.** `sd` is ~1.56 t/s (~2.3 %) run to run; the
  fork-vs-upstream decode percentages above are consistent across repeats but should not be quoted
  to two decimals.
- **MoE numbers are `-ncmoe 0`.** The 16.05 GB MoE file does not fit alongside a 256k KV cache, so
  the declared production operating point streams experts over PCIe and has a different byte mix.
  No measurement exists at that operating point `[open]`.
- **Attention path targets are not met.** Prefill sits at ~31 % of the measured dp4a ceiling; upstream
  parity is not the bar (`docs/gfx1030-directive.md`).
- **RDNA2 switches are compile-time.** No runtime tuning; a wrong value costs a rebuild.
- **Nothing here is upstreamed.** The fork's own policy (see `AGENTS.md`) is that agents produce
  analysis and diffs and a human lands everything.

---

## 8. Document index

| File | Contents |
|---|---|
| `docs/gfx1030-changes-summary.md` | this file — what changed and what it measured |
| `docs/gfx1030-benchmarking.md` | host measurement rules, A/B statistics, resolving power |
| `docs/rdna2-optimization-guide.md` | RDNA 2 kernel optimization playbook |
| `docs/gfx1030-directive.md` | campaign state, work packages, closed lanes |
| `docs/gfx1030-isa-report.md` | the LDS/VGPR measurement that closed the occupancy lane |
| `docs/gfx1030-optimization-plan.md` | the optimization work plan |
| `docs/rdna2-agent-tasks.md` | task breakdown for agents |
| `docs/gfx1030-test-plan.md` | queued experiments T0–T7 with pass/fail predicates |
| `docs/gfx1030-checkpoint.md` | review manifest: per-area diff inventory of the working tree |
| `docs/gfx1030-cjk-audit.md` | EOG/post-EOG pollution audit and reproducer |
| `AGENTS.md` | build commands, hardware, models of record, baselines |
