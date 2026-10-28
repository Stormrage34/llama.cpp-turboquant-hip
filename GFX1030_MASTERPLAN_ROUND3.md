# gfx1030 × Qwen 3.5 — round 3: delta, deeper source mining, gated patch set

Status: uncommitted planning document. **This is a delta to `GFX1030_QWEN35_MASTERPLAN.md`, not a
rewrite.** Sections that the masterplan already settles are cited, not repeated. Where this document
contradicts the masterplan, the contradiction is named and the resolution given; `N1_N4_PATCH_SPEC.md`
governs wherever the two touch N1/N4.

Host: RX 6900 XT, gfx1030, ROCm 7.2.4/7.2.5 (`7.2.53211-3d9ef42`), one card, 15.98 GiB usable.
Tree: `HEAD = d327bf1f0` + 27 modified / 23 untracked files. Model of record:
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, qwen35 hybrid, 65 blocks (48 GDN + 16 full-attn + 1
MTP), 12.109 GB, 3.546 bpw.
No GPU benchmark was run for this round. Every **[P]** below is either an existing artifact in this
repo or an *offline* inspection of the already-built library
(`build-hip-gfx1030/bin/libggml-hip.so.0.27.0`, sha256 `9587e5f81ee2783f…`, built 05:22).

Tag legend unchanged: **[P]** proven on this machine · **[U]** proven upstream, cited to file/hunk ·
**[H]** hypothesis with a stated discriminator · **[X]** refuted / refused.

---

## 0. Delta summary — what is now stale, and what it changes

| # | masterplan said | now | ranking effect |
|---|---|---|---|
| D1 | H-CLOCK open | **closed** — sclk 2485 MHz on both backends in every in-window sample of five runs (47/47, 48/48, 47/47, 53/53, 54/54); ratio 1.0000 vs throughput ratio 1.0597. Card sits in `auto`; sysfs read-only. See masterplan §"Closed hypotheses". | none (removes a candidate) |
| D2 | Vulkan 34.4 t/s | **`-dev` must be explicit**: `build/bin` is dual-backend, `-dev auto` = ROCm0 (32.07). Real Vulkan `-dev Vulkan0` = 34.22 t/s / 29.22 ms. Gap +6.0 % mean. | none (hardens §10.3) |
| D3 | (context note) the new VEC `q8_0-q5_0` instance lands on "the MTP verify width" | **wrong width.** `n_max=2` submits `n_max+1 = 3` rows (`common/speculative.cpp:1588`, `c.reserve((n_max+1)*n_embd)`), and the selector sends **3 → TILE** (`fattn.cu:828-830`: quantized KV, `Q->ne[1] <= 2 → VEC`, else TILE). The new instance lands on **`n_max=1` (width 2)** and on any 2-row batch — not on the recommended config. | **N3 not boosted by it**; the FA_QUANTS change is a *correctness-of-attribution* fix (§0.1) |
| D4 | N1 "to implement" | **implemented in-tree** (`MMVQ_PARAMETERS_RDNA2_BAND`, widths 2–4 → `nwarps 2, rows 2`, width 1 refused). Byte gate already shows the designed signature. | N1 becomes *in flight*, not *planned* (§0.2) |
| D5 | byte law corrected | unchanged | none |
| **D6** | N6 = "prefill attention only, cols=4 row keeps our tile shape" | **half wrong.** The cols=4 row *is* our verify-width tile shape (ncols1=2 × ncols2=2), and its only change is the `__launch_bounds__` occupancy field 6→1. It is a **verify-band codegen** experiment, not prefill-only. | **N6 promoted** to the tier that touches the MTP pass (§1.1, §5) |
| **D7** | tile FA "falls back to f16" (unpriced) | **the TILE path stages the whole K and V to f16 on every call** (`fattn.cu:854-855` → `need_f16_K/V`; `fattn-common.cuh:1031-1058` → `to_fp16(.. ggml_nelements(K) ..)`). At width 3 this is the deployed verify path. Cost scales with context: **≈1.4 ms per verify pass at 4k, ≈30 ms at 86k** (a pass is ~41 ms — see D12), plus 2.3× the KV read bytes. | **N3 becomes the largest item on the board at depth** and the only exactness-preserving one (§2) |
| **D12** | "the 19.65 ms pass" (round-2 wording, inherited from the §6.3 table) | **mislabel.** 19.65 is **ms per emitted token**, not per pass. With `α = 0.661` and `n_max = 2`, expected accepted per round is `(1−α³)/(1−α) = 2.10`, so **a verify pass is ≈41 ms** and emits ≈2.10 tokens (`2.10/41.3 ms = 50.8 t/s`, matching the measured 50.9). Cross-check: `n_max=1` ⇒ 1.82 tokens/pass at 40.9 ms; `n_max=3` ⇒ 2.06 at 43.8 ms — the pass cost is ~41–44 ms for all three. | every "per pass" figure in §2 is now correct; **N3's d0 value is ~3 %, not ~6 %** |
| **D8** | — | codegen of the exact kernels is **inspectable offline** (`llvm-readobj --notes` on the extracted code object): tile<256,256,2,2> = **159 VGPR, 0 spill, 19 200 B LDS**; GQA kernel = **256 VGPR, 4 spills**; MMVQ iq3_s w1/w3 = **47/62 VGPR, 0 spill**. | new zero-GPU gate for every `__launch_bounds__`/table change (§5, tool T1) |
| **D9** | edwin's Q8_1 cache "note only" | **it cannot fire on this model at all** (`q8_cache_eligible` requires `src0->type == GGML_TYPE_Q8_0`; the census has no Q8_0 weights). Its real asset is the **failure mode** it documents, not the cache. | port nothing; add 3 invariants to the O11′ audit (§1.3) |
| **D10** | stew675's native-KV FA | **MMA-only** — the code says "Always NONE for the tile and vec kernels, which stage whichever type they were compiled for". So the delivery's native-KV work does **not** remove the gfx1030 staging cost. | N3 stays unclaimed by anyone (§1.1) |
| **D11** | §2 C4: "64 KB LDS per WGP (HIP)" | **imprecise.** HIP's `sharedMemPerMultiprocessor = 65536` is *per CU*; `multiProcessorCount = 40` is *per WGP*. The WGP has 2 CUs → 128 KiB aggregate, but one block lives in one CU and is capped at 64 KiB, which is why the tile kernel bounds at 3×19.2 KB = 57.6 KB (90 % of 64 KB) rather than 6 blocks. | corrects C4; matters for any occupancy arithmetic (§1.1) |

### 0.1 What D3 means for the record

The `GGML_CUDA_FA_QUANTS` default now includes `q8_0-q5_0` (`ggml/CMakeLists.txt:208`) and the instance
is generated (12 occurrences of `GGML_CUDA_FA_Q8_0_Q5_0` in `build.make`). That is a **good** change and
it will remove a per-call f16 conversion for VEC — but at `n_q = 2`, which our recommended
configuration does not use. Two configs benefit: `--spec-draft-n-max 1` (44.4 t/s, already the slower
choice) and exactly-2-row batches. **[P]** for the code paths, **[H]** for the gain (~1 conversion
per layer per pass at width 2).

The masterplan's §6.3 finding #3 (three kernels across widths 1/2/3-8) is unaffected and is now the
*strongest* argument for N3, not a footnote: adding a VEC instance does not reduce the number of
kernels a speculative round touches.

### 0.2 What D4 means for the record

N1 is in the tree, `MMVQ_PARAMETERS_RDNA2_BAND` split from `MMVQ_PARAMETERS_RDNA2` (so gfx115x is
untouched), widths 2–4 only, width 1 explicitly refused. The byte gate (Appendix A harness) is run and
matches the design:

| width | q4_K | q6_K | iq3_s | iq2_s | iq4_xs | iq3_xxs |
|---|---|---|---|---|---|---|
| 1 | **identical** | identical | identical | identical | identical | identical |
| 2 | 1.34e-5 | 9.5e-6 | 1.14e-5 | 9.5e-6 | 1.14e-5 | 1.14e-5 |
| 3 | 1.34e-5 | 9.5e-6 | 1.14e-5 | 1.14e-5 | 1.14e-5 | 1.14e-5 |
| 4 | 1.34e-5 | 1.14e-5 | 1.14e-5 | 1.53e-5 | 1.14e-5 | 1.14e-5 |
| 6 | **identical** | identical | identical | identical | identical | identical |
| 8 | **identical** | identical | identical | identical | identical | identical |

(max-abs diff vs the stock build, `n1/b1/*.bin`, 1024 rows; max-rel ≤ 7.4e-4.)
Widths 1/6/8 are untouched by construction and are byte-identical; 2–4 differ by fp32
reduction-reorder only, which is the documented substitute gate. **Do not re-litigate N1's geometry in
this round.**

---

## A. Public RDNA2 / gfx1030 findings — deeper, not wider

Everything below is *new* relative to masterplan §5. The four sources the brief names were re-read at
file/hunk level with our tree open beside them; two of them (ROCm docs, gfx1030 wiki) yielded nothing
new and are disposed of in one line each (§A.4).

### A.1 `stew675/llama-cpp-rdna-boosts` @ `974debf82` (r148) — the five RDNA2 patches, mapped

The add-on is `archive/work/rdna2/pr165/patches-rdna2/000{0..4}` (folded into blocks 05/07). Each was
diffed against our tree. The interesting result is how little of it lands on a path we execute.

| patch | mechanism | our file/function | verdict |
|---|---|---|---|
| **00** tile-FA config `_amd_rdna2` | per-(DKQ,DV,ncols) table + `launch_fattn` occupancy recompute + `cols_per_block`/`ncols2` caps | `fattn-tile.cuh:237` (`_amd_rdna`), `:315/:328` (host/device select), `fattn-common.cuh:1137` (occupancy query) | **two rows matter, one idea refused** — see below |
| **01** partial-batch load | branch instead of pointer-select between `KV` and a local `zero[]` | `fattn-tile.cuh:407-410` and `:457-460` (**present verbatim today**) | **port, 12 lines, byte-identical** (N5) |
| **02** Q from global half2 | `flash_attn_q_to_half2` kernel + `Q_h2` buffer + kernel-arg plumbing; `Q_tmp` shrinks to a stub | `fattn-tile.cuh` Q staging block, `fattn-common.cuh:58` (`fattn_kernel_t`), tile/vec/mma kernel signatures | **portable, prefill-only, deferred** (N9) |
| **03** k-quant | `ggml_cuda_mmq_scale_sumi` float scale + `MMVQ_PARAMETERS_NAVI21` table | `vecdotq.cuh:552/615/678`; `mmvq.cu:597/631` | N4 = bit-identical [U]; **N1 already took the table, as a band, not an arch row** [P] |
| **04** build guards | gfx1030-only build without WMMA builtins | — | **not needed**: our tree gates WMMA at runtime (`amd_wmma_available()`) |

**Patch 00, row by row.** Only two of the twenty rows are on a path this model executes.

| row | upstream (`_amd_rdna`) | RDNA2 table | change | is it ours? |
|---|---|---|---|---|
| (256,256,**4**) | 128 thr, occ **6**, fa 32, K 256 | 128 thr, occ **1**, fa 32, K 256 | **launch-bounds field only** | **yes — this is the width-3 verify tile** (ncols1=2 × ncols2=2 = 4). Codegen today: **159 VGPR, 0 spill, 19 200 B LDS** [P]. occ 6 needs 24 waves × 159 VGPR = 122 112 ≤ 131 072, so the bound is *satisfiable but tight*; relaxing it to 1 lets the allocator spend up to 1024 VGPR. This is a real, one-line, width-3 experiment (N6′). |
| (256,256,**32**) | 256 thr, occ **3**, fa **64**, K **128** (line 297, not 216 — the two tables differ) | 256 thr, occ **1**, fa 64, K **64** | **nbatch_K halved** (numerics: online-softmax chunking) + launch-bounds hint | yes, prefill (`Q->ne[1]=512`, ncols2=2 → cols 32). Codegen today: **207 VGPR, 0 spill** [P] against a 3-block bound that needs ≤170 VGPR ⇒ the bound is not met and the occ field is *dead information* for this row; LDS ≈34 KB ⇒ 2 blocks. The RDNA2 row (K 64) ⇒ ≈29 KB, still 2 blocks. **Q-from-global is the only one of the three that changes the block count** (34→18 KB ⇒ 3 blocks). Upstream never measured the row and Q-from-global apart, which is why N9 must come *after* the row is tested, not before. |
| all D=128 / 512 / 576, and the `ncols2`/`cols_per_block` caps | — | caps keyed on `DKQ==128/512/576`, or `DKQ==256 && ncols2==1` | — | **inert for this model**: `DKQ=256, gqa_ratio=6 ⇒ ncols2=2`, so `rdna256_mha` is false and no cap fires. |

**The occupancy recompute is refused, and the reason is now stronger than in round 2.** The patch
replaces `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with its own arithmetic on the stated premise
"HIP assumes regsPerMultiprocessor = 32768 on RDNA2, a quarter of the real file". We measured
**131 072** [P], so the premise is absent on ROCm 7.2.4. Independently, the patch's own arithmetic
divides LDS by `128*1024` while its comment says "64 KB LDS/WGP" — the two are only consistent if
"multiprocessor" means WGP for registers and CU for LDS, which is exactly the confusion to avoid. Our
own measurement sides with the *CU* reading: the tile kernel's 19 200 B block bounds at 3 per 64 KB
(94 %), not 6. **Do not port this function; do not copy its LDS constant.**

**Patch 01 (N5).** Our lines are the pre-image of the patch, character for character apart from
`half2` vs the patch base's `T_in` template (`fattn-tile.cuh:407` and `:457`). It is a latent-fault
class, not a measured one: our full `FLASH_ATTN_EXT` suite (4045/4045) already includes the upstream
kv=113 case. Cost 12 lines, zero perf, zero numerics. Verification is the suite plus a byte-diff, and
it future-proofs a ROCm upgrade (the upstream reproducer is ROCm 7.14).

**Patch 02 (N9).** Bit-identical *by construction* on this arch: our Q staging is
`tmp_f[i] *= scale` then `make_half2` with **no 0.25 down-scale**, because `V_DOT2_F32_F16_AVAILABLE`
is defined for RDNA2 (`common.cuh:761-763`). The patch's conversion kernel computes exactly
`make_half2(q[2i]*scale, q[2i+1]*scale)`, so the Q bytes are identical and the KQ accumulation order is
untouched. The cost is not the conversion kernel — it is the **API change**: one extra `Q_h2` argument
on `fattn_kernel_t` and on all three kernels, the delivery calls it "the only unconditional change".
Value for us is prefill-only (`ncols1*ncols2 ≥ 16`): LDS ≈34 KB → ≈18 KB at ncols=32 flips 2 → 3 blocks
(reg-bound at 207 VGPR). Deferred behind N3 because prefill is already +23 % over Vulkan.

**Patch 03 / N4.** Unchanged from round 2: `mmq_scale_sumi` is claimed bit-identical because
`(float)(sc*sumi)` and `(float)sc*(float)sumi` are both the correctly-rounded exact integer product
(`sc ≤ 255`, `sumi < 2^17`). Exposure on this model is **MMQ-only, ≤2 % of prefill** (1.966/25.6 Gelem
= 7.7 % of weights, of which only the MMQ-covered share). Keep behind N3/N5.

### A.2 Two failure modes in the same worklog that are worth more than the patches

**F-ARCH (#218 part 1, gfx1100 primary + gfx1030 secondary).** Every per-arch decision in `mmb.cu`
read `ggml_cuda_info().devices[0].cc`. On a mixed-arch box the policy froze at the primary's, and the
**gfx1030 card ran RDNA3-tuned MMB kernels** — byte-exact *zero* output-projection logits (PPL == vocab)
or garbled tail-layer GEMMs. Fix: read the *running* device's cc; plus `ggml_cuda_set_device(cuda_ctx->device)`
at the top of `ggml_backend_cuda_graph_optimize()`, because the marking pass ran with a stale current
device. **[U]**

Audit of our tree: `devices[0]` appears once, `allreduce.cu:493`, and it means "first device in the
pipeline", not "arch oracle" — clean. `ggml_backend_cuda_graph_optimize` (`ggml-cuda.cu:5257`) does
**not** call `ggml_cuda_set_device` first **[P]**. On one GPU that is a no-op; it is the same latent
class if this box ever gains a second card. Record it; do not patch it (untestable here).

**F-CACHE (#218 part 2, V620).** Their Q8_1 cache was keyed by the **root tensor's identity** but
filled from the caller's (possibly offset) **view** pointer. An offset view stored its rows under the
root's key; a later op with the same root took them; after a compute-buffer reuse the pointer was
stale → `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on a V620. Fix: stop the key walk at the first
pointer change, and validate the source read extent against the root allocation before every launch.

Our O11′ prequant is safe against all three, and it is worth writing down *why*, because it is a
three-line argument someone will one day break:

1. **Producer is the writer.** The slot is claimed by the fused RMS_NORM+MUL that *writes* `mul`
   (not by an unrelated producer keyed on the same root), and eligibility refuses `view_src != nullptr`
   and non-contiguous output (`common.cuh:1592-1600`).
2. **Consumer matches on the exact tensor object and size**, not on a root identity:
   `e.src1 == src1 && e.nbytes == nbytes` (`common.cuh:1602-1612`). A view of `mul` is a different
   tensor pointer and misses.
3. **Storage is a context-owned arena**, not a ggml tensor and not the compute buffer
   (`common.cuh:1615+`, `cudaMalloc` 4 MiB), so `ggml-alloc` reuse cannot invalidate it; and the
   `stamp` is bumped once per graph execution, so a slot from a previous execution never matches.

Add these three to the O11′ comment block if that code is ever touched again.

### A.3 `edwinbrowwn/llama.cpp-rdna2` @ `e194cbddd`

**Chunked GDN prefill.** The mechanism is smaller than the name: for `n_tokens > 1`, lane 0 loads
`beta`, the column's `v`, and (non-KDA) `g`, then `__shfl_sync`-broadcasts them, and `expf(*g_t)` is
evaluated **once per warp instead of once per lane** (`gated_delta_net.cu:98-136` in their tree). That
last part is the win: the address is block-uniform and transcendentals run at a fraction of the FMA
rate, so 31 of 32 evaluations are redundant. Measured **+7.9 % @256 / +17.7 % @512 direct kernel**, 36/36 backend cases;
no end-to-end claim ("sensitive to process order and GPU temperature"). **[U]**

Mapping to our tree: **their patch does not apply**, because our kernel is a newer design —
`cols_per_warp = 4` with segmented reductions (`gated_delta_net.cu:7,41-50`), plus a GDN-MMA path. That
path is arch-gated away from us twice (`gated-delta-net-mma.cu:394`: HIP requires
`GGML_CUDA_CC_IS_RDNA3_5` **and** `n_tokens ≥ 2048`), so gfx1030 always runs the scalar kernel. The
portable content is one idea: **hoist the warp-uniform `expf` and the uniform `beta` load out of the
per-lane path**. In our structure `v_t[col]` is only uniform per 4-lane column segment, so the correct
translation is one `__shfl_sync` per segment (or `__builtin_amdgcn_readfirstlane` for `col`), not a
wholesale copy. Prefill-only by construction (`n_tokens == 1` keeps the stock path). This is N7, and
its gate is a *direct-kernel* A/B first — the delivery's own end-to-end number is unclaimed.

**Graph-scoped Q8_1 cache.** Analysis in §A.2 and D9: the contract is narrower than ours (single-token,
non-routed `MUL_MAT`, **Q8_0 weights**, standard `block_q8_1` layout, MMQ/ID/packed layouts excluded),
the storage is graph-owned with a per-execution readiness reset and an intervening-output overlap scan,
and the measured gain is **+0.62 %** with `quantize_q8_1` dropping 935 → 815 dispatches. Our O11′ is
already −249 launches / −0.353 ms and is weight-type-agnostic, because it is producer-side. **Port
nothing.** **[P]/[U]**

**Their negatives are the more useful half.**

| result | number | what it means for us |
|---|---|---|
| width-6 `nwarps=2` screen | **−24.1 %** target-dominant Q4_0, and it *changed raw F32 results* for Q4_0/Q4_K/Q6_K | independent support for **refusing widths 5–8** in N1. Our band stops at 4 for this reason as well as for `n_max ≥ 4` being refused. **[U]** |
| Q4_0 DOT8 (`v_dot8_u32_u4`) | exact, **−0.71 %** e2e | the ALU is not the bound; independent confirmation of the byte law. **[U]** |
| rows/block=4 at width 1 | **+2.9 % regression** e2e, reverted | matches our Q4_K result; width 1 stays 1 row. **[U]** |
| tiled FA with `fdot2` + native wave reductions | 2920/2920 `FLASH_ATTN_EXT`, **no end-to-end gain claimed** | do not build a second tile path for gfx1030; it is not where the money is. **[U]** |
| `sdot2` INT16 helper | capability only, no consumer | refuse: no I16 matmul consumer exists, and routing Q4/Q8 through it is mathematically wrong. **[X]** |
| wave-reduction builtins (`v_add_*_dpp` + `ds_swizzle`) | DPP order ≠ shuffle order | numerics-changing for a kernel that is already 4.4–4.5× over VEC; refuse. **[X]** |

### A.4 The other three sources — one line each, unchanged

- **`pkoscik/rdna2-local-opencode`** — 11 files, no kernels. Its only live content is host hygiene
  (`GPU_TARGETS=gfx1030`, no `HSA_OVERRIDE_GFX_VERSION`) and a wrong `--spec-draft-n-max 3` default.
  The `GGML_HIP_ROCWMMA_FATTN=OFF` flag is obsolete here (runtime `amd_wmma_available()`). **[P]**
- **ROCm llama.cpp docs** — install-level; gfx1030 is in the supported list (W6800, V620). Its
  "Flash Attention … through rocWMMA" sentence is **false for gfx1030** (no WMMA). No IC statement, no
  kernel guidance. **[P]**
- **gfx1030 wiki** — self-rated "Community"; its llama.cpp tables are `GGML_CUDA_DISABLE_GRAPHS=1`
  (which inflates our wall by ~44 %) and TP2/TP4. Direction only, never a number. **[X]**

### A.5 The three-way split the brief asks for

| portable engineering idea | hardware guidance only (do not code it) | gated / rejected |
|---|---|---|
| tile partial-batch branch (N5) | 2485/2660 MHz sclk ceiling: +7 % on **both** backends, no relative effect | occupancy recompute + its 128 KiB LDS constant |
| `__launch_bounds__` row for the verify shape (N6′) | 128 MiB IC @ 3.9× DRAM: which working sets are cheap, not a tier | Q4_0 DOT8 / `sdot2` / DPP reductions |
| native-KV attention for widths 2–8 (N3) | 64 KiB LDS per CU / 128 KiB per WGP: the reason a tile is 3 blocks, not 6 | arch-wide warp/rows tables; `nwarps=2` at widths 5–8 |
| NV21 geometry as a **band**, via a split table id (N1, done) | per-DKQ tile-table *values* tuned on a V620 — except the single cols=4 launch-bounds field, which is testable here (N6′) | `cols_per_block`/`ncols2` caps (inert for this model) |
| Q-from-global as an **LDS-budget** change (N9, deferred) | "pin clocks / pin DPM state / set `high`" | GDN sibling concat; six-row routed MoE |
| Q8_1 activation reuse **producer-side** (O11′, done) | `HSA_NO_SCRATCH_RECLAIM=1` (adopt only if scratch growth is observed) | edwin's consumer-side cache (needs Q8_0 weights) |
| device-tuple admission + typed refusal (N8) | wiki t/s tables (graphs-off, TP) | Rust executor, mesh TP, offload, speculative-without-exact-validation |

---

## 1. Infinity Cache — a hardware benefit, answered with our own numbers

The brief asks for this explicitly, so it gets its own section and its own arithmetic. **There is no
public repo, commit, API or knob that controls Infinity Cache, and we are not going to invent one.**

### 1.1 What we measured

| level | size | bandwidth | note |
|---|---|---|---|
| L0 (per WGP, vector) | 16 KB | — | not addressable from software |
| L1 / texture array | 128 KB | — | per CU |
| L2 (GL2C) | 4 MB | — | `hipGetDeviceProperties.l2CacheSize = 4 194 304` **[P]** |
| **Infinity Cache (LLC)** | **128 MiB** | **1986.6 GB/s** | **≈3.9× the 504.3 GB/s DRAM copy ceiling** **[P]** |
| DRAM | 16 GB GDDR6 | 504.3 GB/s copy / 513 GB/s best read | 100 % of nominal on `lm_head` **[P]** |

### 1.2 Why it cannot be "used" as a tier

- **The weight stream is 81–85× the LLC's capacity.** 11.351 GB of matvec weights per token against 128 MiB is
  **84.6×** from the census; the traced stream that produced the 81.4× figure is 10.4 GB. Either way
  the reuse distance is **one full token**, i.e. every weight is touched once per token and evicted
  long before it is touched again. No blocking, prefetch or hint can make a 10 GB stream fit in 128 MiB.
  **[P]**
- **What *does* fit is the KV cache, and only up to 4 681 tokens** for all 16 layers
  (134.2 MB ÷ 28.00 KiB/token). **[P]** (masterplan §5.7; unchanged.)
- **New: the per-layer threshold is where the interesting behaviour is.** Per token per layer the
  native cache costs 1 792 B (K q8_0 1 088 + V q5_0 704), so **one layer's native KV fits the LLC up to
  ~74 900 tokens**. If that layer's KV has been *staged to f16*, it costs 4 096 B/token and fits only to
  **~32 800 tokens**. **[H]** (arithmetic on measured sizes, not a counter).
  Consequence, and it is testable: the staged-verify path should degrade in two steps, with a knee near
  32k, while a native-KV verify path degrades once, near 75k. **M1 (§5) samples 4k / 32k / 86k for
  exactly this reason.**

### 1.3 Why the control surface is closed — by the toolchain, not by measurement

`__ldg` / `__ldcg` / `__ldcs` are **no-ops on HIP**; real `glc`/`slc` bits need inline assembly;
`PersistingL2CacheMaxSize` is 4 MB but `AccessPolicyMaxWindowSize = 0`, so the persisting window is
unusable; `rocprofv3` on ROCm 7.2.4 exposes **no IC counters**. **[P]** (masterplan §1, `ARCHITECTURE`).
None of the six sources in this brief implements or claims an IC control, and the two independent
gfx1030 kernel efforts both ran into the byte ceiling instead (DOT8 −0.71 %, rows>1 +2.9 % regression).
**[U]**

### 1.4 What "cache-friendly" therefore means in this codebase

Four rules, all of which are ordinary engineering and none of which mentions the cache:

1. **Move fewer bytes.** Quantisation is the only large unspent lever: 0.1 bpw = 0.32 GB = 0.70 ms =
   2.3 % of the token (§7.2 P1). The byte law is flat at 458 ± 4 % across four quant types **[P]**, so
   time ∝ bytes.
2. **Do not materialise an intermediate bigger than 128 MiB.** This is precisely the f16 KV staging
   critique: at 86k the staged buffer is **176 MB per layer for K alone (352 MB with V)** — 1.4–2.6×
   the LLC — so every byte of it is a DRAM round trip on top of the original. A native-KV read never creates that buffer.
3. **Do not re-read what you just wrote.** Fusion O1/O3/O11′ removed 393 dispatches and 1.29 ms by
   keeping data in registers; N1's `rows=2` shares one q8_1 activation across two verify columns;
   N3's LDS staging reads K/V from LDS once per tile, not once per query row.
4. **Keep the reused operand out of global.** Q-from-global (N9) is the *opposite* trade and is
   justified only because Q is 32 KB and cache-resident while K/V is 10 GB and streaming — i.e. it is
   still rule 2 and rule 3, applied to the smaller operand.

> **Infinity Cache is a property that decides which working sets are cheap. It is not an interface.
> The software responses are fewer bytes, fewer passes, and no oversized intermediates — which is what
> O1/O3/O11′ already do and what N3/P1 would do next.** **[P]/[U]**

---

## 2. The finding that changes the plan: the TILE path stages the whole KV per call

**Mechanism (code-verified).** `fattn.cu:848-860` sets `need_f16_K = need_f16_V = true` for
`BEST_FATTN_KERNEL_TILE`, unconditionally. `fattn-common.cuh:1031-1058` then converts
`ggml_nelements(K)` and `ggml_nelements(V)` elements to f16 into a scratch region sized *inside the op's
own allocation* (`ggml_cuda_flash_attn_ext_get_f16_extra_data`, `fattn-common.cuh:53-84`) — i.e. **per
call, per layer, not cached**. `n_kv` is the *used* context (`llama-kv-cache.cpp:1301-1318`), so the
cost scales with depth, not with `-c`. **[P]**

**Who pays it.** The selector sends quantized-KV attention to TILE for any `Q->ne[1] ≥ 3`
(`fattn.cu:828-830`). A `--spec-draft-n-max 2` verify pass is exactly 3 rows. So **every speculative
round stages the entire KV cache twice (K and V) × 16 full-attention layers.** The 48 GDN layers are
linear attention and unaffected. Plain AR decode never pays it (our GQA kernel reads q8_0/q5_0 natively
and is gated to `Q->ne[1] == 1`).

**The arithmetic** (measured sizes; 458 GB/s; 16 layers; 4 KV heads; D=256):

| context | K+V staged bytes/pass | staging time | staged-read time | native-read time | **N3 removes** |
|---:|---:|---:|---:|---:|---:|
| 4k | 386 MB | 0.84 ms | 0.59 ms | 0.26 ms | **≈1.2 ms/pass = 0.6 ms/token ≈ 3 %** **[H]** |
| 32k | 3.1 GB | 6.7 ms | 4.7 ms | 2.1 ms | **≈9.3 ms/pass ≈ 4.4 ms/token** **[H]** |
| 86k | 8.3 GB | 18.1 ms | 12.6 ms | 5.5 ms | **≈25 ms/pass ≈ 12 ms/token** **[H]** |

Two conclusions, both new:

1. **At d0, MTP's measured +58 % already pays a ~1.2 ms/pass tax** that nobody has attributed — 3 % of
   the ~41 ms pass, i.e. ~+1.5 t/s of the 50.9. Real but not the headline.
2. **At depth the tax becomes the headline, because the pass is bandwidth-bound and the staging is pure
   extra traffic.** First-order (non-KV pass cost held constant; AR at 86k ≈ 23–27 t/s between the
   measured 27.3 @32k and 22.16 @112k ⇒ ≈24):

   | 86k, MTP `n_max=2` | pass | t/s | vs AR (~24) |
   |---|---:|---:|---:|
   | today (TILE, staged f16) | 41.3 − 1.4 + 30 ≈ **70 ms** | **≈30** | +25 % |
   | with N3 (native KV) | 41.3 − 1.4 + 5.5 ≈ **45 ms** | **≈46** | +92 % |

   So MTP stays ahead of AR at depth, but **staging eats roughly two-thirds of its advantage**, and N3
   is worth **≈+53 %** on the MTP number at 86k against **≈+3 %** at d0. **[H]** — the discriminator is
   one measurement (M1), not an argument.

**Why nobody else has fixed it.** The delivery's `FATTN_KV_NATIVE_*` machinery — the natural place to
look — is **MMA-only**: *"Always NONE for the tile and vec kernels, which stage whichever type they were
compiled for."* gfx1030 has no WMMA, so the one device class that cannot use MMA is also the one that
cannot avoid the staging. **[U]** There is no upstream or fork fix to port; N3 is ours to write or
nobody's.

---

## 3. B — hipfire, delta only

The ledger is masterplan §4.1 and is unchanged; this is only what the delta adds. **[U]** throughout.

| PR | what it contributes | translate / defer / reject |
|---|---|---|
| **#915** decode layers as typed `Step`s | a Rust executor abstraction over `program.rs` | **reject** — ggml's `cgraph` + `ggml_backend_sched` own this. The *discipline* (one route per layer kind, no per-op branches) is already ours: one GQA kernel, one tile path, one MMVQ. |
| **#919** MTP/DFlash/VL-mrope on the step program | speculative plumbing in the executor | **reject**; the transferable rule is `#690`'s single admit predicate, already landed as P1/P2. |
| **#933** dense FFN on `GatedMlp`'s Product ladder | a Rust FFN fusion ladder | **reject code**; the FFN is 5.704 Gelem × 3 = 6.80 GB of the token's 11.35 GB, and the byte law says its only lever is bpw (P1). No C++ fusion can move a byte-limited stream. |
| **#934** dense prefill+decode TP on the mesh executor | multi-GPU split | **defer** — needs all-reduce; one card. The abstract idea (split only when VRAM forces it) is already our §7.3 context ladder. |
| **#936** `head_mixer_admission` / `gated_mlp_admission` | typed refusal per failure mode | **record only**; our equivalent is the L4 negative-case table (§10.5). |
| **#935** restores 3 memcpys into the GDN rollback tape | a *correctness* fix for partially-accepted drafts | **not applicable structurally** (§6.1), but it is the reason our exactness gate runs `-fa 0 -ctk f16 -ctv f16`. |
| **#897** gfx1100 VMM AR lanes | arch+dtype-specific | **reject** for gfx1030 (dtype cannot exist in a GGUF). |
| **#690** one MQ-V2 admit rule | the one portable rule | **landed** (P1/P2). |
| **#793 / #858** host spill / MoE spill | closed unmerged; re-implement `-ngl`/`-ot` | **reject as evidence**. |

**Judgement.** hipfire's contribution to this host is *design discipline and a negative result*, not
code: every Qwen 3.5 decode gate keys on gfx1100/gfx1151/gfx1201, its only gfx1030 admission is scalar
HFQ3 prefill, and its own default gfx1030 Qwen 3.5 model crashes (issue #478, open). Nothing in the
delta changes that. What has changed is that the *RDNA2* ecosystem (stew675, edwinbrowwn) does have
real gfx1030 work — and the most valuable things in it are a 12-line latent-fault fix, a one-line
`__launch_bounds__` row, and two documented failure modes, not the kernels.

---

## 4. C — port map to llama.cpp

Every row names the exact site, the mechanism, and its status. "Measurable here" is about *end-to-end*
effect, not correctness.

| capability | llama.cpp site | mechanism | status | tag |
|---|---|---|---|---|
| HIP backend capability check | `fattn.cu:561` `ggml_cuda_fattn_gqa_gfx1030_supported(cc,dst)`; `mmvq.cu:143` `get_device_table_id(cc)` | explicit cc + exact tensor-tuple predicate, env kill switch, VEC/TILE fallback | **landed (attention) / in flight (MMVQ)** | **[P]** |
| exact tuned route, per device | same two sites | currently admitted on the whole `GGML_CUDA_CC_IS_RDNA2` family (gfx1030–gfx1036) | **gap** → N8 (device tuple + INFO log) | **[P]** |
| GEMM/GEMV dispatch | `mmvq.cu:597` `calc_nwarps`, `:631` `calc_rows_per_block` | band-scoped geometry via a *split table id*, never an arch macro in a host+device constexpr | **landed (N1), gates running** | **[P]** |
| MATVEC activation reuse | `common.cuh:1585-1650` prequant; `ggml-cuda.cu:4678` producer; `mmvq.cu:1606` consumer | context-owned arena keyed by (tensor, nbytes, execution stamp) | **landed (O11′)** | **[P]** |
| attention/token-decode | `fattn.cu` GQA kernel (D=256, GQA 6, K q8_0, V q8_0/q5_0, `n_q==1`) | native quantised KV, no staging, 4.4–4.5× VEC at kv=32768 | **landed** | **[P]** |
| attention at width 2–8 | `fattn.cu:828-830` → TILE | stages whole K/V to f16 per call | **N3** | **[P]** mech / **[H]** gain |
| FFN/MLP | `mmvq` per block × 3; no fusion target | byte-limited; only bpw moves it | **P1** (quantisation A/B) | **[P]** |
| KV cache efficiency | `-ctk q8_0 -ctv q5_0`, 28.00 KiB/token | native types everywhere the GQA kernel runs | **landed for width 1; N3 extends it to 2–8** | **[P]** |
| memory management | `-ngl 99 -fitt 128`, §7.3 ladder | weights+KV resident; context reduced before offload | **landed as policy** | **[P]** |
| fallback / fail-closed | `fattn.cu:852` (switch default) and the final `return 1` in both `mmvq` geometry functions | generic path is the default; no route is taken on a tuple miss | **landed** | **[P]** |
| prefill attention | `fattn-tile.cuh:297` (256,256,32 row), `:407/:457` loads | tile shape, occupancy row, pointer-select | **N5 (loads), N6′/N9 (shape)** | **[U]** |
| GDN prefill | `gated_delta_net.cu:84/97` | warp-uniform `expf`/`beta` hoist | **N7** | **[U]/[H]** |

---

## 5. D — ranked, gated patch set

Order is EV per unit of work *given what is now known*. "Size" is the diff. Every accepted change
carries the standing bar: **bit-identical (or the documented substitute) plus a paired A/B above MDE
(0.30 ms / 0.30 t/s at d0)**, one `llama-bench` at a time, `llama-server` never active.

| id | item | size | tag | verification | measurable here? |
|---|---|---|---|---|---|
| **M1** | **MTP at 4k / 32k / 86k** (was N2), spec on, `-fa 1`, same prompt, VRAM peak per depth. Pre-registered: the pass grows by ≈+9 / +25 ms net from staging, so t/s at 86k lands in **≈28–32** (0.55–0.63× the d0 number) if the mechanism is right, and ≈45 with N3. | 0 code | **[P]** mech | paired, ≥5 pairs per depth; report α *and* tokens/pass; `trace --ledger` per depth | **yes** (1 h) |
| **M2** | **attribute the staging**: one `trace --ledger` per depth, count `to_fp16`/`convert_unary` rows per verify pass and their bytes; expect 2 per layer per pass, 32/pass. | 0 code | **[P]** | census must show the predicted 32 dispatches and ≈ the byte count of §2 | **yes** (20 min) |
| **N1** | verify-band MMVQ geometry (**in flight** — do not re-specify) | +81/−12 (done) | **[P]** | width 1 byte-identical (already green), 2–4 token-identical at `-fa 0 -ctk f16`, paired A/B at `n_max=2` | **yes** |
| **N3** | **one attention kernel for widths 1–8, native KV** — design A: extend the GQA kernel with an LDS-staged native K/V tile and a row loop (DRAM traffic 1× regardless of `n_q`); design B (native-KV tile kernel) refused, it duplicates the validation surface | ~250–400 lines | **[P]** mech, **[H]** gain | **row-0 byte identity**: with `n_q=3`, row 0 must equal the width-1 GQA kernel byte-for-byte; then `test-backend-ops -o FLASH_ATTN_EXT` full; then paired A/B at 4k and 32k | **yes** — the only item that scales with depth |
| **N5** | tile partial-batch branch (both sites) | 12 lines | **[U]** | `FLASH_ATTN_EXT` 4045/4045 unchanged + byte-diff of the L1 artifacts; no perf claim | no (correctness only) |
| **N6′** | the **(256,256,4) launch-bounds row only** (occ 6→1), reached only by widths 3–4 (ncols1=2, ncols2=2); the query-width-keyed split is the #97 pattern | ~6 lines (a narrow RDNA2 override, not the whole table) | **[U]/[H]** | **T1 first** (offline VGPR/spill/LDS of the new instantiation), then byte/token identity is *not* expected — gate is token-identical at `-fa 0 -ctk f16` + paired A/B at `n_max=2` | **yes** |
| **N4** | `ggml_cuda_mmq_scale_sumi` (3 sites) | 6 lines | **[U]** | `test-backend-ops -o MUL_MAT` byte-identical with the patch forced off; paired pp512 | ≤2 % (probably below MDE) |
| **N7** | GDN prefill: hoist the warp-uniform `expf`/`beta` into lane 0 + shuffle, re-derived for `cols_per_warp=4` | ~20 lines | **[U]/[H]** | direct-kernel A/B first (their own e2e claim is unqualified), then pp512 ≥3 % | yes but small |
| **N8** | **device-tuple admission + INFO route log**: admit the tuned routes on the tested tuple (`cc == gfx1030` **and** a known device name) and fall back on the rest of the family; elevate the "route selected / route declined" line from `GGML_LOG_DEBUG` to one-time INFO | ~15 lines | **[P]** | L4 negative table: a forced non-gfx1030 cc must log the decline and take VEC/TILE; no perf change here | no (policy) |
| **N9** | Q-from-global wide tile kernels (**deferred**) | ~120 lines + signature | **[U]/[H]** | trigger: N3 landed **and** pp512 attention still ≥15 % of prefill; then bit-identical Q bytes + paired pp512 | yes but prefill-only |
| — | prefill tile row (256,256,32: **K 128→64**, fa is already 64) | 1 line | **[U]/[H]** | fold into N6′ as a second experiment **only** if N9 lands first (they are coupled through LDS) | marginal |
| — | fused GDN chain (`PLAN_3MS` P8) | — | — | entry only if P5 < 2.5 ms total | — |

**T1 — new zero-GPU instrument (use it before every table/launch-bounds change).**
`llvm-objdump --offloading libggml-hip.so.0.N` writes per-TU code objects next to the library;
`llvm-readobj --notes` on the one containing `flash_attn_tile` prints
`.vgpr_count`, `.vgpr_spill_count`, `.group_segment_fixed_size` per instantiation. Measured on the
current build:

| kernel | VGPR | spills | LDS | consequence |
|---|---:|---:|---:|---|
| `flash_attn_tile<256,256,2,2,false>` (width-3 verify) | 159 | 0 | 19 200 B | occ 6 is satisfiable (122 112 of 131 072 regs); LDS bounds at 3 blocks |
| `flash_attn_tile<256,256,2,2,true>` (softcap) | 102 | 0 | — | the softcap instantiation is not register-bound either |
| `flash_attn_tile<256,256,16,2,false>` (prefill, cols 32) | 207 | 0 | ≈34 KB (arith.) | 2 blocks; the RDNA2 row (K 64) is ≈29 KB ⇒ still 2 blocks |
| `flash_attn_ext_gqa_gfx1030<Q8_0>` | **256** | **4** | 0 | at the wave32 VGPR ceiling — N3 must not simply multiply accumulators by `n_q` |
| `mul_mat_vec_q<iq3_s,1>` / `<iq3_s,3>` | 47 / 62 | 0 / 0 | 0 | N1's 2-warp band has wide occupancy headroom (33 blocks/WGP) |

Rule: **if a proposed table row or launch bound raises VGPRs without removing spills, or pushes LDS
past 64 KB/3 blocks, refuse it before spending GPU time.** Note the prefill row already violates its own
table (`occ 3` with 207 VGPR can only reach 2 blocks) — the hint is advisory, which is exactly why the
*measured* VGPR count, not the table, decides.

**Sequencing.** M1/M2 (≈1.5 h total, no code) → N5 (12 lines, any gap) → N3 (the only depth-scaling
item) → N6′/N4/N7 (independent, cheap, each with its own gate) → N8 (policy, no GPU) → N9 only if its
trigger fires. N1 finishes its existing gates first and is not to be re-opened.

**What I would not do next, and why.** Not another single-token decode kernel: 12 documented attempts,
byte law flat at 458 ± 4 % **[P]**. Not a second tile path for gfx1030: edwin measured one, no e2e gain
**[U]**. Not N6′ before T1: it is a codegen hypothesis and half the answer is free.

---

## 6. E — risks and unsupported items

### 6.1 Device assumptions
- **`GGML_CUDA_CC_IS_RDNA2` is gfx1030–gfx1036**, i.e. seven device ids, of which we have measured one
  (`common.cuh:82,90`) **[P]**. Navi22/23/24 differ in CU count and LDS/IC size; gfx1036 is an iGPU with
  no 128 MiB LLC. N8 exists to stop the tuned routes leaking onto them. Risk if unaddressed: a
  correctness-preserving but *mis-tuned* route on hardware we cannot test.
- **The card is at `power_dpm_force_performance_level = auto` and all sysfs clock files are read-only**
  **[P]**. Any guidance that says "pin clocks before benchmarking" is inapplicable; and H-CLOCK is
  closed, so relative comparisons are safe without pinning.
- `-dev` must be explicit in every backend comparison (dual-backend `build/bin`). Historical "Vulkan"
  numbers without `-dev Vulkan0` are suspect.

### 6.2 Kernel assumptions
- **No WMMA, no MFMA, no fp8** on gfx1030. The delivery's native-KV FA is MMA-only, so it is not a
  route here (§2). Any patch that assumes WMMA, MFMA, `V_DOT2_F32_BF16`, or a >64 KB LDS block must be
  refused.
- **`v_dot8_u32_u4` exists and does not help** (−0.71 % measured by a fork, exact) **[U]**. Do not
  reopen native-arithmetic substitutions.
- **The GQA kernel is at 256 VGPR with 4 spills** **[P]**. Width generalisation (N3) must reuse
  accumulators per row over an LDS-resident tile; a naive `n_q × accumulators` design will spill and
  lose.
- **HIP's device properties are internally inconsistent on this part**: `multiProcessorCount = 40` is
  per WGP, `regsPerMultiprocessor = 131 072` is the full WGP file, but
  `sharedMemPerMultiprocessor = 65 536` is per CU. Do not mix the three in one formula (this is the trap
  the ported occupancy recompute falls into).

### 6.3 Speculative risk
- **α at depth is unmeasured.** 50.9 t/s / α 0.661 is a d0 number. §2 predicts the verify pass grows by
  ~25 ms **net** at 86k from KV staging alone. M1 is the gate; until it runs, "MTP is the largest lever" is a
  d0 claim only.
- **`-fa 1` flips greedy tokens vs `-fa 0`** (measured) **[P]**. Any exactness claim runs
  `-fa 0 -ctk f16 -ctv f16`. N3's whole point is to make AR and verify share one kernel, which is the
  only way to get bit-identical speculative validation at `-fa 1`.
- **MTP's VRAM overhead is flat 0.96 GiB** (332 MiB weights + 449 MiB snapshots + 16 MiB MTP KV)
  against 28.00 KiB/token, hence the 86k ceiling. Do not spend it on `-ctv f16` (+21.5 KiB/token ≈
  +1.85 GB at 86k) to dodge the staging cost — that trades the context ladder for a kernel problem.

### 6.4 Memory-fit risk
- Fit is `weights 12.109 GB + KV 28.00 KiB/token + compute buffers ± 0.96 GiB spec`. `-fitt 128` is the
  conservative estimator; §7.3's ladder (112k spec-off / 86k @ n_max=2 / 80k @ n_max=3) is
  VRAM-derived, not measured beyond 4k — M1 closes part of that gap.
- Offload is closed by arithmetic (§7.6): PCIe at ~16 GB/s vs 458 GB/s is a 28× penalty on any byte
  that crosses. Do not offer it as a supported config.

### 6.5 Infinity Cache — the specific misunderstandings to avoid
- ✗ "Enable Infinity Cache" / "pin weights in IC" / "IC-aware allocation" — no such control exists; the
  stream is 81–85× the cache.
- ✗ "Bypass the cache with `__ldcs` for the weight stream" — no-op on HIP; `glc`/`slc` need inline asm
  and would not change a stream with zero reuse.
- ✗ "Prefetch the next layer's weights into IC" — the reuse distance is one full token; a prefetch of
  the next layer evicts the current one for no gain, and there is no prefetch primitive.
- ✗ "The IC explains the HIP/Vulkan gap" — the gap is 6 % with identical clocks; IC is shared by both
  backends and both stream the same 11.35 GB.
- ✓ "The KV working set fits the IC up to 4.7k tokens for all layers, or ~75k for one layer" — true, and
  the reason N3 (native KV) is worth more at depth than any cache hint.

### 6.6 Refuse list (explicit, this round)

**From this brief's sources — do not implement:**
1. The RDNA2 tile-FA **occupancy recompute** in `launch_fattn` (premise false here: 131 072 regs
   measured; its own LDS divisor contradicts its comment). **[X]**
2. The `cols_per_block` / `ncols2` **caps** (inert for D=256, ncols2=2; they exist for D=128/512/576 and
   MHA). **[X]** for us.
3. **`MMVQ_PARAMETERS_NAVI21` as an arch-wide table** — N1 is the band-scoped version and width 1 is
   refused; edwin's width-6 screen regressed 24.1 %. **[X]**
4. **Q4_0 DOT8 / any native-arithmetic substitution** (−0.71 %, exact). **[X]**
5. **rows/blocks > 1 at width 1** (−34 % isolated / 0.0 % e2e here; +2.9 % regression upstream). **[X]**
6. **`nwarps=2` at widths 5–8** (edwin: −24.1 %, and raw F32 changed). **[X]**
7. **GDN sibling-projection concat** (`[wqkv|z]`) — needs a uniform dtype we do not have. **[X]**
8. **Six-row routed MoE MMVQ** — no expert path in a dense 65-block model. **[X]**
9. **TP / mesh all-reduce / output-head sharding / backend sampling / P2P host snapshots / DFlash** —
   one card, CLI-only. **[X]**
10. **A native-KV *tile* kernel** (design B for N3) — duplicates the GQA kernel's purpose and doubles the
    validation surface; do design A. **[X]**
11. **`sdot2` INT16, native wave-reduction (DPP) substitutions in FA** — no consumer / numerics-changing
    for kernels that are not the bound. **[X]**
12. **edwin's Q8_1 cache** — requires Q8_0 weights, which this model does not have; ours is broader and
    already landed. **[X]**
13. **Q-from-global now** — deferred behind N3 (N9); prefill-only, signature change. **[H]** pending.

**From the brief's framing — refuse as engineering practice:**
14. **The Rust executor model**: `Step`/`OpKind`, `program.rs`, `DispatchCtx`, `GdnTape`, `RowTwins`.
    ggml's `cgraph` owns it. **[X]**
15. **Any hipfire gfx1100/gfx1151/gfx1201 Qwen 3.5 fusion** — arch-gated away; dead code here. **[X]**
16. **Infinity Cache as a software tier, cache hints, prefetch programmes, L2 persistence.** **[X]**
17. **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** — the device is native gfx1030. **[X]**
18. **`power_dpm_force_performance_level` writes** — read-only here, and H-CLOCK is closed. **[X]**
19. **Numbers copied from the gfx1030 wiki** (graphs-off, TP2/TP4, different quant). **[X]**
20. **Enabling MTP by default** in `common/common.h` — it is a host flag and costs 0.96 GiB. **[X]**
21. **Crediting the new `q8_0-q5_0` VEC instance with a verify-band win** — it is the width-2 path (§0.1). **[X]**
22. **`-ctv f16` as a workaround for the staging cost** — +1.85 GB at 86k, breaks the ladder. **[X]**

### 6.7 Judgement on the brief's Priority 1 ("exact tuned route, gated per device")

**Already satisfied for attention, in flight for MMVQ — with one literal gap.**
`ggml_cuda_fattn_gqa_gfx1030_supported()` is exactly the requested shape: explicit HIP/arch check,
exact tensor tuple (head_dim 256, GQA 6, Q F32, K Q8_0, V ∈ {Q8_0, Q5_0}, no sinks, mask/softcap/ALiBi
refused), env kill switch, generic VEC/TILE fallback, and a log line that distinguishes "gfx1030,
validated shape" from "ISA-compatible, shape unvalidated". N1 is the same pattern for MMVQ via a split
table id. **There is nothing to re-architect.**

The two things worth doing are 15 lines total: **(a)** N8 — admit on the tested *device tuple* rather
than the whole gfx103x family, and **(b)** raise the route log from DEBUG to one-time INFO so a silent
regression to VEC is visible in a normal run. Everything else the brief's Priority 1 asks for is present
and measured.

---

## 7. Open questions (delta only)

| # | question | artifact | cost |
|---|---|---|---|
| R1 | Does the verify pass really pay ~1.2 ms net at 4k and ~25 ms net at 86k for KV staging? | `trace --ledger` at 4k/32k/86k, spec on; count `to_fp16` rows and bytes | 30 min |
| R2 | Is MTP still a win at 32k and 86k? | paired A/B, 3 depths, α reported | 1 h |
| R3 | Does the width-3 tile kernel improve when its launch-bounds occupancy drops 6→1? | T1 (VGPR/spill) offline, then one paired A/B at `n_max=2` | 0.5 day |
| R4 | Can `n_q=3` row 0 be made byte-identical to the width-1 GQA kernel? | a row-0-vs-1-row byte test on the new kernel | 2 days |
| R5 | Is the (256,256,32) prefill row worth anything *without* Q-from-global? | trace pp512 attention component + one paired A/B | 0.5 day |
| R6 | Does the GDN uniform-`expf` hoist reproduce on our `cols_per_warp=4` kernel? | direct-kernel A/B, then pp512 | 0.5 day |
| R7 | Is the 4.7k / 75k / 33k residency arithmetic visible as a knee in the depth curve? | R1's bytes at 32k vs 86k | with R1 |

---

*Nothing here is committed. No GPU benchmark was run for this round; the codegen numbers came from
`llvm-readobj --notes` on the built `libggml-hip.so` and the byte-gate numbers from existing artifacts
under `/tmp/opencode/n1/`. All future measurements via `scripts/bench-exclusive.sh`, one `llama-bench` at
a time, `llama-server` never active.*
