# gfx1030 × Qwen 3.5 — masterplan (consolidated, evidence-tagged)

Status: uncommitted planning document. Supersedes nothing automatically — where it conflicts
with an earlier doc, the conflict is named and the resolution given.
Host: RX 6900 XT, gfx1030, ROCm 7.2.4 (`7.2.53211-3d9ef42`), HIP + Vulkan, one card.
Tree: `HEAD = d327bf1f0` + 27 modified / 23 untracked files (all perf work uncommitted).
Model of record: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` — qwen35 hybrid,
65 blocks (48 GDN + 16 full-attn + 1 MTP), 866 tensors, 12.109 GB, 3.546 bpw.
Upstream examined (all read locally, pins below):

| source | pin | what it is |
|---|---|---|
| `warpfront/hipfire` | `56367a75a` (2026-10-24) | Rust engine, Qwen 3.5 dense/MoE/TP, gfx11+/gfx12 only |
| `stew675/llama-cpp-rdna-boosts` | `974debf82` (2026-10-26), r148 | 15-patch C++ delivery, **contains a gfx1030 add-on** |
| `edwinbrowwn/llama.cpp-rdna2` | `e194cbddd` (2026-09-06) | the V620/gfx1030 fork the gfx1030 wiki documents |
| `pkoscik/rdna2-local-opencode` | `75f196059` (2026-10-01) | 11 files; build/config wrapper, no kernels |
| ROCm llama.cpp docs | ROCm 7.14.0 | install + device-selection guide |
| gfx1030 wiki (`blivioniag.github.io`) | fetched 2026-10-27 | community bench snapshots, self-rated "Community" |

**Tag legend — every claim carries one.**

| tag | meaning |
|---|---|
| **[P]** | proven on this machine, number and artifact in the repo |
| **[U]** | proven upstream — a fork's own code or its quoted measurement, cited to file/hunk |
| **[H]** | hypothesis / nearest analog: arithmetic support, not measured here |
| **[X]** | unsupported or refuted — do not claim, do not ship |

Prior documents this one consolidates: `ARCHITECTURE_GFX1030.md` (memory system),
`HIPFIRE_PORT_ROUND2.md` (PR verdicts), `HIPFIRE_QWEN35_PORT_BRIEF.md` (round-1 port map),
`PLAN_3MS_GFX1030.md` (landed-change log), `PLAN_GQA_DECODE_GFX1030.md` (attention kernel),
`MASTER_PLAN_GFX1030.md` (programme history), `PLAN_GFX1030_DECODE_V2.md` (matvec analysis).

---

## 0. Verdicts

1. **The four landed changes and MTP are the whole win so far.** 31.0 ms/token, 32.2 t/s, versus
   Vulkan 29.06 ms / 34.4 t/s — a 1.95 ms (−6.3 %) residual, down from −10 %. **[P]**
2. **MTP is worth more than every kernel change combined**: 50.9 t/s at `n_max=2` (+58 % over
   AR), and `n_max=2` beats the shipped default `n_max=3` by 8 % *and* costs 150 MiB less. **[P]**
3. **hipfire has nothing to port for gfx1030** — every Qwen 3.5 decode gate keys on gfx1100 /
   gfx1151 / gfx1201, its only gfx1030 admission is scalar HFQ3 prefill, and its own default
   gfx1030 Qwen 3.5 model crashes (issue #478, open). **[U]**
4. **Verdict 3 is a statement about hipfire, not about the RDNA2 ecosystem — and the ecosystem
   does have gfx1030 kernel work.** Two C++ forks carry real, gated, externally-validated gfx1030
   changes: a tile-FA/kv=113/carve-out set in `stew675` (from `sixvolts/llama-halo-hybrid`, PR #165,
   validated on a V620) and a per-shape MMVQ/Q8-cache/GDN set in `edwinbrowwn`. The previous
   version of this document generalised one repo's null result into an ecosystem claim. **That was
   an overreach and is corrected here.** **[U]**
5. **The "decode cost tracks elements, not bytes, so requantization is dead" claim does not survive
   the in-situ trace** (§2, C1). Per-token weight traffic is 11.351 GB and it moves at 458 GB/s;
   achieved GB/s is flat (±4 %) across 3.06–4.50 bpw, the signature of a **byte-limited** kernel.
   The element-rate law was measured in an Infinity-Cache-resident harness. **[P]** — and it
   re-opens quantization as the largest remaining lever (§7.2).
6. **The remaining token budget is: 22.13 ms irreducible bytes + 2.66 ms in-kernel block overhead
   + 4.15 ms non-matvec kernels + 2.11 ms dispatch gap = 31.05 ms.** **[P]**
7. **gfx1030 has no matrix cores and no fp8.** RDNA2 tops out at `v_dot4_i32_i8` (dp4a),
   `v_dot2_f32_f16`, and — a fact this version adds — `v_dot8_u32_u4` / `v_dot8_i32_i4`, the native
   4-bit dot. **The 4-bit dot was implemented and measured on gfx1030 by a public fork: exact, and
   0.7 % *slower* end-to-end** (§5.3). Prefill runs at ~35 % of the dp4a MAC peak. **[U]/[P]**
8. **Single-stream decode cannot reach Vulkan parity by kernel work.** 12 documented attempts to
   move the matvec's byte rate have failed. The levers with unspent headroom are dispatch count
   (2.11 ms), the non-matvec 4.15 ms, bytes (quantization), and tokens-per-byte (MTP/batching). **[P]**
9. **The card's real throughput is batching, and it is unused.** k=2 → 55.6 t/s/slot (111 agg),
   k=4 → 80.1 t/s/slot (320 agg) at 4k, versus 32.2 single-stream. **[P]**
10. **The single most under-examined region is the speculative verify band** — batch widths 2…8,
   which is *not* the decode case, was **not** covered by our seven geometry experiments, and has
   measured upstream gains on gfx1030 (§5.2, §2 C5). It is now priority-ordered above kernel work
   on single-token decode. **[U]** for the upstream numbers, **[H]** for the transfer.
11. **Deployment is a routing policy, not a runtime hybrid.** HIP is the default (MTP + prefill);
    Vulkan is a decode-only fallback worth **+6.8 %** at d0 and +4.8 % at 112k, at −19 % prefill. **[P]**
12. **No public project implements Infinity Cache control, and two independent gfx1030 kernel
    efforts confirm the byte ceiling instead.** Infinity Cache remains a hardware property that
    decides *which working sets are cheap* (KV ≤ 4.6k), not a software tier (§5.7). **[U]**

---

## 1. Established on this machine — do not re-derive

| fact | value | source |
|---|---|---|
| decode, current tree | 31.01–31.05 ms → **32.2 t/s** (`-ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128`) | `PLAN_3MS` §results |
| Vulkan reference | 29.06 ms → 34.4 t/s | `PLAN_3MS` |
| prefill pp512 | HIP **625 t/s** vs Vulkan 507 (**+23 %**) | `MASTER_PLAN` §10.5 |
| depth | 32k: 27.3 HIP / 29.2 VK; 112k: 22.16 / 23.23 | `PLAN_GQA` §9 |
| MTP | 44.4 / **50.9** / 46.9 t/s at n_max 1/2/3, α 0.817 / 0.661 / 0.562 | `PLAN_3MS` |
| MTP VRAM | 332 MiB weights + ~150 MiB per draft slot; **+0.96 GB** measured at −c 4096 | `PLAN_3MS` |
| context ceilings | 112k spec-off, **86k @ n_max=2**, 80k @ n_max=3 | measured |
| batching | k=2 17.98 ms/token → 55.6/slot, 111 agg; k=4 12.48 → 80.1/slot, 320 agg | `ARCHITECTURE` |
| dispatch census | 1,788/token; matvec 425 @ 24.79 ms; all other kernels 4.15 ms; gap 2.11 ms @ 1.19 µs | `PLAN_3MS` |
| DRAM | 504.3 GB/s copy ceiling; **513 GB/s** achieved read (lm_head, 100 % of nominal) | `ARCHITECTURE` |
| KV | 28.00 KiB/token, K row 272 B = 5 sectors; 2.99 GiB @ 112k | `gfx1030.py audit` |
| cache levers | closed **by the toolchain**: `__ldg/_ldcg/_ldcs` are no-ops on HIP; `glc/slc` need inline asm; persisting-L2 window = 0; no IC counters in rocprofv3 | `ARCHITECTURE` §"toolchain" |
| upstream PR ledger | 690/897/915/919/933/934/935/936 merged 2026-10-22; **793 and 858 closed unmerged** | GitHub API, re-verified |

### 1.1 Device properties, measured 2026-10-27 with `hipGetDeviceProperties` (new in this version)

A 40-line HIP probe (`/tmp/opencode/probe/props.cpp`, `-O2 --offload-arch=gfx1030`) on this exact
host, ROCm 7.2.4. These numbers settle three claims that were previously argued from documents.

| property | value | consequence |
|---|---|---|
| `multiProcessorCount` | **40** | HIP's "multiprocessor" on gfx1030 is the **WGP**, not the CU (80 CU / 2) |
| `maxThreadsPerMultiProcessor` | 2048 (= 64 waves/WGP) | 32 threads/wave, wave32 |
| `regsPerMultiprocessor` | **131 072** | = 4 SIMD32 × 1024 VGPR × 32 lanes: the *full* WGP register file |
| `regsPerBlock` | 131 072 | one block may claim the whole WGP file |
| `sharedMemPerMultiprocessor` | **65 536** | LDS budget as HIP reports it, **per WGP** |
| `MaxBlocksPerMultiProcessor` | 64 | — |
| `l2CacheSize` | 4 194 304 | 4 MB GL2C, as documented |
| `PersistingL2CacheMaxSize` | 4 194 304 | the whole L2 — but `AccessPolicyMaxWindowSize = 0` |
| `clockRate` / `memoryClockRate` | 2 660 000 kHz / 1 000 000 kHz | 2.66 GHz peak sclk; 16 Gbps GDDR6 on 256 bit |
| `totalGlobalMem` | 17 163 091 968 (15.98 GiB) | the card's usable budget |
| occupancy (256-thread dummy, 0 LDS) | **8 blocks** | no occupancy starvation at any sane VGPR count |

Current clock state, read-only: `power_dpm_force_performance_level = auto` (a reboot cleared an
earlier `manual`), sclk under load **2485 MHz of a 2660 MHz maximum**, package power 245 W.
The sclk table is dynamic (2 states idle, 3 under load) — an idle read does not describe a run.
**H-CLOCK is CLOSED by measurement: HIP and Vulkan both run at 2485 MHz, so DPM state is not a
lever and must not be re-tested. See "Closed hypotheses — DO NOT RETEST" at the end.**

### 1.2 Which gfx1030 fixes already exist here (grep-verified this round)

| upstream artefact | present in our tree? | evidence |
|---|---|---|
| `ggml_cuda_fattn_tile_get_config_amd_rdna2` (per-DKQ gfx1030 tile table) | **no** — RDNA2 falls into the generic `_amd_rdna` table | `fattn-tile.cuh:237,318,331` |
| RDNA2 occupancy recompute in `launch_fattn` | **no** — plain `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `fattn-common.cuh:1137` |
| kv=113 tail-check / 511-row guard tail | **no** — the pointer-select that faults is present verbatim | `fattn-tile.cuh:410`, `:460` |
| Q-from-global half2 for wide kernels | no | — |
| `MMVQ_PARAMETERS_NAVI21` geometry (rows/nwarps for verify widths) | **no** — RDNA2 returns 1/1 | `mmvq.cu:463-580`, `:607-630` |
| `ggml_cuda_mmq_scale_sumi` (float scale in K-quant MMQ) | **no** — integer multiply at three sites | `vecdotq.cuh:552,615,678` |
| `GGML_HIP_ROCWMMA_FATTN` build switch | **obsolete here** — the option no longer exists; WMMA FA is runtime-gated by `amd_wmma_available()` | `grep` across `ggml/` |
| Q8_1 producer/consumer reuse | **yes**, different design (arena + `prequant_find`) | `mmvq.cu` diff, O11′ |

---

## 2. Corrections to the standing record

### C1 — the byte/element law was measured in the wrong regime *(highest-consequence correction)*

Exact per-token weight census from the GGUF (`gguf-py`, main-graph 2-D tensors only, MTP block and
`token_embd` excluded): **25.624 Gelem, 11.351 GB, 3.544 bpw** — this reproduces `MASTER_PLAN`'s
11.353 GB to 0.02 %, so the byte figure was always right; the *rate* was not.

Joining that census to the per-type kernel times in the trace (`/tmp/opencode/tr_ab/`,
`hip_after_kernel_trace.csv`, 17 tokens) gives in-situ rates per type:

| type | bpw | GB/token | ms/token | **GB/s** | Gelem/s |
|---|---:|---:|---:|---:|---:|
| IQ3_S | 3.4375 | 3.643 | 8.120 | **448.7** | 1 044 |
| IQ4_XS | 4.2500 | 3.022 | 6.441 | **469.2** | 883 |
| IQ3_XXS | 3.0625 | 2.076 | 4.658 | **445.7** | 1 164 |
| Q4_K | 4.5000 | 1.583 | 3.316 | **477.4** | 849 |
| all matvec | — | 11.351 | 24.785 | **458.0** | 1 034 |

> **In situ, decode matvec time is proportional to bytes, and achieved byte rate is ~458 GB/s
> (89 % of the 513 GB/s this backend reaches on lm_head, 91 % of the copy ceiling).**
> Two-parameter fit: `t = bytes/513 GB/s + 2.66 ms`, residual 6.9–13.1 % per type.

`ARCHITECTURE_GFX1030.md`'s headline table (841–912 Gelem/s across a 39 % byte span) was taken
from `--test-file` shapes of 34–47 MB, which are **Infinity-Cache resident**. Consequence:
**"requantization cannot buy decode speed below 4.42 bpw" is not established.**
Under the corrected model, **0.1 bpw ≈ 0.32 GB ≈ 0.70 ms ≈ 2.3 % of the token.** **[P]**

*Counter-evidence, stated so it is not lost:* hipfire on gfx1030 at 3.25 bpw measured 28.70 t/s
against our 30.9 at 3.55 bpw (`HIPFIRE_27B_GFX1030_TEST.md`) — fewer bytes, slower. That is
cross-implementation, so it is consistent with the byte law plus a slower kernel. It does not
overturn C1; it is why C1 must be closed by a direct A/B. **New corroboration for C1 from an
independent fork:** `edwinbrowwn` implemented the gfx1030-native 4-bit dot product
(`v_dot8_u32_u4`/`v_dot8_i32_i4`) for Q4_0 decode — a genuine ISA-level win in principle — and
measured **−0.71 %** end-to-end, i.e. the arithmetic was not the bottleneck. **[U]** (§5.3)

### C2 — 420.6 GB/s was an arithmetic error

The published "matvec in situ 420.6 GB/s = 83 % of peak" divides **10.414 GB**, an assumed average
bpw. This file's matvec set is **3.544 bpw → 11.351 GB**, which gives **458 GB/s / 89 %**. Every
downstream sentence that said "17 % of bandwidth never requested" should read "11 % of the byte
rate, concentrated in a per-block overhead term". The conclusion is unchanged and *stronger*: the
headroom is 2.66 ms, and it sits in a specific place (§3.3).

### C3 — the MTP optimum is derivable, and the shipped default is not it

A draft step costs **MTP block (0.348 GB) + a full `lm_head` pass (0.715 GB) = 1.06 GB ≈ 10 % of a
trunk pass**, and buys α tokens. Measured: α(1)=0.817, α(2)=0.661, α(3)=0.562, and n_max=2 wins.
The naive throughput model `(1+n·α)/(1+0.1n)` predicts n=3 — so a second cost the weight model
misses exists: **each draft is its own serial pass and pays its own share of the 4.15 ms small-op
budget and the 2.11 ms gap.** Record: the optimum at α₁≈0.8 is n_max=2, and **do not change the
global default in `common/common.h:329`** — it is a per-host deployment flag. Corroboration from a
second, independent host: the gfx1030 wiki records "MTP n=3 often beats n=4 (27B Q8 TP4)". **[U]**

### C4 — thin corrections

- `gfx1030.py audit` reports "2 dispatch default arms". One is a **false positive**
  (`get_vec_dot_q_cuda` aborts: `mmvq.cu:76 GGML_ABORT`); the other is **benign**
  (`get_vdr_mmvq` default `return 1`). The dispatch layer is fail-closed for correctness. **[P]**
- Row-boundary sector waste is a **2.072 % upper bound and ~0 in practice**. Do not open an
  alignment programme. **[P]**
- KV per token is **28.00 KiB** (16 attn blocks × 4 KV heads); the 17th block is the MTP head. **[P]**
- **LDS attribution (new):** this document previously said "64 KB LDS per CU". HIP reports
  `sharedMemPerMultiprocessor = 65 536` with `multiProcessorCount = 40`, i.e. **64 KB per WGP as
  HIP counts it**. The GQA kernel's arithmetic — 20 992 B × 3 blocks = 61.4 KB ≤ 64 KB — is
  therefore *tighter* than the old wording implied (it consumes 94 % of the budget, not 47 %), and
  the 3-block launch bound is exactly at the limit. Conclusion unchanged; the margin is not. **[P]**

### C5 — the geometry null result covers one band, not all of them *(new)*

The seven geometry experiments that came back flat-to-worse were **decode** configurations
(`ncols_dst == 1`). Upstream `stew675` carries a gfx1030 table that changes the *verify* band
(`ncols_dst` 2…8) and reports **+6.3 % at widths 2–4 and +7.6 % at widths 5–8** on a V620 with
Qwen3.8-27B (§5.2). It also leaves `ncols_dst == 1` at one row per block. Independently,
`edwinbrowwn` swept rows/block 1/2/4/8 on gfx1030 and found **rows=4 regressed decode by 2.9 %**,
reverted it, and kept multi-row only for width-8 verify (§5.3). Three sources, one consistent
picture:

> **Multi-row / multi-warp blocks pay only when a batch of 2–8 tokens amortises the activation
> load. At width 1 they lose. Our null result is therefore correct for decode and was never a
> test of the verify band.** **[U]/[P]**

### C6 — "HIP's occupancy query undercounts the RDNA2 register file 4×" is false here *(new)*

`stew675`'s RDNA2 add-on recomputes occupancy by hand because *"HIP's occupancy calculator assumes
regsPerMultiprocessor = 32768 on RDNA2, a quarter of a WGP's real register file … it reports 0
blocks for kernels above 128 VGPRs"*. **Measured on this host, ROCm 7.2.4: `regsPerMultiprocessor
= 131072`, `regsPerBlock = 131072`, and a 256-thread kernel gets 8 blocks.** The premise does not
hold on our toolchain, so the *mechanism* cannot be ported. The symptom it was written to fix also
does not appear: our 256-thread tile kernels run, and MTP at widths 3–4 completes at 86k. **[P]**

> Rule this establishes: an upstream fix whose *stated cause* is a toolchain bug must be
> re-measured on our toolchain before the fix is admitted, even when the fix is small and gated.

### C7 — gfx1030 tensor-parallel is externally validated for correctness *(new, narrow)*

`HIPFIRE_PORT_ROUND2.md` §D.5 said RDNA2 TP "is unverified and unverifiable here". The second half
stands; the first is now too strong. `stew675`'s own resolution record (`archive/work/rdna2/`,
TODO #66 closed 2026-10-20) documents a **2× gfx1100 + 1× V620 `gfx1030` `-sm tensor` split** running
`pp2048 1549 t/s`, `tg256 30.1 t/s` at d0 and 28.9 t/s at 32k, with `-fa 1` and the internal
all-reduce path selected — i.e. the mixed-generation meta split works. It is still **irrelevant on
one card** and still unmeasurable here. **[U]**

---

## 3. gfx1030 compute profile

### 3.1 Topology and the resources that bind

| resource | value | binds what |
|---|---|---|
| CU / WGP / shader arrays | 80 CU = 40 WGP, 2 SE × 4 SA, **10 CU per array** | — |
| SIMD | **2 × SIMD32 per CU**, wave32 native (wave64 at half rate) | 160 SIMD32 units |
| VGPR file | 256 KB/CU ⇒ **1024 VGPR/lane**; HIP reports 131 072 regs/WGP | decode occupancy |
| LDS | **64 KB per WGP** (HIP `sharedMemPerMultiprocessor`) | the attention kernel, not matvec |
| L0 vector | 16 KB per WGP, shared by both CUs | the batching knee |
| L1 | 128 KB per shader array × 8 ≈ 1 MB | — |
| L2 (GL2C) | 4 MB (`l2CacheSize`) | — |
| Infinity Cache | 128 MB = 2 × 64 MB, **1 986 GB/s internal** | KV at ≤4.6k only |
| DRAM | 16 GB GDDR6, 512 GB/s nominal, **504.3 GB/s copy, 513 GB/s read** | decode, absolutely |
| FP32 FMA | **11.52 TMAC/s** = 23.04 TFLOPS @ 2.25 GHz | the ceiling that makes int8 mandatory |
| int8 dp4a | **46.1 TMAC/s** (`v_dot4_i32_i8`) | prefill reaches ~35 % of it |
| int4 dot8 | `v_dot8_u32_u4` / `v_dot8_i32_i4` — **present**, measured neutral-to-worse (§5.3) | nothing |
| matrix cores / fp8 | **none** — WMMA is gfx11+, FP8 KV is gfx12+ | nothing to import **[X]** |

Occupancy arithmetic that explains the seven null decode geometry experiments: the quantised
matvec kernels compile to **48 VGPR** (unfused) and **64 VGPR** (fused), i.e. 1024/48 = 21 or
1024/64 = 16 waves per SIMD32 — 32–42 resident blocks per WGP, 2 560–3 360 card-wide — with **zero
LDS** in use. Raising `nwarps` or rows-per-block cannot add memory parallelism *at width 1*:
1 × 2.0 GHz-class clock, each block is one wave, and the change only reroutes the reduction tree.
**Both RDNA2 forks reached the same conclusion by measurement** (§5.2, §5.3). This is now a
three-way-corroborated null rather than a single-host one — *for width 1 only* (C5). **[P]/[U]**

### 3.2 The two block geometries that matter

| | decode MMVQ | prefill MMQ |
|---|---|---|
| geometry | `nwarps=1`, `rows_per_block=1` — RDNA2 falls through both `calc_nwarps` and `calc_rows_per_block` to 1 (`mmvq.cu:463-580`, `:607-630`) | `CASE(type, nthreads=256, occupancy=2, I=128, J=8…64)` |
| block | 1 wave32 per output row, grid = n_rows (×2 for a fused pair) | 128 weight rows × J tokens, 8 warps, 2 blocks/CU |
| VGPR / LDS | 24–64 / 0 (Q4_0 stock measured 24, upstream) | MMQ SRAM layout, `MMQ_ITER_K` |
| regime | **byte-limited at 458 GB/s** | **unpack/issue-limited: 35 % of dp4a peak** |
| evidence | §2 C1 table | 625 t/s × 25.624 G params = **16.0 TMAC/s** |

Note what `rows_per_block = 1` costs: every block re-reads the entire q8_1 activation row. For
lm_head that is 248 320 × 5.76 KB ≈ 1.4 GB of L1/L2 traffic per token against 715 MB of weights —
a 2:1 amplification absorbed by cache. `rows_per_block = 2` was tried for decode and is flat-to-
worse in situ; upstream tried it too and measured **+2.9 % regression** (§5.3). Closed at width 1.
**Open at widths 2–8** (C5). **[P]/[U]**

### 3.3 Kernel families: which bound applies

| family | ms/token | share | bound | headroom |
|---|---:|---:|---|---|
| quantised matvec (MMVQ) | 24.79 | 79.8 % | DRAM bytes @ 458 GB/s | **2.66 ms** (the block-overhead term) |
| attention (`flash_attn_ext_gqa_gfx1030`) | 0.74 | 2.4 % | KV bytes ÷ IC/DRAM residency | 0 at ≤4k; grows at depth |
| norms (305 calls) | 0.96 | 3.1 % | **dispatch floor** (3.2 µs/call) | ~0.7 ms, only by absorption |
| SSM conv/GDN/fwht (160) | 0.78 | 2.5 % | dispatch floor | ~0.5 ms by fusion |
| quantize_q8_1 (128 after O11′) | 0.23 | 0.7 % | dispatch floor | mostly captured |
| activations / copy / rope / concat / get_rows / set_rows / elementwise / blit | 1.35 | 4.4 % | dispatch floor | the O5-class fusions |
| dispatch gap (1 788 launches) | 2.11 | 6.8 % | front-end + dependent-drain | **≤0.58 ms, only by deleting ≥480** |

**Safe targets:** dispatch count, bytes, tokens-per-byte, batching, GQA sharing, KV quantisation at
depth, MTP — **and, new, the verify-band kernel geometry.**
**Unsafe assumptions:** that Infinity Cache is a managed tier; that cache hints exist on this
toolchain; that gfx11 results transfer; that per-element cost is constant; that the 1.67 µs floor is
reducible; that TP or offload has a role on one card; **that "geometry is closed" applies outside
`ncols_dst == 1`.**

### 3.4 Why 1.19 µs per dispatch, and why 1.67 µs is the floor

Measured: 1 788 dispatches/token, wall − Σkernel = **2.125 ms → 1.19 µs per dispatch** with
hipGraphs **on**. An isolated bare-kernel launch costs **1.67 µs** round trip. These are different
quantities: 1.67 µs is host-submit + GPU start + GPU done for one launch with nothing else in
flight; 1.19 µs is the *marginal* cost of one node inside a captured graph.

**1 788 × 1.67 µs = 2.99 ms = 9.6 % of the token.** The 1 299 small ops average 2.46 µs/call —
roughly two-thirds floor, one-third work. That is why the plan's unit of account is *dispatches
deleted*, not *kernels made faster*. Mechanism **[H]**: each node boundary is a full drain, so what
is measured is packet front-end + wave-launch ramp + completion fence. The decisive experiment is
`PLAN_3MS` Phase 0b. **Do not re-propose it; run it if Phase 2 resumes.**

### 3.5 Favourable shapes and tiles, with the evidence for each

| shape rule | evidence | status |
|---|---|---|
| **Batch k ≤ 8 is free at the kernel level** (`MMVQ_MAX_BATCH_SIZE=8`; weight address invariant in the token loop) | lm_head n=2: 1 396.28 vs 1 393.61 µs (**+0.19 % for 2× the tokens**) | **[P]** |
| **k=4 is the practical knee for K=5120**, k=8 for K=17408 | a(2)=1.0019, a(4)=1.201; cause = activation working set 23 KB vs 46 KB against a 16 KB L0 | **[P]** |
| **Verify widths 2–4 want 2 waves + 2 rows/block; 5–8 want 1 wave + 4 rows/block** | V620 gfx1030, Qwen3.8-27B: +6.3 % / +7.6 % | **[U]** — §5.2 |
| **Width 8 wants 2 rows/block on a *named shape list*, not on the arch** | edwin `w8_rows2` whitelist; eight-wave Q4_0 variant *failed raw-F32 exactness* | **[U]** — §5.3 |
| **Attention: one workgroup per (KV head, 256-key tile) serving all 6 Q heads** | `GQA GROUP 6 / TILE 256 / CHUNK 24 / WARPS 6`, `launch_bounds(192,3)`; 19.7 → 27.3 t/s at 32k, 9.9 → 22.2 at 112k; slope 0.103 µs/KV-token vs Vulkan 0.111 | **[P]** |
| **20 992 B LDS × 3 = 61.4 KB of a 64 KB WGP budget** | kernel LDS from trace; `launch_bounds(192,3)` | **[P]** |
| **Prefill wants J ≤ 64 tokens per 128-row tile**, unpack-bound not FMA-bound | MMQ RDNA2 config `I=128, J=8…64`; 35 % of dp4a peak | **[P]** |
| per-shape MMQ tile selection | issue #300 (+35…149 % pp32/pp128 on gfx1031) | **[H]** — their shape |
| **Do not** force GQA 6 through the `ncols2` template; a non-power-of-2 needs its own kernel | `PLAN_GQA` §2 | **[P]** |
| **Do not** use the f16 TILE path for q8/q5 KV — it sets `need_f16_K/V` and doubles KV | `PLAN_GQA` §6 | **[P]** |

---

## 4. hipfire's Qwen 3.5 work: what it is, and the gfx1030 answer

### 4.1 Verified ledger (re-checked)

| PR | state | size | what it actually is | for us |
|---|---|---|---|---|
| #915 | merged 10-22 | +2 190/−1 969 | decode layers as typed `Step`s | **reject** — ggml builds a cgraph |
| #919 | merged 10-22 | +18 461/−18 843 | MTP/DFlash/VL-mrope on the step program | **reject** — we run MTP; it is a measurement |
| #933 | merged 10-22 | +18 825/−14 849 | dense FFN on `GatedMlp`'s Product ladder | **reject code**, adopt gate-table discipline |
| #934 | merged 10-22 | +20 428/−14 855 | dense prefill+decode TP on the mesh executor | **reject** — needs all-reduce; one card |
| #935 | merged 10-22 | **+32/−0** | restores 3 `memcpy` into the GDN rollback tape | **structurally impossible here** (§6.1) |
| #936 | merged 10-22 | +21 229/−15 088 | `head_mixer_admission` / `gated_mlp_admission` split rules | **record the rule**, implement nothing |
| #897 | merged 10-22 | +821/−169 | gfx1100 exact-VMM AR lanes + double-rotate marker warning | **gfx1030 excluded by dtype** |
| #690 | merged 09-13 | +391/−125 | one MQ-V2 admit rule; llama path refuses V2 on every arch | **the one real portable rule** (P1/P2) |
| #793 | **closed unmerged** | +14 904 | dense host-RAM spill | **refuse** — re-implements `-ngl`/`-ot` |
| #858 | **closed unmerged** | +32 418 | MoE expert spill, MTP over a spill | **refuse** — same, plus no MoE path |

The five merged dense PRs are **one refactor in five slices**; the performance claim attached to
each is *parity*. There is no new math to port. **This section remains correct as a statement about
hipfire.** It is not correct as a statement about RDNA2 in general — see §5.

### 4.2 The four facts that settle hipfire's gfx1030 story

1. **gfx1030 is excluded from every Qwen 3.5 decode gate** — `hybrid.rs` admits only gfx1100 /
   gfx1151 / gfx1201. **[U]**
2. **The one positive gfx1030 admission is scalar HFQ3 prefill** (`mq3_uniform_with_gfx10_scalar`),
   the same class of route llama.cpp already has. **[U]**
3. **`#897`'s gfx1030 exclusion is by dtype, not policy**: the route requires uniform `MQ4G256V2`,
   which `#690` pins to gfx1100+ and which cannot exist in a GGUF. **[U]**
4. **Upstream's gfx1030 Qwen 3.5 support is broken on its own default model** — issue #478
   (`GemvMq6G256Prerotated`), issue #533 (MQ2 prefill 22× behind llama.cpp). **[U]**

### 4.3 Portable / adapt / reject

| upstream idea | llama.cpp location | verdict |
|---|---|---|
| one admit rule, N hand-copies, drift = silent corruption (#690) | the 5 `n_rs_seq > 0 ? n_rs_seq + 1 : 0` sites | **ported** (P1/P2) |
| predicate co-located with the arms it dispatches (#690) | `ggml_cuda_should_use_mmq`, `fattn.cu` selectors | **landed discipline**; one twin remains (§8.3) |
| typed refusal per failure mode (#936) | `ggml_backend_meta_split_state` | record only |
| `heads % parts == 0`; `parts == 1 ⇒ no reduction` (#936) | `llama-graph.cpp:3978` already has it | **same idea present** |
| fusion-boundary double-rotation marker (#897) | `scripts/test-graph-census.cpp`, `gfx1030.py rotcheck` | **adapted as a guard** (landed) |
| loud load-time refusal for unvalidated dtypes | loader diagnostics | adapt the *message style* |
| `Step`/`OpKind`/`program.rs`/`DispatchCtx`/`GdnTape`/`RowTwins` | — | **reject** (Rust executor internals) |
| MQ/MQ-V2/HFQ containers, every `mq*` kernel | — | **reject** (cannot appear in a GGUF) |
| mesh TP, `Accumulate`→`AllReduceAdd` | `ggml-backend-meta.cpp`, `ggml-cuda.cu:1144-1240` | **reject for this host**; correctness validated elsewhere (C7) **[X]** |
| #793/#858 offload | `-ngl`, `--n-cpu-moe`, `-ot` | **reject** (§7.6) |

---

## 5. The RDNA2 ecosystem — what the four new sources actually contain

This is the section the previous version lacked. Each source is graded by *what was read*, not by
its README's claims. Two of the four are not kernel work at all, and saying so is the finding.

### 5.1 Source inventory

| source | size / shape | contains kernel changes? | gfx1030 claims | grade |
|---|---|---|---|---|
| `stew675/llama-cpp-rdna-boosts` | 15-patch C++ delivery, 460 KB patch, 1.5 MB worklog | **yes** — tile FA, MMVQ/MMQ, GDN, MoE, TP | **external V620 validation**, r76 add-on from `sixvolts/llama-halo-hybrid` PR #165 | **[U]** for code existence; **[U-level]** for numbers, no perf gate claimed by its own maintainer |
| `edwinbrowwn/llama.cpp-rdna2` | full C++ fork, `docs/gfx1030-*`, per-feature env switches | **yes** — MMVQ specialisations, tiled FA, GDN prefill, Q8 cache | four V620s, each path A/B'd with exactness gates | **[U]** — best-documented gfx1030 work found |
| `pkoscik/rdna2-local-opencode` | 11 files, 176 KB, git submodule | **no** | RX 6800 XT, build flags + config preset | **[U]** that it is a wrapper; no findings of its own |
| ROCm llama.cpp docs | install guide, ROCm 7.14.0 | no | gfx1030 in the supported list (W6800, V620) | **[U]** install-level only |
| gfx1030 wiki | community bench notes | no | points at edwinbrowwn, hosts numbers | **[U]** but self-rated *Community* (its own verification page says so) |

Nothing in any of them implements, exposes, or claims an Infinity Cache control (§5.7).

### 5.2 `stew675` — the RDNA2 add-on, claim by claim

The add-on is five patches folded into blocks 05/07 (`archive/work/rdna2/pr165/patches-rdna2/`),
attributed to `sixvolts/llama-halo-hybrid`. Each is small and each is gated `#if defined(RDNA2)` or
by an RDNA2 device-table lookup, so gfx11/gfx12 codegen is unchanged — a discipline we should copy.
Verified against our tree line by line:

| # | change | claim | our verdict |
|---|---|---|---|
| 00 | `ggml_cuda_fattn_tile_get_config_amd_rdna2`: per-(DKQ,DV,ncols) table; `launch_fattn` recomputes occupancy; caps `cols_per_block`/`ncols2` | "occupancy query undercounts 4×"; D=256 ncols 8/16/32 re-tuned: **nbatch_fa 64 vs 32, +31 % prefill attention, −17 %/−30 % on 8- and 4-token verify at 64k** | **occupancy premise false here (C6).** For our widths the D=256 **cols=4 row has the same tile shape as ours** (128 threads, `nbatch_fa` 32, `nbatch_K` 256) and differs only in the `occupancy` launch-bounds field (1 vs 6 — a VGPR-budget hint, satisfiable either way at 131 072 regs) ⇒ **no expected effect on MTP n_max=2**. cols=8 does change the shape but n_max≈7 is past our optimum. The real content is **prefill attention** (cols 16/32), a small share of pp512. **[U], low value, numerics-changing** |
| 01 | partial-batch (unaligned KV) tile loads branch instead of pointer-select to a local zero array | "gfx1030 has no architected flat scratch … can fault at a near-null address for unaligned KV lengths (kv=113)" | **Real bug class; not reproduced here.** Our tree still has the pointer-select (`fattn-tile.cuh:410,460`) and our `test-backend-ops` passes the exact upstream case (D=256, GQA 2, n_q={16,1}, **kv=113**, Q8_0/Q8_0) in 4045/4045. Latent, compiler-dependent (ROCm 7.14 + V620 upstream vs 7.2.4 here). **[U]/[P]** |
| 02 | wide tile kernels read Q from a global half2 buffer instead of LDS | "+30 % on the 512-column kernel"; LDS is saturated by K reads; RDNA2 has 64 KB LDS/WGP | plausible and cheap; **prefill-only** (needs `ncols1*ncols2 ≥ 16`). **[U]/[H]** |
| 03 | `ggml_cuda_mmq_scale_sumi` — float instead of `v_mul_lo_u32` in K-quant MMQ; `MMVQ_PARAMETERS_NAVI21` table | integer mul is quarter-rate on RDNA2; Navi21 table: width 1 → 4 waves, 2–4 → 2 waves + 2 rows, 5–8 → 1 wave + 4 rows | **both portable and interesting.** The float scale is bit-identical by round-to-nearest (§10.2). The Navi21 table is the only public evidence that the *verify band* has headroom on gfx1030 (**+6.3 %/+7.6 %**), and its width-1 branch is unmeasured upstream and contradicted by our data — **admit by band, not by arch** |
| 04 | build guards so a `-DGPU_TARGETS=gfx1030`-only build compiles | WMMA builtins unavailable on gfx10 | matches what pkoscik's `GGML_HIP_ROCWMMA_FATTN=OFF` hack was for; our tree needs neither (runtime `amd_wmma_available()`) **[P]** |

Also recorded, because it is the one place this fork's own maintainer sets a boundary:
**"RDNA2 is best-effort, not a supported target; no RDNA2 device in the maintainer's fleet; an
RDNA2 result is informational, never a delivery gate."** The only gfx1030 validation is *external*
(a V620, `FLASH_ATTN_EXT` 6389/6389, `MUL_MAT` 2444/2444, coherent `-fa 1` generation, and the
3-card mixed split of C7). **No gfx1030 performance number in that repo is gated.** **[U]**

### 5.3 `edwinbrowwn/llama.cpp-rdna2` — the fork the wiki documents

The most useful source found, because it is *gfx1030-native, per-feature gated, and honest about
negatives*. Read from source (`ggml/src/ggml-cuda/mmvq-rdna2-config.h`,
`gated_delta_net.cu`, `docs/gfx1030-native-optimizations.md`, `docs/rdna2-v620-q4-0-dot8-experiment.md`).

| path | mechanism | measured | verdict for us |
|---|---|---|---|
| **Q4_0 DOT8 MMVQ** | `v_dot8_u32_u4` + `v_dot8_i32_i4` on Q4_0, two accumulators, 2-byte `sum_hi` sidecar per q8_1 block; stock weights unchanged | **−0.71 %** (92.27 → 91.61 t/s TG512, 4×V620, 5 reps); exhaustive digit checks 34 096/34 096; Q4_0 0/2048 mismatches | **reject** — exact but slower. Independent confirmation of C1: the ALU is not the bound. **[U]** |
| **rows/block sweep 1/2/4/8** | isolated kernel sometimes faster at 4 rows; activation load reused | **+2.9 % regression** end-to-end at rows=4, reverted; "the fast rows-four isolated-kernel result was not representative" | **reject at width 1**; matches our null result and our Q4_K precedent (−34 % isolated, 0.0 % e2e) **[U]** |
| **exact-shape gates** | `w8_rows2` = a *whitelist of (type, N, K) tuples* at width 8; eight-wave Q4_0 variant **failed raw-F32 exactness** and was dropped; `6656×128` +17.4 % but `6656×2048` −6.2 % ⇒ "the selector is exact rather than a general RDNA2 warp table" | +1.54 % (rows4), part of +6.97 % with backend sampling | **adopt the pattern** (this is our §8 gate discipline, evidenced) **[U]** |
| **chunked GDN prefill** | for `n_tokens > 1`, lane 0 loads scalar `beta`/gate and broadcasts across the wave; decode keeps the stock template | **+7.9 % @256 tokens, +17.7 % @512**, direct kernel; 36/36 backend cases; e2e "sensitive to process order and temperature", no claim | **candidate, prefill-only** — decode untouched by construction; at `n_tokens > 1` the scalar source changes but not the order, so it *should* be byte-identical — **verify, do not assume** **[U]** |
| **graph-scoped Q8_1 cache + dual RMSNorm producer** | cache the exact q8_1 bytes per (source tensor, graph, stream) with an intervening-output overlap scan; fused RMSNorm writes F32 **and** q8_1 | 935 → 815 `quantize_q8_1` dispatches/token (35B MoE graph), **+0.62 %** | **we already have the better half** (O11′: −249 launchers, −0.353 ms). Their *cache* half is what we lack when 2+ consumers share one activation — our O11′ covers 1 producer/1 consumer. Note only. **[U]/[P]** |
| **GDN sibling projection fusion** | load-time row-concat of `[wqkv \| z]` and `[beta \| alpha]` into persistent weights, originals exposed as strided views; 4 matmuls → 2 | +4.39 % tgen (PR #10), Qwen35MoE 35B layer-split only | **rejected for us**: requires **uniform dtype** across the pair. Our GDN block is IQ3_S qkv / Q3-ish gate / bf16 alpha-beta — concatenation is impossible without requantising to a common type. **[X]** (§11) |
| TP4 output-head sharding, backend sampling, P2P host snapshots, DFlash | multi-GPU / server / speculative-drafter specific | +4.6…+15.5 % on TP2/TP4 | **not applicable** on one card, CLI-only **[X]** |

**What this source settles.** (i) The rows>1 / warp-count question is answered per *band*, and the
answer at width 1 is no; (ii) a native 4-bit dot on gfx1030 does not move decode, which is the
strongest independent support for the byte law; (iii) wide-block selection must be a shape list, not
an arch table; (iv) a *second* independent fork, on a different model, reproduces our Q4_K
"isolated win does not transfer" result.

### 5.4 `pkoscik/rdna2-local-opencode` — a config wrapper, and that is the finding

Eleven files (three config presets, three shell scripts, README/licence/dotfiles) plus a submodule
pointing at `TheTom/llama-cpp-turboquant` (`feature/turboquant-kv-cache`). **No kernel changes, no
patches, no benchmarks of its own.** Its value is three host-level knobs, one wrong default and one
confirmation:

| item | content | verdict |
|---|---|---|
| build flags | `-DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=OFF` | **partly obsolete here**: the ROCWMMA switch no longer exists in our tree and WMMA FA is runtime-gated; `GPU_TARGETS=gfx1030` is a build-time/bin-size choice with no runtime effect. Keep as build hygiene. **[P]** |
| iGPU hiding | `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0` because ROCm enumerates the Ryzen `gfx1036` | **not needed on this host** — `rocminfo` lists exactly one GPU agent (gfx1030) and the CPU is Intel. Recorded because a second AMD device would silently steal device 0. **[P]** |
| power pinning | writes `power_dpm_force_performance_level=high` and `power/control=on` for every `card*/device` | **do not adopt** — sysfs is read-only here, and H-CLOCK is closed (both backends measured 2485 MHz), so pinning cannot affect a relative comparison. **[X]** |
| dense preset | 27B IQ4_XS, 32k, **q8_0/q8_0** KV, `-fa 1 -ngl 99`, `--spec-type draft-mtp --spec-draft-n-max 3` | **the n_max=3 default is measurably wrong here** (C3: −8 % and +150 MiB vs 2). A configuration copied from a 16 GB 6800 XT with a different quant. **[P]** |
| MoE preset | 35B-A3B Q6_K, 131k, f16 KV, `--n-cpu-moe 28`, MTP 3 | confirms the only viable big-model shape on 16 GB is expert offload — which our dense model does not need (§7.6) **[U]** |

### 5.5 ROCm documentation — install-level, with one wrong statement for this arch

The AI-ecosystem page is an install/verify guide. Two facts are usable and one is wrong:

- **Usable:** gfx1030 (W6800, V620) is in ROCm 7.14's supported Radeon list; `HIP_VISIBLE_DEVICES`
  is the documented way to pin a device when an iGPU is enumerated; `libgomp1`/`libcurl4` are the
  two package prerequisites that silently degrade to CPU when missing.
- **Wrong for us:** *"Flash Attention is enabled with `-fa 1` and is supported on Radeon GPUs
  through rocWMMA"*. gfx1030 has **no WMMA**; `-fa 1` here runs the tile/VEC/GQA paths. This is
  precisely why a third-party repo pins `GGML_HIP_ROCWMMA_FATTN=OFF` (§5.4). Do not use AMD's
  sentence as evidence that a WMMA path exists on this card. **[P]**
- **Absent:** no gfx1030 kernel guidance, no RDNA2 tuning, no Infinity Cache statement. The doc has
  no opinion on anything in §3.

### 5.6 gfx1030 wiki — numbers that do not transfer, and why

The wiki is a community index, and **its own verification page rates every llama.cpp number
"Community"** ("author-reported; run your own before/after"). The important details are in the
measurement conditions:

| claim | condition | why it does not transfer |
|---|---|---|
| "stock vs fork **+56.8 %**, byte-identical outputs" | Qwen3.8-27B **Q6_K**, MTP, **TP2** (2× V620) | two devices; TP is our hardest reject (§11). Not a single-card result. **[X]** |
| "Q4_0 / Q8_0 are the fastest quants" | fork's native MMVQ paths | on our card a uniform Q4_0 build is 14.41 GB/token ⇒ **matvec 30.8 ms vs today's 24.8**, token 37.0 vs 31.0 — 27 % more bytes through the same byte-limited kernel. Only true if a native path lifts byte rate past 600 GB/s, which the byte law forbids. **[X]** |
| "long-context sweep: Q8_0 > Q4_K ≈ Q6_K > bf16" | one host, tensor split, MTP | 4 GPUs, TP; and "tg" numbers move 16.7 → 24.4 t/s between context tiers, i.e. dominated by acceptance, not kernels. **[X]** |
| absolute t/s tables | **`GGML_CUDA_DISABLE_GRAPHS=1`** in the env prefix | graphs-off inflates wall by **~44 % on our host**; these numbers are not comparable with ours in either direction. **[P]** for the 44 %, **[X]** for the tables |
| "MTP n=3 often beats n=4" | 27B Q8, TP4 | direction matches C3 (optimum below the default) but the value does not transfer. **[U]** direction only |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` in every env stack | V620 | **must not be set here**: our device reports `gfx1030` natively and the override only risks selecting foreign code objects. **[X]** |
| `HSA_NO_SCRATCH_RECLAIM=1` | same stacks | a ROCr scratch-reclaim workaround, not a perf knob. Adopt only if we observe scratch growth in a long run. **[H]** |

### 5.7 Infinity Cache, answered from all sources at once

No source — hipfire, stew675, edwinbrowwn, pkoscik, ROCm docs, or the wiki — implements, exposes or
claims an Infinity Cache control. The closest any of them comes is *indirect*: stew675's tile table
is tuned per arch because the register file and LDS differ; edwin's Q8 cache reuses activations
inside a graph; neither addresses the 128 MB LLC. Meanwhile both gfx1030 kernel efforts ran into the
same wall — a native 4-bit dot product did not help (−0.71 %), and rows/blocks changes did not help
at width 1 (+2.9 % regression) — which is what a byte-limited stream looks like from two
independent directions.

> **Infinity Cache decides which working sets are cheap (KV ≤ 4.6k, activation rows); it does not
> create a software tier. The public evidence now supports this from four implementations, not one
> argument.** **[U]/[P]**

---

## 6. MoE and speculative decode

### 6.1 #935 is real upstream and structurally impossible here

Upstream's batched MoE DeltaNet body stopped writing the rollback tape, so a partially-accepted
draft replayed the **previous window's** q/k/v/α/β: τ 7.47 → 2.5, replies truncated at 15 tokens;
their changelog says *"Dense models were not affected."*

In llama.cpp the same state is written by **graph nodes emitted by one shared builder**
(`build_conv_state` at `delta-net-base.cpp:493/517`, `build_rs` at `llama-graph.cpp:3913-3925`),
and dense and MoE call the *same* builders (`qwen35.cpp:405/407` vs `qwen35moe.cpp:407/409`).
**Immunity applies to missing writes, not to wrong offsets** — the offset arithmetic at
`delta-net-base.cpp:502` is correct only because `split_equal(n_keep_tail = n_rs_seq + 1)`
guarantees the trailing K tokens share a ubatch. That is why P1/P2 were the right ports of #690.

### 6.2 MoE: zero portable content for our model — but one transferable pattern

No merged MoE-execution change exists in the hipfire PR set; #858 is closed unmerged; our model is
dense. `edwinbrowwn`'s **bounded six-row routed MMVQ** is the only public gfx1030 MoE work found:
extend MMVQ from 5 to 6 routed token rows when type ∈ {Q4_K, Q6_K}, rows ≤ 6 and top-k ≤ 4, with
**−15.9…−61.2 % operation latency across 48 cases**, and *stock dispatch retained* where top-k 6 or
concentrated top-k 8 regressed (by up to 54.5 %). It is a **band-admission pattern**, not a kernel:
prove the bound, enumerate the shapes, refuse everything else. Our dense model cannot use it; our
awareness of the verify band (§5.2) did come from reading this class of work. **Record; implement
nothing for MoE.** **[U]**

### 6.3 Speculative decode: measured, and it is the largest single lever

| mode | t/s | ms/token | α | RS buffer |
|---|---:|---:|---:|---:|
| plain AR | 30.57 | 32.72 | — | 149.6 MiB |
| MTP n_max=1 | 44.4 | 22.52 | 0.817 | 299.3 MiB |
| **MTP n_max=2** | **50.9** | **19.65** | 0.661 | 448.9 MiB |
| MTP n_max=3 (default) | 46.9 | 21.31 | 0.562 | 598.5 MiB |

Break-even α > 0.164; measured 0.497–0.773, so 3× margin at the worst prompt. These t/s were taken
at the **30.57 t/s** baseline (pre-O11′), so absolutes are ~5 % conservative; the **+58 % ratio** is
the transferable quantity. **Four findings that must not be lost:**

1. **The O1 fast path is lost under speculation, and it is already priced in.** Any spec type sets
   `n_rs_seq > 0`, breaking O1's precondition: **+1.33 ms/token measured**. The proposed clamp was
   **wrong** — qwen35's driver chains one head across positions, so the rollback distance is
   `n_draft − n_accepted ∈ {1,2,3}` and `seq_rm` refuses `> n_rs_seq`
   (`llama-memory-recurrent.cpp:214`). Lowering `n_max` is the only correct lever. **[P]**
2. **Token-exactness is unobservable at `-fa 1`.** MTP diverged from plain AR on 2/5 prompts at
   production flags; isolated cause is that **flash attention alone flips greedy tokens**. At
   `-fa 0 -ctk f16 -ctv f16` MTP is token-exact over 3 prompts × 129 tokens × 2 reps × 2 modes.
   **Rule: any exactness test on this host runs `-fa 0 -ctk f16 -ctv f16`.** **[P]**
3. **The mechanism is now explicable, and it is a kernel-family discontinuity (new).** The
   dispatcher gives `n_q = 1` to our GQA kernel, `n_q = 2` to VEC (when the KV length is a multiple
   of 256 — otherwise TILE), and `n_q = 3…8` to the tile kernel (`fattn.cu` selector;
   `Q->ne[1] <= 2 → VEC`, else TILE). **A speculative verify pass and an AR step therefore rarely
   run the same attention kernel** — up to three kernels across the four widths that MTP
   n_max=1/2/3 actually uses, and which one you get depends on the KV length. Bit-identical tokens
   between AR and verify are impossible while that holds, regardless of `-fa`. **[P]** for the
   dispatch; **[H]** for the share it costs.
4. **Secondary width dependence inside the tile kernel (new).** `launch_fattn`'s KV-split
   heuristic uses `ntiles_dst`, which is a function of `Q->ne[1]`
   (`fattn-common.cuh:1190-1210`): `nblocks_total = ntiles_dst * parallel_blocks_test`. So a
   3-token verify and a 1-token decode pick *different* partial-sum groupings and drift in the last
   bits. Upstream fixed exactly this class (`stew675` issue #25: "evaluate the heuristic as if
   n_q == 1 so every small batch agrees", `fattn-common.cuh` equivalent). Adopting it makes the
   tile path width-invariant; it does **not** by itself make AR == verify, because finding 3 puts
   AR on a different kernel. **[U]** for the fix, **[H]** for the gain.

**VRAM, not speed, is the constraint.** 12.88 GB spec-off vs 13.84 GB spec-on at −c 4096 on a
15.98 GB card; +2.99 GB of KV for 112k does not fit with spec on. Hence the ladder in §7.3. **[P]**

---

## 7. Deployment policy for the RX 6900 XT

### 7.1 The token budget, and what each part costs to remove

```
22.13 ms   11.351 GB of weights at the card's best-achieved 513 GB/s   ← bytes only
 2.66 ms   per-block overhead in the matvec (the 6.9–13.1 % residual)  ← 12 attempts, all null
 4.15 ms   every non-matvec kernel (attention 0.74, norm 0.96, ssm 0.78, rest 1.67)
 2.11 ms   dispatch gap (1 788 launches × 1.19 µs)
---------
31.05 ms   measured 31.01–31.05
```

Perfect-kernel bound: 22.13 ms → **45.2 t/s** single-stream. The MTP multiplier is orthogonal
(×1.58 measured). **The MTP pass itself has no published budget** — the +58 % was measured, never
decomposed. That is now a 1-hour measurement, not a guess (§9, N0).

### 7.2 Quantization: the only large unspent lever, and it is a reversal

Under C1, **time ∝ bytes at 458–513 GB/s**, so removing a gigabyte is worth **2.0–2.2 ms**:

| build | bpw | weights | matvec | token | t/s | vs today | quality risk |
|---|---:|---:|---:|---:|---:|---:|---|
| today (mixed IQ3_S/IQ4_XS/Q4_K) | 3.544 | 11.35 GB | 24.8 | 31.0 | **32.2** | — | baseline |
| uniform IQ3_XXS | 3.06 | 9.80 GB | 21.8 | 28.0 | **~35.7** | +11 % | low |
| uniform IQ2_S | 2.56 | 8.21 GB | 18.7 | 24.9 | **~40.1** | +25 % | moderate |
| uniform IQ2_XXS | 2.06 | 6.60 GB | 15.5 | 21.8 | **~45.9** | +43 % | high |
| *uniform Q4_0 (for contrast)* | 4.50 | 14.41 GB | 30.8 | 37.0 | *~27.0* | *−16 %* | — |

The last row is the arithmetic refutation of the wiki's "Q4_0 is fastest" finding on **our** card
(§5.6): its native 4-bit dot path does not lift byte rate, and 4.5 bpw is 27 % more bytes than the
current mix. **[H]** for the first four rows — arithmetic from the corrected model, *not measured*.
The old model predicts **+0 %** for all of them, so this is maximally discriminating: one
`llama-bench` A/B settles it and dominates every kernel item in §9.
**Do not requantize the IQ3_S file downward** (lossy-on-lossy) — obtain a build quantized from
originals.

**New (2026-10-27):** both gfx1030 forks that hand-wrote *native arithmetic* for a format
(4-bit dot8, tiled-FA `fdot2`) measured **neutral-to-negative** end-to-end. The lever is bytes, not
ISA features. **[U]**

### 7.3 Residency and context ladder

Everything is GPU-resident: no configuration with an offloaded dense matvec is viable (§7.6).
Fixed per-process overhead at 4k is 1.53 GB over the weights (KV 117 MB + recurrent state 157 MB +
compute buffers).

| mode | max ctx | what binds | measured |
|---|---:|---|---|
| spec off | **112k** | KV 2.99 GiB + weights 11.35 GB | 22.16 t/s |
| MTP n_max=2 | **86k** | +0.96 GB spec | 50.9 t/s at short ctx; **32k unmeasured** |
| MTP n_max=3 | 80k | +1.1 GB spec | 46.9 t/s |
| multi-slot k=2, 4k | 55k | 2× KV + 2× state | 55.6 t/s/slot |
| multi-slot k=4, 4k | 26k | 4× KV + 4× state | 80.1 t/s/slot |
| dropping to 3.06 bpw | **+1.55 GB** headroom | — | would put MTP at ~112k |

Recurrent state is **156.9 MB, context-independent** (48 GDN layers × (`n_embd_r` 30 720 +
`n_embd_s` 786 432) × 4 B) — a fixed tax on every slot, which is why multi-slot dies at long
context before KV does. **[P]**

### 7.4 Backend routing (numbers, not vibes)

| workload | backend | why | delta |
|---|---|---|---|
| interactive decode ≤86k, single user | **HIP + MTP n_max=2** | 50.9 t/s vs 34.4 (VK, no MTP) | **+48 %** |
| decode >86k (86–112k) | **Vulkan** | 23.23 vs 22.16 | +4.8 % |
| prompt-heavy / RAG / batch ingest | **HIP** | 625 vs 507 t/s | **+23 %** |
| many concurrent short requests | **HIP, spec off, `-np 4`** | 320 t/s aggregate at 4k | **+894 %** |
| single-backend constraint | **HIP** | +58 % with MTP, +23 % prefill, −6.3 % bare decode | — |

**Not a runtime hybrid** — KV lives in a backend buffer. The `llama_state_seq_save/load` bridge
(prefill HIP, decode Vulkan) remains **[H]** and unverified for GDN recurrent state.

**The one unmeasured number that could change this table: MTP on Vulkan.** Predicted 48–54 t/s;
cost: one pair of `llama-bench` runs. **Still P0.**

### 7.5 KV policy

- `-ctk q8_0 -ctv q5_0` is current. K row = 272 B = 5 sectors (**85 %** sector efficiency);
  V = 176 B = 3 sectors (91.7 %). **[P]**
- `-ctk q5_0` → K row 176 B: **−19.6 % KV bytes**, 2.99 → 2.40 GiB at 112k; worth **+4 % at 112k
  and 0 at d0**. Needs a PPL check. **[H]**
- Below **~4 570 tokens** the KV working set (117.4 MB @ 4k) fits inside Infinity Cache and
  attention cost is flat in depth — measured. That is why ≤4k is the batching regime. **[P]**
- The GQA kernel made V quantisation free (q8 V 22.39 vs q5 V 22.16 at 112k, inside noise). **[P]**

### 7.6 Offload: the arithmetic that closes it

Decode needs 11.351 GB per token. At 32 t/s that is **363 GB/s**, 14× PCIe 4.0 x16. Any offload
fraction above **~7 % of weights** makes decode PCIe-bound. Prefill is the milder case (0.52 s of
PCIe against 0.82 s of compute), so an offloaded prefill roughly halves pp512 — survivable, pointless.
**Policy: `-ngl 99` or a smaller quant. Never a partial offload.** This is also why #793/#858 had
nothing to offer a 16 GB card. **[P]**

### 7.7 Recommended invocations

```bash
# A. interactive default — single user, ≤86k
llama-cli -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 \
          --spec-type draft-mtp --spec-draft-n-max 2 -c 86000

# B. long context (>86k) — spec must be off at 16 GB
llama-cli -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -c 112000

# C. throughput / agentic — 4 slots at ≤26k, spec off
llama-server -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -c 104000 -np 4

# D. decode-only, single user, long prompt served elsewhere
#    Vulkan build, same flags, no --spec-type
```

`-fitt 128` is load-bearing. `--spec-draft-n-max 2` is a *host* choice, not a default to patch (C3).
**Host preconditions:** none beyond the benchmarking rules in `docs/gfx1030-benchmarking.md`.
**DPM pinning is not required and not possible** — sysfs clock files are read-only to this user and the card runs `auto`. H-CLOCK is closed; do not re-test clocks.

---

## 8. Capability gates and fail-closed design

### 8.1 Gate matrix as it stands

| gate | condition | on reject | validated | evidence |
|---|---|---|---|---|
| GQA decode attention | `GGML_CUDA_CC_IS_RDNA2(cc)` ∧ head_dim 256 ∧ `Q->ne[1]==1` ∧ `Q->ne[2]==6·K->ne[2]` ∧ K=Q8_0 ∧ V∈{Q8_0,Q5_0} ∧ no sinks ∧ softcap 0 ∧ max_bias 0 ∧ mask null-or-F16 ∧ env≠0 | falls through → VEC (bit-identical contract) | gfx1030; gfx1031/1032 admitted **as ISA**, logged shape-unvalidated | `fattn.cu:529-618` |
| MMQ config | per-dtype `CASE(type, 256, 2, 128, J)`; host `cc` switch, device `#ifdef` ladder | terminal `#else → rdna2` for every unlisted HIP arch | gfx1030 ✓ | `mmq-config-rdna2.cuh`, `mmq.cuh:205` |
| MMVQ dtype dispatch | `get_vec_dot_q_cuda` | **`GGML_ABORT`** | — | `mmvq.cu:76` |
| MMVQ VDR | `get_vdr_mmvq` | `return 1` (conservative, correct) | — | `mmvq.cu:101` |
| O1 state alias | `allow_direct ∧ n_seq_tokens==1 ∧ n_seqs==1 ∧ n_rs_seq==0 ∧ n_rs==1 ∧ get_size()==1` | materialised gather | gfx1030, bit-identical | `llama-graph.cpp:3978` |
| O11′ prequantised norm | context-owned arena, mmvq lookup miss → quantise itself | self-quantise | gfx1030, byte-exact | `norm.cu`, `mmvq.cu` |
| MTP | explicit `--spec-type draft-mtp` only; never `auto` | plain AR | gfx1030 | `common/common.h:329` |
| RDNA2 dtype warning | **absent here** — upstream analogue only (`load.rs:2601`) | — | — | **[U]** |
| **N1 verify-band MMVQ geometry** *(proposed)* | `IS_RDNA2` ∧ `2 ≤ ncols_dst ≤ 8` ∧ type ∈ measured set ∧ `MMVQ_MAX_BATCH_SIZE≥8` | `calc_rows_per_block → 1`, `nwarps → 1` (today's behaviour) | gfx1030 (upstream V620) | §5.2 |
| **N4 MMQ float scale** *(proposed)* | `#if defined(RDNA2)` only | integer multiply (today) | bit-identical by RN | §10.2 |
| **N5 tile tail check** *(proposed)* | `#if defined(RDNA2)` ∧ last KV tile of the `ncols2>1` branch | current unchecked load | gfx1030 (upstream V620) | §5.2 |

### 8.2 The four rules this matrix encodes

1. **Exact arch, exact shape, all-or-nothing dtype.** No gate may admit on "same family" unless the
   family *is* the ISA floor it depends on (`IS_RDNA2` is defensible because `sdot4` is the
   dependency; `== gfx1030` would be a guess dressed as caution).
2. **Refusal keeps the incumbent path.** A gate must return a *worse* path, never an error and never
   a silent alternative.
3. **Every gate must be provable to reject.** A gate with no negative test is an unverified claim.
4. **New, from two forks: prefer a shape list to an arch table.** `edwinbrowwn` shipped an
   eight-wave Q8_0 kernel that helped `6656×128` by 17.4 % and hurt `6656×2048` by 6.2 % — the same
   `nwarps` on the same arch, opposite sign. Their selector is therefore an explicit tuple list.
   Any N1/N3 admission must be tuple-scoped or band-scoped with a measured sign, never "RDNA2 likes
   wider blocks". **[U]**

### 8.3 Known gaps (small, real)

- **`mmq.cuh:265-281` host/device predicate twins.** Unlisted HIP arches silently receive the RDNA2
  config; the safe form is `#else #error`. Build-only, zero runtime change. **[H]**
- **`gfx1030.py audit`'s fallback-hazard detector** needs one false positive removed (§2 C4).
- **`ggml/src/ggml-hip/CMakeLists.txt:73-81` sets per-file options on
  `../ggml-cuda/mmq-hipfire-dispatch.cu`, which does not exist.** `set_property(SOURCE …)` does not
  require existence, so the build passes; it is dead configuration from a reverted experiment and
  should be deleted so a future reader does not believe an MMQ dispatch TU exists. **[P]**

---

## 9. Implementation priority

Ordered by measured (or arithmetically bounded) EV per unit of work.

| id | item | type | EV | gate |
|---|---|---|---|---|
| **N0** | **Trace + census under `--spec-type draft-mtp`** — attribute the 19.65 ms pass: which FA kernel runs at width 3, what attention/norm/dispatch cost inside a speculative round | measurement, 0 code | decides whether N1/N3 can matter at all; no existing instrument has ever looked inside the MTP pass | `gfx1030.py trace --ledger` + `census`, spec on, ≥8 rounds, ledger must close to the 19.65 ms pass within 3 % |
| **P0** | **Measure `--spec-type draft-mtp` on Vulkan** | measurement, 0 code | decides the interactive default (±? on 50.9 t/s) | paired A/B, ≥5 pairs; report α as well as t/s |
| **N1** | **Verify-band MMVQ geometry** (rows 2 + nwarps 2 at widths 2–4; rows 4 at 5–8), band-gated, decode untouched | kernel dispatch | upstream **+6.3 % / +7.6 %** on gfx1030 ⇒ ~+3–6 % on the MTP pass ⇒ **52–54 t/s** | bit-identical at width 1 (must be zero diff); width 2–8 diff vs stock must be *zero token* change at `-fa 0 -ctk f16`; paired A/B ≥4/5 same sign |
| **P1** | **Lower-bpw A/B (uniform IQ3_XXS, then IQ2_S)** | product work | **+11 % / +25 %** predicted; +0 % under the old model | same corpus PPL + paired A/B decode and prefill |
| **N4** | **`ggml_cuda_mmq_scale_sumi` on RDNA2** (3 sites in `vecdotq.cuh`) | kernel, 6 lines | prefill only, **[H]** 1–3 % pp512; bit-identical by construction | `test-backend-ops -o MUL_MAT` bit-identical + paired pp512 A/B |
| **N2** | **MTP at 32k and 86k** (ladder is VRAM-derived, not measured) | measurement | confirms §7.3 or moves the ceiling | VRAM peak + t/s at 3 depths |
| **P3** | **`-ctk q5_0` PPL + depth A/B** | 1-line config | +4 % at 112k, 0 at d0 | PPL delta ≤ noise; no 32k regression |
| **N3** | **Extend the gfx1030 GQA attention kernel to `n_q ≤ 8`** | kernel | **(a)** removes the VEC/TILE discontinuity at widths 2–8, the structural cause of AR ≠ verify (§6.3 #3); **(b)** at 4k attention is 0.74 ms/token, ×3 in a verify pass; **[H]** 0.3–0.6 ms/pass | exactness first: row 0 of a 3-row call must equal the 1-row call byte-for-byte; then paired A/B at n_max=2 |
| **P5** | **Dispatch programme, `PLAN_3MS` Phases 0b→2** | kernel/graph | ≤0.58 ms gap + ≤1.5 ms kernel | census `diff` must show ΔN_dispatch < 0; ≥0.70 ms combined |
| **P6** | **`#else #error` in `mmq.cuh`** + delete the dead `mmq-hipfire-dispatch.cu` CMake stanza | hygiene | 0 here, prevents a silent class | builds; `test-backend-ops -o MUL_MAT` unchanged |
| **N5** | **RDNA2 tile tail check + 511-row guard tail** | safety | 0 perf; removes a latent fault class if we move to ROCm 7.14 | `test-backend-ops -o FLASH_ATTN_EXT` unchanged (it already passes); byte-diff 0 |
| **N6** | **RDNA2 tile-FA config table** | kernel table | **prefill attention only** for our widths (+31 % claim); **changes numerics**; the cols=4 row keeps our tile shape and only relaxes a launch-bounds hint | ≥5 % pp512 **and** token stream unchanged over 512 greedy tokens; 1-day cap |
| **N7** | **Chunked GDN prefill** (lane-0 broadcast) | kernel | upstream +7.9 %/+17.7 % *direct*; e2e unclaimed | direct-kernel A/B first; adopt only if pp512 ≥3 % |
| P8 | fused GDN chain (`PLAN_3MS` Phase 5) | structural | ≥1.2 ms or abandon | entry only if P5 < 2.5 ms total |
| P9 | KV/attention at depth beyond the GQA kernel | — | ≤0.4 ms | — |

**Blocked / not in the order:** everything MoE; everything TP; anything offload; anything that
depends on Infinity Cache control; any widening of gfx1030 admission on gfx11 evidence; anything
whose only evidence is a wiki throughput table (§5.6).

**Sequencing note.** N0, P0, N2, P3 are measurement or config and cost ~1.5 days of GPU time
combined. They run **before** N1/N3, because N1's sign depends on where the MTP pass actually
spends its 19.65 ms, and because P0/P1 can invalidate the deployment table that everything else is
optimising inside of. N4 and N5 are side-quests that can land in any gap; both are bit-identical.

**What happened to the previous version's P4 and P7.** *P4 (MMQ per-shape tile sweep, 2 days)* is
**demoted to open question #9**, not deleted: its only evidence is a gfx1031 sweep (`[A]`, issue
#300), no gfx1030 fork reports one, and N4 captures the cheap part of the same idea (one multiply
in the same loop) with a bit-identical gate. Re-promote it if N4 moves pp512 ≥1 % *and* the
dispatcher still shows MMQ headroom. *P7 (`ArchCaps`-style struct)* is **dropped**: §8.2 rule 4
says admissions must be tuple/band-scoped per consumer, and a shared capability struct pushes in
the opposite direction — it invites "RDNA2 is fast at X" reasoning. The one thing it was for (a
single read-once predicate set) is already served by `ggml_cuda_device_info` plus per-gate env
kill switches.

---

## 10. Validation matrix

Our standing bar is **bit-identical numerics plus a paired A/B** — never a pass/fail test. Two
landed fusions passed correctness tests while writing corrupted bytes (§10.2), so the bar is
structural, not aspirational.

### 10.1 Levels

| L | name | what it proves | instrument |
|---|---|---|---|
| **L0** | build | compiles clean, both backends | `-DLLAMA_FATAL_WARNINGS=ON`, `clang-format` |
| **L1** | numeric identity | the change is a re-expression, not a new computation | §10.2 |
| **L2** | performance | faster above MDE, reproducibly | §10.3 |
| **L3** | fallback | every reject path produces the incumbent's output | force the gate false, diff L1 artifacts |
| **L4** | unsupported route | unsupported inputs fail cleanly, not silently | negative-case table (§10.5) |
| **L5** | architecture smoke | the gfx1030 path is *taken*, and no gfx11 assumption leaked in | dispatch census, route log, `gfx1030.py audit` |
| **L6** | capacity | VRAM and context behave as documented | peak VRAM per (ctx, spec), `-fitt` margin |
| **L7** | *new* band invariance | a width-N call and a width-1 call agree where the math allows it | §6.3: per-width token/byte comparison at fixed `-fa` |

### 10.2 L1 — the four checks, all mandatory

1. `llama-eval-callback -ub 1` — **byte-level** `name||sum` stream over ≥512 decode steps.
2. `llama-eval-callback -ub 512 -n 16` — batched/prefill path.
3. **PPL on a fixed corpus, exact match** (O1/O3: 4.4122 ± 0.64159 both).
4. `llama-cli --temp 0 -s 42` greedy stream, byte-identical.

Plus, for anything touching a quantised path: **a byte test, not a tolerance test.** And for
anything touching a fused output: **an explicit output-vs-output overlap assertion** (O3's
`ggml-alloc` aliasing was invisible to every numeric test except the byte-level one).

**Two changes in §9 cannot be bit-identical, and the plan says so up front:**

| change | why not | the substitute gate |
|---|---|---|
| N1 verify-band geometry | `nwarps` participates in the K-split accumulation order | width 1 must be **byte-identical** (it is untouched); widths 2–8 must be **token-identical** at `-fa 0 -ctk f16 -ctv f16` over ≥512 tokens, plus a per-element max-abs ≤ the kernel's existing test tolerance |
| N6 tile config / N3 kernel | tile shape/warp count changes the online-softmax reduction order | token stream identical over ≥512 greedy tokens at fixed `-fa 1`, PPL within noise, `test-backend-ops -o FLASH_ATTN_EXT` full pass |
| N4 MMQ float scale | *claimed* bit-identical: `(float)(sc*sumi) == (float)sc*(float)sumi` because both are the correctly-rounded exact integer product (no double rounding, `sc ≤ 255`, `sumi < 2^17`) | verify, do not assume: `test-backend-ops -o MUL_MAT` byte comparison with the patch forced off |

### 10.3 L2 — the paired A/B protocol

- `scripts/bench-exclusive.sh` for every run. **`llama-server` must never be active**; exactly one
  `llama-bench` at a time.
- `scripts/gfx1030.py bench --ab BASE.so FIX.so --pairs 5` (10 for anything ≥0.5 ms), interleaved,
  with `cmp` verifying the restored library.
- **Swap every library whose bytes changed, and prove it with sha256.** O1 was pure graph
  construction, so `libggml-hip.so.0` was byte-identical; O3/O11′ changed kernels *and* matchers, so
  both libraries move. Swapping the wrong one measures a no-op — this has already happened once.
- Record with every result: both sha256s, ROCm version, **selected DPM state and sclk**, run order.
  **MDE at d0 is 0.30 ms / 0.30 t/s; no verdict below it, regardless of sign.**
- Kernel-level `trace --ledger` accompanies every end-to-end claim; isolated kernel wins have
  repeatedly not transferred (Q4_K: −34 % isolated, 0.0 % e2e; upstream: rows=4, "+2.9 % regression").
- **Token-exactness for anything speculative runs `-fa 0 -ctk f16 -ctv f16`** (§6.3).

### 10.4 Minimum pass criteria (all four, or the change is rejected)

1. **L1 identical**, or the substitute gate in §10.2 for the three documented exceptions.
2. **Paired A/B median above MDE**, with ≥4/5 (ideally 10/10) pairs of the same sign.
3. **Census delta has the predicted sign**: `census` + `diff` must show ΔN_dispatch < 0 for any
   fusion.
4. **No new `cont`/`view`/`concat` node without ≥3 µs of savings per node added.**

### 10.5 Unsupported routes that must fail cleanly (L4)

| route | required behaviour | how to prove |
|---|---|---|
| KV type outside {Q8_0, Q5_0} | selector rejects, VEC runs, same tokens | `-ctv q4_0`, diff L1 |
| head_dim ≠ 256, GQA ≠ 6, `Q->ne[1] > 1` | reject (GQA kernel); N3 must reject `n_q > 8` | `test-backend-ops` cases + forced-shape run |
| sinks / softcap / ALiBi / non-F16 mask | reject | forced tensors, assert the selector returns false |
| `GGML_HIP_GQA_GFX1030=0` | VEC path, **identical output** | one full L1 set with the env off |
| non-RDNA2 device | gate not selected | cc check; log absent |
| MMA/WMMA/fp8 KV request on gfx1030 | must not exist in a reachable path | grep + predicate audit |
| MQ/HFQ/MQ-V2 dtype in a GGUF | load-time refusal | convert attempt must fail |
| offloaded matvec | not offered as a supported config | docs + `-ngl <99` behaviour |
| **N1 at width 1** | must be byte-identical to today | L1 check 1 diff = 0 |

### 10.6 Instrument traps (each has already cost this project a wrong conclusion)

| trap | rule |
|---|---|
| **÷17, not ÷16** — llama-bench runs a 1-token warmup generation | divide per-token counts by 17 |
| **Infinity-Cache illusion** — any working set <128 MB is IC-resident | every isolated number paired with a DRAM-regime number at k ≥ 131 072 |
| **`GGML_CUDA_DISABLE_GRAPHS=1` inflates wall by ~44 %** | trace for per-kernel time; never for wall. **A public fork benchmarks with graphs off — its t/s are not comparable to ours** (§5.6) |
| **Event overhead** ~3.5 µs per `hipEvent` pair | idle inferred from node counts, not a graphs-off run |
| **Thermal/clock drift** — Vulkan moved 33.3 → 34.4 t/s in one session | always interleave; record run order. **Clocks are NOT a lever: sclk measured 2485 MHz on both backends, every in-window sample — hypothesis H-CLOCK is closed, see "Closed hypotheses — DO NOT RETEST"** |
| **Concurrency** — two `llama-bench` silently corrupt each other | `bench-exclusive.sh` only |
| **`llama-server` must never run** on this host | `pgrep -a -f "llama-server\|llama-bench"` must be empty pre-flight |
| **`-fa 1` flips greedy tokens vs `-fa 0`** | exactness tests run `-fa 0 -ctk f16 -ctv f16` |
| **Cross-implementation numbers do not cross** — same arch, same model family, different kernels | an upstream V620 delta is an *expectation*, never a result. Re-measure or do not claim |
| **A cost model can rank items backwards** | `gfx1030.py predict` is a sanity check only; it inverted the O1/O3 ranking |

---

## 11. Refusals

- **Porting the `Step`/`OpKind` executor, `program.rs`, `DispatchCtx`, `GdnTape`, `RowTwins`.**
  ggml's `cgraph` + `ggml_backend_sched` already own the responsibility. **[X]**
- **Porting any gfx1100/gfx1151/gfx1201 Qwen 3.5 fusion** (`gdn_compact2/3`,
  `gated_norm_mq_rotate`, `fused_attention_prep/epilogue`, `conv_scalar_prep`, `qkvza_scalar_prep`).
  All are arch-gated away from gfx1030; importing them is dead code or a support claim with no
  evidence. **[X]**
- **The RDNA2 occupancy recompute.** Its stated cause (HIP reports 32768 regs/WGP) is **false on
  ROCm 7.2.4**, measured: 131 072. Re-measure before admitting any toolchain-bug workaround (C6). **[X]**
- **Q4_0 DOT8 / any native-arithmetic substitution.** Implemented on gfx1030 by a public fork,
  exact, **−0.71 %**. The ALU is not the bound (C1). **[X]**
- **Rows/blocks > 1 at `ncols_dst == 1`.** Measured negative here and upstream (+2.9 % regression);
  Navi21's own table returns 1 row there. **[X]** — *distinct from N1, which is widths 2–8.*
- **An arch-wide RDNA2 warp table.** The same `nwarps` helped 17.4 % and hurt 6.2 % on two shapes of
  one arch in one fork. Shape lists only (§8.2 rule 4). **[X]**
- **GDN sibling-projection concat (`[wqkv|z]`, `[beta|alpha]`).** Requires a uniform dtype across the
  pair; our GDN block is IQ3_S / Q3-ish / bf16. Would force a requantisation to enable a fusion worth
  ≤0.1 ms after O11′. **[X]**
- **Tensor parallel on this host.** One card; the internal all-reduce asserts `n_backends == 2`,
  accepts only F32/F16/BF16, and TP4 has no fast path. Correctness *has* been shown in a 3-card
  mixed split upstream (C7) — which does not make it relevant here. **Do not claim gfx1030 TP here.** **[X]**
- **Any MoE patch.** No expert path in a 65-block dense 27B; the only public gfx1030 MoE work is a
  routed-token band for `MUL_MAT_ID` (§6.2). Record, do not port. **[X]**
- **#793 / #858 as evidence.** Both **closed unmerged**; both re-implement `-ngl`/`-ot`. **[X]**
- **Infinity Cache as a managed software tier.** 81.4× oversubscription on the weight stream; the
  reusable data is already resident; the control surface (persisting-L2 window, `glc`/`slc`,
  prefetch) does not exist on this toolchain; rocprofv3 exposes no IC counters; **and no public
  project implements it** (§5.7). **[X]**
- **Cache-hint / `__ldcs` / prefetch / LDS-staging programmes.** Closed by the toolchain. **[X]**
- **Widening `IS_RDNA2` to `== gfx1030`,** or the reverse (tightening). The family check *is* the
  ISA floor (`sdot4`). **[X]**
- **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** — native gfx1030 here; the override can only select foreign
  code objects. **[X]**
- **Benchmark numbers copied from the gfx1030 wiki.** Graphs-off, TP2/TP4, different quant.
  Direction only. **[X]**
- **Enabling MTP by default** (`common/common.h:329` to 2). It is a host flag. **[X]**
- **Claiming a prefill improvement from hipfire**, or any performance number from the two RDNA2
  forks without re-measuring it here. **[X]**
- **Touching `llama-server` while measuring, or running two benchmarks.** **[X]**

---

## 12. Open questions, each with the artifact that answers it

| # | question | artifact | cost |
|---|---|---|---|
| 1 | Where do the 19.65 ms of an MTP n_max=2 pass go, and which FA kernel runs at width 3? | `trace --ledger` + `census`, spec on | 1 h |
| 2 | Does the verify band really want 2 rows / 2 waves on our shape+dtype? | N1 paired A/B at n_max=2 | 0.5 day |
| 3 | Does MTP raise Vulkan decode above HIP's 50.9 t/s? | one paired pair per backend | 1 h |
| 4 | Does decode really scale with bytes? (C1) | any lower-bpw build, decode A/B + PPL | 0.5 day |
| 5 | What is MTP's t/s at 32k and 86k? | depth sweep with spec on | 1 h |
| 6 | Is `-ctk q5_0` free at 112k? | PPL + depth A/B | 1 h |
| 7 | Can one attention kernel serve widths 1…8 exactly? (N3) | row-0-vs-1-row byte comparison | 2 days |
| 8 | Is the 1.19 µs/dispatch gap hardware or our submission path? | `PLAN_3MS` Phase 0b | 0.5 day |
| 9 | Can MMQ per-shape tiles buy ≥5 % prefill? | J sweep on the 4 GEMM shapes | 2 days |
| 10 | Is `-fa 1` ≠ `-fa 0` a defect or a kernel-selection tie? | `test-backend-ops -o FLASH_ATTN_EXT` at `ne[1]=1` vs tile | 0.5 day |
| 11 | Does the kv=113 tile fault reproduce on ROCm 7.2.4? | **answered no** — our full `FLASH_ATTN_EXT` suite (4045/4045) includes the upstream case (D=256, GQA 2, `n_q={16,1}`, kv=113, Q8_0); re-open only after a ROCm upgrade | — |
| 12 | Is the RDNA1→RDNA2 MMQ config leak real? | gfx1010/1013 device — **unanswerable here** | — |
| 13 | Does the meta-backend all-reduce work on 2× gfx1030 without RCCL? | second card — **unanswerable here** (correctness shown for 2×gfx1100+1×gfx1030 upstream, C7) | — |

---

## 13. Appendix — exact census and the arithmetic

Per-token matvec weight census, main graph only (`gguf-py`, 2-D tensors, `blk.64.*` and
`token_embd` excluded):

| role | tensors | Gelem | GB | bpw |
|---|---:|---:|---:|---:|
| ffn_down | 64 | 5.704 | 2.597 | 3.64 |
| ffn_gate | 64 | 5.704 | 2.390 | 3.35 |
| ffn_up | 64 | 5.704 | 2.410 | 3.38 |
| attn_qkv (GDN) | 48 | 2.517 | 1.083 | 3.44 |
| attn_gate | 48 | 1.510 | 0.667 | 3.53 |
| ssm_out | 48 | 1.510 | 0.719 | 3.81 |
| output (lm_head) | 1 | 1.271 | 0.715 | 4.50 |
| attn_q | 16 | 1.007 | 0.396 | 3.14 |
| attn_output | 16 | 0.503 | 0.234 | 3.71 |
| attn_k / attn_v | 32 | 0.168 | 0.085 | 4.05 |
| ssm_alpha / ssm_beta (bf16) | 96 | 0.024 | 0.048 | 16.0 |
| ssm_conv1d (f32, **not** a matvec) | 48 | 0.002 | 0.008 | 32.0 |
| **matvec total** | | **25.624** | **11.351** | **3.544** |
| MTP block (`blk.64.*`, 18 tensors) | 18 | 0.425 | 0.348 | 6.55 |
| token_embd (GET_ROWS only) | 1 | 1.271 | 0.407 | 2.56 |
| file total | 866 | 27.321 | 12.109 | 3.546 |

Byte model: `t_matvec = 11.351 GB / 513 GB/s + 2.66 ms = 22.13 + 2.66 = 24.79 ms` (measured 24.79).
Quantization conversion: **0.1 bpw = 0.32 GB = 0.70 ms = 2.3 % of the token.**
Per-token hardware floor at zero dispatch cost: 22.13 ms → 45.2 t/s. With MTP ×1.58: ~71 t/s.

Device facts measured for §1.1 (`hipGetDeviceProperties`, gfx1030, ROCm 7.2.4):
`multiProcessorCount 40`, `maxThreadsPerMultiProcessor 2048`, `regsPerMultiprocessor 131072`,
`sharedMemPerMultiprocessor 65536`, `l2CacheSize 4194304`, `PersistingL2CacheMaxSize 4194304`,
`AccessPolicyMaxWindowSize 0`, `MaxBlocksPerMultiProcessor 64`, `clockRate 2660000 kHz`,
`memoryClockRate 1000000 kHz`, `memoryBusWidth 256`, `totalGlobalMem 17163091968`,
occupancy(dummy 256 thr) 8 blocks.

---

## 14. What changed versus the previous version, and what I refuse to include

### Changed

1. **Verdict 3 is narrowed and verdict 4 is new.** The old text generalised hipfire's null result to
   the whole RDNA2 ecosystem ("hipfire has nothing to port … and that is now a documented upstream
   position"). Two C++ forks carry real, gated, externally-validated gfx1030 kernel work. hipfire is
   still port-nothing; the ecosystem is not.
2. **New §5** — the four mission sources plus the fork the wiki points at, read at file level,
   graded, and each claim marked usable / low-value / refuted / not-applicable. Highlights: two of
   the four sources contain **no kernel code at all**; the RDNA2 occupancy fix's premise is
   **falsified on this toolchain**; the RDNA2 tile table's cols=4 entry keeps our tile shape and
   changes only a launch-bounds hint, so its measured gains **do not reach MTP n_max=2**; a native
   4-bit dot on gfx1030 measured **−0.71 %**.
3. **New §2 C5, C6, C7** — the geometry null is band-limited; the occupancy-undercount claim is
   false here; gfx1030 TP is externally validated for correctness (still irrelevant on one card).
4. **New §2 C4 bullet** correcting the LDS attribution from "64 KB per CU" to "64 KB per WGP
   (HIP)", which makes the GQA kernel's 3-block bound a 94 % budget rather than 47 %.
5. **MTP section (§6.3) gained two mechanisms**: the VEC/GQA/TILE kernel discontinuity at widths
   1/2/3–8, and the `ntiles_dst`-dependent KV split. Both are new, both are testable, and together
   they explain the `-fa 1` token divergence as a kernel-selection property rather than a mystery.
6. **§9 renumbered around a new measurement (N0) and six new items** — N1 verify-band MMVQ
   geometry, N3 GQA width extension, N4 MMQ float scale, N5 tile tail check, N6 tile table, N7
   chunked GDN prefill. The verify-band geometry (N1) is now ranked above every kernel item, because
   it is the only *external* gfx1030 performance evidence that lands on a band we actually use.
   P4 (MMQ sweep) is demoted with its reason stated; P7 (`ArchCaps`) is dropped with its reason
   stated.
7. **§10.2 now tables the three changes that cannot be bit-identical** and gives each a substitute
   gate, instead of asserting "bit-identical" as a universal bar.
8. **§11 grew** Q4_0-DOT8, rows>1-at-width-1, arch-wide warp tables, GDN sibling concat,
   `HSA_OVERRIDE_GFX_VERSION`, and wiki-copied numbers.
9. **§7.7 and §10.3 now pin the DPM state** — the card was found at `manual` / state 1 (2485 MHz of
   a 2660 MHz ceiling) while both RDNA2 tuning repos pin `high`.

### What I refuse to include

- **Any performance number from a source I could not run.** The V620 deltas (+6.3 %, +7.6 %, +31 %,
  −17 %/−30 %, +7.9 %/+17.7 %) appear only as *expectations* attached to a re-measurement gate; none
  is used as a claim about this host.
- **The wiki's llama.cpp tables.** Graphs-off, TP2/TP4, different quantisation — structurally
  incomparable with §1 and therefore with every A/B in this document.
- **The RDNA2 occupancy patch.** Not because it is wrong upstream, but because its cause is absent
  here and the symptom does not reproduce. Including it would be cargo-culting.
- **Q4_0 DOT8, rows>1 at width 1, MoE six-row routing, TP/output-sharding, backend sampling,
  DFlash, sibling-projection concat** — each with its stated reason above.
- **"Infinity Cache tuning".** Four implementations, zero control surfaces, one measurement
  (KV ≤ 4.6k) and one arithmetic reason (81.4× oversubscription). I will not dress a hardware
  property as a feature.

---

*Nothing here is committed. All measurements via `scripts/bench-exclusive.sh`; `llama-server` was
never active. Where this document contradicts `ARCHITECTURE_GFX1030.md` §"governing fact", §2 C1 of
this document governs and the experiment in §12 #4 is the arbiter. Where it contradicts
`HIPFIRE_PORT_ROUND2.md` §D.5 on tensor parallel, §2 C7 governs.*

---

## Closed hypotheses — DO NOT RETEST

### H-CLOCK — REJECTED [X] (measured 2026-10-27)

**Hypothesis:** HIP runs at a lower DPM state than Vulkan, so the backend gap is a clock
artifact (2485 MHz observed vs 2660 MHz max = 7.0 %).

**Verdict: DEAD.** sclk sampled every ~100 ms *during* decode, 711 samples across five
runs:

| run | in-window samples | sclk |
|---|---:|---|
| HIP `build-hip-gfx1030` | 47/47 | 2485 MHz |
| HIP again | 48/48 | 2485 MHz |
| `build/bin -dev ROCm0` | 47/47 | 2485 MHz |
| `build/bin -dev Vulkan0` #1 | 53/53 | 2485 MHz |
| `build/bin -dev Vulkan0` #2 | 54/54 | 2485 MHz |

**2660 MHz was never observed on either backend in any sample.** The card sits in
`power_dpm_force_performance_level = auto`, and the sclk table is dynamic (2 states idle,
3 under load) — an idle read does not describe a run and must never be used as evidence.

**The arithmetic that closes it:** clock ratio 2485/2485 = **1.0000** against throughput
ratio 34.00/32.083 = **1.0597** (+6.0 % mean, +6.6 % best pair). Scaling *both* backends
to 2660 (x1.0704) leaves the ratio **identical at 1.0597**. A deficit applied equally to
both sides cannot produce a relative gap.

**Consequences, to be treated as settled:**
- The backend gap is a **real kernel/backend difference**. Do not re-measure clocks to
  explain it; do not treat DPM state or P-state pinning as a lever.
- The only surviving clock fact is the absolute ceiling: +7.0 % headroom exists if sclk
  ever reaches 2660, and it applies to **both** backends equally.
- mclk is **unobservable via sysfs during load** (the active-state marker is absent in
  310/310 load samples). If it is ever needed, use `rocm-smi --showclkfreq` or
  `amdgpu_top`, not `/sys/class/drm/*/pp_dpm_*`.

### Ops rule — `-dev` must be explicit in every backend comparison

`build/bin/llama-bench` is a **dual-backend build** (`GGML_HIP=ON` *and*
`GGML_VULKAN=ON`; backend column reads `ROCm,Vulkan`). With `-dev auto` it selects
**ROCm0**, which returned 32.07 / 32.23 t/s — indistinguishable from HIP. The real Vulkan
number only appears with `-dev Vulkan0`: 34.22 t/s / 29.22 ms.

Any historical "Vulkan" figure taken from `build/bin` without an explicit `-dev Vulkan0`
is **suspect and must be re-taken** before it is cited.
