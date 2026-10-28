# gfx1030 (RX 6900 XT) decode plan v2.4

Target: this machine only (gfx1030 / Navi 21 / RX 6900 XT, 16 GB, ROCm 7.2.4, wave32, 80 CU).
Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (27.32 B params, 11.28 GiB, 65 blocks, **dense**, MTP head = blk.64).
Status: plan only. The FlashAttention kernel from v1 is already in the tree (uncommitted).

Supersedes v2 (2026-10-25 20:20) by adding: the prefill verdict, the exact RDNA2
parameter-table finding, the hipfire issue audit, the L2-traffic model, and the
retraction list in §9. v1 (`PLAN_GQA_DECODE_GFX1030.md`) is history except for §9.

---

## 1. Verdict

1. **Attention is finished as a project.** At depth 0, where FlashAttention never runs,
   HIP is already **30.87 vs Vulkan 33.15 t/s (-7.4 %)**. The entire remaining gap is in
   the decode weight path (~600 quantized matvecs per token). `fattn-gqa-gfx1030.cu`
   contributes at most the ~0.4 ms visible at 8k. Do not touch it.

2. **Prefill is already beating Vulkan by ~30 %, and there is no prefill gap.**
   Identical prompt (15 428 tokens), identical flags, two runs each, measurement order
   inverted: **HIP 556.1 / 556.9 t/s, Vulkan 425.4 / 423.6 t/s**; HIP re-run after the
   Vulkan arm on a hotter card (junction 78 °C) still 547.2 / 554.4 t/s. So the
   "HIP prefill 195.6 t/s, 2.4× behind Vulkan" story was an artefact of a `llama-cli`
   run and is retracted (`HIPFIRE_27B_GFX1030_TEST.md` §3 now carries the retraction).
   Nothing in this plan touches MMQ/GEMM, and §6 lists "improve prefill" as explicitly
   out of scope.

3. **hipfire has nothing to port for gfx1030 attention or gfx1030 decode GEMV.**
   All **35** `*.gfx1030.hip` files in `~/.hipfire/src/kernels/src` are GEMM/GEMV (verified:
   every path matches `gemm_*` or `gemv_*`); there is
   no gfx1030 attention kernel (the kernel we ported is the **gfx1100** sibling), and
   their gfx1030 KV path is FWHT-rotated 3-bit, not Q8_0/Q5_0, so the K/V math is not
   reusable. Their gfx1030 GEMV family is a *reference design*, not a source of code we
   can transplant: its quant format (HFQ4/MQ*) is not ours.

4. **The single biggest lever is Q4_K**, because `output.weight` (lm_head) is
   **Q4_K [5120 × 248320] = 1.271 B params**, re-read every token, plus ~1.23 B params of
   Q4_K `attn_output` — and Q4_K is exactly where HIP trails Vulkan most
   (**1.61×**, 1.82× at the k=4096 shape). Q4_K is 10 % of decoded bytes but the worst
   kernel.

5. **RDNA2 is the only modern AMD MMVQ table that is completely untuned.**
   `MMVQ_PARAMETERS_RDNA2` exists as an enum value and `get_device_table_id()` already
   selects it for gfx1030, but both tuning functions fall through to `1`/`1`:
   RDNA3_0, RDNA4, Turing and GB10 all return `nwarps` 2–8 and/or `rows_per_block` 2 for
   the same types. This is a two-function fill-in, not a new kernel.

6. **Ceiling if the weight path reaches DRAM roofline: ~33–37 t/s at d0**
   (Vulkan is 33.15). Parity is the floor of this work, not the ceiling — see §2.3.
   MTP speculative decoding was considered as the larger multiplier and **rejected for
   this deployment** (§5 P5): it costs a full extra layer of KV cache at full context
   and its throughput gain at 112k is unproven at the sampling settings in use. Every
   item in this plan is context-neutral and bit-identical in output.

7. **Final recommendation: do P0 → P1, and stop there unless P1 misses.** One
   workstream, one env switch (`GGML_HIP_MMVQ_RDNA2`), no output change, no context
   cost. Expected end state **33–35 t/s at d0** (from 30.87) and **24–25 t/s at 112k**
   (from 22.2–22.5), i.e. Vulkan parity plus whatever the roofline leaves. P2 is the
   only follow-on that is justified *by measurement* (Q4_K still >1.3× behind after
   P1); P3 is optional polish worth ≤2 % and should not gate anything; P5 stays
   closed. **Keep the Vulkan server as production until P1 + P4 both land** — a
   half-tuned MMVQ table that regresses any single type is worse than today's uniform
   behaviour, which is what the acceptance gate in §5 P1 is for.

---

## 2. Measurements

### 2.1 Depth sweep (`llama-bench -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -p 0 -n 64 -r 2`)

| depth | HIP | Vulkan | gap | HIP ms/tok | VK ms/tok |
|---|---:|---:|---:|---:|---:|
| 0 | 30.87 ± 0.03 | 33.15 ± 0.05 | 7.4 % | 32.39 | 30.17 |
| 8 192 | 29.28 ± 0.14 | 31.76 ± 0.02 | 8.5 % | 34.15 | 31.49 |
| 32 768 | 27.27–27.37 | 29.24 | 6.7 % | 36.6 | 34.2 |
| 112 000 | 22.16–22.47 | 23.23 | 4.5 % | 44.8 | 43.0 |

Depth fit: HIP slope **0.103 µs/KV-token**, fixed **33.3 ms**; Vulkan **0.111**, **30.7 ms**.
HIP's depth slope is *better* than Vulkan's — the loss is a constant per-token weight cost,
so attention depth scaling is not the problem. The worse 0–8k slope (HIP 0.215 vs VK 0.161)
is block-count underutilisation (4 kv-heads × 32 tiles = 128 blocks vs 240 slots), worth
≤0.44 ms and only relevant ≤8k (P3).

### 2.2 Per-type decode GEMV (µs/run, lower is better)

`test-backend-ops perf -o MUL_MAT`, `m=4096 n=1 k=14336`. These tensors fit in the 128 MB
Infinity Cache, so absolute µs measure the issue/IC ceiling; the **ratios** isolate kernel
efficiency and already bracket the end-to-end gap.

| type | HIP | Vulkan | HIP/VK | bytes/token here | share |
|---|---:|---:|---:|---:|---:|
| iq3_s | 45.53 | 41.99 | 1.084 | 3.64 GB | 31 % |
| iq4_xs | 28.46 | 22.00 | 1.294 | 3.02 GB | 21 % |
| iq3_xxs | 40.96 | 36.25 | 1.130 | 2.08 GB | 20 % |
| iq2_s | 44.23 | 35.67 | 1.240 | 0.39 GB | 9 % |
| q4_K | 39.67 | 24.69 | **1.607** | 1.58 GB | 10 % |
| q4_K (k=4096) | 14.89 | 8.17 | **1.822** | — | — |

Byte-weighted ratio ≈ **1.107**. Decode traffic = **11.64 GB/token** →
HIP 359 GB/s, Vulkan 386 GB/s of a 512 GB/s peak (70 % / 75 %).

### 2.3 Hypothesis H2′: the decode path is L2/IC-bandwidth-bound, not DRAM-bound

`mul_mat_vec_q` loads the activation row inside the block and indexes it per row; with
`rows_per_block = 1` **every block re-reads the entire quantised activation**. Per matvec:

```
y bytes ≈ rows × (k/32) × 36 B = rows × k × 1.125 B
w bytes ≈ rows × k × bpw/8
```

so the activation re-read is **2.0–2.6× the weight traffic** for every type in this model
(Q4_K 2.0×, IQ4_XS 2.1×, IQ3_S 2.6×). Model-wide that is ≈ 11.6 GB weights +
≈ 26 GB of activation re-reads ≈ **38 GB of L2/IC traffic per token**. At the measured
32.4 ms that implies **≈1.17 TB/s** — a plausible Navi 21 L2/IC figure, and it lands on
the observed time almost exactly.

If H2′ holds, the DRAM floor is `11.64 GB / ~0.42 TB/s ≈ 27.7 ms` (**≈36 t/s at d0**),
and `rows_per_block = 4` drops the L2 term to ~7 GB, i.e. below the DRAM floor, so decode
becomes DRAM-bound and Vulkan parity falls out for free. H2′ is a hypothesis, not a
result: it is falsified if P1's sweep shows no gain from multi-row blocks (then the
deficit is per-row latency/MLP and the lever is §5 P2, hipfire's occupancy posture).
`rocprof` is not installed, so P0 must measure the DRAM-shaped matvecs directly.

---

## 3. Root cause in our code

`ggml/src/ggml-cuda/mmvq.cu` (line numbers current as of this writing):

| line | symbol | RDNA2 state | RDNA3_0 / RDNA4 / Turing / GB10 state |
|---|---|---|---|
| 96–104 | `enum mmvq_parameter_table_id` | `MMVQ_PARAMETERS_RDNA2` **exists** | — |
| 111 | `get_device_table_id()` | `defined(RDNA2) \|\| defined(RDNA3_5)` → `MMVQ_PARAMETERS_RDNA2` | mapped by arch |
| 452 | `calc_nwarps()` | **no branch → `return 1`** | 8 (Q4_0/Q5_0/Q8_0/IQ4_NL/Q2_K/Q4_K/Q5_K/Q6_K/IQ4_XS on RDNA4), 8/2 on RDNA3_0, 2/4 on Turing, 2× on GB10 |
| 582 | `mmvq_two_rows_ncols1()` | Q1_0/Q2_K/Q3_K/IQ1_S/IQ1_M only | same |
| 595 | `calc_rows_per_block()` | **no branch → `return 1`** | 2 for `ncols_dst` 2–8; GB10 returns **2 for IQ2_S/IQ3_S** at `ncols_dst == 1` |
| 622 | kernel `__launch_bounds__(nwarps*warp_size, 1)` | 32 threads, **min 1 block/CU** | same macro, higher `nwarps` |

Dispatch order confirms this is a tuning gap, not a dispatch bug: in `ggml-cuda.cu`
(~2020) `ggml_cuda_should_use_mmvq()` is tested **before** `should_use_mmq()`, and for
non-CDNA AMD returns `ne11 <= MMVQ_MAX_BATCH_SIZE` (8) → **every decode matvec takes
MMVQ**, including Q4_K and the lm_head.

Reference posture (hipfire, `kernels/src/gemv_hfq4g256.gfx1030.v2.hip`):
`__launch_bounds__(32, 20)` — one warp per row, **20 waves per SIMD**, sized to ~51 VGPRs,
2× group unroll with dual scalar accumulators, plain f32 FMA. ggml's RDNA2 kernel is the
mirror image: `launch_bounds(32, 1)`, no unroll hint, no multi-row block. hipfire's
conclusion that their GEMV family is DRAM/occupancy-bound (issue #300: v1–v5 tie at
~58 t/s; HFQ3 v2 61.9 vs 62.6) describes **their already-tuned ceiling**, not ours.

---

## 4. hipfire comparison and issue audit

### 4.1 Kernel-by-kernel

| aspect | hipfire gfx1030 / gfx1100 | ours | verdict |
|---|---|---|---|
| gfx1030 attention | **none**; generic `attention_flash_fwht3_tile` with 3-bit FWHT KV | `fattn-gqa-gfx1030.cu`, Q8_0/Q5_0, TILE 256, CHUNK 24, 192 thr `(192,3)` | we are ahead; nothing to port |
| gfx1030 decode GEMV | `gemv_hfq4g256.gfx1030.v1..v5`, `launch_bounds(32,20)`, 1 row/block, 2× unroll, dual acc | ggml MMVQ, 1 warp, 1 row, `(32,1)` | **design idea only** (occupancy intent, unroll) |
| gfx1030 prefill GEMM | `gemm_q8_0_mmq.gfx1030.hip`: 32-row × 128-col tile, 256 thr, both operands LDS-staged, weights fetched exactly 1× per workgroup; plus `gemm_*hfq3g256_mmq_x{8,16,32}[_y{32,64,96}]` tile sweep over one body | ggml MMQ + `mmq-config-rdna2.cuh` | **nothing to gain** — our prefill is ~20× faster than their MQ3 path (556 vs 27.5 t/s, different prompt lengths) and 30 % faster than our own Vulkan |
| GQA attention design | 128-tile, raw buffer descriptors, V kept raw Q8_0 in 64-key LDS slabs, one max/sum per tile, always partials+reduce | 256-tile, sdot4, V dequantised to f16 once, online per-24-key softmax, direct write when `gridDim.y==1` | ideas only; ours already uses the better trade |
| RDNA2 KV format | refuses `q8` KV, uses FWHT-3 | Q8_0 K + Q5_0 V working | ours strictly better |

Their `gemm_q8_0_mmq` comment is the one transferable *lesson*: v3 "left every lane loading
its own row pair, so each weight block was fetched 8× per workgroup — weight traffic
unchanged vs v2". That is the same class of error as our activation re-read in §2.3, and it
is the reason P1 exists.

### 4.2 Issues (repo `warpfront/hipfire`)

| # | state | what it establishes for us |
|---|---|---|
| **#771** | closed | gfx1030 + qwen3.8:27b + dflash → `spec_step: HipError(0): qt=44 gemm_mq4g256v2_batched_lmhead: scalar fallback has no v2 source`; decode only works with dflash off (18.5–19.5 t/s). Same failure in installed v0.4.3.1 for MQ3 (`qt=49`). **Their spec decode is broken on this card** — our MTP lane (P5) has no competitor here. |
| **#478** | open | gfx1030 cannot serve `qwen3.5:9b-mq6`: `no implementation for GemvMq6G256Prerotated`, daemon panics. MQ6/HFQ6 is gfx11/gfx12-only. Confirms their RDNA2 support is a reduced-function path, not a tuned one. |
| **#533** | open | MQ2G256 has no batched prefill → per-token GEMV → **22× prefill gap vs llama.cpp** (88 vs 1923 t/s) on a gfx1030 RX 6800CT. llama.cpp is named as the reference implementation on this arch. |
| **#558** | open | Request to create a `perf/gfx1030` branch to land MQ2 prefill + Q8_0 MMQ + Redline replay (fork `rakaarwaky/hipfire:perf/gfx1030`). |
| **#300** | closed | Decode-tuning close-out: HFQ3 GEMV v2 with `(32,20)` + dual acc gave **61.9 vs 62.6 t/s** (no gain); rocprof showed no kernel >25 % of decode time; hipGraph, epilogue fusion, persistent kernel, MMQ_Y=256, VGPR reduction all falsified. Recommendation: spec decode / lower bpw. |
| **#490** | open | Their `mem_latency_ns` pointer-chase probe uses a fixed 128 MiB buffer, which **fits inside L2+Infinity-Cache on gfx1030**, so their committed DRAM latencies (148–276 ns) are cache latency and "the latency arm of the roofline `bound_class` is unreliable on all IC parts". Rated HIGH by their own review. |
| **#80** | closed | gfx1030 speed baseline on this exact card (RX 6900 XT, 16 GB, ROCm 7.2.2): 27B weights 13.95 GiB → **OOM at load**, so they cannot run 27B here at all; 9b decode 66.8 t/s, prefill 506–550 t/s; "no WMMA → FP16 packed dot product path". |

Net for us: (a) no gfx1030 attention or GEMV code worth porting, (b) their negative
occupancy result applies to a kernel that already had 20 waves, so it does **not** license
skipping P1, (c) **their "decode is DRAM-bound, tuning is exhausted" verdict rests on a
roofline they have themselves documented as cache-deflated on Infinity-Cache parts
(#490)** — the A/B tie in #300 stands, but the reason "we are at the DRAM roof" is not
established on this card, which is exactly the question H2′ asks, (d) their roadmap ends
at spec decode, which is also where our P5 is, and on this card their spec decode is
broken (#771), so P5 is uncontested ground.

### 4.3 Literature check

Searched arXiv full-text for `RDNA2`, `gfx1030`, `GEMV` + `occupancy`, `weight-only
quantization` + `GPU`, `memory-bound` + `decode`. Findings that change or confirm the plan:

| source | claim | does it transfer? |
|---|---|---|
| **arXiv 2605.30571**, *Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode* (2026-05) | batch-1 decode across H100/A100/L40S/L4: **the achieved fraction of peak bandwidth falls as peak bandwidth rises** (L4 ≈81 % of its analytic memory floor, H100 ≈27 %); CUDA Graphs gives **1.259×** on H100 but only **1.028×** on L4; and quantized paths often **fail to recover the 4× traffic cut** — 62.32 ms bf16 → 59.36 ms bnb-nf4 → 45.24 ms AutoAWQ+Marlin → **17.36 ms GPTQ+ExLlamaV2** | **Yes, on all three points.** (a) Our 359 GB/s = 70 % of 512 is *normal* for this class, so "we are bandwidth-bound" explains nothing; (b) three int4 kernels on identical byte counts spread **3.4×** — that is the same phenomenon as our HIP-vs-Vulkan 1.08–1.82× per type, i.e. the kernel decides whether the traffic reduction is realised; (c) their L4 arm (1.028×) is our regime — **launch overhead is not worth chasing**, confirming §6. |
| **arXiv 2206.09557**, *LUT-GEMM* | LUT-based quantized matmul beats dequant-then-multiply (2.1× over OPTQ at 3-bit) because *dequantization*, not arithmetic, is the cost | **Yes, as design validation.** Our IQ3_S/IQ3_XXS `vec_dot` already uses grid LUTs (`iq3s_grid`) + `dp4a`, i.e. the correct family. Direct consequence for P2: fix Q4_K **inside** integer/LUT math (`vdr`, dword loads, hoisted super-block `d`/`dmin`); never dequantize per token. |
| **arXiv 2311.01282**, *FlashDecoding++* | the *synchronized* partial-softmax update costs ~20 % of attention time; "unified max" removes the sync; flat GEMM wants double buffering; 2.18× on AMD | **Partly.** Our kernel rescales every 24 keys (2 barriers each). A per-tile (256-key) max with one rescale is the same idea at smaller scale — this is exactly P3, still low priority because HIP's attention *slope* is already better than Vulkan's (0.103 vs 0.111 µs/KV-token). |
| **arXiv 2603.10031** (the example cited by the user) | production study on 8× MI325X / vLLM: all four models saturate at a **memory-bandwidth bottleneck**; architecture-aware dispatch is essential (AITER helps MoE/MLA 3–5 % but multiplies variance 2–16×) | **Methodology only.** Confirms "architecture-specific kernel beats generic runtime switch", and that bandwidth saturation is the steady state. Its concurrency, MLA and TP findings do not apply to single-stream gfx1030/16 GB. |
| **null result** | arXiv has **no** paper matching `RDNA2` or `gfx1030`, and none matching `GEMV` + `occupancy` | There is no peer-reviewed RDNA2 decode-kernel literature. The authoritative primitives are Mesa `ac_gpu_info.c` (gfx10.3: 16 waves/SIMD, 512 wave64 VGPRs/SIMD ⇒ **≤64 VGPRs to reach 16 waves**, L1 128 KB/CU) plus our own A/B measurements. **Every item in §5 is therefore measurement-first and falsifiable; none is a literature transplant.** |

Net: the literature agrees with the shape of the plan (kernel realisation dominates,
launch overhead is a fast-GPU artefact, LUT/integer math beats dequant, softmax sync is
the attention tax) and supplies **no** short-cut that could replace P1.

### 4.4 Can the 128 MB Infinity Cache be used for something?

Searched arXiv for `Infinity Cache`, `last-level cache` + GPU + inference, `L2 cache`
persistence/residency, `KV cache` + cache hierarchy. The honest answer is **no, not for
the weights, and it is already load-bearing in exactly one place we are trying to
remove.**

The arithmetic first: Navi 21 has **128 MB IC / ~4 MB L2** (Mesa: `num_tcc_blocks ×
128 KB`) against a **11.64 GB/token** weight stream — a 90× shortfall — and each weight
byte is touched **once** per token and never again by that token. There is no reuse to
exploit, so IC cannot raise weight throughput; this is why decode lands at 359 GB/s of
512 and why the DRAM roofline *is* the right frame for the weight term. Cache-residency
work aimed at LLM state (below) cannot change that either.

Where IC genuinely participates in our decode:

1. **The activation re-read (H2′, §2.3).** `y` is 5.76 KB for the lm_head, so every one
   of the 248 320 row-blocks reads a fully IC/L2-resident vector. That is the one place
   where IC is load-bearing — and the correct response is to **reduce those reads**
   (`rows_per_block`), not to exploit them. IC bandwidth is a resource we are currently
   wasting ~26 GB/token of, not a resource we are failing to use.
2. **Prefill**, where each weight tile *is* reused across the ~15 k token batch; IC
   absorbs that reuse, which is part of why HIP prefill is already ~30 % ahead of
   Vulkan (§1.2) and why prefill needs no work.
3. **GQA attention** — reuse is served by LDS staging of the shared V head across 6
   query heads, which is faster than any cache.

Papers, and why none of them is a decode lever for us:

| source | claim | verdict for gfx1030 |
|---|---|---|
| **arXiv 2605.04178**, *Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures* | analytical models for B200 and **MI300A that explicitly capture the Infinity Cache hierarchy, VGPR constraints and occupancy**: **0.09 % MAE** on MI300A vs **>95 % error for naive roofline** on the same kernels | **Most valuable of the three — as a method, not a feature.** It is the quantitative version of hipfire #490: on an IC part, a roofline that ignores IC + VGPR + occupancy is not merely imprecise, it is *wrong by >95 %*. Directly justifies P0 (DRAM-shaped measurement) and the VGPR/occupancy focus of P1. |
| **arXiv 2609.02029**, *HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid Long-Context Language Models* | training-free, per-KV-head multilevel history windows for **hybrid** models; on **Qwen3.6-27B at 112K** cuts sampled peak device memory **8.59 %** and extends verified context 114K → **161K** at near-Full-KV RULER/LoCoMo quality | **The only paper that would change our deployment shape**, and the only lever that *grows* with depth (attention is 3.42 GB/token at 112k = 23 % of traffic). But it is a quality-affecting serving-layer feature (masking + grouped-cache runtime), large, and violates the "bit-identical output" constraint of the current plan → recorded as **P6, conditional**, not scheduled. |
| **arXiv 2512.00083**, *LLaMCAT* | LLC/IC-aware MSHR arbitration + thread throttling for **KV-cache** decode stalls; 1.26× (1.58× with small caches) | **Not implementable here** — it proposes cache/architecture policy, not a kernel we can write. Useful only as corroboration that KV-vs-LLC contention is a real decode bottleneck at long context. |
| other hits (`EXAM`, `SLAC`, `C2P-Cache`, `FluxMoE`) | Apple system-level-cache side channels; MoE expert residency; L1 sharing | irrelevant to a dense 27B single-stream decode on gfx1030 |

**Consequence for the plan:** IC changes nothing about P0–P2 (they target per-byte
efficiency and the IC-wasting re-read). It adds one measurement *warning* — both our
§2.2 sweep and hipfire #490 are IC-resident, so neither may be read as a DRAM number —
and one deferred idea (P6, §5).

---

## 5. Ranked plan

### P0 — Baseline at DRAM shapes (half day, no kernel change; gates P1)

The §2.2 bench is Infinity-Cache-resident. Add perf cases for the five real matvec shapes
to `tests/test-backend-ops.cpp` (`make_test_cases_perf()`), rebuild the HIP build
incrementally, and record µs:

| case | m | k | types |
|---|---:|---:|---|
| lm_head (exact) | 248 320 | 5 120 | q4_K |
| ffn_gate/up | 14 336 | 5 120 | iq3_s, iq4_xs, iq2_s |
| ffn_down | 5 120 | 14 336 | iq3_s, iq4_xs, iq3_xxs |
| attn_qkv | 9 216 | 5 120 | iq3_s |
| attn_output | 6 144 | 5 120 | iq3_s, iq4_xs, q4_K |

Run with `pkill -x llama-server` first (never `pkill -f` — it matches your own shell).
Vulkan numbers are optional here: P1 is a self-referential A/B sweep, so HIP-vs-HIP is
what decides. This step exists to (i) test H2′ cheaply — compare each shape's µs against
`bytes/0.42 TB/s` and `(w+y)/1.17 TB/s` — and (ii) give the sweep a baseline.

### P1 — Fill in the RDNA2 MMVQ table (main kernel work)

Only file: `ggml/src/ggml-cuda/mmvq.cu`.

1. Add `nwarps` for `MMVQ_PARAMETERS_RDNA2` in `calc_nwarps()` (line 452), modelled on the
   RDNA4 whitelist but starting conservative: `{q4_K, iq4_xs, q2_K, q6_K, q4_0, q8_0} → 2`.
   Types with table-lookup vec_dots (IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS) keep 1 initially — the
   RDNA4 comment records that raising threads for those regresses on register pressure and
   LUT contention.
2. Allow multi-row blocks in `calc_rows_per_block()` (line 595): add `MMVQ_PARAMETERS_RDNA2`
   to the accepted-table list and return 2 (later 4) at `ncols_dst == 1` for the same
   types. The GB10 precedent (`IQ2_S`/`IQ3_S → 2`) is the exact analogue: it exists there
   because those types "load few weight bytes per thread", which is also true at 3.4 bpw.
3. Thread a `bool rdna2_tuned` template parameter through the kernel so both variants are
   instantiated **only** under `#if defined(RDNA2)`, and pick it from a host-side cached
   `GGML_HIP_MMVQ_RDNA2=0|1` (default 1). Same pattern as `GGML_HIP_GQA_GFX1030`
   (`fattn.cu:531`). Other arches must not change at all.
4. Only if the sweep finds it necessary: give the RDNA2 kernel an explicit
   `__launch_bounds__` min-blocks (line 622) so VGPRs stay inside the multi-row budget
   (hipfire's ~51 VGPR / 20-wave posture is the anchor), and/or the 2× K-loop unroll.

Sweep matrix (recompile per config; only `mmvq.cu` changes, ~2–4 min incremental):

| id | nwarps | rows/block | launch_bounds | intent |
|---|---:|---:|---|---|
| A | 1 | 1 | (32,1) | baseline, current code |
| B | 1 | 2 | (64,4) | halve activation re-read |
| C | 1 | 4 | (128,2) | quarter it (H2′ maximum) |
| D | 2 | 1 | (64,4) | halve K trips per lane, no multi-row |
| E | 2 | 2 | (128,2) | both |
| F | 4 | 1 | (128,2) | latency hiding only |
| G | 1 | 1 | (32,16) | hipfire posture exactly |
| H | 1 | 2 | (64,8) | multi-row + high occupancy |

Types: `iq3_s, iq4_xs, iq3_xxs, iq2_s, q4_K`; controls `q6_K, iq2_xs` (must not regress).
Shapes: the five P0 cases.

**Accept:** ≥5 % µs gain on lm_head **and** on ≥2 of the four layer shapes, no type
regressing ≥2 %, and end-to-end d0 ≥ +3 % (30.87 → ≥31.8 t/s).
**Expected:** B–E give 5–20 % on the L2-bound shapes; if H2′ is right the lm_head moves
most and d0 lands 33–36 t/s. **Rollback:** `GGML_HIP_MMVQ_RDNA2=0`.

### P2 — Q4_K-specific dot product (only if P1 leaves Q4_K >1.3× behind)

In order of cost:
1. `VDR_Q4_K_Q8_1_MMVQ` as instantiated for RDNA2 — check whether vdr 2 or 4 halves the
   dot calls per thread without blowing the register budget.
2. 128-bit (`dwordx4`) loads of `qs` — one 16 B load per thread per super-block instead of
   four 4 B loads.
3. Hoist the Q4_K super-block `d`/`dmin` out of the eight sub-blocks: the current per-32
   dp4a path re-derives them per sub-block, ~8 scalar ops per 144 B.

Guardrail: every change `#if defined(RDNA2)`, validated with `test-backend-ops -o MUL_MAT`
across all K-quants.

### P3 — Low-depth attention occupancy (small, do last)

Only affects ≤8k contexts, ≤0.44 ms. At 8k, `4 kv_heads × ceil(k/256)` = 128 blocks vs 240
slots. Two launcher-only options: instantiate `launch_fattn<128,1,6>` and pick tile size
from `k_max` on the host; or force `gridDim.y = 2` below ~16k and use the existing
partials+combine path. Accept only if ≥0.2 ms at 4k or 8k. Do **not** restructure the
24-key chunk loop, and do **not** pre-quantise Q once per token: at 32k+ the kernel is
already at/below Vulkan's per-token attention cost, and the whole idea can buy ≤1 % of
end-to-end.

### P4 — Validation (lands with P1)

- `test-backend-ops -o MUL_MAT` for every touched type **and** the untouched controls;
- full `test-backend-ops -o FLASH_ATTN_EXT` (never re-run after the final sdot4/tail
  edits — outstanding debt from v1);
- `-DLLAMA_FATAL_WARNINGS=ON -DGGML_NATIVE=OFF` HIP build (also outstanding);
- one 5 000-token `llama-cli` coherence run plus the four depth points.

### P5 — MTP spec decode: **rejected** for the 112k deployment (kept as a conditional)

The GGUF ships a NextN layer (`blk.64.*`, Q6_K, 65 blocks) and the tree implements it
(`LLAMA_CONTEXT_TYPE_MTP`, `hparams.n_layer_nextn = 1`, `--spec-type draft-mtp`,
`common/speculative.cpp`). It was the headline recommendation in v2.1 and is now demoted.
Two independent costs, one of them measured:

**1. Context cost — hard, code-verified, not tunable.** `src/llama-model.cpp:2848`
allocates the MTP KV cache with `filter = il >= hparams.n_layer()` and
`cparams.n_ctx_seq` — i.e. **the full context length**, for one full-attention layer
(`blk.64` has real `attn_q/k/v/output` + FFN). Cost per token = (K + V) for 4 kv heads ×
256 dim = (1024/32)·34 B + (1024/32)·22 B = **1.75 KiB/token**, so at 112 128 tokens
**+201 MB ≈ +6.6 k tokens of lost context** on a 16 GB card that is already the binding
constraint (112k KV across 17 full-attn layers ≈ 3.4 GB). There is no flag to size the
MTP cache independently.

**2. Throughput at 112k — genuinely unproven, and the attention term is the reason.**
A verify pass processes k draft tokens against the whole prefix, and
`fattn-gqa-gfx1030.cu` only claims `Q->ne[1] == 1`, so **batch-k verification falls out of
our specialized kernel into the generic path** whose 112k cost is unmeasured (the ~11.5 ms
attention term at 112k is what that kernel was written to fix). Break-even in accepted
tokens per pass is (attention + 33.3 ms) / (11.5 ms + 33.3 ms); at the generic path's
plausible 2–3× attention penalty that is ~1.4–1.8 tokens per pass, i.e. per-token
acceptance ≳ 0.45–0.5. At `--temp 1.0 --top-p 0.95` on 3.5-bit weights with long CoT that
is not a safe assumption, and a failed pass is *slower* than baseline. hipfire's own
close-out reaches the same conclusion structurally and their gfx1030 spec path is broken
outright (#771).

**Conditional trigger:** revisit only if (a) context is dropped to ≤32k, where the
attention term is ~3 ms and break-even acceptance falls to ~0.15, or (b) a k-token
verification shape is added to `fattn-gqa-gfx1030.cu` — which is a bigger project than
P1+P2 combined. Do not spend P1's budget here.

### P6 — KV-read reduction at long context (**deferred, conditional**)

The only lever whose payoff *grows* with depth: at 112k the attention term is
**3.42 GB/token**, 23 % of the 15.06 GB read, and it runs at ~296 GB/s — worse than the
359 GB/s the weight path achieves. Per-head history windows for the 17 full-attention
layers (the *HeadWiseKV* approach, §4.4) are the published way to shrink that read set;
8.59 % peak-memory reduction and 114K → 161K verified context at near-Full-KV quality.
**Not scheduled** because it changes outputs (violates the bit-identical constraint that
the rest of this plan holds) and is a serving-layer feature, not a kernel. Revisit only
after P1/P2 land and only if the user accepts a quality/QALM tradeoff at long context.

---

## 6. Explicitly not doing

- **Not** improving prefill: HIP is 30 % ahead of Vulkan and 20× ahead of hipfire's gfx1030
  path at 15.4k; there is no gap. MMQ/GEMM work is out of scope.
- **Not** porting hipfire attention kernels: no gfx1030 GQA kernel exists, and their KV
  format is FWHT-3.
- **Not** porting hipfire's GEMV code: the quant formats differ (HFQ/MQ vs IQ/K-quants);
  only the occupancy/unroll intent transfers, and that is P1.
- **Not** chasing 20-wave occupancy as a goal in itself: hipfire measured that exact
  experiment flat-to-negative — but on a kernel that already had 20 waves, so it does not
  excuse skipping P1.
- **Not** MMQ or WMMA at batch 1: `should_use_mmvq()` is checked first, and batch-1 MMQ on
  RDNA2 is a known loss.
- **Not** blaming launch overhead: `GGML_HIP_GRAPHS=ON` is set in both builds and ~600
  launches are captured per token.
- **Not** shipping MTP spec decode: +201 MB KV at 112k (≈6.6 k tokens of context, not
  tunable) and the k-token verify pass escapes `fattn-gqa-gfx1030.cu` (`ne[1] == 1` only).
  Revisit only at ≤32k context or with a k-query kernel — see P5.
- **Not** touching `fattn-gqa-gfx1030.cu` before P1 reports.

## 7. Order of work

1. P0 DRAM-shape baseline (half day) → confirms or kills H2′.
2. P1 sweep A–H → winning config behind `GGML_HIP_MMVQ_RDNA2`.
3. P4 validation in the same change.
4. P2 / P3 only with leftover budget, in that order.
5. P5 stays closed: it trades ~6.6 k tokens of context for an unproven gain at 112k.

## 8. Files

- `ggml/src/ggml-cuda/mmvq.cu` — P1/P2; the only file the main lever needs
- `tests/test-backend-ops.cpp` — P0 perf cases, P4 coverage
- `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu`, `fattn.cu`, `fattn.cuh`,
  `ggml/src/ggml-hip/CMakeLists.txt` — v1 kernel, already landed
- `PLAN_GQA_DECODE_GFX1030.md` §9 — implementation record; §10 superseded (marked in place)
- `HIPFIRE_FINDINGS_GFX1030.md` — issue/code audit of record
- `HIPFIRE_27B_GFX1030_TEST.md` §3 — prefill comparison, now with the corrected numbers

## 9. Claims retracted / doc debt

| claim | where | now |
|---|---|---|
| "HIP prefill 195.6 t/s, 2.4× behind Vulkan, 10.2 vs 24.6 TFLOP/s" | earlier drafts of `HIPFIRE_27B_GFX1030_TEST.md` | **false** — unrepresentative `llama-cli` run. Correct: HIP 556 vs Vulkan 425 t/s at 15.4k, i.e. HIP ~30 % ahead. Fixed in §3. |
| "the residual 4–7 % gap is per-token fixed overhead inside the attention kernel" | `PLAN_GQA_DECODE_GFX1030.md` §10 | superseded: d0 shows the gap exists with no attention at all. Section is marked superseded in place. |
| "hipfire has a gfx1030 GQA/attention kernel we could compare against" | early task framing | **false** — 35 gfx1030 `.hip` files, all GEMM/GEMV; the ported reference kernel is gfx1100. |
| "MTP spec decode is the real multiplier, 40–55 t/s" (v2.1 §P5) | v2.1 | **rejected for this deployment.** Costs +201 MB KV at 112k (≈6.6 k tokens, `n_ctx_seq`-sized, not tunable) and the k-token verify pass escapes our `ne[1]==1` attention kernel. Gain unproven at `temp 1.0 / top-p 0.95`. |
| "rows_per_block=2 is worth it" (asserted in v2) | v2 §P1 | still a hypothesis; P0/P1 exist to test it. H2′ quantifies the expected size. |
