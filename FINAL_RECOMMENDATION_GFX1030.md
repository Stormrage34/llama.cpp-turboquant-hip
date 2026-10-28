# Final recommendation — gfx1030 (RX 6900 XT) decode

**Date:** Sun Oct 25 2026, updated Mon Oct 26 2026 (2-day gap-verification cycle)
**Hardware:** AMD Radeon RX 6900 XT, gfx1030, Navi 21, 80 CU, 16 GB, 128 MB Infinity Cache, 512 GB/s.
**Model:** Swift-1.5-Qwen3.8-27B (IQ3_S, 3.44 bpw, 27.32 B params, 11.64 GB/token, dense, no MoE).
**Production:** Vulkan server on port 9931, 112k ctx, `-ctk q8_0 -ctv q5_0 -fa 1`. *(Do not start this server from this host; benchmarking rules prohibit launching llama-server.)*
**Source of truth:** `PLAN_GFX1030_DECODE_V2.md` (v2.4). This file is the executive summary.

## Verdict in one paragraph

Prefill is **already ~30 % ahead of Vulkan** (HIP 556 vs 425 t/s at a 15.4k prompt; 25.1 % ahead) and needs no work. The decode gap (−7.4 % at d0, −8.5 % at 8k, ~−4 % at 112k) **survived a full 2-day verification cycle (2026-10-26)**: the matvec path runs at 386 GB/s = 76.5 % of the *measured* DRAM ceiling (504.3 GB/s), the residual is launch/tail at model k plus type-dependent kernel efficiency, and every kernel-side lever tested on this card is negative — including one pre-gated port of Vulkan's proven f32-activation design, which measured **5.35× slower** and was reverted. The single biggest remaining lever is **Q4_K** (lm_head is Q4_K [5120×248320], read every token, 1.61–1.82× behind Vulkan; P2, unattempted). **Deployment verdict: Vulkan for decode (34.24 vs 31.03 t/s pinned floor, +10.3 %), HIP for prefill.** MTP speculative decoding remains **rejected** for this deployment. Full scorecard in the 2-day verification cycle section below. *(Floor later re-pinned 2026-10-28 after the N1/N4/GQA/dual-matvec wave: 33.12 vs 31.22, +6.1 %; see "Binary integrity — RESOLVED".)*

## Measurements

### Decode depth sweep (`-fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -p 0 -n 64 -r 2`)

| ctx | HIP t/s | Vulkan t/s | gap |
|---|---:|---:|---:|
| 0 | 30.87 | 33.15 | −7.4 % |
| 8 192 | 29.28 | 31.76 | −8.5 % |
| 32 768 | 27.27–27.37 | 29.24 | −6.7 % |
| 112 000 | 22.16–22.47 | 23.23 | ~−4 % |

Depth fit: HIP slope 0.103 µs/KV-token + 33.3 ms fixed; Vulkan 0.111 + 30.7 ms. HIP's *slope* is better; the loss is a constant per-token weight cost.

### Per-type decode GEMV (`test-backend-ops`, `MUL_MAT`, m=4096 n=1 k=14336, µs/run, lower better)

| type | HIP | Vulkan | ratio | decode byte share |
|---|---:|---:|---:|---:|
| IQ3_S | 45.53 | 41.99 | 1.084 | 31 % |
| IQ4_XS | 28.46 | 22.00 | 1.294 | 21 % |
| IQ3_XXS | 40.96 | 36.25 | 1.130 | 20 % |
| IQ2_S | 44.23 | 35.67 | 1.240 | 9 % |
| Q4_K | 39.67 | 24.69 | **1.607** | 10 % |
| Q4_K (k=4096) | 14.89 | 8.17 | **1.822** | — |

Caveat: these tensors fit in the 128 MB Infinity Cache, so the absolute µs are IC-bandwidth-bound; the **ratios** isolate kernel efficiency. Byte-weighted ≈1.107 brackets the 7.4–8.5 % gap.

## 2-day gap-verification cycle (2026-10-26)

Scope: verify and attempt to close the surviving 6–11 % HIP-vs-Vulkan matvec gap on the deployed IQ3_S model. Steps: −1 (pin floor), 0 (large-k A/B), 2 (pipelining probe), 3 (decomposition + one bounded fix attempt), 4 (this consolidation). All thresholds pre-registered before runs.

### Corrected gate arithmetic

- Decode weight payload: **11.353 GB** = 12.109 − 0.407 (token_embd) − 0.349 (MTP). HIP 386 GB/s = 11.353 GB / 29.38 ms (e2e, graphs on).
- Measured DRAM ceiling: **504.3 GB/s** (14 GB streaming read, 2000 iters × 2) → 386 is **76.5 % of the measured ceiling**, not of the 512 GB/s spec.
- The old "2.77 µs hard bound" was **inverted**: 9.561 ms / 3454 nodes is the *observed* instrument cost in graphs-off mode, not an admissible cost.
- **The budget does not close in any mode.** Graphs on: 32.15 ms/token (31.10 t/s). Graphs off (`HIP_NODE_TIMING`): total 47.85 ms, Σ nodes 39.92 ms (matvec 30.39 + non-matvec 9.54), idle 6.24 ms (13.5 %), closure residual **16.6 %**. The graphs-off penalty itself is 15.7 ms/token (49 %) — all node-timing numbers are non-production-mode and labelled as such.
- VK closure test fails (predicted 28.52 vs measured 34.27 ms; residual 16.8 %) → "non-matvec is backend-equal" is **not** established.

### Large-k A/B (Step 0) — test-backend-ops, 3 interleaved reps, medians, hash-verified

| type | k | HIP µs | HIP GB/s | VK µs | VK GB/s | time ratio |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_S | 14336 (model k, control) | 46.32 | — | 42.18 | — | 1.098 |
| IQ3_S | 131072 | 516.86 | 446.3 | 534.27 | 431.8 | 0.967 |
| IQ4_XS | 131072 | 672.78 | 423.9 | 594.14 | 480.0 | 1.132 |
| Q4_K | 131072 | 711.59 | 424.4 | 647.89 | 466.1 | 1.098 |
| F16 | 131072 | 2082.91 | 515.5 | 2102.16 | 510.8 | 0.991 |
| IQ3_S | 262144 (≥3× IC control) | 1045.32 | 441.4 | 1050.52 | 439.2 | 0.995 |

Kernel selection verified on both sides (HIP `mul_mat_vec_q`; VK dmmv `_f32` shader, no tile/coopmat at n=1). GB/s computed from payload/time — the harness reports TFLOPS for MUL_MAT.

Findings:

- IQ3_S ratio **improves with k** (1.098 → 0.967 → 0.995): a **launch/tail problem, not a sustained-bandwidth ceiling**. The k=131072 "pass" (ratio ≥ 0.95) is regime-mismatched for the 7-iteration decode loop; the model-k ratio 1.098 ↔ 0.911 GB/s sits **inside the pre-registered 0.89–0.95 one-attempt band**.
- The gap is **type-dependent, not bit-width monotonic**: F16 (no unpack) 0.991, IQ3_S 0.967, Q4_K 1.098, IQ4_XS 1.132 → the "four types converge ≤ 7 %" pre-registration **fails**.
- The harness replays a captured 94-node graph, which favours HIP: the k=131072 pass is an upper bound on HIP's relative performance; the fails are strong evidence.

### Step 2 (pipelining) — cancelled on evidence, no kernel written

- Stock `mul_mat_vec_q` IQ3_S: **55 VGPR** (the 67-VGPR figure belongs to the `vec_dot_iq3_s_q8_1_rdna2` variant, not stock).
- `s_waitcnt vmcnt(0)` once in the function; vmcnt thresholds up to **7** → multiple weight loads already in flight per iteration of the kbx loop.
- The (nwarps=1, rpb=4) sweep point had already tested 4× independent weight loads in flight: 1.00033 vs 1.00041 winner — a wash. A register double-buffer is a **weaker instrument aimed at the same target**.
- `__launch_bounds__(32,16)` on stock: VGPR unchanged (55), **2.1 % slower** at k=131072. Reverted.

### Step 3 — decomposition + one bounded attempt

Per-op decode token (graphs off, 47.989 ms wall — non-production mode): MUL_MAT 30.458 ms (63.5 %), RMS_NORM 2.895, UNARY 1.328, SCALE 1.114, GET_ROWS 1.068, FLASH_ATTN_EXT 0.749, GATED_DELTA_NET 0.685, remainder 2.39 → non-matvec total **10.70 ms/token**. A 2× Vulkan advantage across all of non-matvec would be **5.35 ms, not 1–2 ms**; the old "1–2 ms" claim is plausible only for the RMS_NORM+GET_ROWS subset (2.14 ms → 1.07 ms) and is **unverified** (no VK perf cases in the harness for those ops).

Quantize isolation: **not separable** — `quantize_row_q8_1_cuda` is fused into the `mul_mat_vec_q` path, the node histogram counts it inside MUL_MAT, and the harness has no QUANTIZE_ROW q8_1 perf op.

The single pre-gated attempt — f32-activation IQ3_S matvec (`mul_mat_vec_iq3_s_f32_rdna2`), RDNA2-guarded, porting Vulkan's proven `mul_mat_vec_iq3_s_f32_f32` design (f32 activations, no per-matvec q8_1 pass):

- Correctness: **49/49** MUL_MAT IQ3_S cases pass.
- Perf: k=14336 = **225.54 µs vs 46.32 µs stock → 5.35× slower**. Gate (e2e ≥ 31.81 t/s AND k=14336 ratio ≤ 1.03) failed decisively. Reverted; .so hash restored.
- Interpretation: the f32/LDS-staged design **does not transfer to RDNA2** — the dp4a/q8_1 path is structurally superior for matvec on this card. Vulkan's decode advantage is not a design that can be ported.

### Pre-registration scorecard

| pre-registration | range | measured | status |
|---|---|---|---|
| HIP IQ3_S DRAM | 380–400 (390) | 386 (e2e, 11.353 GB / 29.38 ms) | ✓ in band |
| VK DRAM | 405–433 | 410–433 by subtraction | ✓ in band |
| e2e ratio | 1.05–1.10 | 1.103 (34.24/31.03); 1.098 @ k=14336 | ✓ (at the edge) |
| four types converge ≤ 7 % | ≤ 7 % | 0.967 / 1.132 / 1.098 / 0.991 @ k=131072 | ✗ fails — type-dependent |
| stop if HIP/VK ≥ 0.95 | — | 0.967 @ k=131072 → stop; model-k 0.911 in the one-attempt band | one attempt taken |
| accept ≥ +2.5 % vs pinned floor | 31.81 t/s | floor 31.03; attempt failed | ✗ not met |
| no fourth correct-but-non-transferring kernel | — | f32 port correct (49/49), 5.35× slower, reverted | ✗ violated — documented below |

### Baseline hygiene

- git `d327bf1f0780f87b8c01b367b61853b0c291d7c4`; 11 locally patched files + untracked hipfire/tuning set (see `git status`).
- HIP .so `a47dc1993fed91264d2742802337183cc4bbbfe81826409a3fd01f08e1054aa5`; VK .so `f9ec687444aab0c224a665e7a6ad4cb0ffc3fa86ea6e81c2bd9d4c41d881809b`.
- Tune header `mmvq-rdna2-tune.h` stock no-op **in the .so** (despite `scripts/.tune_state.json` claiming nw=2, rp=2 — the state file is stale).
- `GGML_HIP_GRAPHS=ON` (production). Clocks sclk 2485 MHz / mclk 1000 MHz; temps edge 55–58 °C, junction 59–61 °C, memory 64 °C.
- Pinned floor (5 paired medians, env-set state): **HIP 31.03 t/s, VK 34.24 t/s**. The env var behind the ~3 % floor band is still unidentified (pending); runs are labelled by observed state.

### Final deployment verdict (confirmed)

- **Decode: Vulkan.** 34.24 vs 31.03 t/s (+10.3 % at pinned floor). Residual composition: (a) IQ3_S launch/tail at model k — kernel-side levers exhausted (ablation flat, sweep wash, loads already in flight, launch-bounds negative, f32 port 5.35×); (b) type-dependent efficiency — Q4_K 1.607× at model k (lm_head, ~10 % of bytes) is the largest documented unattempted lever (P2); (c) non-matvec 10.70 ms/token (graphs-off) with no backend comparison measurable in the current harness.
- **Prefill: HIP** (+25.1 %).
- The f32-port failure is itself a finding: do **not** port Vulkan's matvec design to RDNA2; the dp4a path is the correct structure.

### Open items

1. Env var name behind the 3 % floor band (pending from user; runs labelled by state until supplied).
2. No VK per-op perf cases (RMS_NORM/GET_ROWS/UNARY/ADD) in test-backend-ops → the "2× advantage" claim stays unverified.
3. Graphs-off node-timing budget does not close (16.6 % residual) — instrumentation limitation, not a model property.
4. P2 (Q4_K-specific dot, 1.6× at model k) unattempted — future lane, outside the 2-day scope.
5. Upstreamability: `mmvq.cu` is CUDA-shared; RDNA2-guarded tuning is local-only. Deliverable is a local deployment, not an upstreamable patch.

## Root cause in code

File: `ggml/src/ggml-cuda/mmvq.cu`. Dispatch (`ggml-cuda.cu` ~2020) tests `should_use_mmvq` **before** `should_use_mmq`; for non-CDNA AMD it returns `ne11 <= MMVQ_MAX_BATCH_SIZE` (8), so **all decode matvecs use MMVQ**, including Q4_K/lm_head. The table `MMVQ_PARAMETERS_RDNA2` exists and `get_device_table_id()` selects it for gfx1030, but:

- `calc_nwarps()` (line ~452): RDNA4/RDNA3_0/Turing/GB10 branches exist; **no RDNA2 branch → `return 1`**.
- `calc_rows_per_block()` (line ~595): accepts only GENERIC/GCN/TURING/GB10; **RDNA2 falls through → `return 1`** (GB10 precedent returns 2 for IQ2_S/IQ3_S at `ncols_dst==1`).
- `mmvq_two_rows_ncols1()` (line ~582): only Q1_0/Q2_K/Q3_K/IQ1_S/IQ1_M.
- Kernel declares `__launch_bounds__(nwarps*warp_size, 1)` (line ~622) — min 1 block/SM lets hipcc spend VGPRs freely (Mesa: gfx10.3 → 16 waves/SIMD only if ≤64 VGPRs/thread).

Consequence of `rows_per_block == 1`: every block re-reads the whole quantised activation vector. For the lm_head that is 248 320 × 5.76 KB ≈ **1.4 GB of L2/IC traffic per token vs 714 MB of weights** — a plausible large share of the 1.6× Q4_K deficit.

## hipfire audit (repo `warpfront/hipfire`, 42 gfx1030 hits)

- **35** `*.gfx1030.hip` files in `kernels/src`, **all** match `gemm_*` or `gemv_*`. No gfx1030 attention/GQA kernel exists; the kernel we ported is the **gfx1100** sibling. Their KV format is FWHT-rotated 3-bit, not Q8_0/Q5_0.
- **#771** (closed): dflash on gfx1030 → `spec_step: … gemm_mq4g256v2_batched_lmhead: scalar fallback has no v2 source`; decode only with spec off (18.5–19.5 t/s); same failure in v0.4.3.1 for MQ3 (`qt=49`).
- **#478** (open): gfx1030 cannot serve mq6 → `no implementation for GemvMq6G256Prerotated`; MQ6/HFQ6 is gfx11/12-only.
- **#533** (open): MQ2 has no batched prefill → 22× gap vs llama.cpp (88 vs 1923 t/s).
- **#558** (open): request for `perf/gfx1030` branch (fork `rakaarwaky/hipfire:perf/gfx1030`).
- **#300** (closed): HFQ3 GEMV v2 `(32,20)`+dual acc = 61.9 vs 62.6 t/s (no gain); rocprof no kernel >25 %; hipGraph/fusion/persistent-kernel/MMQ_Y=256/VGPR reduction all falsified.
- **#490** (open, HIGH): their `mem_latency_ns` pointer-chase uses a fixed 128 MiB buffer that **fits in L2+IC on gfx1030**, so their DRAM latencies (148–276 ns) are cache latency; "the latency arm of the roofline `bound_class` is unreliable on all IC parts". **Their "DRAM-bound, exhausted" verdict is not established on this card.**
- **#80** (closed): gfx1030 baseline on this exact card: 27B weights 13.95 GiB → OOM at load; 9b decode 66.8 t/s; "no WMMA → FP16 packed dot product path".

Net: nothing to port; their negative occupancy result applies to a kernel that already had 20 waves, so it does **not** license skipping P1.

## Final recommendation

**Status after the 2026-10-26 cycle:** P0 done (large-k A/B above). P1 done in the prior cycle (sweep winner (2,2,lb=1) = 1.00041 — a wash; header is stock no-op in the .so). P2 **unattempted** (trigger met: Q4_K 1.607× at model k). The f32-activation port — the one untried structural lever, not in this P list — was attempted inside the 2-day budget and failed decisively (5.35×, reverted). P3/P4/P6 untouched. The 33–35 t/s expected end state is **not** reached; the pinned floor stands at 31.03 t/s.

1. **P0 — Baseline at DRAM shapes** (half day, no kernel change; gates P1). Add perf cases for the five real matvec shapes to `tests/test-backend-ops.cpp`, rebuild the HIP build incrementally, baseline µs, test H2′ (whether the multi-row sweep shows a gain).
2. **P1 — Fill in the RDNA2 MMVQ table** (main kernel work). `calc_nwarps`, `calc_rows_per_block`, plus a `bool rdna2_tuned` template param instantiated only under `#if defined(RDNA2)`, host-cached `GGML_HIP_MMVQ_RDNA2=0|1` default 1 (pattern from `GGML_HIP_GQA_GFX1030` at `fattn.cu:531`). Sweep matrix A–H over nwarps {1,2,4} × rows/block {1,2,4} × launch_bounds. Accept: ≥5 % on lm_head, ≥2 layer shapes, no type −2 %, d0 ≥ +3 %. Rollback via env.
3. **P2 — Q4_K-specific dot** (only if P1 leaves Q4_K >1.3× behind): `vdr`, dwordx4 `qs` loads, hoist super-block `d`/`dmin`.
4. **P3 — Low-depth attention occupancy** (small, do last): tile 128 or gridDim.y=2 below 16k, ≤0.44 ms, accept ≥0.2 ms.
5. **P4 — Validation** (lands with P1): full `FLASH_ATTN_EXT` suite (never re-run after the sdot4/tail edits), `-DLLAMA_FATAL_WARNINGS=ON`, 5k CLI run, 4 depth points.
6. **P5 — MTP spec decode: REJECTED** for the 112k deployment. Cost: `src/llama-model.cpp:2848` allocates the MTP KV cache at `cparams.n_ctx_seq` for one extra full layer → 1.75 KiB/token → +201 MB ≈ +6.6k tokens lost on a 16 GB card. Throughput at 112k unproven: `fattn-gqa-gfx1030.cu` only claims `Q->ne[1] == 1`, so batch-k verification leaves the generic path (the ~11.5 ms attention term). Break-even ≈1.4–1.8 tokens/pass → acceptance ≥0.45–0.50, not safe at `--temp 1.0 --top-p 0.95` on 3.5-bit weights. Plus hipfire #771 shows spec decode is broken on gfx1030 anyway. Revisit only at ≤32k (break-even ~0.15) or with a k-token verification kernel.
7. **P6 — KV-read reduction at long context (deferred, conditional)**: per-head history windows for the 17 full-attention layers (HeadWiseKV), 8.59 % peak-memory reduction, 114K→161K verified context. Changes outputs; revisit only after P1/P2 and only if a quality tradeoff is accepted.

**Expected end state:** 33–35 t/s at d0, 24–25 t/s at 112k. **Keep the Vulkan server as production until P1 + P4 both land.**

## Explicitly not doing

- Improving prefill (HIP 30 % ahead of Vulkan; 20× ahead of hipfire's gfx1030 path).
- Porting hipfire attention kernels (none exist for gfx1030) or GEMV code (quant formats differ).
- Pursuing 20 waves as a goal (hipfire's `(32,20)` is an RDNA1 number; gfx10.3 caps at 16 waves/SIMD, needing ≤64 VGPRs).
- Chasing launch overhead (2605.30571: 1.028× on L4-class cards).
- Batch-1 MMQ, WMMA on gfx1030, or touching `fattn-gqa-gfx1030.cu` unless P3.
- MTP (P5).
- Register double-buffer in the kbx loop (cancelled on ISA evidence: stock 55 VGPR, vmcnt up to 7 in flight, rpb=4 sweep wash, launch-bounds −2.1 %).
- Porting Vulkan's f32-activation matvec design to RDNA2 (attempted 2026-10-26: correct, 5.35× slower, reverted — the fourth correct-but-non-transferring kernel, documented in the scorecard).
- Using the 128 MB Infinity Cache as a lever for weights (90× shortfall, single touch per token). IC only matters for the activation re-read (the H2′ term — reduce it) and prefill (already ahead).

## Literature (arXiv)

| ID | claim | transfers? |
|---|---|---|
| 2605.30571 | batch-1 decode: achieved bandwidth fraction falls as peak rises (L4 ~81 %, H100 ~27 %); quantized paths often miss the 4× traffic cut (62.32 → 17.36 ms) | Yes — our 70 % is normal; the 3.4× int4 spread mirrors our per-type ratios; launch overhead is an L4 artefact. |
| 2206.09557 (LUT-GEMM) | LUT math beats dequant (2.1×) | Yes — validates IQ LUT+dp4a; P2 stays in integer math. |
| 2311.01282 (FlashDecoding++) | synchronized partial softmax ≈20 % of attention; unified max removes it | Partly — same idea as P3, still low priority. |
| 2603.10031 | MI325X/vLLM: bandwidth saturation; architecture-aware dispatch | Methodology only. |
| 2605.04178 | IC+VGPR+occupancy model: 0.09 % MAE vs >95 % roofline error | Yes — justifies P0 and the VGPR focus. |
| 2609.02029 (HeadWiseKV) | per-head windows on Qwen3.6-27B at 112K: −8.59 % peak memory, 114K→161K | P6 only (changes outputs). |

Null result: no arXiv paper matches `RDNA2` or `gfx1030`, and none matches `GEMV` + `occupancy`. No peer-reviewed RDNA2 decode-kernel literature exists; the primitives are Mesa `ac_gpu_info.c` + our measurements.

## Retracted claims

- "HIP prefill 2.4× behind Vulkan / 10.2 vs 24.6 TFLOP/s" — retracted; HIP is 30 % *ahead* (556 vs 425 t/s). Origin: unrepresentative `llama-cli` run.
- "MTP is the real multiplier, 40–55 t/s" — rejected; context cost + unproven gain at 112k.
- "30 gfx1030 .hip files" — corrected to 35.
- "DRAM-bound and exhausted" (hipfire #300) — their roofline is cache-deflated on IC parts (#490); the A/B tie stands, the explanation does not.
- "Prefill is 7.1× slower" — corrected to ~20× slower (27.50 vs 556 t/s) for the 920-token prompt.
- "Residual 3.19 ms" / "2.77 µs hard bound" — undefined: the budget does not close in any mode (16.6 % graphs-off residual); 9.561 ms / 3454 nodes is the observed instrument cost, not an admissible bound; the 29.38 ms and 9.561 ms figures come from different graph modes.
- "67-VGPR stock kernel" — stock `mul_mat_vec_q` IQ3_S is **55 VGPR**; 67 belongs to the `vec_dot_iq3_s_q8_1_rdna2` variant.
- "2× Vulkan advantage worth 1–2 ms" — unverified: 2× across all non-matvec (10.70 ms) would be 5.35 ms; plausible only for the RMS_NORM+GET_ROWS subset.
- Absolute 405 GB/s gate — replaced by a ratio gate; and even the k=131072 ratio gate is regime-mismatched for the 7-iteration decode loop (model k=14336 is the representative point).

## Files

- `PLAN_GFX1030_DECODE_V2.md` — full plan (v2.4).
- `HIPFIRE_27B_GFX1030_TEST.md` — hipfire 27B test (§3 corrected).
- `FINAL_RECOMMENDATION_GFX1030.md` — this file.
## Addendum — ALU-ablation proof that decode is memory-stalled (2026-10-26)

The decisive experiment: replace the IQ3_S / IQ3_S decode kernel with a stub that
issues **exactly the same memory traffic** (8 B `qs`, 1 B `qh`, 4 B `signs`, 32 B
activations per 32 weights) but deletes **all** unpack arithmetic — the 8
`iq3s_grid` LUT gathers, every index-build/sign-mask bit-op, `__vcmpne4`/`__vsub4`,
and all 8 `v_dot4c` (dp4a). Every loaded value is folded into one XOR accumulator,
so logits are numerically wrong by design and the correctness gate is skipped.

Measured on the production model (`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`,
`-ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -p 0 -n 128 -d 0`):

| kernel | reps | tg128 t/s | sd |
|---|---:|---:|---:|
| stock IQ3_S | 8 | 31.121 | 0.294 |
| IQ3_S with all unpack ALU deleted | 8 | 31.121 | 0.294 |

**Delta: 0.000 t/s.** IQ3_S consumes 41 % of decode wall time, yet its arithmetic
costs nothing measurable — the kernel is stalled on memory, not ALU.

### Consequences

1. **Every ALU-side optimisation is capped at ~0.5 % end-to-end.** The planned
   `v_bfe`/`perm` unpack rewrite (projected 448 -> ~420 static instructions,
   bit-ops 152 -> ~120) cannot close a 10 % gap; it was not built.
2. This retro-explains the entire experiment ledger:

| experiment | isolated effect | end-to-end |
|---|---|---|
| Q4_K VDR=8 wide load | 40.61 -> 26.76 us (-34 %) | 31.13 (flat) |
| Q4_K + VGPR shave | 65 -> 63 VGPR, 12 -> 16 waves | 31.13 (flat) |
| IQ3_S VDR=4 wide load | wider loads, 63 VGPR / 16 waves | 30.84 / 30.11 (worse) |
| occupancy / rows / launch_bounds sweep | 7 configs | 28.13-30.80 (worse) |
| HIP VMM (`GGML_HIP_NO_VMM=OFF`) | VMM active | 31.132 (flat); prefill aborts |
| IQ3_S ALU ablation | all unpack work removed | 31.121 (flat) |

3. The remaining gap is a pure achieved-bandwidth difference in the weight stream:
   **HIP 377 GB/s vs Vulkan 417 GB/s** (12.11 GB/token), and it is not reachable
   from kernel code, occupancy, page granularity, or unpack arithmetic.

### Why the HIP VMM path is unusable here

`GGML_HIP_NO_VMM=OFF` engages VMM (`VMM: yes` at init) and decodes at the same
speed, but any prefill-sized workload aborts:

```
ggml-cuda.cu:705: HipVMM Failure: invalid argument
  ggml_cuda_pool_vmm::alloc <- ggml_cuda_mul_mat_q <- ggml_cuda_graph_evaluate_and_capture
```

## Phase 1 — hipfire non-kernel archaeology (submission / sync / allocator)

Full ledger: `/tmp/opencode/hipfire_nonkernel_ledger.json` (22 findings, id/ref/layer/summary/transferable/transfer_note).
Summary: `/tmp/opencode/hipfire_archaeology.md`.

### Direct answer to "does hipfire bypass hipGraph with raw queues?"

Yes, but **only on the experimental Redline route, and it is not the winning transport**:

- `crates/redline` (research-only) talks to `/dev/dri/renderD128` via `libdrm_amdgpu` — raw KMD/PM4 submission.
- Product Redline (`redline-dispatch` + `redline-rocr`) records ordinary HIP launches into a typed tape and submits the whole decode as **one PM4 indirect-buffer AQL packet through the public ROCr/HSA queue** (single doorbell).
- **Amendment 7** measured that PM4 backend at **−8% vs default hipGraph replay** (byte-identical output) and concluded "PM4 retained-replay bookkeeping costs more per node than hipGraph dispatch" — **hipGraph replay is optimal for this decode path**.

Conclusion for our fork: keep `GGML_HIP_GRAPHS=ON` (already production). The residual ~3.2 ms/token is **not** per-launch overhead in graph mode (graphs already recover the submission component, +4.4%). hipfire's transferable value is in the **micro-optimizations** around kernarg placement, synchronization coalescing, and allocator discipline — not in replacing the submission abstraction.

### Ranked transferable ideas (with evidence)

1. **VRAM-resident kernargs (med-high, +3–7% potential).** hipfire moved kernarg segments from host memory to a host-writable GPU-agent (VRAM) pool: +3.3% R9700, +7.2% 7900 XTX, +0.2% Halo. Kernels whose prologue chains dependent kernarg `s_load`s paid host-memory latency on every replay. Per-replay patches stay host stores through the BAR, published by one SeqCst fence + one-byte readback before the doorbell (commits `3bf6a0e96`, `bde59a078`). **Action:** determine whether ROCm places ggml kernel arguments in device or host memory on gfx1030; if host, force a device-resident pool / pre-stage. Highest-leverage untried non-kernel lever.
2. **Resource-derived wait reduction (med, +2.3% potential).** Derive compute-idle waits from allocation-wide read/write effects instead of a kernel-name allowlist; removed 50 waits from the 833-dispatch tape (+2.34%) (`redline-pm4-resource-waits.md`). **Action:** audit `ggml-backend-hip` and `ggml-cuda` for redundant `hipStreamSynchronize` / event waits / dependency edges in both graph and non-graph paths — our graphs-off idle is 6.24 ms/token (13.5%).
3. **Fence coalescing for host transfers (med).** One fence per host transfer op instead of per-buffer (`d241bfe9f`). **Action:** coalesce synchronization around `quantize_row_q8_1` / activation uploads into one fence per transfer batch.
4. **Allocator discipline for graph capture (med).** hipMalloc during stream capture = hipError 900 (issue #36); hoist allocations out of captured regions. **Action:** keep all allocations/VMM reserve outside captured regions; do not attempt VMM for the compute buffer on this card (already negative — see "Why the HIP VMM path is unusable").
5. **`GPU_MAX_HW_QUEUES=1` probe (low-med).** Cap hardware queues on RDNA4 to avoid stochastic MTP+PCIe-offload loss (`d0eb815be`). **Action:** probe on gfx1030; unverified on our card, low risk.
6. **Clock/DPM (null).** MCLK reaches top state within ~8 s and holds; forcing performance-level high changed nothing (`8042bf2e8`). Not a lever — clocks already maxed. Do not pursue.
7. **PM4-level micro-opts (low):** fence specialization (+1.16%), stateful dirty gate (22891→18090 dwords), NOP pacing (+3%, gfx12 only), GCR trim (gfx12 only) are PM4-encoding- or gfx12-specific and do not transfer to the hipGraph path. Context-bucketed tapes and dispatch interleave were neutral/negative.

### Null results (do not spend effort)

- MCLK/DPM (clocks already maxed).
- Dispatch interleave (null), L2-skip (−0.48%), context-bucketed tapes (neutral).
- PM4 retained-replay vs hipGraph (−8%).
- VMM compute buffer (unusable on this card; `cuMemSetAccess` failure at `ggml-cuda.cu:705`).
- VRAM kernargs on Halo (+0.2% only) — gain is card-dependent; measure on gfx1030 before committing.

### Caveat on hipfire's roofline claims

Issue #490: hipfire's `mem_latency_ns` pointer-chase uses a fixed 128 MiB buffer that fits in L2+IC on gfx1030, so reported "DRAM" latencies (148–276 ns) are cache latency; its "DRAM-bound, exhausted" verdict is not established on IC parts. **Our own 76.5%-of-ceiling claim rests on a 14 GB streaming read**, so it survives — but it cautions against trusting hipfire's roofline on this card.

---

## Phase 2 / 3 — results (chunk sweep + DPM telemetry)

Zero-build. 15 llama-bench runs: HIP chunks 0/128/256/512 × 3 reps + VK default × 3 reps, 112k depth, production flags. Raw logs: `/tmp/opencode/sweep/*.jsonl`, `/tmp/opencode/dpm_*_chunk*.log`.

### Chunk sweep (t/s, 112k depth)

| Backend | Chunk | Rep1 | Rep2 | Rep3 | Median | ms/token |
|---|---:|---:|---:|---:|---:|---:|
| HIP | 512 (default) | 23.106 | 18.422 | 23.145 | 23.11 | 43.29 |
| HIP | 256 | 23.305 | 23.134 | 23.114 | 23.13 | 43.23 |
| HIP | 128 | 23.089 | 22.839 | 23.027 | 23.03 | 43.42 |
| HIP | 0 (single graph) | 23.015 | 23.080 | 23.054 | 23.05 | 43.38 |
| VK | default (control) | 23.050 | 23.081 | 22.987 | 23.05 | 43.38 |

Vulkan does not read `GGML_CUDA_GRAPH_CHUNK_SIZE` (no CUDA-style graph-chunk capture path); run as a control at default.

**Verdict (i): NULL.** chunk=0 does not recover ms/token. All HIP medians within 0.10 t/s (≈0.19 ms/token); chunk=0 (23.05) is slightly *below* chunk=512 (23.11). The 18.422 rep is an outlier (thermal/power dip) but the median is unaffected. Hypothesis rejected.

### DPM / clock telemetry

SCLK during load:

| Chunk | Samples | Mean MHz | Min MHz | Max MHz |
|---|---:|---:|---:|---:|
| 0 | 1453 | 2485 | 2485 | 2485 |
| 128 | 1511 | 2485 | 2485 | 2485 |
| 256 | 999 | 2485 | 2485 | 2485 |
| 512 | 1085 | 2476 | 2245 | 2485 |
| VK | 1007 | 2485 | 2485 | 2485 |

MCLK: `pp_dpm_mclk` has no active-state marker (`*`) on this card, so the poller could not identify the active memory-clock level; all levels logged as empty. Power: `power1_input` does not exist on this card; `power1_average` is available but was not polled. Clock data above is valid.

**Verdict (ii): PARTIAL NULL.** A clock divergence exists but only for chunk=512, where sclk dips to 2245–2485 MHz (mean 2476) during the long run. Chunks 0/128/256 and VK hold 2485 MHz throughout. Consistent with thermal/power throttling over the longer wall time of the chunk=512 sequence, not with chunk size itself. Does not explain the decode gap: at 112k depth HIP (23.05–23.13) and VK (23.05) are effectively identical. The gap remains a d0 / launch-tail issue, not a long-context or DPM issue.

### Interpretation

At 112k context the backends are at parity (≈23.05 t/s both). The gap is concentrated at d0 (model k=14336): HIP/VK time ratio 1.098 at k=14336 vs 0.967 at k=131072 — i.e. launch/tail, not sustained bandwidth. Graph chunking (host-side graph-launch boundaries) and DPM/clocks are ruled out. This is consistent with the kernel-side conclusion that the matvec is memory-stalled at 76.5% of the measured ceiling and the residual is not recoverable from submission granularity.

---

## Phase 4 — Synthesis & Action Gate

### Stop/Go evaluation

| Probe | Outcome |
|---|---|
| Phase 2.2 chunk sweep (0/128/256/512) | NULL — no ms/token recovered |
| Phase 3.1 DPM/clocks | NULL for causation (thermal dip on chunk=512 only; not causal) |
| Phase 1 hipfire archaeology | hipGraph confirmed optimal (PM4 −8%); transferable micro-levers identified (VRAM kernargs, waits, fences) |
| Kernel-side levers (f32 port, launch_bounds, ALU ablation, wide-load, VMM) | All exhausted or negative (f32 port 5.35× slower) |

**Decision: the submission-layer and DPM hypotheses are closed (null).** The remaining, untried levers are the hipfire micro-optimizations and the unaddressed P2 lane. These are **not** null results — they have not been attempted on this card.

### Remaining levers (ranked, with expected gain)

1. **VRAM-resident kernargs** (med-high, +3–7% potential) — ~~highest untried non-kernel lever.~~ **CLOSED for the production hipGraph path**: hipamd's `GraphKernelArgManager::AllocGraphKernargPool` allocates the graph's kernarg pool in VRAM (`deviceLocalAlloc`) when `largeBar_` is true, which holds on this card. hipfire's win was on retained-PM4 replay (−8% vs hipGraph per Amendment 7). At most helps the graph-disabled/test path; measure `HIP_FORCE_DEV_KERNARG=1` as a control only. See `/tmp/opencode/kernarg_placement.md`.
2. **Resource-derived wait reduction** (med, +2.3% potential) — audit `ggml-backend-hip`/`ggml-cuda` for redundant `hipStreamSynchronize`/event waits/dependency edges; graphs-off idle is 6.24 ms/token (13.5%).
3. **Fence coalescing for host transfers** (med) — coalesce sync around `quantize_row_q8_1`/activation uploads into one fence per batch.
4. **Q4_K lm_head 1.6×** (P2, unattempted) — the lm_head is Q4_K at 1.6× the IQ3_S weight-stream cost; unaddressed.
5. **`GPU_MAX_HW_QUEUES=1` probe** (low-med) — low risk, unverified on gfx1030.
6. **Non-matvec** (10.70 ms/token, unverified "2× Vulkan" claim) — no VK per-op harness cases exist; cannot be attributed without them.

### Binary integrity — RESOLVED (re-pinned)

Re-pinned on 2026-10-28 against the current tree (post N1/N4/GQA/dual-matvec). Builds were flaky (failing non-ECC RAM: clang segfault / glslc exit -1 on first attempts) but succeeded on retry with `-j4` and reproduced identical hashes on a clean rebuild.

- **HIP hash** (`libggml-hip.so.0`): `d044065601cfb981dad05bc991c84ac1289500e403f120cf41e00683f172f23f` — **stable**.
- **VK hash** (`libggml-vulkan.so.0`): `9f2f2963d78a9cddc9e1d821d1a967a638bb7613884aed74e0ae91221989a4a5` — **stable**.

New d0 floor (3 interleaved reps each, median, `-p 0 -n 128 -d 0 -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0`):
- **HIP: 31.22 t/s** (31.22, 31.22, 31.24)
- **VK: 33.12 t/s** (33.13, 33.09, 33.12)
- Gap: +6.1 % VK (down from +10.3 % on the pre-implementation tree).

Earlier floors (31.03 / 34.24, and the a47… / f9ec… hashes) were measured against the pre-implementation tree and are superseded; the sweep's 23.05 t/s was at 112k depth on a different binary and is not comparable to this d0 floor. All future numbers reference the hashes above.

### Open items (carried forward)

1. **Env-var name** behind the ~3% floor band (env-set 31.03 vs env-unset 30.10–30.45) — still unnamed; runs are labelled by state.
2. **VK per-op harness cases** — needed to validate the "2× Vulkan advantage" non-matvec claim.
3. **Graphs-off budget closure** (16.6% residual unexplained).
4. **P2 Q4_K lm_head lane** — unattempted.
5. **Binary re-pinning** — blocking (see above).
6. **Upstream-ability** — RDNA2 tuning is local-only; `mmvq.cu` is CUDA-shared, so any change needs a CUDA regression check.

### Final recommendation (unchanged, now evidence-locked)

- **Decode: Vulkan** (33.12 vs 31.22 t/s, +6.1%; at 112k depth the two are at parity).
- **Prefill: HIP** (+25.1%).
- Kernel side exhausted; submission/DPM hypotheses closed (null); VRAM-kernargs lever closed for the hipGraph path; remaining levers are wait reduction (+2.3%, graphs-off idle 6.24 ms), fence coalescing, Q4_K lm_head 1.6× (P2), and the GPU_MAX_HW_QUEUES=1 probe — all gated on the now-resolved binary pin (d044… / 9f2f…).
