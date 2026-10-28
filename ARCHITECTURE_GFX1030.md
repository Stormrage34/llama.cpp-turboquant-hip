# gfx1030 decode: what the memory system can and cannot give us

Architecture assessment. Read this before proposing any cache, bandwidth or
quantization work on this card.

## The governing fact: decode cost tracks elements, not bytes

Three FFN matrices of one layer, identical element count (89.129 M), three quant
types, byte counts spanning 39 %:

| shape (M×K) | type | bytes | time | GB/s | Gelem/s |
|---|---|---:|---:|---:|---:|
| 17408×5120 | IQ3_XXS | 34.12 MB | 106.0 µs | 322 | **841** |
| 17408×5120 | IQ3_S | 38.30 MB | 99.5 µs | 385 | **896** |
| 5120×17408 | IQ4_XS | 47.35 MB | 98.6 µs | 480 | **904** |
| 248320×5120 (lm_head) | Q4_K | 715.2 MB | 1393.6 µs | 513 | **912** |

+39 % bytes buys −0.9 % time. Across a 14× size range, three quant families and two K
values, throughput is **841–912 Gelem/s (±4 %)**.

`GB/s = Gelem/s × bpw / 8`. So the model is not reading 420 GB/s because the memory
system is inefficient — it is retiring **~1.0 Telem/s**, and 420 GB/s is what that costs
at 3.26 bpw.

This also explains the one result that had no explanation: the ALU ablation showed decode
arithmetic is free (+0.38 %), yet the narrowest type is the slowest. It is slowest
**because it is narrowest** — same elements, fewer bytes, so the same time divided by
fewer bytes.

**Consequence: lowering the quant width cannot speed up decode below ~4.42 bpw.** The
byte ceiling binds when `Gelem/s × bpw/8 = 504.3`, i.e. at bpw ≈ 4.42. IQ2/IQ3/IQ4 all
cost the same per element. Any requantization campaign to buy decode speed is dead
(E1 tests this directly).

## Where reuse exists

**(a) Batching — the only real source, and already implemented in the kernel.**
`mmvq.cu:756` is the K loop; the token loop at `mmvq.cu:783-796` passes a weight address
that is **invariant in `j`**:

```cuda
for (int j = 0; j < ncols_dst; ++j)
    tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby],
                                kbx_offset + i*stride_row_x + kbx, kqs);
```

`vec_dot_q_cuda` is a `constexpr` function pointer (`mmvq.cu:656`) so it inlines and the
load is hoistable. Measured at n=2: 1396.28 vs 1393.61 µs, **+0.19 % for 2× the tokens**.
`MMVQ_MAX_BATCH_SIZE` is 8 (`mmvq.cuh:3`), so **k ≤ 8 needs no kernel work at all**.

Knee: n=4 → +20.1 % (3.33× throughput), n=8 → +187.7 % (2.78×). Cause is the activation
working set per block: a quantized `y` row is 5.76 KB at K=5120, 19.6 KB at K=17408; four
tokens = 23 KB (fine), eight = 46 KB (spills).

**(b) Intra-token activation reuse — already at ~100 %.** `y` would cost 28.8 GB of
potential traffic per token (1.125 B per weight element over 25.6 G elements) against
10.414 GB of weights. Uncached that alone would be ≥57 ms; measured matvec is 24.785 ms.
The cache already absorbs all of it. Nothing to win here.

**(c) Cross-kernel intermediates — negligible traffic, only launch count matters.**
1,788 dispatches × ~40 KB ≈ 71 MB/token = 0.14 ms at 504 GB/s, against a 2.125 ms
non-kernel gap. The gap is **15× the traffic time**: dispatch latency, not DRAM round-trips.

**(d) Retaining weight lines across tokens — arithmetically impossible.** Reuse distance
is 10.414 GB against 128 MB, an **81.4×** oversubscription. Best case with random
replacement: 1.23 % hit, worth **0.19 ms (0.6 % of the token)**. Not a programme.

**(e) The asset that is genuinely unused: the KV cache fits in Infinity Cache at 4k.**
At ctx 4096 with q8_0 K / q5_0 V, the KV cache is **117.4 MB**, inside the 128 MB.
Crossover is **~4,570 tokens**. Below that, the only DRAM stream in the token is the
weights — which is exactly why ≤4k is the ideal batching regime.

## The 420.6 vs 512 GB/s shortfall is not a cache problem

| hypothesis | verdict | evidence |
|---|---|---|
| L2/IC pollution from dead weight lines | **falsified** | lm_head is the most polluting stream in the model (715 MB, every line dead) and runs at 513 GB/s = 100 % of nominal, 101.7 % of the 504.3 copy ceiling. The victim would be `y`, which is per-CU L1-resident (5.76 KB shared by all blocks). |
| slice / channel imbalance | **falsified** | same datum — one linear 715 MB range through the same address hash at 100 % of peak. |
| request pattern | not separable, not worth separating | all four shapes are single contiguous tensors; no stride to camp on. Activation-row strides live in the 3.195 ms small-op budget, ≤0.5 ms upside. |
| DRAM efficiency / latency equilibrium | **most consistent** | the kernel stalls ~99 % of issue slots but cannot convert it into bandwidth by adding bytes-in-flight: every MLP lever tried (VDR↑, rows/block↑, nwarps↑, `__launch_bounds__`, register double-buffer) is flat or negative in situ, while isolated wins did not transfer. |

Stated cleanly: **the memory system does 513 GB/s on this exact kernel.** At 3.26 bpw the
kernel only *asks* for 420. It retires ~1.0 Telem/s. The 17 % is bandwidth never
requested, not lost.

## Batching arithmetic

Pass-time multipliers, measured: a(2) = 1.0019, a(4) = 1.201.

| | k=1 | k=2 multi-slot | k=4 multi-slot | k=4 spec, α=0.7 |
|---|---:|---:|---:|---:|
| matvec | 24.79 | 24.83 | 29.77 | 29.77 |
| small ops (no amortization) | 3.20 | 6.39 | 12.78 | 4.15 |
| attention + GDN | 1.31 | 2.62 | 5.24 | 2.2 @4k |
| non-kernel gap | 2.13 | 2.13 | 2.13 | 2.13 |
| **pass total ms** | 31.41 | 35.97 | 49.92 | 38.3 @4k |
| tokens/pass | 1 | 2 | 4 | 2.53 |
| **ms/token** | 31.41 | **17.98** | **12.48** | 15.1 @4k |
| **per-sequence t/s** | 31.8 | **55.6** | **80.1** | 66 @4k |
| **aggregate t/s** | 31.8 | **111** | **320** | 66 |

Per-slot state: KV 117.4 MB at 4k / 3.211 GB at 112k; recurrent state **156.9 MB,
context-independent** (48 GDN layers, `n_embd_r` 30,720 + `n_embd_s` 786,432 × 4 B).

VRAM with 11.353 GB of weights on 16 GB: **k=4 fits to ~26k, k=2 to ~55k; k=2 at 112k is
18.1 GB — infeasible.** The hardware rewards multi-slot; VRAM permits it only at short
context.

Two cautions the arithmetic exposes:
- Multi-slot pays a **0.63 GB gather per k=4 pass** because the direct-state path in
  `build_rs` only applies at `n_seqs == 1`. Spec decode avoids this: one sequence, one
  state, and `n_tokens` loops inside the kernel.
- Batching's limit is **not** the weight stream. At k=4 the non-matvec terms are 18.0 of
  49.9 ms and do not amortize.

## Ceilings

| ceiling | value | property of |
|---|---:|---|
| single-stream hardware (10.845 GB @ 504.3 GB/s) | **21.5 ms → 46.5 t/s** | hardware |
| single-stream realistic (byte-rate matvec, small ops unchanged) | 26.8 ms → 37.3 t/s | software; needs the element-rate stall removed — 7 attempts failed |
| **today** | **31.41 ms → 31.8 t/s** | — |
| k=2 measured | 17.98 ms/token → 55.6 t/s/slot (111 agg) | floor 10.75 → 93 t/s/slot |
| k=4 measured | 12.48 ms/token → 80.1 t/s/slot (320 agg) | floor 5.38 → 186 t/s/slot |
| matvec GB/s | 504.3 hardware; 428–470 reachable (+2…+12 %) | 504.3 hardware, the rest software |

## Experiments, in order

| # | hypothesis | measurement | predicted | gates |
|---|---|---|---|---|
| **E7** | batching transfers from lm_head to the whole model | `llama-batched-bench -npl 1,2,4` at 4k + per-shape E2 | per-slot **55.6 / 80.1**, aggregate **111 / 320** t/s | the deployment change |
| **E6** | MTP rejection at 112k rests on a stale number | `FLASH_ATTN_EXT` perf, q8_0/q5_0, `ne[1]`=4 vs 1, at 4k and 112k | at `ne[1]=4` the dispatcher leaves the GQA kernel (`fattn.cu:555`) for `BEST_FATTN_KERNEL_TILE` (`fattn.cu:779-783`) — cost unmeasured | re-open or finally close spec decode at long context (break-even: pass ≤1.36× single needs α≈0.28) |
| **E1** | cost is per-element below ~4.4 bpw | 4–5 perf cases, same 89.1 M-element shape, IQ3_XXS→Q8_0 | flat ~99–106 µs through Q4_K; Q8_0 (947 MB) ≥ 1.88 ms | whether *any* quantization work can buy decode speed |
| **E2** | the n=4 knee is L1 activation capacity | n=1/2/4/8 on 17408×5120 (y=5.76 KB) and 5120×17408 (y=19.6 KB) | K=5120 ≤5 % to n=4, breaks at 8; K=17408 ≥15 % by n=4 | max safe k |
| **E3** | `__ldcs`/`__ldcg` on weights protects L1 | 2 builds, else identical | **0 ± 1 %** | closes the cache-hint lane |
| **E4** | persisting-L2 is available/useful | read `hipDeviceAttributePersistingL2CacheMaxSize` | **0 on gfx1030**; even if nonzero, no reusable set → gain 0 | closes the lane |
| **E5** | KV is IC-resident below ~4.6k | depth sweep 0…12k, attention residual | slope ~0 below **4,571** tokens, then ~0.10–0.11 µs/KV-token | whether KV/attention work matters at ≤4k (it does not) |

E3 and E4 are worth running **once each** to close those lanes with evidence rather than
argument. Do not run them twice.

## What not to do

- **No cache-hint / persisting-L2 / `__ldcs` programme.** No reuse to exploit (81.4×
  oversubscription, 0.19 ms best case), `y` already resident, KV already fits IC at 4k.
- **No working-set sweep for weights.** At 10.414 GB/token there is no working set to fit.
  Only the KV knee (E5) is meaningful.
- **No re-run of VDR / rows-per-block / nwarps / occupancy / `__launch_bounds__`.** Seven
  configurations, flat-to-worse in situ, isolated wins that did not transfer. That null
  result *is* the finding.
- **No attempt to close 420.6 → 504 with kernel work.** Reachable gain is +2…+12 % of
  matvec (0.5–3 ms) against a proven-negative search space.
- **No multi-slot at 112k** — does not fit. **No MTP work at 112k before E6.**

## One-line summary

Stop measuring this card's caches and start measuring it at k=4. The exploitable assets
are 128 MB of Infinity Cache holding a 117 MB KV working set at 4k, and 504 GB/s of DRAM
feeding one weight read that four tokens can share — not the weight stream itself, which
will never be cached and is already within 12 % of what this kernel structure extracts
from it.

## Verified cache hierarchy (gfx1030 / Navi 21) and why none of it is a lever

| level | size / scope | line | source |
|---|---|---|---|
| L0 vector | 16 KB per WGP (shared by the 2 CUs) | — | AMD/Wikipedia; `rocminfo` reports it as "L1 16 KB" |
| L0 scalar | 16 KB per WGP | — | same |
| L1 | 128 KB per shader array × 8 arrays ≈ 1 MB | — | Wikipedia RDNA 2 |
| L2 (GL2C) | **4 MB** total, all CUs, on-die | 128 B | `rocminfo` |
| Infinity Cache (L3) | **128 MB** = 2 × 64 MB sets, own clock | 128 B | `rocminfo` L3 131072 KB |
| IC internal bandwidth | **1986.6 GB/s** (~3.9× DRAM) | — | Wikipedia RDNA 2 |
| DRAM | 16 GB GDDR6, 256-bit, 512 GB/s nominal | 128 B | measured copy ceiling 504.3 GB/s |

The IC bandwidth figure settles the question of whether the cache is fast: it is, by
~3.9×. It does not settle whether we can use it, and we cannot, for a structural reason
that no code change addresses: **a weight is read once per token into a 10.4 GB stream
against a 128 MB cache — 81.4× oversubscription.** No cache level can help a one-shot
read. The reusable data in a token — the activation row `y`, reused by every block, and
the KV cache below ~4,570 tokens — is already resident, which is why the matvec runs at
420 GB/s instead of collapsing.

### The cache-control avenue is closed by the toolchain, not by measurement

- `__ldg` / `__ldcg` / `__ldcs` are **no-ops on AMD HIP**: `hip/amd_detail/hip_ldg.h`
  defines all three as plain `*ptr`; `__ldlu` / `__ldcv` are not defined at all. They emit
  **no** cache modifier. Our tree uses `__ldg` only in `pool1d.cu:33` / `pool2d.cu:39`,
  and it changes nothing there either.
- The real modifiers are `glc` (L1 coherent) and `slc` (L2 streaming mode). Neither is
  reachable from HIP intrinsics on gfx10 — only via inline asm, which the tree uses
  nowhere. `dlc`, `sc0`, `sc1`, `nt` are not available on gfx10 (they are CDNA/gfx94
  encodings).
- The one prefetch in the codebase, `prefetch.global.L2` in `mmvq.cu:762-775`, is gated
  `#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK` — i.e. NVIDIA-only. **There is no
  prefetch at all on gfx1030.**
- Persisting-L2 is unusable: `hipDeviceAttributePersistingL2CacheMaxSize` reports 4 MB
  (the whole L2) but `hipDeviceAttributeAccessPolicyMaxWindowSize = 0`, so the access
  policy window cannot be set. No code in the tree uses it.
- No IC counters exist in `rocprofv3 -L` on ROCm 7.2.4 — only `GL2C_HIT` / `GL2C_MISS` /
  `GL2C_MC_RDREQ` / `GL2C_EA_RDREQ_*` for the 4 MB L2 (`TCC_*` are the gfx9 names). So IC
  behaviour cannot be measured on this host at all.

### Lever verdicts (all no-op, from hardware)

| lever | verdict |
|---|---|
| `slc` streaming hint on weight loads (`mmvq.cu:756-796`) | no-op: reuse distance 10.4 GB vs 128 MB; best case ~0.19 ms (0.6 %) |
| port the L2 prefetch to gfx1030 | no-op/negative: every MLP lever (VDR, rows/block, nwarps, launch bounds) measured flat or worse in situ |
| LDS staging of weights | no-op by construction: no reuse to stage for, only an extra copy |
| `MMVQ_MAX_BATCH_SIZE` above 8 | would spill L0 (n=4 activation set 23 KB fits the 16 KB vector L0 imperfectly, n=8 46 KB spills); batching itself is the win, not the constant |
| KV residency / persisting-L2 | already captured; window unusable, KV buffers already device-local |
| `__ldg` → plain load in pooling | clarity only; no separate texture cache on RDNA2 |

**Corrected E3 prediction:** previously "0 ± 1 %", now **0, proven by hardware** — the
modifier cannot be emitted, so the experiment cannot be run as specified.

The remaining levers are therefore exactly two, and neither is a cache: **dispatch count**
(1,788/token; gap 2.13 ms = 15× the traffic time) and **batching** (k=4 measured
31.41 → 12.48 ms/token).
