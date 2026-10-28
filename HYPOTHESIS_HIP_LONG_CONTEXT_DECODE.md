# Hypothesis: HIP long-context decode collapse is a quantized-V attention-path defect

Status: proposed, not confirmed. No new benchmark sweeps. One discriminator is defined at the end.

## 1. What has to be explained

Decode throughput on the local RX 6900 XT (gfx1030, 80 CUs), IQ3_S 27B, FA1, `-fitt 128`:

| depth | HIP q8_0 K / q5_0 V | HIP q8_0 K / q8_0 V | Vulkan q8_0 K / q5_0 V | Vulkan q8_0 K / q8_0 V |
|---|---:|---:|---:|---:|
| 0 | 29.4-30.0 | - | 32.1-32.5 | - |
| 8k | 26.86 | 27.91 | - | - |
| 32k | 19.82 | 22.59 | 29.24 | 30.34 |
| 112k | 9.86 | 13.57 | 23.23 | OOM (device lost) |

Prefill goes the other way: HIP 594-603 t/s vs Vulkan 474-482 t/s. So this is not a
global HIP defect; it is specific to decode at depth.

Two facts dominate:

1. The HIP-Vulkan gap grows with depth: ~10% at 0, ~33-48% at 32k, ~2.35x at 112k.
2. On HIP, q8_0 V is 4% faster at 8k, 14% at 32k, and 38% at 112k than q5_0 V,
   even though q8_0 V moves 54.5% more bytes (1.0625 vs 0.6875 B/value).

A bandwidth-bound path cannot produce fact 2. The extra bytes should make q8_0 V
slower, not faster.

## 2. Decomposing token time: fixed cost vs depth slope

Convert the measurements to ms/token and fit `t(depth) = t_fixed + s * depth`.
Using the 32k and 112k points:

| config | t(32k) | t(112k) | slope `s` | fixed `t_fixed` | depth share @112k |
|---|---:|---:|---:|---:|---:|
| HIP q5_0 V | 50.45 ms | 101.42 ms | 0.637 us/KV-token | 30.1 ms | 70% |
| HIP q8_0 V | 44.27 ms | 73.69 ms | 0.368 us/KV-token | 32.5 ms | 56% |
| Vulkan q5_0 V | 34.20 ms | 43.05 ms | 0.111 us/KV-token | 30.7 ms | 29% |

The fitted fixed cost is ~30-33 ms/token on both backends, i.e. 30-33 t/s. That is the
short-context ceiling and it contains the weight matvec (MMVQ), lm_head, sampling, and
launch overhead. It is essentially backend-independent.

The entire long-context gap is the slope:

- HIP q5_0 V: **0.637 us per KV token** (5.8x Vulkan's slope).
- HIP q8_0 V: **0.368 us** (42% better than q5_0 V, still 3.3x Vulkan).
- Vulkan q5_0 V: **0.111 us**.

At 112k, the depth-dependent term is ~71 ms of HIP's 101 ms token time, but only
~12 ms of Vulkan's 43 ms. Tuning MMVQ or clocks cannot move this; only the per-KV-token
attention path can.

Caveat: depth points are single runs (`-r 1`), so the slopes are estimates. The effect
is large enough that the conclusion is robust, but the exact constants are not.

## 3. Primary hypothesis

**H1.** On gfx1030, the HIP flash-attention/KV path has a per-KV-token cost for q5_0 V
that is dominated by unpacking the 5-bit payload of the 22-byte q5_0 block, and/or by the
resulting uncoalesced access pattern. This cost is paid once per KV element per decode
token, so it scales linearly with depth. q8_0 V avoids the bit-unpack (byte-addressable
32 x int8 payload), which is why it is faster despite moving more bytes. Vulkan's FA
dequant is vectorized/shared across lanes, so its slope is 5.8x lower and q8_0 V buys
only 3.8% at 32k.

Sub-mechanisms, in order of likelihood:

- **H1a - unpack ALU bound.** q5_0 needs shifts/masks to reconstruct 32 five-bit values;
  q8_0 is a direct byte load plus scale. The FA inner loop is instruction-bound on the
  V side at depth.
- **H1b - access/coalescing bound.** 22-byte q5_0 blocks force scalar or narrow loads;
  q8_0's 34-byte blocks are still odd-sized but byte-addressable, so wider loads are
  possible. Effective bandwidth, not raw bytes, is the limit.
- **H1c - kernel selection.** HIP picks a slower FA variant for q5_0 V (e.g. a vec path
  without K/V tile reuse), while Vulkan always runs the tiled path. q8_0 V would then
  benefit from a different kernel, not just cheaper unpacking.

H1a/H1b are V-type defects; H1c is a dispatch/tiling defect. The 32k f16-V test below
separates them.

## 4. Alternatives considered and why they are weaker

- **KV bandwidth.** Rejected by fact 2: q8_0 V is faster with 54.5% more bytes. The
  only surviving bandwidth variant is H1b (effective bandwidth/coalescing), not raw bytes.
- **Clocks/thermals/power.** Clocks were identical on both backends (2520 MHz sclk,
  1000 MHz mclk); HIP power was slightly lower. Vulkan runs the same GPU at 23.2 t/s
  at 112k without collapsing.
- **MMVQ / weight path.** Weights, batch, and graph are identical across depths; only
  KV length changes. The fixed cost is the same ~30-33 ms on both backends, so the
  matvec is not the differentiator.
- **KV spill to GTT/host memory.** A spill signature exists: HIP f16 K + q5 V at 112k
  collapsed to 1.69 t/s. The q5_0 and q8_0 runs at 9.86/13.57 do not look like that,
  and the q8_0 run (more VRAM) is faster, not slower. If a GTT sample later shows
  >1 GiB for the 9.86 run, revisit this.
- **`-fitt` silently truncating context or offloading layers.** That would produce a
  step change, not a smooth 32k-to-112k slope with a stable ~30 ms intercept. The load
  log can confirm the actual context/layers without a new benchmark.
- **FA disabled on HIP.** FA1 beats FA0 at 8k (29.2 vs 24.7 t/s), so FA is active and
  beneficial. The defect is inside the FA path, not a fallback to non-FA.

## 5. Falsifiable predictions

- **P1 (slope).** A fix that works must move the slope, not the intercept. Target:
  HIP q5_0 V slope < 0.20 us/KV-token (from 0.637), which implies >=18 t/s at 112k
  and >=26 t/s at 32k, with the same ~30 ms fixed cost.
- **P2 (V path).** A HIP FA variant that stages q5_0 V through a vectorized 32-value
  unpack, or converts q5_0 V to q8_0/f16 once per layer per token into a scratch tile,
  should recover >=80% of the q5_0-to-q8_0 gap. If it recovers <20%, H1a/H1b are false.
- **P3 (backend isolation at 32k).** With f16 V (no dequant) at 32k, where it fits,
  HIP should come within ~10% of Vulkan f16. If HIP is still >=25% slower with f16 V,
  the defect is H1c (FA tiling/scheduling), not V unpacking.
- **P4 (bandwidth).** If q5_0 V were raw-bandwidth-bound, q8_0 V would be slower at
  every depth. Measured opposite at 32k and 112k; reject.
- **P5 (spill).** GTT usage during the HIP 112k q5_0 run should be <0.5 GiB. If >1 GiB,
  the spill alternative is back in play.

## 6. Decision rule

- If P3 shows HIP f16 V is close to Vulkan f16 V at 32k: the defect is narrow. A single
  focused patch (vectorized q5_0 V unpack or q5_0 -> q8_0/f16 staging in FA) is
  justified, because it would lift both 32k and 112k decode.
- If P3 shows HIP is still far behind with f16 V: the problem is HIP FA tiling/scheduling
  on RDNA2. That is a larger project with uncertain payoff. Close the HIP decode
  workstream and keep Vulkan.
- Either way, the production recommendation does not change: **Vulkan for long-context
  decode; HIP only for prefill-dominated requests.**

## 7. The one experiment worth running later

Not a sweep. Two 32k decode runs, both fit in VRAM:

1. HIP `-ctk q8_0 -ctv f16`, 32k, 3 reps.
2. Vulkan `-ctk q8_0 -ctv f16`, 32k, 3 reps.

If HIP f16 V closes the gap, implement the q5_0 V path fix. If it does not, stop.
No profiler is required; 32k is the discriminator. f16 V does not fit at 112k, so do
not repeat there.

## 8. Practical consequences now

- **Do not run more backend comparisons.** Vulkan is the answer for this workload.
- **Current 112k config is already the fastest fitting Vulkan config measured:**
  `q8_0` K / `q5_0` V, 23.2 t/s at full depth. `q8_0` V OOMs at 112k.
- **If context can be capped at <=32k:** `q8_0` K / `q8_0` V gives +3.8% decode on
  Vulkan (30.34 vs 29.24) and is higher precision. At 8k and below the gain is smaller.
- **Prefill routing only:** HIP is ~23% faster at prefill. Using the measured rates,
  HIP wins only when prompt tokens dominate: roughly P > 42 x G at 32k and
  P > 150 x G at 112k. For long-context chat, Vulkan wins regardless.
- **No in-process hybrid.** One GPU, backend-owned buffers, no zero-copy cross-API,
  and the 11.28 GiB weights cannot be duplicated. Per-workload backend selection is the
  only viable "hybrid".
- **If more decode speed is still required:** the next real lever is speculative
  decoding with a Qwen-compatible draft model, not more HIP MMVQ tuning.
