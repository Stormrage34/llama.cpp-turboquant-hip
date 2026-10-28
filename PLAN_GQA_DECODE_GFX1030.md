# Plan: port hipfire's GQA-6 decode tile from Rust/HIP to ggml-hip C++ (gfx1030)

Status: implemented and measured (2026-10-25); see section 9 for results and section 10
for the residual-gap hypothesis. Target: RX 6900 XT, gfx1030, 80 CU, wave32, local build only.
Scope: single-token decode, Qwen3.8-27B geometry (head_dim 256, GQA 6, 17 full-attention
layers), quantized K/V (Q8_0 K, Q8_0/Q5_0 V). No commits, no PRs.

## 0. Why this kernel

Measured on this host:

| backend | 112k decode | 32k decode | attention slope |
|---|---:|---:|---:|
| Vulkan q8_0/q5_0 | 23.2 t/s | 29.2 t/s | 0.111 us/KV-token |
| HIP q8_0/q5_0 | 9.9 t/s | 19.8 t/s | 0.637 us/KV-token |
| HIP q8_0/q8_0 | 13.6 t/s | 22.6 t/s | 0.368 us/KV-token |

The HIP/Vulkan slope ratio is 5.7x; the model's GQA ratio is 6. That is not a
coincidence. llama.cpp's HIP FA selects the **VEC** kernel for quantized decode
(`Q->ne[1] <= 2`), and `fattn-vec.cuh` runs one Q head per block with
`K/V += head / gqa_ratio`. The same K/V is read and dequantized six times.

hipfire already has the fix: a GQA-shared Q8 decode tile for **head_dim 256, GQA 6**,
one 256-thread workgroup per (KV head, tile) serving all six Q heads. It exists only for
gfx1100/gfx1151/gfx1201. This plan ports it to ggml-hip/gfx1030.

Expected from the measured 71 ms attention term at 112k: 2x sharing -> 15.2 t/s,
4x -> 20.9, 6x -> 23.8; at 32k: 25.5 / 28.4 / 29.9. The 6x case matches Vulkan at
112k and beats it at 32k.

## 1. Source map: Rust -> C++

| hipfire (Rust/HIP) | llama.cpp C++ target | Role |
|---|---|---|
| `kernels/src/attention_flash_q8_0_tile_gqa.gfx1100.hip` | new `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cuh` | the kernel |
| `attention_flash_q8_0_reduce.hip` | existing `flash_attn_combine_results<DV>` in `fattn-common.cuh` | combine partials |
| `crates/rdna-compute/src/attention.rs::q8_decode_attn_gqa_gfx1100_admitted` | new predicate in `fattn.cu` | shape/type guard |
| `attention.rs::DecodeGqaPair`, `Q8_DECODE_GQA_GFX1100` | constants in the new header | tile/group/waves |
| `attention.rs::attention_flash_decode_gqa_pair` | new `ggml_cuda_flash_attn_ext_gqa_gfx1030(ctx, dst)` | launcher |
| `attention.rs::VmmFlashDecodePlan` / `max_tiles` sizing | existing `dst_meta` + `blocks_num.y` allocation | split-K layout |
| `crates/hipfire-arch-qwen35/src/qwen35/forward.rs::q8_decode_tile` | `nbatch_fa`/tile 128 in the launcher | tile size |
| `qwen35_flash_partials_len` | `ggml_cuda_flash_attn_ext_get_alloc_size` | extra buffer |

Do not port the Rust runtime, KV cache layout, or `VmmFlashDecodePlan` structs. Only the
kernel body, constants, guard, and launcher arithmetic map to C++.

## 2. Kernel design

### 2.1 Geometry

```
GQA_TILE   = 128      // KV keys per workgroup
GQA_HD     = 256      // head_dim
GQA_GROUP  = 6        // Q heads per KV head
GQA_WAVES  = 8
GQA_THREADS= 256      // 8 * 32
```

One workgroup per `(sequence, kv_head, kv_tile)`. It computes all 6 Q heads of that KV
head. Grid: `(n_kv_tiles, 1, n_seqs * n_kv_heads)`; block: 256 threads. The final combine
runs over `n_kv_tiles` partials.

### 2.2 Data path

- Q: 6 x 256 f32, staged once in LDS/registers. `Q` base = `(kv_head * 6) * nb02 + seq * nb03`.
- K: Q8_0, 8 blocks of 34 B per key per head. Load the 128-key tile once into LDS.
- V: Q8_0 first (34 B/block), then Q5_0 (22 B/block). Load 32 or 64 V rows at a time;
  do not assume the full 128-row tile fits in 64 KiB LDS.
- Compute `S = Q @ K^T` for 6 heads, tile-wise softmax (max, exp, sum), then
  `O = softmax(S) @ V` for 6 heads. Each head writes its own partial and `dst_meta`.
- Causal: decode query position is `pos = KV_max ? KV_max-1 : K->ne[1]-1`; only keys
  `<= pos` contribute. First version supports standard causal decode with
  `mask == nullptr` or the standard causal mask; no ALiBi/custom mask.
- `logit_softcap == 0`, `max_bias == 0`, no sinks for the first version.

### 2.3 gfx1030-specific details

- Wave32 only; `__launch_bounds__(256, 1)` initially, then try `(256, 2)` only if LDS
  allows and spills stay 0.
- **No RDNA3 unaligned-dword mode.** The gfx1100 kernel reads a 4-byte code word at
  `34*b + 2 + 4*(r%8)`. On gfx1030 use two aligned dword loads + `__byte_perm`/shifts,
  or a 2-byte tail load as in the Q8_0 MMQ v7 port. Assert the 2-byte base alignment.
- `#pragma unroll 1` on the outer KV-tile and V-row loops; unroll only the 32-value
  dequant and the 6-head inner loops.
- LDS budget target <= 64 KiB. If Q + K + V exceeds it, stage 32 V rows or 64 K keys at
  a time; never silently drop the tail.
- 0 spills; record VGPR/LDS in the PR notes. If VGPR pressure forces spills, drop to
  4 waves (128 threads) and re-measure before giving up.

### 2.4 Output contract

Reuse the existing `fattn-vec` contract exactly:

```
dst[(((seq*ne01 + j)*ne02 + head)*gridDim.y + blockIdx.y)*DV + d] = out;
dst_meta[((seq*ne01 + j)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(max, sum);
```

The kernel writes this for each of its 6 heads. `flash_attn_combine_results<256>` then
combines over `gridDim.y` without change. This avoids a new partials buffer and a custom
reduce kernel.

## 3. C++ integration

### 3.1 Files

- **new** `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cuh`: kernel + host launcher.
- **new** `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu`: thin TU if per-file HIP flags are
  needed (preferred; see 3.4).
- `ggml/src/ggml-cuda/fattn.cu`: enum, guard, dispatch, alloc size.
- `ggml/src/ggml-cuda/fattn.cuh`: launcher declaration.
- `ggml/src/ggml-hip/CMakeLists.txt`: add the TU + `-mno-wavefrontsize64`.
- `tests/test-backend-ops.cpp`: add `FLASH_ATTN_EXT` cases (or rely on the generic
  coverage plus a parity harness).

### 3.2 Selection

Add a value after `BEST_FATTN_KERNEL_TILE`:

```cpp
BEST_FATTN_KERNEL_GQA_GFX1030 = 300,
```

In `ggml_cuda_get_best_fattn_kernel()`, before the generic `can_use_vector_kernel`
branch:

```cpp
if (ggml_cuda_fattn_gqa_gfx1030_supported(cc, dst)) {
    return BEST_FATTN_KERNEL_GQA_GFX1030;
}
```

Predicate:

```cpp
static bool ggml_cuda_fattn_gqa_gfx1030_supported(int cc, const ggml_tensor * dst) {
    if (!GGML_CUDA_CC_IS_RDNA2(cc)) return false;       // gfx1030/1031
    const auto * Q = dst->src[0]; const auto * K = dst->src[1]; const auto * V = dst->src[2];
    if (Q->ne[0] != 256 || V->ne[0] != 256) return false;
    if (Q->ne[1] > 1) return false;                     // decode only
    if (K->ne[2] == 0 || Q->ne[2] != 6 * K->ne[2]) return false;
    if (K->type != GGML_TYPE_Q8_0) return false;
    if (V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q5_0) return false;
    if (dst->src[3] != nullptr && !standard_causal_mask(dst)) return false;
    // max_bias/logit_softcap/sinks must be 0/null
    return true;
}
```

`standard_causal_mask()` should accept a null mask and the ordinary causal mask; any
sliding-window/ALiBi/custom mask falls back to VEC.

### 3.3 Dispatch and allocation

- `ggml_cuda_flash_attn_ext()`: add a `case` that calls
  `ggml_cuda_flash_attn_ext_gqa_gfx1030(ctx, dst)`.
- `ggml_cuda_flash_attn_ext_get_alloc_size()`: new case sets
  `need_f16_K = false`, `need_f16_V = false` (the kernel reads quantized K/V directly).
  Ensure the `dst`/`dst_meta` extra allocation covers `Q->ne[2] * n_kv_tiles`
  float2 entries; copy the VEC arithmetic.
- If `blocks_num.y` is capped (grid-y limits), loop tiles inside the kernel or chunk the
  grid; gfx1030 grid.y is large enough for 112k/128 = 876, but keep the cap check.

### 3.4 Build

- Put the kernel in a separate `fattn-gqa-gfx1030.cu` and add:
  `set_property(SOURCE ... PROPERTY COMPILE_OPTIONS "-mno-wavefrontsize64")`
  in `ggml/src/ggml-hip/CMakeLists.txt`, matching
  `mmq-hipfire-dispatch.cu`.
- Guard the TU with `#ifdef GGML_USE_HIP` and compile an empty TU under CUDA, so the
  CUDA build and `-DLLAMA_FATAL_WARNINGS=ON` stay clean.
- Do not add the new file to a second glob; it is already covered by the HIP source
  list. Do not compile it for CUDA.

## 4. Phases

### Phase 0 — baseline + parity harness (0.5 day)

- Record VEC vs Vulkan at d0/8k/32k/112k, q8/q5 and q8/q8, 3 reps.
- Build a standalone `FLASH_ATTN_EXT` parity case: GQA 6, head_dim 256, Q tokens 1,
  KV 4k/32k, K q8_0, V q8_0/q5_0. Compare against CPU/Vulkan max-abs/rel error and
  argmax.
- Confirm in a debug print that the current selector returns VEC for this shape.

### Phase 1 — Q8/Q8 GQA-6 kernel (3–5 days)

- Port the tile body; Q8 K/V only; standard causal decode; `mask == nullptr` or
  standard causal.
- Write the 6 heads' partials + `dst_meta`; reuse `flash_attn_combine_results<256>`.
- Validate 4k/32k parity. Measure 32k/112k.
- **Gate:** >= 1.5x over VEC at 32k and >= 1.5x at 112k, 0 spills, no correctness
  failures. If not, stop and record the negative result.

### Phase 2 — Q5_0 V (1–2 days)

- Add a q5_0 V dequant variant: 22-byte blocks, 5-bit unpack, vectorized 32 values,
  safe unaligned loads; same partials contract.
- Measure q8 K / q5 V at 32k/112k; compare with q8 V.
- **Gate:** q5 V must be within 10% of q8 V at 32k; otherwise ship q8 V only.

### Phase 3 — tuning (1–2 days)

- Tile 64 vs 128, V staging 32 vs 64 rows, waves 4 vs 8, `launch_bounds(256,1)` vs
  `(256,2)`, grid-y cap. Keep only measured wins.
- Check LDS bank conflicts and VGPR; no `MMQ_Y`/VGPR-cap tuning (upstream negative).

### Phase 4 — integration + regression (1 day)

- `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` and the full server decode at 112k.
- Multi-slot (`-np 4`) and multiple sequences; fallback correctness when the guard
  rejects (mask/ALiBi/other head_dim).
- `-DLLAMA_FATAL_WARNINGS=ON` build for `ggml-hip`; clang-format.
- Final: keep the kernel only if it beat VEC at 112k by >= 25% and did not regress 32k.

## 5. Validation matrix

| Case | Shape | Expected |
|---|---|---|
| decode short | Q=1, KV=4k, GQA 6, q8/q8 | parity + speed |
| decode 32k | Q=1, KV=32k, q8/q8, q8/q5 | >= 1.5x VEC |
| decode 112k | Q=1, KV=112128, q8/q5 | >= 18 t/s target |
| multi-seq | Q=1, KV=32k, 4 seqs | parity |
| fallback | GQA 4, head_dim 128, f16 KV | selector must reject |
| correctness | `test-backend-ops -o FLASH_ATTN_EXT` | all pass |

Performance is measured at depth 0/8k/32k/112k with `llama-bench -p 0 -n 128 -r 3`,
GPU idle, same clocks; report median and slope.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| gfx1030 unaligned 34-byte Q8 loads | two aligned dwords + byte permutes; 2-byte tail; assert base alignment |
| LDS over 64 KiB with Q+K+V | stage 32 V rows / 64 K keys; measure; never drop tails |
| VGPR spills at 6 heads | 4 waves or split heads into 2 groups of 3; gate on 0 spills |
| GQA 6 not a power of two | dedicated kernel; do not force `ncols2=6` through the existing tile template |
| existing TILE path is f16-only | do not use it for q8/q5; it sets `need_f16_K/V=true`, doubling KV and adding conversion |
| mask/ALiBi/custom | first version standard causal only; selector rejects the rest |
| multi-slot KV | validate `-np 4`; `dst_meta`/grid indexing per sequence |
| q5 V dequant cost | q8 V first; q5 variant gated on <10% gap |
| build break on CUDA | empty TU under `!GGML_USE_HIP`; per-file HIP flags |

## 7. Decision gates

1. Phase 1: if the Q8/Q8 kernel is < 1.5x VEC at 32k or 112k, stop. The GQA-sharing
   hypothesis is wrong or the kernel is memory-latency-bound elsewhere.
2. Phase 2: if q5 V is > 10% slower than q8 V, ship q8 V only and document the VRAM
   tradeoff.
3. Final: keep only if 112k >= 18 t/s (>= 1.8x VEC) and 32k >= 26 t/s. Otherwise revert
   and keep the findings note.

## 8. Out of scope

- MQ3/MQ4/Q8_0 MMQ ports: prefill-only, already exhausted or slower.
- GEMV variant tuning: v1–v5 tie; VGPR caps slower.
- MQ2 prefill: open upstream #533, unrelated to this model.
- Vulkan: its FA shader already uses GQA tiling; no equivalent gap.
- MTP/speculative decode: separate architecture/runtime work.

## 9. Implementation result (2026-10-25)

Status: implemented and measured. Known-good source configuration:

```
GQA_GFX1030_GROUP = 6
GQA_GFX1030_TILE  = 256
GQA_GFX1030_CHUNK = 24
GQA_GFX1030_WARPS = 6      // 192 threads, launch_bounds(192, 3)
K = Q8_0, V = Q8_0 or Q5_0, native (no f16 conversion)
QK = q8_0 Q quantized once per head + sdot4 against raw Q8_0 K
V = dequantized to f16 in LDS, read as half2
```

Files:

- **new** `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu` (kernel + launcher)
- `fattn.cu`: `BEST_FATTN_KERNEL_GQA_GFX1030`, selector, alloc-size case, dispatch
- `fattn.cuh`: launcher declaration
- `ggml/src/ggml-hip/CMakeLists.txt`: per-file `-mno-wavefrontsize64`, `-Wno-pass-failed`
- `tests/test-backend-ops.cpp`: four GQA-6/head_dim-256 cases (4k q8/q8, 4k q8/q5,
  5003 tail q8/q5, 32768 q8/q5)
- Runtime switch: `GGML_HIP_GQA_GFX1030=0` restores the VEC path.

Measured on the local RX 6900 XT, `-fa 1 -fitt 128`, q8_0 K / q5_0 V:

| depth | old VEC | GQA gfx1030 | Vulkan |
|---|---:|---:|---:|
| 32k | 19.72-19.76 | **27.27-27.37** | 29.24 |
| 112k | 9.88-9.91 | **22.16-22.47** | 23.23 |

q8_0 V gives the same result as q5_0 V with the new kernel (22.39 at 112k, 27.28 at
32k), so the old q5-to-f16 fallback is gone and native q5 V is not a bottleneck.

Correctness: 4/4 targeted `FLASH_ATTN_EXT` cases pass (4k q8/q8, 4k q8/q5, 5003 tail,
32768 q8/q5). The full suite was last run before the final sdot4/tail changes; re-run
before merging. `ggml-hip` builds; a fatal-warnings build is still pending.

Gate result: **passed** (112k >= 18 t/s, 32k >= 26 t/s). HIP is now within ~4% of
Vulkan at 112k and ~7% at 32k, versus 2.35x behind before the kernel.

## 10. Hypothesis for the residual 4-7% gap

> **Superseded (2026-10-25).** Measured: at depth 0, where FlashAttention never runs, HIP is
> already 30.87 vs Vulkan 33.15 t/s (-7.4%). The gap is in the decode weight path (quantized
> MMVQ), not in this kernel. See `PLAN_GFX1030_DECODE_V2.md` §2 and §4 (P0/P1). The hypotheses
> below are kept as a record; they can account for at most the ~0.4 ms extra attention cost
> observed at 8k depth.

**H2: the remaining HIP/Vulkan gap is per-token fixed overhead, not attention depth.**

Depth fit from the measured points:

| backend | t(32k) | t(112k) | slope | fixed |
|---|---:|---:|---:|---:|
| HIP GQA | 36.6 ms | 44.8 ms | **0.103 us/KV-token** | **33.3 ms** |
| Vulkan | 34.2 ms | 43.0 ms | 0.111 us/KV-token | 30.7 ms |

HIP's depth slope is now slightly **better** than Vulkan's. The entire residual gap is
the ~2.7 ms higher fixed term. This is the opposite of the pre-kernel situation, where
HIP's slope was 5.7x worse.

Candidate mechanisms, in order:

1. **Per-block Q quantization is redundant.** Each of the ~240 blocks per layer
   re-quantizes the same six Q heads (6 x 256 values). A separate pre-quantization pass
   per token, or a q8 Q scratch buffer, would do it once.
2. **Partials/combine traffic.** 24 heads x `parallel_blocks` (~60) x 256 f32 partials
   plus a second combine kernel. This is small in bytes but adds a launch and a full
   read/write pass per layer.
3. **Load/compute serialization.** Two `__syncthreads` per 24-key chunk; the three WGs
   per CU alternate load and compute phases. Double-buffering 16-key chunks would
   overlap them.
4. **Tail/scheduling.** 240 blocks over 80 CUs, with 438 tiles at 112k and 128 at 32k;
   the last wave is uneven.
5. **Measurement noise.** The depth points are single runs; a five-point sweep is
   needed to trust the intercept difference at the ~2 ms level.

Falsifiable predictions:

- **P1:** a 0/8k/16k/32k/64k/112k sweep still shows a constant ~2-3 ms intercept gap.
  If the gap grows with depth, H2 is false.
- **P2:** pre-quantizing Q once per token (global q8 scratch) removes >= 1 ms of fixed
  cost. If it does not, mechanism 1 is not material.
- **P3:** reducing `parallel_blocks` or fusing the combine changes the fixed term but
  not the slope. If the slope moves, the model is wrong.
- **P4:** double-buffered load/compute removes the sync stall; if the fixed term is
  unchanged, mechanism 3 is not material.
- **P5:** q8 V vs q5 V is already equal (22.39 vs 22.16), so V dequant is not the gap.

Decision: the kernel meets the gate. The remaining 4-7% is not worth another kernel
rewrite unless the goal is specifically to beat Vulkan at every depth. If pursued, the
cheapest order is P2 (Q pre-quantization), then P4 (double buffer), then P3 (combine).

## 11. Current state

- Source is at the known-good WARPS=6 / CHUNK=24 / TILE=256 / `(192,3)` configuration.
  The last build before the WARPS revert was a test of WARPS=8; rebuild before measuring.
- All changes are uncommitted. Do not commit build directories.
- The Vulkan server on port 9931 was stopped for the HIP benchmarks and must be
  restarted with the original `-c 112000 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128` command.
- Next if continuing: run the full `FLASH_ATTN_EXT` suite and the fatal-warnings build,
  then P1/P2. Otherwise ship the kernel behind the env switch and keep Vulkan as the
  production default until the fixed term is closed.
