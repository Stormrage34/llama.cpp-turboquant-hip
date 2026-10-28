# hipfire @ 56367a7 — gfx1030 kernel & issue findings

Scope: [`warpfront/hipfire` kernels tree at commit `56367a75`](https://github.com/warpfront/hipfire/tree/56367a75aefa3093520a5b3880395bc65daf7a8d/kernels)
(2026-10-24, "README: ghcr image is public"), plus the repository's issue/PR tracker as of
the same date. Findings are filtered for relevance to the `ggml-hip` gfx1030 port in this
tree (`ggml/src/ggml-cuda/mmq-hipfire-*`).

Method: shallow clone of the pinned commit, inventory of `kernels/src` (1,278 files),
filename/content search for `gfx1030`/`RDNA2`, read of every gfx1030-specific kernel and
its shared body, and a scan of all 968 issue/PR titles plus keyword searches
(`gfx1030`, `RDNA2`, `mmq`, `gemv`, `q8_0`) with the relevant bodies and comments.

---

## 1. Executive summary

1. **The gfx1030 MMQ family is a shared-body + preprocessor-tile design**, not a template
   explosion: one `*_body.cuh` per quant family, with `MMQ_X`/`MMQ_Y`/`KERNEL_NAME`
   defined by 4–16 line wrappers. There is no `rdna2/` directory and no `radiowave/hip.h`
   include tree at this commit.
2. **Tile selection at this commit is `N <= 63 → x16/y128`, else `x32/y64`** for the
   HFQ4 and HFQ3 residual kernels. Issue text (#299/#315) still quotes the older
   `13..127 → x16`, `>= 128 → x32` boundaries; the code is authoritative.
3. **`__launch_bounds__` is `(128, 2)` for every gfx1030 MMQ body, including `y64`.**
   Upstream never used `(128, 4)`. Occupancy comes from the LDS budget, and the runtime
   is not occupancy-bound (see §6).
4. **The `tid < MMQ_Y` metadata gate is a documented correctness fix**, not a stylistic
   choice: at `y64`, threads 64..127 would write `x_dm[64..127]` into the adjacent
   `tile_y` LDS region and emit NaN.
5. **Q8_0 MMQ (`gemm_q8_0_mmq.gfx1030.hip`, v7) is a channel-test probe**: its file
   comment and launcher both say no production dispatch admits it. Local measurement
   confirms why: it is **24-257% slower** than llama.cpp's stock Q8_0 MMQ on the RX 6900
   XT, so the llama.cpp hook is opt-in only (default disabled).
6. **MQ2-Lloyd on gfx1030 exists only as a grouped-MoE probe.** Batched MQ2 prefill is
   still open (#533/#558); a contributor's fork measured 14–21x, but it is unmerged.
7. **Quality alarm on MMQ was a phantom** (#302 resolved): Q8_1 X quantization is
   KLD-neutral vs dot2 (all four paths within 0.07% at n=256), and the later default-on
   PR (#335) confirmed KLD-neutrality on MQ4 and MQ3.
8. **The remaining levers were exhaustively falsified** (#300): prefill is 88.6% two MMQ
   GEMMs at 15–20 GiB/s, 9–12 waves/SIMD, 0 spills; GEMV variants v1–v5 tie; VGPR/occupancy
   capping is a pessimization. Do not tune occupancy further.
9. **Correctness bugs in this area are tails, caps, and dispatch tables**, not the inner
   `sdot4` loop: K-tail group dropping (#878, #956), Q8 batched row cap 64 with silent
   skipping (#732/#914), residual/SwiGLU dispatch tables drifting (#912), and
   head_dim-dependent FWHT thresholds (#871).
10. **RDNA2 is a second-class target**: no gfx1030 proof/ISA artifacts in `kernels/`
    (only gfx1100/1151/1201), no gfx1030 entries in the kernel-registry coverage test,
    and 3 of the 5 open gfx1030 issues are missing-dispatch/admission gaps.

---

## 2. gfx1030 kernel inventory at the pinned commit

`ls kernels/src | grep gfx1030` → 36 files. Grouped by family:

| Family | Files | Lines | Tile / block | LDS / WG | Launch bounds | Status |
|---|---|---|---|---|---|---|
| HFQ4 residual MMQ | body + `_x16`, `_x32`, `_x32_y64` | 199 + 4–13 | (32,4,1); `x32/y128`, `x32/y64`, `x16/y128` | 26,112 / 15,360 / 23,808 B | `(128, 2)` | production, auto-selected |
| HFQ3 residual MMQ | body + `_x8`, `_x16`, `_x32`, `_x32_y32`, `_x32_y64` | ~199 + 4–12 | same topology, 104 B group | 26,112 / 15,360 / 9,984 / 23,808 B | `(128, 2)` | `y64` production, `x8`/`y32` probes |
| HFQ4 qkv / qkvza / gate_up MMQ | body + `_x16`, `_x32` each | ~200 + 4–8 | fused 3/4/2-way, same tile scheme | 26,112 / 23,808 B | `(128, 2)` | production |
| HFQ3 qkv / qkvza / gate_up MMQ | body + `_x8`, `_x16`, `_x32`; gate_up also `_x32_y64`, `_x32_y96` | ~200 + 4–16 | fused, 104 B group | as above | `(128, 2)` | `y128` production, `y64`/`y96` gated probes |
| HFQ3 qkv / gate_up dp4a | `gemm_qkv_hfq3g256_dp4a.gfx1030.hip`, `gemm_gate_up_...` | 172 / 123 | `BATCH_TILE 16`, 32 threads | 0 | none | **negative result**, env-gated (`HIPFIRE_HFQ3_DP4A`) |
| HFQ4 GEMV | `gemv_hfq4g256.gfx1030.v1..v5.hip` | 113/116/99/116/147 | 32 threads, 1 row/WG | 0 | `(32,16)`, `(32,20)`, `(32,12)`, `(32,16)`, `(32,16)` | production, default v1, `HIPFIRE_RDNA2_VARIANT` selects 1–5 |
| Q8_0 MMQ | `gemm_q8_0_mmq.gfx1030.hip` (v7) | 183 | 32 rows × 128 cols, 256 threads | 6,400 B static | `(256, 1)` | **probe only**, no production dispatch |
| MQ2-Lloyd grouped MMQ | `gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1030.hip` | 190 | 16 rows × 16 slots, 32 threads | 0 (shfl + `v_perm`) | `(32, 4)` | **probe only**, grouped MoE |

Other kernels merely mention gfx1030 in a guard (`attention_flash_q8_0_tile`, fused qkv,
dot2 families); there are no gfx1030-specific variants there.

### LDS budget and WGs/CU (64 KiB LDS)

| `MMQ_Y` | `MMQ_X` | x_qs + x_dm + tile_y | Total | WGs/CU |
|---|---|---|---|---|
| 128 | 16 | 20,480 + 1,024 + 2,304 | 23,808 B | 2 |
| 128 | 32 | 20,480 + 1,024 + 4,608 | 26,112 B | 2 |
| 96 | 32 | 15,360 + 768 + 4,608 | 20,736 B | 3 |
| 64 | 32 | 10,240 + 512 + 4,608 | 15,360 B | 4 |
| 32 | 32 | 5,120 + 256 + 4,608 | 9,984 B | 6 |

Formula: `MMQ_Y*40*4 + MMQ_Y*8 + MMQ_X*36*4`. `X_STRIDE=40` is kept even for `MMQ_X=8/16`
(no per-tile stride switch), and `Y_STRIDE=36`.

---

## 3. Kernel design details worth copying / preserving

### 3.1 Shared MMQ body

- 128 threads `(32,4,1)`, `tid = threadIdx.y*32 + threadIdx.x`.
- Two 128-K windows per 256-element group; per window: load `tile_y` (Q8_1, 36 ints per
  column, including the 4 `half2` scale/sum words), load/unpack weights, one
  `__syncthreads`, four sub-blocks back-to-back, second `__syncthreads`.
- `#pragma unroll 1` on the `kg` and `window` loops; `#pragma unroll` on loaders, the
  `j0`/`i0` loops, and the 8-`sdot4` chain. No `asm` register fences in these bodies.
- Accumulators: `(MMQ_X/4) × (MMQ_Y/32)` floats — 16 (x16/y128, x32/y64), 32 (x32/y128),
  24 (x32/y96), 8 (x32/y32).
- Weight-side metadata is one `float2 (scale, zero)` per row, loaded once per group.
- X loader task count scales as `(MMQ_Y*16)/128`: 16 (y128), 12 (y96), 8 (y64), 4 (y32).
- Out-of-range rows clamp to `M-1`; out-of-range columns read `N-1` and write zero into
  the tile.

### 3.2 Metadata gate (the NaN fix)

Both bodies carry the same comment, which matches our implementation:

```c
// One thread per output row writes one float2 to x_dm[i] (sized MMQ_Y).
// Gating on `tid < MMQ_Y` (NOT the workgroup width 128) is required: when
// MMQ_Y=64, threads 64..127 must NOT write, since `x_dm[64..127]` overlaps
// the adjacent `tile_y` LDS region. Without this gate, the y64 variant
// corrupts the Q8_1 Y tile and emits NaN.
if (window == 0 && tid < MMQ_Y) { ... }
```

### 3.3 Bias folding

- HFQ4: `zp_eff = zp + 8.0f*sc` (4-bit center 8), 136 B group (8 B header + 128 B nibbles).
- HFQ3: `zp_eff = zp + 4.0f*sc` (3-bit center 4), 104 B group (8 B + 96 B trits).
- Inner math: `sum[idx] += scale_w*d_x*sumi + zp_eff*sum_x`, where `d_x`/`sum_x` are the
  Q8_1 scale and per-block sum. No per-sub-block bias correction inside the sdot4 chain.

### 3.4 HFQ3 unpack

Three-byte words at `gp + 8 + window*48 + chunk*3`, combined byte-by-byte into a uint24
(deliberately avoids a 4-byte load past the group), then 8 × 3-bit fields, `n - 4`, packed
into two int8 dwords. The `x8/x16/x32` wrappers all use the same 40-int LDS stride; the
body comment notes that a per-`MMQ_X` stride switch (40 vs 33) is possible if PMC shows
`ds_read` starvation at small `X`, but it was not needed.

### 3.5 Tile-sweep outcome

Per #300 and `docs/plans/gfx10_mq3_prefill.md`:

- Residual `x32/y64`: the only winning `MMQ_Y` change (LDS 26 KB → 15 KB, 2 → 4 WGs/CU).
- `y32`: **-20–40%** (dispatch/cache pressure from 4x WGs).
- `y96` (gate_up): negative (VGR pressure asymmetry).
- `gate_up y64`: negative — gate_up is not LDS-limited the same way.
- `x8`: never best, kept as a probe only.
- Note a reporting inconsistency: `docs/plans/gfx10_mq3_prefill.md` says the y64 residual
  microbench was **-5..-17% at N≥64** with only +1.5% at daemon level, while PR #315 says
  **+5–14% at N≥64**. Both are on gfx1031; treat the E2E win as small and measurement-
  dependent, not as a large kernel-level win.

---

## 4. Dispatch/selector facts

`crates/rdna-compute/src/gemm.rs` at the pinned commit:

- HFQ4 residual auto (`gemm_hfq4g256_residual_mmq_rdna2_auto`, ~line 9079):
  `batch_size <= 63 → x16/y128`, else `x32/y64`.
- HFQ3 residual auto (~line 7893): no sdot4 → dot2/fp16 fallback; `N <= 12 → dot2`;
  `N <= 63 → x16/y128`; else `x32/y64`.
- The doc comment above the HFQ3 function still says `13..127 → x16`, `>=128 → mmq_x=32`
  and was not updated when the boundaries/tile changed. This matches the stale numbers
  quoted in #299/#315. **Use the code, not the issue text.**
- HFQ3 dp4a is behind `HIPFIRE_HFQ3_DP4A=1` and measured ~15% slower than dot2.
- GEMV variant selection is `HIPFIRE_RDNA2_VARIANT` (default 1); v5 exists but all five
  tie at ~58 tok/s on 9B decode (#300).
- Q8_0 MMQ and MQ2-Lloyd grouped MMQ are launched only from channel-test examples; a
  source search finds no production caller.

---

## 5. Q8_0 v7 kernel (the one we ported)

File comment: *"Direct channel-test probe only; no production or A3B dispatch admits it."*
Launcher comment: *"intentionally not selected by any production dispatch path."*

Design:

- Grid `[ceil(M/32), ceil(N/128)]`, block 256, `__launch_bounds__(256,1)`.
- `row_local = tid & 15`, `slot_oct = tid >> 4` → each thread produces 2 rows × 8 cols;
  `acc0[8]`/`acc1[8]`.
- Weight LDS: `lds_w[32*9]` + `lds_ws[32]`; activation LDS: `lds_b[128*9]` + `lds_dx[128]`;
  6,400 B total, 9-dword stride to avoid gfx10 bank aliasing.
- Lanes 0..31 extract one 34-byte Q8_0 block each: `__byte_perm` funnel shifts for the
  unaligned `qs` (offset 2 in a 34-byte block), with a **2-byte** load for the final
  dword so the last row cannot read past the buffer.
- Scale is `(float)(*(const _Float16*)bp)`; accumulation is `acc += wsc*d_x*sumi`.
- Output layout is `Y[col*M + row]`; K must be a multiple of 128.

Our port matches this kernel (same structure, 64-bit-safe strides, `stride_col_dst`
instead of hard-coded `M`). Alignment invariant: `sh = (2 + 34*bi) & 3 ∈ {0,2}`, so the
single `sh==2` selector (`0x05040302`) is sufficient; this holds as long as the tensor
base is even. The hook is opt-in and its weight-base assert is 2-byte, matching that
invariant.

### 5.1 Local Q8_0 sweep vs stock llama.cpp MMQ (RX 6900 XT, 80 CU)

Harness: ggml GPU backend graph, `MUL_MAT` Q8_0 x F32, 200 async computes after 8
warmups, 3 rounds interleaved with the GPU idle (no KFD processes), median,
`GGML_HIPFIRE_Q8_0_MIN_N=1024` (stock) vs `=9` (hipfire). Both paths are llama.cpp
MMQ (`N >= 9`); MMVQ only covers `N <= 8`.

Median slowdown of hipfire vs stock across 5 shapes
(`M,K` = 1024x2048, 2048x2048, 4096x2048, 4096x4096, 8192x1024):

| N | hipfire vs stock | range |
|---:|---:|---|
| 9 | +194% slower | +119..+257% |
| 12 | +196% slower | +120..+256% |
| 16 | +194% slower | +118..+256% |
| 32 | +155% slower | +71..+197% |
| 64 | +91% slower | +29..+111% |
| 96 | +64% slower | +48..+76% |
| 128 | +36% slower | +24..+45% |

The hipfire Q8_0 kernel is never faster in this sweep, including at `N=128` where its
128-column tile is fully used. This is expected for a channel probe and is why the hook
defaults to off. A tuned Q8_0 kernel (or a different tile) would be a new workstream.

Incidental bug found while sweeping: the hook's contiguity check compared `s01` (in
34-byte type blocks) with `ne00` (elements), so it never fired and all earlier "hipfire"
test runs were actually stock. Fixed with `s01 * ggml_blck_size(type) == ne00`; the
opt-in path then passed 113/113.

---

## 6. Performance and quality record (upstream gfx1031; local gfx1030)

All public RDNA2 numbers in this repo are **gfx1031 (RX 6700 XT, Navi22, 40 CU, 96 MB IC)**.
The local partition is **gfx1030 (RX 6900 XT) and reports 80 CUs / 320 SIMDs via
`rocminfo`** (128 MB IC, 512 GB/s). This is host-specific: re-derive occupancy targets
from `rocminfo` on a different gfx1030 partition. Directionally useful; magnitudes will
not transfer.

| Source | Result |
|---|---|
| #299 probe | LDS-tiled MMQ vs dot2/fp16 at pf=240: MQ3 290→350 (+21%), MQ4 288→351 (+22%) |
| #298 | Full HFQ3 MMQ family: LRU 290→545 (+88%); cumulative 9.73x; KLD slice-mean 0.191 |
| #315 | HFQ4 family: 9B MQ4 pp128 289→719 (+149%); y64 +5–14% claim |
| #335 default-on | 4B MQ4 pp32 476→847 (+78%), pp128 531→1222 (+130%); 9B MQ4 pp32 277→511 (+84%), pp128 289→682 (+136%); 9B MQ3 pp128 213→670 (+214%); pp512 ~1245 |
| #335 accuracy | MQ4 KLD 0.0755→0.0769 (+1.9%, noise); MQ3 0.2227→0.2226 (−0.06%) |
| #300 final | 88.6% of 9B prefill = 2 MMQ GEMMs at 15–20 GiB/s; kernels 9–12 waves/SIMD, 0 spills; not occupancy-bound; GEMV v1–v5 tie ~58 tok/s (~80% peak BW) |
| #300 VGPR test | MQ3 GEMV v2: VGPR 61→38 as predicted, decode 62.6→61.9 tok/s (slower); reverted |
| `docs/perf-checkpoints/2026-08-18-...` | RV6: capping VGPRs 97→96/84/80/76/72/64 to raise occupancy 12→16 made a bandwidth-bound GEMV 7–27% slower; uncapped fastest with 0 spills |

`__launch_bounds__` lesson (same doc set): the second argument is **advisory** — on a
GFX12 GEMV the compiler emitted the same VGPR/occupancy for every value tried and only
reordered code; shipped value was within 1.5% of best. This matches our local finding that
literal `(128,4)` and `(128,2)` produced identical gfx1030 code.

---

## 7. Issue findings relevant to this port

| Issue | State | Finding | Implication |
|---|---|---|---|
| [#298](https://github.com/warpfront/hipfire/pull/298) | closed | HFQ3 MMQ family + tile auto-selector; +88% long prefill; dot2 fallback at N≤12 and unaligned M | Confirms the shared-body/tile approach |
| [#299](https://github.com/warpfront/hipfire/issues/299) | closed | HFQ4 MMQ follow-up; LDS tiling is the dominant factor, not unpack cost; `x8` never best | Confirms 3-bit/4-bit share topology |
| [#300](https://github.com/warpfront/hipfire/issues/300) | closed | Remaining levers exhausted; exact negative results for y32/y96/gate_up y64, hipGraph prefill, epilogue fusion, persistent kernels, VGPR 110→88, MMQ_Y=256, HFQ3 GEMV v2 | Do not re-open occupancy/ILP tuning |
| [#302](https://github.com/warpfront/hipfire/issues/302) | closed | +19% KLD was a stale baseline; fresh dot2, MMQ, MMQ-layer-gated and gfx1151 WMMA all within 0.07% | No need for Q8_K-style finer X quantization |
| [#315](https://github.com/warpfront/hipfire/pull/315) | closed | HFQ4 family + qkvza split routing; y64 residual; y96/y64 gate_up and y32 negative | Split routing is for fused projections only |
| [#335](https://github.com/warpfront/hipfire/pull/335) | closed | MMQ default-on gfx10; KLD-neutral; perf table above | Kernel is production-worthy |
| [#533](https://github.com/warpfront/hipfire/issues/533) | **open** | MQ2G256 has no batched prefill on any arch → 22x gap vs llama.cpp (RX 6800 CT, gfx1030); `is_batchable_la` excludes it | Upstream MQ2 prefill is **not** solved; our MQ2 kernel is ahead of upstream |
| [#558](https://github.com/warpfront/hipfire/issues/558) | **open** | Contributor fork has MQ2 prefill + Q8_0 MMQ batched for A3B on gfx1030; comment measured pp128 89→1242, pp512 88→1781, pp1024 88→1911; asks for a `perf/gfx1030` branch | Still unmerged as of the pin; treat as out-of-tree evidence |
| [#478](https://github.com/warpfront/hipfire/issues/478) | open | gfx1030 + MQ6 panics: `no implementation for GemvMq6G256Prerotated`; MQ6 is gfx11/12-only | Format admission must be arch-gated, not `unwrap()` |
| [#771](https://github.com/warpfront/hipfire/issues/771) | closed | gfx1030 MQ4V2 DFlash batched lm_head: "scalar fallback has no v2 source" | Every fast-path needs a working scalar fallback |
| [#732](https://github.com/warpfront/hipfire/issues/732) | closed | Q8 batched kernel `MAX_BATCH=64`; larger N crashed MTP (70/76/256) | Cap must be enforced/chunked at the entry point |
| [#914](https://github.com/warpfront/hipfire/pull/914) | closed | Hardened #732: `gemm_q8_0_batched` now chunks above 64 itself; kernel **silently skips** rows past 64 | Never rely on callers to respect an implicit cap |
| [#878](https://github.com/warpfront/hipfire/issues/878) | closed | MQ4 expanded MoE down GEMV read only the first 512 K for K≠512 → whitespace loops | K-tail handling; add K≠512 parity tests |
| [#956](https://github.com/warpfront/hipfire/issues/956) | closed | Tail-less MoE gate_up dropped partial quads (`K/256 % 4 != 0`) → zero gate/up | Same class: group/quad tails |
| [#912](https://github.com/warpfront/hipfire/pull/912) | closed | Legacy MQ3 `swiglu_residual` had no dispatch arm because two tables drifted | Single dtype→kernel table + exhaustive test |
| [#871](https://github.com/warpfront/hipfire/pull/871) | closed | FWHT 2/4-bit K used head_dim-128 thresholds at 256 → ~40% extra MSE | Norm/scale must track the rotated block count |
| [#223](https://github.com/warpfront/hipfire/issues/223) / [#531](https://github.com/warpfront/hipfire/issues/531) / [#532](https://github.com/warpfront/hipfire/pull/532) | closed | `RDNA2 GEMV variant: v1` `eprintln!` leaked into model output/logs | Keep variant selection silent or behind a flag |
| [#490](https://github.com/warpfront/hipfire/issues/490) | open | 128 MiB pointer-chase buffer fits in Infinity Cache → gfx1030 latency probe measures cache, not DRAM | Local roofline/latency numbers on the 6900 XT need >128 MiB buffers |
| [#669](https://github.com/warpfront/hipfire/issues/669) | open | `hipfire diag`/`serve` ignore `hardware.devices`, always GPU 0 (RX 6800 XT was device 1) | Not kernel-level, but affects local multi-GPU testing |
| [#12](https://github.com/warpfront/hipfire/issues/12) / [#80](https://github.com/warpfront/hipfire/pull/80) | closed | Early gfx1030 support and the first gfx1030 speed baseline | RDNA2 has been a supported but secondary target since day one |

---

## 8. Corrections to the proposed "Master Architecture & Kernel Analysis"

The blueprint supplied after the search contains several items that do not match the pinned
tree. Applying it literally would create a parallel kernel copy and duplicate symbols.

| Blueprint claim | Pinned-tree reality |
|---|---|
| `kernels/include/radiowave/hip.h` | Does not exist. `radiowave` is a Rust autotuning crate (`crates/radiowave`); no C++ include tree under `kernels/`. |
| `gemm_mq4g256v2_residual_mmq.hip` is the RDNA2 residual MMQ | It is a **gfx1100/gfx1151 WMMA** kernel (`MMQ_X=128`, `HIPFIRE_RDNA3`). The gfx1030 MMQ is `gemm_hfq4g256_residual_mmq_body.cuh` + wrappers. |
| `gemm_mq3g256_mmq.gfx1030.hip` | Does not exist. Actual: `gemm_hfq3g256_residual_mmq_{x8,x16,x32,x32_y32,x32_y64}.gfx1030.hip`. |
| `gemm_mq2g256_lloyd_moe_grouped_mmq.hip` | The gfx1030 file is `..._grouped_mmq.gfx1030.hip`; it is a **grouped-MoE probe**, not a dense MQ2 GEMM. |
| `__launch_bounds__(128, 4)` on the MMQ kernel | Upstream uses `(128, 2)` for every gfx1030 MMQ body; `(128,4)` appears in 2 of 1,278 files, neither of these. Local test: literal 4 vs 2 produced identical VGPR/occupancy. |
| `pin_reg` fences after each sub-block | Zero `asm volatile("" : "+v"(...))` occurrences in the gfx1030 MMQ bodies, Q8_0 kernel, or GEMV v5. The bodies rely on `#pragma unroll 1` outer loops; we measured 0 spills without fences. |
| Static `__shared__` arrays sized `[32][40]` / `[36][32]` | Upstream uses `extern __shared__` with per-launch size from `MMQ_Y*40*4 + MMQ_Y*8 + MMQ_X*36*4`. Static sizing would break the y64/y96 variants. |
| "MQ3 uses buffer resource descriptors (`buffer_load_dword`)" | The gfx1030 MMQ/GEMV kernels use ordinary global loads and LDS; `make_buffer_rsrc`/raw buffer loads appear in 44 files, none of the gfx1030 MMQ family. |
| `rdna2/*.hip` + CMake glob | Would duplicate the already-wired single-TU sources (`mmq-hipfire-dispatch.cu` is compiled by the existing `../ggml-cuda/*.cu` glob with gfx1030 flags). A second glob would compile the same kernels twice. |
| FWHT description as "256-point in `mq_rotate_x_128_v2.hip`" | `mq_rotate_x_128_v2.hip` is a **128-point** transform (qt=53, `__launch_bounds__(32,16)`) with `ds_swizzle` butterflies. Our 256-point FWHT lives in `ggml-hip-rdna2-hipfire.cuh` and follows the older hipfire design. |

Everything else in the blueprint (sdot4 inner loop, 2D LDS tiling, no WMMA, X_STRIDE 40 /
Y_STRIDE 36, zp folding, 0-byte-LDS FWHT) is consistent with the tree.

---

## 9. Concrete follow-ups for this tree

1. **Keep the current routing** (`N ≤ 63 → x16/y128`, else `x32/y64`); it matches the code
   at the pin. Ignore the `13..127 / ≥128` numbers in #299/#315.
2. **Keep the metadata gate** exactly as is (`window == 0 && tid < MMQ_Y`).
3. **Do not add y32/y96 variants**; both lost upstream. y64 is the only occupancy win and
   only for the residual shape.
4. **Stop tuning `__launch_bounds__`/VGPR caps.** LDS is the limiter (2/4 WGs), the runtime
   is not occupancy-bound, and capping VGPRs is measured slower upstream.
5. **Keep the Q8_0 hook opt-in.** It is numerically correct (`test-backend-ops` 113/113
   with `GGML_HIPFIRE_Q8_0_MIN_N=64`) but measured 24-257% slower than the stock Q8_0 MMQ
   at every tested N (9..128), consistent with upstream calling it a channel probe. The
   default is therefore the stock path; a tuned kernel is a separate workstream.
6. **Q8_0 alignment invariant is now asserted at dispatch** (`GGML_ASSERT` on the
   weight base) and documented in the kernel; no code change was needed for the
   `sh ∈ {0,2}` selector. The hook passes `test-backend-ops` 113/113 when enabled
   (`GGML_HIPFIRE_Q8_0_MIN_N=64`); it stays disabled by default on the perf result above.
7. **MQ2-Lloyd hardening is ported.** The grouped kernel now bounds `expert_id`
   against `n_experts` and clamps `my_x_row` to `x_src_rows`, matching the pinned
   commit's newer guards. K≠512 / partial-quad tail tests remain a follow-up because
   no ggml MQ2 type is registered yet.
8. **MQ3 small-batch dot2 routing is ported.** `N <= 12` now runs the dot2 kernel with
   an F32→F16 activation copy (upstream `ensure_fp16_x`), mirroring
   `gemm_hfq3g256_residual_mmq`. It is compiled and spill-free (VGPR 23, 0 spills) but
   unreachable until MQ3 is a registered `ggml_type`.
9. **Do not pursue dp4a on RDNA2**; the pinned tree retains it only as a gated negative
   result.
10. **For local gfx1030 measurements**, size latency probes above the 128 MiB Infinity Cache
    and prefer end-to-end throughput over reported occupancy (see #490 and the VGPR-cap
    checkpoint).

### 9.1 Rust launcher logic → C++ (this tree)

| hipfire Rust | C++ counterpart |
|---|---|
| `gemm_hfq4g256_residual_mmq_rdna2_auto` | `mmq-hipfire-dispatch.cu`: `N<=63 → x16/y128`, else `x32/y64` (already matched) |
| `gemm_hfq3g256_residual_mmq` (dot2 ≤12, x16 ≤63, else x32_y64) | **ported**: early `N<=12` dot2 branch in dispatch |
| `ensure_fp16_x` | **ported**: `convert_f32_to_f16_gfx1030_kernel` |
| `ensure_q8_1_mmq_x` | already `quantize_q8_1_mmq_ds4_gfx1030` |
| `gemm_q8_0_batched` 64-row chunking (#732/#914) | not applicable: the ported Q8_0 MMQ tiles N by 128 and has no row cap |
| MQ2 grouped launcher index guards | **ported**: `n_experts` bound + `x_src_rows` clamp |
| `gemv_mq2g256_lloyd_residual` | **ported**: `gemv_mq2g256_lloyd_gfx1030_kernel<RESIDUAL_ADD>`, used at `N==1` |
| GEMV `HIPFIRE_RDNA2_VARIANT` v1–v5 | intentionally not ported: all five tie (#300) |
| fused qkv / qkvza / gate_up launchers | intentionally not ported: llama.cpp fuses at graph level |

### 9.2 Implementation status (2026-10-25)

| Finding | Status |
|---|---|
| `N <= 63 → x16/y128`, else `x32/y64` | already matched (MQ4 + MQ3 MMQ) |
| `tid < MMQ_Y` metadata gate | already present in both MMQ headers |
| no y32/y96 / no dp4a / no GEMV v2–v5 | intentionally not ported (upstream negatives) |
| MQ2 `n_experts` + `x_src_rows` bounds | **added** (`mmq-hipfire-mq2-q8-gfx1030.cuh`) |
| MQ2 decode GEMV | **added** (`gemv_mq2g256_lloyd_gfx1030_kernel`, VGPR 24) |
| MQ3 `N <= 12` dot2 fallback | **added** (F32→F16 kernel + selector in dispatch) |
| Q8_0 2-byte alignment invariant | **asserted** in dispatch, documented in kernel |
| Q8_0 hook default | **disabled** (opt-in `GGML_HIPFIRE_Q8_0_MIN_N`); measured slower than stock |
| `-DLLAMA_FATAL_WARNINGS=ON` | passes for `ggml-hip`; no hipfire warnings |
| launch bounds / VGPR caps | left at `(128,2)`; `(128,4)` measured identical |

Verification: `cmake --build build-hip-gfx1030` clean; `test-backend-ops -b ROCm0
-o MUL_MAT -p q8_0` 113/113 and `-o MUL_MAT_ID -p q8_0` 75/75; resource pass still
0 spills on every hipfire kernel (MQ4 64/79, MQ3 69/71, Q8_0 75, MQ2 72, MQ2 GEMV 24,
dot2 23, convert 4 VGPR).

---

## 10. Source map

- MMQ bodies: `kernels/src/gemm_hfq{3,4}g256_residual_mmq_body.cuh`,
  `gemm_{qkv,qkvza,gate_up}_hfq{3,4}g256_mmq_body.cuh`.
- Tile wrappers: `kernels/src/gemm_*_x*.gfx1030.hip`.
- GEMV variants: `kernels/src/gemv_hfq4g256.gfx1030.v{1..5}.hip`.
- Q8_0 v7: `kernels/src/gemm_q8_0_mmq.gfx1030.hip`.
- MQ2 probe: `kernels/src/gemm_mq2g256_lloyd_moe_grouped_mmq.gfx1030.hip`.
- Selectors/launchers: `crates/rdna-compute/src/gemm.rs` (HFQ4 auto ~9079, HFQ3 auto
  ~7893, Q8_0 launcher ~31199, MQ2 launcher ~16583), `crates/rdna-compute/src/kernels.rs`
  (GEMV variant table ~7224).
- Perf/design records: `docs/plans/gfx10_mq3_prefill.md`, `docs/plans/mq3_gfx10_perf.md`,
  `docs/perf-checkpoints/2026-08-18-vgpr-capping-is-a-pessimization-occupancy-is-not-throughput.md`,
  `docs/investigations/2026-08-18-autotune-the-occupancy-knobs-radiowave-already-has-the-frame.md`.
- Local port: `ggml/src/ggml-cuda/ggml-hip-rdna2-hipfire.cuh`,
  `mmq-hipfire-mq4-gfx1030.cuh`, `mmq-hipfire-mq3-gfx1030.cuh`,
  `mmq-hipfire-mq2-q8-gfx1030.cuh`, `mmq-hipfire-dispatch.cu`, `mmq.cu`,
  `ggml/src/ggml-hip/CMakeLists.txt`.

---

## 11. Tracker re-scan (2026-10-25): the missing gfx1030 kernel is GQA-shared decode attention

Sources: local clone `bfb90b9c8` plus the GitHub tracker (`warpfront/hipfire`); docs
`docs/plans/gfx10_mq3_prefill.md`, `docs/plans/mq3_gfx10_perf.md`; issues/PRs
[#80](https://github.com/warpfront/hipfire/issues/80),
[#298](https://github.com/warpfront/hipfire/pull/298),
[#300](https://github.com/warpfront/hipfire/issues/300),
[#315](https://github.com/warpfront/hipfire/pull/315),
[#335](https://github.com/warpfront/hipfire/pull/335),
[#533](https://github.com/warpfront/hipfire/issues/533),
[#558](https://github.com/warpfront/hipfire/issues/558),
[#561](https://github.com/warpfront/hipfire/pull/561),
[#771](https://github.com/warpfront/hipfire/issues/771),
[#774](https://github.com/warpfront/hipfire/pull/774).

### 11.1 What actually improved on gfx1030/gfx1031

| Work | Result | Evidence |
|---|---|---|
| HFQ3 (MQ3) prefill MMQ family | 9B MQ3 pp240 290 -> 545 tok/s (+88%); 56 -> 545 cumulative (9.73x); KLD 0.191 | #298 |
| HFQ4 (MQ4) prefill MMQ family | 4B pp128 531 -> 1276 (+140%); 9B pp128 289 -> 719 (+149%); 9B pp32 +70% | #315 |
| MMQ default-on | 9B MQ3 pp128 213 -> 670 (+214%); 9B MQ4 pp128 289 -> 682 (+136%); 4B MQ4 pp128 531 -> 1222 (+130%); KLD-neutral | #335 |
| Residual `MMQ_Y=64` | +5-17% at N>=64; only ~+1.5% daemon-level | #300/#315 |
| GQA-shared Q8 decode tile (gfx1100/1151/1201) | one workgroup per (KV head, tile) serves all 6 Q heads; K/V read once | `attention_flash_q8_0_tile_gqa.gfx{1100,1151}.hip` |
| Tiled Q8 flash decode (gfx11/12) | long-context decode 7.73-15.88x (legacy Q8 path), PP4K/8K 1.65-2.69x | #561 |
| Qwen3.8 27B on gfx1030 | ~18.5-19.5 tok/s decode with DFlash off, MQ4V2, V620 | #771 |

All public RDNA2 numbers are **gfx1031 (RX 6700 XT, 40 CU)** unless stated; the 6900 XT
partition is gfx1030, 80 CU, 128 MB Infinity Cache. The only 6900 XT baseline in the
tracker is #80: 9B MQ3 decode 66.8 tok/s, pp128 550.4; 27B MQ4 OOM on 16 GB because the
weights alone are 13.95 GiB.

### 11.2 What did not work (do not port)

- **Q8_0 MMQ v7**: channel probe, 24-257% slower than stock at N=9..128; no production
  dispatch. Our local sweep reproduces this.
- **HFQ3 GEMV v2 high-occupancy**: VGPR 61 -> 38 as predicted, but 9B MQ3 decode
  62.6 -> 61.9 tok/s (slower); reverted. GEMV v1-v5 tie ~58 tok/s (~80% peak BW).
- **MQ3 dp4a**: ~15% slower than dot2; bit extraction dominates.
- **`MMQ_Y=32/96`**: -20-40% / negative; residual y64 is the only tile win.
- **PreFill hipGraph**: predicted +10-20%, rocprof showed only 4.3% addressable startup.
- **VGPR caps / occupancy**: slower; kernels already 9-12 waves/SIMD, 0 spills, not
  occupancy-bound.
- **Final #300 verdict (gfx1031)**: 88.6% of prefill is two MMQ GEMMs at 15-20 GiB/s;
  "past the wall needs spec-decode/lower quant, not kernel tuning."

### 11.3 Open gfx1030 gaps

- **MQ2G256 batched prefill** (#533): no batched GEMM on any arch; llama.cpp IQ2_XXS
  prefill 1923 tok/s vs hipfire MQ2 88 tok/s (22x). An unmerged fork (#558) claims
  pp128 89 -> 1242, pp512 88 -> 1781, pp1024 88 -> 1911 on gfx1030.
- **MQ6 on gfx1030** (#478): `GemvMq6G256Prerotated` missing; MQ6 is gfx11/12-only.
- **Qwen3.8 + DFlash on gfx1030** (#771): missing MQ4V2 scalar fallback for batched
  lm_head; DFlash crashes, DFlash-off decode works.
- **Infinity Cache latency probe** (#490): 128 MiB pointer chase fits in IC; size
  >512 MiB for gfx1030 DRAM latency.
- **GPU selection** (#669): `diag`/`serve` always use GPU 0.

### 11.4 The kernel that is actually missing on gfx1030

The local model is `qwen35`, 65 blocks: **17 full-attention layers + 48 SSM layers**,
24 Q heads, 4 KV heads (**GQA 6**), head_dim 256. That is exactly the geometry of
hipfire's gfx1100 GQA decode tile (`head_dim 256, group 6, tile 128`).

- hipfire has a GQA-shared Q8 tile for **gfx1100, gfx1151, gfx1201** only. The generic
  `attention_flash_q8_0_tile` works on gfx1030 but its grid is `[n_heads, n_tiles]`:
  **one Q head per workgroup**, so it reads/dequantizes the same K/V six times.
- llama.cpp's HIP FA is worse for this shape: `ggml_cuda_get_best_fattn_kernel` returns
  `BEST_FATTN_KERNEL_VEC` for quantized K/V with `Q->ne[1] <= 2`. `fattn-vec.cuh`
  indexes one `head` per block and points K/V at `head / gqa_ratio`; it does not share
  K/V across the six Q heads.
- Vulkan's FA shader does use `gqa_ratio` / `gqa_iq1` tiling. The measured HIP attention
  slope is **0.637 us/KV-token vs Vulkan 0.111**, a ratio of **5.7x** -- essentially the
  GQA ratio of 6.

This is the concrete kernel gap: **gfx1030 has no GQA-shared decode tile, and the HIP FA
dispatcher does not select one.** It is not a weight-kernel problem, which is why the
MQ3/MQ4/Q8_0 ports did not move 112k decode.

### 11.5 Port target and expected result

Port `attention_flash_q8_0_tile_gqa.gfx1100.hip` to gfx1030:

- One 256-thread workgroup per (KV head, tile), serving all 6 Q heads.
- Q8_0 K/V first (q8 V fits at 112k on HIP: 13.57 tok/s measured); add a q5_0 V
  dequant variant after.
- gfx1030 has no RDNA3 unaligned-dword mode; replace the 34-byte-row dword loads with
  two aligned loads / byte permutes (the Q8_0 MMQ v7 kernel already does this).
- Keep the same partials + reduce layout; 0 spills; tile 128; 8 waves.

Expected, from the measured 71 ms attention term at 112k and 20 ms at 32k:

| sharing | 112k attention | 112k total | 112k tok/s | 32k tok/s |
|---|---:|---:|---:|---:|
| none (today) | 71 ms | 101 ms | 9.9 | 19.8 |
| 2x (tile ncols2=2) | 36 ms | 66 ms | 15.2 | 25.5 |
| 4x | 18 ms | 48 ms | 20.9 | 28.4 |
| 6x (hipfire GQA-6) | 12 ms | 42 ms | 23.8 | 29.9 |

That is the only kernel project with a >2x payoff at 112k. It would make HIP match
Vulkan at 112k and beat it at 32k; without it, no amount of MQ/GEMV/MMQ tuning closes
the gap.

---

## 12. gfx1030 GQA-6 kernel: port result (2026-10-25)

The port was implemented in `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu` and selected from
`fattn.cu` for RDNA2, head_dim 256, GQA ratio 6, single-token decode, Q8_0 K and
Q8_0/Q5_0 V. `GGML_HIP_GQA_GFX1030=0` restores the old VEC path.

### 12.1 Measured

RX 6900 XT, `-fa 1 -fitt 128`, q8_0 K / q5_0 V:

| depth | old VEC | GQA gfx1030 | Vulkan | delta vs VEC |
|---|---:|---:|---:|---:|
| 32k | 19.72-19.76 | **27.27-27.37** | 29.24 | +38% |
| 112k | 9.88-9.91 | **22.16-22.47** | 23.23 | +125% |

q8_0 V: 22.39 at 112k, 27.28 at 32k -- same as q5_0 V. Native q5 V works; the old
q8_0/q5_0 VEC fallback converted K/V to f16 and was a major part of the original
112k collapse.

Correctness: 4/4 targeted `FLASH_ATTN_EXT` cases (4k q8/q8, 4k q8/q5, 5003 tail q8/q5,
32768 q8/q5) pass. The full suite was last run before the final sdot4/tail changes.

### 12.2 What made the difference

1. **GQA sharing.** One 6-warp workgroup per (sequence, KV head, tile) serves all six
   Q heads; K/V is read and dequantized once instead of six times.
2. **`sdot4` QK.** Q is quantized to q8_0 once per head; K stays raw Q8_0 bytes in LDS;
   `__builtin_amdgcn_sdot4` replaces the f32 FMA chain. This was the largest single
   speedup (18.8 -> 22.3 at 112k).
3. **Native Q5_0 V.** The dequant fix is the q5_0 layout: values 0..15 are the low
   nibbles of `qs[0..15]`, values 16..31 are the high nibbles, with `qh` bits 0..31.
4. **LDS sizing.** 24-key chunks keep K/V/Q at ~20 KiB, allowing three 192-thread
   workgroups per CU (`launch_bounds(192,3)`); 32-key chunks need 30 KiB and drop to two.
5. **Tail correctness.** The inner chunk must be clipped to `min(k0+TILE, k_max)`; with
   CHUNK=24 and TILE=256, keys 256..263 of the last chunk otherwise get counted twice.

### 12.3 What did not help

- `v_dot2_f32_f16` with f16 Q was slower than f32 Q / half2 K (15.2 vs 17.5 at 112k).
- 8-key-aligned V reads (contiguous per lane) were not faster than the strided layout.
- CHUNK=16, WARPS=8, TILE=128, and full/partial V unrolls were neutral or slower.
- The old q5-to-f16 fallback was the wrong fix; native q5 dequant is required.

### 12.4 Hypothesis for the residual gap

HIP is now 4% behind Vulkan at 112k and 7% behind at 32k. The depth fit is:

- HIP: slope 0.103 us/KV-token, fixed 33.3 ms.
- Vulkan: slope 0.111 us/KV-token, fixed 30.7 ms.

HIP's slope is slightly better; the residual is a ~2.7 ms **fixed** per-token cost.
Candidates: redundant per-block Q quantization (240 blocks/layer re-quantize the same
six heads), partials/combine traffic, load/compute serialization (two `__syncthreads`
per 24-key chunk), and tail scheduling. Falsification tests and the decision gate are in
`PLAN_GQA_DECODE_GFX1030.md` section 10. The cheapest next lever is a per-token Q
pre-quantization pass; the next is double-buffering load/compute. If those do not move
the fixed term, the residual is not worth another kernel rewrite.
