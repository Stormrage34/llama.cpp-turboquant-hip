# Plan: hipfire RDNA2 kernels in ggml-hip (local gfx1030)

Status: source of truth for this build. Supersedes the earlier "144 CU" draft.
Companion evidence: `HIPFIRE_FINDINGS_GFX1030.md`.

## 1. Verified local baseline

```text
$ rocminfo | grep -E "Marketing Name|Compute Unit"
Marketing Name: AMD Radeon RX 6900 XT
Compute Unit:   80
```

**RX 6900 XT, local gfx1030 build: 80 CUs / 320 SIMDs.**

This host reports 80 CUs. All occupancy and grid-fill calculations in this plan use
**80 CUs / 320 SIMDs**. If a different gfx1030 partition is tested later, re-derive the
targets from `rocminfo`; do not generalize this number to every gfx1030 SKU.

| Property | Value for this host |
|---|---:|
| Target | RX 6900 XT, local gfx1030 |
| CUs | 80 |
| SIMDs (4/CU) | 320 |
| Wavefront size | 32 |
| LDS per CU | 64 KiB |
| Infinity Cache | 128 MiB |
| Memory | 16 GB GDDR6, 512 GB/s |
| ISA | `v_dot4_i32_i8`, `v_dot2_f32_f16`, `v_perm_b32`, `ds_swizzle_b32`, buffer loads |
| WMMA | not applicable on RDNA2 |

Grid-fill examples:

| Case | WGs | Resident slots | Fill |
|---|---:|---:|---:|
| `M=4096, N=64`, `<32,64>` | 128 | 320 (4 WGs/CU) | 40% |
| `M=4096, N=128`, `<32,64>` | 256 | 320 | 80% |
| `M=4096, N=256`, `<32,64>` | 512 | 320 | saturated |
| `M=4096, N=64`, `<16,128>` | 128 | 160 (2 WGs/CU) | 80% |

The `N=64` underfill is real but modest on 80 CUs. Split-K is not automatically
justified; it stays measurement-gated (§8).

## 2. llama.cpp dispatch reality

```text
MMVF → MMF → MMVQ if ne11 <= 8 → MMQ → hipBLAS
```

- `N = 1..8` is normally handled by **MMVQ** (`MMVQ_MAX_BATCH_SIZE = 8`), not MMQ.
- `N >= 9` can reach `ggml_cuda_mul_mat_q` and the hipfire hook.
- `ggml_cuda_mul_mat_q` already asserts `src1->type == F32`; no new FP32 contract flag.
- The hook therefore must not claim `N == 1` GEMV or `N = 2..8` dot2 wins. Those are
  MMVQ workstream items.

Current hook (opt-in only):

```cpp
RDNA2 && ids == nullptr && type == Q8_0 && ne00 % 256 == 0
&& ne11 >= GGML_HIPFIRE_Q8_0_MIN_N   // no env var -> disabled
&& single batch && contiguous s01/s1
```

The default is **disabled**: the hipfire Q8_0 kernel is a 128-column channel probe and
measured 24-257% slower than the stock Q8_0 MMQ at every N from 9 to 128 on this GPU
(§12). `GGML_HIPFIRE_Q8_0_MIN_N` exists only to re-run that sweep.

The contiguity check must compare **elements**, not type blocks:
`s01 * ggml_blck_size(type) == ne00`. The original `s01 == ne00` compared 34-byte blocks
with elements and silently kept every run on the stock path.

## 3. Revised MMQ dispatch table

Written around the fact that the hook sees `N >= 9`:

| `N` | MQ4 | MQ3 | MQ2-Lloyd | Q8_0 |
|---:|---|---|---|---|
| 9..12 | MMQ `<16,128>` (padding ≤ 1.8x) | dot2 preferred | 16x16 Lloyd acceptable | stock (hipfire loses) |
| 13..63 | MMQ `<16,128>` | MMQ `<16,128>` | 16x16 Lloyd | stock (hipfire loses) |
| >= 64 | MMQ `<32,64>` | MMQ `<32,64>` | 16x16 Lloyd | stock (hipfire loses 35-75%) |
| low-WG | split-K only after measurement | same | same | same |

Notes:

- MQ3 `9..12 → dot2` is the only small-batch route upstream justifies.
- MQ4 `<16,128>` at 9..12 wastes at most 1.78x columns; acceptable as a first cut.
- MQ2-Lloyd's 16x16 tile is naturally closer to small N than a 128-column tile.
- MQ2/MQ4 `N==1` GEMV code stays for forced-MMQ tests, not as a production claim.

## 4. Resource contract (measured on this build)

| Kernel / path | Threads | LDS | VGPR | Spills | Occ (waves/SIMD) |
|---|---:|---:|---:|---:|---:|
| Activation quantizer | 32 | 0 | 31 | 0 | 16 |
| F32→F16 convert | 256 | 0 | 4 | 0 | 16 |
| MQ4 GEMV | 32 | 0 | 39 | 0 | 16 |
| MQ2 GEMV | 32 | 0 | 24 | 0 | 16 |
| MQ3 dot2 | 32 | 0 | 23 | 0 | 16 |
| MQ4 `<16,128>` | 128 | 23,808 | 64 | 0 | 16 |
| MQ4 `<32,64>` | 128 | 15,360 | 79 | 0 | 12 |
| MQ3 `<16,128>` | 128 | 23,808 | 69 | 0 | 12 |
| MQ3 `<32,64>` | 128 | 15,360 | 71 | 0 | 12 |
| MQ2-Lloyd | 32 | 0 | 72 | 0 | 12 |
| Q8_0 MMQ | 256 | 6,400 | 75 | 0 | 12 |

Rules:

1. **0 spills/scratch.**
2. VGPRs within the measured table (±8 regression triggers review).
3. LDS within `MMQ_Y*40*4 + MMQ_Y*8 + MMQ_X*36*4` (y64 = 15,360; y128/x16 = 23,808;
   y128/x32 = 26,112).
4. Occupancy is taken from the resource pass, not inferred from `__launch_bounds__`.

Corrections to the old contract:

- MQ4/MQ3 are **64/79** and **69/71**, not 89..107.
- Q8_0 LDS is **6,400 B**, not 6,336 B.
- Q8_0 is not "1 WG/CU by LDS"; LDS allows 10, the 256-thread/75-VGPR shape is
  register/occupancy-limited.
- `(128,4)` did **not** change codegen vs `(128,2)` for `<32,64>` (79 VGPR, 12 waves/SIMD
  both). The hint is kept for documentation only.
- Quantizer `(32,8)` is sufficient; `(32,16)` is not justified. `minBlocks` is a WG/CU
  hint, not a wave count.

## 5. Build integration

1. **Do not append the dispatch TU again.** It is already in `GGML_SOURCES_ROCM` via
   `file(GLOB ... "../ggml-cuda/*.cu")`.
2. Use list-form `COMPILE_OPTIONS` via `set_property`:

   ```cmake
   set_property(SOURCE "../ggml-cuda/mmq-hipfire-dispatch.cu" PROPERTY COMPILE_OPTIONS
       "-mno-wavefrontsize64"
       "-Wno-pass-failed")
   ```

3. Correct flag is `-mno-wavefrontsize64`; remove `--wavefront-size=32`.
4. Use conservative parallelism for HIP builds:

   ```bash
   cmake --build build-hip-gfx1030 -j4
   ```

   Observed: `-j24` crashed clang-22 on unrelated upstream template instances; the same
   files compile cleanly in isolation and the full build succeeds at `-j4`. Do not
   attribute parallel-build crashes to hipfire kernels without isolated reproduction.
5. Multi-arch caveat: the per-file options apply to every arch in the same compilation;
   a `gfx1030;gfx90a` build would force wave32 on CDNA. Split the TU per arch or exclude
   hipfire from CDNA builds.

## 6. Alignment contract

- MQ weights: group strides 136/104/72 B, all 4-byte multiples; header loads need a
  4-byte-aligned base.
- Q8_0: **2-byte alignment is sufficient.** `block_q8_0` is 34 B, so for any even base
  the qs funnel-shift class is `sh ∈ {0, 2}`. Assert `A_q8 % 2 == 0`; do **not** require
  256 B.
- Q8_1 scratch: `block_q8_1_mmq_hipfire` is 144 B (16-byte multiple); pool allocations
  are at least 16-byte aligned.

## 7. FWHT sign strategy: benchmark first

Keep the current PRNG path: 8 evaluations per lane per 256-element group, register-only,
deterministic. Optional `signs1`/`signs2` buffers already exist.

Embedded 256-float tables are **not mandatory**: they add per-lane indexed loads and
binary footprint. Benchmark PRNG vs table vs precomputed buffer, then switch only if the
table wins after load pressure is accounted for.

## 8. Split-K policy

Deferred and measurement-gated. Upstream ships `x32/y64` without split-K on 72/80-CU
parts; the `N=64` underfill is 40%, not catastrophic.

Gate:

```text
M=4096, K=4096, N=64:
  baseline <32,64> vs split-K=2 vs split-K=4
adopt only if end-to-end pp64 improves enough to cover the reduction overhead
```

No floating-point atomics as the primary path (determinism); allocate partials and
reduce.

## 9. MQ type integration scope

Q8_0-first is the mergeable path. MQ4/MQ3/MQ2 need:

1. `ggml` type enum.
2. Type table entry (block 256; sizes 136/104/72).
3. Type traits.
4. CPU dequantize.
5. CPU quantize (if conversion is supported).
6. GGUF type id.
7. Model-loading support.
8. Converter from `.mq4` / `.mq3` / `.mq2` artifacts.
9. Backend dispatch.
10. `test-backend-ops` coverage.

Without the converter the types cannot load real models. Do not describe this as a small
patch; gate it behind an experimental CMake option.

## 10. MoE scope

Dense is `ggml_cuda_mul_mat_q`; MoE is `ggml_cuda_mul_mat_id`. The MQ2-Lloyd kernel has
the routing parameters and bounds guards, but it remains a probe. Merge dense first; MoE
is a separate `mul_mat_id` workstream with its own numerical and benchmark gates.

## 11. Issue #300 wording

The NaN bug occurs when the `x_dm` metadata write is gated on `tid < 128` instead of
`tid < MMQ_Y`. For `MMQ_Y=64`, threads 64..127 overwrite `tile_y` and corrupt the Q8_1
activation tile. Issue #300 tracks the related occupancy/tuning levers; the gate itself
is a kernel correctness invariant.

## 12. Verification status

| Item | Status |
|---|---|
| Build | clean with `-j4` |
| `-j24` crashes | unrelated upstream template instances; isolated compile succeeds |
| `MUL_MAT -p q8_0` | 113/113 (default = stock path) |
| `MUL_MAT -p q8_0`, `GGML_HIPFIRE_Q8_0_MIN_N=64` | 113/113 (hipfire path exercised) |
| `MUL_MAT_ID -p q8_0` | 75/75 (stock by design) |
| Q8_0 tail sweep (GPU idle, 3 rounds) | hipfire 24-257% slower at N=9..128; hook disabled |
| Spills | 0 on all measured kernels |
| Measured VGPRs | MQ4 64/79, MQ3 69/71, Q8_0 75, MQ2 72, MQ2 GEMV 24, dot2 23, convert 4 |
| Local CU baseline | 80 CUs / 320 SIMDs |
| Format | clang-format clean |
| `-DLLAMA_FATAL_WARNINGS=ON` (`--target ggml-hip`) | passes; no hipfire warnings (only unrelated `conv2d.cu` pass-failed notes) |

Q8_0 tail sweep (GPU idle, no KFD processes; 3 rounds, median, 200 async computes per
shape; shapes 1024x2048, 2048x2048, 4096x2048, 4096x4096, 8192x1024):

| N | hipfire vs stock | range |
|---:|---:|---|
| 9 | +194% slower | +119..+257% |
| 12 | +196% slower | +120..+256% |
| 16 | +194% slower | +118..+256% |
| 32 | +155% slower | +71..+197% |
| 64 | +91% slower | +29..+111% |
| 96 | +64% slower | +48..+76% |
| 128 | +36% slower | +24..+45% |

Both paths are llama.cpp MMQ (MMVQ covers only `N <= 8`). The hipfire kernel never wins,
including at `N=128` where its 128-column tile is full.

Q8_0 hipfire is numerically correct but not performance-competitive, so the mergeable
artifact is the kernel/dispatch infrastructure with the hook **off by default**.

## 13. Outstanding items

1. **Q8_0 tail sweep — done.** 5 shapes x N=9/12/16/32/64/96/128, stock vs hipfire:
   hipfire is 24-257% slower (median +194% at N=9, +91% at N=64, +36% at N=128). Do not
   lower or raise the gate; the hook stays disabled. A tuned Q8_0 kernel is a new task.
2. **`-DLLAMA_FATAL_WARNINGS=ON` build — done.** `build-hip-werror`, target `ggml-hip`:
   passes with no hipfire warnings.
3. **PMC LDS bank-conflict validation** before changing `X_STRIDE`/`Y_STRIDE`; do not
   change strides on theory alone.
4. **Real Q8_0 model KLD/coherence run** (no Q8_0 model on this host yet).
5. **MQ type converter** (blocker for MQ4/MQ3/MQ2 loading).
6. **Optional N=64 split-K measurement** on this 80-CU part.
7. **Optional MMVQ decode workstream** if `N=1..8` decode is a target; separate from
   the MMQ hook.
8. **Long-context HIP decode hypothesis** (quantized-V FA path): see
   `HYPOTHESIS_HIP_LONG_CONTEXT_DECODE.md`. One 32k f16-V discriminator before any
   HIP FA work; otherwise stay on Vulkan.

## 14. Keep / drop

Keep:

1. Q8_0 infrastructure (kernels + dispatch), with the hook opt-in only.
2. Measured resource gates.
3. Alignment asserts with the 2-byte Q8_0 contract.
4. PMC validation before LDS stride changes.
5. MoE isolation.
6. Split-K only after measurement.

Drop:

1. The 144-CU / 576-SIMD premise and all targets derived from it.
2. Mandatory `N==1` / `2..12` kernels in the MMQ path (MMVQ owns them).
3. Embedded FWHT sign tables as a mandatory change.
4. The 256-byte Q8_0 alignment requirement.
5. The redundant CMake source-list append.
6. `--wavefront-size=32`.
7. The assumption that `(128,4)` changes codegen occupancy.
8. The 89..107 VGPR contract.
9. Automatic split-K for `N=64`.

## 15. Final stance

**Merge now:** the kernel/dispatch infrastructure, local 80-CU baseline, measured
resource gates, 2-byte alignment assert, `COMPILE_OPTIONS` + `-mno-wavefrontsize64`,
dense-only scope with MoE deferred. The Q8_0 hook ships **disabled by default**; it is
opt-in via `GGML_HIPFIRE_Q8_0_MIN_N` for benchmarking only.

**Do not enable:** Q8_0 hipfire for production until a tuned kernel beats stock; MQ4/MQ3/MQ2
model loading; MQ converter; MMVQ decode hook; split-K; FWHT table change; LDS stride
changes.

**Measure next:** N=64 split-K, PMC bank conflicts, a real Q8_0 KLD run, and the MMVQ
`N=1..8` path if decode is a goal. The Q8_0 tail sweep is complete and negative.
