# RDNA2 MMQ Implementation Plan — Porting hipfire's Validated Optimizations

**Date:** 2026-07-02
**Target GPU:** gfx1030/gfx1131 (RX 6800 XT / RX 6900 XT / RX 6700 XT / RX 6600) — RDNA2 wave32
**Source PRs (merged):**
- PR #298 (May 20, 2026): MQ3 prefill — HFQ3 batched-prefill + full MMQ family
- PR #315 (May 25, 2026): HFQ4 MMQ family + HFQ3 polish + split routing
- PR #434 (Jun 10, 2026): Dispatch unification — consistent kernel selection across archs
- PR #477 (Jun 28, 2026): Arch-generic speculative decode seam
**gfx1030 confirmed:** v_dot2_f32_f16 available, GEMV single-row default
**Gate:** HIPFIRE_HFQ4_MMQ_RDNA2=1 (opt-in, default OFF pending KLD validation matrix)
**Research doc:** `docs/HIPFIRE_RDNA2_RESEARCH.md`

---

## Section 1: Kernel Topology Comparison

### 1.1 Launch Configuration

| Aspect | hipfire (gemm_hfq4g256_residual_mmq.gfx1030.hip) | llama.cpp existing MMQ (mmq.cuh) |
|--------|--------------------------------------------------|----------------------------------|
| Warp size | 32 (wave32 native) | 32 (gfx1030) |
| Threads/block | (32, 4, 1) = 128 | Variable: 64-256 depending on arch |
| Warps/block | 4 | 4 (RDNA2 via `get_mmq_nwarps_device()` → 256/32 = 8, but RDNA2 limits) [ASSUMPTION: needs verification] |
| `__launch_bounds__` | `(128, 2)` — 2 blocks/CU | Varies by type/arch |
| Grid | `(M/MMQ_Y, N/MMQ_X)` | `(ne01, ne11)` with padding |
| Block dims | Fixed 128 | Dynamic based on `calc_launch_params` |

**Key difference:** hipfire hardcodes 128 threads/warp×4 = 4 warps. llama.cpp's dispatch uses `get_mmq_nwarps_device()` which returns `256/warp_size = 8` on RDNA2 — but RDNA2 cannot sustain 8 waves due to VGPR pressure. **Mitigation:** Use `__launch_bounds__(128, 2)` explicitly (matching hipfire).

### 1.2 LDS Layout and Sizing

| Region | hipfire | llama.cpp adaptation |
|--------|---------|---------------------|
| `x_qs` (INT8 quantized) | 128 × 40 ints = 20,480 B | Same dimensions; Q4_K needs same stride since 128 rows × 16 uints/row = 2048 uints |
| `x_dm` (X metadata/scales) | 128 float2 = 1,024 B | Will expand to ~6,144 B to hold per-sub-block scales (see §3.2) |
| `tile_y` (Y tile) | 32 cols × 36 ints = 4,608 B | Identical — `block_q8_1_mmq` layout matches exactly |
| **Total** | **~26 KB** | **~31 KB** (still under 64 KB CU limit → 2 WGs/CU) |

**Critical compatibility:** Both use identical `block_q8_1_mmq` Y tile format (lines 41-46 of hipfire, lines 28-47 of llama.cpp mmq.cuh). No Y-side changes needed.

### 1.3 Inner Loop Structure

| Aspect | hipfire | llama.cpp sdot4 port |
|--------|---------|---------------------|
| Sub-blocks per window | 4 | 4 (same) |
| K-dimension per sub-block | 8 elements | 8 elements (sdot4 computes 4 pairs × 2 int4 = 8) |
| Instruction | `__builtin_amdgcn_sdot4(x_v, y_v, sumi, false)` × 8 | Same instruction × 8 |
| Accumulator | `int sumi` (32-bit integer) | Same |
| Scale application | `scale_w * d_x * sumi + zp_eff * sum_x` | Must adapt for per-sub-block scales |

**Key adaptation:** llama.cpp's Q4_K has 8 sub-blocks per super-block (not 4). Each sub-block has its own scale. The hipfire kernel assumes 4 sub-blocks with a single shared scale. **Solution:** Load per-sub-block scales during X-tile load phase, apply individually in inner loop.

### 1.4 X-Tile Loading (Format-Specific Part)

This is the ONLY section requiring format-specific logic.

**hipfire HFQ4 (lines 100-128):**
```
group_stride = 136 B  (sc[4] + zp[4] + qs[128])
body = 128 B (4-bit nibbles in 4-byte uints)
window_stride = window * 64 + chunk * 4
nibble_extract = (qs0 >> (i*4)) & 0xF
signed_bias = n - 8   (center at 8)
```

**llama.cpp Q4_K (existing dp4a path, lines 2093-2199 of mmq.cuh):**
```
super_block = 256 elements, 8 sub-blocks × 32
per super-block: d[2B] + dm[2B] + scales[12B] + qs[128B] = 144B
per sub-block scale: 1 of 8 values from packed 12-byte array
scale_unpack = unpack_scales_q45_K(scales, ksc) → 4 int8 values
effective_scale = d * scale_factor OR dmin * scale_factor (based on sign bit)
```

**Translation table:**

| hipfire concept | llama.cpp equivalent | Notes |
|----------------|---------------------|-------|
| sc (float) | d (half) + scale unpack | Need to reconstruct float from half + unpacked scale |
| zp (float) | Not present as separate field | Q4_K uses dm (negative direction), not zp |
| nibble centering | Subtract 8 from each nibble | Same centering, same math |
| 2 windows × 64B each | 8 sub-blocks × 32 elements each | Different grouping, same 256-element coverage |
| Window stride formula | Sub-block index within super-block | Need new addressing: `superblock_offset + sub_block * 32` |

### 1.5 Y Write-back Pattern

| Aspect | hipfire | llama.cpp port |
|--------|---------|---------------|
| Mode | Residual add (`Y[out_idx] += sum[idx]`) | Residual add (matching) |
| Index calculation | `(long long)col * M + row` | Same (column-major output) |
| Bounds check | `if (col >= N) continue` | Same |
| Thread mapping | `(j0/MMQ_NWARPS) * (MMQ_Y/WAVE_SIZE) + (i0/WAVE_SIZE)` | Identical formula |

Write-back is a direct copy. No changes needed.

### 1.6 Data Types

| Register type | hipfire | llama.cpp sdot4 port |
|--------------|---------|---------------------|
| VGPR (accumulators) | 32 floats per thread | 32 floats per thread — identical |
| VGPR (X data) | int4 × 4 + float2 = ~24 regs | Same |
| VGPR (Y data) | half2 × 1 + int4 × 2 = ~6 regs | Same |
| SGPR (addresses) | Row/col offsets, pointers | Same |
| LDS bank padding | X_STRIDE=40 (16-B aligned) | Same — prevents bank conflicts |
| Y_STRIDE | 36 (half2[4] + int8[128]) | Identical `block_q8_1_mmq` layout |

---

## Section 2: Q4K Weight Format Analysis

### 2.1 Block Layout Comparison

**hipfire HFQ4 group (136 bytes):**
```
Offset  Size  Content
------  ----  -------
0       4     sc (float scale)
4       4     zp (float zero-point = zp + 8*sc)
8      128    qs (4-bit nibbles, 256 elements packed)
```

**llama.cpp Q4_K super-block (144 bytes):**
```
Offset  Size  Content
------  ----  -------
0       2     d (half — super-block scale)
2       2     dm (half — negative direction scale)
4      12     scales (8 × 6-bit quantized scales + 2 signs)
16    128     qs (4-bit quants, 256 elements packed as nibbles)
```

### 2.2 Dequantization Path: Raw Bytes → INT8 Values That sdot4 Needs

**Step 1: Read raw bytes** (line 2120 of mmq.cuh `load_tiles_q4_K`):
```cpp
const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
const int qs0 = get_int_b4(bxi->qs, txi);  // reads 4 bytes as int
```

**Step 2: Extract nibbles with centering** (lines 2123-2128):
```cpp
// For MMA path: subtract 8 via vsubss4
x_qs[...] = (qs0 >> 0) & 0x0F0F0F0F;   // raw nibbles 0-3
x_qs[...] = (qs0 >> 4) & 0x0F0F0F0F;   // raw nibbles 4-7
// For DP4A path (what sdot4 needs): subtract 8 to get signed INT8
x_qs[...] = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
x_qs[...] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
```

**Step 3: Unpack per-sub-block scales** (lines 2083-2091 `unpack_scales_q45_K`):
```cpp
int unpack_scales_q45_K(const int * scales, const int ksc) {
    // Returns 4 signed 6-bit scale values from packed format
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}
```

**Step 4: Apply scale to quants** (lines 2154-2166):
```cpp
const int sc32 = unpack_scales_q45_K(scales, ksc + 0);  // positive scale
const int  m32 = unpack_scales_q45_K(scales, ksc + 2);  // negative scale (sign bit)
const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);     // d and -dmin

// Final scale = dm.x * sc OR dm.y * m (depending on sign bit)
x_dm[...] = dm * make_half2(sc8[l], m8[l]);
```

**Step 5: Effective computation:**
```
For element e in sub-block s:
  raw_nibble = (qs >> (e*4)) & 0xF
  signed_val = raw_nibble - 8           // center at 0
  scale_idx  = s                        // 0..7 within super-block
  scale_sign = (scales >> (scale_idx/4)) & 1   // bit tells us d vs dmin
  scale_mag  = unpacked_scale[scale_idx]         // 6-bit magnitude
  effective_scale = scale_sign ? d : dmin) * scale_mag / 64
  result = signed_val * effective_scale
```

### 2.3 What Changes for the sdot4 Port

The dequantization math is identical — only the **storage location** and **access pattern** differ:

| Existing DP4A path | New sdot4 path |
|--------------------|----------------|
| Loads qs + dm into LDS via `load_tiles_q4_K` | Same load pattern |
| Uses `half2 x_dm[]` for per-row dm | Keep `half2 x_dm[]` for d/dmin |
| Computes `x_dm * y_ds` in vec_dot | Compute `scale_w * d_x * sumi + ...` per sub-block |
| Single scale per row (D4 layout) or half2 dm (DS4 layout) | Need per-sub-block scale: expand to float array or recompute from dm/scales each iteration |

**Recommendation:** Store pre-computed per-sub-block effective scales in an additional LDS region (`x_scales[MMQ_Y][8]`). This amortizes the scale unpacking cost across all 8 sub-blocks computed per window.

---

## Section 3: Phased Implementation Plan

### Phase 1: Core Kernel Body (NEW FILE)

**File to CREATE:** `ggml/src/ggml-cuda/mmq-gfx1030.cuh`

**What to implement:** A self-contained MMQ body template for gfx1030 using sdot4, parameterized by MMQ_X and MMQ_Y (matching hipfire's include-based approach).

**Referencing:** hipfire `gemm_hfq4g256_residual_mmq_body.cuh` (lines 1-199) as topology template.

**Code structure:**
```cpp
// mmq-gfx1030.cuh — RDNA2 wave32 sdot4 MMQ body for Q4_K

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#define MMQ_NWARPS 4
#define WAVE_SIZE 32
#define QK8_1 32
#define X_STRIDE 40
#define Y_STRIDE 36

struct block_q8_1_mmq { /* identical to mmq.cuh lines 28-47 */ };

__launch_bounds__(128, 2)
extern "C" __global__ void KERNEL_NAME(
    const char* __restrict__ A,        // Q4_K super-block pointer
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    // Block grid: blockIdx.x → row tile, blockIdx.y → col tile
    // Thread layout: threadIdx = (threadIdx.x, threadIdx.y) where y ∈ [0,3]
    
    extern __shared__ int smem[];
    int*    x_qs   = smem;                              // [MMQ_Y][X_STRIDE]
    float2* x_dm   = (float2*)(x_qs + MMQ_Y * X_STRIDE); // [MMQ_Y] d/dmin
    float*  x_sc   = (float*)(x_dm + MMQ_Y);             // [MMQ_Y][8] per-sub-block scales
    int*    tile_y = (int*)(x_sc + MMQ_Y * 8);           // [MMQ_X][Y_STRIDE]
    
    // X-tile loader: adapt load_tiles_q4_K for Q4_K super-block format
    // For each row i in [0, MMQ_Y):
    //   1. Read block_q4_K at (row0+i)
    //   2. Unpack nibbles → INT8 into x_qs[i*X_STRIDE..]
    //   3. Compute per-sub-block effective scales → x_sc[i*8..i*8+8]
    //   4. Store dm as float2 → x_dm[i]
    
    // Inner loop: 2 windows × 4 sub-blocks × sdot4 × 8 calls
    // Same structure as hipfire lines 132-175
    
    // Write-back: residual add, identical to hipfire lines 181-198
}
```

**Success criteria:**
- Compiles cleanly under ROCm 7.13 gfx1030 target
- `static_assert(sizeof(block_q8_1_mmq) == 144)` passes
- LDS total ≤ 34 KB (leaving margin for 2 WGs/CU)
- Occupancy ≥ 50% (measured via `rocm-smi --show-utilization`)

**Risk assessment:**
| Risk | Probability | Mitigation |
|------|-------------|-----------|
| LDS overflow | Medium | Start with MMQ_Y=64 variant (halves x_qs + x_dm + tile_y); bump to 128 if VGPR allows |
| VGPR pressure → spill | High | Use `__launch_bounds__(128, 2)` to force 2 blocks/CU; if spills detected, reduce MMQ_Y to 96 or 64 |
| Scale computation overhead | Low | Pre-compute in X-loader (already reading block); amortized over 8 sub-blocks |

**Estimated effort:** 16 hours (coding + debugging)

**Dependencies:** None — self-contained header.

### Phase 2: Q4_K X-Tile Loader Adaptation

**File to MODIFY:** `mmq-gfx1030.cuh` (within Phase 1 file)

**What to implement:** The format-specific X-tile loader that replaces hipfire's HFQ4 loader with Q4_K super-block logic.

**Referencing:** llama.cpp `load_tiles_q4_K` (lines 2093-2199 of mmq.cuh) for the scale unpacking logic.

**Key differences from hipfire loader:**

| Concern | hipfire (lines 100-128) | Q4_K adaptation needed |
|---------|------------------------|----------------------|
| Group stride | 136 B fixed | No fixed stride — super-blocks indexed by `kbx` |
| Window structure | 2 windows × 64B body | 8 sub-blocks × 32 elements (but grouped as 2×4 for sdot4) |
| Scale format | Single float sc + zp | Per-sub-block: d or dmin × unpacked 6-bit scale |
| Zero-point | Explicit zp field | Implicit: nibble - 8 (same centering, no separate zp) |

**Loader pseudocode:**
```cpp
// For each row i in [0, MMQ_Y):
//   const block_q4_K* bxi = (block_q4_K*)(A + row_offset);
//   
//   // Unpack quants: same nibble extraction as hipfire but with Q4_K qs layout
//   for (int loop = 0; loop < X_LOADER_TASKS_PER_THREAD; ++loop) {
//       int task_id = tid * tasks_per_thread + loop;
//       int i_row = task_id / 16;
//       int chunk = task_id % 16;
//       
//       // Read 4 bytes from super-block qs region
//       unsigned int qs0 = *(unsigned int*)(bxi->qs + chunk * 4);
//       
//       // Extract 8 nibbles, subtract 8, pack into 2 int32
//       // (identical to hipfire lines 113-124)
//       x_qs[i_row * X_STRIDE + 2*chunk + 0] = int_a;
//       x_qs[i_row * X_STRIDE + 2*chunk + 1] = int_b;
//   }
//   
//   // Compute per-sub-block scales
//   const int* raw_scales = (const int*)bxi->scales;
//   for (int s = 0; s < 8; ++s) {
//       int sc = unpack_scales_q45_K(raw_scales, s);    // positive scale
//       int mn = unpack_scales_q45_K(raw_scales, s+4);  // negative scale
//       float d_f = __half2float(bxi->d);
//       float dm_f = -__half2float(bxi->dm);            // dmin is stored as negative half
//       
//       // Sign bit determines which direction to use
//       int sign_bit = (sc >> 31) & 1;
//       float effective_d = sign_bit ? dm_f : d_f;
//       x_sc[i * 8 + s] = effective_d * (sc & 0x3F) / 64.0f;
//   }
//   
//   // Store dm as float2
//   x_dm[i] = make_float2(__half2float(bxi->d), -__half2float(bxi->dm));
```

**Success criteria:**
- Numerical parity with existing dp4a path: KLD@n=256 delta < 0.001
- All nibbles correctly unpacked (spot-check first 1000 weights against reference)
- Per-sub-block scales match `ggml_vec_dot_q4_K` output within 1 ULP

**Risk assessment:**
| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Scale unpacking math error | Medium | Cross-validate against existing `unpack_scales_q45_K` in mmq.cuh line 2083 |
| Sign bit misinterpretation | Medium | Q4_K uses dm as negative direction; verify with `make_half2(1.0f, -1.0f)` pattern from line 2160 |
| LDS alignment issues | Low | Use same stride constants (X_STRIDE=40, Y_STRIDE=36) as proven hipfire layout |

**Estimated effort:** 8 hours

**Dependencies:** Phase 1 header skeleton must compile first.

### Phase 3: Dispatch Integration

**File to MODIFY:** `ggml/src/ggml-cuda/mmq.cu`

**What to implement:** Add gfx1030 dispatch case for Q4_K in the kernel launch path.

**Referencing:** Existing `ggml_cuda_mul_mat_q_switch_type` (lines 6-75 of mq.cu).

**Changes needed:**

1. **Add include** at top:
```cpp
#include "mmq-gfx1030.cuh"
```

2. **Add dispatch case** in the non-stream-K path. The existing code quantizes src1 to Q8_1 and then calls the MMQ kernel. We need to detect gfx1030 + Q4_K and route to our new kernel:

```cpp
// In ggml_cuda_mul_mat_q or the relevant dispatch function:
if (GGML_CUDA_CC_IS_RDNA2(cc) && src0->type == GGML_TYPE_Q4_K && !use_stream_k) {
    // Launch mmq-gfx1030 kernel instead of standard mmq
    const int mmq_x = 32;  // Single tile size for RDNA2
    const int mmq_y = 128; // Or 64 if occupancy requires it
    
    // Compute grid dimensions
    const dim3 block(mmq_x / MMQ_NWARPS, mmq_y / WAVE_SIZE, 1);
    const dim3 grid(ceil_div(ne01, mmq_y), ceil_div(ne11, mmq_x));
    
    // Shared memory: same as hipfire calculation
    const size_t shared_mem = mmq_y * X_STRIDE * sizeof(int) +
                               mmq_y * sizeof(float2) +
                               mmq_y * 8 * sizeof(float) +  // per-sub-block scales
                               (mmq_x / MMQ_NWARPS) * (mmq_y / WAVE_SIZE) * sizeof(float); // accumulators
    
    // Kernel name: gemm_q4k_gfx1030_mmq_res
    // Arguments: A (src0), Xq (quantized src1), Y (dst), M=ne01, K=ne00, N=ne11
}
```

3. **Register in switch_type** — add `GGML_TYPE_Q4_K` case that checks for gfx1030 and routes accordingly.

**Success criteria:**
- Kernel launches without errors on gfx1030 hardware
- Correct output for synthetic matmul (A@X = Y, verify against CPU reference)
- Performance > existing dp4a path at batch sizes ≥ 16

**Risk assessment:**
| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Dispatch path mismatch | Medium | Follow existing `ggml_cuda_mul_mat_q` pattern exactly; only change the kernel invocation |
| Shared memory size miscalculation | Low | Use hipfire's exact formula as baseline, add 6KB for scale storage |
| Grid dimension rounding errors | Low | Use same ceiling division as existing code |

**Estimated effort:** 6 hours

**Dependencies:** Phases 1-2 must compile and produce correct results.

### Phase 4: CMakeLists Integration

**File to MODIFY:** `ggml/src/ggml-cuda/CMakeLists.txt`

**What to implement:** Add `mmq-gfx1030.cuh` to the build, with gfx1030-specific compile flags.

**Referencing:** Existing mmq.cuh entry in CMakeLists.txt.

**Changes needed:**
```cmake
# Add new source file
set(GGML_CUDA_SOURCES
    ${GGML_CUDA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/mmq-gfx1030.cuh   # RDNA2 sdot4 MMQ body
)

# If separate .hip compilation is needed:
if(GGML_BACKEND_METAL OR GGML_CUDA)
    # The .cuh is included by other .cu files, no separate compilation needed
endif()
```

**Note:** Since `mmq-gfx1030.cuh` is a header included by other `.cu` files, it follows the same build pattern as `mmq.cuh`. No separate compilation step needed.

**Success criteria:** Clean build with no new warnings under ROCm 7.13.

**Estimated effort:** 1 hour

**Dependencies:** None.

### Phase 5: MMQ_Y Tuning Sweep

**File to MODIFY:** `mmq-gfx1030.cuh` (parameter changes only)

**What to implement:** Per-kernel MMQ_Y variants based on hipfire's findings.

**Referencing:** hipfire research doc § "MMQ_Y per-kernel tuning" and commit `01469af`.

| Kernel type | Optimal MMQ_Y | Rationale |
|-------------|--------------|-----------|
| Residual (A@X += Y) | **64** | LDS-bound; smaller per-WG work → higher occupancy wins |
| Overwrite (Y = A@X) | **128** | Need per-WG compute to hide latency |

**Implementation:** Two instantiations of the body template:
```cpp
// Variant 1: y=64 for residual paths
#define MMQ_Y 64
#define MMQ_X 32
#define KERNEL_NAME gemm_q4k_gfx1030_mmq_res_y64
#include "mmq-gfx1030.cuh"

// Variant 2: y=128 for overwrite paths
#define MMQ_Y 128
#define MMQ_X 32
#define KERNEL_NAME gemm_q4k_gfx1030_mmq_ow_y128
#include "mmq-gfx1030.cuh"
```

**Success criteria:**
- y=64 residual: +5-10% over y=128 at batch sizes 13-127
- y=128 overwrite: +5-15% over y=64 at batch sizes ≥ 128
- No NaN or correctness regression

**Risk assessment:**
| Risk | Probability | Mitigation |
|------|-------------|-----------|
| y=64 regresses on llama.cpp workloads | Medium | Benchmark both; keep the winner; fallback to single y=128 if needed |
| Occupancy too low at y=128 | Low | hipfire confirmed 2 WGs/CU works for residual; overwrite may need profiling |

**Estimated effort:** 4 hours (mostly benchmarking)

**Dependencies:** Phase 3 must be working.

### Phase 6: Batch-Size Auto-Selector

**File to MODIFY:** `mmq.cu` dispatch logic

**What to implement:** Route different batch sizes to optimal mmq_x variant.

**Referencing:** hipfire commit `ce95ca1` (auto-selector).

| Batch size (ne11) | mmq_x | Rationale |
|-------------------|-------|-----------|
| ≤ 12 | Use existing scalar/dp4a path | MMQ tile overhead wastes compute |
| 13-127 | mmq_x = 16 | Fits in LDS with y=128, good occupancy |
| ≥ 128 | mmq_x = 32 | Full tile utilization |

**Implementation:**
```cpp
// In dispatch function:
int optimal_mmq_x;
if (ne11 <= 12) {
    optimal_mmq_x = 0;  // use existing path
} else if (ne11 <= 127) {
    optimal_mmq_x = 16;
} else {
    optimal_mmq_x = 32;
}
```

**Success criteria:**
- No regression vs baseline at any batch size
- Peak throughput at each batch size within 5% of manually-tuned optimum
- Correctness at all batch sizes (edge cases: ne11 = 1, ne11 = prime numbers)

**Risk assessment:**
| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Selector wrong for edge cases | Low | Test ne11 = {1, 2, 3, 7, 13, 15, 128, 255, 256} |
| Overhead of selector decision | Negligible | Single if-else chain; compiler optimizes to branchless |

**Estimated effort:** 4 hours

**Dependencies:** Phase 5 completed.

---

## Section 4: What NOT to Implement (With Evidence)

### 4.1 dp4a Instruction Path

**Evidence:** hipfire commit `2cf265a` — "dp4a wave32 port" on gfx1031.
**Result:** NEGATIVE — -15% median vs dot2 inner loop.
**Extended validation:** Commit `e4548ae` tested pf=30/240/1188 — always negative (-12% to -17%).
**Conclusion:** Do NOT implement any dp4a-based kernel variant. Use sdot4 exclusively.

### 4.2 MMQ_Y = 32 for Residual Kernels

**Evidence:** hipfire commit `a9a5658` — "residual MMQ_Y=32".
**Result:** NEGATIVE — -20% to -40% regression due to L2 cache pressure.
**Conclusion:** Never use MMQ_Y ≤ 32. Minimum is 64 (and only for residual).

### 4.3 MMQ_Y = 64 for gate_up / qkv Kernels

**Evidence:** hipfire commits `4ee8a90` ("gate_up MMQ_Y=64") → -4%. Commit `eb50436` ("gate_up y64 sweep N=64..1024") → y64 ALWAYS slower at N≥128 (3-21% worse).
**Conclusion:** Only use y=64 for residual kernels. gate_up/qkv must stay at y=128.

### 4.4 hipGraph for Prefill Capture

**Evidence:** Research doc § "F1: hipGraph for prefill" — blocked by hipMalloc-during-capture restriction. llama.cpp's diagnostic shows 0% measured gain because dispatch pattern differs from hipfire's.
**Conclusion:** Defer entirely. Not part of this implementation scope.

### 4.5 Persistent Kernels

**Evidence:** Research doc § "F3: Persistent kernels" — estimated +10-30%, not started by hipfire. High complexity, not validated.
**Conclusion:** Out of scope. Phase 1-6 deliver sufficient gains.

### 4.6 Epilogue Fusion (SwiGLU/RMSnorm)

**Evidence:** Research doc § "F2: MMQ epilogue fusion" — MEDIUM priority, +5-15% expected, not started.
**Conclusion:** Future optimization. Focus on raw GEMM throughput first.

### 4.7 F16-packed for gfx1010

**Evidence:** hipfire commit `059e829` — "fp16-packed for gfx1010". Functional but no bench data (no gfx1010 hardware available).
**Conclusion:** Irrelevant for gfx1030 target. Skip.

---

## Section 5: Validation Plan

### 5.1 Unit Tests (Phase 1-2)

**Test 1: Nibble unpack correctness**
- Generate random Q4_K super-block
- Run both dp4a and sdot4 loaders
- Compare INT8 output byte-for-byte
- Pass criteria: 100% match on 1000 random blocks

**Test 2: Scale computation**
- For each sub-block (0-7), verify effective scale matches reference: `(d_or_dmin * scale_factor)`
- Compare against ggml dequantization reference implementation
- Pass criteria: all values within 1 ULP

**Test 3: Full matmul correctness**
- Random A (Q4_K, M×K) @ X (Q8_1, K×N) → Y
- Compare sdot4 MMQ output vs existing dp4a MMQ output
- Compute max absolute error and relative error
- Pass criteria: max abs error < 0.01, max rel error < 0.1%

### 5.2 KLD Validation (Phase 3+)

**Methodology** (from hipfire research doc § "KLD validation methodology"):
- Use n=256 minimum (n=30 is unreliable)
- KV type: Q8 (standard)
- Model: gemma-4-12B Q4_K_XL (or equivalent dense Q4_K model)
- Metric: KLD between standard inference and MMQ-instrumented inference

**Procedure:**
1. Run baseline inference (existing dp4a path) → record KLD@n=256
2. Build with new sdot4 kernel enabled
3. Run same inference → record KLD@n=256
4. Compute delta

**Pass criteria:** KLD delta < 0.005 (within measurement noise)

**Hipfire precedent:** Commit research showed MMQ is innocent — dot2 and MMQ produce equivalent KLD at n=256 (delta < 0.04%). Our sdot4 should match or beat this.

### 5.3 Benchmark Targets

**Dense model (gemma-4-12B Q4_K_XL):**

| Metric | Current (dp4a) | Target (sdot4 MMQ) | Delta |
|--------|---------------|-------------------|-------|
| pp128 | ~1300 t/s | ~1550 t/s | +20% |
| pp256 | ~1330 t/s | ~1580 t/s | +19% |
| pp512 | ~1333 t/s | ~1580 t/s | +18.5% |
| tg128 | ~54.6 t/s | ~55 t/s | +1% (no change expected) |

**MoE model (gemma-4-26B-A4B Q4_K_XL):**

| Metric | Current (dp4a) | Target (sdot4 MMQ) | Delta |
|--------|---------------|-------------------|-------|
| pp128 | ~2500 t/s | ~3000 t/s | +20% |
| tg128 | ~101.8 t/s | ~103 t/s | +1% |

**Rationale:** hipfire delivered +21% on MQ3 residual at pf=240 (commit adc1558). HFQ4 delivered +22% at pp128 (PR #315 Phase 1). Q4_K should land in the same range since LDS tiling is the dominant factor, not the unpack cost difference between 3-bit and 4-bit weights (hipfire Issue #299).

### 5.4 Regression Checks

**Every phase must pass:**
1. **Build check:** Clean build under ROCm 7.13 with gfx1030 target
2. **Sanity check:** Random matmul produces finite output (no NaN/Inf)
3. **Correctness:** KLD@n=256 within tolerance of baseline
4. **Performance:** ≥ baseline performance at all tested batch sizes
5. **Memory:** No OOM at max context length (16K tokens)

---

## Section 6: Risk Matrix

### 6.1 Compile Risk

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| ROCm 7.13 doesn't support `__builtin_amdgcn_sdot4` on gfx1030 | High | Low | Verify intrinsics available in `amdgcn-builtins.h` for gfx1030; fallback to inline asm if needed |
| `__launch_bounds__(128, 2)` conflicts with existing launch params | Medium | Low | Test compilation; if conflict, use `__attribute__((num_simd_workgroups(2)))` instead |
| Header inclusion order issues | Low | Low | Follow same include pattern as mmq.cuh |

### 6.2 Numerical Risk

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| Per-sub-block scale precision loss | Medium | Medium | Use float32 for scale storage in LDS; verify KLD@n=256 |
| Sign bit misinterpretation in Q4_K scales | High | Medium | Cross-validate against existing `unpack_scales_q45_K` function; unit test with known-good weights |
| Accumulator overflow at large batch sizes | Low | Low | Float32 accumulators handle ±3.4×10^38; practical limits are ~10^6 elements |

### 6.3 Performance Risk

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| VGPR spill at MMQ_Y=128 | High | Medium | Start with y=64; profile VGPR count via `rocprof --stats`; if > 256, reduce to y=96 or y=64 |
| Decode path unchanged (no benefit) | Low | N/A | Expected — this is a prefill optimization only |
| L2 cache pressure at small batch sizes | Medium | Medium | Auto-selector routes small batches to existing path; verify with benchmarks |
| LDS bank conflicts from scale storage | Low | Low | Use same stride constants (40/36) as proven hipfire layout |

### 6.4 Maintenance Risk

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|-----------|
| Divergence from existing mmq.cuh patterns | Medium | Medium | Follow llama.cpp conventions exactly; use same struct types, same dispatch pattern |
| Extra code to maintain (new .cuh file) | Low | N/A | Single file, ~200 lines; similar complexity to existing kernel variants |
| Hardcoded gfx1030 target limits portability | Low | N/A | Guard with `#if defined(RDNA2) || defined(GFX1030)` preprocessor checks |

---

## Appendix A: File Change Summary

| File | Action | Lines changed (est.) |
|------|--------|---------------------|
| `ggml/src/ggml-cuda/mmq-gfx1030.cuh` | **CREATE** | ~220 (new file) |
| `ggml/src/ggml-cuda/mmq.cu` | MODIFY | ~30 (dispatch integration) |
| `ggml/src/ggml-cuda/CMakeLists.txt` | MODIFY | ~3 (source registration) |
| `ggml/src/ggml-cuda/mmq.cuh` | **NO CHANGE** | 0 |

## Appendix B: hipfire Commit Reference Map

| Our Implementation | hipfire Source | PR/Commit |
|-------------------|---------------|-----------|
| Core kernel topology (wave32, LDS-tiled, sdot4) | `gemm_hfq4g256_residual_mmq.gfx1030.hip` | PR #298, commit `adc1558` |
| Y=64 residual variant | `gemm_hfq3g256_residual_mmq_x32_y64.gfx1030.hip` | PR #315, commit `01469af` |
| Batch-size selector | Template family + auto-selector | PR #298, commit `ce95ca1` |
| QKV fused variant (future) | `gemm_qkv_hfq4g256_mmq_body.cuh` | PR #315 Phase 2 |
| Gate_up fused variant (future) | `gemm_gate_up_hfq4g256_mmq_body.cuh` | PR #315 Phase 3 |
| Dispatch pattern reference | dispatch.rs `has_dot2_f32_f16()` | PR #434 (dispatch unification) |
| Speculative decode seam | spec_decode.rs | PR #477 (arch-generic) |

## Appendix C: Known Limitations

1. **Single tile size (mmq_x=32):** hipfire tested x8/x16/x32 but our initial implementation uses only x=32 for simplicity. Can be extended in a follow-up PR.

2. **Residual-only write-back:** Initial kernel uses residual add (`Y += A@X`). Overwrite mode (`Y = A@X`) needed for QKV/gate_up fused kernels — implement in Phase 5+ as separate instantiation.

3. **No HFQ3/HFQ4 specific types:** This implementation targets llama.cpp's standard Q4_K type. The "HFQ" prefix in hipfire refers to their custom weight format; we're adapting the topology for llama.cpp's native format.

4. **gfx1030/gfx1131 is the primary target.** hipfire's dispatch.rs confirms gfx1030/gfx1031/gfx1032/gfx1131 all have v_dot2_f32_f16 and use single-row GEMV default. Architecture is identical across these GPUs.
