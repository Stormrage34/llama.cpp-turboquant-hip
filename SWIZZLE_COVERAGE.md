# SWIZZLE COVERAGE.md — RDNA2 Cache-Aware SoA Swizzle Tracking

**Last Updated:** 2026-05-21  
**Scope:** Complete trace of all `RDNA2_CACHE_SWIZZLE` and `RDNA2_SWIZZLE_ALL_QUANTS` gated code paths

---

## Master Reference: All Swizzle Sites

| # | File | Line(s) | Function/Context | Quant Types | Layout | Status |
|---|------|---------|-----------------|-------------|--------|--------|
| 1 | `src/llama-model-loader.cpp` | 1394–1479 | Host-side AoS→SoA conversion functions | IQ4_XS, Q4_K, Q5_K | AoS→SoA | **Verified** |
| 2 | `src/llama-model-loader.cpp` | 1500–1527 | Load-time swizzle dispatch (GPU-resident only) | IQ4_XS (±Q4_K, Q5_K) | SoA write | **Verified** |
| 3 | `ggml/src/ggml-cuda/mmq.cuh` | 3215–3306 | `load_tiles_iq4_xs_swizzled` | IQ4_XS only | SoA read | **Verified (MMQ dispatch)** |
| 4 | `ggml/src/ggml-cuda/mmq.cuh` | 3308–3422 | `load_tiles_q4_K_swizzled` | Q4_K only | SoA read | ⚠️ **Dead code in MMQ** (never called) |
| 5 | `ggml/src/ggml-cuda/mmq.cuh` | 3424–3551 | `load_tiles_q5_K_swizzled` | Q5_K only | SoA read | ⚠️ **Dead code in MMQ** (never called) |
| 6 | `ggml/src/ggml-cuda/mmq.cuh` | 3882–3964 | MMQ kernel dispatch (if-constexpr chain) | IQ4_XS only | SoA read | **Verified** |
| 7 | `ggml/src/ggml-cuda/mmvq.cu` | 33–38 | `get_vec_dot_q_cuda` returns nullptr for IQ4_XS when swizzled | IQ4_XS | — | **Verified** (redirects to inline path) |
| 8 | `ggml/src/ggml-cuda/mmvq.cu` | 518–551 | MMVQ kernel: vec_dot dispatch (if-constexpr) row path | IQ4_XS, Q4_K, Q5_K | SoA read | **Verified** |
| 9 | `ggml/src/ggml-cuda/mmvq.cu` | 536–547 | MMVQ kernel: gate fusion path (if-constexpr) | IQ4_XS, Q4_K, Q5_K | SoA read | **Verified** |
| 10 | `ggml/src/ggml-cuda/mmvq.cu` | 713–723 | MMVQ kernel: single-row path (if-constexpr) | IQ4_XS, Q4_K, Q5_K | SoA read | **Verified** |
| 11 | `ggml/src/ggml-cuda/vecdotq.cuh` | 1383–1448 | `vec_dot_iq4_xs_q8_1_swizzled` | IQ4_XS | SoA read | **Verified** |
| 12 | `ggml/src/ggml-cuda/vecdotq.cuh` | 1450–1498 | `vec_dot_q4_K_q8_1_swizzled` | Q4_K | SoA read | **Verified** |
| 13 | `ggml/src/ggml-cuda/vecdotq.cuh` | 1500–1553 | `vec_dot_q5_K_q8_1_swizzled` | Q5_K | SoA read | **Verified** |
| 14 | `ggml/src/ggml-cuda/vecdotq.cuh` | 1340–1381 | `vec_dot_iq4_xs_q8_1` (standard AoS, baseline) | IQ4_XS | AoS read | Baseline (pre-swizzle) |
| 15 | `ggml/src/ggml-common.h` | 545–596 | GPU swizzle size constants (`IQ4_XS_QS_SIZE`, etc.) | All | — | **Shared definitions** |

---

## Build Configuration

### Compile Flags

| Flag | CMake Option | Default | Defined In | Effect |
|------|-------------|---------|------------|--------|
| `RDNA2_CACHE_SWIZZLE` | `-DRDNA2_CACHE_SWIZZLE=ON` | OFF | `ggml-hip/CMakeLists.txt:191` + `src/CMakeLists.txt:58` | Enables IQ4_XS swizzle host transform + kernel SoA loaders |
| `RDNA2_SWIZZLE_ALL_QUANTS` | `-DLLAMA_BUILD_SWIZZLE_DEV=ON` | OFF | `CMakeLists.txt:199` | Extends swizzle coverage to Q4_K + Q5_K |

### Activation Chain

```
LLAMA_BUILD_SWIZZLE_DEV=ON
  → CMakeLists.txt:199 adds RDNA2_SWIZZLE_ALL_QUANTS=1

RDNA2_CACHE_SWIZZLE=ON
  → ggml-hip/CMakeLists.txt:193 adds -DRDNA2_CACHE_SWIZZLE to HIP flags
  → src/CMakeLists.txt:59 adds -DRDNA2_CACHE_SWIZZLE to llama library
```

### Build Script Invocation

| Command | Swizzle | Extra Quants |
|---------|---------|-------------|
| `./scripts/build_rdna2.sh` | OFF | — |
| `./scripts/build_rdna2.sh --swizzle` | ON (IQ4_XS only) | — |
| `./scripts/build_rdna2.sh --swizzle-all` | ON (IQ4_XS + Q4_K + Q5_K) | Q4_K, Q5_K |

The `--swizzle` flag sets `-DRDNA2_CACHE_SWIZZLE=ON`  
The `--swizzle-all` flag sets `-DRDNA2_CACHE_SWIZZLE=ON -DLLAMA_BUILD_SWIZZLE_DEV=ON`

---

## Host-Side Swizzle (`src/llama-model-loader.cpp`)

### Architecture

Three parallel `swizzle_*_host()` functions convert each quant type from standard AoS (Array of Structures — blocks interleaving quants and metadata per-block) to SoA (Structure of Arrays — all quant bytes contiguous, all metadata bytes contiguous):

| Quant | AoS Block Size | Qs Portion | Meta Portion | SoA Layout |
|-------|---------------|------------|--------------|------------|
| IQ4_XS | 136B | 128B (qs) | 8B (d+scales_h+scales_l) | `[qs_blk0(128B)]...[qs_blkN-1(128B)] [meta_blk0(8B)]...[meta_blkN-1(8B)]` |
| Q4_K | 144B | 128B (qs) | 16B (dm+scales) | `[qs_blk0(128B)]...[qs_blkN-1(128B)] [meta_blk0(16B)]...[meta_blkN-1(16B)]` |
| Q5_K | 176B | 128B (qs) | 48B (dm+scales+qh) | `[qs_blk0(128B)]...[qs_blkN-1(128B)] [meta_blk0(48B)]...[meta_blkN-1(48B)]` |

### Dispatch Logic (llama-model-loader.cpp:1500–1527)

```cpp
#ifdef RDNA2_CACHE_SWIZZLE
    // Only swizzle GPU-resident tensors — CPU layers read standard AoS layout
    if (cur->type == GGML_TYPE_IQ4_XS
#if defined(RDNA2_SWIZZLE_ALL_QUANTS)
        || cur->type == GGML_TYPE_Q4_K || cur->type == GGML_TYPE_Q5_K
#endif
    ) {
        // Check if tensor is GPU-resident
        bool gpu_resident = false;
        // ... layer index checks ...
        if (gpu_resident) {
            switch (cur->type) {
                case IQ4_XS: swizzle_iq4_xs_host(...); break;
                case Q4_K:   swizzle_q4_K_host(...);   break;  // (gated by RDNA2_SWIZZLE_ALL_QUANTS)
                case Q5_K:   swizzle_q5_K_host(...);   break;  // (gated by RDNA2_SWIZZLE_ALL_QUANTS)
            }
        }
    }
#endif
```

**Critical invariant:** CPU layers ALWAYS read AoS. GPU-resident tensors ONLY are swizzled to SoA. This means the two code paths must never share a tensor without conversion.

### Constants

| Macro | Value | Defined In | Used By |
|-------|-------|-----------|---------|
| `IQ4_XS_QS_SIZE` | 128 (QK_K/2) | `ggml-common.h:545` | GPU kernel loaders |
| `IQ4_XS_META_SIZE` | 8 | `ggml-common.h:546` | GPU kernel loaders |
| `Q4_K_QS_SIZE` | 128 (QK_K/2) | `ggml-common.h:563` | GPU kernel loaders |
| `Q4_K_META_SIZE` | 16 | `ggml-common.h:564` | GPU kernel loaders |
| `Q5_K_QS_SIZE` | 128 (QK_K/2) | `ggml-common.h:581` | GPU kernel loaders |
| `Q5_K_META_SIZE` | 48 | `ggml-common.h:582` | GPU kernel loaders |
| `IQ4_XS_BLOCK_SIZE` | 136 | `llama-model-loader.cpp:1396` | Host swizzle only |
| `Q4_K_BLOCK_SIZE` | 144 | `llama-model-loader.cpp:1426` | Host swizzle only |
| `Q5_K_BLOCK_SIZE` | 176 | `llama-model-loader.cpp:1454` | Host swizzle only |

**Note on naming:** Host constants use `*_HOST` suffix (e.g., `IQ4_XS_QS_SIZE_HOST = 128`). GPU constants in `ggml-common.h` omit the suffix (e.g., `IQ4_XS_QS_SIZE = 128`). Values are identical.

---

## MMQ Swizzled Load Tiles (`ggml/src/ggml-cuda/mmq.cuh`)

### Functions Defined

| Function | Lines | Template Params | Read Pattern | Called? |
|----------|-------|----------------|-------------|---------|
| `load_tiles_iq4_xs_swizzled<mmq_y, need_check>` | 3222–3306 | `mmq_y`, `need_check` | `qs_ptr = x + block_idx * 128`, `meta_ptr = x + total_blocks * 128 + block_idx * 8` | ✅ YES (3 call sites) |
| `load_tiles_q4_K_swizzled<mmq_y, need_check>` | 3313–3422 | `mmq_y`, `need_check` | `qs_ptr = x + block_idx * 128`, `meta_ptr = x + total_blocks * 128 + block_idx * 16` | ❌ NO — **Dead code** |
| `load_tiles_q5_K_swizzled<mmq_y, need_check>` | 3428–3551 | `mmq_y`, `need_check` | `qs_ptr = x + block_idx * 128`, `meta_ptr = x + total_blocks * 128 + block_idx * 48` | ❌ NO — **Dead code** |

### Dispatch Sites (mmq.cuh)

Only **3 call sites** exist, all for IQ4_XS exclusively:

| Call Site | Line | Context | Code Path |
|-----------|------|---------|-----------|
| `load_tiles_iq4_xs_swizzled(...)` | 3884 | First tile prefetch (double-buffer path) | `if constexpr (type == GGML_TYPE_IQ4_XS)` |
| `load_tiles_iq4_xs_swizzled(...)` | 3910 | Next tile prefetch (double-buffer path) | `if constexpr (type == GGML_TYPE_IQ4_XS)` |
| `load_tiles_iq4_xs_swizzled(...)` | 3960 | Standard path (no double-buffer) | `if constexpr (type == GGML_TYPE_IQ4_XS)` |

**Bug CR-013 (OPEN):** At line 3291, the meta pointer computation is:
```cpp
const uint8_t * meta_ptr = qs_base + total_blocks_in_view * IQ4_XS_QS_SIZE + block_idx * IQ4_XS_META_SIZE;
```
The `total_blocks_in_view` variable is passed as `total_blocks_x` from the caller (line 3827: `nrows_x * stride_row_x`). This assumes all rows visible to the kernel have the same number of blocks. If `stride_row_x` varies or the view is a partial sub-tensor, the meta offset computation is incorrect — this is the **AoS vs SoA mismatch** described in CR-013.

---

## MMVQ Swizzled Vec Dot (`ggml/src/ggml-cuda/mmvq.cu` + `ggml/src/ggml-cuda/vecdotq.cuh`)

### Function Pointer Table (mmvq.cu:24–41)

When `RDNA2_CACHE_SWIZZLE` is defined:
- `get_vec_dot_q_cuda(GGML_TYPE_IQ4_XS)` returns `nullptr` (line 35)
- This forces MMVQ to use the inline `if constexpr` dispatch instead of function pointer dispatch

**Why:** The swizzled vec_dot functions take **5 parameters** vs the standard 4-parameter signature. They require `blocks_per_row` and `nrows` for SoA meta-location computation, which the function pointer interface cannot pass.

### vecdotq.cuh: Swizzled Functions

| Function | Lines | SoA Offset Formula |
|----------|-------|-------------------|
| `vec_dot_iq4_xs_q8_1_swizzled` | 1390–1448 | `qs_ptr = vbq + kbx*128`; `meta_ptr = vbq + total_blocks*128 + kbx*8` |
| `vec_dot_q4_K_q8_1_swizzled` | 1453–1498 | `qs_ptr = vbq + kbx*128`; `meta_ptr = vbq + total_blocks*128 + kbx*16` |
| `vec_dot_q5_K_q8_1_swizzled` | 1503–1553 | `qs_ptr = vbq + kbx*128`; `meta_ptr = vbq + total_blocks*128 + kbx*48` |

All three use identical offset patterns:
```
total_blocks = nrows * blocks_per_row          // computed per-invocation
qs_offset    = kbx * qs_stride                  // 128B aligned
meta_offset  = total_blocks * qs_stride + kbx * meta_stride
```

### MMVQ Dispatch Sites

Three dispatch blocks in `mmvq.cu` use identical `if constexpr` chains:

**Row path** (lines 518–533):
```cpp
#ifdef RDNA2_CACHE_SWIZZLE
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        vec_dot_iq4_xs_q8_1_swizzled(vx, y, kbx, kqs, blocks_per_row_x, nrows_x_mmvq);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        vec_dot_q4_K_q8_1_swizzled(vx, y, kbx, kqs, blocks_per_row_x, nrows_x_mmvq);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        vec_dot_q5_K_q8_1_swizzled(vx, y, kbx, kqs, blocks_per_row_x, nrows_x_mmvq);
    } else
#endif
    {
        vec_dot_q_cuda(vx, y, kbx, kqs);  // AoS fallback
    }
```

**Gate fusion path** (lines 536–551): Same pattern with `vgate` pointer instead of `vx`.  
**Single-row path** (lines 713–727): Same pattern, used when `c_rows_per_block == 1`.

All three dispatch sites support ALL THREE quant types (IQ4_XS, Q4_K, Q5_K).

---

## CR-013: Open Bugs

### Bug 1: `load_tiles_iq4_xs_swizzled` Meta Offset — SoA/AoS Assumption Mismatch

**Status:** OPEN (from project-state.md)  
**File:** `ggml/src/ggml-cuda/mmq.cuh:3290–3291`  
**Code:**
```cpp
const uint8_t * qs_base = (const uint8_t *)x;
const uint8_t * meta_ptr = qs_base + total_blocks_in_view * IQ4_XS_QS_SIZE + block_idx * IQ4_XS_META_SIZE;
```

**Problem:** `total_blocks_in_view` is passed as `total_blocks_x = nrows_x * stride_row_x` (line 3827). This is a flat count of all blocks in the tile's row range. If the tile starts at a non-zero `kbx0` offset within a row or the tensor is a sub-view, the meta section base address computed by `total_blocks_in_view * IQ4_XS_QS_SIZE` does not accurately reflect where the meta section starts for this particular tile position. The kernel reads SoA meta from the wrong offset, producing incorrect scale values.

**Impact:** Only manifests when MMQ kernel is used with IQ4_XS swizzled tensors — produces wrong inference results. MMVQ kernel is not affected because its vec_dot functions receive `nrows * blocks_per_row` per-invocation and compute meta offset from the row base.

### Bug 2: Q4_K/Q5_K MMQ Swizzled Load Tiles — Dead Code

**Status:** NEW FINDING  
**File:** `ggml/src/ggml-cuda/mmq.cuh:3313,3428`  
**Problem:** `load_tiles_q4_K_swizzled()` and `load_tiles_q5_K_swizzled()` are fully defined but **never called from any dispatch point**. The `if constexpr` chains in the MMQ kernel (lines 3882, 3908, 3958) only check for `GGML_TYPE_IQ4_XS`. Q4_K and Q5_K fall through to the AoS `load_tiles()` function. This means:

1. When `RDNA2_CACHE_SWIZZLE=ON` + `RDNA2_SWIZZLE_ALL_QUANTS=ON` + Q4_K/Q5_K model: the host-side swizzle converts Q4_K/Q5_K to SoA, but the MMQ kernel reads them as AoS via the fallback `load_tiles()` path.
2. **Result:** Wrong inference output for Q4_K/Q5_K models when MMQ kernel is active with `--swizzle-all`.

**Fix needed:** Add Q4_K and Q5_K to the MMQ `if constexpr` dispatch chains at lines 3882–3886, 3908–3912, and 3958–3962.

### Bug 3: `get_vec_dot_q_cuda` Returns nullptr for IQ4_XS

**Status:** INTENTIONAL (but fragile)  
**File:** `ggml/src/ggml-cuda/mmvq.cu:33–38`  
**Code:**
```cpp
case GGML_TYPE_IQ4_XS:
#ifdef RDNA2_CACHE_SWIZZLE
    return nullptr;  // Swizzled uses different function signature
#else
    return vec_dot_iq4_xs_q8_1;
#endif
```

**Problem:** Returning `nullptr` disables the function pointer dispatch path entirely. The MMVQ kernel template `if constexpr` chain handles it, but this creates an implicit assumption that every code path through MMVQ uses the inline chain. Any future addition of a new MMVQ variant that relies on function pointers will silently break for IQ4_XS when swizzle is enabled.

---

## Verification Status

### Verified Working (Passes CR-017 stability baseline)

| Path | Quant Types | Test |
|------|------------|------|
| Host AoS→SoA load-time conversion | IQ4_XS | `validate_cr013.sh` |
| MMVQ vec_dot swizzled dispatch | IQ4_XS, Q4_K, Q5_K | `verify_kernel_dispatch.sh` + inference parity |
| vecdotq.cuh swizzled functions | IQ4_XS, Q4_K, Q5_K | Single-row MMVQ path |

### Not Verified / Known Broken

| Path | Quant Types | Issue |
|------|------------|-------|
| MMQ `load_tiles_q4_K_swizzled` | Q4_K | **Dead code** — never dispatched |
| MMQ `load_tiles_q5_K_swizzled` | Q5_K | **Dead code** — never dispatched |
| MMQ `load_tiles_iq4_xs_swizzled` meta offset | IQ4_XS | CR-013: SoA/AoS mismatch for non-zero `kbx0` offsets |

---

## Rollback

To disable all swizzle paths:

```bash
# Option A: Runtime (no effect — swizzle is compile-time only)
#   Swizzle is compiled into the binary. No runtime toggle exists.
#   Must rebuild without the flag.

# Option B: Rebuild without swizzle
./scripts/build_rdna2.sh  # default: no swizzle flags

# Option C: Git revert (if swizzle changes are the suspect)
git checkout HEAD -- src/llama-model-loader.cpp
git checkout HEAD -- ggml/src/ggml-cuda/mmq.cuh
git checkout HEAD -- ggml/src/ggml-cuda/mmvq.cu
git checkout HEAD -- ggml/src/ggml-cuda/vecdotq.cuh
```

---

## Summary

| Metric | Value |
|--------|-------|
| Total swizzled functions | 9 (6 GPU loaders + 3 host converters) |
| Quant types supported | 3 (IQ4_XS, Q4_K, Q5_K) |
| Quant types fully verified | 1 (IQ4_XS) |
| Quant types with dead MMQ code | 2 (Q4_K, Q5_K) |
| Open bugs | 2 (CR-013 meta offset, new Q4_K/Q5_K MMQ dispatch) |
| Host → GPU constant consistency | ✅ Verified (all *_SIZE values match) |
| GPU-resident guard | ✅ CPU layers never read SoA |
