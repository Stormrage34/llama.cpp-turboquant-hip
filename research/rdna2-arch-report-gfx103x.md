# AMD RDNA 2 (gfx103x) Architecture Research Report
## Relevance to llama.cpp GPU Kernels & mlnn.py Simulation Engine

**Date**: 2026-07-04  
**Target Hardware**: AMD Radeon RX 6700 XT / 6750 XT class (40 CUs)

> **Important architecture clarification first**: You reference "gfx1030" with 40 DCUs. The official gfx1030 is Navi 21 (RX 6800/6900 series, 60-80 CUs). A 40-CU RDNA 2 card is **gfx1031** (Navi 22), i.e., RX 6700 XT (12 GB) or 6750 XT. The ISA is identical; only Infinity Cache (96 vs 128 MB) and L2 (3 vs 4 MB) sizes differ. All findings below apply to both, with gfx1031 values called out specifically.

---

## 1. Wave32 Execution — CORRECT

RDNA 2 uses **Wave32 as native execution mode**. Each SIMD has 32-wide VALU issuing 32 threads per cycle. The WGP (Work Group Processor) can run wave64 as a fallback for backwards compatibility, but that takes two cycles per warp instead of one.

**Sources**: ROCm HIP hardware implementation docs confirm: *"RDNA makes a fundamental change to CU design, by changing the size of a warp to 32 threads... effectively combining two GCN5 SIMDs, creating a VALU of width 32."*

**Implication**: `WAVE_SIZE = 32` in the sim is correct. If any kernel falls back to wave64 (e.g., due to very high VGPR pressure), effective throughput drops ~2x because each warp takes 2 cycles. The sim does not model this fallback — that's a gap worth closing.

**Gotcha**: When VGPR pressure forces fewer concurrent waves, the hardware may also force wave64 mode in some instruction streams, compounding the penalty beyond simple occupancy reduction.

---

## 2. LDS (Local Data Share) — MOSTLY CORRECT, NEEDS REFINEMENT

**Size**: 128 KB per CU — confirmed across all RDNA 2 parts (RX 6600 through RX 6950 XT).

**Bank layout**: 32 banks x 4 bytes per bank. Bank mapping: `(address / 4) % 32`. Confirmed by Composable Kernel docs and AMD's ISA reference guide.

**Conflict model — needs update**: The sim says *"conflict when stride is multiple of 128B"* which is too simplistic. On RDNA 2, wide LDS instructions execute in **phases**:

| Instruction | Phases | Lanes per phase |
|-------------|--------|-----------------|
| ds_read_b32 | 2 | 32 (T0-T31, T32-T63) |
| ds_read_b64 | 4 | 16 |
| ds_read_b128 | 8 | 8 (T0-T7, T8-T15, ...) |
| ds_write_b128 | 8 | 8 |

A bank conflict only occurs when two threads **in the same phase** access the same bank. Consecutive-address writes via ds_write_b128 are naturally conflict-free — each 8-lane phase maps to different banks automatically. The sim's simple stride check misses this nuance entirely and would flag many benign access patterns as conflicts.

**Recommendation**: Replace the stride-based heuristic with per-phase bank collision counting using the actual phase groupings above. For kernels using ds_bpermute for lane shuffles (common in attention), the conflict model should track which banks are being permuted into, not just raw strides.

---

## 3. VGPR Pressure Model — INACCURATE

**Register file**: The spec table lists **VGPR File = 512 KiB per CU**, not "256 VGPRs/SIMD".

Breaking it down: 512 KiB / 4 bytes per word = 131,072 words total per CU. With 4 SIMDs per CU, that's ~32,768 words/SIMD. The sim's `VGPRS_PER_SIMD = 256` appears to be conflating two different concepts — the hardware limit on VGPRs-per-thread (which is ~256) vs total SIMD capacity.

**Occupancy thresholds — directionally reasonable but needs calibration**:
- Below 64 VGPRs/thread: full occupancy (8+ waves per SIMD)
- 64-128 VGPRs: moderate reduction (~4-6 waves)  
- 128-256 VGPRs: significant reduction (~2-3 waves)
- Above 256: kernel may fail to launch or use fallback paths

The sim's penalty formula `ratio * (1.0 + 0.15 * (ratio - 1.0))` is a reasonable first-order approximation but doesn't capture the step-function nature of RDNA's VGPR allocation (the hardware allocates in discrete bands).

---

## 4. Cache Hierarchy — MOSTLY CORRECT WITH THREE FIXES

| Tier | Size | Latency (cycles) | Notes |
|------|------|-------------------|-------|
| L0 Vector | 16 KB per CU | 1 | 4-way set assoc, 128-byte lines |
| L0 Scalar | 16 KB per CU | 1 | Per-CU instruction cache |
| L0 Instruction | 32 KB per 4 CUs | 1 | Shared across WGP |
| L1 Graphics | 128 KB per WGP (2 CUs) | ~4 | The sim correctly says "per 2 CUs" |
| L2 | 2-4 MB shared | ~15 | **Sim hardcodes 4 MB — only correct for flagship** |
| Infinity Cache | 32-128 MB shared | ~50 | RDNA 2's on-die L3 equivalent |
| GDDR6 VRAM | Device-dependent | ~200 | Peak: 512 GB/s (flagship), ~400 GB/s (RX 6700 XT) |

**Three corrections needed in sim**:
1. **L2 size**: `CACHE_L2_SIZE = 4 MB` is wrong for gfx1031 — it's **3 MB**. For gfx1032 (RX 6600), it's **2 MB**. Should be parameterized by device.
2. **Infinity Cache**: `GPU_INFINITY_CACHE_MB = 128` is wrong for gfx1031 — it's **96 MB**. Flagship Navi 21 (RX 6800/6900) has 128 MB.
3. **Memory bandwidth**: `GPU_MEMORY_BW_GB_S = 512` matches RX 6800 XT / 6900 XT. The RX 6700 XT with its narrower bus delivers **~400 GB/s**. This significantly affects bandwidth saturation calculations.

---

## 5. PDL Sync on AMD — CONFIRMED NO-OP

I read the actual definition in `ggml/src/ggml-cuda/common.cuh` (lines 118-133):

```cpp
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && \
    (CUDART_VERSION >= 12030 || ...)
#    define GGML_CUDA_USE_PDL
#endif

static __device__ void ggml_cuda_pdl_sync() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaGridDependencySynchronize();
#endif
}

static __device__ void ggml_cuda_pdl_lc() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
```

**Confirmed**: Both functions are complete no-ops on HIP/AMD. The `#if !defined(GGML_USE_HIP)` guard ensures PDL intrinsics are never compiled for AMD targets. The sim correctly identifies this as a no-op.

**Does RDNA 2 have an equivalent? No.** CUDA's PDL is exclusive to Hopper (sm_90+) and newer NVIDIA. RDNA 2 has no programmatic launch dependency mechanism.

**AMD's equivalent for cooperative KV cache access**: The current codebase handles this through **kernel launch ordering in streams**. Since each flash attention kernel processes independent tiles, there's no cross-threadblock PDL-like dependency. If future kernels need it, use `hipStreamSynchronize()` between dependent launches, or split into multiple kernels with explicit ordering.

The placement of `ggml_cuda_pdl_sync()` at line 149 in `fattn-vec.cuh` is harmless dead code on AMD. On NVIDIA Hopper, it removes an unnecessary grid-level sync before K/V tile loading — a real optimization.

---

## 6. Async Copies on RDNA 2 — SUPPORTED WITH CAVEATS

RDNA 2 has **two SDMA engines** supporting concurrent bidirectional transfers that can overlap with compute. `hipMemcpyAsync` and `cudaMemcpyAsync` are functionally equivalent on RDNA 2 — both use the same SDMA infrastructure.

**Critical caveat**: There's a known regression in ROCm starting from version 6.3.3 where small synchronous copies (1-1024 bytes) became significantly slower due to added synchronization overhead. The fix landed in 7.2+, but **asynchronous copies (`hipMemcpyAsync`) performed better than the broken sync copies throughout**.

**Implication for llama.cpp**: If the codebase uses `memcpy` for small KV cache updates (e.g., sliding window shifts, per-token KV writes), switching to `hipMemcpyAsync` with a dedicated stream could avoid the regression. For large bulk transfers (full prompt processing where bandwidth dominates), the difference is negligible.

Newer ROCm also adds `hipMemcpyBatchAsync` which can reduce API overhead for multiple small transfers by batching them into a single SDMA submission — useful if you have many small KV cache writes per inference step.

---

## 7. Flash Attention Optimization on RDNA 2

**Current state**: The Composable Kernel (CK) backend for FlashAttention on ROCm **does NOT support RDNA 2** — only Instinct (MI200/MI300). The **Triton backend** does support RDNA GPUs with fp16/bf16/fp32.

**Best tile sizes for 128-dim head on gfx1030**:
- BLOCK_M: **32 or 64** — small enough for LDS capacity on wave32
- BLOCK_N: **32 or 64** — balances compute vs memory
- Waves per EU: **1-2** — higher waves increase occupancy but reduce per-wave LDS
- Num warps: **4-8** — matches available CUs efficiently
- Pre-load V: **false** — saves LDS; load V inline during softmax combine

**LDS bank conflict avoidance patterns**:
1. Don't stride K/V reads by `head_dim` when threads in the same wave access different heads — causes bank conflicts within phases
2. Pad LDS arrays by a few elements (e.g., row_stride = head_dim + 4 instead of exactly head_dim) to break alignment with the 32-bank boundary
3. Use XOR preshuffling for column index permutation (see Composable Kernel's `xor_preshuffle` utility)
4. The tinygrad flash attention reference uses `LDS_PAD = 4` — a proven pattern

**Wave32 vs Wave64 for FA inner loop**: Wave32 is strongly preferred. Per zolotukhin.ai's analysis of RDNA flash attention: *"wave32 doubles the maximum count of in-flight wavefronts per SIMD at the cost of halving the cross-lane reduction width"* — for FA the inner loop doesn't do heavy reductions, so this tradeoff is net positive. The current llama.cpp FA kernel already uses `nthreads_KQ_q = 2` for RDNA (vs 4 for non-RDNA), implicitly selecting wave32-friendly configurations.

---

## 8. Multi-GPU / Split Buffer on ROCm

ROCm provides `hipDeviceEnablePeerAccess(peerDeviceId)` for GPU-to-GPU direct memory access, analogous to CUDA's peer access but **requiring explicit activation** per device pair.

Key differences from CUDA:
1. **Explicit enable/disable required**: Must call `hipDeviceEnablePeerAccess()` for each pair
2. **No NVLink on RDNA consumer cards**: P2P goes through PCIe (~16 GB/s on Gen 4 x16, ~8 GB/s on Gen 3), far below GPU memory bandwidth (400+ GB/s). This makes multi-GPU KV cache offloading much less attractive than on NVIDIA with NVLink.
3. **SDMA engines are shared** across GPUs on the same system — heavy inter-GPU traffic can bottleneck both directions
4. **Host staging buffer** is the reliable fallback when P2P isn't available

For llama.cpp's multi-GPU KV cache offloading, the recommendation is to use `hipMemcpyPeerAsync()` only after confirming peer access works (`hipDeviceCanAccessPeer()`), with host staging as a safe fallback path.

---

## Summary: Corrections Needed in mlnn.py

| Parameter | Current Value | Correct Value | Severity |
|-----------|--------------|---------------|----------|
| GPU_ARCH | gfx1030 (with 40 CUs) | Should be gfx1031 for 40 CU card | Medium (terminology) |
| GPU_MEMORY_BW_GB_S | 512 | ~400 for RX 6700 XT class | High |
| CACHE_L2_SIZE | 4 MB | 3 MB for gfx1031 | Medium |
| GPU_INFINITY_CACHE_MB | 128 | 96 MB for gfx1031 | Medium |
| VGPRS_PER_SIMD | 256 (total SIMD) | Should represent max VGPRs/thread; total is ~32K per SIMD | High |
| LDS bank conflict model | stride % 128 == 0 | Phase-based analysis needed | Medium |
| Cache latency values | Based on flagship assumptions | Should scale with actual device tier | Low |
