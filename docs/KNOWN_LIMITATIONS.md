# Known Limitations — RDNA2 Optimization (gfx1030)

> **Last updated**: 2026-05-14 | **Hardware**: RX 6800 XT (gfx1030) | **ROCm**: 7.13 nightly

## Hardware Scope

| GPU | Architecture | Status | Notes |
|-----|-------------|--------|-------|
| RX 6800 / 6800 XT / 6900 XT | RDNA 2 (gfx1030) | ✅ Fully optimized | Primary target |
| RX 7800 XT / 7900 XT / 7900 XTX | RDNA 3 (gfx1100/1101) | ⚠️ Untested | May work, no guarantees |
| NVIDIA GPUs | Various | ❌ Not applicable | Use upstream llama.cpp |
| Intel Arc | Various | ❌ Not applicable | Use upstream llama.cpp |

**If you're not on gfx1030, these optimizations provide zero benefit.** The runtime gate (`GGML_CUDA_CC_IS_RDNA2`) automatically disables them on non-RDNA2 hardware.

## Model Compatibility

|-----------|-------|----------------|----------------------|----------------------|---------------|
| 35B MoE | IQ4_XS | ✅ Active | ✅ Active (MoE) | ❌ No effect | +110% prefill |
| 27B Dense | IQ4_XS | ✅ Active | ⚠️ Auto-disabled | ❌ No effect | Baseline |
| 26B Dense | Q4_K_M | ✅ Active | ⚠️ Auto-disabled | ⚠️ Experimental | TBD (needs validation) |
| 26B Dense | Q5_K_M | ✅ Active | ⚠️ Auto-disabled | ⚠️ Experimental | TBD (needs validation) |
| Any model | TurboQuant | ✅ Active | Varies | ❌ No effect | Baseline |

**Key limitation**: `RDNA2_BFE_DISPATCHER` only affects Q4_K_M and Q5_K_M dequant kernels. If your model uses IQ4_XS or TurboQuant, BFE has zero effect regardless of the flag setting.

## Known Issues

### 1. BFE Dispatcher — Off-Hot-Path (Cold Path Only)
- **Status**: Code exists, but targets a COLD PATH (standalone dequant), NOT the inference hot path
- **Evidence**: Kernel trace on Llama-3.1-8B-Q4_K_M shows `mul_mat_vec_q` (fused dequant+matmul), NOT `dequantize_row_q4_K_cuda`
- **Impact**: BFE optimization never executes during normal inference — dequant is fused into matmul kernels
- **Default**: OFF (`-DGGML_RDNA2_BFE_DISPATCHER=OFF`) — and should stay OFF
- **Recommendation**: Sunset. Keep code behind `#ifdef` for reference only

### 2. DPP Scale Broadcast — Reverted
- **Status**: Removed from codebase
- **Reason**: Path mismatch — DPP targeted IQ4_XS dequant, but benchmark model used TurboQuant
- **Theoretical ceiling**: <1% even on correct path (8/32 threads active, L1-cached loads)
- **See**: `docs/RESEARCH_LOG.md` for full root cause analysis

### 3. Decode Throughput — Compute-Bound
- **Status**: Expected behavior, not a bug
- **Explanation**: Decode (tg128) is compute-bound on RDNA2. The +110% gain is prefill-only (memory-bound). Decode stays flat (~66 t/s for MoE, ~27 t/s for dense).
- **Do not report as regression**: Compare identical configs (`-ngl`, model, quant) before claiming decode regression.

### 4. `-ngl` Configuration Critical
- **MoE models**: Use `-ngl 99` (all layers GPU offloaded)
- **27B dense models**: Use `-ngl 55` (partial offload, fits in 16 GB VRAM)
- **Using `-ngl 30` on a 27B model**: Will produce ~27 t/s decode, which is expected, not a regression

### 5. Variance — Normal Behavior
- **v0.3.0 (experimental)**: ±635 t/s (bimodal, NaN/garbage tokens)
- **v0.3.1 (stable)**: ±6 t/s (0.17% coefficient of variation)
- **If you see high variance**: Ensure `RDNA2_MATMUL_OPT_V1=1` is set for MoE models

### 6. ROCm Version Sensitivity
- **Tested on**: ROCm 7.13 nightly
- **Minimum**: ROCm 6.1 (per CMakeLists.txt)
- **Known issue**: `VALUBusy` and `VALUUtilization` counters are unavailable on gfx1030 (ROCm limitation, not a bug)

## Testing on Other Hardware

We welcome testing on RDNA3 (gfx1100/1101) hardware! If you have an RX 7900 XT or similar:

1. Build with `-DGPU_TARGETS=gfx1100` instead of `gfx1030`
2. Run `scripts/verify_kernel_dispatch.sh <model.gguf> Q4_K_M`
3. Run `scripts/run_std_bench.sh <model.gguf> dense-99`
4. Open an issue with your telemetry results

**Please include**: GPU model, ROCm version, kernel dispatch trace, and benchmark output.

## Upstream Alignment

The following optimizations are fork-specific and NOT suitable for upstream llama.cpp PRs in their current form:

| Feature | Fork-Specific | Upstream Path |
|---------|---------------|---------------|
| `RDNA2_MATMUL_OPT_V1` LDS double-buffer | Yes — MoE-specific | Separate PR, different scope |
| `iq4_dequant_rdn2.cuh` IQ4_XS kernel | Yes — not validated for upstream | Path-specific, needs more testing |
| `RDNA2_BFE_DISPATCHER` CMake option | Yes — needs runtime detection | PR draft in `docs/PR_upstream_bfe_fence.md` |

The BFE dispatcher and `__threadfence_system()` are the most upstream-ready features. See `docs/PR_upstream_bfe_fence.md` for the draft PR.