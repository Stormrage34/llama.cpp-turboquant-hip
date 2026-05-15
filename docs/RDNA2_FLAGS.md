# RDNA2 Optimization Flags — Compatibility Matrix

> **Rule**: Never enable a flag without kernel-path verification. See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the DPP failure case study.

## Flag Reference

| Flag | Purpose | Default | Compile-Time | Runtime | Compatible Models | Status |
|------|---------|---------|-------------|---------|------------------|--------|
| `RDNA2_MATMUL_OPT_V1` | LDS double-buffer matmul for MoE | ON | `add_compile_definitions(RDNA2_MATMUL_OPT_V1)` | `RDNA2_MATMUL_OPT_V1=1` | MoE models only | ✅ Stable |
| `RDNA2_BFE_DISPATCHER` | `v_bfe_u32` for K-quant nibble unpack | OFF | `-DGGML_RDNA2_BFE_DISPATCHER=ON` | `RDNA2_BFE_DISPATCHER=1` | Q4_K_M, Q5_K_M | ⚠️ Experimental |
| `RDNA2_EXP_DPP_SCALES` | DPP broadcast for scale loads | ~~OFF~~ | ~~Removed~~ | ~~Removed~~ | ~~IQ4_XS~~ | ❌ Reverted |

## How to Enable

### Compile-Time (CMake)

```bash
# Stable features only (production)
cmake -B build -S . \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release

# + Experimental BFE dispatcher (validate first!)
cmake -B build -S . \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RDNA2_BFE_DISPATCHER=ON
```

### Runtime (Environment Variables)

```bash
# Production: all stable flags
export RDNA2_MATMUL_OPT_V1=1

# + Experimental BFE (only after validation!)
export RDNA2_BFE_DISPATCHER=1
```

## Model Compatibility Matrix

|-------|-------|----------------|----------------------|----------------------|-------|
| Qwen3.6-35B-MoE | IQ4_XS | ✅ Active | ✅ Active (MoE) | ❌ No effect | BFE targets Q4_K_M/Q5_K_M, not IQ4_XS |
| Qwen3.6-27B | IQ4_XS | ✅ Active | ⚠️ Auto-disabled (dense) | ❌ No effect | Same as above |
| Gemma-4-26B | Q4_K_M | ✅ Active | ⚠️ Auto-disabled (dense) | ✅ **Target** | BFE exercises Q4_K_M dequant path |
| Gemma-4-26B | Q5_K_M | ✅ Active | ⚠️ Auto-disabled (dense) | ✅ **Target** | BFE exercises Q5_K_M dequant path |
| Any model | TurboQuant | ✅ Active | Varies | ❌ No effect | TurboQuant has its own dequant path |

> **Key insight**: `RDNA2_BFE_DISPATCHER` only affects Q4_K_M and Q5_K_M dequant kernels. If your model uses IQ4_XS or TurboQuant, BFE has zero effect regardless of the flag.

## Validation Required Before Enabling BFE

Before enabling `RDNA2_BFE_DISPATCHER`, you **must** verify:

1. **Kernel dispatch**: `./scripts/verify_kernel_dispatch.sh <model.gguf> Q4_K_M`
   - Confirms `dequantize_row_q4_K_cuda` is actually invoked
   - The DPP failure was caused by skipping this step

2. **A/B telemetry**: `./scripts/run_ab_telemetry.sh <model.gguf> RDNA2_BFE_DISPATCHER 5`
   - Same-build comparison with/without the flag
   - Reports median ± std dev across 5 runs

3. **Gate criteria**:

| Metric | Target | Method |
|--------|--------|--------|
| Kernel invoked | Yes | `rocprofv3 --kernel-trace` |
| `SQ_INSTS_VALU` ↓ | ≥10% (kernel-filtered) | Median across 3 runs, CV < 2% |
| Decode (`tg128`) | ≥26.5 t/s | `llama-bench -r 5` median |
| Variance | ≤±1.5 t/s | Std dev across 5 runs |
| Parity | Zero mismatches @ `temp=0.0` | Token diff vs CPU baseline |

## Safety Gates

All RDNA2 optimizations are protected by a **triple gate**:

1. **Compile-time**: Flag must be set in CMake
2. **Runtime environment**: Env var must be set
3. **Hardware check**: Only activates on RDNA2 (gfx1030) via `GGML_CUDA_CC_IS_RDNA2(cc)`

If any gate fails, the kernel falls back to the stable baseline path with **zero overhead**.

## Reverted Flags

### `RDNA2_EXP_DPP_SCALES` — REVERTED

**Status**: ❌ Removed from codebase.

**Why**: DPP (`v_mov_dpp` / `__builtin_amdgcn_readlane`) broadcast targeted `dequantize_block_iq4_xs_rdn2`, but the benchmark model used TurboQuant, which dispatches through different kernels. DPP code compiled but **never executed**. Counter deltas were build-state noise, not optimization signal. Theoretical ceiling was <1% (8/32 threads active, L1-cached loads).

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full root cause analysis.