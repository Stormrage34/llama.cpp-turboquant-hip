# RDNA2 MoE Prefill Accelerator

> **Status**: Stabilized (v0.3.1) — production-ready. Variance reduced from ±635 to ±6 t/s.

## Overview

The `RDNA2_MATMUL_OPT_V1` flag enables an LDS (Local Data Share) double-buffered matrix multiplication kernel that overlaps weight tile loading with DP4A compute. It targets memory-bound small-matrix operations common in Mixture-of-Experts (MoE) routing on AMD RDNA2 (gfx1030) GPUs.

## Performance Profile

**Hardware**: AMD Radeon RX 6800 XT (gfx1030, 16 GB VRAM)
**Model**: Qwen3.6-35B-MoE-IQ4_XS @ pp512, batch=256, ubatch=128

### Before Stabilization (v0.3.0-experimental)

| Metric | Baseline | Experimental (v0.3.0) | Delta |
|--------|----------|----------------------|-------|
| Prefill (t/s) | ~480 ± 100 | ~1314 ± 635 | **+170%** (unstable) |
| Variance | Low | High (bimodal: 666–1777) | 3–6× baseline |

### After Stabilization (v0.3.1)

| Metric | Baseline | Stabilized (v0.3.1) | Delta |
|--------|----------|---------------------|-------|
| Prefill (t/s) | ~480 ± 100 | **1772 ± 6** | **+269%** (stable) |
| Variance | Low | ±6 t/s | Within noise |
| Decode (t/s) | ~57 ± 4 | ~52 ± 7 | No regression |
| Dense Models | ~480 t/s | ~480 t/s | Auto-disabled |

### 10-Run Stress Test (v0.3.1)

| Run | Prefill (t/s) | Run | Prefill (t/s) |
|-----|---------------|-----|---------------|
| 1 | 1776 | 6 | 1776 |
| 2 | 1774 | 7 | 1757 |
| 3 | 1775 | 8 | 1770 |
| 4 | 1773 | 9 | 1769 |
| 5 | 1776 | 10 | 1773 |

**Mean: 1772 t/s | Std dev: ±6 t/s | Variance reduction: 99%**

## Usage

```bash
# Stable RDNA2 features only (production)
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ./llama-server -m model.gguf -ngl 99

# + MoE prefill accelerator (now stable)
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 ./llama-server -m model.gguf -ngl 99
```

## Safety Gates

The accelerator is protected by a **triple gate** — all three must pass:

1. **Compile-time**: `-DRDNA2_MATMUL_OPT_V1=1` must be passed to the compiler
2. **Runtime environment**: `RDNA2_MATMUL_OPT_V1=1` must be set
3. **Hardware check**: Only activates on RDNA2 (gfx1030) via `GGML_CUDA_CC_IS_RDNA2(cc)`

If any gate fails, the kernel falls back to the stable baseline path with **zero overhead**.

## Stabilization Details

### Root Cause of Original Variance

Two independent issues caused the bimodal distribution (high ~1750, low ~900):

1. **LDS Bank Conflicts**: The gfx1030 LDS has 32 banks. Symmetric tile dimensions (32×N) caused warp-stride bank conflicts during the `tile_x ↔ tile_x_next` buffer swap.

2. **Register Spilling**: The compiler over-allocated registers on some runs, forcing spill to local memory and dropping wave occupancy.

### Fixes Applied (v0.3.1)

| Fix | Description | Impact |
|-----|-------------|--------|
| **LDS Padding** | +1 element offset on `tile_x_next` buffer breaks 32-bank symmetry | Eliminates within-run jitter |
| **Wave32 Occupancy Guard** | `__attribute__((amdgpu_waves_per_eu(4, 8)))` forces 4–8 wavefronts per EU | Eliminates between-run register spilling |

### KV Cache Type Matrix (Stabilized)

| ctk \ ctv | turbo2 (Orig) | turbo2 (Stable) | turbo3 (Orig) | turbo3 (Stable) | turbo4 (Orig) | turbo4 (Stable) |
|-----------|---------------|-----------------|---------------|-----------------|---------------|-----------------|
| **turbo2** | 474 ± 123 | 1772 ± 6 | — | — | — | — |
| **turbo3** | 501 ± 56 | — | 507 ± 68 | — | — | — |
| **turbo4** | 483 ± 101 | — | 552 ± 15 | — | — | — |

## Disabling

Unset `RDNA2_MATMUL_OPT_V1` or compile without `-DRDNA2_MATMUL_OPT_V1=1`. Runtime fallback is instant and zero-cost.

```bash
# Disable at runtime
unset RDNA2_MATMUL_OPT_V1

# Or compile without the flag
cmake -DCMAKE_HIP_FLAGS="-DRDNA2_OPT_V1=1 -DRDNA2_ASYNC_PIPELINE=1" -S . -B build
```
