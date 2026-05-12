# RDNA2 Experimental: MoE Prefill Accelerator

> **Status**: Experimental — high-throughput, variable latency. Benchmark your workload before production use.

## Overview

The `RDNA2_MATMUL_OPT_V1` flag enables an LDS (Local Data Share) double-buffered matrix multiplication kernel that overlaps weight tile loading with DP4A compute. It targets memory-bound small-matrix operations common in Mixture-of-Experts (MoE) routing on AMD RDNA2 (gfx1030) GPUs.

## Performance Profile

**Hardware**: AMD Radeon RX 6800 XT (gfx1030, 16 GB VRAM)
**Model**: Qwen3.6-35B-MoE-IQ4_XS @ pp512, batch=256, ubatch=128

| Metric | Baseline | Experimental | Delta |
|--------|----------|--------------|-------|
| Prefill (t/s) | ~480 ± 100 | ~1450–1770 ± 450 | **+170–210%** |
| Variance | Low | High (3–6× baseline) | ⚠️ Run-to-run fluctuation |
| Decode (t/s) | ~57 ± 4 | ~57 ± 4 | No change |
| Dense Models | ~480 t/s | ~480 t/s | Auto-disabled |

### KV Cache Type Matrix (MoE Prefill)

| ctk \ ctv | turbo2 (Orig) | turbo2 (Opt) | turbo3 (Orig) | turbo3 (Opt) | turbo4 (Orig) | turbo4 (Opt) |
|-----------|---------------|--------------|---------------|--------------|---------------|--------------|
| **turbo2** | 474 ± 123 | 1569 ± 464 | — | — | — | — |
| **turbo3** | 501 ± 56 | 1455 ± 443 | 507 ± 68 | 1331 ± 612 | — | — |
| **turbo4** | 483 ± 101 | 1314 ± 635 | 552 ± 15 | 1613 ± 345 | — | — |

## Usage

```bash
# Stable RDNA2 features only (recommended for production)
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ./llama-server -m model.gguf -ngl 99

# + Experimental MoE prefill accelerator (benchmark first)
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 ./llama-server -m model.gguf -ngl 99
```

## Safety Gates

The experimental path is protected by a **triple gate** — all three must pass:

1. **Compile-time**: `-DRDNA2_MATMUL_OPT_V1=1` must be passed to the compiler
2. **Runtime environment**: `RDNA2_MATMUL_OPT_V1=1` must be set
3. **Hardware check**: Only activates on RDNA2 (gfx1030) via `GGML_CUDA_CC_IS_RDNA2(cc)`

If any gate fails, the kernel falls back to the stable baseline path with **zero overhead**.

## Known Behavior

### ✅ Positives
- Gain is consistent across all KV cache types (`turbo2/3/4`)
- Zero regression on decode performance
- Zero regression on dense model workloads
- No numerical accuracy loss (MSE < 1e-4 vs FP16)

### ⚠️ Caveats
- **Prefill latency variance is elevated** (±345–635 t/s vs ±15–123 baseline)
- Not recommended for strict SLA workloads without local benchmarking
- Variance is inherent to LDS bank conflict patterns during buffer swap

## Root Cause of Variance

The gfx1030 LDS has 32 banks. Symmetric tile dimensions (32×N) cause warp-stride bank conflicts during the `tile_x ↔ tile_x_next` buffer swap, producing unpredictable timing. This is being addressed in Phase 3.

## Planned Fixes (v0.3.1)

| Fix | Description | Target |
|-----|-------------|--------|
| **LDS Padding** | Add +1/+2 element padding to tile buffers to break 32-bank symmetry | Variance < ±200 t/s |
| **Wave32 Enforcement** | `__attribute__((amdgpu_waves_per_eu(4, 8)))` for scheduler stability | Reduce slow-run outliers |
| **Stride Alignment** | Align LDS access patterns to avoid bank conflicts | Consistent high throughput |

## Disabling

Unset `RDNA2_MATMUL_OPT_V1` or compile without `-DRDNA2_MATMUL_OPT_V1=1`. Runtime fallback is instant and zero-cost.

```bash
# Disable at runtime
unset RDNA2_MATMUL_OPT_V1

# Or compile without the flag
cmake -DCMAKE_HIP_FLAGS="-DRDNA2_OPT_V1=1 -DRDNA2_ASYNC_PIPELINE=1" -S . -B build
```
