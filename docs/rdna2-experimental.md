# RDNA2 Experimental: MoE Prefill Accelerator

## Overview

The `RDNA2_MATMUL_OPT_V1` flag enables an LDS double-buffered matrix multiplication kernel that overlaps weight loading with DP4A compute. It targets memory-bound small-matrix operations common in Mixture-of-Experts (MoE) routing on AMD RDNA2 (gfx1030) GPUs.

## Performance Profile

**Hardware**: AMD Radeon RX 6800 XT (gfx1030, 16 GB VRAM)
**Model**: Qwen3.6-35B-MoE-IQ4_XS @ pp512, batch=256, ubatch=128

| Metric | Baseline | Experimental | Notes |
|--------|----------|--------------|-------|
| Prefill (t/s) | ~480 ± 100 | ~1450 ± 450 | **+170–210% mean gain** |
| Variance | Low | High (3–6× baseline) | Run-to-run timing fluctuation |
| Decode (t/s) | ~57 ± 4 | ~57 ± 4 | Zero impact |
| Dense Models | ~480 t/s | ~480 t/s | Auto-disabled |

## Usage

```bash
# Stable RDNA2 features only (recommended for production)
RDNA2_OPT_V1=1 ./llama-server ...

# + Experimental MoE prefill accelerator (benchmark first)
RDNA2_OPT_V1=1 RDNA2_MATMUL_OPT_V1=1 ./llama-server ...
```

## Safety Gates

The experimental path is protected by a **triple gate**:

1. **Compile-time**: `-DRDNA2_MATMUL_OPT_V1=1` must be passed to the compiler
2. **Runtime environment**: `RDNA2_MATMUL_OPT_V1=1` must be set
3. **Hardware check**: Only activates on RDNA2 (gfx1030) via `GGML_CUDA_CC_IS_RDNA2(cc)`

If any gate fails, the kernel falls back to the stable baseline path with zero overhead.

## Known Behavior

- Gain is consistent across KV cache types (`turbo2/3/4`)
- Zero regression on decode or dense workloads
- Prefill latency variance is elevated. Not recommended for strict SLA workloads without local benchmarking
- Stabilization tracking: LDS padding + Wave32 enforcement (planned v0.3.1)

## Disabling

Unset `RDNA2_MATMUL_OPT_V1` or compile without `-DRDNA2_MATMUL_OPT_V1=1`. Runtime fallback is instant and zero-cost.
