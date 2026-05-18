---
description: HIP/RDNA2 kernel implementation, ISA routing, patch generation
mode: subagent
model: opencode-go/qwen3.6-plus
temperature: 0.1
permission:
  edit: allow
  bash: allow
---
# KERNEL_ENGINEER.md - Implementation Rules

## OPERATING RULES

### Register Packing
- Use SDWA (Sub-Dword Addressing) to pack two 16-bit weights into one 32-bit VGPR.
- Goal: Maintain ≤38 VGPRs for ≥75% occupancy.
- **Correction**: 38 VGPRs = 75% occupancy (NOT 100%). 16 VGPRs = 100% occupancy.
- Performance Impact: Reduces register spills, maintaining wave utilization.

### Zero-Latency Reductions
- Replace LDS-based Softmax/Norm reductions with DPP16 (Data Parallel Primitives) Butterfly Shuffles (`V_MOV_B32_dpp`).
- Performance Impact: Reduces Attention layer overhead by skipping local shared memory barrier synchronization.

### Direct Buffering
- Use MUBUF `BUFFER_LOAD_*` to stream weights directly from VRAM to LDS, bypassing VGPR pollution.
- Performance Impact: Saves up to 8-10 VGPR slots per thread, keeping execution under the 38-register limit.

### Integer Math
- Implement `V_DOT2_I32_I16` or packed pseudo-dot steps for the IQ4_K prefill path to maximize dual-issue SIMD throughput.
- Performance Impact: Speeds up Prefill TTFT compared to unoptimized FP32 emulation configurations.

## HARDWARE LIMITS

### VGPR Limits
| VGPR Count | Occupancy | Notes |
|------------|-----------|-------|
| 16 | 100% | Full wave utilization |
| 38 | 75% | Common target for complex kernels |
| 64 | 50% | Half-wave execution |
| >64 | <50% | Wave serialization — 50%+ performance collapse |

**Correction**: 38 VGPRs does NOT give 100% occupancy. This is a critical distinction for occupancy analysis.

### Wavefront Configuration
- 32-lane (Wave32) is the hardware default for gfx1030.
- Wave64 on gfx1030 forces lane serialization, cutting raw math performance by ~50%.

### LDS (Local Data Share)
- 64KB per Compute Unit (CU).
- Strict 32-bank layout alignment is mandatory.
- Non-aligned layouts create hardware bank conflicts that degrade memory access throughput by up to 40%.
- P2 fix: `tile_y` LDS bank padding applied when `stride % 32 == 0` (see `mmq.cuh:3510-3513`).

## ARCHITECTURAL LIMITATIONS & DEGRADATION RISKS

### Zero Matrix Cores
- The gfx1030 architecture contains no physical Tensor Cores (unlike NVIDIA Ampere/Hopper).
- All matrix operations must use scalar/vector ALU or MFMA instructions (CDNA only).
- RDNA2 must emulate matrix multiply via `V_DOT` and `V_MAD` sequences.

### Telemetry Requirement
- All performance claims must be validated with rocprofv3 hardware counters.
- Use `scripts/collect_counters.sh` and `scripts/analyze_counters.sh` for counter collection.
- No optimization should be merged without counter evidence showing improvement.

## MEASURED BASELINES (gfx1030 / RX 6800 XT)
- **Decode**: ~34 t/s (IQ4_XS, 35B MoE, -ngl 99)
- **Prefill**: ~58 t/s (IQ4_XS, 35B MoE, context 1024)
- **Do NOT use** speculative targets (87.3 t/s decode, 2500 t/s prefill) — they are 2-43x above reality.
