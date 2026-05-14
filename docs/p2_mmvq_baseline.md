# P2.1 Baseline Profile — mul_mat_vec_q (Q4_K fusion)
# Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# GPU: AMD Radeon RX 6800 XT (gfx1030)
# ROCm: 7.13 nightly, rocprofv3 v1.3.0
# Date: 2026-05-14
# Benchmark: llama-bench -p 512 -n 128 -b 1 -ub 1 -fa 1 -r 3
# Build: 8ec3d861b (9086)

## Hot-Path Confirmation
| Kernel | Dispatches | Avg Duration (µs) | Total GPU Time (s) | % of Total |
|--------|-----------|-------------------|-------------------|------------|
| mul_mat_vec_q Q4_K (fusion) | 192,207 | 80.24 | 15.42 | 58.5% |
| mul_mat_vec_q Q4_K (no fusion) | 197,073 | 13.6 | 2.69 | 10.2% |
| mul_mat_vec_q Q6_K (fusion) | 38,928 | 97.7 | 3.80 | 14.4% |
| mul_mat_vec_q Q6_K (no fusion) | 41,361 | 57.8 | 2.39 | 9.1% |
| quantize_q8_1 | 469,569 | 1.3 | 0.62 | 2.4% |
| rms_norm_f32 | 158,145 | 3.5 | 0.56 | 2.1% |
| flash_attn_tile | 77,856 | 6.4 | 0.50 | 1.9% |
| **All mul_mat_vec_q** | **469,569** | — | **24.30** | **92.2%** |

## Memory Subsystem (Q4_K fusion kernel)
| Metric | Avg | Min | Max | Interpretation |
|--------|-----|-----|-----|---------------|
| MemUnitBusy | 85.29% | 56.35% | 100% | Moderate utilization, ~15% headroom |
| FETCH_SIZE | 36,316 | 9,237 | 67,742 | Bytes fetched per dispatch (cumulative) |
| SQ_INST_CYCLES_VMEM | 156,454 | 0 | 3,444,224 | Vector memory instruction cycles |
| LDSBankConflict | 211.77 | 0 | 8,120 | Low — LDS is not a bottleneck |
| ALUStalledByLDS | 0.12 | 0 | 12.61 | Negligible |

## Compute Subsystem (Q4_K fusion kernel)
| Metric | Avg | Note |
|--------|-----|------|
| SQ_INSTS_VALU | 171,807 | VALU instructions per dispatch |
| SQ_INSTS_VALU : SQ_INST_CYCLES_VMEM | 1.10 : 1 | Roughly balanced compute vs memory |

## Wave/Stall Analysis (Q4_K fusion kernel)
| Metric | Avg | Note |
|--------|-----|------|
| WAVE_ISSUE_WAIT | 52,560 | Instruction issue wait — primary bottleneck |
| WAVE_DEP_WAIT | 24,655 | Data dependency wait — secondary |
| Ratio ISSUE:DEP | 2.13 : 1 | Kernel is instruction-issue-bound |
| SQ_WAIT_INST_ANY | 1,492,690 | Total wait cycles for instruction dependencies |
| SQ_WAIT_INST_LDS | 185,160 | Wait cycles for LDS — significant proportion |
| MeanOccupancyPerCU | 1,263 | Scale-dependent — needs normalization |
| SQ_WAVES | 9,239,872 | Cumulative wave count |

## Bottleneck Analysis
1. **Primary: Instruction-Issue-Bound** — WAVE_ISSUE_WAIT (52,560) > WAVE_DEP_WAIT (24,655)
   - GPU spends more time waiting for instruction scheduling than data
   - Implies complex instruction mix or high VGPR pressure limits ILP
2. **Secondary: Memory Bandwidth** — MemUnitBusy at 85%, not saturated
   - Room for ~15% throughput gain from memory optimization
   - Coalescing improvements would reduce VMEM cycles and could raise MemUnitBusy
3. **LDS is Not a Bottleneck** — LDSBankConflict low, ALUStalledByLDS negligible

## P2.1 Reality Check: Memory Coalescing is NOT the Bottleneck

**Critical finding:** The kernel is **instruction-issue-bound**, NOT memory-coalescing-bound.

Evidence:
1. **WAVE_ISSUE_WAIT (52,560) > WAVE_DEP_WAIT (24,655)** — 65% of dispatch time the pipeline is stalled waiting for instructions to issue
2. **MemUnitBusy at 85%** — memory bandwidth is NOT the bottleneck (~15% headroom)
3. **VALU:VMEM ratio 1.10:1** — balanced, not memory-heavy
4. **LDSBankConflict at 211 (low), ALUStalledByLDS at 0.12 (negligible)** — no secondary bottlenecks

**Why coalescing doesn't help:** The hardware already coalesces consecutive 4-byte thread accesses into 16-128B transactions. Using `uint2`/`uint4` in source code reduces instruction count by at most 1-2 per vec_dot call (~2% of total instructions), which is within noise for WAVE_ISSUE_WAIT.

**VDR analysis:** Increasing VDR_Q4_K_Q8_1_MMVQ from 2 to 4 would change thread grouping but NOT reduce total instruction count (the vec_dot function is hardcoded to process 2 ints per call regardless of VDR). The loop overhead savings (~4 instructions/iteration) are negligible relative to total instruction count.

## P2.2 Pivot: Target the Real Bottleneck

The root cause of WAVE_ISSUE_WAIT is the complex instruction mix in the vec_dot functions. Each call involves:
- Global memory loads with high latency (q4, q8, scales)
- Bit manipulation for scales unpacking (masks, shifts, conditional)
- DP4A dot product computation
- Mixed-precision math (half-to-float conversions)

The GPU instruction scheduler cannot find enough independent work during the latency of these operations.

| Optimization | Expected Impact | Risk |
|-------------|----------------|------|
| VGPR reduction → higher occupancy → better latency hiding | ↓WAVE_ISSUE_WAIT 20-30% | Requires compiler-flag tuning |
| SALU offload of uniform address calcs | ↓VALU instructions 5-10% | Low |
| Branch reduction in scales unpacking | Instruction flow smoothing | Low |
| LDS tiling for weight sharing | ↓Global loads per wave 8x | High complexity |

**Recommendation**: Pivot P2.1 → P2.2. Target WAVE_ISSUE_WAIT directly. Start with VGPR usage analysis (compile with `-Rpass-analysis=kernel-resource-usage`), then apply SALU offload for the most accessible gains.
