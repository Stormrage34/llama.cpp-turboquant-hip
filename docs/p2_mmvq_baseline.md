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

## P2.1 Opportunity Assessment
| Optimization | Expected Impact | Risk |
|-------------|----------------|------|
| float4/uint4 vector loads (coalescing) | ↓VMEM cycles, ↓FETCH_SIZE | VGPR pressure increase |
| Reduce instruction count (SALU offload) | ↓WAVE_ISSUE_WAIT | Medium complexity |
| Wave32 optimization | ↓SQ_WAVES, better occupancy | Medium |

**Recommendation**: Proceed with coalescing (P2.1) but note the primary bottleneck is instruction-issue; expect modest gains (~5-8% tg32) unless combined with instruction reduction.
