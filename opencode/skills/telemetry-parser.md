---
name: telemetry-parser
description: Parse and interpret rocprofv3 SQLite outputs for RDNA2 validation
triggers: ["rocprofv3", "counter", "telemetry", "WAVE_ISSUE_WAIT", "MemUnitBusy"]
---
### 4. `telemetry-parser.md`
# Telemetry Parser Guide (rocprofv3 + RDNA2)

## Key Counters for MoE PCIe Optimization
| Counter | Target | Interpretation |
|---------|--------|---------------|
| `WAVE_ISSUE_WAIT` | ↓ ≥15% | Scheduler starved for independent work → PCIe latency bottleneck |
| `MemUnitBusy` | ↑ ≥88% | Memory unit saturated; <85% suggests underutilization or stalls |
| `SQ_INSTS_VALU` | ↓ ≥10% | Fewer vector ALU ops → better SALU offload or fused ops |
| `LDSBankConflict` | ≤3% | >5% indicates poor LDS access pattern (bank conflicts) |
| `VM_CNT` | Steady | High variance → irregular memory access, possible cache thrashing |

## SQLite Query Templates
```sql
-- Extract decode throughput + variance from llama-bench JSON
SELECT 
    json_extract(result, '$.tg128') as tg128,
    json_extract(result, '$.tg128_std_dev') as variance
FROM benchmarks 
WHERE model LIKE '%Qwen3_35BMTPIQ4%' 
  AND config LIKE '%-ncmoe 16%';

-- Compute WAVE_ISSUE_WAIT delta vs baseline
SELECT 
    run_id,
    counter_value - (SELECT counter_value FROM counters WHERE run_id = 'baseline') as delta
FROM counters 
WHERE counter_name = 'WAVE_ISSUE_WAIT'
  AND kernel = 'mul_mat_vec_q';
