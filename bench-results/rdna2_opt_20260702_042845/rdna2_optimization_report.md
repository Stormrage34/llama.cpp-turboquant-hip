# RDNA2 Optimization Results

**Date:** 2026-07-02 04:30:15
**GPU:** AMD Radeon RX 6800 XT (gfx1030, RDNA2)
**Source:** hipfire #298-#304 RDNA2 research

## Baselines

### Dense (12B)
| Batch | pp512 (t/s) | tg128 (t/s) |
|-------|-------------|-------------|
| 512 | 1333.3 | 54.6 |

### MoE (26B-A4B)
| Batch | pp512 (t/s) | tg128 (t/s) |
|-------|-------------|-------------|
| 512 | 2589.7 | 101.8 |

## Optimization Results

| Variant | Description | Dense pp512 | Dense tg128 | MoE pp512 | MoE tg128 | Dense pp delta | Dense tg delta |
|---------|-------------|-------------|-------------|-----------|-----------|----------------|----------------|
| baseline | Standard build, no turbo FA instances. B | 1333.3 | 54.6 | 2589.7 | 101.8 | +0.0% | +0.0% |
| hipgraph | hipGraph capture enabled. hipfire #300 F | 1332.1 | 54.8 | 2609.3 | 102.9 | -0.1% | +0.4% |
| fa_turbo | FA instances for all quant types (turbo/ | 1335.2 | 55.0 | 2605.5 | 103.5 | +0.1% | +0.8% |
| fa_graphs | FA + hipGraph combined. | 1333.1 | 54.5 | 2586.5 | 102.2 | -0.0% | -0.1% |

## Batch Size Sweep (Dense 12B)

| Batch | pp512 (t/s) | tg128 (t/s) | Wave alignment |
|-------|-------------|-------------|----------------|
| 512 | 1333.3 | 54.6 | OK |

## hipfire Cross-Reference

| hipfire Lever | hipfire Result | Our Result | Notes |
|---------------|----------------|------------|-------|
| hipgraph | #300 F1: hipGraph for prefill (+10-20%) | -0.1% pp | |
