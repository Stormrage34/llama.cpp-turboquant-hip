# RDNA2 Optimization Results

**Date:** 2026-07-02 04:19:22
**GPU:** AMD Radeon RX 6800 XT (gfx1030, RDNA2)
**Source:** hipfire #298-#304 RDNA2 research

## Baselines

### Dense (12B)
| Batch | pp512 (t/s) | tg128 (t/s) |
|-------|-------------|-------------|
| 512 | 1340.6 | 54.6 |

### MoE (26B-A4B)
| Batch | pp512 (t/s) | tg128 (t/s) |
|-------|-------------|-------------|
| 512 | 2586.2 | 101.8 |

## Optimization Results

| Variant | Description | Dense pp512 | Dense tg128 | MoE pp512 | MoE tg128 | Dense pp delta | Dense tg delta |
|---------|-------------|-------------|-------------|-----------|-----------|----------------|----------------|
| baseline | Standard build, no turbo FA instances. B | 1340.6 | 54.6 | 2586.2 | 101.8 | +0.0% | +0.0% |
| hipgraph | hipGraph capture enabled. hipfire #300 F | 1338.1 | 54.7 | 2602.7 | 102.8 | -0.2% | +0.2% |

## Batch Size Sweep (Dense 12B)

| Batch | pp512 (t/s) | tg128 (t/s) | Wave alignment |
|-------|-------------|-------------|----------------|
| 512 | 1340.6 | 54.6 | OK |

## hipfire Cross-Reference

| hipfire Lever | hipfire Result | Our Result | Notes |
|---------------|----------------|------------|-------|
| hipgraph | #300 F1: hipGraph for prefill (+10-20%) | -0.2% pp | |
