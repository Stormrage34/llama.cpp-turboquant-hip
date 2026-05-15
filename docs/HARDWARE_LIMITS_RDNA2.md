# RDNA2 Hardware Limits (RX 6800 XT)

## VRAM Budget
| Component | Size |
|-----------|------|
| Total VRAM | 16,368 MiB (gfx1030) |
| Soft limit (fence threshold) | ~15,500 MiB |
| Hard OOM boundary | ~15,800 MiB |

## Model Fit Guide
### Fits entirely (≤15.5 GiB model data)
- Up to **~4,400 MiB** model weights at default context (131k)
- Example: Meta-Llama-3.1-8B Q4_K_M (~4.6 GiB total, ~4.4 GiB on GPU) ✅

### Requires context reduction or partial offload
- Models 5–12 GiB: reduce `-c` context to fit (fit params auto-adjusts)
- 35B MoE IQ4 (19 GiB): use `--ncmoe 30` + `-c 4096`
- 35B Q4_K_M (22 GiB): use `--ncmoe 40` + `-c 2048`

### Does not fit (exceeds 15.5 GiB)
- 35B Q5_K_M (24+ GiB): GTT spill → instability
- Gemma-4-26B Q4_K_M (16 GiB): borderline, requires `-c 1024` or split

## KV Cache Sizing
| Context | KV Cache Size | Available for Weights |
|---------|--------------|----------------------|
| 2048 | 256 MiB | ~15,800 MiB |
| 4096 | 512 MiB | ~15,550 MiB |
| 8192 | 1,024 MiB | ~15,000 MiB |
| 16384 | 2,048 MiB | ~14,000 MiB |
| 32768 | 4,096 MiB | ~12,000 MiB |
| 131072 | 16,384 MiB | exceeds VRAM |

## Key Constraints
- **MTP speculative decode**: requires ~256+ MiB free VRAM. Guarded by proactive fence (500 MiB threshold, 10-iter cooldown)
- **MoE expert offload** (`--ncmoe N`): reserve compute buffer per active expert. N=30 for 35B MoE IQ4 is the sweet spot
- **TurboQuant calibration**: uses ~50 MiB of device constants + calibration buffers
