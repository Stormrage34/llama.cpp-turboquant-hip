# v0.3.1.1 Validation Table

Fill this table for each candidate commit before merging.

## Functional

| Test | Command | Pass Criteria | Result |
|------|---------|---------------|--------|
| Numerical parity | `--temp 0.0` 10 runs vs CPU baseline | 0 mismatches, MSE <1e-4 | □ Pass □ Fail |
| Multi-turn stability | 3 prompts, 64 tokens each | Coherent output, no `????` or `HereHere` | □ Pass □ Fail |
| Fallback (no flags) | Unset `RDNA2_OPT_V1` | Identical output to baseline | □ Pass □ Fail |
| Fallback (no asm) | Unset `RDNA2_MATMUL_OPT_V1` | Identical output, no NaN | □ Pass □ Fail |

## Performance

| Test | Config | Expected | Measured |
|------|--------|----------|----------|
| Prompt t/s | `pp512 -r 10` | >80 t/s | ____ |
| Generation t/s | `-n 128 -r 10` | >20 t/s | ____ |
| LDSBankConflict | rocprofv3 | ≤5% | ____ % |
| VALUBusy | rocprofv3 | ≥60% | ____ % |
| MeanOccupancy | rocprofv3 | ≥4 waves/EU | ____ |

## Memory

| Test | Config | Expected | Measured |
|------|--------|----------|----------|
| Peak VRAM | rocm-smi | ≤13.5 GB | ____ GB |
| Post-run VRAM | rocm-smi after exit | ≤2 GB | ____ GB |
| Context alloc | 8k ctx, 27B IQ4_XS | succeeds | □ Success □ Fail |

## Hardware

| Field | Value |
|-------|-------|
| GPU | RX 6800 XT (gfx1030) |
| ROCm | 7.1.3 |
| Driver | amdgpu |
| OS | CachyOS |
| Model | Qwen3.6-27B-IQ4_XS.gguf |

*Run date:* ____________  
*Commit:* ____________  
*Tester:* ____________  
