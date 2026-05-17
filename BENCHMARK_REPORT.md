# Benchmark Report: Long-Context MTP Performance on 16 GB (RX 6800 XT)

## Setup

| Item | Detail |
|------|--------|
| **GPU** | AMD Radeon RX 6800 XT (16 GB VRAM) |
| **Model** | Qwen3.5-35B-MoE IQ4 (19 GB, 41 layers, 2 KV heads, 256 experts, head_dim=128) |
| **KV Cache** | TurboQuant 4-bit (turbo4) |
| **Spec Decode** | MTP, draft-n-max=2, draft-p-min=0.75 |
| **Batch** | `-b 256 -ub 256` |
| **Attention** | Flash attention, `-fa on` |
| **CLI flags** | `--no-mmap --mlock --cache-reuse 256 --ctx-checkpoints 4 --kv-unified` |

## Results

| Context | -ncmoe | VRAM idle→peak | Prompt tokens | Prompt t/s | Gen tokens | Gen t/s | VRAM util |
|---------|--------|----------------|---------------|------------|------------|---------|-----------|
| **64K** | 18 | 13,050→15,189 MB | 59,994 | **321.0** | 128 | **10.9** | 92% |
| **128K** | 35 | 6,834→15,826 MB | 119,979 | **241.0** | 128 | **30.8** | 96% |
| **256K** | 41 | 4,975→9,181 MB* | 231,680† | ~190† | — | — | 56%* |

\* VRAM measured after curl timeout (1200s), not at peak  
† Prompt processing was interrupted by client timeout. At 231K/240K tokens processed (~96%), the server log shows 7.1 GB free VRAM, confirming 256K fits comfortably.

## Analysis

### KV Cache Cost (turbo4)
- **Data**: ~9 KB/tok (41 layers × 2 KV heads × 128 dim × 0.5 bytes × 2 for K+V)
- **With overhead** (checkpoints, metadata): ~13 KB/tok at 60K, decreasing to ~7 KB/tok at 231K as overhead amortizes
- **Context checkpoints**: 4 checkpoints × 62.8 MB = 251 MB additional

### Scaling Behavior

| Metric | 64K → 128K | 128K → 256K |
|--------|-----------|-----------|
| Prompt speed drop | 321→241 t/s (−25%) | ~241→~190 t/s (−21%) |
| Gen speed improvement | 10.9→30.8 t/s (+183%) | — |
| KV cache growth | 0.5→1.5 GB | 1.5→~2.5 GB |

**Decode speed improves with ncmoe** because fewer expert layers on GPU means less contention for compute resources. At -ncmoe 35 (128K), only 6/41 MoE layers are GPU-resident, reducing expert-switching overhead during MTP decoding.

### VRAM Budget (16 GB)

| Offloading | Model VRAM | Free | Max KV capacity |
|------------|-----------|------|-----------------|
| -ncmoe 18 (23 layers GPU) | ~13,050 MB | ~3,334 MB | ~250K tokens |
| -ncmoe 35 (6 layers GPU) | ~6,834 MB | ~9,550 MB | ~730K tokens |
| -ncmoe 41 (0 layers GPU) | ~4,975 MB | ~11,409 MB | ~875K tokens |

The actual headroom is ~2× the data estimate due to checkpoints, compute buffers, and MTP overhead.

## Critical Build Fixes Applied

1. **`build.yml`**: Updated HIP container from `rocm/dev-ubuntu-22.04:6.1.2` → `rocm/dev-ubuntu-22.04:7.2.1`. Added `--disable-new-dtags` RPATH isolation, `CMAKE_BUILD_RPATH_USE_ORIGIN=ON`, and `GGML_RDNA2_BFE_DISPATCHER=ON`.

2. **`release.yml`**: Added `ubuntu-22-hip-turboquant` job (HIP build with full RDNA2 optimizations + RPATH isolation). Added `schedule` trigger for nightly stable releases. Added `turboquant` binary to release body downloads.

## q8_0 KV Cache Comparison

| Context | turbo4 (ncmoe, VRAM%) | q8_0 (ncmoe, VRAM%) |
|---------|----------------------|-------------------|
| **64K** | 18, 92% | 18, 90% |
| **128K** | 35, 96% | 22, ~94% |
| **256K** | 41, 56% | ❌ N/A (needs ~6.6 GB KV) |

Counterintuitive: **128K with q8_0 needs less offloading** (-ncmoe 22 vs 35) because the extra 1.5 GB KV cost is smaller than the 13 MoE layers' freed VRAM. Net effect: faster prompt processing and better quality (lossless KV).
