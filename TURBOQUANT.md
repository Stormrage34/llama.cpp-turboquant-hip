# TurboQuant KV Cache Compression — HIP/ROCm

3-bit KV cache compression for llama.cpp on AMD GPUs. Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Results

### GSM8K Math Accuracy (100 problems, temperature=0)

| Model | f16 | turbo3 | Drop | Compression |
|---|---|---|---|---|
| **Gemma 4 31B Dense** Q4_K_M | 96% | **97%** | **+1%** | 2.9× |
| **Gemma 4 26B-A4B** Q4_K_M | 83% | **83%** | **0%** | 2.9× |
| Qwen3.5-27B Q5_K_M | 66% | TBD | — | 5× |

Gemma 4 best config: `--cache-type-k turbo3 --cache-type-v turbo3 --cache-type-k-swa turbo3 --cache-type-v-swa q8_0`

### WikiText-2 Perplexity

| Model | f16 PPL | turbo3 PPL | Δ | Compression |
|---|---|---|---|---|
| Qwen3.5-27B (4K) | 6.6641 | 6.6657 | +0.02% | 5× |

### Comparison with AmesianX (CUDA)

| | domvox (HIP) | AmesianX (CUDA) |
|---|---|---|
| **Gemma 4 accuracy drop** | **0%** | **-19%** |
| Platform | AMD ROCm | NVIDIA CUDA |
| Compression (Gemma 4) | 2.9× | 5.2× |
| TriAttention pruning | Yes | No |
| MMA tensor core | No | Yes |

Our implementation preserves quality significantly better. AmesianX compresses
all layers (including SWA) which destroys quality on Gemma 4. We keep SWA in
turbo3-K + q8_0-V which maintains full accuracy.

### TriAttention KV Pruning (Qwen3.5-27B, 16K context)

| Config | PPL | KV rows |
|---|---|---|
| Baseline | 6.0729 | 16384 |
| TriAttention 75% | **5.9939 (-1.3%)** | ~5989 |
| TriAttention 50% | 6.0890 (+0.26%) | ~2550 |

TurboQuant + TriAttention = compression × pruning for extreme KV reduction.

## Usage

```bash
# Standard (Qwen, Llama, Mistral — head_dim ≤ 256)
llama-server -m model.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3

# Gemma 4 (recommended — 0% accuracy drop)
llama-server -m gemma4.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --cache-type-k-swa turbo3 --cache-type-v-swa q8_0

# With TriAttention KV pruning
llama-perplexity -m model.gguf -ngl 99 \
  --triattention stats.bin --tri-budget 75 --tri-window 512
```

## Build (ROCm/HIP)

```bash
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Supported models

| Architecture | head_dim | Status |
|---|---|---|
| Qwen3/3.5, Llama, Mistral | 128 | ✅ Full support |
| Qwen3.5-27B (hybrid SSM+attn) | 256 | ✅ Full support |
| Gemma 4 (ISWA, global+SWA) | 512/256 | ✅ Global turbo3, SWA turbo3-K+q8_0-V |
| DeepSeek (MLA) | 576/512 | Untested |

## Features

- **Attention sharpening** — compensates softmax flattening from quantization noise (α = 1 + 1/2×SQNR)
- **TriAttention KV pruning** — frequency-based scoring + GPU compaction kernel
- **Hybrid model support** — SSM+attention (Qwen3.5), ISWA (Gemma 4)
- **FP32 WHT butterfly** — no precision loss (unlike FP16 implementations)

## Hardware

Tested on: AMD Ryzen 9 9950X3D, RX 7900 XTX 24GB, ROCm 6.4, openSUSE Tumbleweed.
