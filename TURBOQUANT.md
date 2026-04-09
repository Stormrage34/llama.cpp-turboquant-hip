# TurboQuant KV Cache Compression — HIP/ROCm

3-bit KV cache compression for llama.cpp on AMD GPUs. Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Results

### Qwen3.5-27B Q5_K_M (RX 7900 XTX, WikiText-2)

| KV cache | PPL (4K) | PPL (16K) | KV memory |
|---|---|---|---|
| f16 | 6.6641 | 6.0729 | 256 MiB |
| **turbo3** | **6.6657** | — | **50 MiB (5×)** |

+0.02% PPL at 5× compression.

### Gemma 4 26B-A4B Q4_K_M (RX 7900 XTX, GSM8K math)

| Config | GSM8K (100) | KV memory | Compression |
|---|---|---|---|
| f16 | 83% | 340 MiB | 1.0× |
| turbo3 + f16 SWA | 82% | ~120 MiB | 2.8× |
| **turbo3 + turbo3-K-SWA + q8_0-V-SWA** | **83%** | **117 MiB** | **2.9×** |

**0% accuracy drop** at 2.9× compression (best config).

### Qwen3.5-27B + TriAttention KV Pruning (16K context)

| Config | PPL | KV cache rows |
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

# Gemma 4 (best quality — SWA in f16)
# llama-server -m gemma4.gguf -ngl 99 \
#   --cache-type-k turbo3 --cache-type-v turbo3 \
#   --cache-type-k-swa f16 --cache-type-v-swa f16

# Gemma 4 (best compression — 0% accuracy drop)
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
| Gemma 4 (ISWA, global+SWA) | 512/256 | ✅ Global turbo3, SWA f16 |
| DeepSeek (MLA) | 576/512 | Untested |

## Features

- **Attention sharpening** — compensates softmax flattening from quantization noise (α = 1 + 1/2×SQNR)
- **TriAttention KV pruning** — frequency-based scoring + GPU compaction kernel
- **Hybrid model support** — SSM+attention (Qwen3.5), ISWA (Gemma 4)
- **FP32 WHT butterfly** — no precision loss (unlike FP16 implementations)

## Comparison with other implementations

| | domvox (HIP) | AmesianX (CUDA) |
|---|---|---|
| Platform | **AMD ROCm** | NVIDIA CUDA |
| Gemma 4 accuracy drop | **0%** | -19% |
| Architecture | 1 kernel template | 20+ block types |
| TriAttention pruning | **Yes** | No |
| MMA tensor core | No | Yes |
| Gemma 4 SWA compression | f16 (preserves quality) | turbo3 (loses quality) |
