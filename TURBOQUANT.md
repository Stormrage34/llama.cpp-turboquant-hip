# TurboQuant KV Cache Compression — HIP/ROCm

3-bit KV cache compression for llama.cpp on AMD GPUs. Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Results

### GSM8K Math Accuracy (temperature=0)

| Model | f16 | turbo3 | Drop | Compression | N |
|---|---|---|---|---|---|
| **Qwen3.5-27B** Q5_K_M | 66%* | **72.0%** | +6% | 5× | 1319 |
| **Gemma 4 26B-A4B** Q4_K_M | 83%* | **81.5%** | -1.5% | 2.9× | 100×2 |
| **Gemma 4 31B Dense** Q4_K_M | 96%* | **97%** | +1% | 2.9× | 100 |

*f16 baselines from 100-problem subset. Qwen3.5 turbo3 validated on full 1319 problems.

Best Gemma 4 config: `--cache-type-k turbo3 --cache-type-v turbo3 --cache-type-k-swa turbo3 --cache-type-v-swa q8_0`

### Needle-in-a-Haystack (Qwen3.5-27B, turbo3 K+V)

| Context | d=0.0 | d=0.25 | d=0.5 | d=0.75 | d=1.0 |
|---|---|---|---|---|---|
| 2K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 16K | ✅ | ✅ | ✅ | ✅ | ✅ |

20/20 passed — no retrieval degradation up to 16K context.

### Tool Calling (Qwen3.5-27B, turbo3 K+V)

15/15 tests passed (100%) — correct tool selection and parameter extraction.
Tested: get_weather, send_email, search_web, calculate, create_reminder.

### WikiText-2 Perplexity

| Model | f16 PPL | turbo3 PPL | Δ | Compression |
|---|---|---|---|---|
| Qwen3.5-27B (4K) | 6.6641 | 6.6657 | +0.02% | 5× |
| Qwen3.5-27B (16K) | 6.0729 | 6.4521 | +6.2% | 5× |

### Speed (RX 7900 XTX, ROCm 6.4)

| Model | Context | Config | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|
| Qwen3.5-27B | 512 | f16 | 427 | 30.15 |
| | 512 | turbo3 | 423 (-1%) | 29.49 (-2%) |
| | 8K | f16 | 340 | 30.13 |
| | 8K | turbo3 | 337 (-1%) | 29.74 (-1%) |
| Gemma 4 26B | 512 | f16 | 2939 | 94.53 |
| | 512 | turbo3 | 2878 (-2%) | 87.13 (-8%) |

Minimal speed overhead: 1-2% on standard models, 8% decode on Gemma 4 (D=512 WHT cost).

### Comparison with AmesianX (CUDA)

| | domvox (HIP) | AmesianX (CUDA) |
|---|---|---|
| **Gemma 4 accuracy drop** | **-1.5%** | **-19%** |
| Platform | AMD ROCm | NVIDIA CUDA |
| Compression (Gemma 4) | 2.9× | 5.2× |
| Speed overhead | 1-8% | N/A |
| TriAttention pruning | Yes | No |

### TriAttention KV Pruning (Qwen3.5-27B, 16K context)

| Config | PPL | KV rows |
|---|---|---|
| Baseline | 6.0729 | 16384 |
| TriAttention 75% | **5.9939 (-1.3%)** | ~5989 |
| TriAttention 50% | 6.0890 (+0.26%) | ~2550 |

## Usage

```bash
# Standard (Qwen, Llama, Mistral — head_dim ≤ 256)
llama-server -m model.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3

# Gemma 4 (recommended — minimal accuracy drop, 2.9× compression)
llama-server -m gemma4.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --cache-type-k-swa turbo3 --cache-type-v-swa q8_0

# With TriAttention KV pruning (long context)
llama-server -m model.gguf -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
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
| Gemma 4 (ISWA, global+SWA) | 512/256 | ✅ Works (see notes) |
| DeepSeek (MLA) | 576/512 | Untested |

### Gemma 4 notes

Gemma 4 has two attention types (ISWA):
- **SWA layers** (25/30): head_dim=256, properly compressed with turbo3
- **Global layers** (5/30): head_dim=512, FA TILE kernel reads turbo3 without dequant

On AMD/HIP, the flash attention dispatch for D=512 falls through to the TILE kernel,
which has no turbo3 dequantization. The 25 SWA layers dominate output quality.
When proper D=512 VEC dequant was added, accuracy dropped from 81% to 68% — structured
quantization noise is more harmful than the TILE fallback behavior.

## Features

- **Attention sharpening** — K-side α = 1 + 1/(2×SQNR) compensates softmax flattening
- **TriAttention KV pruning** — frequency-based scoring + GPU compaction kernel
- **Hybrid model support** — SSM+attention (Qwen3.5), ISWA (Gemma 4)
- **FP32 WHT butterfly** — no precision loss in rotation

## Known issues

### OOM with turbo KV on partial-offload setups (e.g. 16GB VRAM + large model)

When the model doesn't fully fit in VRAM, `--fit` reduces context to fit.
With turbo KV, the KV cache is much smaller, so `--fit` allows a larger context
(e.g. 262K instead of 143K). But the compute buffer still scales with prompt length,
and at large prompts (94K+) it can exceed available VRAM.

**Workaround:** Set `-c` explicitly (e.g. `-c 131072`) instead of relying on the default.

This is not a turbo-specific bug — turbo just exposes a `--fit` limitation by making
the KV cache small enough that context isn't reduced.

## Known limitations

- **GROUP_SIZE=256**: Implemented but decode produces garbage. Root cause unknown.
- **Gemma 4 D=512 FA**: TILE kernel fallback, not proper dequant. See notes above.
- **TriAttention + turbo3 combo**: No additive benefit at short contexts (<1K tokens).

## Hardware

Tested on: AMD Ryzen 9 9950X3D, RX 7900 XTX 24GB, ROCm 6.4, openSUSE Tumbleweed.
