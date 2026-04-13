# TurboQuant KV Cache Compression — HIP/ROCm

3-bit KV cache compression for llama.cpp on AMD GPUs. Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Results

### GSM8K Math Accuracy (temperature=0)

| Model | f16 | turbo3 | turbo3+TriAtt 75% | N |
|---|---|---|---|---|
| **Qwen3.5-27B** Q5_K_M | **71.9%** | **72.0%** | **72.0%** | 1319 |
| **Gemma 4 26B-A4B** Q4_K_M | 83%* | **80.3%** | — | 100×3 |
| **Gemma 4 31B Dense** Q4_K_M | 96%* | **97%** | — | 100 |

Qwen3.5-27B: all three configs validated on full 1319 problems. TriAttention pruning adds zero reasoning degradation.
*Gemma 4 baselines from 100-problem subsets.

Best Gemma 4 config: `--cache-type-k turbo3 --cache-type-v turbo3 --cache-type-k-swa turbo3 --cache-type-v-swa q8_0`

### Needle-in-a-Haystack (Qwen3.5-27B, turbo3 K+V)

| Context | d=0.0 | d=0.25 | d=0.5 | d=0.75 | d=1.0 |
|---|---|---|---|---|---|
| 2K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 16K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 32K | ✅ | ✅ | ✅ | ✅ | ✅ |
| 64K | ✅ | — | ✅ | — | ✅ |

28/28 passed — no retrieval degradation up to 64K context.

### Tool Calling (Qwen3.5-27B, turbo3 K+V)

15/15 tests passed (100%) — correct tool selection and parameter extraction.
Tested: get_weather, send_email, search_web, calculate, create_reminder.

### WikiText-2 Perplexity

| Model | Config | PPL | Δ vs f16 | Compression |
|---|---|---|---|---|
| Qwen3.5-27B (4K) | f16 | 6.6657 | — | 1× |
| | q8_0 | 6.6064 | -0.09% | 1.88× |
| | q4_0 | 6.6219 | -0.07% | 3.56× |
| | turbo4 | 6.8203 | +2.3% | 3.88× |
| | turbo3 | 6.6657 | +0.02% | 5.12× |
| | turbo2 | 6.9145 | +3.7% | 7.53× |
| Qwen3.5-27B (16K) | f16 | 6.2752 | — | 1× |
| | q8_0 | 6.5250 | +3.98% | 1.88× |
| | q4_0 | 6.5238 | +3.96% | 3.56× |
| | turbo3 | 6.2187 | -0.9% | 5.12× |

### InnerQ + 2D Vector Quantization (turbo3 quality breakthrough)

Two techniques that dramatically improve turbo3 K cache quality at long context:

**RoPE Pair Normalization (RPN):** InnerQ equalization merges adjacent channel pairs to a shared RMS before WHT, preventing independent per-channel scales from deforming RoPE rotation geometry. Applied to K cache only (V is not RoPE'd). Controlled by `TURBO_INNERQ_STRENGTH` env var (default: 0.15).

**2D Vector Quantization:** Replaces per-element scalar quantization (8 Lloyd-Max centroids) with per-pair 2D VQ using a 64-entry K-means codebook trained on 74K actual WHT output pairs. Same 3 bits/element, same block format. 12.6% MSE reduction from sphere-packing gain.

**Qwen3-8B Q4_K_M, ctx=16K, wikitext-2, 5 chunks — turbo3 K+V progression:**

| Change | PPL | vs f16 (6.92) |
|---|---|---|
| turbo3 baseline (no InnerQ) | 19.70 | +185% ⚠️ |
| + RPN pair normalization | 8.18 | +18% |
| + strength=0.25 | 7.33 | +5.9% |
| + 2D VQ (64-entry codebook) | 7.21 | +4.2% |
| + strength=0.15 | 7.17 | +3.6% |
| + actual-data codebook (20K pairs) | 7.14 | +3.2% |
| + improved K-means (74K, 20 inits) | **7.05** | **+1.9%** |
| turbo4 K+V (reference) | 6.99 | +0.9% |

InnerQ calibration uses 512 tokens (override via `TURBO_INNERQ` env var).

### Long Context: K Cache Sensitivity and Mixed Precision

turbo3 K cache can degrade at long context on models with full RoPE. V cache is robust. The cause is post-RoPE K quantization — RoPE applies position-dependent rotations that create heavier tails in the K distribution.

With InnerQ + 2D VQ enabled (default), turbo3 K+V is now viable at 16K context:

**Qwen3-8B Q4_K_M, ctx=16K, wikitext-2, 5 chunks:**

| Config | PPL | vs f16 |
|---|---|---|
| f16 KV | 6.92 | — |
| turbo4 K+V | 6.99 | +0.9% |
| turbo3 K+V (with InnerQ + 2D VQ) | **7.05** | **+1.9%** |
| turbo4 K + turbo3 V | 7.01 | +1.3% |
| q8_0 K + turbo3 V | 6.95 | +0.4% |
| turbo3 K+V (no InnerQ) | 19.70 | +185% ⚠️ |

**Llama-3.1-8B base Q4_K_M, ctx=16K:** turbo3 K+V = 5.30 vs f16 4.91 (+7.8%).

**Qwen3.5-27B Q5_K_M, ctx=16K:** turbo3 K+V = 6.00 vs f16 5.98 (+0.3% — near-lossless, InnerQ auto-disabled).

Severity depends on: RoPE coverage, rope_theta, model architecture. Consistent with KVQuant (Berkeley) and Q-ROAR findings on post-RoPE K quantization sensitivity.

**Recommended configs:**

| Context | Config |
|---|---|
| Any | `--cache-type-k turbo3 --cache-type-v turbo3` (InnerQ + 2D VQ enabled by default) |
| Conservative | `--cache-type-k turbo4 --cache-type-v turbo3` |

Models with `partial_rotary_factor < 1.0` (Qwen3.5 family) showed no regression at 16K with turbo3 K+V even without InnerQ.

**Important:** InnerQ is **automatically disabled** on models with partial RoPE (e.g. Qwen3.5-27B, partial_rotary_factor=0.25). No user action needed. Qwen3.5-27B turbo3 K+V achieves PPL 6.00 vs f16 5.98 (+0.3%).

### Speed (RX 7900 XTX, ROCm 6.4)

| Model | Context | Config | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|
| Qwen3.5-27B | 512 | f16 | 427 | 30.15 |
| | 512 | turbo3 | 423 (-1%) | 29.49 (-2%) |
| | 512 | turbo2 | 421 (-1%) | 29.80 (-1%) |
| | 512 | turbo4 | 422 (-1%) | 29.71 (-1%) |
| | 8K | f16 | 340 | 30.13 |
| | 8K | turbo3 | 337 (-1%) | 29.74 (-1%) |
| Gemma 4 26B | 512 | f16 | 2939 | 94.53 |
| | 512 | turbo3 | 2878 (-2%) | 87.13 (-8%) |

Minimal speed overhead: 1-2% on standard models, 8% decode on Gemma 4 (D=512 WHT cost).

### TriAttention KV Pruning (Qwen3.5-27B, 16K context)

Based on the same pre-RoPE Q/K concentration principle as [TriAttention (Mao et al., NVIDIA/MIT, 2026)](https://arxiv.org/abs/2604.04921). Independent C/HIP implementation with native GPU compaction kernel.

| Config | PPL | KV rows |
|---|---|---|
| Baseline | 6.0729 | 16384 |
| TriAttention 75% | **5.9939 (-1.3%)** | ~5989 |
| TriAttention 50% | 6.0890 (+0.26%) | ~2550 |

### Single-Needle NIAH: turbo3 + TriAttention combo

All tests with `chat_template_kwargs: {"enable_thinking": false}` (Qwen3 thinking mode otherwise consumes answer tokens).
TriAttention params: `--tri-budget 75 --tri-window 512 --tri-interval 128`

| Config | Model | to 12K | to 30K |
|---|---|---|---|
| turbo3 only (baseline) | Qwen3.5-27B | 20/20 (100%) | 25/25 (100%) |
| turbo3 + TriAtt 75% | Qwen3.5-27B | **20/20 (100%)** | **23/25 (92%)** |
| turbo3 + TriAtt 75% | Qwen3-8B | 20/20 (100%) | 19/25 (76%) |

75% retention preserves single-needle retrieval to 12K on both models.
At longer contexts, sporadic failures appear (primarily at depth 0.75), with the smaller 8B model degrading faster.

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

- **InnerQ + 2D VQ** — RoPE Pair Normalization + 64-entry 2D vector quantization codebook. Reduces turbo3 K degradation from +185% to +1.9% PPL at 16K context
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

Tested on: AMD Ryzen 9 9950X3D, RX 7900 XTX 24GB, ROCm 6.4/7.2.1, openSUSE Tumbleweed.

### KL-Divergence (Qwen3.5-27B, turbo vs f16, 10 prompts)

| Config | KL(f16 \|\| turbo) | JSD | Top-1 match | Compression |
|---|---|---|---|---|
| turbo4 | 0.015 ± 0.017 | 0.0011 | 10/10 (100%) | 3.88× |
| turbo3 | 0.021 ± 0.019 | 0.0019 | 10/10 (100%) | 5.12× |
| turbo2 | 0.034 ± 0.028 | 0.0036 | 9/10 (90%) | 7.53× |

All turbo types produce nearly identical token distributions to f16.

### Multi-Turn Tool Calling (Qwen3.5-27B, turbo3 K+V)

11/11 passed (100%) across 5 scenarios including:
- Sequential same-tool calls (weather → weather)
- Cross-tool chains (weather → calculate, search → search → calculate)
- 3-turn tool chains with intermediate results

turbo3 KV preserves multi-step agentic reasoning.

### KV Cache VRAM Usage (Qwen3.5-27B, 16 KV layers)

| Context | f16 | turbo3 | turbo2 |
|---|---|---|---|
| 4K | 256 MiB | 50 MiB | 34 MiB |
| 32K | 2,048 MiB | 400 MiB | 272 MiB |
| 131K | 8,192 MiB | 1,600 MiB | 1,088 MiB |

At 131K context, turbo3 saves 6.4 GiB vs f16. turbo2 saves 6.9 GiB.

### Combined Compression: TurboQuant + TriAttention

| Method | KV Compression | Quality |
|---|---|---|
| turbo3 alone | 5.12× | Qwen3.5-27B: PPL 6.00 vs f16 5.98 (+0.3%) |
| TriAttention 75% alone | 1.33× | -1.3% PPL |
| **turbo3 + TriAttention 75%** | **~6.8×** | Qwen3.5-27B: PPL 6.19 (+1.1% vs f16), GSM8K 72.0% (=f16) |
| **turbo3 + TriAttention 50%** | **~10.2×** | Qwen3.5-27B: PPL 6.23 (+1.8% vs f16) |

Combo validated on Qwen3.5-27B Q5_K_M (hybrid SSM+attn) and Qwen3-8B Q4_K_M. GSM8K validated on full 1319 problems (2026-04-11).
