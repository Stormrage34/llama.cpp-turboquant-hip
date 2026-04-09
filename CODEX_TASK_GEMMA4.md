# Codex Task: Enable TurboQuant turbo3 KV cache for Gemma 4 (head_dim=512/256)

## Context

This is a llama.cpp fork with TurboQuant KV cache compression (WHT-based 3-bit quantization).
Branch: `feature/triattention-scoring`
The WHT (Walsh-Hadamard Transform) kernel templates already support GROUP_SIZE=256 
(commit 7d310a7d9), but the dispatch logic still hardcodes GROUP_SIZE=128.

## Problem

Gemma 4 has a hybrid ISWA (interleaved sliding window attention) architecture:
- **Global attention layers** (5 layers): head_dim_k=512, head_dim_v=512
- **SWA layers** (25 layers): head_dim_k=256, head_dim_v=256
- Variable GQA: n_head_kv is 8 or 2 depending on layer
- Two separate KV caches (ISWA): one for global, one for SWA

Current TurboQuant uses GROUP_SIZE=128 for WHT rotation. For head_dim=512, this means
4 groups of 128 per head. We want to use GROUP_SIZE=256 (2 groups of 256 per head) for
better quality, matching the approach used for V cache in AmesianX's implementation.

## What needs to change

### 1. `ggml/src/ggml.c` (line ~6264)
Add 256 to the group_size assert in `ggml_turbo_wht`:
```c
// Current:
GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
// Change to:
GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256);
```

### 2. `src/llama-kv-cache.cpp` — two places where `wht_group = 128` is hardcoded

In `cpy_k()` (~line 1554) and `cpy_v()` (~line 1605), change:
```c
int32_t wht_group = 128;
```
to:
```c
// Use 256 for head_dim>=512 (Gemma 4 global attention), else 128 or 64
int32_t wht_group = (n_embd_head >= 512 && n_embd_gqa % 256 == 0) ? 256 :
                    (n_embd_gqa % 128 == 0) ? 128 : 64;
```
`n_embd_head` is already available in both functions (it's the per-layer head dimension).

### 3. `src/llama-graph.cpp` — two places where `turbo_group` is computed (~lines 1910, 1988)

Change:
```c
const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
```
to:
```c
const uint32_t hd_k = hparams.n_embd_head_k(il);
const int turbo_group = (hd_k >= 512 && group_src->ne[0] % 256 == 0) ? 256 :
                        (group_src->ne[0] % 128 == 0) ? 128 : 64;
```
`il` (layer index) and `hparams` are both in scope in `build_attn_mha()`.

## Critical constraints

- **Do NOT change GROUP_SIZE for head_dim=256 models** (e.g. Qwen3.5-27B). They must 
  continue using GROUP_SIZE=128. Only head_dim>=512 should use 256.
- The encode (set_rows) and decode (turbo_wht inverse) must use the SAME group_size.
- Keep the existing `turbo_attn_sharpening()` function intact.
- The sign arrays `TURBO_WHT_SIGNS1_256` and `TURBO_WHT_SIGNS2_256` are already defined
  in `turbo-quant.cuh`.

## Testing

### Regression test (must pass):
```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity \
  -m ~/models/Qwen3.5-27B/Qwen_Qwen3.5-27B-Q5_K_M.gguf \
  -f wikitext-2-raw/wiki.test.raw \
  -c 4096 --chunks 3 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3
# Expected: PPL ≈ 6.57 (same as before)
```

### Gemma 4 test:
```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity \
  -m ~/models/Gemma-4-26B-A4B/google_gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -f wikitext-2-raw/wiki.test.raw \
  -c 2048 --chunks 5 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3
# Compare with f16 baseline (PPL ≈ 35131 — high because instruct model on raw text)
# turbo3 should be within 2x of f16 baseline
```

## Files to modify
1. `ggml/src/ggml.c` — one assert
2. `src/llama-kv-cache.cpp` — two wht_group assignments  
3. `src/llama-graph.cpp` — two turbo_group computations
