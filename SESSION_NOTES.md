# TurboQuant + TriAttention — Session Notes (2026-04-09/10)

## GSM8K Benchmark Results (100 problems, temperature=0)

| Model | f16 | turbo3 | Drop | Compression |
|---|---|---|---|---|
| Gemma 4 31B Dense Q4_K_M | 96% | **97%** | +1% | 2.9× |
| Gemma 4 26B-A4B Q4_K_M | 83% | **83%** | 0% | 2.9× |
| Qwen3.5-27B Q5_K_M | 66% | **72%** | +6% | 5× |

Best Gemma 4 config: `--cache-type-k turbo3 --cache-type-v turbo3 --cache-type-k-swa turbo3 --cache-type-v-swa q8_0`

## Critical Discovery: Gemma 4 D=512 FA Path

Gemma 4 global attention layers (head_dim=512) with turbo3 KV go to **TILE FA kernel**
on AMD/HIP (not VEC, not MMA). TILE reads turbo3 data as raw f16 bytes — no dequant.

| FA kernel | How it reads turbo3 | GSM8K | PPL (4K) |
|---|---|---|---|
| TILE (current, reads garbage) | Raw bytes as f16 | **83%** | 32891 |
| VEC (proper turbo3 dequant) | centroid × norm | **68%** | 32141 |
| f16 baseline | N/A | **83%** | 40155 |

**Paradox:** Proper dequant gives WORSE accuracy than garbage. PPL is better with VEC
but GSM8K accuracy drops 15 points.

**Root cause:** 4 independent 128-element WHT groups on 512-dim head lose cross-group
correlations. Proper dequant reconstructs a structurally degraded signal. TILE's garbage
effectively drops the 5 global layers (residual connection passes through), and the 25
SWA layers (f16/turbo3) carry the output.

## Gemma 4 Config Ablation (GSM8K, 100 problems)

| Global layers | SWA layers | GSM8K |
|---|---|---|
| f16 | f16 | 83% (baseline) |
| turbo3 (TILE=garbage) | turbo3-K + q8_0-V | **83%** |
| turbo3 (TILE=garbage) | f16 | 82% |
| f16 | turbo3-K + q8_0-V | **77%** |
| turbo3 (VEC=proper) | turbo3-K + q8_0-V | **68%** |

Key insight: turbo3 SWA with q8_0-V needs attention sharpening on global layers too
(even if global turbo3 is garbage) to maintain 83%. Without it (f16 global), drops to 77%.

## Gemma 4 Architecture Facts (verified)

- `attn_soft_cap = false` — NO attention logit softcapping (GPT was right)
- `f_attn_scale = 1.0` — no 1/sqrt(d) scaling
- `f_final_logit_softcapping = 30.0` — only on final logits
- `n_rot = 512` — FULL RoPE on all 512 dims (Gemini was wrong about Partial RoPE)
- Global layers: 5/30 (il=5,11,17,23,29), head_dim_k=512, head_dim_v=512, n_head_kv=2
- SWA layers: 25/30, head_dim=256, n_head_kv=8, sliding_window=1024

## FA Kernel Selection on AMD (gfx1100, RX 7900 XTX)

- `turing_mma_available` = false (NVIDIA only)
- `volta_mma_available` = false (NVIDIA only)
- WMMA excludes D=512 (`Q->ne[0] != 512`)
- Falls through to **TILE** kernel (no turbo3 dequant support)
- VEC kernel requires `Q->ne[0] <= 256` (turbo types excluded from D=512)

**Fix applied (not committed):** Added D=512 FA VEC instantiations for turbo types +
forced VEC for turbo+D>256. But proper dequant gives worse results (see above).

## TriAttention Combo Results

| Config (Qwen3.5-27B) | GSM8K | PPL (16K) |
|---|---|---|
| turbo3 only | 72% | 6.4521 |
| turbo3 + TriAttention 75% | 72% | 6.6361 |
| turbo3 + TriAttention 50% (aggressive) | 72% | — |

TriAttention doesn't activate at short context (GSM8K ~600 tokens).
At 16K, combo PPL is worse than turbo3 alone (noise stacking).

## Alpha Norm Scaling Test (from TheTom's research)

Applied α=1.02 to turbo3 norm at FA dequant time (fattn-common.cuh).
- Qwen3.5-27B PPL: 6.5743 (unchanged — sharpening already compensates)
- Gemma 4 GSM8K: 78% (worse — alpha hurts SWA layers)
- **Conclusion:** Our attention sharpening already does what alpha scaling does.

## LLM Consultation Summary (DeepSeek, Gemini, GPT, Perplexity)

### Consensus
- **H_4 ⊗ H_128 Kronecker WHT** is the best path for D=512
- Stride-4 permutation before grouping is a simpler alternative
- PPL vs accuracy divergence is a known phenomenon in compression

### Key insights per LLM
- **DeepSeek:** Block-diagonal WHT loses cross-group correlations. Kronecker is the fix.
  Algorithm: (1) 128-WHT per group, (2) 4-point WHT across groups per position, (3) signs+normalize.
- **Gemini:** Claimed Partial RoPE concentrates positional features in Group 0 — WRONG
  (verified n_rot=512 = full RoPE). But stride-4 shuffle idea is still valid.
- **GPT:** Verified `attn_logit_softcapping = null` for Gemma 4 — CORRECT.
  Suggested per-layer sensitivity analysis (quantize one global layer at a time).
- **Perplexity:** Suggested overlapping groups and adaptive sharpening with softcap factor.

## Next Steps (Priority Order)

1. **Kronecker H_4 ⊗ H_128 kernel** — requires new kernel architecture where one block
   sees all 4 groups (currently one block = one group). Two approaches:
   a) Separate cross-group mixing kernel before/after per-group WHT
   b) Unified 512-thread kernel that does full Kronecker in one pass

2. **Per-layer sensitivity analysis** — quantize one global layer at a time to find
   which of the 5 layers causes most GSM8K degradation

3. **G=256 debug** — still broken, separate from D=512 issue. Race condition or
   butterfly stage bug in 256-thread kernel on HIP.

## Repository State

- **llama.cpp**: branch `feature/triattention-scoring`, commit `78e7e9cd0`
  - All changes committed and pushed to `domvox` remote
  - Working tree clean (FA Vec D=512 changes were reverted)
  - llama-server.service running (Qwen3.5-27B on port 8080)

- **triattention-ggml**: branch `master`, commit `be2f14d`
  - REVIEW_NOTES.md updated with all findings
  - Pushed to origin

## Key Files for Kronecker Implementation

- `ggml/src/ggml-cuda/set-rows.cu` — K/V encode (needs cross-group mixing)
- `ggml/src/ggml-cuda/turbo-wht.cu` — Q preprocessing + V inverse WHT
- `ggml/src/ggml-cuda/fattn-common.cuh` — FA dequant (vec_dot_fattn_vec_KQ_turbo3_0)
- `ggml/src/ggml-cuda/fattn.cu` — FA kernel selection (D=512 routing)
- `ggml/src/ggml-cuda/fattn-vec.cuh` — VEC kernel template instantiations
- `src/llama-kv-cache.cpp` — KV cache encode dispatch (wht_group in op_params)
- `src/llama-graph.cpp` — V inverse WHT dispatch, attention sharpening
- `ggml/src/ggml.c:6261` — Q auto-detect group_size
