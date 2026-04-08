# Codex Task: TriAttention Physical KV Cache Compaction

## Context

Branch `feature/triattention-scoring` on this repo implements TriAttention KV cache pruning via attention masking. The scoring pipeline works end-to-end and produces correct PPL results:

| Retention | PPL | Δ vs baseline |
|-----------|-----|---------------|
| 100% | 8.1524 | — |
| 75% | 8.2129 | +0.7% |
| 50% | 8.4907 | +4.1% |
| 25% | 8.6415 | +6.0% |
| 10% | 8.7901 | +7.8% |

However, masking alone does NOT save VRAM — Flash Attention still loads all K/V blocks. We need **physical compaction**: after scoring, physically remove evicted tokens from the KV cache so FA operates on a shorter, contiguous cache.

## What exists

- `src/triattention-runtime.h` / `.c` — runtime with `tria_maybe_score()` that scores all tokens and produces `global_scores[n_old]` (z-normalized, max-aggregated across all layer×head pairs) and `global_budget` (how many to keep)
- `src/triattention-bridge.cpp` — bridge to access KV cache from C code
- `src/llama-kv-cache.cpp` — mask injection in `set_input_kq_mask()` after standard causal mask fill
- Scoring hook is in `llama_decode()` in `src/llama-context.cpp`

## The task

Implement `tria_compact_kv()` that physically compacts the KV cache after scoring. Called from `tria_maybe_score()` after global scores are computed.

### Requirements

1. From `global_scores`, determine which token positions to keep (top global_budget scores)
2. For each layer, physically copy retained K and V entries to contiguous positions at the start of the cache
3. Update KV cache metadata (cell positions, sequence info) to reflect the compacted state
4. Update `n_kv` / cache head so subsequent attention operates on the shorter cache
5. Must work with f16 K/V cache on HIP/ROCm (use `ggml_backend_tensor_get` / `ggml_backend_tensor_set` for GPU memory access)
6. After compaction, the attention mask injection in `set_input_kq_mask` should be disabled (no longer needed — evicted tokens are physically gone)

### Key files to study

- `src/llama-kv-cache.h` — `llama_kv_cache` class, `get_k_raw()`, cell management
- `src/llama-kv-cells.h` — `llama_kv_cells` with `move()`, `pos` tracking, `seq_has()`
- `src/llama-kv-cache.cpp` — `seq_rm()` for how metadata is updated on removal
- `src/triattention-runtime.c` — `tria_maybe_score()` where compaction should be called
- `src/triattention-bridge.cpp` — existing bridge functions, add new ones here

### Approach

1. Add `tria_compact_kv(tria_runtime *rt, void *ctx)` to bridge
2. Inside: build sorted retained-index list from global_scores
3. For each layer, use `ggml_backend_tensor_get/set` to copy retained K/V rows to positions 0..budget-1
4. Update cell metadata: clear evicted cells, update positions of retained cells
5. Set a flag on runtime so mask injection is skipped (compaction replaces masking)
6. Reset `n_scored` after compaction

### Constraints

- Do NOT modify ggml kernel code
- Do NOT change the Flash Attention path
- Use existing `ggml_backend_tensor_get` / `ggml_backend_tensor_set` for GPU memory moves
- Keep changes minimal — this is a research prototype, not production code
- The KV cache layout is: K tensor per layer has shape [n_embd_k_gqa, n_ctx], stored row-major, each row is one token's K for all KV heads

### Testing

After implementation, this should work:
```bash
HIP_VISIBLE_DEVICES=0 ./bin/llama-perplexity \
  -m ~/models/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf \
  -f ~/llama.cpp/wikitext-2-raw/wiki.test.raw \
  -c 4096 -ngl 99 --chunks 5 \
  --triattention ~/triattention-stats/qwen3_8b_v2.bin \
  --tri-budget 50 --tri-window 128 --tri-interval 256
```

Expected: PPL similar to masking results (~8.49 at 50%), but with actual VRAM reduction visible in memory breakdown.
