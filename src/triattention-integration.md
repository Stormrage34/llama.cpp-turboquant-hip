/*
 * triattention-integration.md — Notes for llama.cpp integration
 *
 * Hook point: src/llama-graph.cpp, build_attn() variants
 *
 * k_cur arrives pre-RoPE. The hook goes right before:
 *   ggml_build_forward_expand(gf, k_cur);
 *
 * Integration plan (MVP):
 *
 * 1. At model load (llama_model_load):
 *    - If --triattention <path> is set, call tria_load(path)
 *    - Store tria_stats* in llama_model or llama_context
 *
 * 2. At graph build (build_attn, every layer):
 *    - k_cur is [head_dim, n_kv_heads, n_tokens] pre-RoPE
 *    - After k_proj but before RoPE/cache copy
 *    - If tria_stats loaded AND (n_past % 128 == 0) AND (n_past > budget):
 *      - Use ggml_map_custom1 to score cached keys
 *      - Or: defer scoring to a separate pass after forward
 *
 * 3. Scoring pass (separate from graph build):
 *    - After forward pass completes, before next token:
 *    - Read pre-RoPE K from aux buffer (stored during forward)
 *    - Call tria_score_kv_head() for each layer × kv_head
 *    - Compute top-B indices per kv_head
 *    - Update retained-index mask in KV cache metadata
 *
 * 4. Attention dispatch:
 *    - If retained-index mask exists for this layer:
 *      - Gather K/V by retained indices before attention
 *      - Or: apply -inf mask to evicted positions
 *    - Else: normal attention path
 *
 * Key files to modify:
 *   src/llama-graph.cpp    — hook k_cur pre-RoPE
 *   src/llama-kv-cache.cpp — retained-index metadata
 *   src/llama-context.cpp  — tria_stats lifecycle
 *   common/arg.cpp         — CLI flags
 *   src/CMakeLists.txt     — add triattention.c
 *
 * Decision: scoring happens on CPU (our C library), not in ggml graph.
 * Reason: scoring is infrequent (every 128 tokens) and complex
 * (GQA aggregation, z-norm, top-K). Not worth building as ggml ops.
 * ggml_map_custom1 is the bridge if we want GPU scoring later.
 */
