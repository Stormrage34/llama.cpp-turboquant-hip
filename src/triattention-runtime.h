/*
 * triattention-runtime.h — TriAttention runtime scoring for llama.cpp
 *
 * Call tria_maybe_prune() after each decode step.
 * It checks if scoring interval is reached and runs pruning if needed.
 */

#ifndef TRIATTENTION_RUNTIME_H
#define TRIATTENTION_RUNTIME_H

#include "triattention.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime state per context */
struct tria_runtime {
    struct tria_stats * stats;
    int     budget_pct;     /* retention % */
    int     window;         /* recent tokens always kept */
    int     interval;       /* score every N tokens */
    int     n_scored;       /* last position we scored at */

    /* Per layer × kv_head: retained index masks */
    /* NULL until first scoring pass */
    int   **retained;       /* [num_layers * num_kv_heads][budget] */
    int    *retained_count; /* [num_layers * num_kv_heads] */

    /* Global aggregated scores per token position (continuous) */
    float  *global_scores;  /* [n_scored], allocated on first score */
    int     global_n;       /* length of global_scores */
    int     global_budget;  /* how many tokens to keep globally */
    int     compaction_active; /* 1 after physical compaction disables mask injection */
    int     score_pass;     /* counts scoring passes, full rescore every TRIA_FULL_RESCORE_INTERVAL */
};

/*
 * Global runtime pointer for mask injection.
 * Set by the application after tria_runtime_init.
 * Read by graph building code to apply eviction mask.
 */
extern struct tria_runtime * g_tria_rt;

/* Bridge helpers implemented in triattention-bridge.cpp. */
struct ggml_tensor * tria_get_k_tensor(void * ctx, int layer_idx);
int tria_get_n_kv(void * ctx);
int tria_get_used_n_kv(void * ctx);
int tria_get_kv_positions(void * ctx, int * positions, int max_positions);

/* Create runtime. Returns NULL if stats is NULL or budget_pct == 0. */
struct tria_runtime * tria_runtime_init(
    struct tria_stats * stats,
    int budget_pct,
    int window,
    int interval
);

void tria_runtime_free(struct tria_runtime * rt);

/*
 * Check if we should score, and if so, do it.
 * ctx — llama context (for KV cache access)
 * n_kv is read from the KV cache directly.
 */
int tria_maybe_score(
    struct tria_runtime * rt,
    void * ctx  /* llama_context*, passed as void* to avoid C++ header dep */
);

/*
 * Build a global eviction bitmask (union of all layer×head retained sets).
 * evict_mask[i] = 1 means position i is evicted (should be -inf in attn mask).
 * Positions >= n_scored or in the recent window are never evicted.
 * Returns 0 if no scoring has been done yet, 1 if mask was written.
 * Caller must allocate evict_mask with at least n_kv entries.
 */
int tria_get_evict_mask(
    const struct tria_runtime * rt,
    int n_kv,
    int8_t * evict_mask  /* out: [n_kv], 1=evicted 0=kept */
);

/* Physically compact the KV cache using the latest global_scores/global_budget. */
int tria_compact_kv(
    struct tria_runtime * rt,
    void * ctx  /* llama_context* */
);

#ifdef __cplusplus
}
#endif

#endif /* TRIATTENTION_RUNTIME_H */
