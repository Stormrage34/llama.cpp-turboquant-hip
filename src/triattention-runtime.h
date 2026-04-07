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
};

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
 *
 * k_data[il]     — raw K cache data pointer per layer (CPU, dequantized, f32)
 * n_kv           — number of KV entries currently in cache
 * n_embd_k_gqa   — K embedding size (n_kv_heads * head_dim)
 *
 * Returns: number of tokens pruned (0 if no scoring happened)
 */
int tria_maybe_score(
    struct tria_runtime * rt,
    int n_kv
);

#ifdef __cplusplus
}
#endif

#endif /* TRIATTENTION_RUNTIME_H */
