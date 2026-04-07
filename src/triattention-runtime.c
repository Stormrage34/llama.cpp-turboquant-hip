/*
 * triattention-runtime.c — TriAttention runtime scoring
 */

#define _GNU_SOURCE
#include "triattention-runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tria_runtime * tria_runtime_init(
    struct tria_stats * stats,
    int budget_pct,
    int window,
    int interval
) {
    if (!stats || budget_pct <= 0) return NULL;

    struct tria_runtime * rt = calloc(1, sizeof(*rt));
    rt->stats      = stats;
    rt->budget_pct = budget_pct;
    rt->window     = window;
    rt->interval   = interval;
    rt->n_scored   = 0;

    int n_pairs = stats->num_layers * stats->num_kv_heads;
    rt->retained       = calloc(n_pairs, sizeof(int *));
    rt->retained_count = calloc(n_pairs, sizeof(int));

    return rt;
}

void tria_runtime_free(struct tria_runtime * rt) {
    if (!rt) return;
    if (rt->retained) {
        int n_pairs = rt->stats->num_layers * rt->stats->num_kv_heads;
        for (int i = 0; i < n_pairs; i++) {
            free(rt->retained[i]);
        }
        free(rt->retained);
    }
    free(rt->retained_count);
    free(rt);
}

int tria_maybe_score(
    struct tria_runtime * rt,
    int n_kv
) {
    if (!rt || !rt->stats) return 0;

    /* Check if we should score */
    if (n_kv - rt->n_scored < rt->interval) return 0;
    if (n_kv <= rt->window) return 0;

    int nl  = rt->stats->num_layers;
    int nkv = rt->stats->num_kv_heads;
    int fc  = rt->stats->freq_count;

    /* How many old tokens to consider (exclude recent window) */
    int n_old = n_kv - rt->window;
    if (n_old <= 0) return 0;

    int budget = (n_old * rt->budget_pct) / 100;
    if (budget <= 0) budget = 1;

    fprintf(stderr, "tria_score: n_kv=%d n_old=%d budget=%d (trigger at interval=%d)\n",
            n_kv, n_old, budget, rt->interval);

    /*
     * TODO: actual scoring requires reading K cache from GPU.
     * For now, just log that scoring would happen.
     * Full implementation needs:
     *   1. ggml_backend_tensor_get() to copy K to CPU
     *   2. Dequantize if quantized
     *   3. Inverse RoPE to get pre-RoPE K
     *   4. Split into real/imag halves
     *   5. Call tria_score_kv_head() per layer × kv_head
     *   6. Select top-B indices
     *   7. Store in rt->retained[]
     */

    rt->n_scored = n_kv;
    return 0; /* no actual pruning yet */
}
