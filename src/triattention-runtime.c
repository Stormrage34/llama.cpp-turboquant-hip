/*
 * triattention-runtime.c — TriAttention runtime scoring
 */

#define _GNU_SOURCE
#include "triattention-runtime.h"
#include "triattention.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tria_runtime * g_tria_rt = NULL;

/* Access KV cache internals — defined in llama-kv-cache.cpp */
/* We use the raw tensor pointers from llama_memory/kv_cache */
struct llama_kv_layer {
    struct ggml_tensor * k;
    struct ggml_tensor * v;
};

/* Forward declaration — we'll get K tensor via a helper */
extern struct ggml_tensor * tria_get_k_tensor(void * ctx, int layer_idx);
extern int tria_get_n_kv(void * ctx);
extern int tria_get_used_n_kv(void * ctx);
extern int tria_get_kv_positions(void * ctx, int * positions, int max_positions);
extern int tria_compact_kv(struct tria_runtime * rt, void * ctx);

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
    free(rt->global_scores);
    free(rt);
}

int tria_maybe_score(
    struct tria_runtime * rt,
    void * ctx
) {
    if (!rt || !rt->stats || !ctx) return 0;

    int n_kv = tria_get_n_kv(ctx);
    int n_used = tria_get_used_n_kv(ctx);
    if (n_kv <= 0 || n_used <= 0) return 0;

    /* Reset if cache was cleared (perplexity resets between chunks) */
    if (n_kv < rt->n_scored) {
        rt->n_scored = 0;
        rt->compaction_active = 0;
    }

    /* Check if we should score */
    if (n_kv - rt->n_scored < rt->interval) return 0;
    if (n_used <= rt->window) return 0;

    int nl  = rt->stats->num_layers;
    int nkv = rt->stats->num_kv_heads;
    int fc  = rt->stats->freq_count;
    int hd  = rt->stats->head_dim;

    int n_old = n_used - rt->window;
    if (n_old <= 0) return 0;

    int budget = (n_old * rt->budget_pct) / 100;
    if (budget <= 0) budget = 1;

    fprintf(stderr, "tria_score: n_kv=%d n_old=%d budget=%d (trigger at interval=%d)\n",
            n_kv, n_old, budget, rt->interval);

    /* Try to read K from cache and score */
    if (!ctx) { rt->n_scored = n_kv; return 0; }

    /* Allocate buffers for one layer's K data */
    int n_embd_k_gqa = nkv * hd;
    size_t k_bytes = (size_t)n_old * n_embd_k_gqa * sizeof(float);
    float * k_f32 = (float *)malloc(k_bytes);
    float * scores = (float *)malloc(n_old * sizeof(float));
    int * key_pos = (int *)malloc(n_old * sizeof(int));

    /* Global score accumulator: max of z-normalized scores across all heads */
    if (rt->global_n < n_old) {
        free(rt->global_scores);
        rt->global_scores = (float *)malloc(n_old * sizeof(float));
        rt->global_n = n_old;
    }
    for (int i = 0; i < n_old; i++) rt->global_scores[i] = -1e30f;
    rt->global_budget = budget;
    rt->compaction_active = 0;

    if (!k_f32 || !scores || !key_pos || !rt->global_scores) {
        free(k_f32); free(scores); free(key_pos);
        rt->n_scored = n_kv;
        return 0;
    }

    if (tria_get_kv_positions(ctx, key_pos, n_old) != n_old) {
        free(k_f32); free(scores); free(key_pos);
        rt->n_scored = n_kv;
        return 0;
    }

    int total_pruned = 0;

    for (int li = 0; li < nl; li++) {
        struct ggml_tensor * k_tensor = tria_get_k_tensor(ctx, li);
        if (!k_tensor) continue;

        /* Read K data from GPU/backend to CPU */
        size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_gqa);
        size_t read_bytes = row_size * n_old;

        if (k_tensor->type == GGML_TYPE_F16) {
            /* Dequant f16 → f32 */
            uint16_t * k_f16 = (uint16_t *)malloc(read_bytes);
            if (!k_f16) continue;
            ggml_backend_tensor_get(k_tensor, k_f16, 0, read_bytes);
            for (int i = 0; i < n_old * n_embd_k_gqa; i++) {
                k_f32[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)k_f16)[i]);
            }
            free(k_f16);
        } else if (k_tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(k_tensor, k_f32, 0, n_old * n_embd_k_gqa * sizeof(float));
        } else {
            /* Quantized cache (turbo3 etc) — skip for now */
            continue;
        }

        /* Score each KV head */
        for (int kvi = 0; kvi < nkv; kvi++) {
            /* Extract this KV head's data: k_f32 is [n_old, n_embd_k_gqa] */
            /* KV head kvi occupies columns [kvi*hd .. (kvi+1)*hd) */
            float * k_real = (float *)malloc(n_old * fc * sizeof(float));
            float * k_imag = (float *)malloc(n_old * fc * sizeof(float));
            if (!k_real || !k_imag) { free(k_real); free(k_imag); continue; }

            for (int s = 0; s < n_old; s++) {
                float * row = k_f32 + s * n_embd_k_gqa + kvi * hd;
                /* Split head_dim into real (first half) and imag (second half) */
                for (int f = 0; f < fc; f++) {
                    k_real[s * fc + f] = row[f];
                    k_imag[s * fc + f] = row[fc + f];
                }
            }

            tria_score_kv_head(rt->stats, k_real, k_imag, key_pos,
                               n_kv, n_old, li, kvi, scores);

            /* Z-normalize scores for this head, then max-aggregate into global */
            float mean = 0, var = 0;
            for (int s = 0; s < n_old; s++) mean += scores[s];
            mean /= n_old;
            for (int s = 0; s < n_old; s++) {
                float d = scores[s] - mean;
                var += d * d;
            }
            float std = sqrtf(var / n_old + 1e-8f);
            for (int s = 0; s < n_old; s++) {
                float z = (scores[s] - mean) / std;
                if (z > rt->global_scores[s]) {
                    rt->global_scores[s] = z;
                }
            }

            total_pruned += n_old - tria_layer_budget(rt->stats, li, budget);
            free(k_real);
            free(k_imag);
        }
    }

    if (total_pruned > 0) {
        fprintf(stderr, "tria_score: pruned %d tokens across %d×%d heads\n",
                total_pruned, nl, nkv);
    }

    free(k_f32);
    free(scores);
    free(key_pos);

    {
        const int compacted = tria_compact_kv(rt, ctx);
        if (compacted > 0) {
            rt->compaction_active = 1;
            rt->n_scored = tria_get_n_kv(ctx);
            return compacted;
        }
    }

    rt->n_scored = n_kv;
    return total_pruned;
}

int tria_get_evict_mask(
    const struct tria_runtime * rt,
    int n_kv,
    int8_t * evict_mask
) {
    if (!rt || rt->n_scored == 0 || !evict_mask) return 0;
    if (rt->compaction_active) return 0;
    if (!rt->global_scores || rt->global_budget <= 0) return 0;

    int n_old = rt->n_scored - rt->window;
    if (n_old <= 0) return 0;

    int budget = rt->global_budget;
    if (budget >= n_old) {
        memset(evict_mask, 0, n_kv);
        return 1;
    }

    /* Find the budget-th largest score as threshold (partial sort via nth_element idea) */
    /* Simple approach: find threshold by sorting scores */
    static float * sorted = NULL;
    static int sorted_cap = 0;
    if (n_old > sorted_cap) {
        free(sorted);
        sorted = (float *)malloc(n_old * sizeof(float));
        sorted_cap = n_old;
    }
    memcpy(sorted, rt->global_scores, n_old * sizeof(float));

    /* Partial sort: find the budget-th largest value */
    /* Use simple selection: find threshold where exactly budget items are >= it */
    /* Quick approach: sort descending, threshold = sorted[budget-1] */
    for (int i = 0; i < budget; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n_old; j++) {
            if (sorted[j] > sorted[max_idx]) max_idx = j;
        }
        float tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
    }
    float threshold = sorted[budget - 1];

    /* Build mask: evict if below threshold */
    memset(evict_mask, 0, n_kv);
    int kept = 0;
    for (int i = 0; i < n_old && i < n_kv; i++) {
        if (rt->global_scores[i] >= threshold && kept < budget) {
            kept++;
        } else {
            evict_mask[i] = 1;
        }
    }

    return 1;
}
