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
    void * ctx
) {
    if (!rt || !rt->stats || !ctx) return 0;

    int n_kv = tria_get_n_kv(ctx);
    if (n_kv <= 0) return 0;

    /* Debug: log every 64 tokens */
    if (n_kv > 0 && (n_kv % 64 == 0)) {
        fprintf(stderr, "tria_check: n_kv=%d n_scored=%d interval=%d window=%d\n",
                n_kv, rt->n_scored, rt->interval, rt->window);
    }

    /* Check if we should score */
    if (n_kv - rt->n_scored < rt->interval) return 0;
    if (n_kv <= rt->window) return 0;

    int nl  = rt->stats->num_layers;
    int nkv = rt->stats->num_kv_heads;
    int fc  = rt->stats->freq_count;
    int hd  = rt->stats->head_dim;

    int n_old = n_kv - rt->window;
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

    if (!k_f32 || !scores || !key_pos) {
        free(k_f32); free(scores); free(key_pos);
        rt->n_scored = n_kv;
        return 0;
    }

    for (int i = 0; i < n_old; i++) key_pos[i] = i;

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
        int layer_budget = tria_layer_budget(rt->stats, li, budget);

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

            /* Store top-B retained indices */
            int pair_idx = li * nkv + kvi;
            free(rt->retained[pair_idx]);
            rt->retained[pair_idx] = (int *)malloc(layer_budget * sizeof(int));
            rt->retained_count[pair_idx] = layer_budget;

            /* Simple top-K: find largest scores */
            for (int b = 0; b < layer_budget; b++) {
                float best = -1e30f;
                int best_idx = 0;
                for (int s = 0; s < n_old; s++) {
                    if (scores[s] > best) {
                        int dup = 0;
                        for (int p = 0; p < b; p++) {
                            if (rt->retained[pair_idx][p] == s) { dup = 1; break; }
                        }
                        if (!dup) { best = scores[s]; best_idx = s; }
                    }
                }
                rt->retained[pair_idx][b] = best_idx;
            }

            total_pruned += n_old - layer_budget;
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

    rt->n_scored = n_kv;
    return total_pruned;
}

int tria_get_evict_mask(
    const struct tria_runtime * rt,
    int n_kv,
    int8_t * evict_mask
) {
    if (!rt || rt->n_scored == 0 || !evict_mask) return 0;

    int nl  = rt->stats->num_layers;
    int nkv = rt->stats->num_kv_heads;
    int n_old = rt->n_scored - rt->window;
    if (n_old <= 0) return 0;

    /* Start: evict everything in the scored region */
    memset(evict_mask, 1, n_kv);

    /* Recent window + anything beyond scored: never evict */
    for (int i = n_old; i < n_kv; i++) {
        evict_mask[i] = 0;
    }

    /* Union: if ANY layer×head retains a position, keep it */
    for (int pair = 0; pair < nl * nkv; pair++) {
        int cnt = rt->retained_count[pair];
        int *idx = rt->retained[pair];
        if (!idx) continue;
        for (int j = 0; j < cnt; j++) {
            if (idx[j] >= 0 && idx[j] < n_old) {
                evict_mask[idx[j]] = 0;
            }
        }
    }

    return 1;
}
