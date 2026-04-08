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
        rt->score_pass = 0;
        rt->global_n = 0;
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

    /* Incremental scoring: only read/score new tokens unless full rescore needed */
    /* Incremental scoring disabled pending fix for score reordering after compaction
       and z-norm drift between incremental/full passes (Codex review P1a/P1b).
       Force full rescore every pass for now. */
    #define TRIA_FULL_RESCORE_INTERVAL 1
    int full_rescore = (rt->score_pass % TRIA_FULL_RESCORE_INTERVAL == 0)
                     || !rt->global_scores
                     || rt->global_n < 1;

    /* How many rows are "new" since last compaction? */
    int n_prev = 0; /* retained from previous pass */
    int n_new  = n_old; /* tokens to score */
    int score_start = 0; /* first row index to read from GPU */

    if (!full_rescore && rt->global_scores && rt->global_n > 0) {
        /* After compaction, retained tokens are at rows 0..global_n-1 with valid scores */
        n_prev = rt->global_n;
        if (n_prev > n_old) n_prev = n_old;
        score_start = n_prev;
        n_new = n_old - n_prev;
        if (n_new <= 0) {
            /* Nothing new to score, just update budget */
            rt->global_budget = budget;
            rt->n_scored = n_kv;
            rt->score_pass++;
            return 0;
        }
    }

    fprintf(stderr, "tria_score: n_kv=%d n_old=%d budget=%d new=%d mode=%s (pass %d)\n",
            n_kv, n_old, budget, n_new,
            full_rescore ? "full" : "incremental", rt->score_pass);

    /* Try to read K from cache and score */
    if (!ctx) { rt->n_scored = n_kv; return 0; }

    /* Allocate buffers */
    int n_embd_k_gqa = nkv * hd;
    size_t k_bytes = (size_t)n_new * n_embd_k_gqa * sizeof(float);
    float * k_f32 = (float *)malloc(k_bytes);
    float * scores = (float *)malloc(n_new * sizeof(float));
    int * key_pos = (int *)malloc(n_old * sizeof(int));

    /* Resize global scores if needed */
    if (rt->global_n < n_old) {
        float * new_gs = (float *)malloc(n_old * sizeof(float));
        if (new_gs) {
            /* Copy old scores for retained positions */
            if (!full_rescore && rt->global_scores && rt->global_n > 0) {
                int copy_n = rt->global_n < n_old ? rt->global_n : n_old;
                memcpy(new_gs, rt->global_scores, copy_n * sizeof(float));
                for (int i = copy_n; i < n_old; i++) new_gs[i] = -1e30f;
            } else {
                for (int i = 0; i < n_old; i++) new_gs[i] = -1e30f;
            }
            free(rt->global_scores);
            rt->global_scores = new_gs;
            rt->global_n = n_old;
        } else {
            /* Resize failed — bail out to avoid buffer overrun */
            free(k_f32); free(scores); free(key_pos);
            rt->n_scored = n_kv;
            return 0;
        }
    } else if (full_rescore) {
        for (int i = 0; i < n_old; i++) rt->global_scores[i] = -1e30f;
    } else {
        /* Extend with -inf for new positions */
        for (int i = n_prev; i < n_old; i++) rt->global_scores[i] = -1e30f;
    }
    rt->global_budget = budget;
    rt->compaction_active = 0;

    /* Pre-allocate buffers for head extraction */
    float * k_real = (float *)malloc(n_new * fc * sizeof(float));
    float * k_imag = (float *)malloc(n_new * fc * sizeof(float));

    if (!k_f32 || !scores || !key_pos || !rt->global_scores || !k_real || !k_imag) {
        free(k_f32); free(scores); free(key_pos); free(k_real); free(k_imag);
        rt->n_scored = n_kv;
        return 0;
    }

    if (tria_get_kv_positions(ctx, key_pos, n_old) != n_old) {
        free(k_f32); free(scores); free(key_pos); free(k_real); free(k_imag);
        rt->n_scored = n_kv;
        return 0;
    }

    int total_pruned = 0;

    for (int li = 0; li < nl; li++) {
        struct ggml_tensor * k_tensor = tria_get_k_tensor(ctx, li);
        if (!k_tensor) continue;

        /* Read only new K rows from GPU (score_start..n_old-1) */
        size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_gqa);
        size_t read_offset = (size_t)score_start * row_size;
        size_t read_bytes = (size_t)n_new * row_size;

        if (k_tensor->type == GGML_TYPE_F16) {
            uint16_t * k_f16 = (uint16_t *)malloc(read_bytes);
            if (!k_f16) continue;
            ggml_backend_tensor_get(k_tensor, k_f16, read_offset, read_bytes);
            for (int i = 0; i < n_new * n_embd_k_gqa; i++) {
                k_f32[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)k_f16)[i]);
            }
            free(k_f16);
        } else if (k_tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(k_tensor, k_f32, read_offset, n_new * n_embd_k_gqa * sizeof(float));
        } else {
            continue;
        }

        /* Score each KV head — only new tokens */
        for (int kvi = 0; kvi < nkv; kvi++) {
            for (int s = 0; s < n_new; s++) {
                float * row = k_f32 + s * n_embd_k_gqa + kvi * hd;
                for (int f = 0; f < fc; f++) {
                    k_real[s * fc + f] = row[f];
                    k_imag[s * fc + f] = row[fc + f];
                }
            }

            tria_score_kv_head(rt->stats, k_real, k_imag,
                               key_pos + score_start,
                               n_kv, n_new, li, kvi, scores);

            /* Z-normalize new scores, max-aggregate into global */
            float mean = 0, var = 0;
            for (int s = 0; s < n_new; s++) mean += scores[s];
            mean /= n_new;
            for (int s = 0; s < n_new; s++) {
                float d = scores[s] - mean;
                var += d * d;
            }
            float std = sqrtf(var / n_new + 1e-8f);
            for (int s = 0; s < n_new; s++) {
                float z = (scores[s] - mean) / std;
                if (z > rt->global_scores[score_start + s]) {
                    rt->global_scores[score_start + s] = z;
                }
            }
        }
    }

    /* For full rescore, total_pruned is informational */
    total_pruned = (n_old - budget) * nl * nkv;

    if (total_pruned > 0) {
        fprintf(stderr, "tria_score: pruned %d tokens across %d×%d heads\n",
                total_pruned, nl, nkv);
    }

    free(k_f32);
    free(scores);
    free(key_pos);
    free(k_real);
    free(k_imag);

    {
        const int compacted = tria_compact_kv(rt, ctx);
        if (compacted > 0) {
            /* Compact global_scores to match new cache layout:
               retained positions 0..budget-1 keep their scores,
               window positions budget..budget+window-1 get -inf (will be rescored) */
            int new_n = tria_get_used_n_kv(ctx);
            int new_old = new_n - rt->window;
            if (new_old > 0 && new_old <= rt->global_n) {
                /* Scores for retained tokens are already at indices selected by compaction.
                   After compaction, rows 0..new_old-1 are the retained old tokens.
                   Their scores in global_scores correspond to the original indices
                   that were kept — but compaction reordered them. For simplicity,
                   just keep the first new_old scores (they were the top-K). */
                rt->global_n = new_old;
            }
            rt->compaction_active = 1;
            rt->n_scored = tria_get_n_kv(ctx);
            rt->score_pass++;
            return compacted;
        }
    }

    rt->n_scored = n_kv;
    rt->score_pass++;
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

    /* Find the budget-th largest score as threshold via quickselect O(n) */
    static float * sorted = NULL;
    static int sorted_cap = 0;
    if (n_old > sorted_cap) {
        free(sorted);
        sorted = (float *)malloc(n_old * sizeof(float));
        sorted_cap = n_old;
    }
    memcpy(sorted, rt->global_scores, n_old * sizeof(float));

    /* Quickselect: partition around budget-th largest */
    int lo = 0, hi = n_old - 1, target = budget - 1;
    while (lo < hi) {
        float pivot = sorted[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (sorted[i] > pivot) i++;
            while (sorted[j] < pivot) j--;
            if (i <= j) {
                float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
                i++; j--;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }
    float threshold = sorted[target];

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
