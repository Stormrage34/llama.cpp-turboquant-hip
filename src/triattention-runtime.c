/*
 * triattention-runtime.c — TriAttention runtime scoring
 */

#define _GNU_SOURCE
#include "triattention-runtime.h"
#include "triattention.h"
#include "triattention-backend.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tria_runtime * g_tria_rt = NULL;

struct llama_kv_layer {
    struct ggml_tensor * k;
    struct ggml_tensor * v;
};

extern struct ggml_tensor * tria_get_k_tensor(void * ctx, int layer_idx);
extern struct ggml_tensor * tria_get_v_tensor(void * ctx, int layer_idx);
extern int tria_get_n_kv(void * ctx);
extern int tria_get_used_n_kv(void * ctx);
extern int tria_get_kv_positions(void * ctx, int * positions, int max_positions);
extern int tria_compact_kv(struct tria_runtime * rt, void * ctx);

struct tria_runtime * tria_runtime_init(
    struct tria_stats * stats,
    int budget_pct,
    int window,
    int interval,
    int sink
) {
    if (!stats || budget_pct <= 0 || budget_pct > 100) return NULL;
    if (stats->num_kv_heads == 0 || stats->num_layers == 0) return NULL;

    /* Overflow check (Codex review) */
    uint32_t n_pairs_u = (uint32_t)stats->num_layers * (uint32_t)stats->num_kv_heads;
    if (n_pairs_u / stats->num_layers != stats->num_kv_heads) return NULL;
    int n_pairs = (int)n_pairs_u;

    struct tria_runtime * rt = calloc(1, sizeof(*rt));
    if (!rt) return NULL;

    /* Initialize GPU backend (once) */
    static int backend_initialized = 0;
    if (!backend_initialized) {
        tria_backend_init();
        backend_initialized = 1;
    }

    rt->stats      = stats;
    rt->budget_pct = budget_pct;
    rt->window     = window;
    rt->interval   = interval;
    rt->sink       = sink;
    rt->n_scored   = 0;

    rt->retained       = calloc(n_pairs, sizeof(int *));
    rt->retained_count = calloc(n_pairs, sizeof(int));
    if (!rt->retained || !rt->retained_count) {
        free(rt->retained); free(rt->retained_count); free(rt);
        return NULL;
    }

    return rt;
}

void tria_runtime_free(struct tria_runtime * rt) {
    if (!rt) return;

    /* Clear global pointer to prevent UAF (Codex review) */
    if (g_tria_rt == rt) {
        g_tria_rt = NULL;
    }

    if (rt->retained) {
        int n_pairs = rt->stats->num_layers * rt->stats->num_kv_heads;
        for (int i = 0; i < n_pairs; i++) {
            free(rt->retained[i]);
        }
        free(rt->retained);
    }
    free(rt->retained_count);
    free(rt->global_scores);

    /* Free GPU scoring stats */
    if (g_tria_backend.stats_free) {
        g_tria_backend.stats_free(rt->gpu_omega, rt->gpu_q_mean_real, rt->gpu_q_mean_imag);
        if (rt->gpu_global_scores)
            g_tria_backend.stats_free(rt->gpu_global_scores, NULL, NULL);
    }

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

    /* Warn once if multi-slot detected — scoring is approximate */
    static int multi_slot_warned = 0;
    if (!multi_slot_warned && n_used > n_kv + 128) {
        fprintf(stderr, "tria: warning: n_used (%d) >> n_kv (%d), multi-slot detected. "
                "Scoring uses max-over-sequences which is approximate.\n", n_used, n_kv);
        multi_slot_warned = 1;
    }

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
    int absolute_budget = 0;
    {
        const char * bt = getenv("TRIA_BUDGET_TOKENS");
        if (bt) { budget = atoi(bt); absolute_budget = 1; }
    }
    if (budget <= 0) budget = 1;

    /* Exponential ramp eviction (opt-in via TRIA_RAMP_START_PCT).
     * When set: cap doubles each pass (e.g. 10->20->40->80->100%).
     * Prevents abrupt semantic shock after long prefill.
     * Disabled when TRIA_BUDGET_TOKENS is set (absolute budget). */
    if (!absolute_budget) {
        const char * rsp = getenv("TRIA_RAMP_START_PCT");
        if (rsp) {
            int ramp_start = atoi(rsp);
            if (ramp_start < 1) ramp_start = 1;
            if (ramp_start > 100) ramp_start = 100;
            int cap_pct = ramp_start;
            for (int p = 0; p < rt->score_pass && cap_pct < 100; p++)
                cap_pct *= 2;
            if (cap_pct > 100) cap_pct = 100;
            int max_evict = n_old * cap_pct / 100;
            if (max_evict < 256) max_evict = 256;
            int ramp_budget = n_old - max_evict;
            if (ramp_budget > budget) budget = ramp_budget;
        }
    }

    /* Force full rescore every pass (incremental disabled pending fix) */
    #define TRIA_FULL_RESCORE_INTERVAL 1
    int full_rescore = (rt->score_pass % TRIA_FULL_RESCORE_INTERVAL == 0)
                     || !rt->global_scores
                     || rt->global_n < 1;

    int n_prev = 0;
    int n_new  = n_old;
    int score_start = 0;

    if (!full_rescore && rt->global_scores && rt->global_n > 0) {
        n_prev = rt->global_n;
        if (n_prev > n_old) n_prev = n_old;
        score_start = n_prev;
        n_new = n_old - n_prev;
        if (n_new <= 0) {
            rt->global_budget = budget;
            rt->n_scored = n_kv;
            rt->score_pass++;
            return 0;
        }
    }

    fprintf(stderr, "tria_score: n_kv=%d n_old=%d budget=%d%s new=%d mode=%s (pass %d)\n",
            n_kv, n_old, budget,
            absolute_budget ? " (abs)" : " (pct)",
            n_new,
            full_rescore ? "full" : "incremental", rt->score_pass);

    if (!ctx) { rt->n_scored = n_kv; return 0; }

    /* Validate stats dimensions against actual KV tensor (Codex review) */
    {
        struct ggml_tensor * k0 = tria_get_k_tensor(ctx, 0);
        if (k0) {
            int64_t actual_row = k0->ne[0];
            int expected_row = nkv * hd;
            if (actual_row != expected_row) {
                fprintf(stderr, "tria_score: TRIA stats mismatch: expected row %d, got %d — skipping\n",
                        expected_row, (int)actual_row);
                rt->n_scored = n_kv;
                return 0;
            }
        }
    }

    int n_embd_k_gqa = nkv * hd;
    /* Overflow check for element count (Codex review #2) */
    size_t n_elem = (size_t)n_new * (size_t)n_embd_k_gqa;
    if (n_elem / (size_t)n_new != (size_t)n_embd_k_gqa) {
        rt->n_scored = n_kv;
        return 0;
    }
    size_t k_bytes = n_elem * sizeof(float);
    float * k_f32 = (float *)malloc(k_bytes);
    float * scores = (float *)malloc(n_new * sizeof(float));
    int * key_pos = (int *)malloc(n_old * sizeof(int));

    /* Resize global scores if needed */
    if (rt->global_n < n_old) {
        float * new_gs = (float *)malloc(n_old * sizeof(float));
        if (new_gs) {
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
            free(k_f32); free(scores); free(key_pos);
            rt->n_scored = n_kv;
            return 0;
        }
    } else if (full_rescore) {
        for (int i = 0; i < n_old; i++) rt->global_scores[i] = -1e30f;
        rt->global_n = n_old;
    } else {
        for (int i = n_prev; i < n_old; i++) rt->global_scores[i] = -1e30f;
    }
    rt->global_budget = budget;
    rt->compaction_active = 0;

    float * k_real = (float *)malloc((size_t)n_new * fc * sizeof(float));
    float * k_imag = (float *)malloc((size_t)n_new * fc * sizeof(float));

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

    /* Precompute mean layer weight for normalization */
    float layer_weight_mean = 0.0f;
    for (int l = 0; l < nl; l++) layer_weight_mean += rt->stats->layer_budget_scales[l];
    layer_weight_mean /= nl;
    if (layer_weight_mean <= 0.0f) layer_weight_mean = 1.0f;

    /* Detect K cache type and choose scoring path */
    int score_stride = 1;
    int use_gpu_scoring = 0;
    {
        struct ggml_tensor * k0 = tria_get_k_tensor(ctx, 0);
        int is_q8_0 = k0 && k0->type == GGML_TYPE_Q8_0;
        const char * sls = getenv("TRIA_SCORE_LAYER_STRIDE");
        if (sls) score_stride = atoi(sls);
        if (score_stride < 1) score_stride = 1;

        /* GPU scoring path: use for q8_0 if GPU stats are available */
        if (is_q8_0) {
            /* Lazy upload GPU stats on first use */
            if (!rt->gpu_omega || rt->gpu_q_mean_layers != nl || rt->gpu_q_mean_kv_heads != nkv) {
                /* Build flat q_mean_real/imag arrays [nl * nkv * fc] */
                float * qmr = (float *)malloc((size_t)nl * nkv * fc * sizeof(float));
                float * qmi = (float *)malloc((size_t)nl * nkv * fc * sizeof(float));
                if (qmr && qmi) {
                    for (int li = 0; li < nl; li++) {
                        for (int kvi = 0; kvi < nkv; kvi++) {
                            struct tria_head_stats * hs = &rt->stats->heads[li * nkv + kvi];
                            const size_t base = ((size_t)li * nkv + kvi) * fc;
                            for (int f = 0; f < fc; f++) {
                                qmr[base + f] = hs->q_mean_real[f];
                                qmi[base + f] = hs->q_mean_imag[f];
                            }
                        }
                    }
                    if (g_tria_backend.stats_free)
                        g_tria_backend.stats_free(rt->gpu_omega, rt->gpu_q_mean_real, rt->gpu_q_mean_imag);
                    rt->gpu_omega = NULL;
                    rt->gpu_q_mean_real = NULL;
                    rt->gpu_q_mean_imag = NULL;
                    rt->gpu_q_mean_layers = 0;
                    rt->gpu_q_mean_kv_heads = 0;
                    if (g_tria_backend.stats_upload &&
                        g_tria_backend.stats_upload(rt->stats->omega, fc, qmr, qmi, nl * nkv,
                                              &rt->gpu_omega, &rt->gpu_q_mean_real, &rt->gpu_q_mean_imag) == 0) {
                        rt->gpu_q_mean_layers = nl;
                        rt->gpu_q_mean_kv_heads = nkv;
                    }
                }
                free(qmr);
                free(qmi);
            }
            if (rt->gpu_omega) {
                use_gpu_scoring = 1;
                score_stride = 1; /* GPU can score all layers efficiently */
            } else {
                /* Fallback: CPU with stride */
                if (!sls) score_stride = 4;
            }
        }
    }

    /* Upload this pass's global_scores slice once; all GPU layers reuse it. */
    if (use_gpu_scoring) {
        if (rt->gpu_global_scores) {
            if (g_tria_backend.stats_free)
                g_tria_backend.stats_free(rt->gpu_global_scores, NULL, NULL);
            rt->gpu_global_scores = NULL;
            rt->gpu_global_scores_n = 0;
        }
        if (g_tria_backend.stats_upload &&
            g_tria_backend.stats_upload(rt->global_scores + score_start, n_new, NULL, NULL, 0,
                                  &rt->gpu_global_scores, NULL, NULL) == 0) {
            rt->gpu_global_scores_n = n_new;
        } else {
            use_gpu_scoring = 0;
        }
    }

    /* Precompute future offsets (same as CPU path) */
    int offsets[TRIA_N_OFFSETS];
    {
        int o = 1;
        for (int i = 0; i < TRIA_N_OFFSETS; i++) {
            offsets[i] = o;
            o *= 2;
        }
    }

    for (int li = 0; li < nl; li++) {
        if (li % score_stride != 0) continue;  /* sampled-layer skip */
        struct ggml_tensor * k_tensor = tria_get_k_tensor(ctx, li);
        if (!k_tensor) continue;

        /* GPU scoring path for q8_0: score directly on GPU, no CPU transfer */
        if (use_gpu_scoring && k_tensor->type == GGML_TYPE_Q8_0 && rt->gpu_omega) {
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;

            if (rt->gpu_global_scores) {
                g_tria_backend.score_q8_0(
                    k_tensor->data,
                    n_new, score_start,
                    n_embd_k_gqa, nkv, hd, fc,
                    key_pos + score_start,
                    rt->gpu_omega,
                    rt->gpu_q_mean_real, rt->gpu_q_mean_imag,
                    li * nkv * fc,
                    layer_weight,
                    rt->gpu_global_scores,
                    TRIA_N_OFFSETS, offsets);
            }
            continue;  /* skip CPU path for this layer */
        }

        size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_gqa);
        size_t read_offset = (size_t)score_start * row_size;
        size_t read_bytes = (size_t)n_new * row_size;

        if (k_tensor->type == GGML_TYPE_F16) {
            uint16_t * k_f16 = (uint16_t *)malloc(read_bytes);
            if (!k_f16) continue;
            ggml_backend_tensor_get(k_tensor, k_f16, read_offset, read_bytes);
            for (size_t i = 0; i < n_elem; i++) {
                k_f32[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)k_f16)[i]);
            }
            free(k_f16);
        } else if (k_tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(k_tensor, k_f32, read_offset, n_elem * sizeof(float));
        } else if (k_tensor->type == GGML_TYPE_Q8_0) {
            /* Q8_0 block: [fp16 scale d][32 x int8 qs] — sizeof = 34 bytes.
             * Only reached when score_stride reduces layer count to manageable. */
            #define TRIA_QK8_0 32
            #define TRIA_Q8_0_BLOCK_SIZE (sizeof(ggml_fp16_t) + TRIA_QK8_0)
            if (n_embd_k_gqa % TRIA_QK8_0 != 0) { continue; }
            uint8_t * k_q8 = (uint8_t *)malloc(read_bytes);
            if (!k_q8) continue;
            ggml_backend_tensor_get(k_tensor, k_q8, read_offset, read_bytes);
            const int nb = n_embd_k_gqa / TRIA_QK8_0;
            for (int s = 0; s < n_new; s++) {
                const uint8_t * row_q8 = k_q8 + (size_t)s * row_size;
                float * dst = k_f32 + (size_t)s * n_embd_k_gqa;
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = row_q8 + b * TRIA_Q8_0_BLOCK_SIZE;
                    ggml_fp16_t d_fp16;
                    memcpy(&d_fp16, blk, sizeof(ggml_fp16_t));
                    float d = ggml_fp16_to_fp32(d_fp16);
                    const int8_t * qs = (const int8_t *)(blk + sizeof(ggml_fp16_t));
                    for (int j = 0; j < TRIA_QK8_0; j++)
                        dst[b * TRIA_QK8_0 + j] = d * qs[j];
                }
            }
            free(k_q8);
        } else {
            /* Generic dequant path — handles turbo2/3/4 and any future quantized KV type */
            const struct ggml_type_traits * traits = ggml_get_type_traits(k_tensor->type);
            if (!traits || !traits->to_float) { continue; }
            uint8_t * k_raw = (uint8_t *)malloc(read_bytes);
            if (!k_raw) continue;
            ggml_backend_tensor_get(k_tensor, k_raw, read_offset, read_bytes);
            for (int s = 0; s < n_new; s++) {
                traits->to_float(k_raw + (size_t)s * row_size,
                                 k_f32 + (size_t)s * n_embd_k_gqa,
                                 n_embd_k_gqa);
            }
            free(k_raw);
        }

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

            float mean = 0, var = 0;
            for (int s = 0; s < n_new; s++) mean += scores[s];
            mean /= n_new;
            for (int s = 0; s < n_new; s++) {
                float d = scores[s] - mean;
                var += d * d;
            }
            float std = sqrtf(var / n_new + 1e-8f);

            /* Layer-aware max aggregation: weight positive z-scores by layer importance.
             * Diffuse early layers (high layer_budget_scale) contribute more.
             * Leave negative z-scores unweighted: scaling negative scores would invert the
             * intended retention bias by making important layers' below-mean tokens look worse.
             * Preserves max-aggregation semantics (same eviction rate as before)
             * while using calibration data to improve token selection quality. */
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;

            for (int s = 0; s < n_new; s++) {
                float z = (scores[s] - mean) / std;
                float wz = z > 0.0f ? z * layer_weight : z;
                if (wz > rt->global_scores[score_start + s])
                    rt->global_scores[score_start + s] = wz;
            }
        }
    }

    if (use_gpu_scoring && rt->gpu_global_scores) {
        g_tria_backend.scores_download(rt->global_scores + score_start, rt->gpu_global_scores, n_new);
    }

    total_pruned = (n_old - budget) * nl * nkv;

    /* ---- Value-aware scoring boost (VATP/OBCache-inspired) ---- */
    /* Compute per-token V energy across all layers, z-normalize,
       and add lambda * v_z to global_scores (bidirectional).
       Only works with non-transposed V (flash_attn mode).         */
    {
        const float lambda = 0.25f;
        float * v_energy = (float *)calloc(n_old, sizeof(float));
        if (v_energy) {
            int v_layers = 0;
            for (int li = 0; li < nl; li++) {
                struct ggml_tensor * v_tensor = tria_get_v_tensor(ctx, li);
                if (!v_tensor) continue;
                /* Skip transposed V (nb[1] > nb[2]) — row reads would be wrong */
                if (v_tensor->nb[1] > v_tensor->nb[2]) continue;

                int n_embd_v = (int)v_tensor->ne[0];
                size_t v_row_size = ggml_row_size(v_tensor->type, n_embd_v);
                uint8_t * row_buf = (uint8_t *)malloc(v_row_size);
                if (!row_buf) continue;

                if (v_tensor->type == GGML_TYPE_F16 || v_tensor->type == GGML_TYPE_F32) {
                    for (int s = 0; s < n_old; s++) {
                        ggml_backend_tensor_get(v_tensor, row_buf,
                                                (size_t)s * v_row_size, v_row_size);
                        float energy = 0.0f;
                        if (v_tensor->type == GGML_TYPE_F16) {
                            const ggml_fp16_t * vf16 = (const ggml_fp16_t *)row_buf;
                            for (int d = 0; d < n_embd_v; d++) {
                                float v = ggml_fp16_to_fp32(vf16[d]);
                                energy += v * v;
                            }
                        } else {
                            const float * vf32 = (const float *)row_buf;
                            for (int d = 0; d < n_embd_v; d++)
                                energy += vf32[d] * vf32[d];
                        }
                        v_energy[s] += energy;
                    }
                    v_layers++;
                } else if (v_tensor->type == GGML_TYPE_TURBO3_0) {
                    /* turbo3: block norms as energy proxy (WHT equalizes,
                       so vstd guard will likely skip the boost) */
                    int n_blocks = n_embd_v / 128;
                    if (n_blocks > 0) {
                        for (int s = 0; s < n_old; s++) {
                            ggml_backend_tensor_get(v_tensor, row_buf,
                                                    (size_t)s * v_row_size, v_row_size);
                            float energy = 0.0f;
                            for (int b = 0; b < n_blocks; b++) {
                                ggml_fp16_t norm_fp16;
                                memcpy(&norm_fp16, row_buf + b * 14, sizeof(ggml_fp16_t));
                                float norm = ggml_fp16_to_fp32(norm_fp16);
                                energy += norm * norm;
                            }
                            v_energy[s] += energy;
                        }
                        v_layers++;
                    }
                }
                /* Other quant types: skip (no safe norm proxy) */
                free(row_buf);
            }

            if (v_layers > 0) {
                /* log1p + z-normalize */
                for (int s = 0; s < n_old; s++)
                    v_energy[s] = logf(1.0f + v_energy[s] / v_layers);

                float vmean = 0.0f;
                for (int s = 0; s < n_old; s++) vmean += v_energy[s];
                vmean /= n_old;

                float vvar = 0.0f;
                for (int s = 0; s < n_old; s++) {
                    float d = v_energy[s] - vmean;
                    vvar += d * d;
                }
                float vstd = sqrtf(vvar / n_old + 1e-8f);

                /* Guard: if variance collapsed (WHT-normalized turbo3), skip boost */
                if (vstd > 0.01f) {
                    int boosted = 0;
                    for (int s = 0; s < n_old; s++) {
                        float vz = (v_energy[s] - vmean) / vstd;
                        float boost = lambda * vz;
                        rt->global_scores[s] += boost;
                        if (boost > 0.01f) boosted++;
                    }
                    fprintf(stderr, "tria_score: value-aware boost applied to %d/%d tokens (lambda=%.2f, vstd=%.4f)\n",
                            boosted, n_old, lambda, vstd);
                } else {
                    fprintf(stderr, "tria_score: value-aware skipped (vstd=%.6f < 0.01, WHT-normalized)\n", vstd);
                }
            }
            free(v_energy);
        }
    }

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
            int new_n = tria_get_used_n_kv(ctx);
            int new_old = new_n - rt->window;
            if (new_old > 0 && new_old <= rt->global_n) {
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

    /* Clamp n_old to actual mask/scores bounds (Codex review: OOB fix) */
    if (n_old > n_kv) n_old = n_kv;
    if (n_old > rt->global_n) n_old = rt->global_n;

    int budget = rt->global_budget;
    if (budget >= n_old) {
        memset(evict_mask, 0, n_kv);
        return 1;
    }

    /* Quickselect for threshold — use local buffer, not static (Codex review: thread safety) */
    float * sorted = (float *)malloc(n_old * sizeof(float));
    if (!sorted) return 0;
    memcpy(sorted, rt->global_scores, n_old * sizeof(float));

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
    /* threshold is unused by the segment-based eviction below,
       but kept for potential future use */
    (void)sorted[target];
    free(sorted);

    /* Build mask: V3 hybrid — prefix protection + per-segment eviction quota */
    memset(evict_mask, 0, n_kv);

    const int n_segments = 8;
    int prefix = rt->sink > 0 ? rt->sink : 128;
    if (prefix > n_old) prefix = n_old;

    int evictable = n_old - prefix;
    int n_to_evict = n_old - budget;
    if (evictable <= 0 || n_to_evict <= 0) return 1;

    int seg_size = evictable / n_segments;
    if (seg_size < 1) seg_size = 1;
    int actual_segs = (evictable + seg_size - 1) / seg_size;

    int total_evicted = 0;
    for (int s = 0; s < actual_segs && total_evicted < n_to_evict; s++) {
        int seg_start = prefix + s * seg_size;
        int seg_end   = seg_start + seg_size;
        if (seg_end > n_old) seg_end = n_old;
        int seg_len = seg_end - seg_start;
        if (seg_len <= 0) continue;

        int seg_evict = (n_to_evict * seg_len + evictable - 1) / evictable;
        int remaining = n_to_evict - total_evicted;
        if (seg_evict > remaining) seg_evict = remaining;
        if (seg_evict > seg_len)   seg_evict = seg_len;
        if (seg_evict <= 0) continue;

        int * idx = (int *)malloc(seg_len * sizeof(int));
        if (!idx) continue;
        for (int i = 0; i < seg_len; i++) idx[i] = seg_start + i;

        for (int e = 0; e < seg_evict; e++) {
            int min_j = e;
            for (int j = e + 1; j < seg_len; j++) {
                if (rt->global_scores[idx[j]] < rt->global_scores[idx[min_j]])
                    min_j = j;
            }
            if (min_j != e) { int tmp = idx[e]; idx[e] = idx[min_j]; idx[min_j] = tmp; }
            /* Bounds check before writing mask (Codex review: OOB fix) */
            if (idx[e] >= 0 && idx[e] < n_kv) {
                evict_mask[idx[e]] = 1;
            }
            total_evicted++;
        }
        free(idx);
    }

    return 1;
}
