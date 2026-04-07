/*
 * triattention.c — TriAttention scoring implementation
 *
 * Implements TRIA binary loader and per-KV-head scoring with GQA aggregation.
 * Standalone — compiles without ggml for testing, integrates via ggml_map_custom1.
 */

#define _GNU_SOURCE  /* sincosf */
#include "triattention.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* TRIA binary loader                                                  */
/* ------------------------------------------------------------------ */

struct tria_stats * tria_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }

    uint32_t magic, version;
    fread(&magic, 4, 1, fp);
    fread(&version, 4, 1, fp);
    if (magic != TRIA_MAGIC || (version != 1 && version != 2)) {
        fprintf(stderr, "tria_load: bad magic/version: %x v%u\n", magic, version);
        fclose(fp); return NULL;
    }

    struct tria_stats *s = calloc(1, sizeof(*s));
    fread(&s->num_layers,   4, 1, fp);
    fread(&s->num_heads,    4, 1, fp);
    fread(&s->num_kv_heads, 4, 1, fp);
    fread(&s->head_dim,     4, 1, fp);
    fread(&s->freq_count,   4, 1, fp);
    fread(&s->rope_theta,   4, 1, fp);
    fread(&s->attn_scale,   4, 1, fp);

    /* Skip reserved bytes to reach end of 64-byte header */
    fseek(fp, TRIA_HEADER_SIZE, SEEK_SET);

    uint32_t nl = s->num_layers, nh = s->num_heads, fc = s->freq_count;

    /* Per-layer budget scales (v2) */
    s->layer_budget_scales = malloc(nl * sizeof(float));
    if (version >= 2) {
        fread(s->layer_budget_scales, 4, nl, fp);
    } else {
        for (uint32_t i = 0; i < nl; i++) s->layer_budget_scales[i] = 1.0f;
    }

    /* Precompute omega: theta^(-2i/head_dim) */
    s->omega = malloc(fc * sizeof(float));
    for (uint32_t i = 0; i < fc; i++) {
        s->omega[i] = powf(s->rope_theta, -2.0f * i / s->head_dim);
    }

    /* Per-head stats */
    uint32_t total = nl * nh;
    s->heads = calloc(total, sizeof(struct tria_head_stats));
    for (uint32_t h = 0; h < total; h++) {
        struct tria_head_stats *hs = &s->heads[h];
        hs->q_mean_real = malloc(fc * sizeof(float));
        hs->q_mean_imag = malloc(fc * sizeof(float));
        hs->q_abs_mean  = malloc(fc * sizeof(float));
        hs->qma         = malloc(fc * sizeof(float));

        fread(hs->q_mean_real, 4, fc, fp);
        fread(hs->q_mean_imag, 4, fc, fp);
        fread(hs->q_abs_mean,  4, fc, fp);
        fseek(fp, fc * 4, SEEK_CUR);  /* skip mrl */

        /* Precompute |E[q_f]| */
        for (uint32_t f = 0; f < fc; f++) {
            float r = hs->q_mean_real[f], i = hs->q_mean_imag[f];
            hs->qma[f] = sqrtf(r*r + i*i);
        }
    }

    fclose(fp);
    return s;
}

void tria_free(struct tria_stats *s) {
    if (!s) return;
    uint32_t total = s->num_layers * s->num_heads;
    for (uint32_t h = 0; h < total; h++) {
        free(s->heads[h].q_mean_real);
        free(s->heads[h].q_mean_imag);
        free(s->heads[h].q_abs_mean);
        free(s->heads[h].qma);
    }
    free(s->heads);
    free(s->layer_budget_scales);
    free(s->omega);
    free(s);
}

/* ------------------------------------------------------------------ */
/* Single-head scoring (eq 6-11)                                       */
/* ------------------------------------------------------------------ */

/*
 * Precomputed cos/sin table for a set of keys.
 * Shared across all query heads in a GQA group.
 * Layout: [seq_len][N_OFFSETS][fc] for both cos and sin.
 */
struct tria_cs_table {
    float *cos_tab;  /* [seq_len * N_OFFSETS * fc] */
    float *sin_tab;  /* [seq_len * N_OFFSETS * fc] */
    float *ka;       /* [seq_len * fc] — key magnitudes */
    int    seq_len;
    int    fc;
};

static struct tria_cs_table * tria_cs_precompute(
    const float *omega, const float *k_real, const float *k_imag,
    const int *key_pos, int cur_pos, int fc, int seq_len
) {
    static const float offsets[TRIA_N_OFFSETS] = {
        1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536
    };
    struct tria_cs_table *t = malloc(sizeof(*t));
    t->seq_len = seq_len;
    t->fc = fc;
    t->cos_tab = malloc(seq_len * TRIA_N_OFFSETS * fc * sizeof(float));
    t->sin_tab = malloc(seq_len * TRIA_N_OFFSETS * fc * sizeof(float));
    t->ka      = malloc(seq_len * fc * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        const float *kr = k_real + s * fc;
        const float *ki = k_imag + s * fc;
        float base_delta = (float)(cur_pos - key_pos[s]);

        for (int f = 0; f < fc; f++) {
            t->ka[s * fc + f] = sqrtf(kr[f]*kr[f] + ki[f]*ki[f]);
        }
        for (int o = 0; o < TRIA_N_OFFSETS; o++) {
            float delta = base_delta + offsets[o];
            int base = (s * TRIA_N_OFFSETS + o) * fc;
            for (int f = 0; f < fc; f++) {
                float angle = delta * omega[f];
                sincosf(angle, &t->sin_tab[base + f], &t->cos_tab[base + f]);
            }
        }
    }
    return t;
}

static void tria_cs_free(struct tria_cs_table *t) {
    if (!t) return;
    free(t->cos_tab); free(t->sin_tab); free(t->ka); free(t);
}

static void score_keys_single_head(
    const struct tria_head_stats *hs,
    const struct tria_cs_table *cs,
    const float *k_real,
    const float *k_imag,
    int          fc,
    int          seq_len,
    float       *out
) {
    for (int s = 0; s < seq_len; s++) {
        const float *kr = k_real + s * fc;
        const float *ki = k_imag + s * fc;
        float extra = 0.0f;

        /* Precompute rel = q_mean * conj(k) per freq band */
        float rel_r[fc], rel_i[fc];
        for (int f = 0; f < fc; f++) {
            rel_r[f] = hs->q_mean_real[f]*kr[f] + hs->q_mean_imag[f]*ki[f];
            rel_i[f] = hs->q_mean_imag[f]*kr[f] - hs->q_mean_real[f]*ki[f];
            float residual = hs->q_abs_mean[f] - hs->qma[f];
            if (residual > 0.0f) extra += residual * cs->ka[s * fc + f];
        }

        /* Trig score using precomputed cos/sin table */
        float trig_sum = 0.0f;
        for (int o = 0; o < TRIA_N_OFFSETS; o++) {
            int base = (s * TRIA_N_OFFSETS + o) * fc;
            float trig = 0.0f;
            for (int f = 0; f < fc; f++) {
                trig += rel_r[f] * cs->cos_tab[base + f]
                      - rel_i[f] * cs->sin_tab[base + f];
            }
            trig_sum += trig;
        }
        out[s] = trig_sum / (float)TRIA_N_OFFSETS + extra;
    }
}

/* ------------------------------------------------------------------ */
/* GQA-aggregated scoring (eq 12-13)                                   */
/* ------------------------------------------------------------------ */

void tria_score_kv_head(
    const struct tria_stats *stats,
    const float *k_pre_real,
    const float *k_pre_imag,
    const int   *key_pos,
    int          cur_pos,
    int          seq_len,
    int          layer_idx,
    int          kv_head_idx,
    float       *out_scores
) {
    int nh = stats->num_heads;
    int nkv = stats->num_kv_heads;
    int fc = stats->freq_count;
    int gqa = nh / nkv;

    /* Precompute cos/sin table — shared across all GQA query heads */
    struct tria_cs_table *cs = tria_cs_precompute(
        stats->omega, k_pre_real, k_pre_imag, key_pos, cur_pos, fc, seq_len);

    float *tmp = malloc(seq_len * sizeof(float));
    bool first = true;

    for (int g = 0; g < gqa; g++) {
        int ah = kv_head_idx * gqa + g;
        const struct tria_head_stats *hs = &stats->heads[layer_idx * nh + ah];

        score_keys_single_head(hs, cs, k_pre_real, k_pre_imag, fc, seq_len, tmp);

        /* Z-normalize */
        float mean = 0.0f;
        for (int s = 0; s < seq_len; s++) mean += tmp[s];
        mean /= seq_len;

        float var = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            float d = tmp[s] - mean;
            var += d * d;
        }
        float std = sqrtf(var / seq_len);
        if (std < 1e-6f) std = 1e-6f;

        for (int s = 0; s < seq_len; s++) {
            float z = (tmp[s] - mean) / std;
            if (first) {
                out_scores[s] = z;
            } else if (z > out_scores[s]) {
                out_scores[s] = z;
            }
        }
        first = false;
    }

    free(tmp);
    tria_cs_free(cs);
}
