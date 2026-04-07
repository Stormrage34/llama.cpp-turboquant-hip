/*
 * triattention.h — TriAttention scoring for ggml/llama.cpp
 *
 * Loads TRIA v2 calibration stats and provides scoring function
 * compatible with ggml_map_custom1 callback.
 *
 * Usage in llama.cpp build graph:
 *   1. Load stats at model init: tria_load(path)
 *   2. Every 128 tokens, score keys: tria_score_keys(...)
 *   3. Select top-B per KV head
 *
 * This file is self-contained — no ggml dependency for the math.
 * Integration with ggml compute graph via ggml_map_custom1.
 */

#ifndef TRIATTENTION_H
#define TRIATTENTION_H

#include <stdint.h>
#include <stdbool.h>

#define TRIA_MAGIC      0x54524941
#define TRIA_HEADER_SIZE 64
#define TRIA_N_OFFSETS   17

#ifdef __cplusplus
extern "C" {
#endif

/* Per-head calibration stats */
struct tria_head_stats {
    float *q_mean_real;   /* [fc] */
    float *q_mean_imag;   /* [fc] */
    float *q_abs_mean;    /* [fc] */
    float *qma;           /* [fc] precomputed |E[q_f]| */
};

/* Full calibration data */
struct tria_stats {
    uint32_t num_layers;
    uint32_t num_heads;      /* attention heads */
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t freq_count;     /* head_dim / 2 */
    float    rope_theta;
    float    attn_scale;
    float   *layer_budget_scales;  /* [num_layers] */
    float   *omega;                /* [freq_count] precomputed */
    struct tria_head_stats *heads;  /* [num_layers * num_heads] */
};

/* Load TRIA v1/v2 binary stats file. Returns NULL on error. */
struct tria_stats * tria_load(const char *path);

/* Free stats. */
void tria_free(struct tria_stats *stats);

/*
 * Score keys for one KV head.
 * Aggregates across GQA query heads (z-normalize + max).
 *
 * k_pre_real[seq_len][fc] — pre-RoPE key real halves
 * k_pre_imag[seq_len][fc] — pre-RoPE key imag halves
 * key_pos[seq_len]        — absolute positions
 * cur_pos                 — trigger position
 * layer_idx, kv_head_idx  — which KV head
 * out_scores[seq_len]     — output: final aggregated scores
 */
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
);

/*
 * Get per-layer budget for top-K selection.
 * Returns: floor(base_budget * layer_budget_scale)
 */
static inline int tria_layer_budget(const struct tria_stats *s, int layer, int base_budget) {
    float scale = s->layer_budget_scales[layer];
    int b = (int)(base_budget * scale);
    return b > 0 ? b : 1;
}

#ifdef __cplusplus
}
#endif

#endif /* TRIATTENTION_H */
