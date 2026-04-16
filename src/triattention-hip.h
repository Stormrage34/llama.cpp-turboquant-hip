#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes);

/*
 * GPU-side TriAttention scoring for q8_0 K cache.
 * Reads K directly on GPU, computes trig importance scores, returns
 * per-token global_scores (max across layers/heads) to host.
 *
 * k_data_dev   : GPU pointer to q8_0 K tensor (all tokens, this layer)
 * n_tokens     : number of tokens to score (n_new, starting at score_start)
 * score_start  : offset into K tensor (first token to score)
 * n_embd_k_gqa : total K embedding dim (n_kv_heads * head_dim)
 * n_kv_heads   : number of KV heads
 * head_dim     : per-head dimension
 * freq_count   : head_dim / 2
 * key_pos      : host array of token positions [n_tokens]
 * omega_dev    : GPU pointer to precomputed omega[freq_count]
 * q_mean_real  : GPU pointer to q_mean_real[num_layers * n_kv_heads * freq_count]
 * q_mean_imag  : GPU pointer to q_mean_imag[num_layers * n_kv_heads * freq_count]
 * q_mean_offset: element offset for this layer inside q_mean_real/q_mean_imag
 * layer_weight : normalized layer importance weight
 * global_scores_dev : GPU pointer to global_scores[n_tokens] (in/out, max-aggregated)
 * n_offsets    : number of future offsets (TRIA_N_OFFSETS = 17)
 * offsets      : host array of future offsets [n_offsets]
 */
int tria_hip_score_q8_0(
        const void * k_data_dev,
        int n_tokens,
        int score_start,
        int n_embd_k_gqa,
        int n_kv_heads,
        int head_dim,
        int freq_count,
        const int * key_pos,
        const float * omega_dev,
        const float * q_mean_real_dev,
        const float * q_mean_imag_dev,
        int q_mean_offset,
        float layer_weight,
        float * global_scores_dev,
        int n_offsets,
        const int * offsets);

/* Allocate/free persistent GPU buffers for scoring stats */
int  tria_hip_stats_upload(const float * omega, int freq_count,
                            const float * q_mean_real, const float * q_mean_imag,
                            int n_kv_heads,
                            float ** omega_dev_out,
                            float ** q_mean_real_dev_out,
                            float ** q_mean_imag_dev_out);
void tria_hip_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev);
int  tria_hip_scores_download(float * scores, const float * scores_dev, int n_scores);

#ifdef __cplusplus
}
#endif
