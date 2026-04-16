#include "triattention-hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

/* ── existing compact kernel ─────────────────────────────────────────── */

static __global__ void tria_hip_gather_rows_kernel(
        const uint8_t * tensor_data,
        const uint32_t * indices,
        uint8_t * scratch,
        size_t row_bytes,
        uint32_t n_move) {
    const size_t total_bytes = row_bytes * n_move;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_bytes) {
        return;
    }

    const size_t row = idx / row_bytes;
    const size_t col = idx - row * row_bytes;
    const size_t src_row = indices[row];

    scratch[idx] = tensor_data[src_row * row_bytes + col];
}

extern "C" int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes) {
    if (tensor_data == nullptr || h_indices == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    if (first_move >= n_keep || row_bytes == 0) {
        return 0;
    }

    const uint32_t n_move = n_keep - first_move;
    const size_t indices_bytes = static_cast<size_t>(n_move) * sizeof(uint32_t);
    const size_t scratch_bytes = static_cast<size_t>(n_move) * row_bytes;

    uint32_t * d_indices = nullptr;
    uint8_t * d_scratch = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void **>(&d_indices), indices_bytes);
    if (err != hipSuccess) {
        return static_cast<int>(err);
    }

    err = hipMalloc(reinterpret_cast<void **>(&d_scratch), scratch_bytes);
    if (err != hipSuccess) {
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }

    err = hipMemcpy(d_indices, h_indices + first_move, indices_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        (void) hipFree(d_scratch);
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }

    constexpr uint32_t block_size = 256;
    const size_t total_bytes = scratch_bytes;
    const uint32_t n_blocks = static_cast<uint32_t>((total_bytes + block_size - 1) / block_size);

    hipLaunchKernelGGL(
            tria_hip_gather_rows_kernel,
            dim3(n_blocks),
            dim3(block_size),
            0,
            0,
            static_cast<const uint8_t *>(tensor_data),
            d_indices,
            d_scratch,
            static_cast<size_t>(row_bytes),
            n_move);

    err = hipGetLastError();
    if (err == hipSuccess) {
        err = hipMemcpy(
                static_cast<uint8_t *>(tensor_data) + static_cast<size_t>(first_move) * row_bytes,
                d_scratch,
                scratch_bytes,
                hipMemcpyDeviceToDevice);
    }
    if (err == hipSuccess) {
        err = hipDeviceSynchronize();
    }

    (void) hipFree(d_scratch);
    (void) hipFree(d_indices);

    return static_cast<int>(err);
}

/* ── GPU scoring for q8_0 K cache ────────────────────────────────────── */

#define TRIA_QK8_0 32
#define TRIA_Q8_0_BLOCK_SIZE (2 + TRIA_QK8_0)  /* fp16 scale + 32 int8 */
#define TRIA_N_OFFSETS_MAX 17

static inline void tria_hip_free_ptr(void * ptr) {
    if (ptr) {
        (void) hipFree(ptr);
    }
}

/*
 * Pass 1: per-token raw score for one KV head.
 * Each thread handles one token.
 * score[token] = mean over offsets of sum_f(|q_f| * |k_f| * cos(omega_f*delta + phi_f))
 */
static __global__ void tria_score_q8_0_kernel(
        const uint8_t * __restrict__ k_q8,   /* q8_0 K tensor, all tokens */
        int score_start,                       /* first token index */
        int n_tokens,                          /* tokens to score */
        int n_embd_k_gqa,                      /* total K dim */
        int kvi,                               /* which KV head */
        int head_dim,
        int freq_count,                        /* head_dim / 2 */
        const int * __restrict__ key_pos,      /* positions [n_tokens] */
        const float * __restrict__ omega,      /* [freq_count] */
        const float * __restrict__ q_mean_real,/* [freq_count] for this head */
        const float * __restrict__ q_mean_imag,/* [freq_count] for this head */
        const int * __restrict__ offsets,      /* future offsets [n_offsets] */
        int n_offsets,
        float * __restrict__ scores_out        /* [n_tokens] output */
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_tokens) return;

    /* Q8_0 block layout: [fp16 scale (2 bytes)][32 x int8] */
    const int token_idx = score_start + s;
    const int row_bytes = (n_embd_k_gqa / TRIA_QK8_0) * TRIA_Q8_0_BLOCK_SIZE;
    const uint8_t * row = k_q8 + (size_t)token_idx * row_bytes;

    /* Dequantize this token's KV head slice into registers */
    /* head starts at kvi * head_dim, which is kvi * (head_dim/32) blocks */
    const int head_block_start = kvi * (head_dim / TRIA_QK8_0);
    const int n_head_blocks = head_dim / TRIA_QK8_0;

    float k_f32[128];  /* max head_dim=128 */
    for (int b = 0; b < n_head_blocks; b++) {
        const uint8_t * blk = row + (head_block_start + b) * TRIA_Q8_0_BLOCK_SIZE;
        /* fp16 scale */
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        /* manual fp16->fp32 */
        const uint32_t sign = (d_bits >> 15) & 1;
        const uint32_t exp  = (d_bits >> 10) & 0x1f;
        const uint32_t mant = d_bits & 0x3ff;
        float d;
        if (exp == 0) {
            d = (sign ? -1.f : 1.f) * ldexpf((float)mant, -24);
        } else if (exp == 31) {
            d = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
        } else {
            uint32_t f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&d, &f32bits, 4);
        }
        const int8_t * qs = (const int8_t *)(blk + 2);
        for (int j = 0; j < TRIA_QK8_0; j++) {
            k_f32[b * TRIA_QK8_0 + j] = d * (float)qs[j];
        }
    }

    /* Compute trig score: mean over offsets of sum_f(|q_f|*|k_f|*cos(omega_f*delta+phi_f)) */
    const int pos = key_pos[s];
    float score_sum = 0.f;

    for (int oi = 0; oi < n_offsets; oi++) {
        const int delta = offsets[oi];
        float s_trig = 0.f;
        for (int f = 0; f < freq_count; f++) {
            const float kr = k_f32[f];
            const float ki = k_f32[freq_count + f];
            const float qr = q_mean_real[f];
            const float qi = q_mean_imag[f];
            /* |k_f| */
            const float k_norm = sqrtf(kr*kr + ki*ki + 1e-8f);
            /* phi_f = arg(q_f) - arg(k_f) */
            const float phi = atan2f(qi, qr) - atan2f(ki, kr);
            /* S_trig = |q_f| * |k_f| * cos(omega_f * delta + phi_f) */
            const float q_norm = sqrtf(qr*qr + qi*qi + 1e-8f);
            s_trig += q_norm * k_norm * cosf(omega[f] * (float)delta + phi);
        }
        score_sum += s_trig;
    }
    scores_out[s] = score_sum / (float)n_offsets;
}

/*
 * Pass 2: z-normalize scores and max-aggregate into global_scores.
 * One block, shared memory reduction for mean/std.
 */
static __global__ void tria_znorm_maxagg_kernel(
        const float * __restrict__ scores,  /* [n_tokens] raw scores */
        int n_tokens,
        float layer_weight,
        float * __restrict__ global_scores  /* [n_tokens] in/out */
) {
    extern __shared__ float smem[];

    /* Compute mean */
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x)
        local_sum += scores[i];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = smem[0] / (float)n_tokens;
    __syncthreads();

    /* Compute variance */
    float local_var = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        float d = scores[i] - mean;
        local_var += d * d;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float std = sqrtf(smem[0] / (float)n_tokens + 1e-8f);

    /* Z-normalize and max-aggregate */
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        float z = (scores[i] - mean) / std;
        float wz = z > 0.f ? z * layer_weight : z;
        float cur = global_scores[i];
        if (wz > cur) global_scores[i] = wz;
    }
}

extern "C" int tria_hip_stats_upload(
        const float * omega, int freq_count,
        const float * q_mean_real, const float * q_mean_imag,
        int n_kv_heads,
        float ** omega_dev_out,
        float ** q_mean_real_dev_out,
        float ** q_mean_imag_dev_out) {
    hipError_t err;
    const size_t omega_bytes = (size_t)freq_count * sizeof(float);
    const size_t qm_bytes    = (size_t)n_kv_heads * freq_count * sizeof(float);

    if (omega_dev_out && omega && freq_count > 0) {
        *omega_dev_out = nullptr;
        err = hipMalloc((void**)omega_dev_out, omega_bytes);
        if (err != hipSuccess) return (int)err;
        err = hipMemcpy(*omega_dev_out, omega, omega_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            tria_hip_free_ptr(*omega_dev_out);
            *omega_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_real_dev_out && q_mean_real && qm_bytes > 0) {
        *q_mean_real_dev_out = nullptr;
        err = hipMalloc((void**)q_mean_real_dev_out, qm_bytes);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) {
                tria_hip_free_ptr(*omega_dev_out);
                *omega_dev_out = nullptr;
            }
            return (int)err;
        }
        err = hipMemcpy(*q_mean_real_dev_out, q_mean_real, qm_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) {
                tria_hip_free_ptr(*omega_dev_out);
                *omega_dev_out = nullptr;
            }
            tria_hip_free_ptr(*q_mean_real_dev_out);
            *q_mean_real_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_imag_dev_out && q_mean_imag && qm_bytes > 0) {
        *q_mean_imag_dev_out = nullptr;
        err = hipMalloc((void**)q_mean_imag_dev_out, qm_bytes);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) {
                tria_hip_free_ptr(*omega_dev_out);
                *omega_dev_out = nullptr;
            }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) {
                tria_hip_free_ptr(*q_mean_real_dev_out);
                *q_mean_real_dev_out = nullptr;
            }
            return (int)err;
        }
        err = hipMemcpy(*q_mean_imag_dev_out, q_mean_imag, qm_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) {
                tria_hip_free_ptr(*omega_dev_out);
                *omega_dev_out = nullptr;
            }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) {
                tria_hip_free_ptr(*q_mean_real_dev_out);
                *q_mean_real_dev_out = nullptr;
            }
            tria_hip_free_ptr(*q_mean_imag_dev_out);
            *q_mean_imag_dev_out = nullptr;
            return (int)err;
        }
    }

    return 0;
}

extern "C" void tria_hip_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev) {
    tria_hip_free_ptr(omega_dev);
    tria_hip_free_ptr(q_mean_real_dev);
    tria_hip_free_ptr(q_mean_imag_dev);
}

extern "C" int tria_hip_scores_download(float * scores, const float * scores_dev, int n_scores) {
    if (!scores || !scores_dev || n_scores <= 0) return 0;
    return (int)hipMemcpy(scores, scores_dev, (size_t)n_scores * sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" int tria_hip_score_q8_0(
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
        const int * offsets) {

    if (!k_data_dev || n_tokens <= 0) return 0;

    /* Upload key_pos and offsets to GPU */
    int * d_key_pos = nullptr;
    int * d_offsets = nullptr;
    float * d_scores = nullptr;

    hipError_t err;
    err = hipMalloc((void**)&d_key_pos, (size_t)n_tokens * sizeof(int));
    if (err != hipSuccess) return (int)err;
    err = hipMalloc((void**)&d_offsets, (size_t)n_offsets * sizeof(int));
    if (err != hipSuccess) { tria_hip_free_ptr(d_key_pos); return (int)err; }
    err = hipMalloc((void**)&d_scores, (size_t)n_tokens * sizeof(float));
    if (err != hipSuccess) { tria_hip_free_ptr(d_key_pos); tria_hip_free_ptr(d_offsets); return (int)err; }

    err = hipMemcpy(d_key_pos, key_pos, (size_t)n_tokens * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        tria_hip_free_ptr(d_key_pos);
        tria_hip_free_ptr(d_offsets);
        tria_hip_free_ptr(d_scores);
        return (int)err;
    }
    err = hipMemcpy(d_offsets, offsets, (size_t)n_offsets * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        tria_hip_free_ptr(d_key_pos);
        tria_hip_free_ptr(d_offsets);
        tria_hip_free_ptr(d_scores);
        return (int)err;
    }

    const int block = 256;
    const int grid  = (n_tokens + block - 1) / block;

    for (int kvi = 0; kvi < n_kv_heads; kvi++) {
        /* Pass 1: raw scores for this KV head */
        hipLaunchKernelGGL(tria_score_q8_0_kernel,
            dim3(grid), dim3(block), 0, 0,
            (const uint8_t *)k_data_dev,
            score_start, n_tokens, n_embd_k_gqa,
            kvi, head_dim, freq_count,
            d_key_pos, omega_dev,
            q_mean_real_dev + q_mean_offset + kvi * freq_count,
            q_mean_imag_dev + q_mean_offset + kvi * freq_count,
            d_offsets, n_offsets,
            d_scores);

        err = hipGetLastError();
        if (err != hipSuccess) break;

        /* Pass 2: z-normalize and max-aggregate into global_scores */
        const int smem = block * sizeof(float);
        hipLaunchKernelGGL(tria_znorm_maxagg_kernel,
            dim3(1), dim3(block), smem, 0,
            d_scores, n_tokens, layer_weight, global_scores_dev);

        err = hipGetLastError();
        if (err != hipSuccess) break;
    }

    hipError_t sync_err = hipDeviceSynchronize();
    if (err == hipSuccess) err = sync_err;
    tria_hip_free_ptr(d_key_pos);
    tria_hip_free_ptr(d_offsets);
    tria_hip_free_ptr(d_scores);

    return (int)err;
}
