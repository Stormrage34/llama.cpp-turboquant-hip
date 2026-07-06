/*
 * TurboQuant CUDA kernels for KV cache compression
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit PolarQuant, block size 32)
 * Constants, WHT rotation, quantize/dequantize device functions.
 */

#pragma once

#include "common.cuh"
#include "turbo-innerq.cuh"
#include <cstdlib>
#include <cmath>

// HIP compatibility shims
#ifdef __HIP_PLATFORM_AMD__
#ifndef cudaMemcpyToSymbol
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#endif
#ifndef cudaMemcpyFromSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#endif
#endif

// ---- Quantization ratios for dequantize_block template ----
#define QR_TURBO3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO2 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO4 1  // Each dequantize call produces 2 consecutive elements (like q8_0)

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- WHT sign arrays (seed=42) ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- 64-element WHT sign arrays (first 64 of the 128-element arrays) ----

static __constant__ float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- Fast Walsh-Hadamard Transform (in-place, normalized) ----
// O(n log n) = 896 ops for n=128

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// ---- Fast Walsh-Hadamard Transform for 64-element groups ----
// O(n log n) = 384 ops for n=64

static __device__ __forceinline__ void turbo_fwht_64(float * x) {
    for (int h = 1; h < 64; h *= 2) {
        for (int i = 0; i < 64; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_64 = 0.125f;
    for (int i = 0; i < 64; i++) {
        x[i] *= inv_sqrt_64;
    }
}

// ---- Forward rotation: signs1 → FWHT → signs2 ----

static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// ---- Inverse rotation: signs2 → FWHT → signs1 (FWHT is self-inverse up to norm) ----

static __device__ __forceinline__ void turbo_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
}

// ---- Inverse rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_inverse_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
}

// ---- Forward rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_forward_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
}

// ---- InnerQ per-channel equalization ----
// Equalizes K channel variances before WHT rotation to reduce quantization error.
// Enabled via TURBO_INNERQ=N env var (N = calibration token count).
// Math: <Q/s, s*K> = <Q, K> preserves dot products.
// INNERQ_MAX_CHANNELS is defined in turbo-innerq.cuh

static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
 static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
 static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
 static __device__ int   d_innerq_count;
 static __device__ int   d_innerq_active;       // 0 = scales are identity, 1 = scales applied
 static __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

static int  innerq_enabled       = 0;  // host: 0=off, 1=calibrating, 2=active
static int  innerq_target_tokens = 0;
static float innerq_strength     = 0.5f;
static bool  innerq_initialized  = false;

// Host: read TURBO_INNERQ env, start calibration if enabled
static void turbo_innerq_init(void) {
    if (innerq_initialized) return;
    innerq_initialized = true;

    const char * env = getenv("TURBO_INNERQ");
    if (!env || atoi(env) <= 0) {
        innerq_enabled = 0;
        return;
    }
    innerq_target_tokens = atoi(env);
    innerq_enabled = 1;  // calibrating

    const char * env_str = getenv("TURBO_INNERQ_STRENGTH");
    if (env_str) innerq_strength = atof(env_str);
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.5f;

    // Zero accumulators and set calibrating flag on device
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    int zero = 0, one = 1;
    (void)cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros));
    (void)cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int));
    (void)cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int));
    (void)cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int));

    GGML_LOG_INFO("%s: InnerQ calibration started (target=%d tokens, strength=%.2f)\n",
                   __func__, innerq_target_tokens, innerq_strength);
}

// Host: finalize calibration — compute scales, upload, activate
static void turbo_innerq_finalize(int group_size) {
    // Read accumulators from device
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    (void)cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float));
    (void)cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));

    if (count <= 0) {
        GGML_LOG_WARN("%s: InnerQ calibration got 0 tokens, disabling\n", __func__);
        innerq_enabled = 0;
        int zero = 0;
        (void)cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    // Compute per-channel RMS
    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    // Compute scale[i] = (mean_rms / channel_rms[i])^strength, clamp to [0.5, 2.0]
    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, innerq_strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    // Auto-skip if max channel ratio < 1.2 (already balanced)
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        GGML_LOG_INFO("%s: InnerQ auto-disabled (channels already balanced, max_ratio=%.3f)\n",
                       __func__, max_ratio);
        innerq_enabled = 0;
        int zero = 0;
        (void)cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    // Stop calibrating, upload scales, activate
    int zero = 0, one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());  // ensure scales are visible before activating
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int)));

    innerq_enabled = 2;  // active

    // Publish scale_inv to shared host state for cross-TU tensor update
    turbo_innerq_publish(scale_inv, group_size);

    GGML_LOG_INFO("%s: InnerQ finalized (%d tokens, max_ratio=%.3f, min_ratio=%.3f)\n",
                   __func__, count, max_ratio, min_ratio);
}

// Host: called before each set_rows kernel launch
static void turbo_innerq_check_finalize(int group_size, int64_t ne00) {
    if (!innerq_initialized) {
        turbo_innerq_init();
    }
    if (innerq_enabled == 0) return;

    // InnerQ only works when each WHT group = one head (group_size == head_dim).
    // For standard models: ne00 = n_heads * head_dim, group_size = head_dim → ne00 % group_size == 0, fine.
    // For non-standard models (head_dim > group_size, e.g. GLM 576 → 64-group):
    //   ne00 = head_dim (single head), group_size = 64, ne00/group_size = 9 groups per head → WRONG.
    // Detect: if ne00 / group_size doesn't divide evenly into standard head counts (1,2,4,8,16,32,64,128),
    // it's likely multi-group-per-head. Simpler check: group_size < 128 means head_dim > 128.
    const bool multi_group_per_head = (group_size < 128);  // 64-group → head_dim > 128, multi-group
    if (multi_group_per_head) {
        if (innerq_enabled == 1) {
            GGML_LOG_WARN("%s: InnerQ disabled (ne00=%lld != group_size=%d, multi-group heads)\n",
                           __func__, (long long)ne00, group_size);
            innerq_enabled = 0;
            int zero = 0;
            (void)cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        }
        return;
    }

    // Check if calibration is complete
    if (innerq_enabled == 1) {
        int count = 0;
        (void)cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));
        if (count >= innerq_target_tokens) {
            turbo_innerq_finalize(group_size);
        }
    }
}

// Host: check if InnerQ is currently active (finalized)
static bool turbo_innerq_is_active(void) {
    return innerq_enabled == 2;
}

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// NOTE: _H variants omitted — __float2half() cannot initialize __constant__ on HIP.
// Use __float2half(TURBO_CENTROIDS_4BIT[idx]) at call site instead.
#if 0
// __float2half cannot initialize __constant__ on HIP — use TURBO_CENTROIDS_4BIT + __float2half at runtime
static __constant__ const __half TURBO_CENTROIDS_4BIT_H[16] = {
    __float2half(-0.173926f), __float2half(-0.117195f),
    __float2half(-0.089527f), __float2half(-0.068756f),
    __float2half(-0.051262f), __float2half(-0.035597f),
    __float2half(-0.020989f), __float2half(-0.006938f),
    __float2half( 0.006938f), __float2half( 0.020989f),
    __float2half( 0.035597f), __float2half( 0.051262f),
    __float2half( 0.068756f), __float2half( 0.089527f),
    __float2half( 0.117195f), __float2half( 0.173926f),
};
#endif

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- Nearest 4-bit centroid index (branchless binary search) ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    // Level 1: compare with mid7 (index 7, value 0.0f)
    int left_half = (val < TURBO_MID_4BIT[7]) ? 1 : 0;
    
    // Level 2: compare with mid3 if left_half, else with mid11
    int cmp_L2L = (val < TURBO_MID_4BIT[3]) ? 1 : 0;
    int cmp_L2R = (val < TURBO_MID_4BIT[11]) ? 1 : 0;
    int cmp_L2 = left_half ? cmp_L2L : cmp_L2R;
    
    // Level 3: compare with appropriate midpoint
    int cmp_L3LL = (val < TURBO_MID_4BIT[1]) ? 1 : 0;
    int cmp_L3LR = (val < TURBO_MID_4BIT[5]) ? 1 : 0;
    int cmp_L3RL = (val < TURBO_MID_4BIT[9]) ? 1 : 0;
    int cmp_L3RR = (val < TURBO_MID_4BIT[13]) ? 1 : 0;
    
    int cmp_L3 = left_half ? (cmp_L2 ? cmp_L3LL : cmp_L3LR) : (cmp_L2 ? cmp_L3RL : cmp_L3RR);
    
    // Level 4: compare with leaf midpoints
    int cmp_L4LLL = (val < TURBO_MID_4BIT[0]) ? 1 : 0;
    int cmp_L4LLR = (val < TURBO_MID_4BIT[2]) ? 1 : 0;
    int cmp_L4LRL = (val < TURBO_MID_4BIT[4]) ? 1 : 0;
    int cmp_L4LRR = (val < TURBO_MID_4BIT[6]) ? 1 : 0;
    int cmp_L4RLL = (val < TURBO_MID_4BIT[8]) ? 1 : 0;
    int cmp_L4RLR = (val < TURBO_MID_4BIT[10]) ? 1 : 0;
    int cmp_L4RRL = (val < TURBO_MID_4BIT[12]) ? 1 : 0;
    int cmp_L4RRR = (val < TURBO_MID_4BIT[14]) ? 1 : 0;
    
    int cmp_L4 = left_half ? (cmp_L2 ? (cmp_L3 ? cmp_L4LLL : cmp_L4LLR) : (cmp_L3 ? cmp_L4LRL : cmp_L4LRR))
                           : (cmp_L2 ? (cmp_L3 ? cmp_L4RLL : cmp_L4RLR) : (cmp_L3 ? cmp_L4RRL : cmp_L4RRR));
    
    int bits = (left_half << 3) | (cmp_L2 << 2) | (cmp_L3 << 1) | cmp_L4;
    return (uint8_t)(bits ^ 15);
}

// ---- Per-block quantize for turbo4 (128 elements, expects unrotated input) ----

static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src,
                                                    block_turbo4_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j]);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
}

// ---- Inline dequant helper: extract one float from turbo4 block ----

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- Nearest 3-bit centroid index (branchless binary search) ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    // Level 1: compare with mid3 (index 3, value 0.0f)
    int left = (val < TURBO_MID_3BIT[3]) ? 1 : 0;
    
    // Level 2a: compare with mid1 (index 1) if left, else with mid5 (index 5)
    int cmp_left = (val < TURBO_MID_3BIT[1]) ? 1 : 0;
    int cmp_right = (val < TURBO_MID_3BIT[5]) ? 1 : 0;
    int cmp_l2 = left ? cmp_left : cmp_right;
    
    // Level 3: compare with appropriate leaf midpoint
    int cmp_ll = (val < TURBO_MID_3BIT[0]) ? 1 : 0;
    int cmp_lr = (val < TURBO_MID_3BIT[2]) ? 1 : 0;
    int cmp_rl = (val < TURBO_MID_3BIT[4]) ? 1 : 0;
    int cmp_rr = (val < TURBO_MID_3BIT[6]) ? 1 : 0;
    
    int cmp_l3 = left ? (cmp_l2 ? cmp_ll : cmp_lr) : (cmp_l2 ? cmp_rl : cmp_rr);
    
    int bits = (left << 2) | (cmp_l2 << 1) | cmp_l3;
    return (uint8_t)(bits ^ 7);
}

// ---- Per-block quantize (32 elements, expects unrotated input) ----

static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ src,
                                                    block_turbo3_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst->signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// ---- Inline dequant helper: extract one float from turbo3 block ----

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

// ---- Nearest 2-bit centroid index (branchless binary search) ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    // Level 1: compare with mid1 (index 1, value 0.0f)
    int left = (val < TURBO_MID_2BIT[1]) ? 1 : 0;
    
    // Level 2: compare with appropriate leaf midpoint
    int cmp_left = (val < TURBO_MID_2BIT[0]) ? 1 : 0;
    int cmp_right = (val < TURBO_MID_2BIT[2]) ? 1 : 0;
    
    int cmp_l2 = left ? cmp_left : cmp_right;
    
    // Map (left, cmp_l2) to index:
    // (1, 1) -> val < m1 and val < m0 -> 0
    // (1, 0) -> val < m1 and val >= m0 -> 1
    // (0, 1) -> val >= m1 and val < m2 -> 2
    // (0, 0) -> val >= m1 and val >= m2 -> 3
    
    int bits = (left << 1) | cmp_l2;
    return (uint8_t)(bits ^ 3);
}

// ---- Per-block quantize for turbo2 (32 elements, expects unrotated input) ----

static __device__ void quantize_f32_turbo2_0_block(const float * __restrict__ src,
                                                    block_turbo2_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from turbo2 block ----

static __device__ __forceinline__ float turbo2_dequant_element(
        const block_turbo2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

// ============================================================================
// Dequantize kernels (dequantize_kernel_t signature) for KV-cache turbo types
// ============================================================================

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const __half2 norm_h2 = __half2half2(x[ib].norm);
    uint8_t low0 = (x[ib].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x3;
    uint8_t hi0  = (x[ib].signs[iqs / 8] >> (iqs % 8)) & 0x1;
    uint8_t idx0 = low0 | (hi0 << 2);
    int iqs1 = iqs + 1;
    uint8_t low1 = (x[ib].qs[iqs1 / 4] >> ((iqs1 % 4) * 2)) & 0x3;
    uint8_t hi1  = (x[ib].signs[iqs1 / 8] >> (iqs1 % 8)) & 0x1;
    uint8_t idx1 = low1 | (hi1 << 2);
    const __half2 cent_h2 = make_half2(__float2half(TURBO_CENTROIDS_3BIT[idx0]), __float2half(TURBO_CENTROIDS_3BIT[idx1]));
    const __half2 prod    = __hmul2(norm_h2, cent_h2);
    v.x = __half2float(prod.x);
    v.y = __half2float(prod.y);
}

static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const __half2 norm_h2 = __half2half2(x[ib].norm);
    uint8_t idx0 = (x[ib].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0xF;
    int iqs1 = iqs + 1;
    uint8_t idx1 = (x[ib].qs[iqs1 / 2] >> ((iqs1 % 2) * 4)) & 0xF;
    const __half2 cent_h2 = make_half2(__float2half(TURBO_CENTROIDS_4BIT[idx0]), __float2half(TURBO_CENTROIDS_4BIT[idx1]));
    const __half2 prod    = __hmul2(norm_h2, cent_h2);
    v.x = __half2float(prod.x);
    v.y = __half2float(prod.y);
}

static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const __half2 norm_h2 = __half2half2(x[ib].norm);
    uint8_t idx0 = (x[ib].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x3;
    int iqs1 = iqs + 1;
    uint8_t idx1 = (x[ib].qs[iqs1 / 4] >> ((iqs1 % 4) * 2)) & 0x3;
    const __half2 cent_h2 = make_half2(__float2half(TURBO_CENTROIDS_2BIT[idx0]), __float2half(TURBO_CENTROIDS_2BIT[idx1]));
    const __half2 prod    = __hmul2(norm_h2, cent_h2);
    v.x = __half2float(prod.x);
    v.y = __half2float(prod.y);
}

// ============================================================================
// Weight compression types (TQ3_1S, TQ4_1S)
// These use N(0,1) centroids (NOT N(0,1/128) like KV cache types)
// and require inverse WHT (RHT) after centroid lookup.
// ============================================================================

#define QR_TQ4_1S 1  // dequantize produces 2 consecutive elements
#define QR_TQ3_1S 1

// ---- Weight centroids: Lloyd-Max for N(0,1) ----

static __constant__ float TQ4_CENTROIDS_WEIGHT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

static __constant__ float TQ3_CENTROIDS_WEIGHT[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

// ---- Sign array for weight WHT (golden ratio hash, 32 elements) ----

static __constant__ float TQ_WEIGHT_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
};
