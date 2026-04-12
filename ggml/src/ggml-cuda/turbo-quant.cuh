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

// ---- Sign arrays for GROUP_SIZE=256 (Gemma 4 D=256 SWA, D=512 as 2×256 blocks) ----
// Generated with deterministic seeds (256001, 256002)
static __constant__ float TURBO_WHT_SIGNS1_256[256] = {
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
     1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
    -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,
     1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,
     1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f,
     1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,
     1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
     1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,
     1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
     1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,
     1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
     1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,
};
static __constant__ float TURBO_WHT_SIGNS2_256[256] = {
     1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,
     1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
     1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,
     1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,
     1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
     1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
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
static float innerq_strength     = 0.15f;
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
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.15f;

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

    // RPN: merge RoPE pairs to shared RMS before computing scales.
    // RoPE rotates pairs together — independent scales would deform
    // the pair circle into an ellipse, causing phase-dependent distortion.
    // Pair mapping depends on rope_type:
    //   NORM:  (2i, 2i+1)
    //   NEOX/MROPE/IMROPE: (i, i + n_rot/2) for i < n_rot/2
    // Only applied to K cache (V is not RoPE'd).
    if (g_rpn_config.is_key && g_rpn_config.n_rot >= 2) {
        const int n_rot = (g_rpn_config.n_rot <= group_size) ? g_rpn_config.n_rot : group_size;
        const int rtype = g_rpn_config.rope_type;
        // LLAMA_ROPE_TYPE_NORM == 2
        const bool is_norm = (rtype == 2);
        if (is_norm) {
            // NORM: pairs are (2i, 2i+1)
            for (int i = 0; i + 1 < n_rot; i += 2) {
                float pr = sqrtf(0.5f * (rms[i]*rms[i] + rms[i+1]*rms[i+1]));
                rms[i] = pr; rms[i+1] = pr;
            }
        } else {
            // NEOX/MROPE/IMROPE: pairs are (i, i + n_rot/2)
            const int half = n_rot / 2;
            for (int i = 0; i < half; i++) {
                float pr = sqrtf(0.5f * (rms[i]*rms[i] + rms[i+half]*rms[i+half]));
                rms[i] = pr; rms[i+half] = pr;
            }
        }
        GGML_LOG_INFO("%s: RPN pair merge applied (rope_type=%d, n_rot=%d, mode=%s)\n",
                       __func__, rtype, n_rot, is_norm ? "norm" : "neox");
    }

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
    (void)cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
    (void)cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float));
    (void)cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float));
    cudaDeviceSynchronize();  // ensure scales are visible before activating
    (void)cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int));

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
    // InnerQ only works when group_size == INNERQ_MAX_CHANNELS (128).
    // Reject group_size > 128 (Gemma 4 D=256) and < 128 (multi-group-per-head).
    // Codex review P0: group_size=256 overflows 128-entry device symbols and stack arrays.
    const bool incompatible_group = (group_size != INNERQ_MAX_CHANNELS);
    if (incompatible_group) {
        if (innerq_enabled >= 1) {
            GGML_LOG_WARN("%s: InnerQ disabled (group_size=%d != %d, incompatible)\n",
                           __func__, group_size, INNERQ_MAX_CHANNELS);
            innerq_enabled = 0;
            int zero = 0;
            (void)cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
            (void)cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int));
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

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- Nearest 4-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

// ---- Per-block quantize for turbo4 (128 elements, expects already-rotated input) ----

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

// ---- Nearest 3-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Per-block quantize (32 elements, expects already-rotated input) ----
// Used by set_rows after group-level WHT rotation

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

// ---- Nearest 2-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Per-block quantize for turbo2 (32 elements, expects already-rotated input) ----

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

// 2D VQ codebook: 64 entries, trained on actual Qwen3-8B WHT output pairs
static __constant__ float TURBO_VQ2D_X[64] = {
    0.0279071f, -0.1041781f, -0.0497183f, 0.0836585f, 0.0755566f, -0.1593080f, -0.0472192f, 0.1499346f,
    -0.0259202f, -0.0749334f, -0.1060147f, -0.1302685f, 0.0510575f, 0.0321239f, 0.0427720f, 0.2017132f,
    -0.0174130f, 0.0938271f, 0.1514418f, -0.1524931f, -0.0659325f, -0.1347785f, 0.1569419f, 0.0335782f,
    0.2139767f, 0.0298571f, 0.1024047f, -0.1463255f, -0.0380896f, -0.1880937f, 0.1287539f, -0.0810642f,
    -0.0230893f, -0.0325119f, -0.0495625f, 0.0664514f, 0.1864402f, 0.0794077f, -0.2225531f, 0.0198063f,
    -0.0478895f, 0.1485750f, 0.0846328f, 0.0470138f, 0.0562434f, -0.1950971f, 0.0961574f, -0.0095595f,
    -0.0900242f, -0.0080224f, -0.0094565f, -0.1106773f, -0.0637866f, -0.1312685f, 0.0118203f, 0.0150917f,
    0.1209811f, -0.0833506f, -0.1212273f, 0.0995258f, -0.0725997f, 0.1161496f, 0.0609390f, -0.0160979f,
};
static __constant__ float TURBO_VQ2D_Y[64] = {
    -0.0263300f, 0.0685406f, -0.1090837f, 0.1035094f, -0.0896168f, -0.0125089f, -0.0671406f, -0.0187005f,
    0.0717508f, 0.1467829f, -0.0184862f, -0.1144251f, 0.0793044f, -0.1656622f, 0.1358503f, 0.0923961f,
    -0.0055588f, 0.0639664f, -0.1557487f, 0.0507863f, 0.0079050f, 0.1759942f, 0.1642957f, -0.1138678f,
    0.0008668f, -0.0694018f, 0.0207315f, -0.1742128f, -0.2115104f, 0.1064816f, 0.1005936f, -0.1476948f,
    0.0305579f, 0.1157162f, -0.0274784f, -0.0479926f, -0.0830491f, -0.2145961f, 0.0121470f, 0.0135110f,
    0.2169799f, 0.0451792f, -0.1392938f, 0.2095770f, 0.0380025f, -0.0849720f, 0.1537713f, -0.0449162f,
    -0.0923609f, 0.1603929f, -0.0886196f, 0.0234268f, 0.0493861f, -0.0623466f, 0.1004377f, 0.0549886f,
    -0.1051118f, -0.0522899f, 0.1113740f, -0.0216251f, 0.0940127f, -0.0629645f, -0.0015928f, -0.1436934f,
};

// 2D VQ dequant pair helper
static __device__ __forceinline__ float2 turbo3_dequant_pair(
        const block_turbo3_0 * __restrict__ x, int j_even, float norm) {
    uint8_t low2_0 = (x->qs[j_even / 4] >> ((j_even % 4) * 2)) & 0x3;
    uint8_t hi1_0  = (x->signs[j_even / 8] >> (j_even % 8)) & 0x1;
    uint8_t idx0   = low2_0 | (hi1_0 << 2);
    int j_odd = j_even + 1;
    uint8_t low2_1 = (x->qs[j_odd / 4] >> ((j_odd % 4) * 2)) & 0x3;
    uint8_t hi1_1  = (x->signs[j_odd / 8] >> (j_odd % 8)) & 0x1;
    uint8_t idx1   = low2_1 | (hi1_1 << 2);
    uint8_t vq = (idx0 << 3) | idx1;
    return make_float2(TURBO_VQ2D_X[vq] * norm, TURBO_VQ2D_Y[vq] * norm);
}
