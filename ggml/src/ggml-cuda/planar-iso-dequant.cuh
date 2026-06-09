/*
 * PlanarQuant / IsoQuant dequantize kernels for KV cache
 * Based on: https://github.com/scrya-com/rotorquant
 *
 * Implements GGML_TYPE_PLANAR3_0 (planar3_0) and GGML_TYPE_ISO3_0 (iso3_0)
 * dequantize device functions and CUDA kernel wrappers.
 *
 * PlanarQuant: 2D Givens rotations (SO(2)), 64 groups for d=128
 * IsoQuant: 4D quaternion rotations (SO(4)), 32 groups for d=128
 *
 * Block format: 50 bytes per 128 elements
 *   - 2B norm (fp16)
 *   - 32B quantized indices (3-bit Lloyd-Max with split low2/high1)
 *   - 16B signs (1-bit per element)
 *
 * NOTE: Guarded by QK_PLANAR3 — types not yet present in the codebase.
 * See https://github.com/scrya-com/rotorquant for upstream type definitions.
 */

#pragma once

#include "common.cuh"

#ifndef QK_PLANAR3
// Not ready for compilation — upstream rotorquant types not yet integrated
#else

// ---- Quantization ratios for dequantize_block template ----
#define QR_PLANAR3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_ISO3 1     // Each dequantize call produces 2 consecutive elements (like q8_0)

// ============================================================================
// PlanarQuant 3-bit (planar3_0)
// 2D Givens rotations, 64 groups for d=128
// ============================================================================

// 3-bit centroids (Lloyd-Max for N(0,1))
static __constant__ float PLANAR3_CENTROIDS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __constant__ float PLANAR3_MID[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// Rotation params: 64 groups × 2 floats (cos, sin) per 2D pair
static __constant__ float PLANAR3_ROTATION_PARAMS[64 * 2];

// Nearest centroid lookup for planar3
static __device__ __forceinline__ uint8_t planar3_nearest_centroid(float val) {
    if      (val < PLANAR3_MID[0]) return 0;
    else if (val < PLANAR3_MID[1]) return 1;
    else if (val < PLANAR3_MID[2]) return 2;
    else if (val < PLANAR3_MID[3]) return 3;
    else if (val < PLANAR3_MID[4]) return 4;
    else if (val < PLANAR3_MID[5]) return 5;
    else if (val < PLANAR3_MID[6]) return 6;
    else                           return 7;
}

// Dequantize one element from planar3 block
static __device__ __forceinline__ float planar3_dequant_element(
        const block_planar3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return PLANAR3_CENTROIDS[idx] * norm;
}

// Inverse 2D Givens rotation: [c, s; -s, c] * [a, b]
// Forward: c*a - s*b, s*a + c*b
// Inverse: c*a + s*b, -s*a + c*b
static __device__ __forceinline__ void planar3_inverse_givens_2d(
        float & a, float & b, float cos_t, float sin_t) {
    float new_a = cos_t * a + sin_t * b;
    float new_b = -sin_t * a + cos_t * b;
    a = new_a;
    b = new_b;
}

// Dequantize + inverse rotation for planar3_0
// Each warp handles one 128-element group (64 pairs)
static __device__ void planar3_dequant_and_inverse(
        float * __restrict__ dst,
        const block_planar3_0 * __restrict__ x,
        int group_idx,
        int ncols) {
    const int lane = threadIdx.x;
    const int base = group_idx * QK_PLANAR3;
    
    // Get rotation params for this group
    const int cos_sin_base = group_idx * 2;
    float cos_t = PLANAR3_ROTATION_PARAMS[cos_sin_base + 0];
    float sin_t = PLANAR3_ROTATION_PARAMS[cos_sin_base + 1];
    
    // Dequantize 128 elements (32 threads × 4 elements each via unroll)
    float vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j = lane * 4 + i;
        if (j < QK_PLANAR3) {
            vals[i] = planar3_dequant_element(x, j, __half2float(x->norm));
        }
    }
    
    // Apply inverse 2D Givens rotation per pair (64 pairs per group)
    // Each thread handles 2 pairs (4 elements)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int pair = lane * 2 + i;
        if (pair < 64) {
            planar3_inverse_givens_2d(
                vals[i * 2 + 0], vals[i * 2 + 1],
                cos_t, sin_t);
        }
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j = lane * 4 + i;
        if (j < QK_PLANAR3) {
            dst[base + j] = vals[i];
        }
    }
}

// ============================================================================
// IsoQuant 3-bit (iso3_0)
// 4D quaternion rotations (SO(4)), 32 groups for d=128
// ============================================================================

// 3-bit centroids (Lloyd-Max for N(0,1))
static __constant__ float ISO3_CENTROIDS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __constant__ float ISO3_MID[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// Rotation params: 32 groups × 4 floats (quaternion w, x, y, z)
static __constant__ float ISO3_ROTATION_PARAMS[32 * 4];

// Nearest centroid lookup for iso3
static __device__ __forceinline__ uint8_t iso3_nearest_centroid(float val) {
    if      (val < ISO3_MID[0]) return 0;
    else if (val < ISO3_MID[1]) return 1;
    else if (val < ISO3_MID[2]) return 2;
    else if (val < ISO3_MID[3]) return 3;
    else if (val < ISO3_MID[4]) return 4;
    else if (val < ISO3_MID[5]) return 5;
    else if (val < ISO3_MID[6]) return 6;
    else                        return 7;
}

// Dequantize one element from iso3 block
static __device__ __forceinline__ float iso3_dequant_element(
        const block_iso3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return ISO3_CENTROIDS[idx] * norm;
}

// Quaternion conjugate: (w, -x, -y, -z)
static __device__ __forceinline__ void quat_conj(float & w, float & x, float & y, float & z) {
    x = -x;
    y = -y;
    z = -z;
}

// Quaternion × vector (pure quaternion): q * (0, vx, vy, vz)
// Returns result as (r0, r1, r2, r3) where r0 is scalar part (unused)
static __device__ __forceinline__ void quat_vec_mul(
        float q_w, float q_x, float q_y, float q_z,
        float vx, float vy, float vz,
        float & r0, float & r1, float & r2, float & r3) {
    r0 = -q_x*vx - q_y*vy - q_z*vz;
    r1 =  q_w*vx + q_y*vz - q_z*vy;
    r2 =  q_w*vy - q_x*vz + q_z*vx;
    r3 =  q_w*vz + q_x*vy - q_y*vx;
}

// Apply inverse 4D quaternion rotation to 4-element vector
// Inverse: conj(q_L) * v (IsoQuant-Fast mode)
static __device__ __forceinline__ void iso3_inverse_quat_4d(
        float & v0, float & v1, float & v2, float & v3,
        float q_w, float q_x, float q_y, float q_z) {
    float r0, r1, r2, r3;
    quat_vec_mul(q_w, q_x, q_y, q_z, v0, v1, v2, r0, r1, r2, r3);
    v0 = r1;
    v1 = r2;
    v2 = r3;
    // v3 is unused (pure quaternion)
}

// Dequantize + inverse rotation for iso3_0
// Each warp handles one 128-element group (32 quaternions)
static __device__ void iso3_dequant_and_inverse(
        float * __restrict__ dst,
        const block_iso3_0 * __restrict__ x,
        int group_idx,
        int ncols) {
    const int lane = threadIdx.x;
    const int base = group_idx * QK_ISO3;
    
    // Get rotation params for this group
    const int q_base = group_idx * 4;
    float q_w = ISO3_ROTATION_PARAMS[q_base + 0];
    float q_x = ISO3_ROTATION_PARAMS[q_base + 1];
    float q_y = ISO3_ROTATION_PARAMS[q_base + 2];
    float q_z = ISO3_ROTATION_PARAMS[q_base + 3];
    
    // Conjugate for inverse rotation
    quat_conj(q_w, q_x, q_y, q_z);
    
    // Dequantize 128 elements (32 threads × 4 elements each via unroll)
    float vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j = lane * 4 + i;
        if (j < QK_ISO3) {
            vals[i] = iso3_dequant_element(x, j, __half2float(x->norm));
        }
    }
    
    // Apply inverse 4D quaternion rotation per 4D group (32 quaternions per group)
    // Each thread handles 4 quaternions (16 elements)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int quat = lane * 4 + i;
        if (quat < 32) {
            int base_idx = quat * 4;
            iso3_inverse_quat_4d(
                vals[i * 4 + 0], vals[i * 4 + 1], vals[i * 4 + 2], vals[i * 4 + 3],
                q_w, q_x, q_y, q_z);
        }
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int j = lane * 4 + i;
        if (j < QK_ISO3) {
            dst[base + j] = vals[i];
        }
    }
}

// ============================================================================
// Device dequantize functions (matching dequantize_tq4_1s pattern)
// These dequantize a full block and apply inverse rotation,
// then return 2 elements via float2 reference.
// Used by dequantize_block_cont_cuda template.
// ============================================================================

// Planar3_0: dequantize full 128-element block with inverse 2D Givens rotation
static __device__ __forceinline__ void dequantize_planar3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    
    // Step 1: Dequantize full 128-element block
    float buf[128];
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t low2 = (x[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        uint8_t hi1  = (x[ib].signs[j / 8] >> (j % 8)) & 0x1;
        uint8_t idx  = low2 | (hi1 << 2);
        buf[j] = PLANAR3_CENTROIDS[idx] * norm;
    }
    
    // Step 2: Apply inverse 2D Givens rotation per pair (64 pairs per group)
    // Each group has its own rotation params
    const int group_idx = ib;  // ib is the block index within the row
    const int cos_sin_base = group_idx * 2;
    float cos_t = PLANAR3_ROTATION_PARAMS[cos_sin_base + 0];
    float sin_t = PLANAR3_ROTATION_PARAMS[cos_sin_base + 1];
    
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float a = buf[p * 2 + 0];
        float b = buf[p * 2 + 1];
        buf[p * 2 + 0] = cos_t * a + sin_t * b;
        buf[p * 2 + 1] = -sin_t * a + cos_t * b;
    }
    
    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// Iso3_0: dequantize full 128-element block with inverse quaternion rotation
static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    
    // Step 1: Dequantize full 128-element block
    float buf[128];
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t low2 = (x[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        uint8_t hi1  = (x[ib].signs[j / 8] >> (j % 8)) & 0x1;
        uint8_t idx  = low2 | (hi1 << 2);
        buf[j] = ISO3_CENTROIDS[idx] * norm;
    }
    
    // Step 2: Apply inverse 4D quaternion rotation per 4D group (32 quaternions per group)
    const int group_idx = ib;  // ib is the block index within the row
    const int q_base = group_idx * 4;
    float q_w = ISO3_ROTATION_PARAMS[q_base + 0];
    float q_x = ISO3_ROTATION_PARAMS[q_base + 1];
    float q_y = ISO3_ROTATION_PARAMS[q_base + 2];
    float q_z = ISO3_ROTATION_PARAMS[q_base + 3];
    
    // Conjugate for inverse rotation
    q_x = -q_x; q_y = -q_y; q_z = -q_z;
    
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float vx = buf[g * 4 + 0];
        float vy = buf[g * 4 + 1];
        float vz = buf[g * 4 + 2];
        // quat * vec (pure quaternion)
        float r0 = -q_x*vx - q_y*vy - q_z*vz;
        float r1 =  q_w*vx + q_y*vz - q_z*vy;
        float r2 =  q_w*vy - q_x*vz + q_z*vx;
        float r3 =  q_w*vz + q_x*vy - q_y*vx;
        buf[g * 4 + 0] = r1;
        buf[g * 4 + 1] = r2;
        buf[g * 4 + 2] = r3;
        // buf[g*4+3] unused
    }
    
    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// ============================================================================
// Host-side rotation params generation and upload
// Generates rotation params using seed=42 LCG (matching CPU implementation)
// and uploads them to device constant arrays.
// ============================================================================

static void planar3_generate_rotation_params(float * params, int n_groups) {
    // seed=42 LCG: next = (a * seed + c) % m
    // Using parameters from Numerical Recipes: a=1664525, c=1013904245, m=2^32
    uint32_t state = 42;
    for (int g = 0; g < n_groups; g++) {
        // Generate two random floats in [0, 1)
        state = 1664525u * state + 1013904245u;
        float r0 = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        state = 1664525u * state + 1013904245u;
        float r1 = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        
        // Normalize to get cos/sin from random angle in [0, 2*pi]
        float angle = fmodf(r0 * 6.283185307179586f, 6.283185307179586f);
        params[g * 2 + 0] = cosf(angle);
        params[g * 2 + 1] = sinf(angle);
    }
}

static void iso3_generate_rotation_params(float * params, int n_groups) {
    // seed=42 LCG: generates unit quaternions via normalized random 4D vector
    uint32_t state = 42;
    for (int g = 0; g < n_groups; g++) {
        // Generate 4 random floats
        float vals[4];
        for (int i = 0; i < 4; i++) {
            state = 1664525u * state + 1013904245u;
            vals[i] = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        }
        
        // Normalize to unit quaternion
        float norm = sqrtf(vals[0]*vals[0] + vals[1]*vals[1] + vals[2]*vals[2] + vals[3]*vals[3]);
        if (norm > 1e-10f) {
            params[g * 4 + 0] = vals[0] / norm;
            params[g * 4 + 1] = vals[1] / norm;
            params[g * 4 + 2] = vals[2] / norm;
            params[g * 4 + 3] = vals[3] / norm;
        } else {
            params[g * 4 + 0] = 1.0f;
            params[g * 4 + 1] = 0.0f;
            params[g * 4 + 2] = 0.0f;
            params[g * 4 + 3] = 0.0f;
        }
    }
}

// Upload rotation params to device constant arrays
static void planar3_load_rotation_params(void) {
    float params[128];  // 64 groups × 2 floats
    planar3_generate_rotation_params(params, 64);
    cudaMemcpyToSymbol(PLANAR3_ROTATION_PARAMS, params, sizeof(params));
}

static void iso3_load_rotation_params(void) {
    float params[128];  // 32 groups × 4 floats
    iso3_generate_rotation_params(params, 32);
    cudaMemcpyToSymbol(ISO3_ROTATION_PARAMS, params, sizeof(params));
}

#endif // QK_PLANAR3
