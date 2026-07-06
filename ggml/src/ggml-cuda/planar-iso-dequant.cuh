/*
 * PlanarQuant / IsoQuant dequantize kernels for KV cache.
 * Dequant returns centroids x norm with per-pair/per-block inverse rotation
 * applied (restores original direction vector before orthogonal transform).
 */

#pragma once

#include "common.cuh"
#include "planar-iso-constants.cuh"

// ============================================================================
// Rotation helpers (Givens + Quaternion)
// Uses precomputed constants from planar-iso-constants.cuh
// (Python torch.manual_seed(42), matching the paper reference).
// ============================================================================

static __device__ __forceinline__ void givens_forward(float &a, float &b, float c, float s) {
    float an = a*c - b*s, bn = a*s + b*c;
    a = an; b = bn;
}

static __device__ __forceinline__ void givens_inverse(float &a, float &b, float c, float s) {
    float an = a*c + b*s, bn = -a*s + b*c;
    a = an; b = bn;
}

static __device__ __forceinline__ void quat_mul(float r[4], const float p[4], const float q[4]) {
    r[0]=p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3];
    r[1]=p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2];
    r[2]=p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1];
    r[3]=p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0];
}

static __device__ __forceinline__ void quat_conj(float r[4], const float q[4]) {
    r[0]=q[0]; r[1]=-q[1]; r[2]=-q[2]; r[3]=-q[3];
}

// Planar: precomputed cos/sin from Python (seed=42)
static __device__ __forceinline__ void planar3_get_rotation(int pair_idx, float &c, float &s) {
    c = PI_COS[pair_idx % 64];
    s = PI_SIN[pair_idx % 64];
}

// Iso: precomputed unit quaternions from Python (seed=42)
static __device__ __forceinline__ void iso3_get_rotation(int block_idx, float q_L[4], float q_R[4]) {
    int idx = block_idx % 32;
    // q_L from left-isoclinic array
    q_L[0] = PI_QW[idx]; q_L[1] = PI_QX[idx]; q_L[2] = PI_QY[idx]; q_L[3] = PI_QZ[idx];
    // q_R from RIGHT-isoclinic array (INDEPENDENT values!)
    q_R[0] = PI_QW_R[idx]; q_R[1] = PI_QX_R[idx]; q_R[2] = PI_QY_R[idx]; q_R[3] = PI_QZ_R[idx];
}

// ---- Quantization ratios for dequantize_block template ----
#define QR_PLANAR3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_ISO3 1     // Each dequantize call produces 2 consecutive elements (like q8_0)

// ============================================================================
// PlanarQuant 3-bit (planar3_0)
// 2-element pairs, 64 groups for d=128
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

// ============================================================================
// IsoQuant 3-bit (iso3_0)
// 4-element quaternion blocks for d=128
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

// ============================================================================
// Device dequantize functions (matching dequantize_tq4_1s pattern)
// Each call dequantizes 2 consecutive elements with inverse rotation applied.
// Quantizer (set_rows) applies forward rotation; VEC FA applies inverse.
// ============================================================================

static __device__ __forceinline__ void dequantize_planar3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // Dequantize target pair at (iqs, iqs+1) — inverse Givens rotation.
    uint8_t low0 = (x[ib].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x3;
    uint8_t hi0  = (x[ib].signs[iqs / 8] >> (iqs % 8)) & 0x1;
    uint8_t idx0 = low0 | (hi0 << 2);

    int iqs1 = iqs + 1;
    uint8_t low1 = (x[ib].qs[iqs1 / 4] >> ((iqs1 % 4) * 2)) & 0x3;
    uint8_t hi1  = (x[ib].signs[iqs1 / 8] >> (iqs1 % 8)) & 0x1;
    uint8_t idx1 = low1 | (hi1 << 2);

    float kv0 = PLANAR3_CENTROIDS[idx0] * norm;
    float kv1 = PLANAR3_CENTROIDS[idx1] * norm;

    // Inverse Givens rotation — pair_index = iqs / 2.
    float c, s;
    planar3_get_rotation(iqs / 2, c, s);
    givens_inverse(kv0, kv1, c, s);

    v.x = kv0;
    v.y = kv1;
}

static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // Align to quaternion block boundary (4 elements).  Since the float2
    // interface processes 2 elements per call (QR_ISO3=1) but the inverse
    // quaternion rotation needs all 4, we load all 4, apply the full 4x4
    // rotation, and return the pair at [offset..offset+1].
    const int base   = (iqs / 4) * 4;
    const int offset = iqs - base;  // 0 or 2

    // Load ALL 4 centroid values from the quaternion block.
    float kv[4];
    for (int k = 0; k < 4; k++) {
        const int j = base + k;
        const uint8_t low2 = (x[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j / 8] >> (j % 8)) & 0x1;
        const uint8_t idx  = low2 | (hi1 << 2);
        kv[k] = ISO3_CENTROIDS[idx] * norm;
    }

    // Full inverse quaternion rotation: conj(q_L) * kv * q_R
    float q_L[4], q_R[4];
    iso3_get_rotation(base / 4, q_L, q_R);
    float conj_L[4], tmp[4], result[4];
    quat_conj(conj_L, q_L);
    quat_mul(tmp, conj_L, kv);
    quat_mul(result, tmp, q_R);

    v.x = result[offset];
    v.y = result[offset + 1];
}

// ============================================================================
// Host-side loading: generate rotation params matching GPU k_set_rows LCG
// LCG: state = 1664525u * state + 1013904245u, seed = 42.
// ============================================================================

// No-ops: rotation params generated on-device via LCG (seed 42 base).
static void planar3_load_rotation_params(void) {}
static void iso3_load_rotation_params(void) {}
