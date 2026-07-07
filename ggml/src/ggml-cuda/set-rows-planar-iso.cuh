#pragma once

// Device quantize functions for planar3/iso3/planar4/iso4 set_rows.
// These wrap the rotation + quantize logic into the __device__ function
// signature required by k_set_rows_quant.
//
// Uses static __constant__ arrays from planar-iso-constants.cuh so each
// compilation unit gets its own initialized copy — no cross-TU extern needed.

#include "ggml-common.h"
#include "planar-iso-constants.cuh"
#include "planar-iso-dequant.cuh"
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <cmath>

// Init function from cpy-planar-iso.cu (still needed for cpy path)
extern void ggml_cuda_init_planar_iso_constants();

// ── Helpers ─────────────────────────────────────────────────────────

__device__ __forceinline__ uint8_t sr_quantize_3bit(float val, const float * mid) {
    uint8_t idx = 0;
    if      (val < mid[0]) idx = 0;
    else if (val < mid[1]) idx = 1;
    else if (val < mid[2]) idx = 2;
    else if (val < mid[3]) idx = 3;
    else if (val < mid[4]) idx = 4;
    else if (val < mid[5]) idx = 5;
    else if (val < mid[6]) idx = 6;
    else                   idx = 7;
    return idx;
}

__device__ __forceinline__ uint8_t sr_quantize_4bit(float val, const float * centroids) {
    uint8_t best = 0;
    float best_d = fabsf(val - centroids[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float d = fabsf(val - centroids[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// ── Planar3: F32[128] → block_planar3_0 ─────────────────────────────

__device__ void quantize_f32_planar3_block(const float * x, block_planar3_0 * dst) {
    // Norm
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    // Forward Givens rotation
    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], s = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1] = s * buf[p*2] + c * buf[p*2+1];
    }

    // Quantize + pack
    for (int j = 0; j < QK_PLANAR3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = sr_quantize_3bit(rotated[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Iso3: F32[128] → block_iso3_0 (quaternion rotation) ────────────

__device__ void quantize_f32_iso3_block(const float * x, block_iso3_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    // Forward quaternion rotation: q_L * v * conj(q_R) per 4D group.
    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float q_L[4] = {PI_QW[g], PI_QX[g], PI_QY[g], PI_QZ[g]};
        float q_R[4] = {PI_QW_R[g], PI_QX_R[g], PI_QY_R[g], PI_QZ_R[g]};
        float v[4]   = {buf[g*4], buf[g*4+1], buf[g*4+2], buf[g*4+3]};
        float tmp[4], result[4];
        quat_mul(tmp, q_L, v);
        quat_conj(q_R, q_R);
        quat_mul(result, tmp, q_R);
        rotated[g*4]   = result[0];
        rotated[g*4+1] = result[1];
        rotated[g*4+2] = result[2];
        rotated[g*4+3] = result[3];
    }

    for (int j = 0; j < QK_ISO3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_ISO3/8; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = sr_quantize_3bit(rotated[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Planar4: F32[128] → block_planar4_0 (Givens + 4-bit nibble) ────
// 4-bit types (block_planar4_0, block_iso4_0, QK_PLANAR4, QK_ISO4)
// are not yet defined in ggml-common.h. Guard until they are added.
#if 0
__device__ void quantize_f32_planar4_block(const float * x, block_planar4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR4; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv_norm;

    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], s = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1] = s * buf[p*2] + c * buf[p*2+1];
    }

    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(rotated[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

// ── Iso4: F32[128] → block_iso4_0 (quaternion + 4-bit nibble) ──────

__device__ void quantize_f32_iso4_block(const float * x, block_iso4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_ISO4; j++) {
        buf[j] = x[j];
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO4; j++) buf[j] *= inv_norm;

    // Forward quaternion rotation: q_L * v * conj(q_R) per 4D group.
    float rotated[128];
    #pragma unroll
    for (int g = 0; g < 32; g++) {
        float q_L[4] = {PI_QW[g], PI_QX[g], PI_QY[g], PI_QZ[g]};
        float q_R[4] = {PI_QW_R[g], PI_QX_R[g], PI_QY_R[g], PI_QZ_R[g]};
        float v[4]   = {buf[g*4], buf[g*4+1], buf[g*4+2], buf[g*4+3]};
        float tmp[4], result[4];
        quat_mul(tmp, q_L, v);
        quat_conj(q_R, q_R);
        quat_mul(result, tmp, q_R);
        rotated[g*4]   = result[0];
        rotated[g*4+1] = result[1];
        rotated[g*4+2] = result[2];
        rotated[g*4+3] = result[3];
    }

    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(rotated[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

#endif // 0 — end 4-bit guard; V-cache 3-bit norot functions below

// ══════════════════════════════════════════════════════════════════════
// V-cache variants: NO ROTATION (for transposed V cache)
// ══════════════════════════════════════════════════════════════════════

// Verify iso3/planar3 share the same packed layout (cast in _norot variants).
static_assert(sizeof(block_iso3_0) == sizeof(block_planar3_0), "iso3/planar3 block size mismatch");
// 4-bit types not yet defined — guard until planar4/iso4 are added.
#if 0
static_assert(sizeof(block_iso4_0) == sizeof(block_planar4_0), "iso4/planar4 block size mismatch");
#endif

__device__ void quantize_f32_planar3_block_norot(const float * x, block_planar3_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR3; j++) { buf[j] = x[j]; norm_sq += buf[j]*buf[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv;
    for (int j = 0; j < QK_PLANAR3/4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) dst->signs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = sr_quantize_3bit(buf[j], PI_MID_3BIT);
        dst->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) dst->signs[j/8] |= (1 << (j%8));
        recon_sq += PI_CENTROIDS_3BIT[idx] * PI_CENTROIDS_3BIT[idx];
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half(rn > 1e-10f ? grp_norm / rn : grp_norm);
}

__device__ void quantize_f32_iso3_block_norot(const float * x, block_iso3_0 * dst) {
    quantize_f32_planar3_block_norot(x, (block_planar3_0 *)dst);
}

// 4-bit norot variants — guarded until block_planar4_0/block_iso4_0 are defined
#if 0
__device__ void quantize_f32_planar4_block_norot(const float * x, block_planar4_0 * dst) {
    float norm_sq = 0.0f;
    float buf[128];
    for (int j = 0; j < QK_PLANAR4; j++) { buf[j] = x[j]; norm_sq += buf[j]*buf[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv;
    for (int j = 0; j < 64; j++) dst->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = sr_quantize_4bit(buf[j], PI_CENTROIDS_4BIT);
        dst->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += PI_CENTROIDS_4BIT[idx] * PI_CENTROIDS_4BIT[idx];
    }
    float rn = sqrtf(recon_sq);
    dst->norm = __float2half(rn > 1e-10f ? grp_norm / rn : grp_norm);
    dst->rnorm = __float2half(0.0f);
}

__device__ void quantize_f32_iso4_block_norot(const float * x, block_iso4_0 * dst) {
    quantize_f32_planar4_block_norot(x, (block_planar4_0 *)dst);
}
#endif // 0 — 4-bit types not yet defined
