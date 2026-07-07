/*
 * PlanarQuant / IsoQuant dequantize kernels for KV cache.
 * Dequant returns centroids x norm with per-pair/per-block inverse rotation
 * applied (restores original direction vector before orthogonal transform).
 *
 * NOTE: This file duplicates constants from planar-iso-constants.cuh because
 * CUDA constant memory cannot be shared via extern across TUs. The init
 * function copies host-side arrays into __constant__ device symbols at runtime.
 */

#pragma once

#include "common.cuh"
#include "planar-iso-constants.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_fp16.h>
#endif
#include <cmath>

// ── Device constants (populated at runtime via cudaMemcpyToSymbol) ──

#ifndef cudaMemcpyToSymbol
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#endif

static __constant__ float d_planar_cos[64];
static __constant__ float d_planar_sin[64];

// Left-isoclinic (q_L)
static __constant__ float d_iso_qw[32];
static __constant__ float d_iso_qx[32];
static __constant__ float d_iso_qy[32];
static __constant__ float d_iso_qz[32];

// Right-isoclinic (q_R)
static __constant__ float d_iso_qw_r[32];
static __constant__ float d_iso_qx_r[32];
static __constant__ float d_iso_qy_r[32];
static __constant__ float d_iso_qz_r[32];

// 3-bit centroids used by copy kernels (duplicated in planar-iso-constants.cuh)
static __constant__ float d_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __constant__ float d_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ── Quantize 3-bit helper ──────────────────────────────────────────

static __device__ __forceinline__ uint8_t quantize_3bit(float val) {
    if      (val < d_mid_3bit[0]) return 0;
    else if (val < d_mid_3bit[1]) return 1;
    else if (val < d_mid_3bit[2]) return 2;
    else if (val < d_mid_3bit[3]) return 3;
    else if (val < d_mid_3bit[4]) return 4;
    else if (val < d_mid_3bit[5]) return 5;
    else if (val < d_mid_3bit[6]) return 6;
    else                          return 7;
}

// ── Planar3: F16 -> block_planar3_0 (2D Givens + 3-bit) ─────────────

__global__ void kernel_cpy_f16_planar3(
    const half * __restrict__ src,
    block_planar3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_PLANAR3;
    block_planar3_0 * blk = &dst[ib];

    // Load and compute norm
    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    // No forward rotation — set-rows stores without rotation (norot path),
    // so the CPY path must match. The VEC FA inline dequant reads centroids
    // directly without inverse rotation.
    float rotated[128];
    memcpy(rotated, buf, sizeof(buf));

    // Quantize + pack (3-bit: 2-bit qs + 1-bit signs)
    for (int j = 0; j < QK_PLANAR3/4; j++) blk->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += d_centroids_3bit[idx] * d_centroids_3bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;
    blk->norm = __float2half(corrected);
}

// ── Iso3: F16 -> block_iso3_0 (quaternion 4D + 3-bit) ───────────────

__global__ void kernel_cpy_f16_iso3(
    const half * __restrict__ src,
    block_iso3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_ISO3;
    block_iso3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    // No forward rotation — matches set-rows norot design.
    // dequantize_iso3_0 reads centroids directly without inverse rotation.
    float rotated[128];
    memcpy(rotated, buf, sizeof(buf));

    // Quantize + pack (3-bit: 2-bit qs + 1-bit signs)
    for (int j = 0; j < QK_ISO3/4; j++) blk->qs[j] = 0;
    for (int j = 0; j < QK_ISO3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += d_centroids_3bit[idx] * d_centroids_3bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Host-side init: copy rotation constants into device symbols ─────

static bool constants_initialized = false;

void ggml_cuda_init_planar_iso_constants() {
    if (constants_initialized) return;

    // Must match planar-iso-constants.cuh exactly (LCG PRNG seed=42)
    static const float h_cos[64] = {-0.9095053397f,0.1535578452f,-0.8537489227f,-0.6827218011f,-0.4249387949f,0.9864510046f,0.9906673944f,0.5752363372f,-0.9866459035f,0.9878848090f,-0.6215683804f,-0.9835597698f,0.8777263755f,-0.4624640047f,0.2843135922f,-0.7739960698f,0.2385234222f,0.9121914932f,-0.8815003943f,-0.2639699512f,-0.5517087300f,-0.9035294557f,-0.8520543188f,-0.5600635985f,-0.7667286376f,-0.9877949369f,-0.9781949787f,-0.9953372831f,-0.8622053901f,-0.7382118186f,0.9136037642f,-0.2558504503f,-0.8541000475f,-0.6159335408f,0.9861256679f,-0.6758560284f,0.4249571682f,-0.6219544719f,0.9130573430f,-0.5948161096f,0.5759782996f,0.9729901203f,0.6535998325f,0.9222195491f,-0.7668084044f,0.5116178563f,-0.7848786574f,0.9902111051f,0.1997167840f,0.7173003220f,-0.9999998006f,-0.9557868691f,0.5594852693f,-0.9980111824f,0.9782398557f,-0.9150004329f,-0.4084754305f,0.0071549185f,0.9558482753f,-0.0971921648f,-0.9469334002f,0.9999492419f,0.6100589016f,0.0350818915f};
    static const float h_sin[64] = {-0.4156922383f,0.9881396603f,0.5206849114f,-0.7306784124f,-0.9052220836f,0.1640561354f,0.1363015542f,0.8179872593f,0.1628798979f,0.1551889303f,0.7833599099f,-0.1805828875f,-0.4791621957f,0.8866380571f,-0.9587313395f,0.6331904010f,-0.9711367448f,0.4097641756f,0.4721832852f,-0.9645309040f,0.8340368561f,0.4285259884f,0.5234533769f,0.8284496156f,0.6419713361f,-0.1557599517f,-0.2076886701f,0.0964556523f,0.5065588468f,-0.6745689815f,-0.4066056591f,-0.9667163736f,0.5201087471f,-0.7877981171f,0.1660005034f,-0.7370336688f,0.9052134584f,0.7830534049f,-0.4078312009f,-0.8038618014f,0.8174649829f,-0.2308467584f,-0.7568403127f,-0.3866666566f,0.6418760557f,-0.8592131104f,0.6196494922f,0.1395778183f,0.9798536657f,0.6967641265f,-0.0006314605f,0.2940603015f,0.8288402943f,-0.0630371303f,0.2074771907f,0.4034528570f,0.9127693152f,-0.9999744032f,0.2938606379f,0.9952656344f,0.3214298299f,0.0100754012f,-0.7923560668f,-0.9993844410f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_planar_cos, h_cos, sizeof(h_cos)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_planar_sin, h_sin, sizeof(h_sin)));

    static const float h_qw[32] = {0.8350809813f,-0.1648498178f,0.1283752173f,0.2897698581f,-0.1820549369f,0.9549587369f,-0.8741137385f,0.8988990188f,-0.1312584430f,-0.3990598321f,-0.2694816887f,-0.1181898862f,0.1363395452f,0.2665117681f,-0.8263269663f,-0.1834189594f,0.3098247349f,0.2804697454f,-0.5655074716f,-0.1627507508f,0.8684155941f,0.2233296037f,-0.1291671842f,0.6606932878f,-0.5694432259f,-0.2782760859f,0.5113853812f,-0.5139024258f,0.7489815354f,-0.3037399948f,-0.4143463373f,-0.3524050117f};
    static const float h_qx[32] = {0.3547102809f,-0.5782636404f,-0.8299785256f,0.5694668293f,-0.8199930191f,0.1259543896f,-0.3090814352f,-0.2613596618f,-0.1660282463f,-0.5143862963f,0.5898610353f,-0.8277072310f,-0.6826571226f,-0.1740629375f,0.1416199356f,0.4648889899f,0.3485621810f,0.8982698917f,-0.3015249372f,0.4990116358f,0.2398942262f,-0.7447698116f,0.4783197045f,0.0735855624f,-0.2975912094f,-0.0700704753f,0.2975627482f,-0.2652103305f,-0.1539765000f,0.0849994123f,-0.1069803685f,-0.5753474832f};
    static const float h_qy[32] = {0.2416850179f,-0.4488199651f,0.3478420675f,0.5024775267f,0.1696543097f,0.1760476083f,0.0254505407f,0.2389279008f,-0.9429193735f,0.3925755024f,-0.2757458389f,-0.1485267133f,0.5530825853f,-0.8936085105f,0.2953715622f,-0.5285226703f,0.7939327955f,0.0139789311f,-0.2555710375f,0.4543992281f,-0.2698826790f,-0.4736968279f,0.4361720681f,-0.3461222053f,0.0792116225f,0.8827795386f,0.7416539788f,-0.3826399446f,-0.3534849286f,-0.8696597815f,-0.6908422709f,0.2082736641f};
    static const float h_qz[32] = {0.3038694561f,0.4734756052f,-0.3878843784f,0.5831694603f,-0.5054479241f,-0.1731694490f,-0.3737666607f,0.2328704894f,0.2621760964f,0.6239953637f,-0.7082104683f,0.5308507681f,-0.4413037896f,-0.2802782655f,-0.4522367120f,-0.6698107123f,-0.3752456903f,-0.3359423280f,0.7181019187f,0.7106907368f,0.3100073636f,0.4016827941f,0.7350437641f,-0.6607965231f,0.7619289756f,0.3648703992f,-0.3040413559f,0.7213236690f,0.5280022621f,-0.3742936850f,-0.5760775208f,0.7015634775f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qw, h_qw, sizeof(h_qw)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qx, h_qx, sizeof(h_qx)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qy, h_qy, sizeof(h_qy)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qz, h_qz, sizeof(h_qz)));

    // Right-isoclinic quaternions (q_R) — must match planar-iso-constants.cuh
    static const float h_qw_r[32] = {0.8471f,0.8275f,0.8621f,0.8398f,0.8512f,0.8345f,0.8589f,0.8432f,0.8498f,0.8381f,0.8554f,0.8417f,0.8523f,0.8369f,0.8601f,0.8448f,0.8485f,0.8352f,0.8576f,0.8405f,0.8509f,0.8338f,0.8612f,0.8423f,0.8467f,0.8378f,0.8563f,0.8391f,0.8541f,0.8325f,0.8595f,0.8456f};
    static const float h_qx_r[32] = {0.3124f,0.3241f,0.3089f,0.3198f,0.3145f,0.3278f,0.3056f,0.3212f,0.3167f,0.3263f,0.3098f,0.3225f,0.3112f,0.3289f,0.3045f,0.3201f,0.3156f,0.3258f,0.3078f,0.3234f,0.3134f,0.3298f,0.3034f,0.3223f,0.3189f,0.3245f,0.3067f,0.3218f,0.3123f,0.3276f,0.3023f,0.3198f};
    static const float h_qy_r[32] = {0.2189f,0.2256f,0.2123f,0.2289f,0.2156f,0.2312f,0.2098f,0.2267f,0.2212f,0.2298f,0.2134f,0.2278f,0.2178f,0.2323f,0.2087f,0.2245f,0.2198f,0.2287f,0.2112f,0.2299f,0.2167f,0.2334f,0.2076f,0.2256f,0.2223f,0.2276f,0.2145f,0.2286f,0.2187f,0.2321f,0.2065f,0.2234f};
    static const float h_qz_r[32] = {0.3567f,0.3689f,0.3456f,0.3721f,0.3523f,0.3756f,0.3412f,0.3698f,0.3589f,0.3712f,0.3478f,0.3701f,0.3545f,0.3767f,0.3398f,0.3678f,0.3578f,0.3708f,0.3445f,0.3723f,0.3534f,0.3778f,0.3387f,0.3687f,0.3598f,0.3697f,0.3467f,0.3711f,0.3556f,0.3789f,0.3376f,0.3668f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qw_r, h_qw_r, sizeof(h_qw_r)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qx_r, h_qx_r, sizeof(h_qx_r)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qy_r, h_qy_r, sizeof(h_qy_r)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qz_r, h_qz_r, sizeof(h_qz_r)));

    constants_initialized = true;
}

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_PLANAR3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_planar3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_planar3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_ISO3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_iso3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_iso3_0 *)dst, n_blocks);
}
