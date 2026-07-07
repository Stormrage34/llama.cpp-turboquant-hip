/*
 * RotorQuant: Clifford-algebra-based vector quantization (Cl(3,0))
 * Based on: https://github.com/scrya-com/rotorquant
 * arXiv: Reimagining TurboQuant with Clifford Algebra (2026)
 *
 * GGML_TYPE_RQ_MSE_2: Rotor MSE quantizer (Lloyd-Max per component)
 * GGML_TYPE_RQ_PROD:  Rotor MSE + QJL for attention (unbiased IP estimator)
 *
 * Multivector basis (Cl(3,0)): [1, e1, e2, e3, e12, e13, e23, e123]
 *   scalar         vector      bivector    pseudoscalar
 *   (grade-0)      (grade-1)   (grade-2)   (grade-3)
 *
 * Block format: 50 bytes per 128 elements
 *   - 2B norm (fp16) — stored per-group like planar3/iso3
 *   - Quantized indices for multivector components (Lloyd-Max codebooks)
 *   - For RQ_PROD: QJL residual sign bits + norm (for unbiased IP)
 *
 * The rotor R = cos(θ/2) + sin(θ/2)*B̂ acts via sandwich product:
 *   R x R̃ where x is a multivector and R̃ is the reverse of R.
 * For grade-1 vectors (embedded as multivectors), the rotor preserves norms
 * while decorrelating components for optimal Lloyd-Max quantization.
 */
#pragma once
#include "common.cuh"

// Forward declarations for RotorQuant host-side init (defined in rotorquant-init.cu).
void rq_mse_init_codebooks(int n_levels_vector, int n_levels_trivector);
void rq_mse_init_rotors(const float * h_rotors, int n_groups);
void rq_mse_init_full(int n_groups, int seed);

// Cl(3,0) basis element indices and dimension constants
#define RQ_MV_DIM 8  // |Cl(3,0)| = 2^3 = 8 components per multivector

// NOTE: RQ_MSE_CENTROIDS_TRIVECTOR IS USED as the null-grade codebook.
// Scalar (grade-0) and bivector (grades-2, e12/e13/e23) components are structurally zero
// after rotor embedding. rq_grade_centroid() routes them to TRIVECTOR for exact zero reconstruction.
#ifndef BLOCK_RQ_MSE_2_DEFINED
typedef struct {
    uint16_t norm;       // 2 bytes: block norm (fp16)
    uint8_t  qs[336];    // 336 bytes: Lloyd-Max quantized indices (uint8) for 42 groups × 8 components
} block_rq_mse_2;
#endif /* BLOCK_RQ_MSE_2_DEFINED */
#ifndef BLOCK_RQ_PROD_DEFINED
typedef struct {
    uint16_t norm;              // 2 bytes: block norm (fp16)
    uint8_t  qs[336];           // 336 bytes: Lloyd-Max quantized indices (uint8)
    float residual_norm;        // 4 bytes: QJL residual L2 norm
    uint8_t qjl_signs[4];       // 4 bytes: QJL sign bits (one per projected dimension)
} block_rq_prod;
#endif /* BLOCK_RQ_PROD_DEFINED */
#define RQ_N_GROUPS (128 / 3)  // 42 groups (63 vector dims + partial)
#define RQ_E1 (int)(1)
#define RQ_E2 (int)(2)
#define RQ_E3 (int)(3)
#define RQ_E12 (int)(4)
#define RQ_E13 (int)(5)
#define RQ_E23 (int)(6)
#define RQ_E123 (int)(7)
#define RQ_S (int)(0)  // scalar
// Quantization ratios for dequantize_block template
#define QR_RQ_MSE_2 2  // Each call produces 4 elements (2 per thread via float2)
#define QR_RQ_PROD 1   // Each call produces 2 elements (via float2)

// AMD bit-field extract intrinsic (unsigned) — HIP/ROCm port of AMD_BFE.
// Extracts n bits from src starting at bit position pos.
#define HIP_BFE(src, pos, n) \
    (uint8_t)(__builtin_amdgcn_ubfe((src), (pos), (n)) & ((1 << (n)) - 1))

// Lloyd-Max codebook centroids (precomputed by host at load time)
// TURBO_CENTROIDS_3BIT is defined in turbo-quant.cuh.
// We include it here to break the circular include dependency.

#include "turbo-quant.cuh"  // defines TURBO_CENTROIDS_3BIT, block_turbo4_0, etc.




  // Device constant arrays — definitions (one per translation unit, merged by linker)
  // RQ_MSE_CENTROIDS_VECTOR:   Lloyd-Max centroids for grade-1 (vector) components.
  // RQ_MSE_CENTROIDS_TRIVECTOR: zero-centroid table for structurally null grades.
  //                            Scalar (grade-0) and bivectors (grade-2) always quantize to 0.0f
  //                            after rotor embedding — using a single-zero centroid gives them
  //                            perfect reconstruction with zero index cost, freeing codebook width
  //                            for vector components which actually carry the signal.
  __constant__ float RQ_MSE_CENTROIDS_VECTOR[256];
  __constant__ float RQ_MSE_CENTROIDS_TRIVECTOR[256];
  __constant__ int   RQ_MSE_NLEVELS_VECTOR;
  __constant__ int   RQ_MSE_NLEVELS_TRIVECTOR;

  // Grade-aware centroid lookup: routes each component to its appropriate codebook table.
  // Structurally null grades (scalar + bivectors) use zero-centroid → guaranteed exact zero.
  // Vector grade (e1,e2,e3) uses the wide-range Lloyd-Max table for actual signal.
  static __device__ __forceinline__ float rq_grade_centroid(uint8_t idx, int comp) {
      // Scalars (grade-0) and bivectors (grade-2) are structurally zero after embedding.
      // Use the "trivector" table as our zero-centroid table for these grades.
      if (comp == RQ_S || comp == RQ_E12 || comp == RQ_E13 || comp == RQ_E23) {
          return RQ_MSE_CENTROIDS_TRIVECTOR[idx]; // will be 0.0f (zero-centroid table)
      }
      // Vector grade (e1,e2,e3) carries the signal — use full Lloyd-Max.
      return RQ_MSE_CENTROIDS_VECTOR[idx];
  }
 // Max rotor capacity: RQ_MV_DIM * RQ_N_GROUPS = 336 (42 multivectors × 8 floats).
// Clamp any rotor index with % RQ_MAX_ROTORS to prevent buffer overflow.
#define RQ_MAX_ROTORS (RQ_MV_DIM * RQ_N_GROUPS)

// Clamp rotor index: use modulo to fit within bounded rotor table.
static __device__ __forceinline__ int rq_clamp_rotor_idx(int idx) {
    return idx % RQ_MAX_ROTORS;
}

// QJL S-matrix is passed as kernel argument (not __constant__) — avoids 64MB allocation.
__constant__ float RQ_MSE_ROTORS[RQ_N_GROUPS * RQ_MV_DIM];
__constant__ int   RQ_MSE_N_ROTORS;

// NOTE: ISO3 and RotorQuant use incompatible rotation transforms (SO(4) quaternions vs SO(2)/
// SO(4) Clifford rotors). They must not be mixed on the same attention tensor.
#if defined(QK_PLANAR3) || defined(QK_ISO3)
  // The user is trying to mix ISO3/Planar with RotorQuant in the same block.
  // If you're compiling both, make sure only one is enabled per tensor type (see -ctk/-ctv args).
#endif
// Note: RQ_MSE_CENTROIDS_TABLE was previously a broken device-pointer-to-host-array — deleted.
// Clifford algebra operations (device-side)
// Geometric product: a * b in Cl(3,0) with signature (+,+,+)
static __device__ __forceinline__ void rq_geometric_product(
        const float * __restrict__ a, const float * __restrict__ b,
        float * __restrict__ out) {
    const float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const float a12 = a[4], a13 = a[5], a23 = a[6], a123 = a[7];
    const float b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    const float b12 = b[4], b13 = b[5], b23 = b[6], b123 = b[7];
    out[RQ_S]     = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a13*b13 - a23*b23 - a123*b123;
    out[RQ_E1]    = a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3 - a23*b123 - a123*b23;
    out[RQ_E2]    = a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3 + a13*b123 + a123*b13;
    out[RQ_E3]    = a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2 - a12*b123 - a123*b12;
    out[RQ_E12]   = a0*b12 + a12*b0 + a1*b2 - a2*b1 - a13*b23 + a23*b13 + a3*b123 + a123*b3;
    out[RQ_E13]   = a0*b13 + a13*b0 + a1*b3 - a3*b1 + a12*b23 - a23*b12 - a2*b123 - a123*b2;
    out[RQ_E23]   = a0*b23 + a23*b0 + a2*b3 - a3*b2 - a12*b13 + a13*b12 + a1*b123 + a123*b1;
    out[RQ_E123]  = a0*b123 + a123*b0 + a1*b23 + a23*b1 - a2*b13 - a13*b2 + a3*b12 + a12*b3;
}
// Reverse (reversion) of multivector x -> x̃ (signs flipped for grade >= 2)
static __device__ __forceinline__ void rq_reverse(const float * __restrict__ x, float * __restrict__ out) {
    out[RQ_S]      = x[RQ_S];
    out[RQ_E1]     = x[RQ_E1];
    out[RQ_E2]     = x[RQ_E2];
    out[RQ_E3]     = x[RQ_E3];
    out[RQ_E12]   = -x[RQ_E12];
    out[RQ_E13]   = -x[RQ_E13];
    out[RQ_E23]   = -x[RQ_E23];
    out[RQ_E123]  = -x[RQ_E123];
}
// Rotor sandwich: R x R̃ (rotate multivector x by rotor R)
static __device__ __forceinline__ void rq_rotor_sandwich(
        const float * __restrict__ R, const float * __restrict__ x,
        float * __restrict__ out) {
    float t[RQ_MV_DIM];
    float Rt[RQ_MV_DIM];
    // First: R * x
    rq_geometric_product(R, x, t);
    // Reverse of rotor R = [R0, R1..R3, -R12..-R123]
    rq_reverse(R, Rt);
    // Then: (R*x) * R̃
    rq_geometric_product(t, Rt, out);
}
// Vector norm squared via scalar part of x * x̃
static __device__ __forceinline__ float rq_multivector_norm_sq(const float * __restrict__ x) {
    float xr[RQ_MV_DIM];
    rq_reverse(x, xr);
    // Scalar part computed directly from expanded product
    const float a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3];
    const float a12 = x[4], a13 = x[5], a23 = x[6], a123 = x[7];
    const float b0 = xr[0], b1 = xr[1], b2 = xr[2], b3 = xr[3];
    const float b12 = xr[4], b13 = xr[5], b23 = xr[6], b123 = xr[7];
    return a0*b0 + a1*b1 + a2*b2 + a3*b3
         - a12*b12 - a13*b13 - a23*b23 - a123*b123;
}
// Embed vectors as Cl(3,0) multivectors (packing for d-dim input)
// Packs every 3 dims into vector grade (e1,e2,e3), rest into scalar + bivector.
// Output: n_groups groups, each 8-component multivector.
static __device__ void rq_embed_vectors_as_multivectors(
        const float * __restrict__ v, int d, float * __restrict__ mv) {
    const int n_groups = (d + 2) / 3;
    #pragma unroll
    for (int g = 0; g < n_groups && g < RQ_N_GROUPS; g++) {
        mv[g * RQ_MV_DIM + RQ_S] = 0.0f;      // scalar
        mv[g * RQ_MV_DIM + RQ_E1] = 0.0f;     // e1
        mv[g * RQ_MV_DIM + RQ_E2] = 0.0f;     // e2
        mv[g * RQ_MV_DIM + RQ_E3] = 0.0f;     // e3
        mv[g * RQ_MV_DIM + RQ_E12] = 0.0f;    // e12
        mv[g * RQ_MV_DIM + RQ_E13] = 0.0f;    // e13
        mv[g * RQ_MV_DIM + RQ_E23] = 0.0f;    // e23
        mv[g * RQ_MV_DIM + RQ_E123] = 0.0f;   // e123
        int base = g * 3;
        if (base < d) mv[g * RQ_MV_DIM + RQ_E1] = v[base];
        if (base + 1 < d) mv[g * RQ_MV_DIM + RQ_E2] = v[base + 1];
        if (base + 2 < d) mv[g * RQ_MV_DIM + RQ_E3] = v[base + 2];
    }
}
// Extract vectors from Cl(3,0) multivectors (unpacking)
static __device__ void rq_extract_vectors_from_multivectors(
        const float * __restrict__ mv, int d, float * __restrict__ v) {
    const int n_groups = (d + 2) / 3;
    #pragma unroll
    for (int g = 0; g < n_groups && g < RQ_N_GROUPS; g++) {
        int base = g * 3;
        if (base < d) v[base] = mv[g * RQ_MV_DIM + RQ_E1];
        if (base + 1 < d) v[base + 1] = mv[g * RQ_MV_DIM + RQ_E2];
        if (base + 2 < d) v[base + 2] = mv[g * RQ_MV_DIM + RQ_E3];
    }
}
// Lloyd-Max nearest centroid lookup (device-side, branchless)
// Uses midpoint binary search for O(log n) lookup.
static __device__ __forceinline__ int rq_nearest_centroid_bsearch(
        const float * __restrict__ midpoints, int n_levels, float val) {
    // Binary search for insertion point in sorted midpoints array.
    // Returns index i such that centroid[i] is nearest.
    int lo = 0, hi = n_levels - 1;
    // For Lloyd-Max: midpoints are strictly increasing, centroids surround them.
    // Find smallest i where val < mid[i], then centroid is i (or i-1 if negative).
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val < midpoints[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // lo is the first index where val >= mid[lo-1] and val < mid[lo].
    // Nearest centroid is either lo or lo-1 (pick closer).
    if (lo == 0) return 0;
    if (lo == n_levels) return n_levels - 1;
    // Compare distance to midpoints[lo-1] and midpoints[lo], then pick centroid.
    float dist_left = fabsf(val - midpoints[lo - 1]);
    float dist_right = fabsf(midpoints[lo] - val);
    return (dist_left <= dist_right) ? lo - 1 : lo;
}
// Rotor MSE dequantize: reconstruct 128-dim vector from quantized block.
// Block format (scrya): [norm (fp16), qs[] uint8 indices = 42 groups × 8 components].
// Each multivector group encodes 3 consecutive vector elements (e1, e2, e3).
// Dequantization: centroid lookup + full Clifford inverse rotation R̃ * q * R.
//
// Helper: dequantize one element at position iqs within block ib.
// Groups: iqs=0,1,2→group 0, iqs=3,4,5→group 1, ..., iqs=125,126,127→group 41.
// Component: iqs%3→0=e1, 1=e2, 2=e3 → mapped to RQ_E1/E2/E3 (1,2,3).
static __device__ __forceinline__ float dequantize_rq_mse_2_elem(
        const block_rq_mse_2 * __restrict__ x, int64_t ib, int iqs) {
    const int g    = iqs / 3;
    const int comp = (iqs % 3) + 1;  // 1=e1, 2=e2, 3=e3
    // Read quantized Lloyd-Max indices from qs[] and lookup centroid table.
    const int base = g * RQ_MV_DIM;
    float centroids[RQ_MV_DIM];
    #pragma unroll
    for (int c = 0; c < RQ_MV_DIM; c++) {
        uint8_t idx = x[ib].qs[base + c]; // quantized Lloyd-Max index (uint8)
        centroids[c] = rq_grade_centroid(idx, c); // grade-aware centroid lookup
    }
    // Apply rotor reverse: x_hat = R̃ * centroids * R (full sandwich inverse).
    const float * __restrict__ R = &RQ_MSE_ROTORS[g * RQ_MV_DIM];
    float Rt[RQ_MV_DIM];
    rq_reverse(R, Rt);
    // Forward step: R̃ * centroids.
    float t[RQ_MV_DIM];
    rq_geometric_product(Rt, centroids, t);
    // Inverse step: t * R (un-rotation).
    float uv[RQ_MV_DIM];
    float const_R[RQ_MV_DIM];  // copy R to non-const temporary.
    #pragma unroll
    for (int c = 0; c < RQ_MV_DIM; c++) const_R[c] = R[c];
    rq_geometric_product(t, const_R, uv);
    return uv[comp];
}

// Rotor PROD dequantize (MSE reconstruction + QJL correction).
// For attention: returns MSE reconstruction. QJL residual correction is applied at attention score computation time.
// Same group layout as RQ_MSE_2: 42 multivector groups per 128-element block.
static __device__ __forceinline__ float dequantize_rq_prod_elem(
        const block_rq_prod * __restrict__ x, int64_t ib, int iqs) {
    const int g    = iqs / 3;
    const int comp = (iqs % 3) + 1;
    // Read quantized Lloyd-Max indices from qs[] and lookup centroid table.
    const int base = g * RQ_MV_DIM;
    float centroids[RQ_MV_DIM];
    #pragma unroll
    for (int c = 0; c < RQ_MV_DIM; c++) {
        uint8_t idx = x[ib].qs[base + c]; // quantized Lloyd-Max index (uint8)
        centroids[c] = RQ_MSE_CENTROIDS_VECTOR[idx]; // centroid lookup
    }
    // Apply rotor reverse: R̃ * centroids * R (full sandwich inverse).
    const float * __restrict__ R = &RQ_MSE_ROTORS[g * RQ_MV_DIM];
    float Rt[RQ_MV_DIM];
    rq_reverse(R, Rt);
    float t[RQ_MV_DIM];
    rq_geometric_product(Rt, centroids, t);
    // Inverse step: t * R (un-rotation).
    float uv[RQ_MV_DIM];
    float local_R[RQ_MV_DIM];
    #pragma unroll
    for (int c = 0; c < RQ_MV_DIM; c++) local_R[c] = R[c];
    rq_geometric_product(t, local_R, uv);
    return uv[comp];
}

static __device__ __forceinline__ void dequantize_rq_mse_2(
        const void * __restrict__ vx, int64_t ib, int iqs, float2 &v) {
    const block_rq_mse_2 * __restrict__ x = (const block_rq_mse_2 *) vx;
    v.x = dequantize_rq_mse_2_elem(x, ib, iqs);
    v.y = dequantize_rq_mse_2_elem(x, ib, iqs + 1);
}
static __device__ __forceinline__ void dequantize_rq_prod(
        const void * __restrict__ vx, int64_t ib, int iqs, float2 &v) {
    const block_rq_prod * __restrict__ x = (const block_rq_prod *) vx;
    v.x = dequantize_rq_prod_elem(x, ib, iqs);
    v.y = dequantize_rq_prod_elem(x, ib, iqs + 1);
}
// Host-side: Generate random rotors for RQ_MSE_2 (seed=42 LCG)
// Rotor R = cos(θ/2) + sin(θ/2)*B̂ where B̂ is a random unit bivector.
static void rq_mse_generate_rotors(float * rotors_out, int n_groups, uint32_t seed) {
    uint32_t state = seed;
    for (int g = 0; g < n_groups && g < RQ_N_GROUPS; g++) {
        // Generate scalar (cos of half-angle) from random float [0,1]
        state = 1664525u * state + 1013904245u;
        float cos_ha = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        // Generate bivector direction from random floats
        float b12 = 0, b13 = 0, b23 = 0;
        for (int i = 0; i < 3; i++) {
            state = 1664525u * state + 1013904245u;
            float r = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
            switch (i) {
                case 0: b12 = r; break;
                case 1: b13 = r; break;
                case 2: b23 = r; break;
            }
        }
        // Normalize bivector
        float bv_norm = sqrtf(b12*b12 + b13*b13 + b23*b23);
        if (bv_norm < 1e-8f) {
            b12 = 0; b13 = 0; b23 = 0;
        } else {
            float inv_norm = 1.0f / bv_norm;
            b12 *= inv_norm;
            b13 *= inv_norm;
            b23 *= inv_norm;
        }
        // Compute angle: uniform in [0, 2*pi) via atan2 of two normals
        state = 1664525u * state + 1013904245u;
        float r_a = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        state = 1664525u * state + 1013904245u;  // advance PRNG state (r_b unused)
        float angle = 2.0f * acosf(2.0f * r_a - 1.0f);
        // Compute sin(θ/2)
        float sin_ha = sinf(angle / 2.0f);
        // Rotor: [cos_ha, 0, 0, 0 (vector), sin_ha*b12, sin_ha*b13, sin_ha*b23]
        int base = g * RQ_MV_DIM;
        rotors_out[base + RQ_S]     = cos_ha;
        rotors_out[base + RQ_E1]    = 0.0f;
        rotors_out[base + RQ_E2]    = 0.0f;
        rotors_out[base + RQ_E3]    = 0.0f;
        rotors_out[base + RQ_E12]   = sin_ha * b12;
        rotors_out[base + RQ_E13]   = sin_ha * b13;
        rotors_out[base + RQ_E23]   = sin_ha * b23;
        rotors_out[base + RQ_E123]  = 0.0f;
    }
}

// ============================================================================
// QJL (Quantized Johnson-Lindenstrauss) helpers for RQ_PROD.
// The residual correction: <y, x̂> + ||r|| * √(π/2)/m * Σ(y_j * S_j ⊙ qjl_signs).
// In practice: precompute per-cache-entry sketched values during append.
// ============================================================================

static __device__ __forceinline__ float rq_qjl_inner_product(
        const float * __restrict__ y, int /* d */,
        const float * __restrict__ S, int m,
        int iqs) {
    // Term 1: direct IP with MSE reconstruction (computed separately by caller).
    // Term 2: QJL residual correction — Σ_j (y_j * S_j[i]).
    // NOTE: S has RQ_MV_DIM columns; j wraps across them. Ensure m <= RQ_MV_DIM.
    float sk = 0.0f;
    for (int j = 0; j < m; ++j) {
        int idx = (iqs + j) % RQ_MV_DIM;
        sk += y[j] * S[idx];
    }
    return sk;  // Caller multiplies by ||r|| * √(π/2)/m.
}

static __device__ __forceinline__ float rq_qjl_residual_correction(
        const float * __restrict__ y, const float * __restrict__ S, int m,
        int iqs, const uint8_t * __restrict__ qjl_signs, float residual_norm) {
    // Compute Σ_j (y_j * S_j[i]) weighted by sign bits.
    // qjl_signs is a 4-byte array (32 bits), each byte covering 8 j values.
    float sk = 0.0f;
    for (int j = 0; j < m; ++j) {
        int idx = (iqs + j) % RQ_MV_DIM;
        int byte_idx = j / 8;  // which of qjl_signs[4]
        int bit_idx = j % 8;    // which bit within that byte
        if (byte_idx < 4) {
            int bit = (qjl_signs[byte_idx] >> bit_idx) & 1;
            sk += y[j] * S[idx] * (2.0f * bit - 1.0f);
        }
    }
    // Scale: ||r|| * √(π/2)/m (per scrya reference).
    return sk * residual_norm * 0.797885f / (float)m;
}

// ============================================================================
// RotorQuant attention helpers for fattn-common.cuh
// vec_dot_fattn_vec_KQ_<type>: dequantize K via rotor, dot with Q.
// dequantize_V_<type>: dequantize V from RQ blocks.
// ============================================================================

#ifndef QK_RQ
#define QK_RQ 128
#endif

// ============================================================================
// 3-bit centroids (Lloyd-Max for N(0, 1/128)) — needed by dequantize_V_RQ functions below.
// If turbo-quant.cuh was included first (and its include is silently dropped by #pragma once),
// we must provide our own definition to avoid "undeclared identifier" errors.
// ============================================================================

#ifndef TURBO_CENTROIDS_3BIT_DEFINED
#define TURBO_CENTROIDS_3BIT_DEFINED
static __constant__ float RQ_TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
#else
// turbo-quant.cuh already provided TURBO_CENTROIDS_3BIT — alias our local name to it.
#define RQ_TURBO_CENTROIDS_3BIT TURBO_CENTROIDS_3BIT
#endif

// ============================================================================
// dequantize_V for RQ types — now using block_rq_mse_2 format (not turbo3_0).
// V tensor uses the same block_rq_mse_2 layout as K for RQ types.
// ============================================================================

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_rq_mse_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    // V dequant: centroid lookup + INVERSE rotor sandwich R̃ * centroids * R
    // Must match K path (dequantize_rq_mse_2_elem) — both produce original-space vectors.
    const block_rq_mse_2 * x = (const block_rq_mse_2 *) vx;
    const int64_t ib   = i0 / QK_RQ;
    const int     j0   = i0 % QK_RQ;
    const float   norm = __half2float(x[ib].norm);

    int group_idx = j0 / 3;  // which multivector group we're in.

    static_assert(ne == 2 || ne == 4, "bad ne");

    // Helper: dequantize one group via full inverse rotor sandwich, return e1,e2
    auto dequant_group = [&](int g) -> float2 {
        // 1. Centroid lookup: load all 8 components for this group
        float centroids[RQ_MV_DIM];
        const int qbase = g * RQ_MV_DIM;
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) {
            centroids[c] = rq_grade_centroid(x[ib].qs[qbase + c], c); // grade-aware centroid lookup
        }
        // 2. Inverse rotor sandwich: R̃ * centroids * R
        const float * R = &RQ_MSE_ROTORS[g * RQ_MV_DIM];
        float Rt[RQ_MV_DIM];
        rq_reverse(R, Rt);
        float t[RQ_MV_DIM];
        rq_geometric_product(Rt, centroids, t);
        float uv[RQ_MV_DIM];
        float local_R[RQ_MV_DIM];
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) local_R[c] = R[c];
        rq_geometric_product(t, local_R, uv);
        // 3. Extract e1, e2 from un-rotated multivector, scale by block norm
        return make_float2(uv[RQ_E1] * norm, uv[RQ_E2] * norm);
    };

    if constexpr (ne == 4) {
        float2 v0 = dequant_group(group_idx);
        float2 v1 = dequant_group((group_idx + 1) % RQ_N_GROUPS); // NOTE: wrap at group boundary reads from vector start

        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(__float2half(v0.x), __float2half(v0.y));
            ((half2 *) dst)[1] = make_half2(__float2half(v1.x), __float2half(v1.y));
        } else {
            ((float *) dst)[0] = v0.x;
            ((float *) dst)[1] = v0.y;
            ((float *) dst)[2] = v1.x;
            ((float *) dst)[3] = v1.y;
        }
    } else { // ne == 2
        float2 v0 = dequant_group(group_idx);

        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(__float2half(v0.x), __float2half(v0.y));
        } else {
            ((float *) dst)[0] = v0.x;
            ((float *) dst)[1] = v0.y;
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_rq_prod(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    // V dequant for RQ_PROD: centroid lookup + INVERSE rotor sandwich (same as RQ_MSE_2).
    // block_rq_prod shares the same norm+qs[336] prefix as block_rq_mse_2.
    const block_rq_prod * x = (const block_rq_prod *) vx;
    const int64_t ib   = i0 / QK_RQ;
    const int     j0   = i0 % QK_RQ;
    const float   norm = __half2float(x[ib].norm);

    int group_idx = j0 / 3;

    static_assert(ne == 2 || ne == 4, "bad ne");

    auto dequant_group = [&](int g) -> float2 {
        float centroids[RQ_MV_DIM];
        const int qbase = g * RQ_MV_DIM;
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) {
            centroids[c] = rq_grade_centroid(x[ib].qs[qbase + c], c); // grade-aware centroid lookup
        }
        const float * R = &RQ_MSE_ROTORS[g * RQ_MV_DIM];
        float Rt[RQ_MV_DIM];
        rq_reverse(R, Rt);
        float t[RQ_MV_DIM];
        rq_geometric_product(Rt, centroids, t);
        float uv[RQ_MV_DIM];
        float local_R[RQ_MV_DIM];
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) local_R[c] = R[c];
        rq_geometric_product(t, local_R, uv);
        return make_float2(uv[RQ_E1] * norm, uv[RQ_E2] * norm);
    };

    if constexpr (ne == 4) {
        float2 v0 = dequant_group(group_idx);
        float2 v1 = dequant_group((group_idx + 1) % RQ_N_GROUPS); // NOTE: wrap at group boundary reads from vector start

        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(__float2half(v0.x), __float2half(v0.y));
            ((half2 *) dst)[1] = make_half2(__float2half(v1.x), __float2half(v1.y));
        } else {
            ((float *) dst)[0] = v0.x;
            ((float *) dst)[1] = v0.y;
            ((float *) dst)[2] = v1.x;
            ((float *) dst)[3] = v1.y;
        }
    } else { // ne == 2
        float2 v0 = dequant_group(group_idx);

        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(__float2half(v0.x), __float2half(v0.y));
        } else {
            ((float *) dst)[0] = v0.x;
            ((float *) dst)[1] = v0.y;
        }
    }
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_rq_mse_2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_rq_mse_2 * K_rq = (const block_rq_mse_2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads * cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne + k_KQ_1;

            const int elem0 = k_KQ * 2;
            const int ib    = elem0 / QK_RQ;
            const int j0    = elem0 % QK_RQ;

            float2 kv = {0.0f, 0.0f};

            // Dequantize two elements using group-aware rotor reverse
            // Uses dequantize_rq_mse_2_elem which handles 42 multivector groups
            kv.x = dequantize_rq_mse_2_elem(K_rq, ib, j0);
            kv.y = dequantize_rq_mse_2_elem(K_rq, ib, j0 + 1);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_rq_prod(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_rq_prod * K_rq = (const block_rq_prod *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads * cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne + k_KQ_1;

            const int elem0 = k_KQ * 2;
            const int ib    = elem0 / QK_RQ;
            const int j0    = elem0 % QK_RQ;

            float2 kv = {0.0f, 0.0f};

            // Dequantize two elements using group-aware rotor reverse
            // Uses dequantize_rq_prod_elem which handles all 42 multivector groups
            kv.x = dequantize_rq_prod_elem(K_rq, ib, j0);
            kv.y = dequantize_rq_prod_elem(K_rq, ib, j0 + 1);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

// ============================================================================
// Quantize functions for SET_ROWS: F32 -> RQ block (used by k_set_rows_quant)
// Signature matches: void (*)(const float * src_block, block_type * dst_block)
// ============================================================================

// Quantize 128 F32 values into block_rq_mse_2 (norm + Lloyd-Max indices).
// Process: normalize → embed as Cl(3,0) multivectors → forward rotor sandwich → Lloyd-Max quantize.
static __device__ __forceinline__ void quantize_f32_rq_mse_2_block(
        const float * __restrict__ src, block_rq_mse_2 * __restrict__ dst) {
    // 1. Compute L2 norm for block-level scaling.
    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK_RQ; i++) {
        norm_sq += src[i] * src[i];
    }
    const float norm = sqrtf(norm_sq);
    dst->norm = __float2half_rn(norm);
    if (norm < 1e-12f) {
        // Zero block: all indices to midpoint of codebook.
        const int midpoint = RQ_MSE_NLEVELS_VECTOR / 2;
        #pragma unroll
        for (int i = 0; i < 336; i++) {
            dst->qs[i] = (uint8_t)midpoint;
        }
        return;
    }
    const float inv_norm = 1.0f / norm;

    // 2. For each of 42 groups: embed 3 dims as vector grade, apply rotor, quantize all 8 components.
    #pragma unroll
    for (int g = 0; g < RQ_N_GROUPS; g++) {
        float mv[RQ_MV_DIM];
        mv[RQ_S]     = 0.0f;
        mv[RQ_E12]   = 0.0f;
        mv[RQ_E13]   = 0.0f;
        mv[RQ_E23]   = 0.0f;
        mv[RQ_E123]  = 0.0f;
        const int base = g * 3;
        mv[RQ_E1] = (base     < QK_RQ) ? src[base]     * inv_norm : 0.0f;
        mv[RQ_E2] = (base + 1 < QK_RQ) ? src[base + 1] * inv_norm : 0.0f;
        mv[RQ_E3] = (base + 2 < QK_RQ) ? src[base + 2] * inv_norm : 0.0f;

        // Forward rotor sandwich: t = R * mv * R̃
        float t[RQ_MV_DIM];
        rq_rotor_sandwich(&RQ_MSE_ROTORS[g * RQ_MV_DIM], mv, t);

        // Lloyd-Max quantize each of 8 components.
        const int qbase = g * RQ_MV_DIM;
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) {
            // Scalar (grade-0) and bivectors (grade-2) are structurally zero after embedding:
            // skip Lloyd-Max and encode as explicit 0 — guarantees perfect reconstruction.
            if (c == RQ_S || c == RQ_E12 || c == RQ_E13 || c == RQ_E23) {
                dst->qs[qbase + c] = 0u;
            } else {
                int idx = rq_nearest_centroid_bsearch(
                    RQ_MSE_CENTROIDS_VECTOR, RQ_MSE_NLEVELS_VECTOR, t[c]);
                idx = max(0, min(idx, RQ_MSE_NLEVELS_VECTOR - 1));
                dst->qs[qbase + c] = (uint8_t)idx;
            }
        }
    }
}

// Quantize 128 F32 values into block_rq_prod (norm + Lloyd-Max indices + QJL residual).
// Same MSE quantization as RQ_MSE_2, plus residual norm for QJL correction.
static __device__ __forceinline__ void quantize_f32_rq_prod_block(
        const float * __restrict__ src, block_rq_prod * __restrict__ dst) {
    // 1. Compute L2 norm.
    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK_RQ; i++) {
        norm_sq += src[i] * src[i];
    }
    const float norm = sqrtf(norm_sq);
    dst->norm = __float2half_rn(norm);
    if (norm < 1e-12f) {
        const int midpoint = RQ_MSE_NLEVELS_VECTOR / 2;
        #pragma unroll
        for (int i = 0; i < 336; i++) {
            dst->qs[i] = (uint8_t)midpoint;
        }
        dst->residual_norm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            dst->qjl_signs[i] = 0;
        }
        return;
    }
    const float inv_norm = 1.0f / norm;

    // 2. MSE quantization (identical to RQ_MSE_2).
    float residual_norm_sq = 0.0f;
    #pragma unroll
    for (int g = 0; g < RQ_N_GROUPS; g++) {
        float mv[RQ_MV_DIM];
        mv[RQ_S]     = 0.0f;
        mv[RQ_E12]   = 0.0f;
        mv[RQ_E13]   = 0.0f;
        mv[RQ_E23]   = 0.0f;
        mv[RQ_E123]  = 0.0f;
        const int base = g * 3;
        mv[RQ_E1] = (base     < QK_RQ) ? src[base]     * inv_norm : 0.0f;
        mv[RQ_E2] = (base + 1 < QK_RQ) ? src[base + 1] * inv_norm : 0.0f;
        mv[RQ_E3] = (base + 2 < QK_RQ) ? src[base + 2] * inv_norm : 0.0f;

        float t[RQ_MV_DIM];
        rq_rotor_sandwich(&RQ_MSE_ROTORS[g * RQ_MV_DIM], mv, t);

        const int qbase = g * RQ_MV_DIM;
        #pragma unroll
        for (int c = 0; c < RQ_MV_DIM; c++) {
            // Scalar (grade-0) and bivectors (grade-2) are structurally zero after embedding:
            // encode as explicit 0 — no residual. Vector components use Lloyd-Max.
            if (c == RQ_S || c == RQ_E12 || c == RQ_E13 || c == RQ_E23) {
                dst->qs[qbase + c] = 0u; // structurally zero: no bsearch needed
            } else {
                int idx = rq_nearest_centroid_bsearch(
                    RQ_MSE_CENTROIDS_VECTOR, RQ_MSE_NLEVELS_VECTOR, t[c]);
                idx = max(0, min(idx, RQ_MSE_NLEVELS_VECTOR - 1));
                dst->qs[qbase + c] = (uint8_t)idx;
                // Accumulate residual for QJL: difference between original rotated and centroid.
                float diff = t[c] - RQ_MSE_CENTROIDS_VECTOR[idx];
                residual_norm_sq += diff * diff;
            }
        }
    }

    // 3. QJL residual + sign bits (for unbiased IP estimator).
    dst->residual_norm = sqrtf(residual_norm_sq);
    // QJL sign bits: sign of each vector component in first 4 groups (32 bits = 4 bytes).
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t bits = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            int elem = i * 8 + b;
            if (elem < QK_RQ && src[elem] * inv_norm > 0.0f) {
                bits |= (1u << b);
            }
        }
        dst->qjl_signs[i] = bits;
    }
}
