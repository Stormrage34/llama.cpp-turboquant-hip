#ifndef GGML_CUDA_QUANT_LAYOUTS_RDN2_CUH
#define GGML_CUDA_QUANT_LAYOUTS_RDN2_CUH

#include <cstdint>

// RDNA2 BFE-based weight unpacking for K-quant dequant kernels.
// Uses v_bfe_u32 (1-cycle bit-field extract) instead of shift+mask.
// ONLY for per-element dequant path — matmul path uses SIMD shift+mask
// which processes 8 nibbles simultaneously (BFE would be slower there).
//
// Gate: #ifdef RDNA2_BFE_DISPATCHER (compile-time)
// Runtime: RDNA2_OPT_V1=1 env var must also be set for RDNA2 path.

#ifdef RDNA2_BFE_DISPATCHER

#include "ggml-common.h"  // block_q4_K, block_q5_K definitions

// Trait: per-element BFE unpack for Q4_K_M dequant
// Q4_K_M: 256 elements/block, 4-bit nibbles packed 8 per uint32 in qs[]
// Block-level scale d (half) + 16 sub-block scales
template <typename QuantType>
struct rdna2_bfe_traits {
    static constexpr bool use_bfe = false;
};

template <>
struct rdna2_bfe_traits<block_q4_K> {
    static constexpr bool use_bfe = true;
    static constexpr int vgpr_count = 7;

    // Extract 4-bit nibble at position idx within a Q4_K block.
    // idx: element index (0-255) within the 256-element block.
    // qs layout: 8 nibbles per uint32, packed as [n0|n1|n2|n3|n4|n5|n6|n7]
    // BFE extracts 4 bits starting at bit (idx%8)*4.
    __device__ static inline float unpack(const block_q4_K * block, int idx, float scale) {
        const uint32_t * qs32 = (const uint32_t *)block->qs;
        const uint32_t packed = qs32[idx / 8];
        const uint32_t nibble = __builtin_amdgcn_bfe_u32(packed, (idx % 8) * 4, 4);
        return __builtin_amdgcn_cvt_f32_u32(nibble) * scale;
    }
};

// Trait: per-element BFE unpack for Q5_K_M dequant
// Q5_K_M: 256 elements/block, 4+1 bits (4 in qs[], 5th in qh[])
// qh layout: interleaved — element idx's 5th bit is at qh[idx%32] bit (idx/32)
template <>
struct rdna2_bfe_traits<block_q5_K> {
    static constexpr bool use_bfe = true;
    static constexpr int vgpr_count = 8;

    __device__ static inline float unpack(const block_q5_K * block, int idx, float scale) {
        const uint32_t * qs32 = (const uint32_t *)block->qs;
        const uint32_t packed = qs32[idx / 8];
        const uint32_t nibble = __builtin_amdgcn_bfe_u32(packed, (idx % 8) * 4, 4);

        // 5th bit: interleaved layout — qh[idx%32] bit (idx/32)
        // qh is uint8_t[32]; cast to uint32_t for BFE access
        const uint32_t * qh32 = (const uint32_t *)block->qh;
        const uint32_t qh_word = qh32[(idx % 32) / 8];  // 4 bytes per uint32
        const uint32_t bit5 = __builtin_amdgcn_bfe_u32(qh_word, ((idx % 32) % 8) * 4 + (idx / 32), 1);

        const uint32_t weight = nibble | (bit5 << 4);
        return __builtin_amdgcn_cvt_f32_u32(weight) * scale;
    }
};

#endif // RDNA2_BFE_DISPATCHER

#endif // GGML_CUDA_QUANT_LAYOUTS_RDN2_CUH