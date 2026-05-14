#ifndef GGML_CUDA_IQ4_DEQUANT_RDN2_CUH
#define GGML_CUDA_IQ4_DEQUANT_RDN2_CUH

#include <hip/hip_runtime.h>

// RDNA2-optimized IQ4_XS dequant kernel template (shared header)
// Included by both convert.cu and iq4_dequant_rdn2.cu

// block_iq4_xs layout: ggml_half d + uint16_t scales_h + uint8_t[4] scales_l + uint8_t[128] qs
// QK_K = 256 elements per block, 72 bytes total

// Type-safe output helper for RDNA2
template<typename dst_t> static __device__ __forceinline__ dst_t rdn2_cvt(float v);
template<> __device__ __forceinline__ half  rdn2_cvt<half>(float v)  { return __float2half(v); }
template<> __device__ __forceinline__ float rdn2_cvt<float>(float v) { return v; }

// Optimized IQ4_XS dequant kernel for RDNA2:
// - Wave-cooperative scale extraction (threads 0-7 each handle one sub-block)
// - Direct global memory writes (no LDS staging)
template <bool need_check, typename dst_t>
static __global__ void dequantize_block_iq4_xs_rdn2(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i0 = 256LL * blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx + blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < 8) {
        const int sl_idx = tid / 2;
        const int sl_shift = (tid % 2) * 4;
        const int sh_shift = 2 * tid;
        const float dl = (float)x->d * (((float)((x->scales_l[sl_idx] >> sl_shift) & 0xf) + (float)((x->scales_h >> sh_shift) & 3) * 16.0f) - 32.0f);

        const uint8_t * q4 = x->qs + tid * 16;
        dst_t * out_lower = y + i0 + tid * 32;
        dst_t * out_upper = y + i0 + tid * 32 + 16;

        for (int j = 0; j < 16; ++j) {
            uint8_t qbyte = q4[j];
            out_lower[j] = rdn2_cvt<dst_t>(dl * kvalues_iq4nl[qbyte & 0xF]);
            out_upper[j] = rdn2_cvt<dst_t>(dl * kvalues_iq4nl[qbyte >> 4]);
        }
    }

#if defined(__gfx1030__) || defined(__gfx1010__)
    __threadfence_system();
#endif
}

// Host wrapper: launches the RDNA2 IQ4_XS dequant kernel (type-safe)
template <typename dst_t>
static void ggml_dequant_iq4_xs_rdn2(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + 256 - 1) / 256;
    if (k % 256 == 0) {
        dequantize_block_iq4_xs_rdn2<false, dst_t><<<nb, 32, 0, stream>>>(vx, y, k);
    } else {
        dequantize_block_iq4_xs_rdn2<true, dst_t><<<nb, 32, 0, stream>>>(vx, y, k);
    }
}

#endif // GGML_CUDA_IQ4_DEQUANT_RDN2_CUH
