// RDNA2-optimized dequant helper (Kernel A)
// Compiled only when RDNA2_OPT_V1 is set for this file.
#include "convert.cuh"

#ifdef RDNA2_OPT_V1

// Lightweight vector type for 128-bit loads
struct int4_t { int v0, v1, v2, v3; };

// Optimized kernel: vectorized block loads into shared memory + half2 stores.
// This mirrors the baseline dequantize_block_q8_0_f16 logic but performs
// wider loads/stores intended to improve coalescing on RDNA2.
template <bool need_check>
static __global__ void dequantize_block_q8_0_f16_rdn2(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    // Shared staging buffer (same size as baseline)
    __shared__ int vals[nint];

    // Vectorized store into shared memory: each thread copies multiple ints via int4_t
    const int nvec = nint / 4;
    int4_t * x0v = (int4_t *) x0;
    int4_t * valsv = (int4_t *) vals;

    // Each warp-sized block of threads cooperatively loads vectors
    for (int ixv0 = threadIdx.x; ixv0 < nvec; ixv0 += WARP_SIZE) {
        if (need_check) {
            // conservative boundary check: compute byte offset
            const size_t byte_off = (size_t)(ixv0*4) * sizeof(int) + (size_t)blockIdx.x * CUDA_Q8_0_NE_ALIGN * sizeof(int);
            if (byte_off >= (size_t)k * sizeof(int)) break;
        }
        valsv[ixv0] = x0v[ixv0];
    }

    // handle any remaining ints
    const int rem_start = (nvec*4);
    for (int ix = rem_start + threadIdx.x; ix < nint; ix += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*ix >= k*sizeof(block_q8_0)/QK8_0) break;
        vals[ix] = x0[ix];
    }

    __syncthreads();

    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif
}

#endif // RDNA2_OPT_V1
