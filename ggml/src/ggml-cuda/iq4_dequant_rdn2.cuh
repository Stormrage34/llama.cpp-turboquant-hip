#ifndef GGML_CUDA_IQ4_DEQUANT_RDN2_CUH
#define GGML_CUDA_IQ4_DEQUANT_RDN2_CUH

#include <hip/hip_runtime.h>

// RDNA2-optimized IQ4_XS dequant kernel template (shared header)
// Included by both convert.cu and iq4_dequant_rdn2.cu

// block_iq4_xs layout: ggml_half d + uint16_t scales_h + uint8_t[4] scales_l + uint8_t[128] qs
// QK_K = 256 elements per block, 72 bytes total

// Optimized IQ4_XS dequant kernel for RDNA2:
// - Wave-cooperative scale extraction (threads 0-7 each handle one sub-block)
// - Direct global memory writes (no LDS staging)
template <bool need_check>
static __global__ void dequantize_block_iq4_xs_rdn2(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
    const int64_t i0 = 256LL * blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx + blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < 8) {
        const int sl_idx = tid / 2;
        const int sl_shift = (tid % 2) * 4;
        const int sh_shift = 2 * tid;
        const float dl = (float)x->d * (((float)((x->scales_l[sl_idx] >> sl_shift) & 0xf) + (float)((x->scales_h >> sh_shift) & 3) * 16.0f) - 32.0f);

        const uint8_t * q4 = x->qs + tid * 16;
        half * out_lower = y + i0 + tid * 32;
        half * out_upper = y + i0 + tid * 32 + 16;

        for (int j = 0; j < 16; ++j) {
            uint8_t qbyte = q4[j];
            out_lower[j] = __float2half(dl * kvalues_iq4nl[qbyte & 0xF]);
            out_upper[j] = __float2half(dl * kvalues_iq4nl[qbyte >> 4]);
        }
    }
}

#ifdef RDNA2_MODULE_CACHE
// Module-based launch for reduced overhead (Phase 2D)
static hipModule_t g_rdn2_module = nullptr;
static hipFunction_t g_dequant_fn = nullptr;
static bool g_rdn2_module_init = false;

static void rdn2_module_init() {
    if (g_rdn2_module_init) return;

    const char* module_path = getenv("RDNA2_MODULE_PATH");
    if (module_path) {
        hipError_t err = hipModuleLoad(&g_rdn2_module, module_path);
        if (err == hipSuccess) {
            // Try both possible mangled names
            err = hipModuleGetFunction(&g_dequant_fn, g_rdn2_module,
                                       "_Z32dequantize_block_iq4_xs_rdn2ILb0EEvPKvP5__halfx");
            if (err != hipSuccess) {
                err = hipModuleGetFunction(&g_dequant_fn, g_rdn2_module,
                                           "_Z32dequantize_block_iq4_xs_rdn2ILb0EEvPKvP6__halfx");
            }
            if (err != hipSuccess) {
                fprintf(stderr, "RDNA2 module: failed to get function, falling back to launch\n");
                hipModuleUnload(g_rdn2_module);
                g_rdn2_module = nullptr;
            }
        } else {
            fprintf(stderr, "RDNA2 module: failed to load %s, falling back to launch\n", module_path);
        }
    }
    g_rdn2_module_init = true;
}

static void dequant_iq4_xs_rdn2_module(const void * vx, half * y, const int64_t k, cudaStream_t stream) {
    rdn2_module_init();
    if (!g_dequant_fn) {
        // Fallback to standard launch
        const int nb = (k + 255) / 256;
        dequantize_block_iq4_xs_rdn2<false><<<nb, 32, 0, stream>>>(vx, y, k);
        return;
    }

    const int nb = (k + 256 - 1) / 256;
    const void * p_vx = vx;
    half * p_y = y;
    const int64_t p_k = k;
    void* args[] = { (void*)&p_vx, (void*)&p_y, (void*)&p_k };
    hipModuleLaunchKernel(g_dequant_fn, nb, 1, 1, 32, 1, 1, 0, stream, args, nullptr);
}
#endif

// Host wrapper: launches the RDNA2 IQ4_XS dequant kernel
static void ggml_dequant_iq4_xs_rdn2(const void * vx, half * y, const int64_t k, cudaStream_t stream) {
#ifdef RDNA2_MODULE_CACHE
    dequant_iq4_xs_rdn2_module(vx, y, k, stream);
#else
    const int nb = (k + 256 - 1) / 256;
    if (k % 256 == 0) {
        dequantize_block_iq4_xs_rdn2<false><<<nb, 32, 0, stream>>>(vx, y, k);
    } else {
        dequantize_block_iq4_xs_rdn2<true><<<nb, 32, 0, stream>>>(vx, y, k);
    }
#endif
}

#endif // GGML_CUDA_IQ4_DEQUANT_RDN2_CUH
