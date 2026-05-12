// Unit test for RDNA2-optimized IQ4_XS dequant
// Compares CPU baseline vs rdn2 GPU dequant outputs
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define GGML_COMMON_DECL_HIP
#include "ggml-common.h"
#include <hip/hip_runtime.h>

extern "C" void dequantize_row_iq4_xs(const void * x, float * y, int64_t k);
#ifdef RDNA2_OPT_V1
extern "C" void ggml_dequant_iq4_xs_rdn2(const void * vx, half * y, const int64_t k);
#endif

int main() {
    printf("RDNA2 IQ4_XS dequant unit test\n");

    // VRAM guard: abort if usage exceeds 13.5 GB
    size_t free, total;
    hipMemGetInfo(&free, &total);
    size_t used = total - free;
    if (used > 13.5 * 1024 * 1024 * 1024ULL) {
        fprintf(stderr, "VRAM GUARD ABORT: %.2f GB used (cap: 13.5 GB)\n", used / 1e9);
        exit(1);
    }

    const int64_t k = 256 * 64; // 64 blocks of 256 elements
    const size_t bytes_q = ((size_t)k / 256) * sizeof(block_iq4_xs);

    // allocate host source packed buffer
    std::vector<uint8_t> src(bytes_q);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i & 0xFF);

    // CPU baseline (float output)
    std::vector<float> baseline_f(k);
    dequantize_row_iq4_xs(src.data(), baseline_f.data(), k);

#ifdef RDNA2_OPT_V1
    // GPU RDNA2 (half output)
    std::vector<half> rdn2_h(k);
    void *d_src = nullptr; half *d_out = nullptr;
    hipMalloc(&d_src, src.size());
    hipMalloc(&d_out, k * sizeof(half));
    hipMemcpy(d_src, src.data(), src.size(), hipMemcpyHostToDevice);
    ggml_dequant_iq4_xs_rdn2(d_src, d_out, k);
    hipMemcpy(rdn2_h.data(), d_out, k * sizeof(half), hipMemcpyDeviceToHost);
    hipFree(d_src); hipFree(d_out);

    // Compare: convert baseline float to double, rdn2 half to float, compute MSE
    double mse = 0.0; size_t n = k;
    for (size_t i = 0; i < n; ++i) {
        float a = baseline_f[i];
        float b = __half2float(rdn2_h[i]);
        double diff = (double)a - (double)b;
        mse += diff*diff;
    }
    mse /= (double)n;
    printf("MSE: %.8e\n", mse);
    if (!(mse < 1e-4)) {
        printf("MSE threshold failed (target <1e-4)\n");
        return 2;
    }
#else
    printf("RDNA2_OPT_V1 not compiled; test skipped\n");
#endif

    printf("RDNA2 IQ4_XS dequant unit test passed\n");
    return 0;
}
