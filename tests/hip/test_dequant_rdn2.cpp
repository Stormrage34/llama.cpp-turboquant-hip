// Unit test for RDNA2-optimized dequant
// Minimal unit test to compare baseline vs rdn2 dequant outputs on device
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"
#include <hip/hip_runtime.h>
#include "../testing.h"

extern "C" void ggml_dequant_q8_0_baseline(const void * vx, half * y, const int64_t k);
#ifdef RDNA2_OPT_V1
extern "C" void ggml_dequant_q8_0_rdn2(const void * vx, half * y, const int64_t k);
#endif

int main() {
    printf("RDNA2 dequant unit test\n");

    const int64_t k = 1024; // elements
    const size_t bytes_q = ((size_t)k / QK8_0) * sizeof(block_q8_0);

    // allocate host source packed buffer and two outputs
    std::vector<uint8_t> src(bytes_q);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i & 0xFF);

    std::vector<half> out_baseline(k);
    std::vector<half> out_rdn2(k);

    // Device buffers via host pinned for simplicity - use existing helpers if available
    void *d_src = nullptr; half *d_out = nullptr;
    hipMalloc(&d_src, src.size()); hipMalloc(&d_out, k * sizeof(half));
    hipMemcpy(d_src, src.data(), src.size(), hipMemcpyHostToDevice);

    // baseline
    ggml_dequant_q8_0_baseline(d_src, d_out, k);
    hipMemcpy(out_baseline.data(), d_out, k * sizeof(half), hipMemcpyDeviceToHost);

#ifdef RDNA2_OPT_V1
    // optimized
    ggml_dequant_q8_0_rdn2(d_src, d_out, k);
    hipMemcpy(out_rdn2.data(), d_out, k * sizeof(half), hipMemcpyDeviceToHost);

    // compare
    double mse = 0.0; size_t n = k;
    for (size_t i = 0; i < n; ++i) {
        float a = __half2float(out_baseline[i]);
        float b = __half2float(out_rdn2[i]);
        double diff = (double)a - (double)b;
        mse += diff*diff;
    }
    mse /= (double)n;
    printf("MSE: %.8e\n", mse);
    if (!(mse < 1e-6)) {
        printf("MSE threshold failed\n");
        return 2;
    }
#else
    printf("RDNA2_OPT_V1 not compiled; test skipped\n");
#endif

    hipFree(d_src); hipFree(d_out);
    printf("RDNA2 dequant unit test passed\n");
    return 0;
}
