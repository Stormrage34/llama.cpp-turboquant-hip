// Microbenchmark for RDNA2-optimized IQ4_XS dequant
#define GGML_COMMON_DECL_HIP
#include "ggml-common.h"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

extern "C" void ggml_dequant_iq4_xs_rdn2_extern(const void * vx, half * y, const int64_t k, hipStream_t stream);

int main() {
    printf("IQ4_XS dequant microbench start\n");

    // VRAM guard: abort if usage exceeds 13.5 GB
    size_t free, total;
    hipMemGetInfo(&free, &total);
    size_t used = total - free;
    if (used > 13.5 * 1024 * 1024 * 1024ULL) {
        fprintf(stderr, "VRAM GUARD ABORT: %.2f GB used (cap: 13.5 GB)\n", used / 1e9);
        exit(1);
    }

    const int64_t k = 16 * 1024 * 1024; // 16M elements
    const size_t bytes_q = ((size_t)k / 256) * sizeof(block_iq4_xs);

    void *d_src; half *d_out;
    hipMalloc(&d_src, bytes_q);
    hipMalloc(&d_out, k * sizeof(half));
    hipMemset(d_src, 0x5A, bytes_q);

    hipStream_t stream;
hipStreamCreate(&stream);
hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
hipDeviceSynchronize();
hipEventRecord(s, stream);
ggml_dequant_iq4_xs_rdn2_extern(d_src, d_out, k, stream);
hipEventRecord(e, stream);
hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e);
    double bw = (double)bytes_q / (ms/1000.0) / (1024.0*1024.0*1024.0);
    printf("time: %.3f ms, GB/s: %.2f\n", ms, bw);

    hipFree(d_src); hipFree(d_out);
    hipEventDestroy(s); hipEventDestroy(e);
    return 0;
}
