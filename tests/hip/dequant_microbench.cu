// Microbenchmark for dequant optimized path
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

extern "C" void ggml_dequant_q8_0_rdn2(const void * vx, void * y, const int64_t k);

int main() {
    printf("dequant microbench start\n");
    const int64_t k = 16 * 1024 * 1024; // large number of elements
    const size_t bytes_q = ((size_t)k / 256) * 256; // rough

    void *d_src; void *d_out;
    hipMalloc(&d_src, bytes_q);
    hipMalloc(&d_out, k * sizeof(short));
    hipMemset(d_src, 0x5A, bytes_q);

    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipDeviceSynchronize();
    hipEventRecord(s);
    ggml_dequant_q8_0_rdn2(d_src, (half*)d_out, k);
    hipEventRecord(e);
    hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e);
    double bw = (double)bytes_q / (ms/1000.0) / (1024.0*1024.0*1024.0);
    printf("time: %.3f ms, GB/s: %.2f\n", ms, bw);

    hipFree(d_src); hipFree(d_out);
    return 0;
}
