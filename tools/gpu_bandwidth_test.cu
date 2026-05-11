#include <hip/hip_runtime.h>
#include <stdio.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Simple device-to-device copy kernel
__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("        HIP GPU Memory Bandwidth Test\n");
    printf("============================================================\n\n");

    // Query device properties
    int deviceId = 0;
    HIP_CHECK(hipGetDevice(&deviceId));

    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, deviceId));

    printf("Device: %s\n", deviceProp.name);
    printf("Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory clock rate: %.2f GHz\n", deviceProp.memoryClockRate / 1e6);
    // For AMD GPUs, use l2CacheSize and compute units to estimate peak
    // RX 6800 XT: 16 Gbps * 256-bit bus / 8 = 512 GB/s
    // Use memoryClockRate and typical bus width for RDNA2 (256-bit)
    double bus_width = 256.0;  // bits, typical for RDNA2
    double theoretical_gbps = (deviceProp.memoryClockRate / 1e6) * bus_width / 8.0;
    printf("Theoretical memory bandwidth: %.0f GB/s\n", theoretical_gbps);
    printf("(Based on memory clock rate and 256-bit bus)\n\n");

    // Large buffer sizes for accurate measurement
    const size_t MB = 1024 * 1024;
    const size_t GB = 1024 * 1024 * 1024;
    
    // Test with 512MB buffers
    const size_t buffer_size = 512 * MB;
    const size_t num_floats = buffer_size / sizeof(float);
    
    printf("Test configuration:\n");
    printf("  Buffer size: %zu MB (%.2f GB)\n", buffer_size / MB, (float)buffer_size / GB);
    printf("  Number of elements: %zu\n\n", num_floats);

    // Allocate device buffers
    float* d_src;
    float* d_dst;
    HIP_CHECK(hipMalloc(&d_src, buffer_size));
    HIP_CHECK(hipMalloc(&d_dst, buffer_size));

    // Allocate pinned host memory for comparison
    float* h_src;
    float* h_dst;
    HIP_CHECK(hipHostMalloc(&h_src, buffer_size, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_dst, buffer_size, hipHostMallocDefault));

    // Initialize host buffer with data
    printf("Initializing host buffer with test pattern...\n");
    for (size_t i = 0; i < num_floats; i++) {
        h_src[i] = (float)(i & 0xFF);
    }

    // Copy to device first
    printf("Copying data to device...\n");
    HIP_CHECK(hipMemcpy(d_src, h_src, buffer_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Create events for precise timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Configuration
    const int block_size = 256;
    const int grid_size = deviceProp.multiProcessorCount * 16; // ~16 blocks per SM
    const int num_iterations = 100;

    printf("Kernel configuration:\n");
    printf("  Block size: %d\n", block_size);
    printf("  Grid size: %d\n", grid_size);
    printf("  Iterations per test: %d\n\n", num_iterations);

    printf("============================================================\n");
    printf("                    TEST RESULTS\n");
    printf("============================================================\n\n");

    // Warm-up
    printf("Warming up...\n");
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(copy_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                          d_dst, d_src, num_floats);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Test 1: Device-to-device copy (pure GPU memory bandwidth)
    printf("TEST 1: Device-to-Device Memory Copy\n");
    printf("----------------------------------------\n");
    
    float total_time_d2d = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(copy_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                          d_dst, d_src, num_floats);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        total_time_d2d += elapsed;
    }

    float avg_time_d2d = total_time_d2d / num_iterations;
    double bandwidth_d2d = (buffer_size * 2.0 / (1024.0 * 1024.0 * 1024.0)) / (avg_time_d2d / 1000.0);
    
    printf("  Iterations: %d\n", num_iterations);
    printf("  Average time: %.3f ms\n", avg_time_d2d);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_d2d);
    printf("  %% of theoretical peak: %.1f%%\n\n", (bandwidth_d2d / theoretical_gbps) * 100.0);

    // Test 2: Host-to-device ( PCIe bandwidth)
    printf("TEST 2: Host-to-Device Copy (PCIe)\n");
    printf("----------------------------------------\n");
    
    float total_time_h2d = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        HIP_CHECK(hipEventRecord(start));
        HIP_CHECK(hipMemcpy(d_src, h_src, buffer_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        total_time_h2d += elapsed;
    }

    float avg_time_h2d = total_time_h2d / num_iterations;
    double bandwidth_h2d = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_time_h2d / 1000.0);
    
    printf("  Iterations: %d\n", num_iterations);
    printf("  Average time: %.3f ms\n", avg_time_h2d);
    printf("  Bandwidth: %.2f GB/s\n\n", bandwidth_h2d);

    // Test 3: Host-to-device with pinned memory
    printf("TEST 3: Host-to-Device (Pinned Memory)\n");
    printf("----------------------------------------\n");
    
    float total_time_pinned = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        HIP_CHECK(hipEventRecord(start));
        HIP_CHECK(hipMemcpy(d_src, h_src, buffer_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        total_time_pinned += elapsed;
    }

    float avg_time_pinned = total_time_pinned / num_iterations;
    double bandwidth_pinned = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (avg_time_pinned / 1000.0);
    
    printf("  Iterations: %d\n", num_iterations);
    printf("  Average time: %.3f ms\n", avg_time_pinned);
    printf("  Bandwidth: %.2f GB/s\n\n", bandwidth_pinned);

    // Summary
    printf("============================================================\n");
    printf("                      SUMMARY\n");
    printf("============================================================\n\n");
    printf("%-30s %10.2f GB/s\n", "Device-to-Device (GPU VRAM):", bandwidth_d2d);
    printf("%-30s %10.2f GB/s\n", "Host-to-Device (PCIe):", bandwidth_h2d);
    printf("%-30s %10.2f GB/s\n\n", "Theoretical Peak:", theoretical_gbps);

    printf("Conclusion: ");
    if (bandwidth_d2d > 300.0) {
        printf("GPU memory bandwidth is working correctly (%.1f%% of peak)\n", 
               (bandwidth_d2d / theoretical_gbps) * 100.0);
    } else {
        printf("LOW bandwidth detected - system/configuration issue\n");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipHostFree(h_src));
    HIP_CHECK(hipHostFree(h_dst));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    printf("\n============================================================\n");

    return 0;
}