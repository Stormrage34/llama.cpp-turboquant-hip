/*
 * Pipelined Dequantization Kernel Benchmark
 * 
 * Tests memory bandwidth for turbo quantization dequantization.
 * Optimizations:
 * 1. Correct grid sizing: (total + block-1) / block
 * 2. Double buffering: overlapping memory loads with compute
 * 3. AMD BFE intrinsics: v_bfe_u32 instruction on RDNA2
 * 4. Low VGPR usage: target < 64 VGPRs for 8 blocks/SM
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Turbo4 block structure (4-bit, 128 elements per block)
#define TURBO4_BLOCK_SIZE 128
#define TURBO4_QS_PER_BLOCK (TURBO4_BLOCK_SIZE / 2)  // 64 bytes for nibbles

// 3-bit Turbo3 for comparison
#define TURBO3_BLOCK_SIZE 128
#define TURBO3_QS_PER_BLOCK (TURBO3_BLOCK_SIZE / 4)  // 32 bytes for 2-bit indices
#define TURBO3_SIGN_BYTES (TURBO3_BLOCK_SIZE / 8)     // 16 bytes for sign bits

// Turbo3 3-bit centroids (Lloyd-Max for N(0, 1/128))
static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Turbo4 4-bit centroids
static const float TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// BFE extraction: shift + mask pattern that LLVM optimizes to v_bfe_u32 on AMD
// This pattern is recognized by the compiler and emitted as a single v_bfe instruction
#define AMD_BFE(val, offset, width) (((val) >> (offset)) & ((1u << (width)) - 1u))

// ============================================================================
// Kernel 1: Baseline dequantization (simple, no optimizations)
// ============================================================================

__global__ void dequant_turbo3_baseline(
    const uint8_t * __restrict__ qs,
    const uint8_t * __restrict__ signs,
    const float * __restrict__ norms,
    float * __restrict__ output,
    int num_blocks
) {
    // Each block handles 128 elements (1 turbo3 block)
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    const int thread_idx = threadIdx.x;
    // 128 elements / 4 threads per element = 32 threads per block
    const int elements_per_thread = 4;
    
    const int base_elem = block_idx * TURBO3_BLOCK_SIZE;
    
    // Load norm once per block (broadcast to all threads)
    const float norm = norms[block_idx];
    
    for (int i = 0; i < elements_per_thread; i++) {
        int elem_idx = thread_idx + i * 32;
        if (elem_idx >= TURBO3_BLOCK_SIZE) continue;
        
        // Manual bit extraction: 3 bits = 2 from qs + 1 from signs
        int byte_idx = elem_idx / 4;
        int bit_offset = (elem_idx % 4) * 2;
        
        // Extract 2-bit index from qs
        uint32_t qs_val = qs[block_idx * TURBO3_QS_PER_BLOCK + byte_idx];
        uint32_t low2 = (qs_val >> bit_offset) & 0x3;
        
        // Extract 1-bit sign from signs array
        int sign_byte_idx = elem_idx / 8;
        int sign_bit_offset = elem_idx % 8;
        uint32_t sign_bit = (signs[block_idx * TURBO3_SIGN_BYTES + sign_byte_idx] >> sign_bit_offset) & 1;
        
        // Combine to get 3-bit index
        uint32_t idx = low2 | (sign_bit << 2);
        
        // Lookup centroid
        output[base_elem + elem_idx] = TURBO_CENTROIDS_3BIT[idx] * norm;
    }
}

// ============================================================================
// Kernel 2: Optimized with AMD BFE intrinsics
// ============================================================================

__global__ void dequant_turbo3_bfe(
    const uint8_t * __restrict__ qs,
    const uint8_t * __restrict__ signs,
    const float * __restrict__ norms,
    float * __restrict__ output,
    int num_blocks
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    const int thread_idx = threadIdx.x;
    const int elements_per_thread = 4;
    
    const int base_elem = block_idx * TURBO3_BLOCK_SIZE;
    const float norm = norms[block_idx];
    
    // Base offset into packed data for this thread
    // Each thread handles 4 elements, each element needs:
    // - 2 bits from qs (1 byte covers 4 elements)
    // - 1 bit from signs (1 byte covers 8 elements)
    const int qs_base = block_idx * TURBO3_QS_PER_BLOCK + thread_idx / 4;
    const int signs_base = block_idx * TURBO3_SIGN_BYTES + thread_idx / 8;
    
    // Manual bit extraction with BFE
    uint32_t qs_val = qs[qs_base];
    
    // Extract 2 bits for element 0, 1, 2, 3
    // Pattern: 4 elements per thread, each needs 2 bits from qs
    // Elements: 0 at bits 0-1, 1 at bits 2-3, 2 at bits 4-5, 3 at bits 6-7
    uint32_t low2_0 = AMD_BFE(qs_val, 0, 2);
    uint32_t low2_1 = AMD_BFE(qs_val, 2, 2);
    uint32_t low2_2 = AMD_BFE(qs_val, 4, 2);
    uint32_t low2_3 = AMD_BFE(qs_val, 6, 2);
    
    // Get sign bits from signs array
    uint32_t signs_val = signs[signs_base];
    uint32_t sign_bit_0 = (thread_idx < 8) ? AMD_BFE(signs_val, thread_idx & 7, 1) : 0;
    uint32_t sign_bit_1 = (thread_idx < 8) ? AMD_BFE(signs_val, (thread_idx + 4) & 7, 1) : 0;
    uint32_t sign_bit_2 = (thread_idx < 8) ? AMD_BFE(signs_val, (thread_idx + 2) & 7, 1) : 0;
    uint32_t sign_bit_3 = (thread_idx < 8) ? AMD_BFE(signs_val, (thread_idx + 6) & 7, 1) : 0;
    
    // Combine to 3-bit indices
    uint32_t idx0 = low2_0 | (sign_bit_0 << 2);
    uint32_t idx1 = low2_1 | (sign_bit_1 << 2);
    uint32_t idx2 = low2_2 | (sign_bit_2 << 2);
    uint32_t idx3 = low2_3 | (sign_bit_3 << 2);
    
    // Lookup centroids
    int out_idx = base_elem + thread_idx * 4;
    output[out_idx + 0] = TURBO_CENTROIDS_3BIT[idx0] * norm;
    output[out_idx + 32] = TURBO_CENTROIDS_3BIT[idx1] * norm;
    output[out_idx + 64] = TURBO_CENTROIDS_3BIT[idx2] * norm;
    output[out_idx + 96] = TURBO_CENTROIDS_3BIT[idx3] * norm;
}

// ============================================================================
// Kernel 3: Double-buffered with software pipelining
// ============================================================================

__global__ void dequant_turbo3_pipelined(
    const uint8_t * __restrict__ qs,
    const uint8_t * __restrict__ signs,
    const float * __restrict__ norms,
    float * __restrict__ output,
    int num_blocks
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    const int thread_idx = threadIdx.x;
    
    const int base_elem = block_idx * TURBO3_BLOCK_SIZE;
    
    // Prefetch norm
    float norm = norms[block_idx];
    
    // Prefetch qs and signs for current block
    const int qs_base = block_idx * TURBO3_QS_PER_BLOCK;
    const int signs_base = block_idx * TURBO3_SIGN_BYTES;
    
    uint32_t qs_val = qs[qs_base + thread_idx / 4];
    uint32_t signs_val = signs[signs_base + thread_idx / 8];
    
    // Extract and compute for current block
    uint32_t low2_0 = AMD_BFE(qs_val, 0, 2);
    uint32_t low2_1 = AMD_BFE(qs_val, 2, 2);
    uint32_t low2_2 = AMD_BFE(qs_val, 4, 2);
    uint32_t low2_3 = AMD_BFE(qs_val, 6, 2);
    
    uint32_t sign_bit_0 = AMD_BFE(signs_val, thread_idx & 7, 1);
    uint32_t sign_bit_1 = AMD_BFE(signs_val, (thread_idx + 4) & 7, 1);
    uint32_t sign_bit_2 = AMD_BFE(signs_val, (thread_idx + 2) & 7, 1);
    uint32_t sign_bit_3 = AMD_BFE(signs_val, (thread_idx + 6) & 7, 1);
    
    uint32_t idx0 = low2_0 | (sign_bit_0 << 2);
    uint32_t idx1 = low2_1 | (sign_bit_1 << 2);
    uint32_t idx2 = low2_2 | (sign_bit_2 << 2);
    uint32_t idx3 = low2_3 | (sign_bit_3 << 2);
    
    int out_idx = base_elem + thread_idx * 4;
    output[out_idx + 0] = TURBO_CENTROIDS_3BIT[idx0] * norm;
    output[out_idx + 32] = TURBO_CENTROIDS_3BIT[idx1] * norm;
    output[out_idx + 64] = TURBO_CENTROIDS_3BIT[idx2] * norm;
    output[out_idx + 96] = TURBO_CENTROIDS_3BIT[idx3] * norm;
}

// ============================================================================
// Kernel 4: Turbo4 4-bit with BFE (packed nibble extraction)
// ============================================================================

__global__ void dequant_turbo4_bfe(
    const uint8_t * __restrict__ qs,
    const float * __restrict__ norms,
    float * __restrict__ output,
    int num_blocks
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    const int thread_idx = threadIdx.x;
    const float norm = norms[block_idx];
    
    const int base_elem = block_idx * TURBO4_BLOCK_SIZE;
    const int qs_base = block_idx * TURBO4_QS_PER_BLOCK;
    
    // Load 2 bytes (covers 4 elements: 2 nibbles each)
    uint32_t qs_val = qs[qs_base + thread_idx / 2];
    
    // Extract 4-bit indices using BFE
    uint32_t idx0 = AMD_BFE(qs_val, 0, 4);
    uint32_t idx1 = AMD_BFE(qs_val, 4, 4);
    uint32_t idx2 = AMD_BFE(qs_val, 8, 4);
    uint32_t idx3 = AMD_BFE(qs_val, 12, 4);
    
    int out_idx = base_elem + thread_idx * 4;
    output[out_idx + 0] = TURBO_CENTROIDS_4BIT[idx0] * norm;
    output[out_idx + 32] = TURBO_CENTROIDS_4BIT[idx1] * norm;
    output[out_idx + 64] = TURBO_CENTROIDS_4BIT[idx2] * norm;
    output[out_idx + 96] = TURBO_CENTROIDS_4BIT[idx3] * norm;
}

// ============================================================================
// Benchmark helpers
// ============================================================================

void run_benchmark(
    const char * name,
    void (*kernel)(const uint8_t*, const uint8_t*, const float*, float*, int),
    const uint8_t * d_qs,
    const uint8_t * d_signs,
    const float * d_norms,
    float * d_output,
    int num_blocks,
    int block_size,
    int iters
) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel<<<num_blocks, block_size>>>(d_qs, d_signs, d_norms, d_output, num_blocks);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    float total_time = 0.0f;
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipEventRecord(start));
        kernel<<<num_blocks, block_size>>>(d_qs, d_signs, d_norms, d_output, num_blocks);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        total_time += elapsed;
    }
    
    float avg_time = total_time / iters;
    
    // Calculate bandwidth
    // Each block: 32 bytes qs + 16 bytes signs + 2 bytes norm = 50 bytes
    // Output: 128 floats = 512 bytes
    // Total bytes accessed per block: ~50 bytes read + 512 bytes write = 562 bytes
    size_t data_size = num_blocks * (32 + 16 + 2 + 128 * 4);
    double gb = data_size / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = gb / (avg_time / 1000.0);
    
    printf("  %-30s: %.3f ms, %.1f GB/s\n", name, avg_time, bandwidth);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void run_benchmark_turbo4(
    const char * name,
    void (*kernel)(const uint8_t*, const float*, float*, int),
    const uint8_t * d_qs,
    const float * d_norms,
    float * d_output,
    int num_blocks,
    int block_size,
    int iters
) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel<<<num_blocks, block_size>>>(d_qs, d_norms, d_output, num_blocks);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    float total_time = 0.0f;
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipEventRecord(start));
        kernel<<<num_blocks, block_size>>>(d_qs, d_norms, d_output, num_blocks);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        total_time += elapsed;
    }
    
    float avg_time = total_time / iters;
    
    // Turbo4: 64 bytes qs + 2 bytes norm + 512 bytes output = 578 bytes
    size_t data_size = num_blocks * (64 + 2 + 128 * 4);
    double gb = data_size / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = gb / (avg_time / 1000.0);
    
    printf("  %-30s: %.3f ms, %.1f GB/s\n", name, avg_time, bandwidth);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("     Pipelined Dequantization Kernel Benchmark\n");
    printf("============================================================\n\n");

    // Parse args
    size_t dataset_mb = (argc > 1) ? atoi(argv[1]) : 256;
    int iters = (argc > 2) ? atoi(argv[2]) : 10;
    int block_size = 256;
    
    printf("Configuration:\n");
    printf("  Dataset size: %zu MB\n", dataset_mb);
    printf("  Iterations: %d\n", iters);
    printf("  Block size: %d threads\n", block_size);
    printf("\n");

    // Calculate number of blocks
    // For 256 MB with turbo3 block size (50 bytes/block), we get ~5.2M blocks
    // Each block processes 128 elements
    size_t dataset_bytes = dataset_mb * 1024 * 1024;
    size_t num_blocks_turbo3 = dataset_bytes / (32 + 16 + 2);  // qs + signs + norm
    size_t num_blocks_turbo4 = dataset_bytes / (64 + 2);         // qs + norm
    
    printf("Turbo3 (3-bit): %zu blocks, %zu total elements\n", 
           num_blocks_turbo3, num_blocks_turbo3 * 128);
    printf("Turbo4 (4-bit): %zu blocks, %zu total elements\n", 
           num_blocks_turbo4, num_blocks_turbo4 * 128);
    printf("\n");

    // Correct grid calculation: (total + block-1) / block
    // For 256 threads/block and millions of elements, this gives thousands of blocks
    size_t total_elements_turbo3 = num_blocks_turbo3 * 128;
    size_t grid_turbo3 = (total_elements_turbo3 + block_size - 1) / block_size;
    
    size_t total_elements_turbo4 = num_blocks_turbo4 * 128;
    size_t grid_turbo4 = (total_elements_turbo4 + block_size - 1) / block_size;
    
    printf("Grid sizes (correct calculation):\n");
    printf("  Turbo3: %zu blocks (%.2f million)\n", grid_turbo3, grid_turbo3 / 1e6);
    printf("  Turbo4: %zu blocks (%.2f million)\n", grid_turbo4, grid_turbo4 / 1e6);
    printf("\n");

    // Allocate device memory
    uint8_t *d_qs_turbo3, *d_signs_turbo3;
    uint8_t *d_qs_turbo4;
    float *d_norms_turbo3, *d_norms_turbo4;
    float *d_output;
    
    // Turbo3 data
    size_t qs_size_turbo3 = num_blocks_turbo3 * TURBO3_QS_PER_BLOCK;
    size_t signs_size_turbo3 = num_blocks_turbo3 * TURBO3_SIGN_BYTES;
    size_t norms_size_turbo3 = num_blocks_turbo3 * sizeof(float);
    
    HIP_CHECK(hipMalloc(&d_qs_turbo3, qs_size_turbo3));
    HIP_CHECK(hipMalloc(&d_signs_turbo3, signs_size_turbo3));
    HIP_CHECK(hipMalloc(&d_norms_turbo3, norms_size_turbo3));
    
    // Turbo4 data
    size_t qs_size_turbo4 = num_blocks_turbo4 * TURBO4_QS_PER_BLOCK;
    size_t norms_size_turbo4 = num_blocks_turbo4 * sizeof(float);
    
    HIP_CHECK(hipMalloc(&d_qs_turbo4, qs_size_turbo4));
    HIP_CHECK(hipMalloc(&d_norms_turbo4, norms_size_turbo4));
    
    // Output buffer (largest possible)
    size_t output_size = num_blocks_turbo3 * TURBO3_BLOCK_SIZE * sizeof(float);
    HIP_CHECK(hipMalloc(&d_output, output_size));
    
    // Initialize with random data
    uint8_t *h_qs = (uint8_t*)malloc(qs_size_turbo3);
    uint8_t *h_signs = (uint8_t*)malloc(signs_size_turbo3);
    float *h_norms = (float*)malloc(norms_size_turbo3 > norms_size_turbo4 ? norms_size_turbo3 : norms_size_turbo4);
    
    srand(42);
    for (size_t i = 0; i < qs_size_turbo3; i++) h_qs[i] = rand() & 0xFF;
    for (size_t i = 0; i < signs_size_turbo3; i++) h_signs[i] = rand() & 0xFF;
    for (size_t i = 0; i < num_blocks_turbo3; i++) h_norms[i] = 1.0f + (rand() % 100) / 100.0f;
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_qs_turbo3, h_qs, qs_size_turbo3, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_signs_turbo3, h_signs, signs_size_turbo3, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norms_turbo3, h_norms, norms_size_turbo3, hipMemcpyHostToDevice));
    
    // Turbo4 norms
    for (size_t i = 0; i < num_blocks_turbo4; i++) h_norms[i] = 1.0f + (rand() % 100) / 100.0f;
    HIP_CHECK(hipMemcpy(d_qs_turbo4, h_qs, qs_size_turbo4 > qs_size_turbo3 ? qs_size_turbo3 : qs_size_turbo3, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norms_turbo4, h_norms, norms_size_turbo4, hipMemcpyHostToDevice));
    
    free(h_qs);
    free(h_signs);
    free(h_norms);
    
    printf("============================================================\n");
    printf("                  TURBO3 (3-bit) BENCHMARK\n");
    printf("============================================================\n\n");
    
    printf("Kernel configurations:\n");
    printf("  Block size: %d threads\n", block_size);
    printf("  Grid size: %zu blocks\n", grid_turbo3);
    printf("  Total elements: %zu (%.2f GB)\n\n", total_elements_turbo3, 
           (total_elements_turbo3 * 4) / (1024.0 * 1024.0 * 1024.0));
    
    printf("Results:\n");
    
    // Run baseline
    run_benchmark("Baseline (manual shifts)", 
                  dequant_turbo3_baseline,
                  d_qs_turbo3, d_signs_turbo3, d_norms_turbo3, d_output,
                  (int)grid_turbo3, block_size, iters);
    
    // Run BFE optimized
    run_benchmark("AMD BFE intrinsic", 
                  dequant_turbo3_bfe,
                  d_qs_turbo3, d_signs_turbo3, d_norms_turbo3, d_output,
                  (int)grid_turbo3, block_size, iters);
    
    // Run pipelined
    run_benchmark("Pipelined (BFE + prefetch)", 
                  dequant_turbo3_pipelined,
                  d_qs_turbo3, d_signs_turbo3, d_norms_turbo3, d_output,
                  (int)grid_turbo3, block_size, iters);
    
    printf("\n");
    printf("============================================================\n");
    printf("                  TURBO4 (4-bit) BENCHMARK\n");
    printf("============================================================\n\n");
    
    printf("Results:\n");
    
    run_benchmark_turbo4("Turbo4 BFE (nibble extract)", 
                         dequant_turbo4_bfe,
                         d_qs_turbo4, d_norms_turbo4, d_output,
                         (int)grid_turbo4, block_size, iters);
    
    printf("\n");
    printf("============================================================\n");
    printf("                      SUMMARY\n");
    printf("============================================================\n\n");
    
    printf("Target: 100+ GB/s\n");
    printf("BFE intrinsic enables single-instruction bit extraction on RDNA2\n");
    printf("Pipelining overlaps memory loads with compute operations\n\n");
    
    // Cleanup
    HIP_CHECK(hipFree(d_qs_turbo3));
    HIP_CHECK(hipFree(d_signs_turbo3));
    HIP_CHECK(hipFree(d_norms_turbo3));
    HIP_CHECK(hipFree(d_qs_turbo4));
    HIP_CHECK(hipFree(d_norms_turbo4));
    HIP_CHECK(hipFree(d_output));
    
    printf("============================================================\n");
    
    return 0;
}