---
name: hip-kernel-patterns
description: Reusable HIP kernel templates for RDNA2 MoE inference
triggers: ["kernel", "template", "mmvq", "dequant", "stream-k", "tile"]
---
### 2. `hip-kernel-patterns.md`
# HIP Kernel Patterns for RDNA2 MoE (gfx1030)

## Pattern 1: Fused Dequant + Matmul (Hot Path)
```cpp
// ggml/src/ggml-hip/mmvq.cu - Stream-K fused kernel
__global__ void mul_mat_vec_q_fused(
    const void* __restrict__ weights_q4,  // Q4_K quantized weights
    const float* __restrict__ activations,
    float* __restrict__ output,
    int n_rows, int n_cols, int block_size
) {
    // Stream-K: split work across CUs, reduce at end
    int stream_id = blockIdx.x % num_streams;
    int tile_start = stream_id * tile_size;
    
    // Shared memory tiling (32x33 stride avoids bank conflicts)
    __shared__ float tile[32][33];  // lds_bank_pad=2
    
    // Load quantized block, dequant inline to registers
    #pragma unroll 4
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        uint4 q_block = ((uint4*)weights_q4)[i];
        // Inline dequant: nibble extract → FP16 → FP32 accumulate
        // Use v_dot4c_i32_i8 for RDNA2 dot-product acceleration
    }
    
    // Accumulate in VGPRs, write final result
    if (threadIdx.x == 0) {
        output[stream_id] = accumulator;
    }
}
