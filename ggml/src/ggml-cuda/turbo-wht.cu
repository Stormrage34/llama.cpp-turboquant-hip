#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ─── half4 vectorized WHT kernel ────────────────────────────────────────────
//
// Optimized for RDNA2 (gfx1030, wave-32): uses 32 threads for group_size=128,
// each thread processes 4 elements via float4. Reduces thread count 4x,
// enables 128-bit coalesced loads/stores, and cuts shared memory bank conflicts.
//
// Butterfly stages 1-2 operate within each thread's 4-element chunk (no sync needed).
// Stages 3+ require cross-thread communication via shared memory (__syncthreads).
//
// For group_size=64: 16 threads, each 4 elements.
// For group_size=32: falls back to scalar kernel (8 threads too few for float4).

template <int direction, int group_size>
static __global__ void k_turbo_wht_half4(const float * __restrict__ src,
                                          float * __restrict__ dst,
                                          const float * __restrict__ scale_inv,
                                          int64_t n_groups,
                                          int64_t head_dim,
                                          int64_t groups_per_head) {
    // Only for group_size=128 and 64 (need at least 16 threads for float4)
    static_assert(group_size == 128 || group_size == 64,
                  "half4 kernel requires group_size 128 or 64");

    constexpr int VEC_WIDTH = 4;  // float4 = 4 floats per thread
    constexpr int N_THREADS = group_size / VEC_WIDTH;  // 32 for 128, 16 for 64

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. N_THREADS-1
    const int t4 = t * VEC_WIDTH;  // first element index for this thread

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    // --- Load from global memory (128-bit coalesced) ---
    float4 tmp = reinterpret_cast<const float4 *>(&src[base + t4])[0];
    // Store to shared memory as individual floats for butterfly access
    x[t4 + 0] = tmp.x;
    x[t4 + 1] = tmp.y;
    x[t4 + 2] = tmp.z;
    x[t4 + 3] = tmp.w;
    __syncthreads();

    // --- InnerQ forward: apply scale_inv BEFORE signs+WHT ---
    if (direction == 0 && scale_inv != nullptr) {
        x[t4 + 0] *= scale_inv[t4 + 0];
        x[t4 + 1] *= scale_inv[t4 + 1];
        x[t4 + 2] *= scale_inv[t4 + 2];
        x[t4 + 3] *= scale_inv[t4 + 3];
        __syncthreads();
    }

    // --- Apply first sign array (4 signs per thread, from packed constant memory) ---
    if (group_size == 128) {
        float4 s1 = reinterpret_cast<const float4 *>(&TURBO_WHT_SIGNS1[t4])[0];
        x[t4 + 0] *= s1.x;
        x[t4 + 1] *= s1.y;
        x[t4 + 2] *= s1.z;
        x[t4 + 3] *= s1.w;
    } else {
        // group_size == 64
        float4 s1 = reinterpret_cast<const float4 *>(&TURBO_WHT_SIGNS1_64[t4])[0];
        x[t4 + 0] *= s1.x;
        x[t4 + 1] *= s1.y;
        x[t4 + 2] *= s1.z;
        x[t4 + 3] *= s1.w;
    }
    __syncthreads();

    // --- WHT butterfly ---
    // Stages 1-2: within-thread (each thread owns 4 consecutive elements)
    // Stage h=1: pairs (t4+0, t4+1) and (t4+2, t4+3)
    {
        float a0 = x[t4], b0 = x[t4 + 1];
        float a1 = x[t4 + 2], b1 = x[t4 + 3];
        x[t4]     = a0 + b0;
        x[t4 + 1] = a0 - b0;
        x[t4 + 2] = a1 + b1;
        x[t4 + 3] = a1 - b1;
    }
    // Stage h=2: pairs (t4+0, t4+2) and (t4+1, t4+3)
    {
        float a0 = x[t4], b0 = x[t4 + 2];
        float a1 = x[t4 + 1], b1 = x[t4 + 3];
        x[t4]     = a0 + b0;
        x[t4 + 2] = a0 - b0;
        x[t4 + 1] = a1 + b1;
        x[t4 + 3] = a1 - b1;
    }

    // Stages 3+: cross-thread via shared memory
    // Stage h=4: thread t pairs with thread t^1 (flip bit 0)
    // Stage h=8: thread t pairs with thread t^2 (flip bit 1)
    // Stage h=16: thread t pairs with thread t^4 (flip bit 2)
    // Stage h=32: thread t pairs with thread t^8 (flip bit 3)
    // Stage h=64: thread t pairs with thread t^16 (flip bit 4)

    // h=4: distance 4 elements = 1 thread offset
    if ((t % 2) == 0) {
        float a = x[t4], b = x[t4 + 4];
        x[t4]     = a + b;
        x[t4 + 4] = a - b;
        a = x[t4 + 1]; b = x[t4 + 5];
        x[t4 + 1] = a + b;
        x[t4 + 5] = a - b;
        a = x[t4 + 2]; b = x[t4 + 6];
        x[t4 + 2] = a + b;
        x[t4 + 6] = a - b;
        a = x[t4 + 3]; b = x[t4 + 7];
        x[t4 + 3] = a + b;
        x[t4 + 7] = a - b;
    }
    __syncthreads();

    // h=8: distance 8 elements = 2 thread offsets
    if ((t % 4) < 2) {
        for (int i = 0; i < VEC_WIDTH; ++i) {
            float a = x[t4 + i], b = x[t4 + i + 8];
            x[t4 + i]     = a + b;
            x[t4 + i + 8] = a - b;
        }
    }
    __syncthreads();

    // h=16: distance 16 elements = 4 thread offsets
    if ((t % 8) < 4) {
        for (int i = 0; i < VEC_WIDTH; ++i) {
            float a = x[t4 + i], b = x[t4 + i + 16];
            x[t4 + i]      = a + b;
            x[t4 + i + 16] = a - b;
        }
    }
    __syncthreads();

    if constexpr (group_size >= 64) {
        // h=32: distance 32 elements = 8 thread offsets
        if ((t % 16) < 8) {
            for (int i = 0; i < VEC_WIDTH; ++i) {
                float a = x[t4 + i], b = x[t4 + i + 32];
                x[t4 + i]      = a + b;
                x[t4 + i + 32] = a - b;
            }
        }
        __syncthreads();
    }

    if constexpr (group_size == 128) {
        // h=64: distance 64 elements = 16 thread offsets
        if (t < 16) {
            for (int i = 0; i < VEC_WIDTH; ++i) {
                float a = x[t4 + i], b = x[t4 + i + 64];
                x[t4 + i]      = a + b;
                x[t4 + i + 64] = a - b;
            }
        }
        __syncthreads();
    }

    // --- Normalize and apply second sign array ---
    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;
    float4 result;
    if (group_size == 128) {
        float4 s2 = reinterpret_cast<const float4 *>(&TURBO_WHT_SIGNS2[t4])[0];
        result.x = x[t4 + 0] * inv_sqrt * s2.x;
        result.y = x[t4 + 1] * inv_sqrt * s2.y;
        result.z = x[t4 + 2] * inv_sqrt * s2.z;
        result.w = x[t4 + 3] * inv_sqrt * s2.w;
    } else {
        float4 s2 = reinterpret_cast<const float4 *>(&TURBO_WHT_SIGNS2_64[t4])[0];
        result.x = x[t4 + 0] * inv_sqrt * s2.x;
        result.y = x[t4 + 1] * inv_sqrt * s2.y;
        result.z = x[t4 + 2] * inv_sqrt * s2.z;
        result.w = x[t4 + 3] * inv_sqrt * s2.w;
    }

    // InnerQ inverse: apply scale_inv AFTER WHT+signs
    if (direction == 1 && scale_inv != nullptr) {
        result.x *= scale_inv[t4 + 0];
        result.y *= scale_inv[t4 + 1];
        result.z *= scale_inv[t4 + 2];
        result.w *= scale_inv[t4 + 3];
    }

    // --- Store to global memory (128-bit coalesced) ---
    reinterpret_cast<float4 *>(&dst[base + t4])[0] = result;
}

// ─── Scalar WHT kernel (original, for group_size=32 and fallback) ────────────
//
// Templated on direction and group_size (128 or 64).
// One block per group, group_size threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// When head_dim is not a multiple of group_size, only the full groups
// within each head are processed.  Tail elements are left unchanged (identity).
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (log2(group_size) stages, in-place)
//   3. Normalize by 1/sqrt(group_size) and apply s_second elementwise
//
// InnerQ scale_inv: when non-null, applies per-channel inverse scaling for
// Q/V equalization. For forward (Q rotation): multiply BEFORE signs+WHT.
// For inverse (V un-rotation): multiply AFTER WHT+signs.

template <int direction, int group_size>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const float * __restrict__ scale_inv,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    static_assert(group_size == 128 || group_size == 64 || group_size == 32, "group_size must be 128, 64, or 32");

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. group_size-1

    // Map group index to position in the tensor:
    // each head has groups_per_head full groups, then a gap of tail elements.
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    // Load from global memory
    x[t] = src[base + t];
    __syncthreads();

    // InnerQ forward: apply scale_inv BEFORE signs+WHT (for Q pre-rotation)
    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    // Apply first sign array
    if (group_size == 128) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else if (group_size == 64) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    } else {
        // group_size == 32: TQ weight signs (same for forward and inverse)
        x[t] *= TQ_WEIGHT_SIGNS[t];
    }
    __syncthreads();

    // WHT butterfly — log2(group_size) stages.
    // In stage h, threads where (t % (2h)) < h read x[t] and x[t+h],
    // then write x[t] = a+b and x[t+h] = a-b.  Each active thread
    // owns a disjoint pair, so no intra-stage conflicts exist.
#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    if (group_size >= 64) { WHT_STAGE(32) }
    if (group_size == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    // Normalize and apply second sign array, write to output
    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f :
                               (group_size == 64)  ? 0.125f :
                                                     0.17677669529663688f; // 1/sqrt(32)
    float result;
    if (group_size == 128) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else if (group_size == 64) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    } else {
        // group_size == 32: normalize only (signs already applied before butterfly)
        result = x[t] * inv_sqrt;
    }

    // InnerQ inverse: apply scale_inv AFTER WHT+signs (for V un-rotation)
    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];  // InnerQ scale_inv (may be NULL)

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction, dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    cudaStream_t stream = ctx.stream();

    // Process full groups
    // Prefer half4 vectorized kernel for group_size 128 and 64 (32/16 threads vs 128/64).
    // Falls back to scalar kernel for group_size=32.
    if (n_groups > 0) {
        dim3 blocks(n_groups);
        if (group_size == 128) {
            dim3 threads_h4(32);  // half4: 32 threads x 4 elements = 128
            if (direction == 0) {
                k_turbo_wht_half4<0, 128><<<blocks, threads_h4, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_half4<1, 128><<<blocks, threads_h4, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else if (group_size == 64) {
            dim3 threads_h4(16);  // half4: 16 threads x 4 elements = 64
            if (direction == 0) {
                k_turbo_wht_half4<0, 64><<<blocks, threads_h4, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_half4<1, 64><<<blocks, threads_h4, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else {
            dim3 threads(32);
            if (direction == 0) {
                k_turbo_wht_f32<0, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        }
    }

    // Pass through tail elements unchanged (no rotation)
    // Not needed for 64-aligned dims but kept for completeness
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * group_size, tail_size);
    }
}
