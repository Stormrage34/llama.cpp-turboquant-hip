/*
 * Fused mul_mat_vec for TQ4_1S / TQ3_1S weight types.
 *
 * V12: Single-phase fused kernel with shmem activation sharing.
 * All warps cooperatively rotate activation into shared memory,
 * then each warp processes one row reading from shmem (broadcast).
 *
 * Eliminates:
 *   - Global memory scratch buffer (no CUDA graph incompatibility)
 *   - Separate pre-rotation kernel launch
 *   - 2x activation bandwidth (was: write global + read global per row)
 *
 * V12 avoids the NR0 regression that killed V3/V6/V11 — the single
 * __syncthreads is OUTSIDE the dot product loop (between rotation and
 * mmvq phases), not inside it.
 *
 * Falls back to V8 two-phase if shmem exceeds 48 KB (ncols > 12288).
 *
 * Based on signalnine's V8 two-phase kernel (commit b107175).
 * Optimization by TheTom.
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"

#define MMVQ_TQ_NWARPS 8

// ============================================================================
// V8 two-phase kernels (fallback for very large ncols that exceed shmem)
// ============================================================================

static __global__ void tq_prerotate_activation_v8(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = val;
}

static __global__ void mul_mat_vec_tq4_1s_v8(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ4_1S + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;

        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

static __global__ void mul_mat_vec_tq3_1s_v8(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ3_0 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);

        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// V12: Single-phase fused kernel — rotate in shmem, no global scratch
//
// All 8 warps cooperatively WHT-rotate activation into shared memory.
// Then each warp processes one row doing centroid×scale dot product
// reading activation from shmem (broadcast reads from L1).
//
// The key insight: the single __syncthreads is between the two phases
// (rotation vs dot product), NOT inside the inner dot product loop.
// This is why V3/V11 regressed (sync per block) but V12 should not.
// ============================================================================

static __global__ void mul_mat_vec_tq4_1s_v12(
        const void  * __restrict__ vx,
        const float * __restrict__ vy,   // UNROTATED activation (raw src1)
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    extern __shared__ float s_act[];  // ncols_x floats

    const int lane    = threadIdx.x;  // 0-31
    const int warp_id = threadIdx.y;  // 0 to MMVQ_TQ_NWARPS-1
    const int blocks_per_row = ncols_x / QK_TQ4_1S;

    // Phase 1: ALL warps cooperatively pre-rotate activation into shmem.
    // Each warp handles a strided subset of 32-element blocks.
    // 8 warps × 32 threads = 256 threads rotating in parallel.
    for (int ib = warp_id; ib < blocks_per_row; ib += MMVQ_TQ_NWARPS) {
        float val = vy[ib * 32 + lane];
        val *= TQ_WEIGHT_SIGNS[lane];

        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffff, val, h);
            val = (lane & h) ? (o - val) : (val + o);
        }
        val *= 0.17677669529663688f;  // 1/sqrt(32)
        s_act[ib * 32 + lane] = val;
    }
    __syncthreads();  // ONE sync — between rotation and dot product, NOT in inner loop

    // Phase 2: Each warp processes one row using shmem activation (broadcast reads).
    const int row = blockIdx.x * MMVQ_TQ_NWARPS + warp_id;
    if (row >= nrows_x) return;

    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = s_act[ib * 32 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static __global__ void mul_mat_vec_tq3_1s_v12(
        const void  * __restrict__ vx,
        const float * __restrict__ vy,   // UNROTATED activation (raw src1)
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    extern __shared__ float s_act[];

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int blocks_per_row = ncols_x / QK_TQ3_0;

    // Phase 1: cooperative rotation into shmem
    for (int ib = warp_id; ib < blocks_per_row; ib += MMVQ_TQ_NWARPS) {
        float val = vy[ib * 32 + lane];
        val *= TQ_WEIGHT_SIGNS[lane];

        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffff, val, h);
            val = (lane & h) ? (o - val) : (val + o);
        }
        val *= 0.17677669529663688f;
        s_act[ib * 32 + lane] = val;
    }
    __syncthreads();

    // Phase 2: mmvq from shmem
    const int row = blockIdx.x * MMVQ_TQ_NWARPS + warp_id;
    if (row >= nrows_x) return;

    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = s_act[ib * 32 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// Dispatch — V12 shmem when it fits, V8 two-phase fallback
// ============================================================================

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0,
                               const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    const size_t shmem_needed = (size_t)ncols_x * sizeof(float);

    // V12: single kernel, activation in shmem (fits for all models up to ncols=12288)
    // V8 fallback: two-phase with global scratch (for hypothetical future huge models)
    if (shmem_needed <= 48 * 1024) {
        const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
        const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);

        if (src0->type == GGML_TYPE_TQ4_1S) {
            mul_mat_vec_tq4_1s_v12<<<grid, block, shmem_needed, stream>>>(src0_d, src1_d, dst_d, ncols_x, nrows_x);
        } else {
            mul_mat_vec_tq3_1s_v12<<<grid, block, shmem_needed, stream>>>(src0_d, src1_d, dst_d, ncols_x, nrows_x);
        }
    } else {
        // V8 fallback: two-phase with global scratch buffer
        static float * d_act_buf = nullptr;
        static size_t  d_act_buf_size = 0;

        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(stream, &capture_status);

        if (capture_status != cudaStreamCaptureStatusNone) {
            GGML_ASSERT(d_act_buf != nullptr && d_act_buf_size >= shmem_needed &&
                         "TQ scratch buffer not pre-allocated before graph capture");
        } else {
            if (shmem_needed > d_act_buf_size) {
                if (d_act_buf) cudaFree(d_act_buf);
                cudaMalloc(&d_act_buf, shmem_needed);
                d_act_buf_size = shmem_needed;
            }
        }

        {
            const int n_blocks = ncols_x / 32;
            const dim3 rot_block(32, 4);
            const dim3 rot_grid((n_blocks + 3) / 4);
            tq_prerotate_activation_v8<<<rot_grid, rot_block, 0, stream>>>(src1_d, d_act_buf, ncols_x);
        }

        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);

            if (src0->type == GGML_TYPE_TQ4_1S) {
                mul_mat_vec_tq4_1s_v8<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            } else {
                mul_mat_vec_tq3_1s_v8<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            }
        }
    }
}

// ============================================================================
// Load-time conversion: TQ4_1S → q8_0
//
// Fused kernel: dequant TQ4_1S (centroid lookup + inverse WHT) → quantize q8_0.
// One warp (32 threads) per block of 32 elements.
// Used at model load to convert TQ4_1S weights to q8_0 in VRAM for dp4a decode.
// ============================================================================

static __global__ void k_convert_tq4_1s_to_q8_0(
        const block_tq4_1s * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;

    const int lane = threadIdx.x;
    const block_tq4_1s * blk = &src[block_idx];

    // Step 1: Dequant — centroid lookup × half-block scale
    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    // Step 2: Inverse WHT via warp shuffle (same as dequant path)
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    val *= TQ_WEIGHT_SIGNS[lane];

    // Step 3: Quantize to q8_0 — find block amax, compute scale, round
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    // Step 4: Write q8_0 block
    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].d = __float2half(d);
    }
}

void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_TQ4_1S == 0);
    const int n_blocks = n_elements / QK_TQ4_1S;

    const int wpb = 4;  // warps per CUDA block
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);

    k_convert_tq4_1s_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_tq4_1s *)src_tq4,
        (block_q8_0 *)dst_q8,
        n_blocks);
}
