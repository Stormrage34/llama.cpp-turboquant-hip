#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"
#include "set-rows-planar-iso.cuh"
#include "rotorquant.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    GGML_ASSERT(s1 % sizeof(block_type) == 0);
    GGML_ASSERT(s2 % sizeof(block_type) == 0);
    GGML_ASSERT(s3 % sizeof(block_type) == 0);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_PLANAR3_0) {
        ggml_cuda_init_planar_iso_constants();
        set_rows_cuda_quant<idx_t, block_planar3_0, QK_PLANAR3, quantize_f32_planar3_block_norot>(
            src0_d, src1_d, (block_planar3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_ISO3_0) {
        ggml_cuda_init_planar_iso_constants();
        set_rows_cuda_quant<idx_t, block_iso3_0, QK_ISO3, quantize_f32_iso3_block_norot>(
            src0_d, src1_d, (block_iso3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_RQ_MSE) {
        set_rows_cuda_quant<idx_t, block_rq_mse_2, QK_RQ, quantize_f32_rq_mse_2_block>(
            src0_d, src1_d, (block_rq_mse_2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_RQ_PROD) {
        set_rows_cuda_quant<idx_t, block_rq_prod, QK_RQ, quantize_f32_rq_prod_block>(
            src0_d, src1_d, (block_rq_prod*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


// ─── TurboQuant set_rows kernels ──────────────────────────────────────────────
//
// Each thread processes one full group (group_size elements). The flow mirrors
// the CPU path in ggml-turbo-quant.c:
//   1. L2 norm over the group
//   2. Normalize by 1/norm
//   3. Forward WHT rotation: signs1 -> FWHT -> signs2  (applied ONCE per group)
//   4. Quantize each QK-sized sub-block from rotated values
//   5. Reconstruction-corrected norm: grp_norm / sqrt(sum(centroid^2))
//
// The dequantize path (fattn-vec.cuh) reads centroids directly without inverse
// rotation, matching the upstream design where both Q and K live in the rotated
// domain for the dot product.

// ─── Turbo3 set_rows kernel ──────────────────────────────────────────────────
// group_size=128, qk=32 (4 sub-blocks per group). Thread processes one full
// 128-element group: normalize -> WHT rotate -> quantize each sub-block.

static __global__ void k_set_rows_turbo3(const float * __restrict__ src0,
                                          const int * __restrict__ src1,
                                          block_turbo3_0 * __restrict__ dst,
                                         const int64_t ne_total,
                                         const int64_t ne10, const int64_t ne11,
                                         const int64_t ne12, const int64_t ne13,
                                         const int64_t s01, const int64_t s02,
                                         const int64_t s03,
                                         const int64_t s10, const int64_t s11,
                                         const int64_t s12,
                                         const int64_t s1, const int64_t s2,
                                         const int64_t s3,
                                         const uint3 ne00, const uint3 ne01,
                                         const uint3 ne02,
                                         const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod            = fast_div_modulo(tmp, ne00);
    const int64_t i00  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne01);
    const int64_t i01  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne02);
    const int64_t i02  = div_mod.y;
    const int64_t i03  = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo3_0);

    // Load full 128-element group into registers
    float buf[128];
    const float * grp_src = src0_row + i00 * QK_TURBO3 * (128 / QK_TURBO3); // i00 * qk * blocks_per_group = i00 * 128
    for (int j = 0; j < 128; j++) {
        buf[j] = grp_src[j];
    }

    // Step 1-2: L2 norm + normalize
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += buf[j] * buf[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) buf[j] *= inv_norm;

    // Step 3: Quantize sub-blocks + reconstruction-corrected norm
    const int n_blocks = 128 / QK_TURBO3; // = 4
    float recon_sq = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const int off = b * QK_TURBO3;
        block_turbo3_0 blk;
        quantize_f32_turbo3_0_block(buf + off, &blk);

        // Reconstruction error: extract raw centroid values (without norm)
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (blk.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1  = (blk.signs[j / 8] >> (j % 8)) & 0x1;
            uint8_t idx  = low2 | (hi1 << 2);
            recon_sq += TURBO_CENTROIDS_3BIT[idx] * TURBO_CENTROIDS_3BIT[idx];
        }

        dst_row_ptr[i00 * n_blocks + b] = blk;
    }

    // Step 5: Corrected norm
    float recon_norm = sqrtf(recon_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < n_blocks; b++) {
        dst_row_ptr[i00 * n_blocks + b].norm = __float2half(corrected_norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// ─── Turbo2 set_rows kernel ──────────────────────────────────────────────────
// group_size=128, qk=32 (4 sub-blocks per group). Thread processes one full
// 128-element group: normalize -> WHT rotate -> quantize each sub-block.

static __global__ void k_set_rows_turbo2(const float * __restrict__ src0,
                                          const int * __restrict__ src1,
                                          block_turbo2_0 * __restrict__ dst,
                                         const int64_t ne_total,
                                         const int64_t ne10, const int64_t ne11,
                                         const int64_t ne12, const int64_t ne13,
                                         const int64_t s01, const int64_t s02,
                                         const int64_t s03,
                                         const int64_t s10, const int64_t s11,
                                         const int64_t s12,
                                         const int64_t s1, const int64_t s2,
                                         const int64_t s3,
                                         const uint3 ne00, const uint3 ne01,
                                         const uint3 ne02,
                                         const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod            = fast_div_modulo(tmp, ne00);
    const int64_t i00  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne01);
    const int64_t i01  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne02);
    const int64_t i02  = div_mod.y;
    const int64_t i03  = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo2_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo2_0);

    // Load full 128-element group into registers
    float buf[128];
    const float * grp_src = src0_row + i00 * QK_TURBO2 * (128 / QK_TURBO2); // i00 * qk * blocks_per_group = i00 * 128
    for (int j = 0; j < 128; j++) {
        buf[j] = grp_src[j];
    }

    // Step 1-2: L2 norm + normalize
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += buf[j] * buf[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) buf[j] *= inv_norm;

    // Step 3: Quantize sub-blocks + reconstruction-corrected norm
    const int n_blocks = 128 / QK_TURBO2; // = 4
    float recon_sq = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const int off = b * QK_TURBO2;
        block_turbo2_0 blk;
        quantize_f32_turbo2_0_block(buf + off, &blk);

        // Reconstruction error: extract 2-bit centroid values
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (blk.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            recon_sq += TURBO_CENTROIDS_2BIT[idx] * TURBO_CENTROIDS_2BIT[idx];
        }

        dst_row_ptr[i00 * n_blocks + b] = blk;
    }

    // Step 5: Corrected norm
    float recon_norm = sqrtf(recon_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < n_blocks; b++) {
        dst_row_ptr[i00 * n_blocks + b].norm = __float2half(corrected_norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// ─── Turbo4 set_rows kernel ──────────────────────────────────────────────────
// group_size=128, qk=128 (1 block per group). Simpler: each thread = one group.

static __global__ void k_set_rows_turbo4(const float * __restrict__ src0,
                                          const int * __restrict__ src1,
                                          block_turbo4_0 * __restrict__ dst,
                                         const int64_t ne_total,
                                         const int64_t ne10, const int64_t ne11,
                                         const int64_t ne12, const int64_t ne13,
                                         const int64_t s01, const int64_t s02,
                                         const int64_t s03,
                                         const int64_t s10, const int64_t s11,
                                         const int64_t s12,
                                         const int64_t s1, const int64_t s2,
                                         const int64_t s3,
                                         const uint3 ne00, const uint3 ne01,
                                         const uint3 ne02,
                                         const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod            = fast_div_modulo(tmp, ne00);
    const int64_t i00  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne01);
    const int64_t i01  = div_mod.y;
    tmp                = div_mod.x;
    div_mod            = fast_div_modulo(tmp, ne02);
    const int64_t i02  = div_mod.y;
    const int64_t i03  = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo4_0);

    // Load full 128-element group into registers
    float buf[128];
    const float * grp_src = src0_row + i00 * QK_TURBO4;
    for (int j = 0; j < 128; j++) {
        buf[j] = grp_src[j];
    }

    // Step 1-2: L2 norm + normalize
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += buf[j] * buf[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) buf[j] *= inv_norm;

    // Step 3: Quantize
    block_turbo4_0 blk;
    quantize_f32_turbo4_0_block(buf, &blk);

    // Step 5: Reconstruction-corrected norm
    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = (blk.qs[j / 2] >> ((j % 2) * 4)) & 0xF;
        recon_sq += TURBO_CENTROIDS_4BIT[idx] * TURBO_CENTROIDS_4BIT[idx];
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    blk.norm   = __float2half(corrected_norm);
    blk.rnorm  = __float2half(recon_norm);

    dst_row_ptr[i00] = blk;

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // TurboQuant types: value-preserved quantization with reconstruction-corrected norm
    // (no WHT rotation; decomposed for VEC dequantize compatibility)
    const char * type_name = ggml_type_name(dst->type);
    bool is_turbo2 = (strstr(type_name, "turbo2") != NULL);
    bool is_turbo3 = (strstr(type_name, "turbo3") != NULL);
    bool is_turbo4 = (strstr(type_name, "turbo4") != NULL);

    if (is_turbo2 || is_turbo3 || is_turbo4) {
        const int64_t ne00 = src0->ne[0];
        const int64_t ne01 = src0->ne[1];
        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];
        const int64_t ne10 = dst->ne[0];
        const int64_t ne11 = dst->ne[1];
        const int64_t ne12 = dst->ne[2];
        const int64_t ne13 = dst->ne[3];

        const int qk = is_turbo4 ? QK_TURBO4 : QK_TURBO3; // turbo2 and turbo3 both use 32
        const int blocks_per_group = 128 / qk;
        const int64_t ne00_groups = ne00 / (qk * blocks_per_group); // groups per row (1 for D=128)
        const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / (qk * blocks_per_group);
        const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
        const dim3 grid_size(num_blocks);

        const int64_t s01 = src0->nb[1]/sizeof(float);
        const int64_t s02 = src0->nb[2]/sizeof(float);
        const int64_t s03 = src0->nb[3]/sizeof(float);
        // nb[0] = element size (4 for I32, 8 for I64).  Kernel reads const int* (4B).
        // s10=1 for I32 → sequential int access. s10=2 for I64 → skip high32 of each int64_t.
        const int64_t s10 = src1->nb[0] / sizeof(int);

        // I64 row indices must be verified to fit in int32_t before the cast to const int*
        if (src1->type == GGML_TYPE_I64) {
            const int64_t * row_data_i64 = (const int64_t *) src1->data;
            const int64_t nrows_i64 = ne12 * ne13;
            for (int64_t i = 0; i < nrows_i64; i++) {
                GGML_ASSERT(row_data_i64[i] <= (int64_t) INT32_MAX
                            && "turbo set-rows: I64 row index exceeds 32-bit range");
            }
        }

        const int64_t s11 = src1->nb[1] / sizeof(int);
        const int64_t s12 = src1->nb[2] / sizeof(int);
        const int64_t s1  = dst->nb[1];
        const int64_t s2  = dst->nb[2];
        const int64_t s3  = dst->nb[3];

        const size_t block_sz_turbo = is_turbo4 ? sizeof(block_turbo4_0) :
                                       is_turbo2 ? sizeof(block_turbo2_0) :
                                                   sizeof(block_turbo3_0);
        GGML_ASSERT(s1 % block_sz_turbo == 0);
        GGML_ASSERT(s2 % block_sz_turbo == 0);
        GGML_ASSERT(s3 % block_sz_turbo == 0);

        if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00_groups);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

            if (is_turbo4) {
                k_set_rows_turbo4<<<grid_size, block_size, 0, ctx.stream()>>>(
                    (const float *)src0->data, (const int *)src1->data,
                    (block_turbo4_0*)dst->data,
                    ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
                    ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                CUDA_CHECK(cudaGetLastError());
            } else if (is_turbo2) {
                k_set_rows_turbo2<<<grid_size, block_size, 0, ctx.stream()>>>(
                    (const float *)src0->data, (const int *)src1->data,
                    (block_turbo2_0*)dst->data,
                    ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
                    ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                CUDA_CHECK(cudaGetLastError());
            } else {
                k_set_rows_turbo3<<<grid_size, block_size, 0, ctx.stream()>>>(
                    (const float *)src0->data, (const int *)src1->data,
                    (block_turbo3_0*)dst->data,
                    ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
                    ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    } else if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
