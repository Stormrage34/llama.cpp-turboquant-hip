// GQA-shared decode flash attention for gfx1030 (RDNA2, wave32).
//
// One workgroup per (query-row tile, KV head, KV tile), with one warp per Q head of the
// group. It computes all GROUP query heads that share one KV head from a single K/V
// read. This is the llama.cpp/HIP port of hipfire's
// attention_flash_q8_0_tile_gqa.gfx1100.hip design, with head_dim 256.
//
// GROUP is the GQA ratio (Q heads per KV head) and therefore also the workgroup shape:
// GROUP warps, GROUP*32 threads. Exactly two ratios are compiled, and the host-side
// predicate in fattn.cu admits nothing else:
//   GROUP = 6: 24 Q heads / 4 KV heads (dense Qwen3.8-27B)
//   GROUP = 8: 16 Q heads / 2 KV heads (MoE Qwen3.8-35B-A3B)
//
// ROWS is the number of query rows one workgroup handles. blockIdx.x selects the row
// tile (launch_fattn is called with ncols1 == ROWS), blockIdx.y splits the KV range and
// blockIdx.z still enumerates (sequence, KV-head group). Rows of one tile share the
// staged K/V chunk: a speculative-decoding verify pass submits 2-4 rows for one
// sequence, and reading K/V once per tile rather than once per row (or once per row and
// per query head, as the VEC kernel does) is what keeps that pass at decode bandwidth.
// ROWS == 1 is the plain decode shape and keeps the original arithmetic order. ROWS ==
// GQA_GFX1030_ROWS_MAX covers 2, 3 and 4 rows with one instance, because the number of
// live rows is a runtime value inside the kernel: a 3-row verify pass uses the same
// compiled kernel as a 4-row one and simply leaves the fourth accumulator idle.
//
// Supported:
//   - Q f32, K Q8_0, V Q8_0 or Q5_0
//   - 1 to GQA_GFX1030_ROWS_MAX (4) query rows (Q->ne[1]); 1 = decode, 2-4 = verify
//   - GQA ratio exactly 6 or 8
//   - standard additive f16 mask (or no mask), no ALiBi/sinks/logit softcap
//
// The kernel reuses the existing fattn-vec partials/combine contract:
//   dst[(((seq*ne01 + col)*ne02 + head)*gridDim.y + blockIdx.y)*D + d]
//   dst_meta[((seq*ne01 + col)*ne02 + head)*gridDim.y + blockIdx.y]
// so it can be launched through launch_fattn<256, ROWS, GROUP>().

#ifdef GGML_USE_HIP
#include "fattn-common.cuh"

// Geometry shared by every instantiation. GROUP is not here: it is the template
// parameter below, because it sets the workgroup shape.
constexpr int GQA_GFX1030_D     = 256; // head dim
constexpr int GQA_GFX1030_TILE  = 256; // KV keys per (block, outer iteration)
constexpr int GQA_GFX1030_CHUNK = 24;  // KV keys staged in LDS at a time
// Widest query-row tile: 1 (decode) plus the 2-4 rows an MTP verify pass submits.
// Qwen3.8's MTP head drafts up to 4 tokens, so 4 covers the whole speculative batch.
constexpr int GQA_GFX1030_ROWS_MAX = 4;

template <ggml_type type_V>
__device__ __forceinline__ void gqa_gfx1030_load_v(
        const char * __restrict__ vp, half * __restrict__ dst) {
    half d;
    memcpy(&d, vp, sizeof(half));
    const float df = __half2float(d);

    if constexpr (type_V == GGML_TYPE_Q8_0) {
        const int8_t * qs = (const int8_t *) (vp + 2);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            dst[i] = __float2half_rn(df * float(qs[i]));
        }
    } else {
        static_assert(type_V == GGML_TYPE_Q5_0, "unsupported V type");
        uint32_t qh;
        memcpy(&qh, vp + 2, sizeof(qh));
        const uint8_t * qs = (const uint8_t *) (vp + 6);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            // q5_0 stores values 0..15 in the low nibbles of qs[0..15] and
            // values 16..31 in the high nibbles of the same bytes.
            const uint32_t nib = i < 16
                ? (uint32_t(qs[i]) & 0x0Fu)
                : (uint32_t(qs[i - 16]) >> 4);
            const uint32_t hi  = (qh >> i) & 1u;
            const float v = float(nib | (hi << 4)) - 16.0f;
            dst[i] = __float2half_rn(df * v);
        }
    }
}

// GROUP warps, one per Q head. `__launch_bounds__` states the workgroup size the block
// actually runs at; the min-blocks argument is 1 because the register budget allows
// exactly one block per CU: gfx1030 holds 8 wave32 slots per CU at the 256-VGPR ceiling
// this kernel already uses, so a 256-thread block fills the CU. Asking for 2+ blocks
// would cap VGPRs at <=128/thread and spill the accumulators.
//
// Adding rows is paid for in registers, not in LDS: the accumulators become
// acc[ROWS][ACC] (ROWS*8 floats/lane, the dominant term), the softmax state becomes
// kq_max[ROWS]/kq_sum[ROWS], and Q staging grows from GROUP*D to ROWS*GROUP*D bytes.
// K/V staging is unchanged, because the whole point of the row tile is that rows share
// one staged chunk. That is why there is one wide instance (ROWS = 4) instead of a
// separate kernel per row count.
template <int GROUP, int ROWS, ggml_type type_V>
__launch_bounds__(GROUP*32, 2)
static __global__ void flash_attn_ext_gqa_gfx1030(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    GGML_UNUSED_VARS(ne00, ne10, ne13, ne31, ne32, nb32, max_bias, m0, m1, n_head_log2,
                     logit_softcap, sinks_ptr);

    constexpr int D      = GQA_GFX1030_D;
    constexpr int TILE   = GQA_GFX1030_TILE;
    constexpr int CHUNK  = GQA_GFX1030_CHUNK;
    constexpr int TIDS   = GROUP*32;
    constexpr int ACC    = D / 32; // output elements per lane

    // A ratio with no validated instantiation cannot reach this kernel: the host predicate
    // and the dispatch switch below both enumerate {6, 8}. This assert is the third lock.
    static_assert(GROUP == 6 || GROUP == 8, "gfx1030 GQA: only the validated ratios 6 and 8 are instantiated");
    static_assert(GROUP <= 32, "one warp per Q head, so GROUP cannot exceed a warp count");
    // ROWS is either the decode shape or the widest verify tile; 2 and 3 rows run on the
    // latter with a runtime row count, so a 3-row instance would only add code.
    static_assert(ROWS == 1 || ROWS == GQA_GFX1030_ROWS_MAX,
                  "gfx1030 GQA: rows are compiled as 1 (decode) or GQA_GFX1030_ROWS_MAX (verify)");
    // Staging assigns one thread per (key, 32-value block): CHUNK*8 threads, and it must fit
    // the workgroup. 24*8 = 192 <= 6*32 = 192 and <= 8*32 = 256, so both ratios fit; for
    // GROUP = 8 the staging loop simply leaves 64 of 256 threads idle.
    static_assert(CHUNK*8 <= TIDS, "K/V staging needs more threads than the workgroup has");

    // K is staged as raw Q8_0 bytes for sdot4. sKq[b][i4][key][0..3] holds
    // one int8x4 group; for a fixed b/i4 the lanes read consecutive keys.
    __shared__ int8_t sKq[D/32][8][CHUNK][4];
    __shared__ half   sKd[D/32][CHUNK];
    // V is staged by key: sV[key][d]. Lanes read sV[k][lane*ACC + j].
    __shared__ half   sV[CHUNK][D];
    // Q is quantized to q8_0 once per (row, head), with one f32 scale per 32 values.
    // This is the only buffer that grows with ROWS.
    __shared__ int8_t sQq[ROWS][GROUP][D];
    __shared__ float  sQd[ROWS][GROUP][D/32];

    const int tid  = threadIdx.y*32 + threadIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    const int gqa_ratio = ne02 / ne12;
    // blockIdx.z enumerates (sequence, KV head). ne02/GROUP == ne12 because GROUP is the
    // admitted ratio and the host asserts Q->ne[2] == GROUP*K->ne[2], so this is an exact
    // division: 24/6 = 4 for the dense model, 16/8 = 2 for the MoE.
    const int sequence  = blockIdx.z / (ne02 / GROUP);
    const int head0     = blockIdx.z * GROUP - sequence * ne02;
    const int kv_head   = head0 / gqa_ratio;

    // Query rows of this tile: blockIdx.x is the row tile and gridDim.x is
    // ceil(Q->ne[1]/ROWS), because the launch passes ncols1 == ROWS. ne01.z is Q->ne[1]
    // (init_fastdiv_values packs the real extent there). col0 is a compile-time 0 for
    // ROWS == 1, so that instance keeps reading row 0 exactly as before.
    const int col0 = ROWS > 1 ? int(blockIdx.x)*ROWS : 0;
    const int nrow = ROWS < int(ne01.z) - col0 ? ROWS : int(ne01.z) - col0;

    ggml_cuda_pdl_lc();

    const char * Qb = Q_ptr + nb03*sequence + nb02*head0;
    const char * Kb = K_ptr + nb13*sequence + nb12*kv_head;
    const char * Vb = V_ptr + nb23*sequence + nb22*kv_head;
    const half * maskh = mask_ptr
        ? (const half *) (mask_ptr + nb33*(sequence % ne33))
        : nullptr;
    // Additive f16 mask row for each query row of the tile, hoisted out of the KV loop.
    // Mask layout is [n_kv, n_tokens, n_head_group, n_seqs]: nb33 selects the sequence,
    // nb31 the *token* row, and dim 2 groups heads, which this kernel leaves at row 0
    // (like the VEC/TILE kernels, which also read the mask as head-agnostic). The token
    // row is the query row, so each row of the tile reads its own mask row -- reusing row
    // 0 for the whole tile is exactly the bug the multi-row cases exist to catch.
    const int mask_s31 = nb31 / int(sizeof(half));
    const half * maskrow[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        maskrow[r] = maskh ? maskh + (col0 + r)*mask_s31 : nullptr;
    }

    // Quantize every Q row of the tile to q8_0 once: one warp per head, one f32
    // scale per 32 values, matching the VEC kernel's q8_1 Q path. Rows are looped rather
    // than given warps because the workgroup has exactly GROUP warps, and because this
    // runs once per block while the KV loop below runs for every key.
    if (warp < GROUP) {
        for (int r = 0; r < nrow; ++r) {
            const float * Qh = (const float *) (Qb + (col0 + r)*nb01 + warp*nb02);
#pragma unroll
            for (int b = 0; b < D/32; ++b) {
                const float q = Qh[b*32 + lane];
                const float amax = warp_reduce_max(fabsf(q));
                const float d = amax > 0.0f ? amax/127.0f : 1.0f;
                int qi = __float2int_rn(q/d);
                qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
                sQq[r][warp][b*32 + lane] = int8_t(qi);
                if (lane == 0) {
                    sQd[r][warp][b] = d;
                }
            }
        }
    }

    const int k_max = KV_max_ptr
        ? KV_max_ptr[sequence*gridDim.x + blockIdx.x]
        : ne11;

    float acc[ROWS][ACC];
    float kq_max[ROWS];
    float kq_sum[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        kq_max[r] = -INFINITY;
        kq_sum[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < ACC; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    ggml_cuda_pdl_sync();
    __syncthreads();

    for (int k0 = blockIdx.y*TILE; k0 < k_max; k0 += gridDim.y*TILE) {
        const int tile_end = k0 + TILE < k_max ? k0 + TILE : k_max;
        for (int c0 = 0; c0 < TILE; c0 += CHUNK) {
            const int base = k0 + c0;

            // One thread per (key, 32-value block). CHUNK*8 <= TIDS.
            if (tid < CHUNK*8) {
                const int key     = tid >> 3;
                const int blk     = tid & 7;
                const int abs_key = base + key;

                if (abs_key < tile_end) {
                    const char * kp = Kb + int64_t(abs_key)*nb11 + blk*34;
                    half d;
                    memcpy(&d, kp, sizeof(half));
                    const int8_t * qs = (const int8_t *) (kp + 2);
#pragma unroll
                    for (int i4 = 0; i4 < 8; ++i4) {
                        int v;
                        memcpy(&v, qs + 4*i4, sizeof(v));
                        memcpy(&sKq[blk][i4][key][0], &v, sizeof(v));
                    }
                    sKd[blk][key] = d;

                    const char * vp = Vb + int64_t(abs_key)*nb21 + blk*(type_V == GGML_TYPE_Q5_0 ? 22 : 34);
                    gqa_gfx1030_load_v<type_V>(vp, &sV[key][blk*32]);
                } else {
#pragma unroll
                    for (int i4 = 0; i4 < 8; ++i4) {
                        const int v = 0;
                        memcpy(&sKq[blk][i4][key][0], &v, sizeof(v));
                    }
                    sKd[blk][key] = __float2half_rn(0.0f);
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        sV[key][blk*32 + i] = __float2half_rn(0.0f);
                    }
                }
            }

            __syncthreads();

            if (warp < GROUP) {
                // Unroll depth of the two inner loops. The wide instance carries four
                // accumulator vectors plus four softmax pairs, which puts it past the 256-VGPR
                // ceiling at full unroll and spills about 200 B per thread into the scratch
                // buffer; halving the unroll trades instruction-level parallelism for registers
                // without reordering any arithmetic. The 1-row instance keeps the original
                // schedule, which fits.
                constexpr int UNROLL_B = ROWS == 1 ? 8 : 4;
                constexpr int UNROLL_K = ROWS == 1 ? CHUNK : 2;
                for (int r = 0; r < nrow; ++r) {
                    // One lane per key in this chunk. Eight sdot4 groups per
                    // 32-value block, with the Q8_0 K and Q scales applied once.
                    float sum = 0.0f;
#pragma unroll UNROLL_B
                    for (int b = 0; b < D/32; ++b) {
                        int dot = 0;
#pragma unroll
                        for (int i4 = 0; i4 < 8; ++i4) {
                            int qv;
                            int kv;
                            memcpy(&qv, &sQq[r][warp][b*32 + 4*i4], sizeof(qv));
                            const int lk = lane < CHUNK ? lane : (CHUNK - 1);
                            memcpy(&kv, &sKq[b][i4][lk][0], sizeof(kv));
                            // ggml_cuda_dp4a selects sdot4 on gfx10+ (RDNA2/CDNA) and gfx906, sudot4 on
                            // RDNA3/4, and a v_mul_i32_i24 fallback on RDNA1/gfx900. Calling the builtin
                            // directly made this TU fail to compile for gfx1010/1011/gfx1100/gfx1101,
                            // which lack dot1-insts.
                            dot = ggml_cuda_dp4a(qv, kv, dot);
                        }
                        const int lk2 = lane < CHUNK ? lane : (CHUNK - 1);
                        sum += sQd[r][warp][b] * __half2float(sKd[b][lk2]) * float(dot);
                    }
                    const int kk = base + lane;
                    const bool valid = lane < CHUNK && kk < tile_end;
                    const half * mrow = maskrow[r];
                    float score = valid ? sum * scale + (mrow ? __half2float(mrow[kk]) : 0.0f) : -INFINITY;

                    const float chunk_max = warp_reduce_max(score) + FATTN_KQ_MAX_OFFSET;
                    // A fully out-of-range chunk (the tail of a partial tile when TILE % CHUNK != 0)
                    // makes every score -INFINITY, so chunk_max is -INFINITY too and
                    // expf(kq_max - new_max) would be expf(-INF - -INF) = NaN, which then poisons
                    // kq_sum and the split metadata written below. Skip such a chunk: it holds no
                    // keys, so it contributes nothing to the softmax. The first chunk of a tile
                    // always has a valid key, so kq_max still becomes finite.
                    if (chunk_max == -INFINITY) {
                        continue;
                    }
                    const float new_max   = fmaxf(kq_max[r], chunk_max);
                    const float corr      = expf(kq_max[r] - new_max);
                    const float w         = expf(score - new_max);

                    kq_sum[r] = kq_sum[r]*corr + warp_reduce_sum(w);
                    kq_max[r] = new_max;

#pragma unroll
                    for (int j = 0; j < ACC; ++j) {
                        acc[r][j] *= corr;
                    }

#pragma unroll UNROLL_K
                    for (int k = 0; k < CHUNK; ++k) {
                        const float wk = __shfl_sync(0xFFFFFFFFu, w, k, 32);
                        if (wk != 0.0f) {
#pragma unroll
                            for (int j = 0; j < ACC; j += 2) {
                                const half2 v2 = *((const half2 *) &sV[k][lane*ACC + j]);
                                const float2 vf = __half22float2(v2);
                                acc[r][j + 0] = fmaf(wk, vf.x, acc[r][j + 0]);
                                acc[r][j + 1] = fmaf(wk, vf.y, acc[r][j + 1]);
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }

    if (warp < GROUP) {
        const int head = head0 + warp;
        for (int r = 0; r < nrow; ++r) {
            // Global query row: rows of the tile are consecutive, so the partial-result
            // index is (sequence*ne01 + col0 + r)*ne02 + head, with the KV split innermost.
            const int row  = sequence*int(ne01.z) + col0 + r;
            const int bidx = (row*ne02 + head)*gridDim.y + blockIdx.y;

            if (gridDim.y == 1) {
                const float inv = kq_sum[r] > 0.0f ? 1.0f/kq_sum[r] : 0.0f;
#pragma unroll
                for (int j = 0; j < ACC; ++j) {
                    dst_ptr[int64_t(bidx)*D + lane*ACC + j] = acc[r][j]*inv;
                }
            } else {
#pragma unroll
                for (int j = 0; j < ACC; ++j) {
                    dst_ptr[int64_t(bidx)*D + lane*ACC + j] = acc[r][j];
                }
                if (lane == 0) {
                    dst_meta_ptr[bidx] = make_float2(kq_max[r], kq_sum[r]);
                }
            }
        }
    }
}

// The only GQA ratios with a compiled kernel instance. Lives here, next to the
// static_assert and the dispatch switch that instantiate them, so the admission test in
// fattn.cu and the kernel cannot drift apart: adding a ratio means editing this list, the
// switch, and the assert, all in one file.
bool ggml_cuda_fattn_gqa_gfx1030_ratio_supported(const int ratio) {
    switch (ratio) {
        case 6: return true;
        case 8: return true;
        default: return false;
    }
}

// The admitted query-row counts. An explicit set, not a range test: 1 is plain decode and
// 2-4 are the speculative verify pass. Anything else (prefill batches, larger speculative
// drafts) keeps the TILE/VEC path, which is compiled for those shapes.
bool ggml_cuda_fattn_gqa_gfx1030_rows_supported(const int rows) {
    switch (rows) {
        case 1:
        case 2:
        case 3:
        case 4: return true;
        default: return false;
    }
}

template <int GROUP, int ROWS>
static void ggml_cuda_flash_attn_ext_gqa_gfx1030_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * V = dst->src[2];

    fattn_kernel_t kernel = V->type == GGML_TYPE_Q5_0
        ? flash_attn_ext_gqa_gfx1030<GROUP, ROWS, GGML_TYPE_Q5_0>
        : flash_attn_ext_gqa_gfx1030<GROUP, ROWS, GGML_TYPE_Q8_0>;

    // ncols1 == ROWS is what makes blockIdx.x the row tile: launch_fattn derives
    // gridDim.x = ceil(Q->ne[1]/ROWS) from it, and keeps the KV-split dimension (y) and
    // the (sequence, KV head) dimension (z) unchanged.
    launch_fattn<GQA_GFX1030_D, ROWS, GROUP>(
        ctx, dst, kernel, GROUP, 0, GQA_GFX1030_TILE, false, false, false, false);
}

template <int GROUP>
static void ggml_cuda_flash_attn_ext_gqa_gfx1030_group(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // 1 row keeps the narrow instance (and its arithmetic order); 2-4 rows share the wide
    // one, which reads K/V once per row tile instead of once per row.
    if (dst->src[0]->ne[1] == 1) {
        ggml_cuda_flash_attn_ext_gqa_gfx1030_impl<GROUP, 1>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_gqa_gfx1030_impl<GROUP, GQA_GFX1030_ROWS_MAX>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_gqa_gfx1030(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa_ratio = dst->src[0]->ne[2] / dst->src[1]->ne[2];

    GGML_ASSERT(ggml_cuda_fattn_gqa_gfx1030_rows_supported(dst->src[0]->ne[1]));

    // Explicit switch, not a range test: an unvalidated ratio is a bug in admission, not
    // something to run on the closest available geometry.
    switch (gqa_ratio) {
        case 6: ggml_cuda_flash_attn_ext_gqa_gfx1030_group<6>(ctx, dst); break;
        case 8: ggml_cuda_flash_attn_ext_gqa_gfx1030_group<8>(ctx, dst); break;
        default:
            GGML_ABORT("%s: no gfx1030 GQA instance for GQA ratio %d (only 6 and 8 are compiled)",
                    __func__, gqa_ratio);
    }
}

#endif // GGML_USE_HIP
