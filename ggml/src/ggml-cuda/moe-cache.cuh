#pragma once

// Stream type: on HIP we use hipStream_t, on CUDA moe_stream_t.
// The header pulls in the runtime via the backend-internal compat path
// so both names work.  We typedef to moe_stream_t for the public API.
#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
typedef hipStream_t moe_stream_t;
#else
#include <cuda_runtime.h>
typedef cudaStream_t moe_stream_t;
#endif
#include <stdint.h>

// MoE expert cache: keeps a fixed-size GPU slot pool of expert weight slabs
// while cold experts live in CPU pinned memory. On routing miss the slab is
// async-copied H2D and an LRU slot is evicted to make room.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda_moe_cache;

struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots);

void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache);

int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    moe_stream_t copy_stream);

bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes);

void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id);

size_t       ggml_cuda_moe_cache_slot_size_bytes(const struct ggml_cuda_moe_cache * cache);
int          ggml_cuda_moe_cache_n_slots(const struct ggml_cuda_moe_cache * cache);
moe_stream_t ggml_cuda_moe_cache_copy_stream(const struct ggml_cuda_moe_cache * cache);

void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions);

void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache);

struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create_for_tensor(
    int          device,
    const void * tensor_data,
    size_t       slot_size_bytes,
    int          n_slots,
    const char * tensor_name_for_log);

struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create(
    int    device,
    size_t slot_size_bytes,
    int    n_slots);

void ggml_cuda_moe_cache_free_all(void);

#ifdef __cplusplus
}
#endif
