// MoE expert cache — GPU slot pool + LRU bookkeeping + buffer type registration.
//
// This file owns:
//   - struct ggml_cuda_moe_cache : the GPU slot pool + LRU bookkeeping
//   - init / free / acquire / slot_ptr / stats : the cache C API
//   - ggml_backend_cuda_moe_cached_buffer_type : a ggml buffer type that
//     marks expert tensors as "live in CPU pinned memory and route GPU access
//     through the cache". The allocator uses the same pinned-memory allocator
//     as the existing CUDA host buffer type but with a distinct name so the
//     dispatch hook in ggml_cuda_mul_mat_id can distinguish them.

#include "moe-cache.cuh"
#include "common.cuh"

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    // Dedicated copy stream so H2D acquires can pipeline with compute
    moe_stream_t copy_stream;

    // Per-slot state
    std::vector<const void *> slot_to_host;  // [n_slots], nullptr if empty
    std::vector<uint64_t>     last_used;     // [n_slots]

    // host_ptr -> slot_id, O(1) lookup
    std::unordered_map<const void *, int> host_to_slot;

    uint64_t access_counter;
    std::mutex mu;

    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
};

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots) {

    if (slot_size_bytes == 0 || n_slots <= 0) {
        return nullptr;
    }

    int prev_device = 0;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(device));

    auto * c = new ggml_cuda_moe_cache;
    c->device          = device;
    c->slot_size_bytes = slot_size_bytes;
    c->n_slots         = n_slots;
    c->slot_pool_d     = nullptr;
    c->copy_stream     = nullptr;
    c->access_counter  = 0;

    cudaError_t err = cudaMalloc(&c->slot_pool_d, (size_t)n_slots * slot_size_bytes);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("moe-cache: device %d cudaMalloc(%zu bytes) failed: %s\n",
                       device, (size_t)n_slots * slot_size_bytes, cudaGetErrorString(err));
        (void)cudaGetLastError();
        delete c;
        return nullptr;
    }

    err = cudaStreamCreateWithFlags(&c->copy_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("moe-cache: device %d cudaStreamCreateWithFlags failed: %s\n",
                       device, cudaGetErrorString(err));
        (void)cudaGetLastError();
        cudaFree(c->slot_pool_d);
        delete c;
        return nullptr;
    }

    c->slot_to_host.assign(n_slots, nullptr);
    c->last_used  .assign(n_slots, 0);
    c->host_to_slot.reserve(n_slots * 2);

    CUDA_CHECK(cudaSetDevice(prev_device));
    return c;
}

extern "C"
void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;

    int prev_device = 0;
    (void)cudaGetDevice(&prev_device);
    (void)cudaSetDevice(cache->device);

    if (cache->copy_stream) {
        CUDA_CHECK(cudaStreamDestroy(cache->copy_stream));
    }
    if (cache->slot_pool_d) {
        CUDA_CHECK(cudaFree(cache->slot_pool_d));
    }
    (void)cudaSetDevice(prev_device);

    delete cache;
}

extern "C"
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    moe_stream_t copy_stream) {

    if (!cache || host_src == nullptr || byte_count == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    if (byte_count > cache->slot_size_bytes) {
        return -1;
    }

    // Hit path: O(1) hash lookup
    auto it = cache->host_to_slot.find(host_src);
    if (it != cache->host_to_slot.end()) {
        int slot = it->second;
        cache->last_used[slot] = ++cache->access_counter;
        cache->hits.fetch_add(1, std::memory_order_relaxed);
        return slot;
    }

    // Miss: pick the LRU slot
    int      lru_slot = 0;
    uint64_t lru_t    = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < cache->n_slots; ++i) {
        if (cache->last_used[i] < lru_t) {
            lru_t    = cache->last_used[i];
            lru_slot = i;
        }
    }

    const void * evicted = cache->slot_to_host[lru_slot];
    if (evicted != nullptr) {
        cache->host_to_slot.erase(evicted);
        cache->evictions.fetch_add(1, std::memory_order_relaxed);
    }

    cache->slot_to_host[lru_slot] = host_src;
    cache->host_to_slot[host_src] = lru_slot;
    cache->last_used[lru_slot]    = ++cache->access_counter;
    cache->misses.fetch_add(1, std::memory_order_relaxed);

    void * dst = (char *)cache->slot_pool_d + (size_t)lru_slot * cache->slot_size_bytes;
    CUDA_CHECK(cudaMemcpyAsync(dst, host_src, byte_count, cudaMemcpyHostToDevice, copy_stream));

    return lru_slot;
}

extern "C"
bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes) {

    if (!cache || min_slot_size_bytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    if (min_slot_size_bytes <= cache->slot_size_bytes) {
        return true;
    }

    int prev_device = 0;
    (void)cudaGetDevice(&prev_device);
    (void)cudaSetDevice(cache->device);

    void * new_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&new_pool, (size_t)cache->n_slots * min_slot_size_bytes));

    // Sync the copy stream before freeing the old pool to avoid use-after-free
    // if an acquire was issued just before grow() on another thread.
    if (cache->copy_stream) {
        CUDA_CHECK(cudaStreamSynchronize(cache->copy_stream));
    }
    if (cache->slot_pool_d) {
        CUDA_CHECK(cudaFree(cache->slot_pool_d));
    }
    cache->slot_pool_d     = new_pool;
    cache->slot_size_bytes = min_slot_size_bytes;

    // Existing cached state is invalidated by realloc
    std::fill(cache->slot_to_host.begin(), cache->slot_to_host.end(), nullptr);
    std::fill(cache->last_used.begin(),    cache->last_used.end(),    0ull);
    cache->host_to_slot.clear();
    cache->access_counter = 0;

    GGML_LOG_INFO("moe-cache: device %d  grew slot_size to %.2f MiB  pool=%.2f MiB\n",
                  cache->device,
                  min_slot_size_bytes / 1024.0 / 1024.0,
                  ((double)cache->n_slots * min_slot_size_bytes) / 1024.0 / 1024.0);

    (void)cudaSetDevice(prev_device);
    return true;
}

extern "C"
void * ggml_cuda_moe_cache_slot_ptr(struct ggml_cuda_moe_cache * cache, int slot_id) {
    if (!cache || slot_id < 0 || slot_id >= cache->n_slots) {
        return nullptr;
    }
    return (char *)cache->slot_pool_d + (size_t)slot_id * cache->slot_size_bytes;
}

extern "C"
size_t ggml_cuda_moe_cache_slot_size_bytes(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->slot_size_bytes : 0;
}

extern "C"
int ggml_cuda_moe_cache_n_slots(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->n_slots : 0;
}

extern "C"
moe_stream_t ggml_cuda_moe_cache_copy_stream(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->copy_stream : nullptr;
}

extern "C"
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions) {
    if (!cache) {
        if (out_hits)      *out_hits      = 0;
        if (out_misses)    *out_misses    = 0;
        if (out_evictions) *out_evictions = 0;
        return;
    }
    if (out_hits)      *out_hits      = cache->hits.load(std::memory_order_relaxed);
    if (out_misses)    *out_misses    = cache->misses.load(std::memory_order_relaxed);
    if (out_evictions) *out_evictions = cache->evictions.load(std::memory_order_relaxed);
}

extern "C"
void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;
    cache->hits.store(0, std::memory_order_relaxed);
    cache->misses.store(0, std::memory_order_relaxed);
    cache->evictions.store(0, std::memory_order_relaxed);
}

// ===========================================================================
// ggml buffer type: CUDA_MoE_Cached
// ===========================================================================

static const char * ggml_backend_cuda_moe_cached_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_CUDA_NAME "_MoE_Cached";
}

static void ggml_backend_cuda_moe_cached_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_moe_cached_pinned_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n",
                       __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {

    void * ptr = ggml_cuda_moe_cached_pinned_malloc(size);

    if (ptr == nullptr) {
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft             = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_buffer_free_buffer;
    return buffer;
}

static bool ggml_backend_cuda_moe_cached_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // Returns false because CUDA_MoE_Cached weights must be routed to the
    // CUDA (or HIP) backend for kernel dispatch, not the CPU backend.
    // Although the backing memory is CPU-accessible (pinned via cudaMallocHost),
    // the scheduler's is_host flag controls backend routing — false ensures
    // the CUDA backend handles these tensors.  GPU kernels on UVA-capable
    // devices (all modern NVIDIA + AMD gfx1030+) can access the pinned host
    // memory directly, so no explicit H2D copy is needed at the buffer level.
    // The optional MoE cache layer (ggml_cuda_moe_cache_acquire) stages slabs
    // to the GPU slot pool for LRU residency when is_host=false tensors are read.
    return false;
}

extern "C"
ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_moe_cached = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_moe_cached_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cuda_moe_cached_buffer_type_is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };
    return &ggml_backend_cuda_buffer_type_moe_cached;
}

extern "C"
bool ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_type_t buft) {
    return buft != nullptr
        && buft->iface.get_name == ggml_backend_cuda_moe_cached_buffer_type_name;
}

// ===========================================================================
// Per-tensor cache registry
// ===========================================================================

namespace {

struct moe_cache_key {
    int         device;
    std::string tensor_name;
    bool operator<(const moe_cache_key & o) const {
        if (device != o.device) return device < o.device;
        return tensor_name < o.tensor_name;
    }
};

struct moe_cache_registry {
    std::mutex mu;
    std::map<moe_cache_key, ggml_cuda_moe_cache *> by_key;
};

static moe_cache_registry & get_registry() {
    static moe_cache_registry inst;
    return inst;
}

} // namespace

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create_for_tensor(
    int          device,
    const void * tensor_data,
    size_t       slot_size_bytes,
    int          n_slots,
    const char * tensor_name_for_log) {
    GGML_UNUSED(tensor_data);

    if (tensor_name_for_log == nullptr || tensor_name_for_log[0] == '\0') {
        return nullptr;
    }

    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    moe_cache_key k{device, std::string(tensor_name_for_log)};
    auto it = reg.by_key.find(k);
    if (it != reg.by_key.end()) {
        return it->second;
    }

    ggml_cuda_moe_cache * c = ggml_cuda_moe_cache_init(device, slot_size_bytes, n_slots);
    if (!c) {
        return nullptr;
    }

    reg.by_key.emplace(k, c);
    GGML_LOG_INFO("load_tensors: CUDA_MoE_Cache_Pool[%-32s] = %7.2f MiB  (%d slots \xd7 %.2f MiB)\n",
                  tensor_name_for_log,
                  ((double)n_slots * slot_size_bytes) / 1024.0 / 1024.0,
                  n_slots,
                  slot_size_bytes / 1024.0 / 1024.0);
    return c;
}

extern "C"
void ggml_cuda_moe_cache_free_all(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    for (auto & kv : reg.by_key) {
        ggml_cuda_moe_cache_free(kv.second);
    }
    reg.by_key.clear();
}

// ===========================================================================
// Slot count config (settable from llama_model_load)
// ===========================================================================

static std::atomic<int> g_moe_cache_slots{0};

extern "C"
void ggml_backend_cuda_moe_set_cache_slots(int n_slots) {
    if (n_slots < 0) n_slots = 0;
    g_moe_cache_slots.store(n_slots, std::memory_order_relaxed);
}

extern "C"
int ggml_backend_cuda_moe_get_cache_slots(void) {
    return g_moe_cache_slots.load(std::memory_order_relaxed);
}

// ===========================================================================
// Per-tensor observation (model loader records tensors for pool preallocation)
// ===========================================================================

namespace {
struct observed_tensor {
    const void * tensor_data;
    std::string  tensor_name;
    size_t       per_expert_bytes;
};
struct observation_state {
    std::mutex mu;
    std::vector<observed_tensor> tensors;
};
static observation_state & get_observation_state() {
    static observation_state inst;
    return inst;
}
} // namespace

extern "C"
void ggml_backend_cuda_moe_observe_expert_tensor(
    const void * tensor_data,
    const char * tensor_name,
    size_t       per_expert_bytes) {
    if (tensor_data == nullptr || per_expert_bytes == 0) return;
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    st.tensors.push_back({tensor_data,
                          tensor_name ? std::string(tensor_name) : std::string(),
                          per_expert_bytes});
}

extern "C"
void ggml_backend_cuda_moe_reset_expert_size_observation(void) {
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    st.tensors.clear();
}

extern "C"
void ggml_backend_cuda_moe_preallocate_pools(int device) {
    const int n_slots = ggml_backend_cuda_moe_get_cache_slots();
    if (n_slots <= 0) return;
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    for (const auto & t : st.tensors) {
        ggml_cuda_moe_cache_get_or_create_for_tensor(
            device, t.tensor_data, t.per_expert_bytes, n_slots,
            t.tensor_name.empty() ? "?" : t.tensor_name.c_str());
    }
}

extern "C"
void ggml_backend_cuda_moe_prefetch_experts(
    int             device,
    const char *    tensor_name,
    const int32_t * eids,
    int             n_eids) {
    if (!tensor_name || !eids || n_eids <= 0) return;

    const observed_tensor * found = nullptr;
    {
        auto & st = get_observation_state();
        std::lock_guard<std::mutex> lk(st.mu);
        for (const auto & t : st.tensors) {
            if (t.tensor_name == tensor_name) {
                found = &t;
                break;
            }
        }
        if (!found) return;
    }

    const void * tensor_data    = found->tensor_data;
    const size_t expert_stride  = found->per_expert_bytes;

    auto & reg = get_registry();
    ggml_cuda_moe_cache * cache = nullptr;
    {
        std::lock_guard<std::mutex> lk(reg.mu);
        moe_cache_key k{device, std::string(tensor_name)};
        auto it = reg.by_key.find(k);
        if (it == reg.by_key.end()) return;
        cache = it->second;
    }

    if (!cache) return;
    moe_stream_t copy_stream = cache->copy_stream;
    const char * src_base = (const char *)tensor_data;

    for (int i = 0; i < n_eids; ++i) {
        int32_t eid = eids[i];
        if (eid < 0) continue;
        const void * host_ptr = src_base + (size_t)eid * expert_stride;
        (void)ggml_cuda_moe_cache_acquire(cache, host_ptr, expert_stride, copy_stream);
    }
}

extern "C"
void ggml_backend_cuda_moe_preallocate_pool(int device) {
    ggml_backend_cuda_moe_preallocate_pools(device);
}

extern "C"
void ggml_backend_cuda_moe_log_and_reset_stats(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    uint64_t total_hits = 0, total_misses = 0, total_evictions = 0;
    size_t   n_caches = reg.by_key.size();
    for (auto & kv : reg.by_key) {
        uint64_t h = 0, m = 0, e = 0;
        ggml_cuda_moe_cache_stats(kv.second, &h, &m, &e);
        total_hits      += h;
        total_misses    += m;
        total_evictions += e;
        ggml_cuda_moe_cache_reset_stats(kv.second);
    }
    const uint64_t total = total_hits + total_misses;
    const double rate = total > 0 ? 100.0 * (double)total_hits / (double)total : 0.0;
    GGML_LOG_INFO("moe-cache: %zu caches  hits=%llu  misses=%llu  evictions=%llu  hit-rate=%.2f%%\n",
                  n_caches,
                  (unsigned long long)total_hits,
                  (unsigned long long)total_misses,
                  (unsigned long long)total_evictions,
                  rate);
}
