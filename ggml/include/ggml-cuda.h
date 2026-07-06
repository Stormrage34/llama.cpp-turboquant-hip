#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// forward declarations
struct ggml_cuda_moe_cache;

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// MoE expert cache buffer types and configuration
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_buffer_type(void);
GGML_BACKEND_API bool ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_type_t buft);

// Slot count: set from llama_model_load, consumed by dispatch hook
GGML_BACKEND_API void ggml_backend_cuda_moe_set_cache_slots(int n_slots);
GGML_BACKEND_API int  ggml_backend_cuda_moe_get_cache_slots(void);

// Observation API: model loader records expert tensors for pool preallocation
GGML_BACKEND_API void ggml_backend_cuda_moe_observe_expert_tensor(
    const void * tensor_data,
    const char * tensor_name,
    size_t       per_expert_bytes);

GGML_BACKEND_API void ggml_backend_cuda_moe_reset_expert_size_observation(void);

// Preallocate all observed pools for a given device
GGML_BACKEND_API void ggml_backend_cuda_moe_preallocate_pools(int device);

// Sibling prefetch: warm up caches for up/gate/down tensors
GGML_BACKEND_API void ggml_backend_cuda_moe_prefetch_experts(
    int             device,
    const char *    tensor_name,
    const int32_t * eids,
    int             n_eids);

// Aggregate stats from all per-tensor caches and reset counters
GGML_BACKEND_API void ggml_backend_cuda_moe_log_and_reset_stats(void);

#ifdef  __cplusplus
}
#endif
