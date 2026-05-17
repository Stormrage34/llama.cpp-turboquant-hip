#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include "common.cuh"

#define RDNA2_MOE_MAX_EXPERTS 128


static uint32_t* g_moe_semaphores = nullptr;
static hipStream_t g_moe_stream = nullptr;
static bool g_moe_pipeline_initialized = false;

extern "C" void ggml_hip_moe_init_async_pipeline(int num_experts) {
    if (g_moe_pipeline_initialized) return;

    hipError_t err = hipStreamCreate(&g_moe_stream);
    if (err != hipSuccess) {
        fprintf(stderr, "[RDNA2 MOE] Failed to create MOE stream: %s\n", hipGetErrorString(err));
        return;
    }

    err = hipHostMalloc(&g_moe_semaphores, sizeof(uint32_t) * RDNA2_MOE_MAX_EXPERTS,
                         hipHostMallocPortable | hipHostMallocMapped);
    if (err != hipSuccess) {
        fprintf(stderr, "[RDNA2 MOE] Failed to allocate pinned semaphore memory: %s\n", hipGetErrorString(err));
        hipStreamDestroy(g_moe_stream);
        g_moe_stream = nullptr;
        return;
    }

    err = hipMemset(g_moe_semaphores, 0, sizeof(uint32_t) * RDNA2_MOE_MAX_EXPERTS);
    if (err != hipSuccess) {
        fprintf(stderr, "[RDNA2 MOE] Failed to zero semaphore memory: %s\n", hipGetErrorString(err));
        hipHostFree(g_moe_semaphores);
        g_moe_semaphores = nullptr;
        hipStreamDestroy(g_moe_stream);
        g_moe_stream = nullptr;
        return;
    }

    g_moe_pipeline_initialized = true;
    fprintf(stderr, "[RDNA2 MOE] Async pipeline initialized: %d experts, stream=%p, semaphores=%p\n",
            num_experts, (void *)g_moe_stream, (void *)g_moe_semaphores);
}

extern "C" void ggml_hip_moe_signal_expert_ready(int expert_id) {
    if (!g_moe_pipeline_initialized || expert_id < 0 || expert_id >= RDNA2_MOE_MAX_EXPERTS) return;

    hipStreamWriteValue32(g_moe_stream, &g_moe_semaphores[expert_id], 1, 0);
}

extern "C" void ggml_hip_moe_wait_expert(int expert_id, uintptr_t device_sem_addr) {
    if (!g_moe_pipeline_initialized || expert_id < 0 || expert_id >= RDNA2_MOE_MAX_EXPERTS) return;

    // Use device-side pointer if provided, otherwise fall back to host-mapped semaphore
    if (device_sem_addr != 0) {
        hipStreamWaitValue32(g_moe_stream, (void*)device_sem_addr, 1, 0);
    } else {
        hipStreamWaitValue32(g_moe_stream, &g_moe_semaphores[expert_id], 1, 0);
    }
}

extern "C" void ggml_hip_moe_destroy_pipeline() {
    if (!g_moe_pipeline_initialized) return;

    if (g_moe_semaphores) {
        hipHostFree(g_moe_semaphores);
        g_moe_semaphores = nullptr;
    }
    if (g_moe_stream) {
        hipStreamSynchronize(g_moe_stream);
        hipStreamDestroy(g_moe_stream);
        g_moe_stream = nullptr;
    }
    g_moe_pipeline_initialized = false;
}

extern "C" hipStream_t ggml_hip_moe_get_stream() {
    return g_moe_stream;
}
