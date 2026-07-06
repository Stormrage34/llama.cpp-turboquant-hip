#include "turbo-innerq.cuh"
#include <atomic>
#include <cstring>

// Host-side shared state for InnerQ cross-TU communication
TURBO_IQ_API std::atomic<bool> g_innerq_finalized{false};
TURBO_IQ_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static std::atomic<bool> g_innerq_tensor_needs_update{false};

void turbo_innerq_publish(const float * scale_inv, int group_size) {
    for (int i = 0; i < group_size && i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = scale_inv[i];
    }
    for (int i = group_size; i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = 1.0f;
    }
    g_innerq_finalized.store(true, std::memory_order_release);
    g_innerq_tensor_needs_update.store(true, std::memory_order_release);
}

TURBO_IQ_API bool turbo_innerq_needs_tensor_update(void) {
    return g_innerq_tensor_needs_update.load(std::memory_order_acquire);
}

TURBO_IQ_API void turbo_innerq_mark_tensor_updated(void) {
    g_innerq_tensor_needs_update.store(false, std::memory_order_release);
}
