#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

void ggml_hip_moe_init_async_pipeline(int num_experts);
void ggml_hip_moe_signal_expert_ready(int expert_id);
void ggml_hip_moe_wait_expert(int expert_id, uintptr_t device_sem_addr);
void ggml_hip_moe_destroy_pipeline();
hipStream_t ggml_hip_moe_get_stream();
