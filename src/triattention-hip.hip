#include "triattention-hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

static __global__ void tria_hip_gather_rows_kernel(
        const uint8_t * tensor_data,
        const uint32_t * indices,
        uint8_t * scratch,
        size_t row_bytes,
        uint32_t n_move) {
    const size_t total_bytes = row_bytes * n_move;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_bytes) {
        return;
    }

    const size_t row = idx / row_bytes;
    const size_t col = idx - row * row_bytes;
    const size_t src_row = indices[row];

    scratch[idx] = tensor_data[src_row * row_bytes + col];
}

extern "C" int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes) {
    if (tensor_data == nullptr || h_indices == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    if (first_move >= n_keep || row_bytes == 0) {
        return 0;
    }

    const uint32_t n_move = n_keep - first_move;
    const size_t indices_bytes = static_cast<size_t>(n_move) * sizeof(uint32_t);
    const size_t scratch_bytes = static_cast<size_t>(n_move) * row_bytes;

    uint32_t * d_indices = nullptr;
    uint8_t * d_scratch = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void **>(&d_indices), indices_bytes);
    if (err != hipSuccess) {
        return static_cast<int>(err);
    }

    err = hipMalloc(reinterpret_cast<void **>(&d_scratch), scratch_bytes);
    if (err != hipSuccess) {
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }

    err = hipMemcpy(d_indices, h_indices + first_move, indices_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        (void) hipFree(d_scratch);
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }

    constexpr uint32_t block_size = 256;
    const size_t total_bytes = scratch_bytes;
    const uint32_t n_blocks = static_cast<uint32_t>((total_bytes + block_size - 1) / block_size);

    hipLaunchKernelGGL(
            tria_hip_gather_rows_kernel,
            dim3(n_blocks),
            dim3(block_size),
            0,
            0,
            static_cast<const uint8_t *>(tensor_data),
            d_indices,
            d_scratch,
            static_cast<size_t>(row_bytes),
            n_move);

    err = hipGetLastError();
    if (err == hipSuccess) {
        err = hipMemcpy(
                static_cast<uint8_t *>(tensor_data) + static_cast<size_t>(first_move) * row_bytes,
                d_scratch,
                scratch_bytes,
                hipMemcpyDeviceToDevice);
    }
    if (err == hipSuccess) {
        err = hipDeviceSynchronize();
    }

    (void) hipFree(d_scratch);
    (void) hipFree(d_indices);

    return static_cast<int>(err);
}
