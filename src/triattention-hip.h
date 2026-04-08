#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes);

#ifdef __cplusplus
}
#endif
