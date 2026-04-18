/*
 * triattention-backend.h — function pointer ABI for optional HIP acceleration
 *
 * When GGML_BACKEND_DL is OFF (static linking), pointers resolve to
 * triattention-hip.hip symbols directly.
 * When GGML_BACKEND_DL is ON (dynamic loading), pointers are NULL
 * and scoring falls back to CPU.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tria_backend {
    int  (*stats_upload)(const float *, int, const float *, const float *, int,
                         float **, float **, float **);
    void (*stats_free)(float *, float *, float *);
    int  (*score_q8_0)(const void *, int, int, int, int, int, int,
                       const int *, const float *, const float *, const float *,
                       int, float, float *, int, const int *);
    int  (*scores_download)(float *, const float *, int);
    int  (*compact_rows)(void *, const uint32_t *, uint32_t, uint32_t, uint32_t);
};

/* Global backend — initialized by tria_backend_init() */
extern struct tria_backend g_tria_backend;

/* Returns 1 if GPU acceleration is available */
int tria_backend_init(void);

#ifdef __cplusplus
}
#endif
