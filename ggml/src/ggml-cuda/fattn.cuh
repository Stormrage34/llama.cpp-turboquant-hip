#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
void ggml_cuda_flash_attn_ext_gqa_gfx1030(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// admission policy of the gfx1030 GQA decode kernel, exposed so that device init can report it
bool        ggml_cuda_fattn_gqa_gfx1030_enabled();
const char * ggml_cuda_fattn_gqa_gfx1030_env();
const char * ggml_cuda_fattn_gqa_gfx1030_ratios();
int         ggml_cuda_fattn_gqa_gfx1030_head_dim();
bool        ggml_cuda_fattn_gqa_gfx1030_admitted(int cc);
// defined in fattn-gqa-gfx1030.cu, next to the kernel instances
bool        ggml_cuda_fattn_gqa_gfx1030_ratio_supported(int ratio);
bool        ggml_cuda_fattn_gqa_gfx1030_rows_supported(int rows);
#endif

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
