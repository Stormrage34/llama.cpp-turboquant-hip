/*
 * triattention-bridge.cpp — C++ bridge for accessing llama internals from C
 */

#include "llama.h"
#include "llama-kv-cache.h"

extern "C" {
#include "triattention-runtime.h"

struct ggml_tensor * tria_get_k_tensor(void * ctx_void, int layer_idx) {
    auto * ctx = (llama_context *)ctx_void;
    auto * mem = llama_get_memory(ctx);
    if (!mem) return nullptr;

    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) return nullptr;

    return kv->get_k_raw(layer_idx);
}

} /* extern "C" */
