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

int tria_get_n_kv(void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    auto * mem = llama_get_memory(ctx);
    if (!mem) return 0;

    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) return 0;

    /* seq_pos_max returns the highest position for seq 0, +1 gives count */
    llama_pos pmax = kv->seq_pos_max(0);
    return (pmax >= 0) ? (int)(pmax + 1) : 0;
}

} /* extern "C" */
