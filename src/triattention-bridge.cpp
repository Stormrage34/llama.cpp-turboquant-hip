/*
 * triattention-bridge.cpp — C++ bridge for accessing llama internals from C
 */

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <numeric>
#include <vector>

/* Helper: extract llama_kv_cache from either pure KV or hybrid memory */
static llama_kv_cache * get_kv(void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    auto * mem = llama_get_memory(ctx);
    if (!mem) return nullptr;

    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv) return kv;

    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (hybrid) return hybrid->get_mem_attn();

    return nullptr;
}

extern "C" {
#include "triattention-runtime.h"

struct ggml_tensor * tria_get_k_tensor(void * ctx_void, int layer_idx) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return nullptr;
    return kv->get_k_raw(layer_idx);
}

int tria_get_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    llama_pos pmax = kv->seq_pos_max(0);
    return (pmax >= 0) ? (int) (pmax + 1) : 0;
}

int tria_get_used_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    return (int) kv->get_used_n_kv();
}

int tria_get_kv_positions(void * ctx_void, int * positions, int max_positions) {
    auto * ctx = (llama_context *)ctx_void;
    if (!ctx || !positions || max_positions <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    std::vector<llama_pos> kv_positions;
    if (!kv->get_cell_positions(kv_positions)) {
        return 0;
    }

    const int n = std::min<int>((int) kv_positions.size(), max_positions);
    for (int i = 0; i < n; ++i) {
        positions[i] = (int) kv_positions[i];
    }

    return n;
}

int tria_compact_kv(struct tria_runtime * rt, void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    if (!rt || !ctx || !rt->global_scores || rt->global_budget <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    const int n_kv = (int) kv->get_used_n_kv();
    const int n_old = n_kv - rt->window;
    if (n_old <= 0) {
        return 0;
    }

    int budget = rt->global_budget;
    budget = std::max(1, std::min(budget, n_old));

    std::vector<uint32_t> ranked(n_old);
    std::iota(ranked.begin(), ranked.end(), 0u);

    std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) {
        if (rt->global_scores[a] == rt->global_scores[b]) {
            return a < b;
        }
        return rt->global_scores[a] > rt->global_scores[b];
    });

    ranked.resize(budget);
    std::sort(ranked.begin(), ranked.end());

    std::vector<uint32_t> keep_positions;
    keep_positions.reserve(budget + rt->window);
    keep_positions.insert(keep_positions.end(), ranked.begin(), ranked.end());

    for (int pos = n_old; pos < n_kv; ++pos) {
        keep_positions.push_back((uint32_t) pos);
    }

    if ((int) keep_positions.size() >= n_kv) {
        return 0;
    }

    llama_synchronize(ctx);

    if (!kv->triattention_compact(keep_positions)) {
        return 0;
    }

    return n_kv - (int) keep_positions.size();
}

} /* extern "C" */
