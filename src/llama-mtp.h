#pragma once

#include "llama.h"

#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // sized to n_ubatch

    // Cross-ubatch shift state: pair (h_p, x_{p+1}) at MTP pos p+1. The last
    // h-row of one ubatch needs the first token of the NEXT ubatch to pair
    // with, so it's stashed here until that next ubatch fires. Resets when
    // pos_start of the new ubatch != pending_pos+1 (new prompt or seq_rm gap).
    std::vector<float> pending_h;
    llama_pos          pending_pos = -1;

    // Deferred MTP decode staging.
    //
    // Inside process_ubatch() we synchronize and copy hidden-state rows into
    // a host buffer (staged_data) but do NOT call llama_decode(ctx_mtp) yet.
    // After the ubatch loop finishes and the target scheduler's graph is freed
    // (ggml_backend_sched_reset), the staged batches are flushed to ctx_mtp.
    // This avoids nested GPU memory allocation that can exhaust VRAM.
    struct StagedBatch {
        std::vector<float>         h;          // [n_embd * n_out] flattened hidden rows
        std::vector<llama_token>   tokens;     // [n_out] token IDs for the MTP batch
        std::vector<llama_pos>     positions;  // [n_out] positions for the MTP batch
        int32_t                    n_out;      // number of MTP tokens in this batch (≤ n_ubatch)
    };
    std::vector<StagedBatch> staged_batches;
};
