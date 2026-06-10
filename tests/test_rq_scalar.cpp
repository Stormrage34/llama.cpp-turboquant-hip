// Test that per‑group scalar rho is correctly computed and stored as half‑precision.
// Uses the host‑side quantizer quantize_row_rq_mse_2_ref.

#include "ggml.h"
#include "../ggml/src/ggml-quants.h"
#include <cstdio>
#include <cstring>
#include <cmath>
// Duplicate fp16_encode from ggml-turbo-quant.c for test purposes
static inline uint16_t fp16_encode_test(float f) {
    uint32_t i = *(uint32_t*)&f;
    int exp = (i >> 23) & 0xFF;
    if (exp == 255) return (i & 0x8000) | 0x7C00; // Inf/NaN -> Inf
    if (exp >= 143) return (i & 0x8001) | ((245 - exp) << 13); // Overflow -> Inf
    if (exp > 126) return (i & 0x8007F) | ((exp - 126) << 13);
    int mant = (i & 0x7F800) >> 7;
    return (i & 0x8000) | (mant >> (126 - exp)) | (((mant >> (125 - exp)) & 1) ^ ((mant >> (125 - exp)) & ((-(mant >> (125 - exp))) & 1)));
}

int main() {
    const int64_t k = 128; // vector length (multiple of 128)
    float x[k];
    // Fill with deterministic values: pattern repeats 0.0, 0.1, ..., 0.9
    for (int i = 0; i < k; ++i) {
        x[i] = static_cast<float>((i % 10) * 0.1f);
    }
    block_rq_mse_2 blk;
    // Zero init
    memset(&blk, 0, sizeof(blk));
    quantize_row_rq_mse_2_ref(x, &blk, k);

    // Compute expected per‑group RMS and compare with stored half‑precision.
    const int n_groups = (k + 2) / 3; // matches quantizer logic
    bool ok = true;
    for (int g = 0; g < n_groups && g < 42; ++g) {
        float sum_sq = 0.0f;
        int base = g * 3;
        for (int c = 0; c < 3; ++c) {
            if (base + c < k) {
                float v = x[base + c];
                sum_sq += v * v;
            }
        }
        float expected = std::sqrt(sum_sq);
        uint16_t encoded = fp16_encode_test(expected);
        if (blk.rho[g] != encoded) {
            printf("Group %d: mismatch expected 0x%04x got 0x%04x\n", g, encoded, blk.rho[g]);
            ok = false;
        }
    }
    if (!ok) {
        return 1;
    }
    printf("All rho values match expected half‑precision encoding.\n");
    return 0;
}
