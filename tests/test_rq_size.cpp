#include <assert.h>
#include "../ggml/src/ggml-quants.h"
int main() {
    static_assert(sizeof(block_rq_mse_2) == 86, "block_rq_mse_2 size mismatch"); // rho[42]*2 + norm = 86
    static_assert(sizeof(block_rq_prod) == 348, "block_rq_prod size mismatch");   // norm + qs[336] + residual_norm + qjl_signs[4] with padding
    return 0;
}
