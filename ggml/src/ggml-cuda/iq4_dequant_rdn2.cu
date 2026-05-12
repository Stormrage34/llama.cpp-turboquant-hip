// RDNA2-optimized IQ4_XS dequant kernel (Kernel A)
// Compiled only when RDNA2_OPT_V1 is set for this file.
#include "convert.cuh"
#include "iq4_dequant_rdn2.cuh"

#ifdef RDNA2_OPT_V1

// Extern C wrapper for linking with test binaries
extern "C" void ggml_dequant_iq4_xs_rdn2_extern(const void * vx, half * y, const int64_t k, cudaStream_t stream) {
    ggml_dequant_iq4_xs_rdn2(vx, y, k, stream);
}

#endif // RDNA2_OPT_V1
