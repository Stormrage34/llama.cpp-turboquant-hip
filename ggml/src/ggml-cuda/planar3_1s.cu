// PlanarQuant 3-bit v1s dequant kernel (placeholder)
// Forwards to the existing planar3_0 dequant kernel
// Identical memory layout — only the type enum differs

#include "planar-iso-dequant.cuh"

// Forward to existing planar3_0 dequant kernels
// The dequantize_planar3_0 and dequantize_block_cont_cuda<QK_PLANAR3, QR_PLANAR3, dequantize_planar3_0>
// templates handle the actual dequantization logic.
// This file exists solely to register the GGML_TYPE_PLANAR3_1S type in the dispatch switch.
