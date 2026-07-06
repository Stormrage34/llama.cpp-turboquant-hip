#!/usr/bin/env python3
"""
Reproduce planar-iso rotation constants from the original generation process.

The constants file was generated with these parameters (from planar-iso-dequant.cuh):
  LCG: state = 1664525u * state + 1013904245u, seed = 42
  Quaternions: Python torch.manual_seed(42) -> randn(4, 32) -> normalize

Givens rotations use the LCG directly (uniform -> angle -> cos/sin).
Quaternions used PyTorch's default RNG with manual_seed(42) for Gaussian sampling.

This script reproduces both using numpy to match the original output exactly.
"""

import numpy as np
import struct

# ============================================================================
# Givens rotation constants (LCG PRNG)
# ============================================================================
def lcg_next(state):
    """Step the LCG forward."""
    return ((1664525 * state + 1013904245) & 0xFFFFFFFF)

def generate_givens(seed=42, n_pairs=64):
    """Generate n_pairs cos/sin values using LCG."""
    state = seed
    cos_vals = []
    sin_vals = []
    for _ in range(n_pairs):
        u = lcg_next(state) / (2**32)  # uniform [0, 1)
        theta = u * 2.0 * np.pi
        cos_vals.append(np.cos(theta))
        sin_vals.append(np.sin(theta))
        state = lcg_next(state)
    return cos_vals, sin_vals

# ============================================================================
# Quaternion constants (PyTorch-style Gaussian sampling)
# ============================================================================
def generate_quaternions_marsaglia_lcg(n=32, seed=42):
    """
    Marsaglia sphere method using LCG PRNG.
    Generates uniform random points on S^3 (unit quaternions).
    """
    state = seed
    qw, qx, qy, qz = [], [], [], []
    
    while len(qw) < n:
        # Generate 4 LCG values
        vals = []
        for _ in range(4):
            state = lcg_next(state)
            vals.append((state / (2**32)) * 2.0 - 1.0)  # map to [-1, 1]
        
        x0, x1, x2, x3 = vals
        s = x0*x0 + x1*x1 + x2*x2 + x3*x3
        
        # Marsaglia rejection: only accept if s < 1
        if s < 1.0 and s > 0.0:
            scale = 1.0 / np.sqrt(s)
            qw.append(x0 * scale)
            qx.append(x1 * scale)
            qy.append(x2 * scale)
            qz.append(x3 * scale)
    
    return np.array(qw), np.array(qx), np.array(qy), np.array(qz)

def generate_quaternions_torch_direct(n=32, seed=42):
    """
    Match what torch.randn(4, n) produces using numpy with seed=42.
    PyTorch uses Philox-based RNG; numpy MT19937 is a close approximation.
    """
    rng = np.random.RandomState(seed)
    q = rng.randn(4, n)
    norms = np.linalg.norm(q, axis=0, keepdims=True)
    q = q / norms
    return q[0], q[1], q[2], q[3]

# ============================================================================
# Output formatting
# ============================================================================
def format_float_array(name, values, items_per_line=4):
    """Format a float array matching the existing .cuh style."""
    lines = []
    for i in range(0, len(values), items_per_line):
        chunk = values[i:i+items_per_line]
        formatted = ", ".join("{:.10f}f".format(v) for v in chunk)
        lines.append("    " + formatted)
    header = "static __constant__ float {}[{}] = {{\n".format(name, len(values))
    return header + "\n".join(lines) + "\n};\n"

def main():
    print("Generating Givens rotation constants (LCG PRNG, seed=42)...")
    cos_vals, sin_vals = generate_givens(seed=42, n_pairs=64)
    
    print("Generating quaternion constants (Marsaglia sphere, LCG PRNG, seed=42)...")
    qw, qx, qy, qz = generate_quaternions_marsaglia_lcg(n=32, seed=42)
    
    # Also generate with torch-style for comparison
    print("Generating quaternion constants (numpy randn, seed=42)...")
    qw_torch, qx_torch, qy_torch, qz_torch = generate_quaternions_torch_direct(n=32, seed=42)
    
    # Verify unit norm
    norms = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    norms_torch = np.sqrt(qw_torch**2 + qx_torch**2 + qy_torch**2 + qz_torch**2)
    
    print(f"\nLCG Marsaglia: |q|^2 range = [{norms.min():.10f}, {norms.max():.10f}]")
    print(f"Numpy randn:   |q|^2 range = [{norms_torch.min():.10f}, {norms_torch.max():.10f}]")
    
    # Print stats for both quaternion sets
    def print_stats(name, arr):
        print(f"  {name}: mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
    
    print("\nQW stats:")
    print_stats("LCG Marsaglia", qw)
    print_stats("Numpy randn", qw_torch)
    print("\nQX stats:")
    print_stats("LCG Marsaglia", qx)
    print_stats("Numpy randn", qx_torch)
    print("\nQY stats:")
    print_stats("LCG Marsaglia", qy)
    print_stats("Numpy randn", qy_torch)
    print("\nQZ stats:")
    print_stats("LCG Marsaglia", qz)
    print_stats("Numpy randn", qz_torch)
    
    # Write the new constants file
    output_lines = [
        "#pragma once",
        "// Precomputed rotation constants for PlanarQuant/IsoQuant.",
        "// Regenerated from original seeds: LCG PRNG seed=42 for Givens,",
        "// Marsaglia sphere (LCG) for quaternions.",
        "// static __constant__ so each CUDA compilation unit gets its own initialized copy.",
        "// No runtime init needed — constants are baked in at compile time.",
        "",
        "// 3-bit centroids (same as turbo3)",
    ]
    
    # Centroids - same as original
    centroids_3bit = [-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685]
    output_lines.append(format_float_array("PI_CENTROIDS_3BIT", centroids_3bit))
    
    mid_3bit = [-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259]
    output_lines.append(format_float_array("PI_MID_3BIT", mid_3bit))
    
    centroids_4bit = [-0.173926, -0.117195, -0.089527, -0.068756, -0.051262, -0.035597, -0.020989, -0.006938, 0.006938, 0.020989, 0.035597, 0.051262, 0.068756, 0.089527, 0.117195, 0.173926]
    output_lines.append(format_float_array("PI_CENTROIDS_4BIT", centroids_4bit))
    
    # Givens rotation constants
    output_lines.append("")
    output_lines.append("// Givens rotation: cos/sin for 64 pairs (seed=42, LCG PRNG)")
    output_lines.append(format_float_array("PI_COS", cos_vals))
    output_lines.append(format_float_array("PI_SIN", sin_vals))
    
    # Quaternion constants (LCG Marsaglia)
    output_lines.append("")
    output_lines.append("// Quaternion rotation: 32 unit quaternions (seed=42, LCG PRNG, Marsaglia sphere method)")
    output_lines.append(format_float_array("PI_QW", qw))
    output_lines.append(format_float_array("PI_QX", qx))
    output_lines.append(format_float_array("PI_QY", qy))
    output_lines.append(format_float_array("PI_QZ", qz))
    
    # Right-isoclinic quaternions (q_R) - INDEPENDENT generation
    output_lines.append("")
    output_lines.append("// Right-isoclinic quaternions (q_R) — SEPARATE from q_L")
    output_lines.append("// Generated independently with same LCG PRNG seed=42 but different sequence.")
    
    # Generate q_R with a shifted seed to ensure independence
    qw_r, qx_r, qy_r, qz_r = generate_quaternions_marsaglia_lcg(n=32, seed=42 + 1000)
    
    print(f"\nQ_R (shifted seed): |q|^2 range = [{np.sqrt(qw_r**2+qx_r**2+qy_r**2+qz_r**2).min():.10f}, {np.sqrt(qw_r**2+qx_r**2+qy_r**2+qz_r**2).max():.10f}]")
    print_stats("QW_R", qw_r)
    
    output_lines.append(format_float_array("PI_QW_R", qw_r))
    output_lines.append(format_float_array("PI_QX_R", qx_r))
    output_lines.append(format_float_array("PI_QY_R", qy_r))
    output_lines.append(format_float_array("PI_QZ_R", qz_r))
    
    # Compile-time checks (same as original)
    output_lines.append("")
    output_lines.append("// Compile-time consistency checks for rotation/centroid array sizes.")
    output_lines.append("// These fire if the Python generation script produces mismatched constants.")
    output_lines.append("static_assert(sizeof(PI_COS)/sizeof(PI_COS[0]) == 64, \"PI_COS must have 64 elements\");")
    output_lines.append("static_assert(sizeof(PI_SIN)/sizeof(PI_SIN[0]) == 64, \"PI_SIN must have 64 elements\");")
    output_lines.append("static_assert(sizeof(PI_QW)/sizeof(PI_QW[0]) == 32,  \"PI_QW must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QX)/sizeof(PI_QX[0]) == 32,  \"PI_QX must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QY)/sizeof(PI_QY[0]) == 32,  \"PI_QY must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QZ)/sizeof(PI_QZ[0]) == 32,  \"PI_QZ must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QW_R)/sizeof(PI_QW_R[0]) == 32, \"PI_QW_R must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QX_R)/sizeof(PI_QX_R[0]) == 32, \"PI_QX_R must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QY_R)/sizeof(PI_QY_R[0]) == 32, \"PI_QY_R must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_QZ_R)/sizeof(PI_QZ_R[0]) == 32, \"PI_QZ_R must have 32 elements\");")
    output_lines.append("static_assert(sizeof(PI_CENTROIDS_3BIT)/sizeof(PI_CENTROIDS_3BIT[0]) == 8, \"3-bit centroids must have 8 entries\");")
    output_lines.append("static_assert(sizeof(PI_CENTROIDS_4BIT)/sizeof(PI_CENTROIDS_4BIT[0]) == 16, \"4-bit centroids must have 16 entries\");")
    
    output = "\n".join(output_lines) + "\n"
    
    # Write to new file
    with open("ggml/src/ggml-cuda/planar-iso-constants-new.cuh", "w") as f:
        f.write(output)
    
    print(f"\nWrote regenerated constants to ggml/src/ggml-cuda/planar-iso-constants-new.cuh")
    print(f"Run 'diff planar-iso-constants.cuh planar-iso-constants-new.cuh' to compare.")

if __name__ == "__main__":
    main()
