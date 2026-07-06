#!/usr/bin/env python3
"""
Compute optimal Lloyd-Max centroids for TurboQuant KV cache compression.

This script samples V-cache vectors from a real model, applies WHT rotation,
computes the actual distribution, and runs Lloyd-Max to find optimal centroids.

Usage:
    python3 scripts/compute_turbo_centroids.py \
        --model /path/to/model.gguf \
        --n_samples 10000 \
        --n_layers 5 \
        --output centroids.json

The script:
1. Loads the model's V projection weights
2. Generates random hidden states and projects through V
3. Applies WHT rotation (same as turbo pipeline)
4. Normalizes each vector to unit norm
5. Runs Lloyd-Max algorithm to find optimal centroids
6. Outputs 3-bit (8) and 4-bit (16) centroid values
7. Compares against current theoretical N(0, 1/128) centroids
"""

import argparse
import json
import numpy as np
from pathlib import Path


def fwht_inplace(a):
    """Fast Walsh-Hadamard Transform in-place (Hadamard order)."""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def fwht(x):
    """Apply WHT to vector x (returns new array)."""
    y = x.copy()
    fwht_inplace(y)
    y /= np.sqrt(len(y))  # normalize to preserve L2 norm
    return y


def lloyd_max_1d(data, n_centroids, max_iter=100, tol=1e-6):
    """
    Lloyd-Max algorithm for 1D optimal scalar quantization.
    Returns (centroids, midpoints, mse).
    """
    # Initialize centroids using quantiles (n_centroids bins -> n_centroids-1 boundaries)
    # Place centroids at the centers of equal-probability bins
    bin_edges = np.linspace(0, 100, n_centroids + 1)
    centroids = np.array([
        np.percentile(data, (bin_edges[i] + bin_edges[i+1]) / 2)
        for i in range(n_centroids)
    ])

    for iteration in range(max_iter):
        # Assignment step: find nearest centroid for each data point
        # Using midpoints for fast assignment
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0

        # Assign each point to nearest centroid
        indices = np.searchsorted(midpoints, data)
        indices = np.clip(indices, 0, n_centroids - 1)

        # Update step: compute new centroids as conditional means
        new_centroids = np.zeros(n_centroids)
        counts = np.zeros(n_centroids)
        for i in range(n_centroids):
            mask = indices == i
            if np.any(mask):
                new_centroids[i] = np.mean(data[mask])
                counts[i] = np.sum(mask)

        # Handle empty clusters (keep old centroid)
        for i in range(n_centroids):
            if counts[i] == 0:
                new_centroids[i] = centroids[i]

        # Check convergence
        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if delta < tol:
            break

    # Compute final midpoints and MSE
    midpoints = np.zeros(n_centroids - 1)
    for i in range(n_centroids - 1):
        midpoints[i] = (centroids[i] + centroids[i + 1]) / 2.0

    # Compute MSE
    indices = np.searchsorted(midpoints, data)
    indices = np.clip(indices, 0, n_centroids - 1)
    reconstructed = centroids[indices]
    mse = np.mean((data - reconstructed) ** 2)

    return centroids, midpoints, mse


def load_gguf_weights(model_path):
    """
    Load V projection weights from GGUF model.
    This is a simplified version - in practice, use gguf library.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(str(model_path), 'r')

        # Find V projection weights
        v_weights = {}
        for tensor in reader.tensors:
            name = tensor.name
            if 'attn_v' in name and 'weight' in name:
                v_weights[name] = tensor.data.astype(np.float32)

        return v_weights
    except ImportError:
        print("Warning: gguf library not found. Using random weights for demonstration.")
        return None


def simulate_v_cache(v_weights, n_samples, head_dim=128):
    """
    Simulate V-cache by projecting random hidden states through V weights.
    Returns vectors of shape (n_samples, head_dim).
    """
    if v_weights is None:
        # Generate random V-cache-like vectors
        # Typical V-cache: unit-norm vectors with some structure
        print("Using random vectors (install gguf library for real data)")
        vectors = np.random.randn(n_samples, head_dim).astype(np.float32)
        # Normalize to unit norm (as done in turbo pipeline)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        return vectors

    # Use real V weights
    all_vectors = []
    for name, weight in v_weights.items():
        # weight shape: (n_heads * head_dim, hidden_dim) or similar
        if weight.ndim == 2:
            # Generate random hidden states and project
            hidden_dim = weight.shape[1]
            n_heads = weight.shape[0] // head_dim

            for _ in range(n_samples // (n_heads * len(v_weights)) + 1):
                h = np.random.randn(hidden_dim).astype(np.float32)
                v = weight @ h  # project through V
                # Split into heads
                v = v.reshape(n_heads, head_dim)
                # Normalize each head vector
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                v = v / (norms + 1e-10)
                all_vectors.append(v)

    if all_vectors:
        all_vectors = np.concatenate(all_vectors, axis=0)
        # Take requested number of samples
        if len(all_vectors) > n_samples:
            idx = np.random.choice(len(all_vectors), n_samples, replace=False)
            all_vectors = all_vectors[idx]
        return all_vectors

    # Fallback
    vectors = np.random.randn(n_samples, head_dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    return vectors


def compute_centroids_from_data(vectors, d=128):
    """
    Compute optimal centroids from real V-cache data.
    1. Apply WHT rotation
    2. Compute distribution statistics
    3. Run Lloyd-Max for 3-bit and 4-bit
    """
    print(f"Input vectors: {vectors.shape}")

    # Apply WHT to each vector
    rotated = np.zeros_like(vectors)
    for i in range(len(vectors)):
        rotated[i] = fwht(vectors[i])

    # Flatten all rotated coordinates
    all_coords = rotated.flatten()

    print(f"Distribution of rotated coordinates:")
    print(f"  Mean: {np.mean(all_coords):.6f}")
    print(f"  Std:  {np.std(all_coords):.6f}")
    print(f"  Min:  {np.min(all_coords):.6f}")
    print(f"  Max:  {np.max(all_coords):.6f}")
    print(f"  Theoretical std for N(0,1/{d}): {1.0/np.sqrt(d):.6f}")

    # Compute per-coordinate variance (should be ~1/d for unit-norm input)
    coord_var = np.var(rotated, axis=0)
    print(f"\nPer-coordinate variance:")
    print(f"  Mean: {np.mean(coord_var):.6f}")
    print(f"  Std:  {np.std(coord_var):.6f}")
    print(f"  Min:  {np.min(coord_var):.6f}")
    print(f"  Max:  {np.max(coord_var):.6f}")

    # Run Lloyd-Max for 3-bit (8 centroids)
    centroids_3bit, midpoints_3bit, mse_3bit = lloyd_max_1d(all_coords, 8)
    print(f"\n3-bit centroids (8 levels):")
    print(f"  Centroids: {centroids_3bit}")
    print(f"  MSE: {mse_3bit:.8f}")

    # Run Lloyd-Max for 4-bit (16 centroids)
    centroids_4bit, midpoints_4bit, mse_4bit = lloyd_max_1d(all_coords, 16)
    print(f"\n4-bit centroids (16 levels):")
    print(f"  Centroids: {centroids_4bit}")
    print(f"  MSE: {mse_4bit:.8f}")

    return {
        'centroids_3bit': centroids_3bit.tolist(),
        'midpoints_3bit': midpoints_3bit.tolist(),
        'centroids_4bit': centroids_4bit.tolist(),
        'midpoints_4bit': midpoints_4bit.tolist(),
        'mse_3bit': float(mse_3bit),
        'mse_4bit': float(mse_4bit),
        'distribution': {
            'mean': float(np.mean(all_coords)),
            'std': float(np.std(all_coords)),
            'theoretical_std': 1.0 / np.sqrt(d),
        }
    }


def compare_with_current(new_centroids):
    """Compare new centroids with current theoretical values."""
    # Current centroids (from turbo-quant.cuh)
    current_3bit = np.array([
        -0.190685, -0.117832, -0.065717, -0.021460,
         0.021460,  0.065717,  0.117832,  0.190685
    ])
    current_4bit = np.array([
        -0.173926, -0.117195, -0.089527, -0.068756,
        -0.051262, -0.035597, -0.020989, -0.006938,
         0.006938,  0.020989,  0.035597,  0.051262,
         0.068756,  0.089527,  0.117195,  0.173926
    ])

    new_3bit = np.array(new_centroids['centroids_3bit'])
    new_4bit = np.array(new_centroids['centroids_4bit'])

    print("\n=== Comparison with Current Centroids ===")
    print(f"\n3-bit centroids:")
    print(f"  Current: {current_3bit}")
    print(f"  New:     {new_3bit}")
    print(f"  Max diff: {np.max(np.abs(new_3bit - current_3bit)):.6f}")
    print(f"  Rel diff: {np.max(np.abs(new_3bit - current_3bit) / (np.abs(current_3bit) + 1e-10)) * 100:.2f}%")

    print(f"\n4-bit centroids:")
    print(f"  Current: {current_4bit}")
    print(f"  New:     {new_4bit}")
    print(f"  Max diff: {np.max(np.abs(new_4bit - current_4bit)):.6f}")
    print(f"  Rel diff: {np.max(np.abs(new_4bit - current_4bit) / (np.abs(current_4bit) + 1e-10)) * 100:.2f}%")


def generate_c_code(centroids_3bit, centroids_4bit, midpoints_3bit, midpoints_4bit):
    """Generate C code for updated centroid arrays."""
    def format_array(name, values, fmt="{:.6f}f"):
        vals = ", ".join(fmt.format(v) for v in values)
        return f"    {name}[{len(values)}] = {{\n        {vals}\n    }};"

    code = f"""
// =====================================================
// Auto-generated centroids from real V-cache data
// Run: python3 scripts/compute_turbo_centroids.py
// =====================================================

// ---- 3-bit centroids (Lloyd-Max for actual V-cache distribution) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {{
    {', '.join(f'{v:.6f}f' for v in centroids_3bit)}
}};

// ---- Midpoints for nearest 3-bit centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {{
    {', '.join(f'{v:.6f}f' for v in midpoints_3bit)}
}};

// ---- 4-bit centroids (Lloyd-Max for actual V-cache distribution) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {{
    {', '.join(f'{v:.6f}f' for v in centroids_4bit)}
}};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {{
    {', '.join(f'{v:.6f}f' for v in midpoints_4bit)}
}};
"""
    return code


def main():
    parser = argparse.ArgumentParser(description="Compute optimal TurboQuant centroids from real V-cache data")
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of V-cache vectors to sample")
    parser.add_argument("--n_layers", type=int, default=5, help="Number of layers to sample from")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--output", type=str, default="centroids.json", help="Output JSON file")
    parser.add_argument("--c_code", action="store_true", help="Generate C code for CUDA/host arrays")
    args = parser.parse_args()

    print("=" * 60)
    print("TurboQuant Centroid Recomputation")
    print("=" * 60)

    # Load model weights
    if args.model:
        print(f"\nLoading model: {args.model}")
        v_weights = load_gguf_weights(args.model)
    else:
        print("\nNo model specified, using random vectors")
        v_weights = None

    # Simulate V-cache
    print(f"\nSimulating V-cache ({args.n_samples} vectors, head_dim={args.head_dim})")
    vectors = simulate_v_cache(v_weights, args.n_samples, args.head_dim)

    # Compute centroids
    print(f"\nComputing optimal centroids...")
    result = compute_centroids_from_data(vectors, args.head_dim)

    # Compare with current
    compare_with_current(result)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate C code if requested
    if args.c_code:
        c_code = generate_c_code(
            result['centroids_3bit'],
            result['centroids_4bit'],
            result['midpoints_3bit'],
            result['midpoints_4bit']
        )
        c_path = output_path.with_suffix('.cuh')
        with open(c_path, 'w') as f:
            f.write(c_code)
        print(f"C code saved to: {c_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
