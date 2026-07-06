#!/usr/bin/env python3
"""
turbo3_roundtrip.py - Verify turbo3_0 quant/dequant fidelity

Tests:
1. Roundtrip: float32 -> turbo3_0 -> float32, check MSE/SNR
2. Attention dot product: K@Q with turbo3 vs f16, check similarity
3. Norm distribution: check for blowup/small-norm patterns
4. InnerQ interference: check if calibration data exists

If roundtrip MSE is high, the issue is quantization precision.
If roundtrip is fine but attention fails, the issue is in the attention kernel.
"""

import numpy as np
import struct
import sys
from pathlib import Path

# ── turbo3_0 constants (from turbo-quant.cuh) ─────────────────────
TURBO_CENTROIDS_3BIT = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
], dtype=np.float32)

TURBO_MID_3BIT = np.array([
    -0.154259, -0.091775, -0.043589, 0.0,
     0.043589,  0.091775,  0.154259
], dtype=np.float32)

QK_TURBO3 = 32


def nearest_centroid_3bit(val):
    """Python equivalent of turbo_nearest_centroid_3bit."""
    # Binary search using midpoints
    idx = 4  # start at center
    if val < TURBO_MID_3BIT[3]:  # < 0.0
        idx = 2
        if val < TURBO_MID_3BIT[1]:  # < -0.091775
            idx = 0
            if val >= TURBO_MID_3BIT[0]:  # >= -0.154259
                idx = 1
        else:  # >= -0.091775
            idx = 2
            if val >= TURBO_MID_3BIT[2]:  # >= -0.043589
                idx = 3
    else:  # >= 0.0
        idx = 6
        if val < TURBO_MID_3BIT[5]:  # < 0.091775
            idx = 4
            if val >= TURBO_MID_3BIT[4]:  # >= 0.043589
                idx = 5
        else:  # >= 0.091775
            idx = 6
            if val >= TURBO_MID_3BIT[6]:  # >= 0.154259
                idx = 7
    return idx


def quantize_turbo3_block(values):
    """Quantize 32 float values into turbo3_0 block.
    Returns (norm_f16, qs_bytes[8], signs_bytes[4])."""
    assert len(values) == QK_TURBO3

    # L2 norm
    grp_norm = np.sqrt(np.sum(values ** 2))
    if grp_norm < 1e-10:
        return np.float16(0.0), bytes(8), bytes(4)

    # Normalize
    normalized = values / grp_norm

    # Quantize each element
    indices = np.array([nearest_centroid_3bit(v) for v in normalized], dtype=np.uint8)

    # Compute reconstruction norm
    reconstructed = TURBO_CENTROIDS_3BIT[indices]
    recon_norm = np.sqrt(np.sum(reconstructed ** 2))

    # Corrected norm (same as CUDA kernel line 489)
    corrected_norm = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm

    # Pack qs: 4 elements per byte, 2 bits each (low 2 bits of index)
    qs = bytearray(8)
    for i in range(QK_TURBO3):
        byte_idx = i // 4
        shift = (i % 4) * 2
        qs[byte_idx] |= (indices[i] & 0x3) << shift

    # Pack signs: 8 elements per byte, 1 bit each (bit 2 of index)
    signs = bytearray(4)
    for i in range(QK_TURBO3):
        byte_idx = i // 8
        shift = i % 8
        if indices[i] & 0x4:
            signs[byte_idx] |= (1 << shift)

    return np.float16(corrected_norm), bytes(qs), bytes(signs)


def dequantize_turbo3_block(norm_f16, qs_bytes, signs_bytes):
    """Dequantize 32 float values from turbo3_0 block."""
    norm = np.float32(norm_f16)
    result = np.zeros(QK_TURBO3, dtype=np.float32)

    for j in range(QK_TURBO3):
        low2 = (qs_bytes[j // 4] >> ((j % 4) * 2)) & 0x3
        hi1 = (signs_bytes[j // 8] >> (j % 8)) & 0x1
        idx = low2 | (hi1 << 2)
        result[j] = TURBO_CENTROIDS_3BIT[idx] * norm

    return result


def test_roundtrip(values, label=""):
    """Test quantize -> dequantize roundtrip."""
    norm, qs, signs = quantize_turbo3_block(values)
    reconstructed = dequantize_turbo3_block(norm, qs, signs)

    mse = np.mean((values - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    max_err = np.max(np.abs(values - reconstructed))

    signal_power = np.mean(values ** 2)
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # Relative error
    rel_err = np.max(np.abs(values - reconstructed) / (np.abs(values) + 1e-10))

    print(f"  {label}")
    print(f"    RMSE:        {rmse:.6f}")
    print(f"    Max error:   {max_err:.6f}")
    print(f"    Rel error:   {rel_err:.4f}")
    print(f"    SNR:         {snr_db:.1f} dB")
    print(f"    Norm:        {float(norm):.6f}")
    print(f"    Input range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"    Input L2:    {np.sqrt(np.sum(values**2)):.4f}")

    return mse, snr_db


def test_attention_dot(K_values, Q_values, label=""):
    """Test attention dot product: K @ Q with turbo3 vs f16."""
    # Turbo3 path
    norm, qs, signs = quantize_turbo3_block(K_values)
    K_turbo = dequantize_turbo3_block(norm, qs, signs)

    # F16 path (simulate with lower precision)
    K_f16 = K_values.astype(np.float16).astype(np.float32)

    dot_turbo = np.dot(K_turbo, Q_values)
    dot_f16 = np.dot(K_f16, Q_values)

    cos_sim = dot_turbo / (np.linalg.norm(K_turbo) * np.linalg.norm(Q_values) + 1e-10)
    cos_f16 = dot_f16 / (np.linalg.norm(K_f16) * np.linalg.norm(Q_values) + 1e-10)

    print(f"  {label}")
    print(f"    dot turbo3:  {dot_turbo:.6f}")
    print(f"    dot f16:     {dot_f16:.6f}")
    print(f"    dot error:   {abs(dot_turbo - dot_f16):.6f}")
    print(f"    cos sim:     {cos_sim:.6f}")

    return abs(dot_turbo - dot_f16)


def test_norm_distribution():
    """Test with realistic KV cache value distributions."""
    print("\n=== Norm Distribution Test ===")
    print("Testing with N(0, sigma) distributions at different scales:")

    for sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        values = np.random.randn(QK_TURBO3).astype(np.float32) * sigma
        norm, qs, signs = quantize_turbo3_block(values)
        reconstructed = dequantize_turbo3_block(norm, qs, signs)
        mse = np.mean((values - reconstructed) ** 2)
        norm_f = float(norm)
        print(f"  sigma={sigma:.2f}: norm={norm_f:.4f}, MSE={mse:.8f}, "
              f"norm_blowup={'YES' if norm_f > sigma * 10 else 'no'}")


def test_edge_cases():
    """Test edge cases that could cause attention collapse."""
    print("\n=== Edge Case Tests ===")

    # All zeros
    values = np.zeros(QK_TURBO3, dtype=np.float32)
    norm, qs, signs = quantize_turbo3_block(values)
    recon = dequantize_turbo3_block(norm, qs, signs)
    print(f"  All zeros: norm={float(norm):.6f}, recon={recon[:4]}...")

    # All same value (degenerate attention pattern)
    values = np.ones(QK_TURBO3, dtype=np.float32) * 0.1
    norm, qs, signs = quantize_turbo3_block(values)
    recon = dequantize_turbo3_block(norm, qs, signs)
    mse = np.mean((values - recon) ** 2)
    print(f"  All 0.1:   norm={float(norm):.6f}, MSE={mse:.8f}, "
          f"recon range=[{recon.min():.4f}, {recon.max():.4f}]")

    # Single spike (one large value, rest small)
    values = np.zeros(QK_TURBO3, dtype=np.float32)
    values[0] = 10.0
    norm, qs, signs = quantize_turbo3_block(values)
    recon = dequantize_turbo3_block(norm, qs, signs)
    mse = np.mean((values - recon) ** 2)
    print(f"  Spike:     norm={float(norm):.6f}, MSE={mse:.8f}, "
          f"recon[0]={recon[0]:.4f}")

    # Very small values (near denormal)
    values = np.random.randn(QK_TURBO3).astype(np.float32) * 1e-6
    norm, qs, signs = quantize_turbo3_block(values)
    recon = dequantize_turbo3_block(norm, qs, signs)
    print(f"  Tiny:      norm={float(norm):.6f}, input L2={np.sqrt(np.sum(values**2)):.2e}")

    # Large values (potential overflow in norm correction)
    values = np.random.randn(QK_TURBO3).astype(np.float32) * 100
    norm, qs, signs = quantize_turbo3_block(values)
    recon = dequantize_turbo3_block(norm, qs, signs)
    mse = np.mean((values - recon) ** 2)
    print(f"  Large:     norm={float(norm):.4f}, MSE={mse:.4f}, "
          f"recon range=[{recon.min():.1f}, {recon.max():.1f}]")


def test_attention_collapse():
    """Simulate attention with turbo3 quantization to detect collapse."""
    print("\n=== Attention Collapse Test ===")
    print("Simulating multi-token attention with turbo3 KV cache:")

    np.random.seed(42)
    seq_len = 50
    head_dim = QK_TURBO3

    # Generate realistic Q, K, V
    Q = np.random.randn(head_dim).astype(np.float32) * 0.1
    K_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

    # Compute attention with f16 KV
    K_f16 = K_all.astype(np.float16).astype(np.float32)
    scores_f16 = K_f16 @ Q / np.sqrt(head_dim)
    weights_f16 = np.exp(scores_f16 - scores_f16.max())
    weights_f16 /= weights_f16.sum()
    output_f16 = weights_f16 @ V_all

    # Compute attention with turbo3 KV
    K_turbo = np.zeros_like(K_all)
    for i in range(seq_len):
        norm, qs, signs = quantize_turbo3_block(K_all[i])
        K_turbo[i] = dequantize_turbo3_block(norm, qs, signs)

    scores_turbo = K_turbo @ Q / np.sqrt(head_dim)
    weights_turbo = np.exp(scores_turbo - scores_turbo.max())
    weights_turbo /= weights_turbo.sum()
    output_turbo = weights_turbo @ V_all

    # Compare
    cos_sim = np.dot(output_f16, output_turbo) / (np.linalg.norm(output_f16) * np.linalg.norm(output_turbo))

    # Check attention weight entropy (low entropy = focused, high = uniform)
    entropy_f16 = -np.sum(weights_f16 * np.log(weights_f16 + 1e-10))
    entropy_turbo = -np.sum(weights_turbo * np.log(weights_turbo + 1e-10))
    max_entropy = np.log(seq_len)

    print(f"  F16 attention:  entropy={entropy_f16:.3f}/{max_entropy:.3f} ({entropy_f16/max_entropy*100:.0f}%)")
    print(f"  Turbo attention: entropy={entropy_turbo:.3f}/{max_entropy:.3f} ({entropy_turbo/max_entropy*100:.0f}%)")
    print(f"  Output cosine similarity: {cos_sim:.6f}")
    print(f"  Max weight ratio: turbo/f16 = {weights_turbo.max()/weights_f16.max():.4f}")

    if entropy_turbo / max_entropy > 0.95:
        print(f"  WARNING: Attention nearly uniform -> token repetition likely!")
    elif cos_sim < 0.9:
        print(f"  WARNING: Low output similarity -> output quality degraded")


def main():
    np.random.seed(42)

    print("=" * 60)
    print("turbo3_0 Roundtrip Fidelity Test")
    print("=" * 60)

    # Test 1: Random normal values (typical KV distribution)
    print("\n=== Test 1: Random Normal (typical KV values) ===")
    for sigma in [0.05, 0.1, 0.5]:
        values = np.random.randn(QK_TURBO3).astype(np.float32) * sigma
        test_roundtrip(values, f"N(0, {sigma})")

    # Test 2: Realistic attention patterns
    print("\n=== Test 2: Attention-like patterns ===")
    # Softmax-like distribution (one dominant value)
    vals = np.random.randn(QK_TURBO3).astype(np.float32) * 0.1
    vals[0] = 3.0  # dominant
    test_roundtrip(vals, "dominant + noise")

    # Uniform distribution
    vals = np.random.uniform(-0.5, 0.5, QK_TURBO3).astype(np.float32)
    test_roundtrip(vals, "uniform(-0.5, 0.5)")

    # Test 3: Edge cases
    test_edge_cases()

    # Test 4: Norm distribution
    test_norm_distribution()

    # Test 5: Multi-token attention simulation
    test_attention_collapse()

    # Test 6: Bit pattern analysis
    print("\n=== Test 6: Bit Pattern Analysis ===")
    values = np.random.randn(QK_TURBO3).astype(np.float32) * 0.1
    norm, qs, signs = quantize_turbo3_block(values)
    indices = []
    for j in range(QK_TURBO3):
        low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
        hi1 = (signs[j // 8] >> (j % 8)) & 0x1
        indices.append(low2 | (hi1 << 2))
    unique = len(set(indices))
    print(f"  3-bit indices: {indices}")
    print(f"  Unique centroids used: {unique}/8")
    if unique <= 2:
        print(f"  WARNING: Only {unique} centroid(s) used -> attention collapse!")

    print("\n" + "=" * 60)
    print("Done. Check for warnings above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
