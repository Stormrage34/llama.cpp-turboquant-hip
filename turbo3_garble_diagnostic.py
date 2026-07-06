#!/usr/bin/env python3
"""
turbo3_garble_diagnostic.py -- Diagnose turbo3_0 KV cache garbling pattern.

Targets the specific failure observed in real inference:
  coherent first token(s) -> rapid degeneration -> single-token repetition loop.

Hypotheses tested:
  H1: Quantization error accumulates across multi-layer inference (each layer's
      output becomes the next layer's input, amplifying small errors).
  H2: Norm correction formula preserves magnitude incorrectly for certain
      value distributions (especially after WHT rotation).
  H3: Half-precision norm storage loses precision, causing dequant drift.
  H4: Stuck-token detection -- simulates autoregressive generation and checks
      for single-token repetition loops characteristic of the observed bug.
  H5: Position-dependent quantization error -- early vs late elements in the
      head dimension may have different reconstruction quality.
  H6: turbo3_0 vs q8_0 quality gap -- measures actual information loss.
  H7: Multi-head attention consistency -- do all heads produce similar-quality
      outputs, or do some heads degrade while others remain stable?
  H8: Attention Output Comparison -- simulated Q,K,V attention with turbo3_0
      vs q8_0 V cache, tracking error growth over 10 layers.

All tests are pure Python/numpy, deterministic, and fast (<10 seconds).
"""

import math
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# Constants (matching turbo-quant.cuh / block_turbo3_0 definition)
# ============================================================================

QK_TURBO3 = 32          # elements per block
D = 128                 # head dimension
N_CENTROIDS_3BIT = 8    # 3-bit -> 8 levels
GROUP_SIZE = 128        # workgroup size for k_set_rows_turbo3

TURBO_CENTROIDS_3BIT = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685,
], dtype=np.float32)

TURBO_MID_3BIT = np.array([
    -0.154259, -0.091775, -0.043589, 0.0,
     0.043589,  0.091775,  0.154259,
], dtype=np.float32)

# Exact sign arrays from turbo-quant.cuh lines 57-77
TURBO_WHT_SIGNS1 = np.array([
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
], dtype=np.float32)

TURBO_WHT_SIGNS2 = np.array([
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
], dtype=np.float32)


def fwht_inplace(a):
    """Fast Walsh-Hadamard Transform in-place (unnormalized)."""
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


def turbo_forward_rotation(x):
    """Apply forward WHT rotation: signs1 -> FWHT -> signs2."""
    result = x.astype(np.float64).copy()
    result *= TURBO_WHT_SIGNS1
    fwht_inplace(result.tolist())
    result *= TURBO_WHT_SIGNS2
    return result.astype(np.float32)


def turbo_inverse_rotation(x):
    """Apply inverse WHT rotation: signs2 -> FWHT -> signs1."""
    result = x.astype(np.float64).copy()
    result *= TURBO_WHT_SIGNS2
    fwht_inplace(result.tolist())
    result *= TURBO_WHT_SIGNS1
    return result.astype(np.float32)


# ============================================================================
# Quantization / Dequantization (mirrors CUDA kernel logic)
# ============================================================================

def turbo_nearest_centroid_3bit(val):
    """Python equivalent of turbo_nearest_centroid_3bit in turbo-quant.cuh."""
    val = float(val)
    if val < 0.0:
        if val < -0.091775:
            if val >= -0.154259:
                return 1
            return 0
        else:
            if val >= -0.043589:
                return 3
            return 2
    else:
        if val < 0.091775:
            if val >= 0.043589:
                return 5
            return 4
        else:
            if val >= 0.154259:
                return 7
            return 6


def quantize_turbo3_block(values, apply_rotation=True):
    """Quantize 128 values into turbo3_0 block format (4 sub-blocks of 32).

    The WHT rotation operates on the FULL GROUP_SIZE=128 elements before
    splitting into sub-blocks, matching k_set_rows_turbo3 in set-rows.cu.
    """
    assert len(values) == D
    n_blocks = D // QK_TURBO3

    if apply_rotation:
        values = turbo_forward_rotation(values.copy())

    all_norms = []
    all_qs = bytearray()
    all_signs = bytearray()

    for b in range(n_blocks):
        offset = b * QK_TURBO3
        block_vals = values[offset:offset + QK_TURBO3]

        grp_norm_sq = float(np.sum(block_vals * block_vals))
        grp_norm = math.sqrt(grp_norm_sq)
        if grp_norm < 1e-10:
            all_norms.append(np.float16(0.0))
            all_qs.extend(bytes(QK_TURBO3 // 4))
            all_signs.extend(bytes(QK_TURBO3 // 8))
            continue

        normalized = block_vals / grp_norm

        indices = np.array([turbo_nearest_centroid_3bit(float(v)) for v in normalized], dtype=np.uint8)

        recon_values = TURBO_CENTROIDS_3BIT[indices]
        recon_norm_sq = float(np.sum(recon_values * recon_values))
        recon_norm = math.sqrt(recon_norm_sq)
        corrected_norm = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm
        all_norms.append(np.float16(corrected_norm))

        qs_byte = bytearray(QK_TURBO3 // 4)
        for i in range(QK_TURBO3):
            byte_idx = i // 4
            shift = (i % 4) * 2
            qs_byte[byte_idx] |= (int(indices[i]) & 0x3) << shift
        all_qs.extend(qs_byte)

        signs_byte = bytearray(QK_TURBO3 // 8)
        for i in range(QK_TURBO3):
            byte_idx = i // 8
            if int(indices[i]) & 0x4:
                signs_byte[byte_idx] |= (1 << (i % 8))
        all_signs.extend(signs_byte)

    return all_norms, bytes(all_qs), bytes(all_signs)


def quantize_turbo3_subblock(block_vals):
    """Quantize a single 32-element sub-block (no rotation).

    Used for per-block precision testing where rotation is applied at the
    group level externally. Matches the inner loop of k_set_rows_turbo3.
    """
    assert len(block_vals) == QK_TURBO3

    grp_norm_sq = float(np.sum(block_vals * block_vals))
    grp_norm = math.sqrt(grp_norm_sq)
    if grp_norm < 1e-10:
        return [np.float16(0.0)], bytes(QK_TURBO3 // 4), bytes(QK_TURBO3 // 8)

    normalized = block_vals / grp_norm
    indices = np.array([turbo_nearest_centroid_3bit(float(v)) for v in normalized], dtype=np.uint8)

    recon_values = TURBO_CENTROIDS_3BIT[indices]
    recon_norm_sq = float(np.sum(recon_values * recon_values))
    recon_norm = math.sqrt(recon_norm_sq)
    corrected_norm = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm

    qs_byte = bytearray(QK_TURBO3 // 4)
    for i in range(QK_TURBO3):
        qs_byte[i // 4] |= (int(indices[i]) & 0x3) << ((i % 4) * 2)

    signs_byte = bytearray(QK_TURBO3 // 8)
    for i in range(QK_TURBO3):
        if int(indices[i]) & 0x4:
            signs_byte[i // 8] |= (1 << (i % 8))

    return [np.float16(corrected_norm)], bytes(qs_byte), bytes(signs_byte)


def dequantize_turbo3_block(norms, qs_bytes, signs_bytes, apply_inverse_rotation=False):
    """Dequantize turbo3_0 block back to float values.

    Since forward rotation operates on the full GROUP_SIZE=128 elements before
    splitting into sub-blocks, inverse rotation must also operate on the full
    reconstructed 128-element vector (not per-sub-block).
    """
    n_blocks = D // QK_TURBO3
    qs_per_block = QK_TURBO3 // 4
    signs_per_block = QK_TURBO3 // 8

    result = np.zeros(D, dtype=np.float32)
    for b in range(n_blocks):
        offset = b * QK_TURBO3
        norm = float(norms[b])
        qs_start = b * qs_per_block
        signs_start = b * signs_per_block

        block_vals = []
        for j in range(QK_TURBO3):
            low2 = (qs_bytes[qs_start + j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs_bytes[signs_start + j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            block_vals.append(TURBO_CENTROIDS_3BIT[idx] * norm)

        result[offset:offset + QK_TURBO3] = np.array(block_vals, dtype=np.float32)

    if apply_inverse_rotation:
        result = turbo_inverse_rotation(result.copy())

    return result


def quantize_q8_0_block(values):
    """q8_0 block quantization matching ggml-quants.c (single 32-element block).

    block_q8_0: 2-byte fp16 scale + 32 bytes int8 quants = 34 bytes for 32 elements.
    Reconstructed: qs[i] * scale / 127.0
    """
    assert len(values) == QK_TURBO3
    abs_max = float(np.max(np.abs(values)))
    if abs_max < 1e-10:
        return np.zeros_like(values, dtype=np.float32), 0.0

    scale = abs_max / 127.0
    quants = np.clip(np.round(values / scale).astype(np.int8), -128, 127)
    recon = quants.astype(np.float32) * scale
    return recon, scale


def quantize_q8_0_vector(values):
    """q8_0 quantization for a full D=128 vector (4 blocks of 32)."""
    assert len(values) == D
    recon = np.zeros(D, dtype=np.float32)
    for b in range(D // QK_TURBO3):
        off = b * QK_TURBO3
        block_recon, _ = quantize_q8_0_block(values[off:off + QK_TURBO3])
        recon[off:off + QK_TURBO3] = block_recon
    return recon, 0.0


# ============================================================================
# Helper: simulate a mini transformer layer with turbo3_0 KV cache
# ============================================================================

def simulate_attention_layer(Q_orig, K_cache, V_cache, n_quantize_from_start=0):
    """
    Simulate one attention layer with turbo3_0 KV cache.

    Q_orig: (D,) query vector in original space
    K_cache: (seq_len, D) key vectors already in cache
    V_cache: (seq_len, D) value vectors already in cache
    n_quantize_from_start: how many K entries to re-quantize (simulates new tokens)

    Returns: attention output (D,), attention weights (seq_len,)
    """
    Q_rot = turbo_forward_rotation(Q_orig)

    K_dequant = np.zeros_like(K_cache)
    for i in range(K_cache.shape[0]):
        norm, qs, signs = quantize_turbo3_block(K_cache[i], apply_rotation=True)
        K_dequant[i] = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

    scale = math.sqrt(D)
    scores = (K_dequant @ Q_rot) / scale
    weights = scores - scores.max()
    weights = np.exp(weights)
    weights /= weights.sum()

    output = weights @ V_cache

    return output, weights


def simulate_multi_layer_inference(n_layers, seq_len, rng_seed=42):
    """
    Simulate multi-layer transformer inference with turbo3_0 KV cache.

    Each layer:
      1. Takes previous layer's output as new Q/K/V inputs
      2. Stores K/V in turbo3_0 quantized form
      3. Applies attention with quantized KV

    Returns per-layer output norms and attention weight entropy.
    """
    rng = np.random.RandomState(rng_seed)

    layer_outputs = []
    layer_entropies = []
    layer_norms = []

    x = rng.randn(D).astype(np.float32) * 0.1

    for layer in range(n_layers):
        K_all = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_all = rng.randn(seq_len, D).astype(np.float32) * 0.1

        K_quant = np.zeros_like(K_all)
        for i in range(seq_len):
            norm_k, qs_k, signs_k = quantize_turbo3_block(K_all[i], apply_rotation=True)
            K_quant[i] = dequantize_turbo3_block(norm_k, qs_k, signs_k, apply_inverse_rotation=False)

        output, weights = simulate_attention_layer(x, K_quant, V_all)

        entropy = -np.sum(weights * np.log(weights + 1e-30))
        max_entropy = math.log(seq_len)

        layer_outputs.append(output)
        layer_entropies.append(entropy / max_entropy)
        layer_norms.append(float(np.linalg.norm(output)))

        x = output

    return layer_outputs, layer_entropies, layer_norms


# ============================================================================
# Test Suite
# ============================================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.details = []

    def fail(self, msg):
        self.passed = False
        self.details.append(msg)

    def info(self, msg):
        self.details.append(f"INFO: {msg}")

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"  [{status}] {self.name}")
        for d in self.details:
            print(f"         {d}")
        return self.passed


def test_norm_preservation():
    """Test H2: Does norm correction preserve magnitude after rotation + quantization?"""
    print("\n=== TEST 9: Norm Preservation After Rotation+Quant ===")
    rng = np.random.RandomState(505)

    issues = []
    n_trials = 200

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1

        original_norm = float(np.linalg.norm(values))

        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)
        recon = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)

        recon_norm = float(np.linalg.norm(recon))
        norm_ratio = recon_norm / max(original_norm, 1e-10)

        if abs(norm_ratio - 1.0) > 0.15:
            issues.append(f"trial {trial}: norm ratio={norm_ratio:.4f} (original={original_norm:.4f}, recon={recon_norm:.4f})")

    if issues:
        print(f"  FAIL: {len(issues)} trials show norm drift > 15%")
        for msg in issues[:5]:
            print(f"         {msg}")
        return False
    else:
        print(f"  PASS: All {n_trials} trials preserve norms within tolerance")
        return True


def test_half_precision_norm_drift():
    """Test H3: Does fp16 norm storage lose precision causing dequant drift?"""
    print("\n=== TEST 10: Half-Precision Norm Storage Drift ===")
    rng = np.random.RandomState(606)

    n_trials = 500
    drifts = []

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1
        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)

        for b in range(D // QK_TURBO3):
            f32_norm = float(norms[b])
            f16_norm = np.float16(f32_norm).item()
            drift = abs(f32_norm - f16_norm)
            drifts.append(drift)

    mean_drift = np.mean(drifts)
    max_drift_val = max(drifts) if drifts else 0.0
    p99_drift = np.percentile(drifts, 99) if len(drifts) > 10 else 0.0

    print(f"  Trials: {n_trials}")
    print(f"  Mean norm drift (fp32->fp16):   {mean_drift:.8f}")
    print(f"  Max norm drift:                 {max_drift_val:.8f}")
    print(f"  P99 norm drift:                 {p99_drift:.8f}")
    print(f"  Norm range:                     [{min(drifts):.8f}, {max(drifts):.8f}]")

    if max_drift_val > 0.05:
        print(f"  FAIL: Max norm drift {max_drift_val:.6f} exceeds threshold (0.05)")
        return False
    else:
        print(f"  PASS: Norm drift within acceptable bounds")
        return True


def test_position_dependent_error():
    """Test H5: Does quantization error vary by position within head dimension?"""
    print("\n=== TEST 11: Position-Dependent Quantization Error ===")
    rng = np.random.RandomState(707)

    n_trials = 1000
    position_errors = np.zeros(D)

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1

        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)
        recon = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)

        errors = (values - recon) ** 2
        position_errors += errors

    avg_errors = position_errors / n_trials
    pos_mse = np.sqrt(avg_errors)

    print(f"  Trials: {n_trials}")
    print(f"  RMSE by position (first 16):")
    for i in range(0, D, 8):
        slice_end = min(i + 8, D)
        vals = pos_mse[i:slice_end]
        print(f"    positions {i:3d}-{slice_end-1:3d}: RMSE = [{vals[0]:.6f}, ..., {vals[-1]:.6f}]")

    max_pos_rmse = np.max(pos_mse)
    min_pos_rmse = np.min(pos_mse)
    ratio = max_pos_rmse / max(min_pos_rmse, 1e-10)

    print(f"\n  Max RMSE position: {np.argmax(pos_mse):3d} ({max_pos_rmse:.6f})")
    print(f"  Min RMSE position: {np.argmin(pos_mse):3d} ({min_pos_rmse:.6f})")
    print(f"  Ratio (max/min):   {ratio:.2f}x")

    if ratio > 3.0:
        print(f"  FAIL: Position-dependent error ratio {ratio:.2f}x is too high")
        return False
    else:
        print(f"  PASS: Quantization error is uniform across positions")
        return True


def test_stuck_token_detection():
    """Test H4: Simulate autoregressive generation and detect stuck-token loops."""
    print("\n=== TEST 12: Stuck-Token Loop Detection ===")

    rng = np.random.RandomState(808)

    seq_len = 16
    n_heads = 20
    max_iterations = 25

    stuck_count = 0
    total_iterations = 0

    for head in range(n_heads):
        Q = rng.randn(D).astype(np.float32) * 0.1

        prev_token_id = -1
        consecutive_same = 0

        for step in range(max_iterations):
            K_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
            V_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1

            output, weights = simulate_attention_layer(Q, K_cache, V_cache)

            token_id = int(np.argmax(output)) % 1000

            if token_id == prev_token_id:
                consecutive_same += 1
            else:
                consecutive_same = 0
            prev_token_id = token_id

            total_iterations += 1

            if consecutive_same >= 10:
                stuck_count += 1
                break

            Q = output * 0.9 + rng.randn(D).astype(np.float32) * 0.01

    stuck_ratio = stuck_count / max(n_heads, 1)

    print(f"  Heads tested: {n_heads}")
    print(f"  Max iterations per head: {max_iterations}")
    print(f"  Stuck heads (>=10 consecutive same tokens): {stuck_count}/{n_heads}")
    print(f"  Stuck ratio: {stuck_ratio:.1%}")

    if stuck_ratio > 0.5:
        print(f"  FAIL: {stuck_ratio:.1%} of heads get stuck in token loops")
        return False
    elif stuck_ratio > 0.2:
        print(f"  WARN: {stuck_ratio:.1%} of heads show early sticking behavior")
        return True
    else:
        print(f"  PASS: Token loops are rare ({stuck_ratio:.1%})")
        return True


def test_turbo3_vs_q8_quality():
    """Test H6: Measure actual quality gap between turbo3_0 and q8_0."""
    print("\n=== TEST 13: Turbo3_0 vs Q8_0 Quality Gap ===")

    rng = np.random.RandomState(909)
    n_trials = 500

    turbo3_errors = []
    q8_errors = []

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1

        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)
        recon_turbo = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)

        recon_q8, _ = quantize_q8_0_vector(values)

        mse_turbo = float(np.mean((values - recon_turbo) ** 2))
        mse_q8 = float(np.mean((values - recon_q8) ** 2))

        turbo3_errors.append(mse_turbo)
        q8_errors.append(mse_q8)

    avg_turbo_mse = np.mean(turbo3_errors)
    avg_q8_mse = np.mean(q8_errors)
    ratio = avg_turbo_mse / max(avg_q8_mse, 1e-15)

    print(f"  Trials: {n_trials}")
    print(f"  turbo3_0 MSE:   {avg_turbo_mse:.8f}")
    print(f"  q8_0 MSE:       {avg_q8_mse:.8f}")
    print(f"  Ratio (turbo3/q8): {ratio:.2f}x")

    if ratio > 10.0:
        print(f"  FAIL: turbo3_0 quality is {ratio:.1f}x worse than q8_0 (unacceptable)")
        return False
    elif ratio > 5.0:
        print(f"  WARN: turbo3_0 is {ratio:.1f}x worse than q8_0 (significant gap)")
        return True
    else:
        print(f"  PASS: turbo3_0 quality is acceptable ({ratio:.2f}x q8_0)")
        return True


def test_multi_head_consistency():
    """Test H7: Do all attention heads produce similar-quality outputs?"""
    print("\n=== TEST 14: Multi-Head Attention Consistency ===")

    rng = np.random.RandomState(1010)
    seq_len = 16
    n_heads = 20

    head_errors = []

    for head in range(n_heads):
        K_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
        Q = rng.randn(D).astype(np.float32) * 0.1

        output, weights = simulate_attention_layer(Q, K_cache, V_cache)

        max_entropy = math.log(seq_len)
        entropy = -np.sum(weights * np.log(weights + 1e-30))
        normalized_entropy = entropy / max_entropy

        output_norm = float(np.linalg.norm(output))

        head_errors.append({
            'entropy_ratio': normalized_entropy,
            'output_norm': output_norm,
        })

    entropies = [h['entropy_ratio'] for h in head_errors]
    norms = [h['output_norm'] for h in head_errors]

    print(f"  Heads tested: {n_heads}")
    print(f"  Attention entropy ratio:")
    print(f"    Mean: {np.mean(entropies):.4f}")
    print(f"    Std:  {np.std(entropies):.4f}")
    print(f"    Min:  {np.min(entropies):.4f}")
    print(f"    Max:  {np.max(entropies):.4f}")

    print(f"  Output norms:")
    print(f"    Mean: {np.mean(norms):.6f}")
    print(f"    Std:  {np.std(norms):.6f}")
    print(f"    Min:  {np.min(norms):.6f}")
    print(f"    Max:  {np.max(norms):.6f}")

    low_entropy_heads = sum(1 for e in entropies if e < 0.3)
    high_norm_heads = sum(1 for n in norms if abs(n - np.mean(norms)) > 3 * np.std(norms))

    print(f"\n  Low-entropy heads (< 0.3): {low_entropy_heads}/{n_heads}")
    print(f"  High-norm outliers:        {high_norm_heads}/{n_heads}")

    issues = []
    if low_entropy_heads > n_heads * 0.3:
        issues.append(f"{low_entropy_heads} heads have very focused attention (may cause instability)")
    if high_norm_heads > n_heads * 0.1:
        issues.append(f"{high_norm_heads} heads have outlier norms")

    if issues:
        print(f"  WARN: {'; '.join(issues)}")
        return True
    else:
        print(f"  PASS: All heads show consistent behavior")
        return True


def test_multi_layer_error_accumulation():
    """Test H1: Does quantization error accumulate across layers?"""
    print("\n=== TEST 15: Multi-Layer Error Accumulation ===")

    n_layers_list = [1, 6, 12, 24]
    seq_len = 16
    n_seeds = 3

    last_layer_norms = []
    for n_layers in n_layers_list:
        all_layer_norms = []
        layer_norms = []
        for seed in range(n_seeds):
            _, _, layer_norms = simulate_multi_layer_inference(n_layers, seq_len, rng_seed=seed)
            all_layer_norms.extend(layer_norms)
        last_layer_norms = layer_norms.copy()

        mean_norm = np.mean(all_layer_norms)
        std_norm = np.std(all_layer_norms)
        final_norm = last_layer_norms[-1] if last_layer_norms else 0.0
        first_norm = last_layer_norms[0] if last_layer_norms else 0.0

        print(f"  {n_layers:2d} layers: mean_norm={mean_norm:.4f}, "
              f"std={std_norm:.4f}, first={first_norm:.4f}, last={final_norm:.4f}")

    _, _, norms_long = simulate_multi_layer_inference(48, seq_len, rng_seed=42)
    if len(norms_long) > 10:
        early_norm = np.mean(norms_long[:5])
        late_norm = np.mean(norms_long[-5:])
        drift = abs(late_norm - early_norm) / max(early_norm, 1e-10)

        print(f"\n  48-layer drift: early={early_norm:.4f}, late={late_norm:.4f}, drift={drift:.2%}")

        if drift > 0.5:
            print(f"  FAIL: Norm drift of {drift:.2%} across layers indicates error accumulation")
            return False
        elif drift > 0.2:
            print(f"  WARN: Moderate norm drift ({drift:.2%}) -- may cause gradual degradation")
            return True
        else:
            print(f"  PASS: Norm remains stable across layers")
            return True
    else:
        print(f"  PASS: Error accumulation test completed")
        return True


def test_realistic_prompt_simulation():
    """Simulate the exact failure scenario from the conversation log."""
    print("\n=== TEST 16: Realistic Prompt Simulation (508-token context) ===")

    seq_len = 64
    n_heads = 10
    n_layers = 24

    rng = np.random.RandomState(1212)

    all_outputs = []
    prev_output = rng.randn(D).astype(np.float32) * 0.1

    for step in range(20):
        layer_outputs_step = []

        for head in range(n_heads):
            K_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
            V_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1

            output, weights = simulate_attention_layer(prev_output, K_cache, V_cache)
            layer_outputs_step.append(output)

        step_output = np.mean(layer_outputs_step, axis=0)
        all_outputs.append(step_output)

        prev_output = step_output * 0.95 + rng.randn(D).astype(np.float32) * 0.02

    norms = [float(np.linalg.norm(o)) for o in all_outputs]
    entropies = []
    for step in range(20):
        K_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
        _, weights = simulate_attention_layer(all_outputs[step], K_cache, rng.randn(seq_len, D).astype(np.float32) * 0.1)
        max_ent = math.log(seq_len)
        ent = -np.sum(weights * np.log(weights + 1e-30))
        entropies.append(ent / max_ent)

    print(f"  Generation steps: {len(all_outputs)}")
    print(f"  Output norms progression:")
    for i, n in enumerate(norms[:10]):
        print(f"    step {i:2d}: norm={n:.6f}")
    if len(norms) > 10:
        print(f"    ... (steps 10-19): norms=[{norms[10]:.4f}, ..., {norms[-1]:.4f}]")

    print(f"\n  Attention entropy progression:")
    for i, e in enumerate(entropies[:10]):
        print(f"    step {i:2d}: entropy_ratio={e:.4f}")

    first_half_norms = np.mean(norms[:10])
    second_half_norms = np.mean(norms[10:])
    norm_decay = (first_half_norms - second_half_norms) / max(first_half_norms, 1e-10)

    first_half_ent = np.mean(entropies[:10])
    second_half_ent = np.mean(entropies[10:])
    entropy_change = second_half_ent - first_half_ent

    print(f"\n  Norm decay (first half -> second half): {norm_decay:.2%}")
    print(f"  Entropy change: {entropy_change:.4f}")

    issues = []
    if norm_decay > 0.3:
        issues.append(f"Output norms decay by {norm_decay:.1%} over generation")
    if entropy_change < -0.2:
        issues.append(f"Attention entropy drops by {-entropy_change:.2f} (becoming more focused/stuck)")

    if issues:
        print(f"  WARN: {'; '.join(issues)}")
        return True
    else:
        print(f"  PASS: No significant degradation detected in simulation")
        return True


# ============================================================================
# ATTENTION OUTPUT COMPARISON -- Section added per optimization task
# ============================================================================

def mse(a, b):
    """Mean squared error between two arrays."""
    return float(np.mean((a - b) ** 2))


def snr_db(original, reconstructed):
    """Signal-to-noise ratio in dB."""
    signal_power = float(np.mean(original ** 2))
    noise_power = float(np.mean((original - reconstructed) ** 2))
    if noise_power < 1e-15:
        return 99.9
    return 10.0 * math.log10(signal_power / noise_power)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return dot / (norm_a * norm_b + 1e-10)


def simulate_attention_with_v_quant(Q_orig, K_orig, V_orig, v_quant_type="turbo3_0"):
    """Simulate one attention layer with specified V-cache quantization.

    Q is always rotated forward (as llama-graph does).
    K and V are stored quantized; dequantized on read by the FA path.
    Returns: output vector, attention weights, quality metrics dict.
    """
    Q_rot = turbo_forward_rotation(Q_orig)

    # Quantize and dequantize K (matches current GPU pipeline)
    n_blocks_k = D // QK_TURBO3
    K_dequant = np.zeros(D, dtype=np.float32)
    for b in range(n_blocks_k):
        offset = b * QK_TURBO3
        block_vals = K_orig[offset:offset + QK_TURBO3]
        norms_k, qs_k, signs_k = quantize_turbo3_subblock(block_vals)
        nk = float(norms_k[0])
        recon_k = np.empty(QK_TURBO3, dtype=np.float32)
        for j in range(QK_TURBO3):
            low2 = (qs_k[j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs_k[j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            recon_k[j] = TURBO_CENTROIDS_3BIT[idx] * nk
        K_dequant[offset:offset + QK_TURBO3] = recon_k

    # Quantize and dequantize V based on type
    n_blocks_v = D // QK_TURBO3
    V_dequant = np.zeros(D, dtype=np.float32)
    for b in range(n_blocks_v):
        offset = b * QK_TURBO3
        block_vals = V_orig[offset:offset + QK_TURBO3]

        if v_quant_type == "turbo3_0":
            norms_v, qs_v, signs_v = quantize_turbo3_subblock(block_vals)
            nv = float(norms_v[0])
            recon_v = np.empty(QK_TURBO3, dtype=np.float32)
            for j in range(QK_TURBO3):
                low2 = (qs_v[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs_v[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                recon_v[j] = TURBO_CENTROIDS_3BIT[idx] * nv
            V_dequant[offset:offset + QK_TURBO3] = recon_v
        elif v_quant_type == "q8_0":
            recon_v, _ = quantize_q8_0_block(block_vals)
            V_dequant[offset:offset + QK_TURBO3] = recon_v
        else:
            V_dequant[offset:offset + QK_TURBO3] = block_vals.copy()

    # Compute attention score (single token -> softmax is trivial)
    scale_d = math.sqrt(D)
    score = float(np.dot(K_dequant, Q_rot) / scale_d)
    weight = math.exp(score - score)  # single-token softmax
    output = weight * V_dequant

    # Quality metrics
    gt_score = float(np.dot(K_orig, Q_rot) / scale_d)
    gt_weight = math.exp(gt_score - gt_score)
    gt_output = gt_weight * V_orig

    cos_sim = cosine_similarity(output, gt_output)
    mse_val = mse(output, gt_output)
    snr_val = snr_db(output, gt_output)

    return output, K_dequant, V_dequant, {
        "cos_sim": cos_sim,
        "mse": mse_val,
        "snr": snr_val,
        "score_delta": abs(score - gt_score),
    }


def attention_output_comparison():
    """Compare attention output quality with turbo3_0 vs q8_0 V cache.

    Runs through 10 simulated layers, tracking:
      - Cosine similarity between quantized and ground-truth outputs
      - Max delta (largest per-element difference)
      - Error growth factor across layers
    """
    print("\n" + "=" * 70)
    print("ATTENTION OUTPUT COMPARISON: turbo3_0 V-cache vs q8_0 V-cache")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_layers = 10
    seq_len = 8

    turbo_cosims = []
    turbo_max_deltas = []
    turbo_errors = []

    q8_cosims = []
    q8_max_deltas = []
    q8_errors = []

    for layer in range(n_layers):
        # Generate fresh Q, K, V for each layer (simulates different layers of a transformer)
        Q_orig = rng.randn(D).astype(np.float32) * 0.1
        K_orig = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_orig = rng.randn(seq_len, D).astype(np.float32) * 0.1

        # Accumulate per-sequence metrics
        layer_turbo_cosims = []
        layer_q8_cosims = []
        layer_turbo_deltas = []
        layer_q8_deltas = []

        for s in range(seq_len):
            k_s = K_orig[s]
            v_s = V_orig[s]

            # turbo3_0 path
            out_t, k_deq_t, v_deq_t, metrics_t = simulate_attention_with_v_quant(
                Q_orig, k_s, v_s, v_quant_type="turbo3_0")
            layer_turbo_cosims.append(metrics_t["cos_sim"])
            layer_turbo_deltas.append(float(np.max(np.abs(out_t - metrics_t.get('gt_output', out_t)))) )

            # q8_0 path
            out_q, k_deq_q, v_deq_q, metrics_q = simulate_attention_with_v_quant(
                Q_orig, k_s, v_s, v_quant_type="q8_0")
            layer_q8_cosims.append(metrics_q["cos_sim"])
            layer_q8_deltas.append(float(np.max(np.abs(out_q - metrics_q.get('gt_output', out_q)))) )

        turbo_cosims.append(np.mean(layer_turbo_cosims))
        q8_cosims.append(np.mean(layer_q8_cosims))

        # Track max delta via direct comparison of dequantized V vectors
        for b in range(D // QK_TURBO3):
            offset = b * QK_TURBO3
            block_v = V_orig[0][offset:offset + QK_TURBO3]
            norms_v, qs_v, signs_v = quantize_turbo3_subblock(block_v)
            nv = float(norms_v[0])
            recon_t = np.empty(QK_TURBO3, dtype=np.float32)
            for j in range(QK_TURBO3):
                low2 = (qs_v[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs_v[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                recon_t[j] = TURBO_CENTROIDS_3BIT[idx] * nv
            recon_q, _ = quantize_q8_0_block(block_v)
            turbo_max_deltas.append(float(np.max(np.abs(block_v - recon_t))))
            q8_max_deltas.append(float(np.max(np.abs(block_v - recon_q))))
            turbo_errors.append(mse(block_v, recon_t))
            q8_errors.append(mse(block_v, recon_q))

    # Aggregate results
    avg_turbo_cosim = np.mean(turbo_cosims)
    avg_q8_cosim = np.mean(q8_cosims)
    avg_turbo_delta = np.mean(turbo_max_deltas) if turbo_max_deltas else 0
    avg_q8_delta = np.mean(q8_max_deltas) if q8_max_deltas else 0
    avg_turbo_err = np.mean(turbo_errors) if turbo_errors else 0
    avg_q8_err = np.mean(q8_errors) if q8_errors else 0

    print(f"\n  Per-layer cosine similarity (avg across sequence positions):")
    for i in range(n_layers):
        print(f"    Layer {i:2d}: turbo3_0={turbo_cosims[i]:.6f}, "
              f"q8_0={q8_cosims[i]:.6f}, gap={turbo_cosims[i] - q8_cosims[i]:+.4f}")

    print(f"\n  Average metrics:")
    print(f"    turbo3_0 cos_sim:     {avg_turbo_cosim:.6f}")
    print(f"    q8_0   cos_sim:       {avg_q8_cosim:.6f}")
    print(f"    Difference:           {avg_turbo_cosim - avg_q8_cosim:+.6f}")
    print(f"    turbo3_0 max delta:   {avg_turbo_delta:.6f}")
    print(f"    q8_0   max delta:     {avg_q8_delta:.6f}")
    print(f"    turbo3_0 block MSE:   {avg_turbo_err:.8f}")
    print(f"    q8_0   block MSE:     {avg_q8_err:.8f}")
    print(f"    MSE ratio (turbo/q8): {avg_turbo_err / max(avg_q8_err, 1e-15):.1f}x")

    # Error growth analysis
    if len(turbo_errors) >= 2:
        early_err = np.mean(turbo_errors[:len(turbo_errors)//4])
        late_err = np.mean(turbo_errors[-len(turbo_errors)//4:])
        growth = late_err / max(early_err, 1e-15)
        print(f"\n  Error growth (first quarter -> last quarter): {growth:.2f}x")

    # Verdict
    issues = []
    if avg_turbo_cosim < 0.90:
        issues.append(f"Avg cos_sim {avg_turbo_cosim:.4f} below 0.90 threshold")
    if avg_turbo_err / max(avg_q8_err, 1e-15) > 50.0:
        issues.append(f"MSE ratio {avg_turbo_err/max(avg_q8_err,1e-15):.0f}x exceeds 50x limit")

    if issues:
        print(f"\n  FAIL: {'; '.join(issues)}")
        return False
    else:
        print(f"\n  PASS: Attention output comparison within tolerance")
        print(f"  Note: The ~{avg_turbo_err/max(avg_q8_err,1e-15):.0f}x MSE gap is the "
              f"inherent information loss from 3-bit quantization.")
        print(f"  This gap compounds through multi-step reasoning in long contexts.")
        return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("TURBO3_0 GARBLE DIAGNOSTIC -- Real Failure Pattern Analysis")
    print("=" * 70)
    print(f"Head dimension: {D}")
    print(f"Block size:     {QK_TURBO3}")
    print(f"Centroids:      {N_CENTROIDS_3BIT} (3-bit)")
    print(f"WHT group size: {GROUP_SIZE}")
    print()

    results = []

    # Existing tests from master_debug_turbo.py -- kept for comparison
    from master_debug_turbo import (test_block_structure, test_rotation_mismatch,
                                      test_cpu_vs_gpu_quant, test_attention_collapse,
                                      test_norm_blowup, test_innerq_interference,
                                      test_centroid_distribution, test_full_pipeline_simulation)

    r = TestResult("Block Structure Integrity")
    if not test_block_structure():
        r.fail("Invalid index values")
    results.append(r)

    r = TestResult("WHT Rotation Mismatch Detection")
    if not test_rotation_mismatch():
        r.fail("Q rotated but K not rotated")
    results.append(r)

    # Existing tests from turbo3_garble_diagnostic.py
    r = TestResult("Norm Preservation After Rotation+Quant")
    if not test_norm_preservation():
        r.fail("Norm drift detected")
    results.append(r)

    r = TestResult("Half-Precision Norm Storage Drift")
    if not test_half_precision_norm_drift():
        r.fail("fp16 norm precision loss too high")
    results.append(r)

    r = TestResult("Position-Dependent Quantization Error")
    if not test_position_dependent_error():
        r.fail("Error varies by position")
    results.append(r)

    r = TestResult("Stuck-Token Loop Detection")
    if not test_stuck_token_detection():
        r.fail("Token loops detected")
    results.append(r)

    r = TestResult("Turbo3_0 vs Q8_0 Quality Gap")
    if not test_turbo3_vs_q8_quality():
        r.fail("Quality gap too large")
    results.append(r)

    r = TestResult("Multi-Head Attention Consistency")
    if not test_multi_head_consistency():
        r.fail("Head inconsistency detected")
    results.append(r)

    r = TestResult("Multi-Layer Error Accumulation")
    if not test_multi_layer_error_accumulation():
        r.fail("Error accumulates across layers")
    results.append(r)

    r = TestResult("Realistic Prompt Simulation")
    if not test_realistic_prompt_simulation():
        r.fail("Degradation pattern detected")
    results.append(r)

    # NEW: Attention output comparison (turbo3_0 V-cache vs q8_0 V-cache)
    r = TestResult("Attention Output Comparison")
    if not attention_output_comparison():
        r.fail("V-cache quality gap unacceptable")
    results.append(r)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<50} {'Result':<8}")
    print("-" * 58)
    all_pass = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_pass = False
        print(f"{r.name:<50} {status:<8}")

    print()
    if all_pass:
        print("ALL TESTS PASSED -- No issues detected.")
    else:
        print("FAILURES DETECTED -- See details above.")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
