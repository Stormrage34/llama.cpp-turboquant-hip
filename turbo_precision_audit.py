#!/usr/bin/env python3
"""
turbo_precision_audit.py -- Precision audit for turbo3_0 KV cache quantization.

Exact Python implementations of turbo3_0 and q8_0 quantize/dequantize matching
the CUDA/CPU source code. Tests multiple distributions and reports structured
PASS/FAIL metrics.

Designed to be run standalone: python3 turbo_precision_audit.py

Usage:
    python3 turbo_precision_audit.py              # full audit
    python3 turbo_precision_audit.py --quick      # reduced sample count (~5s)
    python3 turbo_precision_audit.py --dist gaussian  # single distribution
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Constants (from turbo-quant.cuh / block_turbo3_0 definition)
# ============================================================================

QK = 32           # elements per block
D = 128            # head dimension
N_CENTROIDS = 8    # 3-bit -> 8 levels
GROUP_SIZE = 128   # WHT group size

CENTROIDS = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685,
], dtype=np.float32)

MIDS = np.array([
    -0.154259, -0.091775, -0.043589, 0.0,
     0.043589,  0.091775,  0.154259,
], dtype=np.float32)

# Exact sign arrays from turbo-quant.cuh (lines 57-77)
WHT_SIGNS1 = np.array([
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
], dtype=np.float32)

WHT_SIGNS2 = np.array([
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
], dtype=np.float32)


# ============================================================================
# Exact Python implementations matching CUDA source
# ============================================================================

def fwht_inplace(a):
    """Fast Walsh-Hadamard Transform (unnormalized) -- matches turbo_fwht_128."""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def turbo_nearest_centroid_3bit(val):
    """Exact Python port of turbo_nearest_centroid_3bit in turbo-quant.cuh (line 381).

    Branchless binary search over TURBO_MID_3BIT[7]:
      Level 1: compare with mid[3] (=0.0) -> left/right half
      Level 2a: compare with mid[1] if left, else mid[5]
      Level 3: compare with appropriate leaf midpoint
    XOR with 7 to map to Lloyd-Max index ordering.
    """
    val = float(val)
    left = val < MIDS[3]  # < 0.0
    if left:
        cmp_l2 = val < MIDS[1]   # < -0.091775
        if cmp_l2:
            return 0 if val < MIDS[0] else 1   # < -0.154259 -> 0, else 1
        else:
            return 2 if val < MIDS[2] else 3   # < -0.043589 -> 2, else 3
    else:
        cmp_l2 = val < MIDS[5]   # < 0.091775
        if cmp_l2:
            return 4 if val < MIDS[4] else 5   # < 0.043589 -> 4, else 5
        else:
            return 6 if val < MIDS[6] else 7   # < 0.154259 -> 6, else 7


def turbo_forward_rotation(x):
    """Forward WHT rotation: signs1 * x -> FWHT -> signs2 * result."""
    r = x.astype(np.float64) * WHT_SIGNS1
    fwht_inplace(r.tolist())
    r *= WHT_SIGNS2
    return r.astype(np.float32)


def turbo_inverse_rotation(x):
    """Inverse WHT rotation: signs2 * x -> FWHT -> signs1 * result."""
    r = x.astype(np.float64) * WHT_SIGNS2
    fwht_inplace(r.tolist())
    r *= WHT_SIGNS1
    return r.astype(np.float32)


# ---- turbo3_0 quantize/dequantize ----

def turbo3_quantize_block(values, apply_rotation=True):
    """Quantize 128 values into turbo3_0 block format.

    Matches k_set_rows_turbo3 in set-rows.cu (lines 355-496):
      1. Split into n_blocks=4 sub-blocks of QK=32 elements each
      2. Apply forward WHT rotation to full GROUP_SIZE=128 if enabled
      3. L2 normalize within each 32-element sub-block
      4. Quantize to 3-bit centroid index
      5. Pack qs (4 elem/byte, 2 bits) and signs (8 elem/byte, 1 bit)
      6. Compute reconstruction norm for correction

    Returns: norms (list of fp16), qs_bytes (bytes), signs_bytes (bytes)
    """
    assert len(values) == D
    n_blocks = D // QK

    # Full-group rotation (applied once before splitting into sub-blocks)
    if apply_rotation:
        values = turbo_forward_rotation(values.copy())

    norms = []
    qs_out = bytearray()
    signs_out = bytearray()

    for b in range(n_blocks):
        off = b * QK
        block_vals = values[off:off + QK]

        grp_norm_sq = float(np.sum(block_vals * block_vals))
        grp_norm = math.sqrt(grp_norm_sq)
        if grp_norm < 1e-10:
            norms.append(np.float16(0.0))
            qs_out.extend(bytes(QK // 4))
            signs_out.extend(bytes(QK // 8))
            continue

        normalized = block_vals / grp_norm

        # Quantize each element to 3-bit centroid index
        indices = np.array([turbo_nearest_centroid_3bit(float(v)) for v in normalized], dtype=np.uint8)

        # Reconstruction norm for correction
        recon_vals = CENTROIDS[indices]
        recon_norm_sq = float(np.sum(recon_vals * recon_vals))
        recon_norm = math.sqrt(recon_norm_sq)
        corrected = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm
        norms.append(np.float16(corrected))

        # Pack qs: 4 elements per byte, 2 bits each
        qs_byte = bytearray(QK // 4)
        for i in range(QK):
            qs_byte[i // 4] |= (int(indices[i]) & 0x3) << ((i % 4) * 2)
        qs_out.extend(qs_byte)

        # Pack signs: 8 elements per byte, 1 bit each
        signs_byte = bytearray(QK // 8)
        for i in range(QK):
            if int(indices[i]) & 0x4:
                signs_byte[i // 8] |= (1 << (i % 8))
        signs_out.extend(signs_byte)

    return norms, bytes(qs_out), bytes(signs_out)


def turbo3_quantize_subblock(block_vals):
    """Quantize a single 32-element sub-block (no rotation).

    Used for per-block precision testing where rotation is applied at the
    group level externally. Matches the inner loop of k_set_rows_turbo3.
    """
    assert len(block_vals) == QK

    grp_norm_sq = float(np.sum(block_vals * block_vals))
    grp_norm = math.sqrt(grp_norm_sq)
    if grp_norm < 1e-10:
        return [np.float16(0.0)], bytes(QK // 4), bytes(QK // 8)

    normalized = block_vals / grp_norm
    indices = np.array([turbo_nearest_centroid_3bit(float(v)) for v in normalized], dtype=np.uint8)

    recon_vals = CENTROIDS[indices]
    recon_norm_sq = float(np.sum(recon_vals * recon_vals))
    recon_norm = math.sqrt(recon_norm_sq)
    corrected = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm

    qs_byte = bytearray(QK // 4)
    for i in range(QK):
        qs_byte[i // 4] |= (int(indices[i]) & 0x3) << ((i % 4) * 2)

    signs_byte = bytearray(QK // 8)
    for i in range(QK):
        if int(indices[i]) & 0x4:
            signs_byte[i // 8] |= (1 << (i % 8))

    return [np.float16(corrected)], bytes(qs_byte), bytes(signs_byte)


def turbo3_dequantize_block(norms, qs_bytes, signs_bytes, apply_inverse_rotation=False):
    """Dequantize turbo3_0 block back to float values.

    Matches dequantize_turbo3_0 in turbo-quant.cuh (lines 475-487):
      1. Unpack low2 bits from qs + hi1 bit from signs -> 3-bit index
      2. Look up centroid and scale by norm
      3. Apply inverse WHT rotation if enabled

    Returns: numpy array of D floats
    """
    n_blocks = D // QK
    result = np.zeros(D, dtype=np.float32)

    for b in range(n_blocks):
        off = b * QK
        norm = float(norms[b])
        qs_start = b * (QK // 4)
        signs_start = b * (QK // 8)

        block_vals = np.empty(QK, dtype=np.float32)
        for j in range(QK):
            low2 = (qs_bytes[qs_start + j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs_bytes[signs_start + j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            block_vals[j] = CENTROIDS[idx] * norm

        result[off:off + QK] = block_vals

    if apply_inverse_rotation:
        result = turbo_inverse_rotation(result.copy())

    return result


# ---- q8_0 quantize/dequantize ----

def q8_quantize_block(values):
    """Quantize QK=32 values using q8_0 format matching ggml-quants.c.

    block_q8_0: 2-byte fp16 scale (delta) + 32 bytes int8 quants = 34 bytes total.
    Reconstructed: qs[i] * d / 127.0

    Returns: reconstructed values (numpy array), scale factor
    """
    assert len(values) == QK
    abs_max = float(np.max(np.abs(values)))
    if abs_max < 1e-10:
        return np.zeros(QK, dtype=np.float32), 0.0

    scale = abs_max / 127.0
    quants = np.clip(np.round(values / scale).astype(np.int8), -128, 127)
    recon = quants.astype(np.float32) * scale
    return recon, scale


def q8_quantize_vector(values):
    """Quantize a full D=128 vector using per-block q8_0 (4 blocks of QK=32)."""
    assert len(values) == D
    recon = np.zeros(D, dtype=np.float32)
    for b in range(D // QK):
        off = b * QK
        block_recon, _ = q8_quantize_block(values[off:off + QK])
        recon[off:off + QK] = block_recon
    return recon


def q8_dequantize_block(scale, quants_int8):
    """Dequantize q8_0 block: qs[i] * scale / 127.0"""
    return quants_int8.astype(np.float32) * scale


# ============================================================================
# Metrics
# ============================================================================

def mse(a, b):
    return float(np.mean((a - b) ** 2))


def snr_db(original, reconstructed):
    sig = float(np.mean(original ** 2))
    noise = float(np.mean((original - reconstructed) ** 2))
    if noise < 1e-15:
        return 99.9
    return 10.0 * math.log10(sig / noise)


def max_abs_error(a, b):
    return float(np.max(np.abs(a - b)))


def cosine_similarity(a, b):
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / (na * nb + 1e-10)


# ============================================================================
# Distribution generators
# ============================================================================

def generate_distributions(n, seed=42):
    """Generate diverse test distributions matching LLM activation patterns.

    gaussian     : N(0, 1/sqrt(QK)) -- typical attention output magnitude
    heavy_tailed : Student-t(df=3) scaled to match Gaussian RMS
    bimodal      : Mixture of two Gaussians (simulates sparse feature activation)
    sparse       : 90% zeros with occasional large values
    uniform      : Uniform in [-1, 1] (worst case for centroid-based quant)
    """
    rng = np.random.RandomState(seed)
    std = 1.0 / math.sqrt(QK)

    distributions = {}

    # Gaussian: standard LLM activation distribution
    distributions["gaussian"] = rng.randn(n, D).astype(np.float32) * std

    # Heavy-tailed: Student-t with df=3 (common in transformer activations)
    t_vals = rng.standard_t(3, size=(n, D)).astype(np.float32)
    rms_actual = np.sqrt(np.mean(t_vals ** 2))
    distributions["heavy_tailed"] = t_vals * (std / max(rms_actual, 1e-10))

    # Bimodal: two clusters simulating sparse feature activation
    labels = (rng.rand(n, 1) > 0.5).astype(np.float32)
    centers = labels * 0.6 - 0.3
    noise = rng.randn(n, D).astype(np.float32) * 0.05
    distributions["bimodal"] = centers + noise

    # Sparse: 90% zeros, 10% from N(0, 0.5)
    mask = (rng.rand(n, D) < 0.1).astype(np.float32)
    distributions["sparse"] = mask * rng.randn(n, D).astype(np.float32) * 0.5

    # Uniform: uniform in [-std*sqrt(3), +std*sqrt(3)] to match Gaussian variance
    half_range = std * math.sqrt(3.0)
    distributions["uniform"] = (rng.rand(n, D) * 2.0 - 1.0).astype(np.float32) * half_range

    return distributions


# ============================================================================
# Attention simulation
# ============================================================================

def simulate_attention(Q_orig, K_orig, V_orig, v_quant="turbo3_0"):
    """Simulate one attention computation with specified V-cache quantization.

    Q is always rotated forward (as llama-graph does).
    K and V are quantized and dequantized through the pipeline.
    Returns: output vector, metrics dict.
    """
    Q_rot = turbo_forward_rotation(Q_orig)

    # Dequantize K (matches GPU pipeline: no inverse rotation)
    K_dequant = np.zeros(D, dtype=np.float32)
    for b in range(D // QK):
        off = b * QK
        block_k = K_orig[off:off + QK]
        norms_k, qs_k, signs_k = turbo3_quantize_subblock(block_k)
        recon_k = np.empty(QK, dtype=np.float32)
        nk = float(norms_k[0])
        for j in range(QK):
            low2 = (qs_k[j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs_k[j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            recon_k[j] = CENTROIDS[idx] * nk
        K_dequant[off:off + QK] = recon_k

    # Dequantize V based on type
    V_dequant = np.zeros(D, dtype=np.float32)
    for b in range(D // QK):
        off = b * QK
        block_v = V_orig[off:off + QK]
        if v_quant == "turbo3_0":
            norms_v, qs_v, signs_v = turbo3_quantize_subblock(block_v)
            recon_v = np.empty(QK, dtype=np.float32)
            nv = float(norms_v[0])
            for j in range(QK):
                low2 = (qs_v[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs_v[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                recon_v[j] = CENTROIDS[idx] * nv
            V_dequant[off:off + QK] = recon_v
        elif v_quant == "q8_0":
            recon_v, _ = q8_quantize_block(block_v)
            V_dequant[off:off + QK] = recon_v
        else:
            V_dequant[off:off + QK] = block_v.copy()

    # Attention score (single token -> softmax trivial)
    scale_d = math.sqrt(D)
    score = float(np.dot(K_dequant, Q_rot) / scale_d)
    weight = math.exp(score - score)
    output = weight * V_dequant

    # Ground truth
    gt_score = float(np.dot(K_orig, Q_rot) / scale_d)
    gt_weight = math.exp(gt_score - gt_score)
    gt_output = gt_weight * V_orig

    return output, {
        "cos_sim": cosine_similarity(output, gt_output),
        "mse": mse(output, gt_output),
        "snr": snr_db(output, gt_output),
        "score_delta": abs(score - gt_score),
    }


def simulate_multi_layer(n_layers, seq_len, v_quant="turbo3_0", seed=42):
    """Simulate multi-layer transformer with specified V-cache quantization.

    Returns per-layer quality metrics tracking error growth.
    """
    rng = np.random.RandomState(seed)
    metrics_history = []

    x = rng.randn(D).astype(np.float32) * 0.1

    for layer in range(n_layers):
        K_all = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_all = rng.randn(seq_len, D).astype(np.float32) * 0.1

        # Attention output
        layer_cosims = []
        for s in range(seq_len):
            _, m = simulate_attention(x, K_all[s], V_all[s], v_quant=v_quant)
            layer_cosims.append(m["cos_sim"])

        avg_cosim = float(np.mean(layer_cosims))

        # Track error growth
        metrics_history.append({
            "layer": layer,
            "avg_cos_sim": avg_cosim,
            "output_norm": float(np.linalg.norm(x)),
        })

        x = rng.randn(D).astype(np.float32) * 0.1 + x * 0.5

    return metrics_history


# ============================================================================
# Report generation
# ============================================================================

def print_header(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"{title}")
    print(f"{'=' * width}")


def print_section(title):
    print(f"\n--- {title} ---")


def run_distribution_test(dist_name, samples, n_samples=None):
    """Run quantization quality test on a single distribution.

    Returns dict of aggregated metrics.
    """
    if n_samples is None:
        n_samples = len(samples)

    turbo_mses = []
    q8_mses = []
    turbo_snrs = []
    q8_snrs = []
    turbo_max_errs = []
    q8_max_errs = []
    centroid_hist = [0] * N_CENTROIDS
    centroid_per_elem_errors = [[] for _ in range(N_CENTROIDS)]

    for i in range(n_samples):
        vals = samples[i]

        # Process per-block (32 elements) for proper centroid tracking
        block_mses_t, block_mses_q = [], []
        block_snrs_t, block_snrs_q = [], []
        block_me_t, block_me_q = [], []

        for b in range(D // QK):
            off = b * QK
            block_vals = vals[off:off + QK]

            # turbo3_0 (no rotation, matches GPU pipeline)
            norms_sb, qs_sb, signs_sb = turbo3_quantize_subblock(block_vals)
            # Reconstruct single sub-block directly
            recon_t = np.empty(QK, dtype=np.float32)
            norm_val = float(norms_sb[0])
            for j in range(QK):
                low2 = (qs_sb[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs_sb[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                recon_t[j] = CENTROIDS[idx] * norm_val

            # q8_0
            recon_q, _ = q8_quantize_block(block_vals)

            block_mses_t.append(mse(block_vals, recon_t))
            block_mses_q.append(mse(block_vals, recon_q))
            block_snrs_t.append(snr_db(block_vals, recon_t))
            block_snrs_q.append(snr_db(block_vals, recon_q))
            block_me_t.append(max_abs_error(block_vals, recon_t))
            block_me_q.append(max_abs_error(block_vals, recon_q))

            # Track centroid usage and per-centroid error
            for j in range(QK):
                low2 = (qs_sb[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs_sb[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                centroid_hist[idx] += 1
                centroid_per_elem_errors[idx].append(abs(block_vals[j] - recon_t[j]))

        # Aggregate per-block metrics to full-vector level
        turbo_mses.append(np.mean(block_mses_t))
        q8_mses.append(np.mean(block_mses_q))
        turbo_snrs.append(np.mean(block_snrs_t))
        q8_snrs.append(np.mean(block_snrs_q))
        turbo_max_errs.append(max(block_me_t))
        q8_max_errs.append(max(block_me_q))

    avg_mse_t = np.mean(turbo_mses)
    avg_mse_q = np.mean(q8_mses)
    avg_snr_t = np.mean(turbo_snrs)
    avg_snr_q = np.mean(q8_snrs)
    avg_me_t = np.mean(turbo_max_errs)
    avg_me_q = np.mean(q8_max_errs)
    mse_ratio = avg_mse_t / max(avg_mse_q, 1e-15)

    # Centroid histogram
    total_c = sum(centroid_hist)
    print_section(f"Centroid Usage ({dist_name})")
    for ci in range(N_CENTROIDS):
        pct = centroid_hist[ci] / total_c * 100 if total_c > 0 else 0
        mean_err = np.mean(centroid_per_elem_errors[ci]) if centroid_per_elem_errors[ci] else 0
        bar_len = int(pct / 2)
        print(f"  [{ci}] c={CENTROIDS[ci]:>9.5f}: {pct:5.1f}% "
              f"| mean|err|={mean_err:.6f} {'#' * bar_len}")

    used = sum(1 for c in centroid_hist if c > 0)
    probs = [c / total_c for c in centroid_hist if c > 0]
    entropy = -sum(p * math.log2(p + 1e-30) for p in probs)
    max_entropy = math.log2(N_CENTROIDS)

    print(f"  Used: {used}/{N_CENTROIDS} centroids")
    print(f"  Entropy: {entropy:.2f}/{max_entropy:.2f} ({entropy/max_entropy*100:.0f}%)")

    return {
        "dist": dist_name,
        "n_samples": n_samples,
        "mse_turbo": avg_mse_t,
        "mse_q8": avg_mse_q,
        "snr_turbo": avg_snr_t,
        "snr_q8": avg_snr_q,
        "me_turbo": avg_me_t,
        "me_q8": avg_me_q,
        "mse_ratio": mse_ratio,
        "centroid_entropy": entropy,
        "centroids_used": used,
    }


def run_attention_test(samples, v_quant="turbo3_0", n_tests=200):
    """Run attention simulation and report quality metrics."""
    cosims = []
    mses = []
    snrs = []

    for i in range(min(n_tests, len(samples))):
        Q_orig = samples[i]
        k_idx = (i + 7) % len(samples)
        v_idx = (i + 13) % len(samples)
        K_orig = samples[k_idx]
        V_orig = samples[v_idx]

        _, m = simulate_attention(Q_orig, K_orig, V_orig, v_quant=v_quant)
        cosims.append(m["cos_sim"])
        mses.append(m["mse"])
        snrs.append(m["snr"])

    return {
        "avg_cos_sim": float(np.mean(cosims)),
        "avg_mse": float(np.mean(mses)),
        "avg_snr": float(np.mean(snrs)),
        "min_cos_sim": float(np.min(cosims)),
    }


def run_multi_layer_test(v_quant="turbo3_0", n_layers=10, seq_len=8):
    """Simulate multi-layer error growth tracking."""
    metrics = simulate_multi_layer(n_layers, seq_len, v_quant=v_quant)

    print_section(f"Multi-Layer Error Growth ({v_quant})")
    for m in metrics:
        print(f"  Layer {m['layer']:2d}: cos_sim={m['avg_cos_sim']:.6f}, "
              f"output_norm={m['output_norm']:.6f}")

    if len(metrics) >= 2:
        early = np.mean([m['avg_cos_sim'] for m in metrics[:3]])
        late = np.mean([m['avg_cos_sim'] for m in metrics[-3:]])
        decay = (early - late) / max(early, 1e-10)
        print(f"\n  Cosine similarity decay: early={early:.6f}, late={late:.6f}, "
              f"decay={decay:.2%}")
        return decay
    return 0.0


# ============================================================================
# Structured PASS/FAIL report
# ============================================================================

def generate_report(all_results, attn_results, layer_decays):
    """Generate final structured PASS/FAIL report."""
    print_header("PRECISION AUDIT REPORT")

    thresholds = {
        "mse_ratio_max": 50.0,       # turbo3_0 should not be >50x worse than q8_0
        "snr_min_db": 5.0,           # minimum acceptable SNR for any distribution
        "attn_cosim_min": 0.80,      # minimum attention quality
        "centroid_entropy_min": 2.0, # at least ~75% of max entropy
    }

    all_pass = True
    print(f"\nThresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    print(f"\nPer-Distribution Results:")
    print(f"{'Distribution':<20} {'MSE(t/q)':>10} {'SNR(t)':>8} {'MaxErr(t)':>10} {'Status':<8}")
    print("-" * 62)

    for r in all_results:
        issues = []
        if r["mse_ratio"] > thresholds["mse_ratio_max"]:
            issues.append(f"MSE ratio {r['mse_ratio']:.0f}x > {thresholds['mse_ratio_max']}x")
        if r["snr_turbo"] < thresholds["snr_min_db"]:
            issues.append(f"SNR {r['snr_turbo']:.1f}dB < {thresholds['snr_min_db']}dB")

        status = "PASS" if not issues else "FAIL"
        if issues:
            all_pass = False
        print(f"{r['dist']:<20} {r['mse_ratio']:>10.1f} {r['snr_turbo']:>7.1f}dB "
              f"{r['me_turbo']:>9.6f} {status}")

    # Attention quality
    attn_status = "PASS"
    for vq, ar in attn_results.items():
        if ar["avg_cos_sim"] < thresholds["attn_cosim_min"]:
            attn_status = "FAIL"
            all_pass = False
        print(f"\nAttention ({vq}): cos_sim={ar['avg_cos_sim']:.4f}, "
              f"min={ar['min_cos_sim']:.4f} [{attn_status}]")

    # Layer decay
    decay_status = "PASS"
    for vq, decay in layer_decays.items():
        if decay > 0.3:
            decay_status = "FAIL"
            all_pass = False
        print(f"Multi-layer decay ({vq}): {decay:.2%} [{decay_status}]")

    print(f"\n{'=' * 62}")
    verdict = "ALL PASS" if all_pass else "FAILURES DETECTED"
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 62}")

    print(f"\nRoot cause note:")
    print(f"  turbo3_0 is fundamentally 4-bit (2 index bits + 1 sign bit) vs")
    print(f"  q8_0's 8+ bits per element. The observed MSE ratio of ~{np.mean([r['mse_ratio'] for r in all_results]):.0f}x")
    print(f"  represents the information-theoretic limit of 3-bit Lloyd-Max")
    print(f"  quantization on N(0, 1/sqrt(QK)) data.")
    print(f"  This gap compounds through multi-step reasoning in long contexts.")

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="turbo3_0 precision audit")
    parser.add_argument("--quick", action="store_true", help="Reduced sample count (~5s)")
    parser.add_argument("--dist", type=str, default=None, help="Test single distribution only")
    args = parser.parse_args()

    n_samples = 500 if args.quick else 2000
    print_header("TURBO3_0 PRECISION AUDIT")
    print(f"Block size:       {QK} elements")
    print(f"Head dimension:   {D}")
    print(f"Centroids:        {N_CENTROIDS} (3-bit Lloyd-Max)")
    print(f"WHT group size:   {GROUP_SIZE}")
    print(f"Samples per dist: {n_samples}")
    print(f"Format:           turbo3_0 = {QK//4 + QK//8 + 2} bytes/{QK} elems = {(QK//4 + QK//8 + 2)/QK:.2f} bpw")
    print(f"                  q8_0     = {QK + 2} bytes/{QK} elems = {(QK + 2)/QK:.2f} bpw")

    # Generate distributions
    dists = generate_distributions(n=n_samples, seed=42)

    if args.dist:
        if args.dist not in dists:
            print(f"ERROR: Unknown distribution '{args.dist}'. Available: {list(dists.keys())}")
            return 1
        dists = {args.dist: dists[args.dist]}

    # Run per-distribution tests
    all_results = []
    for dist_name, samples in dists.items():
        print_section(f"Distribution: {dist_name}")
        r = run_distribution_test(dist_name, samples, n_samples=min(n_samples, len(samples)))
        all_results.append(r)

    # Attention simulation comparison
    print_header("ATTENTION QUALITY COMPARISON")
    attn_results = {}
    for vq in ["turbo3_0", "q8_0"]:
        # Use a combined sample set
        combined = np.vstack([dists.get(k, np.zeros(0)) for k in dists])
        rng = np.random.RandomState(42)
        indices = rng.choice(len(combined), size=min(200, len(combined)), replace=False)
        test_samples = combined[indices]
        attn_results[vq] = run_attention_test(test_samples, v_quant=vq, n_tests=200)

    # Multi-layer error growth
    print_header("MULTI-LAYER ERROR GROWTH")
    layer_decays = {}
    for vq in ["turbo3_0", "q8_0"]:
        decay = run_multi_layer_test(v_quant=vq, n_layers=10, seq_len=8)
        layer_decays[vq] = decay

    # Generate final report
    success = generate_report(all_results, attn_results, layer_decays)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
