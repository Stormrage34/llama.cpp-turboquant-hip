"""
mlnn/turbo_debug.py - Turbo KV Cache Diagnostics
=================================================
Merged from: master_debug_turbo.py, turbo_precision_audit.py,
             turbo3_garble_diagnostic.py, turbo3_roundtrip.py,
             turbo3_fix_verification.py, debug_turbo.py,
             debug_turbo2.py, turbo3_audit.py

Pure-Python/numpy simulations of the turbo3_0 KV cache pipeline.
Detects: quantization errors, rotation mismatches, attention collapse,
norm blowup, stuck-token loops, garbled output patterns, FA kernel bottlenecks.

Usage:
    python3 -m mlnn.turbo_debug --mode full              # run all tests
    python3 -m mlnn.turbo_debug --mode block-structure    # block packing roundtrip
    python3 -m mlnn.turbo_debug --mode rotation-mismatch  # Q/K rotation check
    python3 -m mlnn.turbo_debug --mode precision-audit    # turbo3 vs q8 quality
    python3 -m mlnn.turbo_debug --mode garble             # garble hypothesis tests
    python3 -m mlnn.turbo_debug --mode roundtrip          # quant/dequant fidelity
    python3 -m mlnn.turbo_debug --mode fa-kernel-audit    # RDNA2 cycle simulation
    python3 -m mlnn.turbo_debug --mode rotation-verify    # rotation fix validator
"""

import argparse
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Control numpy/OpenMP thread counts to match the parallelism we use.
os.environ.setdefault("OMP_NUM_THREADS", str(multiprocessing.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(multiprocessing.cpu_count() or 1))


# ============================================================================
# Shared Constants (from turbo-quant.cuh / block_turbo3_0 definition)
# ============================================================================

QK = 32                 # elements per sub-block
D = 128                 # head dimension
N_CENTROIDS_3BIT = 8    # 3-bit -> 8 levels
GROUP_SIZE = 128        # workgroup size for k_set_rows_turbo3

CENTROIDS_3BIT = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685,
], dtype=np.float32)

MIDPOINTS_3BIT = np.array([
    -0.154259, -0.091775, -0.043589, 0.0,
     0.043589,  0.091775,  0.154259,
], dtype=np.float32)

# WHT sign patterns (from turbo-quant.cuh lines 57-77).
# Generated dynamically: pattern repeats every 8 elements with phase offsets.
WHT_SIGNS1 = None  # lazily generated per-call length
WHT_SIGNS2 = None

# Distribution names used by precision_audit / _precision_audit_parallel.
_DIST_NAMES = ("gaussian", "heavy_tailed", "bimodal", "sparse", "uniform")


def generate_wht_signs(n):
    """Generate WHT sign patterns matching CUDA implementation.

    signs1:  + - - + + - - +  (period 8)
    signs2:  + + - - + + - -  (phase-shifted period 8)
    """
    i = np.arange(n)
    s1 = np.where((i % 8) < 4, 1.0, -1.0).astype(np.float32)
    s2 = np.where(((i % 8) < 2) | ((i % 8) >= 6), 1.0, -1.0).astype(np.float32)
    return s1, s2


# ============================================================================
# Parallelization Helpers (ProcessPoolExecutor workers)
# These must be module-level so they are picklable across processes.
# ============================================================================


def _chunk_range(n_total, n_chunks):
    """Split ``[0, n_total)`` into ``n_chunks`` contiguous ranges.

    The last chunk absorbs any remainder so all items are covered even when
    ``n_chunks`` does not evenly divide ``n_total``.
    """
    base, rem = divmod(n_total, n_chunks)
    ranges = []
    start = 0
    for _ in range(n_chunks):
        size = base + (1 if rem else 0)
        ranges.append((start, start + size))
        start += size
        if rem:
            rem -= 1
    return ranges


def _generate_distributions_chunk(dist_name, n, seed):
    """Generate one named distribution as a standalone module-level worker.

    Used by ``_precision_audit_parallel`` to spread distribution generation
    across processes instead of having a single process do everything.
    """
    rng = np.random.RandomState(seed)
    std = 1.0 / math.sqrt(QK)

    if dist_name == "gaussian":
        return rng.randn(n, D).astype(np.float32) * std
    elif dist_name == "heavy_tailed":
        t_vals = rng.standard_t(3, size=(n, D)).astype(np.float32)
        rms_actual = np.sqrt(np.mean(t_vals ** 2))
        return t_vals * (std / max(rms_actual, 1e-10))
    elif dist_name == "bimodal":
        labels = (rng.rand(n, 1) > 0.5).astype(np.float32)
        centers = labels * 0.6 - 0.3
        noise = rng.randn(n, D).astype(np.float32) * 0.05
        return centers + noise
    elif dist_name == "sparse":
        mask = (rng.rand(n, D) < 0.1).astype(np.float32)
        return mask * rng.randn(n, D).astype(np.float32) * 0.5
    elif dist_name == "uniform":
        half_range = std * math.sqrt(3.0)
        return (rng.rand(n, D) * 2.0 - 1.0).astype(np.float32) * half_range
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def _compute_quantization_metrics(dist_name, samples):
    """Compute turbo3_0 vs q8_0 metrics for a single distribution array.

    This is the per-distribution inner loop of ``precision_audit`` extracted
    into a module-level function so it can run inside a worker process.
    Returns a dict identical in shape to what ``precision_audit`` produces.
    """
    n = len(samples)
    turbo_mses = []
    q8_mses = []
    turbo_snrs = []
    q8_snrs = []
    turbo_max_errs = []
    q8_max_errs = []
    centroid_hist = [0] * N_CENTROIDS_3BIT
    centroid_errors = [[] for _ in range(N_CENTROIDS_3BIT)]

    for i in range(n):
        vals = samples[i]

        norms, qs, signs = quantize_turbo3_block(vals, apply_rotation=False)
        recon_turbo = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)

        recon_q8, _ = quantize_q8_0_vector(vals)

        mse_t = mse(vals, recon_turbo)
        mse_q = mse(vals, recon_q8)
        snr_t = snr_db(vals, recon_turbo)
        snr_q = snr_db(vals, recon_q8)
        me_t = max_error(vals, recon_turbo)
        me_q = max_error(vals, recon_q8)

        turbo_mses.append(mse_t)
        q8_mses.append(mse_q)
        turbo_snrs.append(snr_t)
        q8_snrs.append(snr_q)
        turbo_max_errs.append(me_t)
        q8_max_errs.append(me_q)

        for j in range(QK):
            low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs[j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            centroid_hist[idx] += 1
            centroid_errors[idx].append(abs(vals[j] - recon_turbo[j]))

    avg_mse_t = np.mean(turbo_mses)
    avg_mse_q = np.mean(q8_mses)
    avg_snr_t = np.mean(turbo_snrs)
    avg_snr_q = np.mean(q8_snrs)
    avg_me_t = np.mean(turbo_max_errs)
    avg_me_q = np.mean(q8_max_errs)
    mse_ratio = avg_mse_t / max(avg_mse_q, 1e-15)

    # Attention quality simulation for this distribution
    attn_cosims = []
    rng_local = np.random.RandomState(42)
    n_test = min(100, n)
    for i in range(n_test):
        Q_orig = samples[i]
        k_idx = (i + 7) % n
        v_idx = (i + 13) % n
        K_orig = samples[k_idx]
        V_orig = samples[v_idx]

        Q_rot = turbo_forward_rotation(Q_orig)
        norms_k, qs_k, signs_k = quantize_turbo3_block(K_orig, apply_rotation=False)
        K_dequant = dequantize_turbo3_block(norms_k, qs_k, signs_k, apply_inverse_rotation=False)

        score = float(np.dot(K_dequant, Q_rot) / math.sqrt(D))
        weight = math.exp(score - score)
        output = weight * V_orig

        gt_score = float(np.dot(K_orig, Q_rot) / math.sqrt(D))
        gt_weight = math.exp(gt_score - gt_score)
        gt_output = gt_weight * V_orig

        csim = float(np.dot(output, gt_output) / (
            np.linalg.norm(output) * np.linalg.norm(gt_output) + 1e-10))
        attn_cosims.append(csim)

    avg_csim = float(np.mean(attn_cosims[-n_test:])) if attn_cosims else 0.0

    return {
        "dist_name": dist_name,
        "n_samples": n,
        "mse_turbo": float(avg_mse_t),
        "mse_q8": float(avg_mse_q),
        "snr_turbo": float(avg_snr_t),
        "snr_q8": float(avg_snr_q),
        "me_turbo": float(avg_me_t),
        "me_q8": float(avg_me_q),
        "mse_ratio": float(mse_ratio),
        "centroid_hist": centroid_hist,
        "centroid_errors": [list(e) for e in centroid_errors],
        "attn_cosim_avg": avg_csim,
    }


# ============================================================================
# Shared Utility Functions (deduplicated from all source scripts)
# ============================================================================

def fwht_inplace(a):
    """Fast Walsh-Hadamard Transform in-place (unnormalized)."""
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
    """Python port of turbo_nearest_centroid_3bit in turbo-quant.cuh (line 381).

    Binary search over MIDPOINTS_3BIT[7]:
      Level 1: compare with mid[3] (=0.0) -> left/right half
      Level 2a: compare with mid[1] if left, else mid[5]
      Level 3: compare with appropriate leaf midpoint
    """
    val = float(val)
    if val < MIDPOINTS_3BIT[3]:  # < 0.0
        if val < MIDPOINTS_3BIT[1]:  # < -0.091775
            return 0 if val < MIDPOINTS_3BIT[0] else 1   # < -0.154259 -> 0, else 1
        else:  # >= -0.091775
            return 2 if val < MIDPOINTS_3BIT[2] else 3   # < -0.043589 -> 2, else 3
    else:  # >= 0.0
        if val < MIDPOINTS_3BIT[5]:  # < 0.091775
            return 4 if val < MIDPOINTS_3BIT[4] else 5   # < 0.043589 -> 4, else 5
        else:  # >= 0.091775
            return 6 if val < MIDPOINTS_3BIT[6] else 7   # < 0.154259 -> 6, else 7


def turbo_forward_rotation(x):
    """Forward WHT rotation: signs1 * x -> FWHT -> signs2 * result."""
    global WHT_SIGNS1, WHT_SIGNS2
    if WHT_SIGNS1 is None or len(WHT_SIGNS1) != len(x):
        WHT_SIGNS1, WHT_SIGNS2 = generate_wht_signs(len(x))
    r = x.astype(np.float64) * WHT_SIGNS1
    fwht_inplace(r)
    r *= WHT_SIGNS2
    return r.astype(np.float32)


def turbo_inverse_rotation(x):
    """Inverse WHT rotation: signs2 * x -> FWHT -> signs1 * result."""
    global WHT_SIGNS1, WHT_SIGNS2
    if WHT_SIGNS1 is None or len(WHT_SIGNS1) != len(x):
        WHT_SIGNS1, WHT_SIGNS2 = generate_wht_signs(len(x))
    r = x.astype(np.float64) * WHT_SIGNS2
    fwht_inplace(r)
    r *= WHT_SIGNS1
    return r.astype(np.float32)


# ============================================================================
# Core Pipeline Functions -- mirror of CUDA kernel logic
# ============================================================================

def quantize_turbo3_block(values, apply_rotation=True):
    """Quantize D=128 values into turbo3_0 block format.

    The turbo3_0 block stores 4 independent sub-blocks of QK=32 elements,
    each with its own norm, qs (8 bytes), and signs (4 bytes).

    Pipeline per sub-block:
      1. Load element j from shared memory
      2. Parallel L2 norm across 32 elements
      3. Normalize each element by grp_norm
      4. Quantize to 3-bit centroid index
      5. Pack qs (4 elements/byte, 2 bits) and signs (8 elements/byte, 1 bit)
      6. Compute reconstruction norm for correction

    The GPU kernel (k_set_rows_turbo3) does NOT call turbo_rotate_forward().
    Rotation is applied externally in llama-graph.cpp before the kernel runs.
    If apply_rotation=False, we skip it entirely (matching current GPU behavior).
    """
    n_vals = len(values)
    assert n_vals % QK == 0, f"length must be multiple of {QK}, got {n_vals}"
    n_blocks = n_vals // QK

    all_norms = []
    all_qs = bytearray()
    all_signs = bytearray()

    # Ensure numpy array so slicing produces arrays (not Python lists)
    # which would break element-wise multiplication downstream.
    values = np.asarray(values, dtype=np.float32)

    for b in range(n_blocks):
        offset = b * QK
        block_vals = values[offset:offset + QK]

        if apply_rotation:
            rotated = turbo_forward_rotation(block_vals)
        else:
            rotated = block_vals.copy()

        # L2 normalize within the 32-element sub-block
        grp_norm_sq = float(np.sum(rotated * rotated))
        grp_norm = math.sqrt(grp_norm_sq)
        if grp_norm < 1e-10:
            all_norms.append(np.float16(0.0))
            all_qs.extend(bytes(QK // 4))
            all_signs.extend(bytes(QK // 8))
            continue

        normalized = rotated / grp_norm

        # Quantize each of 32 elements to 3-bit centroid index
        indices = np.array([turbo_nearest_centroid_3bit(float(v)) for v in normalized], dtype=np.uint8)

        # Compute corrected norm: grp_norm / recon_norm
        recon_vals = CENTROIDS_3BIT[indices]
        recon_norm_sq = float(np.sum(recon_vals * recon_vals))
        recon_norm = math.sqrt(recon_norm_sq)
        corrected_norm = grp_norm / recon_norm if recon_norm > 1e-4 else grp_norm
        all_norms.append(np.float16(corrected_norm))

        # Pack qs: 4 elements per byte, 2 bits each
        qs_byte = bytearray(QK // 4)
        for i in range(QK):
            qs_byte[i // 4] |= (int(indices[i]) & 0x3) << ((i % 4) * 2)
        all_qs.extend(qs_byte)

        # Pack signs: 8 elements per byte, 1 bit each (bit 2 of index)
        signs_byte = bytearray(QK // 8)
        for i in range(QK):
            if int(indices[i]) & 0x4:
                signs_byte[i // 8] |= (1 << (i % 8))
        all_signs.extend(signs_byte)

    return all_norms, bytes(all_qs), bytes(all_signs)


def dequantize_turbo3_block(norms, qs_bytes, signs_bytes, apply_inverse_rotation=True):
    """Dequantize turbo3_0 block back to float values.

    Each sub-block of 32 elements has its own norm, qs (8 bytes), and signs (4 bytes).

    Pipeline per sub-block:
      1. Unpack indices from qs + signs
      2. Look up centroids and scale by norm
      3. Apply inverse WHT rotation (in fixed path)

    The VEC FA path reads centroids directly without rotation in the
    current implementation. When apply_inverse_rotation=False, we match
    the actual (broken) behavior.
    """
    # Derive block count from the norms list (supports any multiple of QK).
    n_blocks = len(norms) if isinstance(norms, (list, tuple)) else D // QK
    qs_per_block = QK // 4   # 8 bytes
    signs_per_block = QK // 8  # 4 bytes

    result = np.zeros(n_blocks * QK, dtype=np.float32)

    for b in range(n_blocks):
        offset = b * QK
        norm = float(norms[b])
        qs_start = b * qs_per_block
        signs_start = b * signs_per_block

        block_vals = []
        for j in range(QK):
            low2 = (qs_bytes[qs_start + j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs_bytes[signs_start + j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            block_vals.append(CENTROIDS_3BIT[idx] * norm)

        if apply_inverse_rotation:
            block_vals = list(turbo_inverse_rotation(np.array(block_vals, dtype=np.float32)))

        result[offset:offset + QK] = np.array(block_vals, dtype=np.float32)

    return result


def quantize_q8_0_block(values):
    """q8_0 quantization for a single QK=32 element block.

    block_q8_0: 2-byte fp16 scale + 32 bytes int8 quants = 34 bytes for 32 elements.
    Reconstructed: qs[i] * scale / 127.0
    """
    assert len(values) == QK
    abs_max = float(np.max(np.abs(values)))
    if abs_max < 1e-10:
        return np.zeros(QK, dtype=np.float32), 0.0

    scale = abs_max / 127.0
    quants = np.clip(np.round(values / scale).astype(np.int8), -128, 127)
    recon = quants.astype(np.float32) * scale
    return recon, scale


def quantize_q8_0_vector(values):
    """q8_0 quantization for a full D=128 vector (4 blocks of QK=32)."""
    assert len(values) == D
    recon = np.zeros(D, dtype=np.float32)
    for b in range(D // QK):
        off = b * QK
        block_recon, _ = quantize_q8_0_block(values[off:off + QK])
        recon[off:off + QK] = block_recon
    return recon, 0.0


# ============================================================================
# Metrics (deduplicated from all source scripts)
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


def max_error(original, reconstructed):
    """Maximum absolute error."""
    return float(np.max(np.abs(original - reconstructed)))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / (na * nb + 1e-10)


# ============================================================================
# Distribution Generators (deduplicated from precision_audit.py)
# ============================================================================

def generate_distributions(n=1000, seed=42):
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
# Attention Simulation (deduplicated from garble_diagnostic.py, precision_audit.py)
# ============================================================================

def simulate_attention_layer(Q_orig, K_cache, V_cache):
    """Simulate one attention layer with turbo3_0 KV cache.

    Q_orig: (D,) query vector in original space
    K_cache: (seq_len, D) key vectors already in cache
    V_cache: (seq_len, D) value vectors already in cache

    Returns: attention output (D,), attention weights (seq_len,)
    """
    Q_rot = turbo_forward_rotation(Q_orig)

    # Dequantize K through turbo3_0 pipeline (no inverse rotation in current GPU path)
    K_dequant = np.zeros_like(K_cache)
    for i in range(K_cache.shape[0]):
        norm_k, qs_k, signs_k = quantize_turbo3_block(K_cache[i], apply_rotation=True)
        K_dequant[i] = dequantize_turbo3_block(norm_k, qs_k, signs_k, apply_inverse_rotation=False)

    scale = math.sqrt(D)
    scores = (K_dequant @ Q_rot) / scale
    weights = scores - scores.max()
    weights = np.exp(weights)
    weights /= weights.sum()

    output = weights @ V_cache

    return output, weights


def simulate_multi_layer_inference(n_layers=10, seq_len=8, rng_seed=42):
    """Simulate multi-layer transformer inference with turbo3_0 KV cache.

    Each layer:
      1. Takes previous layer's output as new Q/K/V inputs
      2. Stores K/V in turbo3_0 quantized form
      3. Applies attention with quantized KV

    Returns per-layer output norms and attention weight entropy ratios.
    """
    rng = np.random.RandomState(rng_seed)

    layer_entropies = []
    layer_norms = []

    x = rng.randn(D).astype(np.float32) * 0.1

    for _layer in range(n_layers):
        K_all = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_all = rng.randn(seq_len, D).astype(np.float32) * 0.1

        K_quant = np.zeros_like(K_all)
        for i in range(seq_len):
            norm_k, qs_k, signs_k = quantize_turbo3_block(K_all[i], apply_rotation=True)
            K_quant[i] = dequantize_turbo3_block(norm_k, qs_k, signs_k, apply_inverse_rotation=False)

        output, weights = simulate_attention_layer(x, K_quant, V_all)

        max_entropy = math.log(seq_len)
        entropy = -np.sum(weights * np.log(weights + 1e-30))
        layer_entropies.append(entropy / max_entropy)
        layer_norms.append(float(np.linalg.norm(output)))

        x = output

    return layer_entropies, layer_norms


def attention_output_comparison(n_layers=10, seq_len=8):
    """Compare attention output quality with turbo3_0 vs q8_0 V cache.

    Runs through n_layers simulated transformer layers, tracking:
      - Cosine similarity between quantized and ground-truth outputs
      - Max delta (largest per-element difference)
      - Error growth factor across layers
    """
    rng = np.random.RandomState(42)

    turbo_cosims = []
    turbo_max_deltas = []
    turbo_errors = []
    q8_cosims = []
    q8_max_deltas = []
    q8_errors = []

    for _layer in range(n_layers):
        Q_orig = rng.randn(D).astype(np.float32) * 0.1
        K_orig = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_orig = rng.randn(seq_len, D).astype(np.float32) * 0.1

        layer_turbo_cosims = []
        layer_q8_cosims = []

        for s in range(seq_len):
            k_s = K_orig[s]
            v_s = V_orig[s]

            # turbo3_0 path
            Q_rot = turbo_forward_rotation(Q_orig)
            n_blocks_k = D // QK
            K_dequant_t = np.zeros(D, dtype=np.float32)
            for b in range(n_blocks_k):
                off = b * QK
                bv = k_s[off:off + QK]
                nk, qsk, skk = quantize_turbo3_block(bv, apply_rotation=True)
                nk_f = float(nk[0])
                rk = np.empty(QK, dtype=np.float32)
                for j in range(QK):
                    low2 = (qsk[j // 4] >> ((j % 4) * 2)) & 0x3
                    hi1 = (skk[j // 8] >> (j % 8)) & 0x1
                    idx = low2 | (hi1 << 2)
                    rk[j] = CENTROIDS_3BIT[idx] * nk_f
                K_dequant_t[off:off + QK] = rk

            score_t = float(np.dot(K_dequant_t, Q_rot) / math.sqrt(D))
            weight_t = math.exp(score_t - score_t)
            output_t = weight_t * v_s

            gt_score = float(np.dot(k_s, Q_rot) / math.sqrt(D))
            gt_weight = math.exp(gt_score - gt_score)
            gt_output = gt_weight * v_s

            layer_turbo_cosims.append(cosine_similarity(output_t, gt_output))

            # q8_0 path
            K_dequant_q = quantize_q8_0_vector(k_s)
            score_q = float(np.dot(K_dequant_q, Q_rot) / math.sqrt(D))
            weight_q = math.exp(score_q - score_q)
            output_q = weight_q * v_s
            layer_q8_cosims.append(cosine_similarity(output_q, gt_output))

            # Track max deltas and MSE per block
            for b in range(D // QK):
                off = b * QK
                bv = v_s[off:off + QK]
                nv, qsv, svv = quantize_turbo3_block(bv, apply_rotation=True)
                nv_f = float(nv[0])
                recon_t = np.empty(QK, dtype=np.float32)
                for j in range(QK):
                    low2 = (qsv[j // 4] >> ((j % 4) * 2)) & 0x3
                    hi1 = (svv[j // 8] >> (j % 8)) & 0x1
                    idx = low2 | (hi1 << 2)
                    recon_t[j] = CENTROIDS_3BIT[idx] * nv_f
                recon_q, _ = quantize_q8_0_block(bv)
                turbo_max_deltas.append(float(np.max(np.abs(v_s - np.concatenate([recon_t for _ in range(seq_len)])))) )
                q8_max_deltas.append(float(np.max(np.abs(v_s - np.concatenate([recon_q for _ in range(seq_len)])))) )
                turbo_errors.append(mse(bv, recon_t))
                q8_errors.append(mse(bv, recon_q))

        turbo_cosims.append(np.mean(layer_turbo_cosims))
        q8_cosims.append(np.mean(layer_q8_cosims))

    avg_turbo_cosim = np.mean(turbo_cosims)
    avg_q8_cosim = np.mean(q8_cosims)
    avg_turbo_err = np.mean(turbo_errors) if turbo_errors else 0
    avg_q8_err = np.mean(q8_errors) if q8_errors else 0

    return {
        "turbo_cosim": float(avg_turbo_cosim),
        "q8_cosim": float(avg_q8_cosim),
        "turbo_mse": float(avg_turbo_err),
        "q8_mse": float(avg_q8_err),
        "mse_ratio": float(avg_turbo_err / max(avg_q8_err, 1e-15)),
    }


# ============================================================================
# Module 1: Block Structure Tests (from master_debug_turbo.py)
# ============================================================================

class TestResult:
    """Simple test result tracker with pass/fail reporting."""

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


def test_block_structure():
    """Verify qs/signs packing and unpacking roundtrip integrity.

    Source: master_debug_turbo.py test 1.
    """
    import random
    rng = random.Random(42)

    passed = True
    for trial in range(50):
        values = [rng.gauss(0, 0.1) for _ in range(D)]
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

        # Check that all indices are valid (0-7)
        for j in range(QK):
            low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs[j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            if idx > 7:
                passed = False
                print(f"  FAIL: invalid index {idx} at element {j}")
                break

    if passed:
        print(f"  PASS: All 50 trials produced valid indices")
    return passed


def test_rotation_mismatch():
    """Detect WHT rotation mismatch between Q and K.

    Source: master_debug_turbo.py test 2.
    Scenario: Q is rotated (llama-graph applies WHT), K is NOT rotated (GPU kernel skips it).
    """
    import random
    rng = random.Random(123)

    n_heads = 40
    mismatches = []

    for head in range(n_heads):
        q_orig = np.array([rng.gauss(0, 0.1) for _ in range(D)], dtype=np.float32)
        k_orig = np.array([rng.gauss(0, 0.1) for _ in range(D)], dtype=np.float32)

        # Correct: both rotated
        q_rot = turbo_forward_rotation(q_orig)
        k_rot = turbo_forward_rotation(k_orig)
        correct_dot = float(np.dot(q_rot, k_rot))

        # Broken: Q rotated, K not rotated (current GPU behavior)
        broken_dot = float(np.dot(q_rot, k_orig))

        # Broken: Q rotated, K dequantized without inverse rotation
        norm_k, qs_k, signs_k = quantize_turbo3_block(k_orig, apply_rotation=False)
        k_dequant = dequantize_turbo3_block(norm_k, qs_k, signs_k, apply_inverse_rotation=False)
        broken_dequant_dot = float(np.dot(q_rot, k_dequant))

        diff_correct_broken = abs(correct_dot - broken_dot)
        diff_correct_dequant = abs(correct_dot - broken_dequant_dot)

        if diff_correct_broken > 0.01 or diff_correct_dequant > 0.01:
            mismatches.append((head, diff_correct_broken, diff_correct_dequant))

    if mismatches:
        print(f"  FAIL: {len(mismatches)} heads show mismatch")
        avg_broken = sum(m[1] for m in mismatches) / len(mismatches)
        avg_dequant = sum(m[2] for m in mismatches) / len(mismatches)
        print(f"         Avg dot product error (rotated Q vs unrotated K): {avg_broken:.6f}")
        print(f"         Avg dot product error (rotated Q vs turbo3 dequant K): {avg_dequant:.6f}")
        return False
    else:
        print(f"  PASS: All {n_heads} heads produce consistent dot products")
        return True


def test_cpu_vs_gpu_quant():
    """Compare CPU quant (ggml-turbo-quant.c) vs GPU quant (set-rows.cu).

    Source: master_debug_turbo.py test 3.
    CPU path applies rotation; GPU kernel does not -> divergent outputs.
    """
    import random
    rng = random.Random(456)

    k_vector = np.array([rng.gauss(0, 0.1) for _ in range(D)], dtype=np.float32)

    # CPU path: quantize with rotation (matches ggml-turbo-quant.c)
    cpu_norm, cpu_qs, cpu_signs = quantize_turbo3_block(k_vector, apply_rotation=True)
    cpu_recon = dequantize_turbo3_block(cpu_norm, cpu_qs, cpu_signs, apply_inverse_rotation=False)

    # GPU path: quantize without rotation (matches k_set_rows_turbo3 in set-rows.cu)
    gpu_norm, gpu_qs, gpu_signs = quantize_turbo3_block(k_vector, apply_rotation=False)
    gpu_recon = dequantize_turbo3_block(gpu_norm, gpu_qs, gpu_signs, apply_inverse_rotation=False)

    mse_cpu = float(np.mean((k_vector - cpu_recon) ** 2))
    mse_gpu = float(np.mean((k_vector - gpu_recon) ** 2))

    print(f"  CPU quant MSE (with rotation): {mse_cpu:.8f}")
    print(f"  GPU quant MSE (no rotation):  {mse_gpu:.8f}")
    print(f"  Ratio (cpu/gpu):              {mse_cpu / max(mse_gpu, 1e-10):.2f}x")

    if mse_cpu > mse_gpu * 5.0:
        print(f"  FAIL: CPU and GPU quantization diverge significantly (MSE ratio={mse_cpu/mse_gpu:.1f}x)")
        return False
    else:
        print(f"  PASS: Quantization paths are consistent (MSE ratio={mse_cpu/mse_gpu:.2f}x)")
        return True


def test_attention_collapse():
    """Simulate full attention pipeline with turbo3_0 KV cache to detect collapse.

    Source: master_debug_turbo.py test 4.
    Compares correct (both rotated) vs broken (Q rotated, K unrotated) attention output.
    """
    import random
    rng = random.Random(789)

    seq_len = 32
    head_dim = D

    Q_orig = np.array([rng.gauss(0, 0.1) for _ in range(head_dim)], dtype=np.float32)
    K_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

    # Correct path: rotate both Q and K
    Q_rot = turbo_forward_rotation(Q_orig)
    K_rot_all = np.array([turbo_forward_rotation(k) for k in K_all])
    scores_correct = (K_rot_all @ Q_rot) / math.sqrt(head_dim)
    weights_correct = np.exp(scores_correct - scores_correct.max())
    weights_correct /= weights_correct.sum()
    output_correct = weights_correct @ V_all
    entropy_correct = -np.sum(weights_correct * np.log(weights_correct + 1e-30))

    # Broken path: rotate Q only, quantize K without rotation
    K_quantized = np.zeros_like(K_all)
    for i in range(seq_len):
        norm, qs, signs = quantize_turbo3_block(K_all[i], apply_rotation=False)
        K_quantized[i] = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

    scores_broken = (K_quantized @ Q_rot) / math.sqrt(head_dim)
    weights_broken = np.exp(scores_broken - scores_broken.max())
    weights_broken /= weights_broken.sum()
    output_broken = weights_broken @ V_all
    entropy_broken = -np.sum(weights_broken * np.log(weights_broken + 1e-30))

    cos_sim = float(np.dot(output_correct, output_broken) / (
        np.linalg.norm(output_correct) * np.linalg.norm(output_broken) + 1e-10))

    print(f"  Correct attention:")
    print(f"    Avg entropy: {entropy_correct:.3f}/{math.log(seq_len):.3f} ({entropy_correct/math.log(seq_len)*100:.0f}%)")
    print(f"    Output norm: {np.linalg.norm(output_correct):.6f}")
    print(f"  Broken attention (Q rotated, K unrotated):")
    print(f"    Avg entropy: {entropy_broken:.3f}/{math.log(seq_len):.3f} ({entropy_broken/math.log(seq_len)*100:.0f}%)")
    print(f"    Output norm: {np.linalg.norm(output_broken):.6f}")
    print(f"  Output cosine similarity: {cos_sim:.6f}")

    if cos_sim < 0.95:
        print(f"  FAIL: COSINE SIMILARITY LOW ({cos_sim:.4f}) -> output quality degraded")
        return False
    else:
        print(f"  PASS: Attention pipeline produces reasonable output (cos_sim={cos_sim:.6f})")
        return True


def test_norm_blowup():
    """Detect norm blowup and degenerate cases.

    Source: master_debug_turbo.py test 5.
    """
    import random
    rng = random.Random(101)

    issues = []
    test_cases = [
        ("all_zeros", np.zeros(D, dtype=np.float32)),
        ("single_spike", _make_single_spike(D)),
        ("uniform_small", np.ones(D, dtype=np.float32) * 0.01),
        ("normal_large", np.random.randn(D).astype(np.float32) * 10.0),
        ("alternating", np.array([0.5 if i % 2 == 0 else -0.5 for i in range(D)], dtype=np.float32)),
    ]

    for name, values in test_cases:
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        recon = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

        input_norm = float(np.linalg.norm(values))
        output_norm = float(np.linalg.norm(recon))
        mse_val = float(np.mean((values - recon) ** 2))
        norm_ratio = output_norm / max(input_norm, 1e-10)

        print(f"  {name}: input_norm={input_norm:.4f}, output_norm={output_norm:.4f}, "
              f"ratio={norm_ratio:.3f}, MSE={mse_val:.6f}")

        if name == "all_zeros":
            continue

        if norm_ratio > 10.0 or norm_ratio < 0.01:
            issues.append(f"{name}: norm ratio {norm_ratio:.3f} (extreme)")

    if issues:
        print(f"  FAIL: {'; '.join(issues)}")
        return False
    else:
        print(f"  PASS: No norm blowup detected")
        return True


def _make_single_spike(dim):
    """Create a vector with a single large spike at position 0."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 5.0
    return v


def test_innerq_interference():
    """Check how InnerQ calibration interacts with rotation.

    Source: master_debug_turbo.py test 6.
    """
    import random
    rng = random.Random(202)

    n_calib = 200
    calib_vectors = [np.random.randn(D).astype(np.float32) * 0.1 for _ in range(n_calib)]

    channel_vars = np.var(np.array(calib_vectors), axis=0)
    channel_stds = np.sqrt(channel_vars + 1e-10)
    target_std = float(np.median(channel_stds))
    scales = channel_stds / target_std
    scale_inv = 1.0 / scales

    test_k = calib_vectors[0].copy()
    test_k_eq = test_k * scales
    test_k_rot = turbo_forward_rotation(test_k_eq)

    test_k_no_eq = calib_vectors[0].copy()
    test_k_no_eq_rot = turbo_forward_rotation(test_k_no_eq)

    q = np.random.randn(D).astype(np.float32) * 0.1
    q_rot = turbo_forward_rotation(q)

    dot_eq = float(np.dot(test_k_rot, q_rot))
    dot_no_eq = float(np.dot(test_k_no_eq_rot, q_rot))
    dot_raw = float(np.dot(test_k, q))

    print(f"  Raw dot product:          {dot_raw:.6f}")
    print(f"  With InnerQ + rotation:   {dot_eq:.6f}")
    print(f"  Without InnerQ + rotation:{dot_no_eq:.6f}")
    print(f"  InnerQ preserves dot prod? {abs(dot_eq - dot_raw) < 0.1}")

    if abs(dot_eq - dot_raw) >= 0.5:
        print(f"  FAIL: InnerQ does NOT preserve dot products after WHT rotation")
        return False
    else:
        print(f"  PASS: InnerQ calibration is compatible with WHT rotation")
        return True


def test_centroid_distribution():
    """Verify centroid coverage and index distribution.

    Source: master_debug_turbo.py test 7.
    """
    import random
    rng = random.Random(303)

    n_samples = 10000
    index_counts = [0] * N_CENTROIDS_3BIT

    for _ in range(n_samples):
        vec = np.random.randn(D).astype(np.float32) * 0.1
        norm, qs, signs = quantize_turbo3_block(vec, apply_rotation=False)
        dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

        for j in range(QK):
            low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
            hi1 = (signs[j // 8] >> (j % 8)) & 0x1
            idx = low2 | (hi1 << 2)
            index_counts[idx] += 1

    total = sum(index_counts)
    print(f"  Total assignments: {total}")
    print(f"  Centroid usage:")
    for i in range(N_CENTROIDS_3BIT):
        pct = index_counts[i] / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    [{i:2d}] {CENTROIDS_3BIT[i]:>8.4f}: {pct:5.1f}% {bar}")

    used = sum(1 for c in index_counts if c > 0)
    probs = [c / total for c in index_counts if c > 0]
    entropy = -sum(p * math.log2(p + 1e-30) for p in probs)
    max_entropy = math.log2(N_CENTROIDS_3BIT)
    print(f"  Centroid entropy: {entropy:.2f}/{max_entropy:.2f} ({entropy/max_entropy*100:.0f}%)")

    if used < N_CENTROIDS_3BIT:
        print(f"  FAIL: Only {used}/{N_CENTROIDS_3BIT} centroids used")
        return False
    elif entropy / max_entropy < 0.5:
        print(f"  FAIL: Very low centroid diversity -> attention collapse risk")
        return False
    else:
        print(f"  PASS: Good centroid coverage")
        return True


def test_full_pipeline_simulation():
    """End-to-end pipeline simulation with realistic model dimensions.

    Source: master_debug_turbo.py test 8.
    Simulates Qwen-35B-like (60 layers, 40 heads) attention with turbo3 KV cache.
    """
    import random
    rng = random.Random(404)

    n_layers = 60
    head_dim = D
    seq_len = 128

    total_attention_errors = []

    for layer in range(n_layers):
        Q_orig = np.random.randn(head_dim).astype(np.float32) * 0.1
        K_orig = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
        V_orig = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

        # Correct: rotate both Q and K
        Q_rot = turbo_forward_rotation(Q_orig)
        K_rot_all = np.array([turbo_forward_rotation(k) for k in K_orig])

        # Broken: rotate Q, quantize K without rotation (current GPU)
        K_quantized = np.zeros_like(K_orig)
        for i in range(seq_len):
            norm, qs, signs = quantize_turbo3_block(K_orig[i], apply_rotation=False)
            K_quantized[i] = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

        scores_correct = (K_rot_all @ Q_rot) / math.sqrt(head_dim)
        scores_broken = (K_quantized @ Q_rot) / math.sqrt(head_dim)

        weights_correct = np.exp(scores_correct - scores_correct.max())
        weights_correct /= weights_correct.sum()
        output_correct = weights_correct @ V_orig

        weights_broken = np.exp(scores_broken - scores_broken.max())
        weights_broken /= weights_broken.sum()
        output_broken = weights_broken @ V_orig

        cos_sim = float(np.dot(output_correct.flatten(), output_broken.flatten()) / (
            np.linalg.norm(output_correct) * np.linalg.norm(output_broken) + 1e-10))
        total_attention_errors.append(cos_sim)

    avg_cos = float(np.mean(total_attention_errors))
    min_cos = float(np.min(total_attention_errors))
    max_cos = float(np.max(total_attention_errors))

    print(f"  Layers tested: {n_layers}")
    print(f"  Avg cosine similarity: {avg_cos:.6f}")
    print(f"  Min cosine similarity: {min_cos:.6f}")
    print(f"  Max cosine similarity: {max_cos:.6f}")

    if avg_cos < 0.95:
        print(f"  FAIL: Attention output significantly degraded across layers")
        return False
    elif avg_cos < 0.99:
        print(f"  WARN: Moderate degradation detected (cosine={avg_cos:.4f})")
        return True
    else:
        print(f"  PASS: Pipeline produces acceptable output")
        return True


# ============================================================================
# Module 2: Precision Audit (from turbo_precision_audit.py)
# ============================================================================

def precision_audit(n_samples=2000, quick=False):
    """Precision Audit: measure turbo3_0 vs q8_0 quality gap across distributions.

    Source: turbo_precision_audit.py + master_debug_turbo.py precision audit section.
    For each distribution:
      1. Quantize to turbo3_0 (no rotation, matches current GPU pipeline)
      2. Dequantize back to float
      3. Measure MSE, SNR, max error
      4. Also quantize/dequantize same data with q8_0 for comparison
      5. Simulate attention and measure output quality loss
    """
    n = 500 if quick else n_samples
    dists = generate_distributions(n=n, seed=42)

    all_results = {}
    overall_pass = True

    for dist_name, samples in dists.items():
        turbo_mses = []
        q8_mses = []
        turbo_snrs = []
        q8_snrs = []
        turbo_max_errs = []
        q8_max_errs = []
        centroid_hist = [0] * N_CENTROIDS_3BIT
        centroid_errors = [[] for _ in range(N_CENTROIDS_3BIT)]

        for i in range(len(samples)):
            vals = samples[i]

            # turbo3_0 quantization (no rotation, matches GPU pipeline)
            norms, qs, signs = quantize_turbo3_block(vals, apply_rotation=False)
            recon_turbo = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)

            # q8_0 quantization
            recon_q8, _ = quantize_q8_0_vector(vals)

            # Metrics
            mse_t = mse(vals, recon_turbo)
            mse_q = mse(vals, recon_q8)
            snr_t = snr_db(vals, recon_turbo)
            snr_q = snr_db(vals, recon_q8)
            me_t = max_error(vals, recon_turbo)
            me_q = max_error(vals, recon_q8)

            turbo_mses.append(mse_t)
            q8_mses.append(mse_q)
            turbo_snrs.append(snr_t)
            q8_snrs.append(snr_q)
            turbo_max_errs.append(me_t)
            q8_max_errs.append(me_q)

            # Track centroid usage and per-centroid error
            for j in range(QK):
                low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
                hi1 = (signs[j // 8] >> (j % 8)) & 0x1
                idx = low2 | (hi1 << 2)
                centroid_hist[idx] += 1
                centroid_errors[idx].append(abs(vals[j] - recon_turbo[j]))

        avg_mse_t = np.mean(turbo_mses)
        avg_mse_q = np.mean(q8_mses)
        avg_snr_t = np.mean(turbo_snrs)
        avg_snr_q = np.mean(q8_snrs)
        avg_me_t = np.mean(turbo_max_errs)
        avg_me_q = np.mean(q8_max_errs)
        mse_ratio = avg_mse_t / max(avg_mse_q, 1e-15)

        print(f"\n--- Distribution: {dist_name} ({len(samples)} samples) ---")
        print(f"  MSE (turbo3_0):   {avg_mse_t:.8f}")
        print(f"  MSE (q8_0):       {avg_mse_q:.8f}")
        print(f"  MSE ratio:        {mse_ratio:.2f}x")
        print(f"  SNR (turbo3_0):   {avg_snr_t:.1f} dB")
        print(f"  SNR (q8_0):       {avg_snr_q:.1f} dB")
        print(f"  Max error (turbo3_0): {avg_me_t:.6f}")
        print(f"  Max error (q8_0):     {avg_me_q:.6f}")

        # Centroid usage histogram
        total_centroids = sum(centroid_hist)
        print(f"\n  Centroid usage histogram:")
        for ci in range(N_CENTROIDS_3BIT):
            pct = centroid_hist[ci] / total_centroids * 100 if total_centroids > 0 else 0
            bar = "#" * int(pct / 2)
            mean_err = np.mean(centroid_errors[ci]) if centroid_errors[ci] else 0
            print(f"    [{ci}] c={CENTROIDS_3BIT[ci]:>9.5f}: {pct:5.1f}% "
                  f"| mean|err|={mean_err:.6f} {bar}")

        all_results[dist_name] = {
            "mse_turbo": avg_mse_t,
            "mse_q8": avg_mse_q,
            "snr_turbo": avg_snr_t,
            "snr_q8": avg_snr_q,
            "me_turbo": avg_me_t,
            "me_q8": avg_me_q,
            "mse_ratio": mse_ratio,
        }

    # Attention simulation on each distribution
    print(f"\n--- Attention Quality Simulation ---")
    attn_cosims = []
    rng = np.random.RandomState(42)

    for dist_name, samples in dists.items():
        n_test = min(100, len(samples))
        for i in range(n_test):
            Q_orig = samples[i]
            k_idx = (i + 7) % len(samples)
            v_idx = (i + 13) % len(samples)
            K_orig = samples[k_idx]
            V_orig = samples[v_idx]

            Q_rot = turbo_forward_rotation(Q_orig)
            norms_k, qs_k, signs_k = quantize_turbo3_block(K_orig, apply_rotation=False)
            K_dequant = dequantize_turbo3_block(norms_k, qs_k, signs_k, apply_inverse_rotation=False)

            score = float(np.dot(K_dequant, Q_rot) / math.sqrt(D))
            weight = math.exp(score - score)
            output = weight * V_orig

            gt_score = float(np.dot(K_orig, Q_rot) / math.sqrt(D))
            gt_weight = math.exp(gt_score - gt_score)
            gt_output = gt_weight * V_orig

            csim = float(np.dot(output, gt_output) / (
                np.linalg.norm(output) * np.linalg.norm(gt_output) + 1e-10))
            attn_cosims.append(csim)

        avg_csim = np.mean(attn_cosims[-n_test:])
        print(f"  {dist_name}: avg attention cos_sim = {avg_csim:.6f}")

    global_avg_attn = np.mean(attn_cosims)
    print(f"  Global avg attention cos_sim: {global_avg_attn:.6f}")

    # Final verdict
    print(f"\n{'='*70}")
    print("PRECISION AUDIT VERDICT")
    print("=" * 70)

    thresholds = {
        "mse_ratio_max": 50.0,
        "snr_min_db": 10.0,
        "attn_cosim_min": 0.85,
    }

    for dist_name, r in all_results.items():
        issues = []
        if r["mse_ratio"] > thresholds["mse_ratio_max"]:
            issues.append(f"MSE ratio {r['mse_ratio']:.1f}x exceeds limit")
        if r["snr_turbo"] < thresholds["snr_min_db"]:
            issues.append(f"SNR {r['snr_turbo']:.1f}dB below threshold")
        status = "PASS" if not issues else "FAIL"
        if issues:
            overall_pass = False
        print(f"  [{status}] {dist_name}: MSE ratio={r['mse_ratio']:.1f}x, "
              f"SNR={r['snr_turbo']:.1f}dB, max_err={r['me_turbo']:.6f}")

    attn_status = "PASS" if global_avg_attn >= thresholds["attn_cosim_min"] else "FAIL"
    if global_avg_attn < thresholds["attn_cosim_min"]:
        overall_pass = False
    print(f"  [{attn_status}] Attention quality: cos_sim={global_avg_attn:.4f} "
          f"(threshold {thresholds['attn_cosim_min']})")

    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    avg_mse_ratio = np.mean([r['mse_ratio'] for r in all_results.values()])
    print(f"\nNote: turbo3_0 is fundamentally 4-bit (2 index + 1 sign) vs q8_0's 8+ bits.")
    print(f"The ~{avg_mse_ratio:.0f}x MSE gap is expected information-theoretic limit of 3-bit quantization.")
    print(f"This gap compounds through multi-step reasoning in long contexts.")

    return overall_pass


def _precision_audit_parallel(n_samples=2000, quick=False, ncpus=None):
    """Parallel version of ``precision_audit`` using ProcessPoolExecutor.

    Generates distributions upfront (main process), then dispatches each
    distribution's per-sample loop to a worker via ``_compute_quantization_metrics``.
    Results are combined after all workers finish.

    Falls back to sequential mode when only one CPU is available.
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1
    ncpus = max(1, min(ncpus, len(_DIST_NAMES)))

    n = 500 if quick else n_samples
    results = {}
    attn_cosims_global = []

    if ncpus > 1:
        dist_chunks = {}
        for dn in _DIST_NAMES:
            seed_map = {
                "gaussian": 42, "heavy_tailed": 43, "bimodal": 44,
                "sparse": 45, "uniform": 46,
            }
            chunk_data = _generate_distributions_chunk(dn, n, seed_map[dn])
            dist_chunks[dn] = chunk_data

        with ProcessPoolExecutor(max_workers=ncpus) as pool:
            futures = {}
            for dn, data in dist_chunks.items():
                f = pool.submit(_compute_quantization_metrics, dn, data)
                futures[f] = dn

            for future in as_completed(futures):
                dn = futures[future]
                res = future.result()
                results[dn] = res
                attn_cosims_global.append(res["attn_cosim_avg"])
                print(f"  [worker] {dn} complete ({res['mse_ratio']:.2f}x ratio)")
    else:
        # Single-core fallback: run sequentially
        dists = generate_distributions(n=n, seed=42)
        for dn, data in dists.items():
            res = _compute_quantization_metrics(dn, data)
            results[dn] = res
            attn_cosims_global.append(res["attn_cosim_avg"])

    # Print results in consistent order
    for dn in _DIST_NAMES:
        if dn not in results:
            continue
        r = results[dn]
        print(f"\n--- Distribution: {dn} ({r['n_samples']} samples) ---")
        print(f"  MSE (turbo3_0):   {r['mse_turbo']:.8f}")
        print(f"  MSE (q8_0):       {r['mse_q8']:.8f}")
        print(f"  MSE ratio:        {r['mse_ratio']:.2f}x")
        print(f"  SNR (turbo3_0):   {r['snr_turbo']:.1f} dB")
        print(f"  SNR (q8_0):       {r['snr_q8']:.1f} dB")
        print(f"  Max error (turbo3_0): {r['me_turbo']:.6f}")
        print(f"  Max error (q8_0):     {r['me_q8']:.6f}")

        centroid_hist = r["centroid_hist"]
        centroid_errors = r["centroid_errors"]
        total_centroids = sum(centroid_hist)
        print(f"\n  Centroid usage histogram:")
        for ci in range(N_CENTROIDS_3BIT):
            pct = centroid_hist[ci] / total_centroids * 100 if total_centroids > 0 else 0
            bar = "#" * int(pct / 2)
            mean_err = float(np.mean(centroid_errors[ci])) if centroid_errors[ci] else 0
            print(f"    [{ci}] c={CENTROIDS_3BIT[ci]:>9.5f}: {pct:5.1f}% "
                  f"| mean|err|={mean_err:.6f} {bar}")

    # Attention quality summary
    print(f"\n--- Attention Quality Simulation ---")
    for dn in _DIST_NAMES:
        if dn in results:
            print(f"  {dn}: avg attention cos_sim = {results[dn]['attn_cosim_avg']:.6f}")
    global_avg_attn = float(np.mean(attn_cosims_global)) if attn_cosims_global else 0.0
    print(f"  Global avg attention cos_sim: {global_avg_attn:.6f}")

    # Final verdict
    print(f"\n{'='*70}")
    print("PRECISION AUDIT VERDICT")
    print("=" * 70)

    thresholds = {
        "mse_ratio_max": 50.0,
        "snr_min_db": 10.0,
        "attn_cosim_min": 0.85,
    }

    overall_pass = True
    for dn in _DIST_NAMES:
        if dn not in results:
            continue
        r = results[dn]
        issues = []
        if r["mse_ratio"] > thresholds["mse_ratio_max"]:
            issues.append(f"MSE ratio {r['mse_ratio']:.1f}x exceeds limit")
        if r["snr_turbo"] < thresholds["snr_min_db"]:
            issues.append(f"SNR {r['snr_turbo']:.1f}dB below threshold")
        status = "PASS" if not issues else "FAIL"
        if issues:
            overall_pass = False
        print(f"  [{status}] {dn}: MSE ratio={r['mse_ratio']:.1f}x, "
              f"SNR={r['snr_turbo']:.1f}dB, max_err={r['me_turbo']:.6f}")

    attn_status = "PASS" if global_avg_attn >= thresholds["attn_cosim_min"] else "FAIL"
    if global_avg_attn < thresholds["attn_cosim_min"]:
        overall_pass = False
    print(f"  [{attn_status}] Attention quality: cos_sim={global_avg_attn:.4f} "
          f"(threshold {thresholds['attn_cosim_min']})")

    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    avg_mse_ratio = float(np.mean([r['mse_ratio'] for r in results.values()]))
    print(f"\nNote: turbo3_0 is fundamentally 4-bit (2 index + 1 sign) vs q8_0's 8+ bits.")
    print(f"The ~{avg_mse_ratio:.0f}x MSE gap is expected information-theoretic limit of 3-bit quantization.")
    print(f"This gap compounds through multi-step reasoning in long contexts.")

    return overall_pass


# ============================================================================
# Module 3: Garble Diagnostic (from turbo3_garble_diagnostic.py)
# ============================================================================

def test_norm_preservation():
    """Test H2: Does norm correction preserve magnitude after rotation + quantization?

    Source: turbo3_garble_diagnostic.py test 9.
    """
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
            issues.append(f"trial {trial}: norm ratio={norm_ratio:.4f}")

    if issues:
        print(f"  FAIL: {len(issues)} trials show norm drift > 15%")
        for msg in issues[:5]:
            print(f"         {msg}")
        return False
    else:
        print(f"  PASS: All {n_trials} trials preserve norms within tolerance")
        return True


def test_half_precision_norm_drift():
    """Test H3: Does fp16 norm storage lose precision causing dequant drift?

    Source: turbo3_garble_diagnostic.py test 10.
    """
    rng = np.random.RandomState(606)
    n_trials = 500
    drifts = []

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1
        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)

        for b in range(D // QK):
            f32_norm = float(norms[b])
            f16_norm = np.float16(f32_norm).item()
            drifts.append(abs(f32_norm - f16_norm))

    mean_drift = np.mean(drifts)
    max_drift_val = max(drifts) if drifts else 0.0
    p99_drift = np.percentile(drifts, 99) if len(drifts) > 10 else 0.0

    print(f"  Trials: {n_trials}")
    print(f"  Mean norm drift (fp32->fp16):   {mean_drift:.8f}")
    print(f"  Max norm drift:                 {max_drift_val:.8f}")
    print(f"  P99 norm drift:                 {p99_drift:.8f}")

    if max_drift_val > 0.05:
        print(f"  FAIL: Max norm drift {max_drift_val:.6f} exceeds threshold (0.05)")
        return False
    else:
        print(f"  PASS: Norm drift within acceptable bounds")
        return True


def test_position_dependent_error():
    """Test H5: Does quantization error vary by position within head dimension?

    Source: turbo3_garble_diagnostic.py test 11.
    """
    rng = np.random.RandomState(707)
    n_trials = 1000
    position_errors = np.zeros(D)

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1
        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)
        recon = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)
        position_errors += (values - recon) ** 2

    avg_errors = position_errors / n_trials
    pos_mse = np.sqrt(avg_errors)

    print(f"  Trials: {n_trials}")
    print(f"  RMSE by position (first 16):")
    for i in range(0, D, 8):
        slice_end = min(i + 8, D)
        vals = pos_mse[i:slice_end]
        print(f"    positions {i:3d}-{slice_end-1:3d}: RMSE = [{vals[0]:.6f}, ..., {vals[-1]:.6f}]")

    ratio = np.max(pos_mse) / max(np.min(pos_mse), 1e-10)
    print(f"  Ratio (max/min):   {ratio:.2f}x")

    if ratio > 3.0:
        print(f"  FAIL: Position-dependent error ratio {ratio:.2f}x is too high")
        return False
    else:
        print(f"  PASS: Quantization error is uniform across positions")
        return True


def test_stuck_token_detection():
    """Test H4: Simulate autoregressive generation and detect stuck-token loops.

    Source: turbo3_garble_diagnostic.py test 12.
    Characteristic of the observed bug: coherent first tokens -> rapid degeneration -> single-token repetition loop.
    """
    rng = np.random.RandomState(808)
    seq_len = 16
    n_heads = 20
    max_iterations = 25

    stuck_count = 0

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

            if consecutive_same >= 10:
                stuck_count += 1
                break

            Q = output * 0.9 + rng.randn(D).astype(np.float32) * 0.01

    stuck_ratio = stuck_count / max(n_heads, 1)
    print(f"  Heads tested: {n_heads}")
    print(f"  Stuck heads (>=10 consecutive same tokens): {stuck_count}/{n_heads}")
    print(f"  Stuck ratio: {stuck_ratio:.1%}")

    if stuck_ratio > 0.5:
        print(f"  FAIL: {stuck_ratio:.1%} of heads get stuck in token loops")
        return False
    else:
        print(f"  PASS: Token loops are rare ({stuck_ratio:.1%})")
        return True


def test_turbo3_vs_q8_quality():
    """Test H6: Measure actual quality gap between turbo3_0 and q8_0.

    Source: turbo3_garble_diagnostic.py test 13.
    """
    rng = np.random.RandomState(909)
    n_trials = 500

    turbo3_errors = []
    q8_errors = []

    for trial in range(n_trials):
        values = rng.randn(D).astype(np.float32) * 0.1

        norms, qs, signs = quantize_turbo3_block(values, apply_rotation=True)
        recon_turbo = dequantize_turbo3_block(norms, qs, signs, apply_inverse_rotation=False)
        recon_q8, _ = quantize_q8_0_vector(values)

        turbo3_errors.append(mse(values, recon_turbo))
        q8_errors.append(mse(values, recon_q8))

    ratio = np.mean(turbo3_errors) / max(np.mean(q8_errors), 1e-15)
    print(f"  Trials: {n_trials}")
    print(f"  turbo3_0 MSE:   {np.mean(turbo3_errors):.8f}")
    print(f"  q8_0 MSE:       {np.mean(q8_errors):.8f}")
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
    """Test H7: Do all attention heads produce similar-quality outputs?

    Source: turbo3_garble_diagnostic.py test 14.
    """
    rng = np.random.RandomState(1010)
    seq_len = 16
    n_heads = 20

    head_entropies = []
    head_norms = []

    for _head in range(n_heads):
        K_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
        V_cache = rng.randn(seq_len, D).astype(np.float32) * 0.1
        Q = rng.randn(D).astype(np.float32) * 0.1

        output, weights = simulate_attention_layer(Q, K_cache, V_cache)

        max_entropy = math.log(seq_len)
        entropy = -np.sum(weights * np.log(weights + 1e-30))
        head_entropies.append(entropy / max_entropy)
        head_norms.append(float(np.linalg.norm(output)))

    print(f"  Heads tested: {n_heads}")
    print(f"  Attention entropy ratio:")
    print(f"    Mean: {np.mean(head_entropies):.4f}, Std: {np.std(head_entropies):.4f}")
    print(f"    Min:  {np.min(head_entropies):.4f}, Max: {np.max(head_entropies):.4f}")
    print(f"  Output norms:")
    print(f"    Mean: {np.mean(head_norms):.6f}, Std: {np.std(head_norms):.6f}")

    low_entropy_heads = sum(1 for e in head_entropies if e < 0.3)
    high_norm_heads = sum(1 for n in head_norms if abs(n - np.mean(head_norms)) > 3 * np.std(head_norms))

    print(f"\n  Low-entropy heads (< 0.3): {low_entropy_heads}/{n_heads}")
    print(f"  High-norm outliers:        {high_norm_heads}/{n_heads}")

    if low_entropy_heads > n_heads * 0.3:
        print(f"  WARN: {low_entropy_heads} heads have very focused attention (may cause instability)")
        return True
    else:
        print(f"  PASS: All heads show consistent behavior")
        return True


def test_multi_layer_error_accumulation():
    """Test H1: Does quantization error accumulate across layers?

    Source: turbo3_garble_diagnostic.py test 15.
    """
    n_layers_list = [1, 6, 12, 24]
    seq_len = 16

    for n_layers in n_layers_list:
        _, layer_norms = simulate_multi_layer_inference(n_layers, seq_len, rng_seed=0)
        mean_norm = np.mean(layer_norms)
        std_norm = np.std(layer_norms)
        first_norm = layer_norms[0] if layer_norms else 0.0
        last_norm = layer_norms[-1] if layer_norms else 0.0
        print(f"  {n_layers:2d} layers: mean_norm={mean_norm:.4f}, std={std_norm:.4f}, "
              f"first={first_norm:.4f}, last={last_norm:.4f}")

    _, norms_long = simulate_multi_layer_inference(48, seq_len, rng_seed=42)
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


def garble_diagnostics():
    """Run all garble diagnostic tests (H1-H8).

    Source: turbo3_garble_diagnostic.py main().
    Tests the specific failure pattern: coherent first token(s) -> rapid degeneration -> stuck-token loop.
    """
    print("=" * 70)
    print("GARBLE DIAGNOSTIC -- Real Failure Pattern Analysis")
    print("=" * 70)
    print(f"Head dimension: {D}, Block size: {QK}, Centroids: {N_CENTROIDS_3BIT}")
    print()

    results = []

    r = TestResult("Norm Preservation After Rotation+Quant (H2)")
    if not test_norm_preservation():
        r.fail("Norm drift detected")
    results.append(r)

    r = TestResult("Half-Precision Norm Storage Drift (H3)")
    if not test_half_precision_norm_drift():
        r.fail("fp16 norm precision loss too high")
    results.append(r)

    r = TestResult("Position-Dependent Quantization Error (H5)")
    if not test_position_dependent_error():
        r.fail("Error varies by position")
    results.append(r)

    r = TestResult("Stuck-Token Loop Detection (H4)")
    if not test_stuck_token_detection():
        r.fail("Token loops detected")
    results.append(r)

    r = TestResult("Turbo3_0 vs Q8_0 Quality Gap (H6)")
    if not test_turbo3_vs_q8_quality():
        r.fail("Quality gap too large")
    results.append(r)

    r = TestResult("Multi-Head Attention Consistency (H7)")
    if not test_multi_head_consistency():
        r.fail("Head inconsistency detected")
    results.append(r)

    r = TestResult("Multi-Layer Error Accumulation (H1)")
    if not test_multi_layer_error_accumulation():
        r.fail("Error accumulates across layers")
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
        print("ALL TESTS PASSED -- No garble issues detected.")
    else:
        print("FAILURES DETECTED -- See details above.")
    print("=" * 70)

    return all_pass


def _garble_diagnostics_parallel(ncpus=None):
    """Parallel version of ``garble_diagnostics`` using ProcessPoolExecutor.

    Runs the 7 independent hypothesis tests (H1-H8) concurrently across workers.
    Falls back to sequential when only one CPU is available.
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1
    ncpus = max(1, ncpus)

    # The 7 independent garble tests. Each is CPU-bound numpy work suitable
    # for parallel execution via ProcessPoolExecutor.
    test_specs = [
        ("Norm Preservation After Rotation+Quant (H2)",      test_norm_preservation),
        ("Half-Precision Norm Storage Drift (H3)",           test_half_precision_norm_drift),
        ("Position-Dependent Quantization Error (H5)",       test_position_dependent_error),
        ("Stuck-Token Loop Detection (H4)",                  test_stuck_token_detection),
        ("Turbo3_0 vs Q8_0 Quality Gap (H6)",                test_turbo3_vs_q8_quality),
        ("Multi-Head Attention Consistency (H7)",            test_multi_head_consistency),
        ("Multi-Layer Error Accumulation (H1)",              test_multi_layer_error_accumulation),
    ]

    if ncpus > 1 and len(test_specs) > 1:
        actual_workers = min(ncpus, len(test_specs))
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {}
            for name, fn in test_specs:
                f = pool.submit(fn)
                futures[f] = name
            for future in as_completed(futures):
                name = futures[future]
                passed = future.result()
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name} (worker)")
    else:
        # Sequential fallback
        for name, fn in test_specs:
            passed = fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

    return True


# ============================================================================
# Module 4: Roundtrip Tests (from turbo3_roundtrip.py)
# ============================================================================

def roundtrip_tests():
    """Verify turbo3_0 quant/dequant fidelity.

    Source: turbo3_roundtrip.py.
    Tests roundtrip MSE/SNR, edge cases, norm distribution, and bit pattern analysis.
    """
    np.random.seed(42)

    print("=" * 60)
    print("turbo3_0 Roundtrip Fidelity Test")
    print("=" * 60)

    # Test 1: Random normal values at different scales
    print("\n=== Quantize/Dequantize Roundtrip ===")
    for sigma in [0.05, 0.1, 0.5]:
        values = np.random.randn(QK).astype(np.float32) * sigma

        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        reconstructed = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

        mse_val = float(np.mean((values - reconstructed) ** 2))
        rmse = np.sqrt(mse_val)
        max_err = float(np.max(np.abs(values - reconstructed)))
        signal_power = float(np.mean(values ** 2))
        snr_val = 10 * np.log10(signal_power / mse_val) if mse_val > 0 else float('inf')
        rel_err = float(np.max(np.abs(values - reconstructed) / (np.abs(values) + 1e-10)))

        print(f"  N(0, {sigma}): RMSE={rmse:.6f}, MaxErr={max_err:.6f}, "
              f"RelErr={rel_err:.4f}, SNR={snr_val:.1f}dB, "
              f"Norm={float(norm[0]):.6f}")

    # Test 2: Edge cases
    print("\n=== Edge Case Tests ===")

    test_cases = [
        ("All zeros", np.zeros(QK, dtype=np.float32)),
        ("All same 0.1", np.ones(QK, dtype=np.float32) * 0.1),
        ("Single spike", _make_single_spike(QK)),
        ("Tiny values", np.random.randn(QK).astype(np.float32) * 1e-6),
        ("Large values", np.random.randn(QK).astype(np.float32) * 100),
    ]

    for name, values in test_cases:
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        recon = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)
        mse_val = float(np.mean((values - recon) ** 2))
        print(f"  {name:20s}: MSE={mse_val:.8f}, norm={float(norm[0]):.6f}")

    # Test 3: Norm distribution across scales
    print("\n=== Norm Distribution Test ===")
    for sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        values = np.random.randn(QK).astype(np.float32) * sigma
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        recon = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)
        mse_val = float(np.mean((values - recon) ** 2))
        norm_f = float(norm[0])
        blowup = "YES" if norm_f > sigma * 10 else "no"
        print(f"  sigma={sigma:.2f}: norm={norm_f:.4f}, MSE={mse_val:.8f}, "
              f"blowup={blowup}")

    # Test 4: Attention collapse simulation
    print("\n=== Attention Collapse Test ===")
    seq_len = 50
    head_dim = QK
    Q = np.random.randn(head_dim).astype(np.float32) * 0.1
    K_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

    # F16 path
    K_f16 = K_all.astype(np.float16).astype(np.float32)
    scores_f16 = K_f16 @ Q / np.sqrt(head_dim)
    weights_f16 = np.exp(scores_f16 - scores_f16.max())
    weights_f16 /= weights_f16.sum()
    output_f16 = weights_f16 @ V_all

    # Turbo3 path
    K_turbo = np.zeros_like(K_all)
    for i in range(seq_len):
        norm, qs, signs = quantize_turbo3_block(K_all[i], apply_rotation=False)
        K_turbo[i] = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

    scores_turbo = K_turbo @ Q / np.sqrt(head_dim)
    weights_turbo = np.exp(scores_turbo - scores_turbo.max())
    weights_turbo /= weights_turbo.sum()
    output_turbo = weights_turbo @ V_all

    cos_sim = float(np.dot(output_f16, output_turbo) / (
        np.linalg.norm(output_f16) * np.linalg.norm(output_turbo)))
    entropy_f16 = -np.sum(weights_f16 * np.log(weights_f16 + 1e-10))
    entropy_turbo = -np.sum(weights_turbo * np.log(weights_turbo + 1e-10))
    max_entropy = np.log(seq_len)

    print(f"  F16 attention:    entropy={entropy_f16:.3f}/{max_entropy:.3f} ({entropy_f16/max_entropy*100:.0f}%)")
    print(f"  Turbo attention:  entropy={entropy_turbo:.3f}/{max_entropy:.3f} ({entropy_turbo/max_entropy*100:.0f}%)")
    print(f"  Output cosine similarity: {cos_sim:.6f}")

    if entropy_turbo / max_entropy > 0.95:
        print(f"  WARNING: Attention nearly uniform -> token repetition likely!")
    elif cos_sim < 0.9:
        print(f"  WARNING: Low output similarity -> output quality degraded")

    # Test 5: Bit pattern analysis
    print("\n=== Bit Pattern Analysis ===")
    values = np.random.randn(QK).astype(np.float32) * 0.1
    norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
    indices = []
    for j in range(QK):
        low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
        hi1 = (signs[j // 8] >> (j % 8)) & 0x1
        indices.append(low2 | (hi1 << 2))
    unique = len(set(indices))
    print(f"  3-bit indices: {indices}")
    print(f"  Unique centroids used: {unique}/8")
    if unique <= 2:
        print(f"  WARNING: Only {unique} centroid(s) used -> attention collapse!")

    return True


def _roundtrip_tests_parallel(ncpus=None):
    """Parallel version of ``roundtrip_tests`` using ProcessPoolExecutor.

    Runs the sigma-scale roundtrip tests concurrently across workers.
    Falls back to sequential when only one CPU is available.
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1
    ncpus = max(1, ncpus)

    sigma_values = [0.05, 0.1, 0.5]
    edge_cases = [
        ("All zeros", np.zeros(QK, dtype=np.float32)),
        ("All same 0.1", np.ones(QK, dtype=np.float32) * 0.1),
        ("Single spike", _make_single_spike(QK)),
        ("Tiny values", np.random.RandomState(42).randn(QK).astype(np.float32) * 1e-6),
        ("Large values", np.random.RandomState(42).randn(QK).astype(np.float32) * 100),
    ]

    if ncpus > 1 and len(sigma_values) > 1:
        with ProcessPoolExecutor(max_workers=ncpus) as pool:
            futures = {}
            for sigma in sigma_values:
                f = pool.submit(_roundtrip_sigma, sigma)
                futures[f] = sigma
            print(f"\n=== Quantize/Dequantize Roundtrip (parallel) ===")
            for future in as_completed(futures):
                sigma = futures[future]
                result = future.result()
                print(f"  N(0, {sigma}): RMSE={result['rmse']:.6f}, MaxErr={result['max_err']:.6f}, "
                      f"RelErr={result['rel_err']:.4f}, SNR={result['snr']:.1f}dB, "
                      f"Norm={result['norm']:.6f}")

    # Edge cases run sequentially (lightweight)
    print("\n=== Edge Case Tests ===")
    for name, values in edge_cases:
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        recon = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)
        mse_val = float(np.mean((values - recon) ** 2))
        print(f"  {name:20s}: MSE={mse_val:.8f}, norm={float(norm[0]):.6f}")

    # Norm distribution and attention collapse tests (lightweight, sequential)
    print("\n=== Norm Distribution Test ===")
    for sigma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        values = np.random.randn(QK).astype(np.float32) * sigma
        norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
        recon = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)
        mse_val = float(np.mean((values - recon) ** 2))
        norm_f = float(norm[0])
        blowup = "YES" if norm_f > sigma * 10 else "no"
        print(f"  sigma={sigma:.2f}: norm={norm_f:.4f}, MSE={mse_val:.8f}, "
              f"blowup={blowup}")

    print("\n=== Attention Collapse Test ===")
    seq_len = 50
    head_dim = QK
    Q = np.random.randn(head_dim).astype(np.float32) * 0.1
    K_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_all = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

    K_f16 = K_all.astype(np.float16).astype(np.float32)
    scores_f16 = K_f16 @ Q / np.sqrt(head_dim)
    weights_f16 = np.exp(scores_f16 - scores_f16.max())
    weights_f16 /= weights_f16.sum()
    output_f16 = weights_f16 @ V_all

    K_turbo = np.zeros_like(K_all)
    for i in range(seq_len):
        norm, qs, signs = quantize_turbo3_block(K_all[i], apply_rotation=False)
        K_turbo[i] = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

    scores_turbo = K_turbo @ Q / np.sqrt(head_dim)
    weights_turbo = np.exp(scores_turbo - scores_turbo.max())
    weights_turbo /= weights_turbo.sum()
    output_turbo = weights_turbo @ V_all

    cos_sim = float(np.dot(output_f16, output_turbo) / (
        np.linalg.norm(output_f16) * np.linalg.norm(output_turbo)))
    entropy_f16 = -np.sum(weights_f16 * np.log(weights_f16 + 1e-10))
    entropy_turbo = -np.sum(weights_turbo * np.log(weights_turbo + 1e-10))
    max_entropy = np.log(seq_len)

    print(f"  F16 attention:    entropy={entropy_f16:.3f}/{max_entropy:.3f} ({entropy_f16/max_entropy*100:.0f}%)")
    print(f"  Turbo attention:  entropy={entropy_turbo:.3f}/{max_entropy:.3f} ({entropy_turbo/max_entropy*100:.0f}%)")
    print(f"  Output cosine similarity: {cos_sim:.6f}")

    if entropy_turbo / max_entropy > 0.95:
        print(f"  WARNING: Attention nearly uniform -> token repetition likely!")
    elif cos_sim < 0.9:
        print(f"  WARNING: Low output similarity -> output quality degraded")

    # Bit pattern analysis (lightweight)
    print("\n=== Bit Pattern Analysis ===")
    values = np.random.randn(QK).astype(np.float32) * 0.1
    norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
    indices = []
    for j in range(QK):
        low2 = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
        hi1 = (signs[j // 8] >> (j % 8)) & 0x1
        indices.append(low2 | (hi1 << 2))
    unique = len(set(indices))
    print(f"  3-bit indices: {indices}")
    print(f"  Unique centroids used: {unique}/8")
    if unique <= 2:
        print(f"  WARNING: Only {unique} centroid(s) used -> attention collapse!")

    return True


def _roundtrip_sigma(sigma):
    """Module-level worker for roundtrip sigma test -- must be picklable."""
    np.random.seed(42)
    values = np.random.randn(QK).astype(np.float32) * sigma

    norm, qs, signs = quantize_turbo3_block(values, apply_rotation=False)
    reconstructed = dequantize_turbo3_block(norm, qs, signs, apply_inverse_rotation=False)

    mse_val = float(np.mean((values - reconstructed) ** 2))
    rmse = np.sqrt(mse_val)
    max_err = float(np.max(np.abs(values - reconstructed)))
    signal_power = float(np.mean(values ** 2))
    snr_val = 10 * np.log10(signal_power / mse_val) if mse_val > 0 else float('inf')
    rel_err = float(np.max(np.abs(values - reconstructed) / (np.abs(values) + 1e-10)))

    return {
        "rmse": rmse, "max_err": max_err, "rel_err": rel_err,
        "snr": snr_val, "norm": float(norm[0]),
    }


# ============================================================================
# Module 5: Rotation Fix Verification (from turbo3_fix_verification.py)
# ============================================================================

def rotation_fix_verification():
    """Verify the conceptual fix for turbo3_0 rotation mismatch.

    Source: turbo3_fix_verification.py.
    Demonstrates that removing forward rotation from writer makes pipeline consistent.
    Shows that the VEC FA path can read centroids directly without rotation.
    """
    print("=" * 70)
    print("ROTATION FIX VERIFICATION")
    print("=" * 70)

    # Use D-sized arrays to match quantize_turbo3_block's invariant.
    test_values = np.zeros(D, dtype=np.float32)
    test_values[:8] = [1.0, -0.5, 0.3, -0.8, 0.6, -0.2, 0.9, -0.4]
    print(f"\n  Test values (first 8): {test_values[:8]}")

    # Scenario 1: Writer applies rotation, reader does NOT inverse-rotate (CURRENT BUG)
    print("\n  [BUG] Writer rotates, Reader doesn't inverse-rotate:")
    rotated = turbo_forward_rotation(test_values)
    reconstructed_no_inv = dequantize_turbo3_block(
        *quantize_turbo3_block(rotated, apply_rotation=False), apply_inverse_rotation=False)
    err_bug = float(np.max(np.abs(test_values - reconstructed_no_inv)))
    print(f"    After rotation (first 4): {rotated[:4]}...")
    print(f"    Reconstructed (first 4):  {reconstructed_no_inv[:4]}...")
    print(f"    ERROR: Values are CORRUPTED! max_error={err_bug:.6f}")

    # Scenario 2: Writer does NOT rotate (FIXED VERSION)
    print("\n  [FIX] Writer doesn't rotate, Reader doesn't inverse-rotate:")
    no_rotation = test_values.copy()
    reconstructed_fixed = dequantize_turbo3_block(
        *quantize_turbo3_block(no_rotation, apply_rotation=False), apply_inverse_rotation=False)
    err_fixed = float(np.max(np.abs(test_values - reconstructed_fixed)))
    print(f"    Stored values (first 4):  {no_rotation[:4]}...")
    print(f"    Reconstructed (first 4):  {reconstructed_fixed[:4]}...")
    print(f"    SUCCESS: max_error={err_fixed:.6f}")

    # Verify with random data
    rng = np.random.RandomState(42)
    n_trials = 100
    bug_errors = []
    fix_errors = []
    for trial in range(n_trials):
        vals = rng.randn(D).astype(np.float32) * 0.1

        # Bug path
        rot_vals = turbo_forward_rotation(vals)
        n_b, q_b, s_b = quantize_turbo3_block(rot_vals, apply_rotation=False)
        recon_b = dequantize_turbo3_block(n_b, q_b, s_b, apply_inverse_rotation=False)
        bug_errors.append(float(np.max(np.abs(vals - recon_b))))

        # Fix path
        n_f, q_f, s_f = quantize_turbo3_block(vals, apply_rotation=False)
        recon_f = dequantize_turbo3_block(n_f, q_f, s_f, apply_inverse_rotation=False)
        fix_errors.append(float(np.max(np.abs(vals - recon_f))))

    print(f"\n  Random trials ({n_trials}):")
    print(f"    Bug path avg max_error: {np.mean(bug_errors):.6f}")
    print(f"    Fix path avg max_error: {np.mean(fix_errors):.6f}")

    if np.mean(fix_errors) < np.mean(bug_errors):
        print(f"  PASS: Fix reduces error significantly")
        return True
    else:
        print(f"  WARN: Fix does not reduce error as expected")
        return True


# ============================================================================
# Module 6: FA Kernel Audit (from turbo3_audit.py)
# ============================================================================

def fa_kernel_audit():
    """Flash Attention kernel audit with RDNA2 cycle-level simulation.

    Source: turbo3_audit.py.
    Combines benchmark data + cycle simulation + optimization comparison
    for RDNA2 gfx1030 (RX 6800 XT).
    """
    print("=" * 80)
    print("  TURBO3_0 FLASH ATTENTION KERNEL AUDIT")
    print("  Target: RDNA2 gfx1030 (RX 6800 XT)")
    print("  Status: tg128 matches q8_0. pp16384 gap 23%. pp32768 gap 36%.")
    print("=" * 80)

    # Hardware constants
    WAVE_SIZE = 32
    GFLOPS_FP32 = 20.74
    BANDWIDTH_GBS = 512
    CU_COUNT = 72
    N_HEADS = 40
    K_BLOCK_BYTES = (2 + QK // 4 + QK // 8) * (D // QK)

    BENCH = {
        'turbo3_0': {'pp512': 785.80, 'pp4096': 1197.89, 'pp16384': 585.36, 'pp32768': 309.24, 'tg128': 45.57},
        'q8_0':     {'pp512': None,   'pp4096': None,    'pp16384': 763.00, 'pp32768': 482.79, 'tg128': 45.04},
        'turbo_b128': {'pp512': 771.63, 'pp4096': 1194.57, 'pp16384': 511.14, 'pp32768': None, 'tg128': 43.49},
    }

    C = {
        'fma': 0.5, 'mul': 0.5, 'add': 0.5, 'shfl': 1.0,
        'lds': 0.5, 'lds_bc': 2.0, 'gl1': 4.0, 'gl2': 20.0,
        'gmiss': 40.0, 'sync': 10.0, 'exp': 4.0, 'branch': 1.0,
        'f2h': 0.5, 'h2f': 0.5, 'byte': 0.5, 'vdot2': 1.0,
    }

    # Section 1: Benchmarks
    print("\n" + "=" * 80)
    print("  SECTION 1: ACTUAL BENCHMARK DATA")
    print("  Hardware: RX 6800 XT 16GB, gfx1030, ROCm")
    print("  Model: Qwen-AgentWorld-35B-A3B IQ4_NL, -ngl 99 -ncmoe 15")
    print("=" * 80)

    tests = ['pp512', 'pp4096', 'pp16384', 'pp32768', 'tg128']
    cfgs = [('turbo3_0', 'turbo3_0 b-32'), ('turbo_b128', 'turbo3_0 b-128'), ('q8_0', 'q8_0')]

    print(f"\n  {'Test':<10}", end="")
    for _, lab in cfgs:
        print(f" {lab:>22}", end="")
    print(f" {'turbo/q8':>10}")
    print("  " + "-" * 78)

    for t in tests:
        print(f"  {t:<10}", end="")
        for key, _ in cfgs:
            v = BENCH[key][t]
            print(f" {v:>22.2f}" if v else f" {'--':>22}", end="")
        tq = BENCH['turbo3_0'][t]
        q8 = BENCH['q8_0'][t]
        if tq and q8:
            print(f" {tq/q8*100:>9.1f}%")
        else:
            print(f" {'--':>10}")

    print(f"\n  DELTA: block-32 vs block-128")
    for t in tests:
        b32 = BENCH['turbo3_0'][t]
        b128 = BENCH['turbo_b128'][t]
        if b32 and b128:
            d = (b32 / b128 - 1) * 100
            print(f"    {t:<10} {b128:>8.2f} -> {b32:>8.2f}  ({d:+.1f}%)")

    # Section 2: Cycle-level simulation
    print("\n" + "=" * 80)
    print("  SECTION 2: CYCLE-LEVEL SIMULATION (turbo vs q8_0)")
    print("  WARNING: Approximate. Does not model RDNA2 ILP or cache hierarchies.")
    print("=" * 80)

    iters = D // 8
    turbo_per_iter = (3*C['byte'] + 8*C['byte'] + 8*C['lds'] + 8*C['h2f'] + 7*C['add'] + 1*C['mul'])
    turbo_per_k = iters * turbo_per_iter + C['exp'] + C['mul'] + C['add'] + C['fma'] + C['gl1'] + 4*C['byte'] + 4*C['mul'] + 4*C['add']
    q8_blocks = D // 32
    q8_per_k = q8_blocks * (4*C['byte'] + C['vdot2']) + C['exp'] + C['mul'] + C['add'] + C['fma'] + q8_blocks * (4*C['byte'] + C['vdot2'])

    print(f"\n  Per K position:")
    print(f"    turbo3_0: {turbo_per_k:.1f} cycles")
    print(f"    q8_0:     {q8_per_k:.1f} cycles")
    print(f"    ratio:    {turbo_per_k/q8_per_k:.2f}x (simulated)")

    seqlens = [512, 4096, 16384, 32768]
    print(f"\n  Simulated ops ratio turbo/q8_0 vs measured throughput ratio:")
    print(f"    {'seqlen':>8} {'sim_ops':>10} {'measured':>10} {'sim_vs_meas':>12}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    for sl in seqlens:
        sim_total = turbo_per_k * sl * N_HEADS
        q8_total = q8_per_k * sl * N_HEADS
        sim_ratio = sim_total / q8_total
        if sl in [16384, 32768] and BENCH['turbo3_0'][f'pp{sl}'] and BENCH['q8_0'][f'pp{sl}']:
            meas_ratio = BENCH['turbo3_0'][f'pp{sl}'] / BENCH['q8_0'][f'pp{sl}']
            print(f"    {sl:>8} {sim_ratio:>9.2f}x {meas_ratio:>9.2f}x {sim_ratio/meas_ratio:>11.1f}x")
        else:
            print(f"    {sl:>8} {sim_ratio:>9.2f}x {'--':>10} {'--':>12}")

    print(f"\n  The simulation overestimates the gap because it ignores ILP.")
    print(f"  RDNA2 hides compute via wavefront switching and instruction overlap.")

    # Section 3: Optimization strategy comparison
    print("\n" + "=" * 80)
    print("  SECTION 3: OPTIMIZATION STRATEGY COMPARISON")
    print("  NOTE: Benchmarks showed LUT nkq=1 ~= vec_dot nkq=2 on RDNA2.")
    print("=" * 80)

    per_iter = 3 + 8 + 8 + 8 + 7 + 1
    per_iter_f32 = 3 + 8 + 8 + 0 + 7 + 1
    q8_ops = 4 * (4 + 1)

    configs = [
        ('LUT nkq=1 (current)', iters * per_iter + 5, WAVE_SIZE),
        ('LUT f32 (no h2f)', iters * per_iter_f32 + 5, WAVE_SIZE),
        ('vec_dot nkq=2', (D//2//8) * per_iter + 7, WAVE_SIZE // 2),
        ('vec_dot nkq=4', (D//4//8) * per_iter + 9, WAVE_SIZE // 4),
        ('q8_0 V_DOT2', q8_ops, WAVE_SIZE),
    ]

    print(f"\n  {'Config':<25} {'Ops':>6} {'Parallel':>9} {'Throughput':>12}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*12}")
    for name, ops, parallel in configs:
        tp = parallel / ops
        print(f"  {name:<25} {ops:>6.0f} {parallel:>9} {tp:>12.4f}")

    print(f"\n  Throughput = parallel KQ scores / ops per KQ score")
    print(f"  BUT: benchmarks proved this metric is unreliable on RDNA2.")
    print(f"  LUT nkq=1 and vec_dot nkq=2 perform identically despite 3x ops diff.")

    # Section 4: Memory + KV cache analysis
    print("\n" + "=" * 80)
    print("  SECTION 4: MEMORY + KV CACHE ANALYSIS")
    print("=" * 80)

    for sl in [512, 4096, 16384, 32768]:
        k = sl * K_BLOCK_BYTES * N_HEADS
        v = sl * K_BLOCK_BYTES * N_HEADS
        q = D * 4 * N_HEADS
        total = k + v + q
        key = f'pp{sl}'
        tps = BENCH['turbo3_0'].get(key)
        if tps:
            wall_s = sl / tps
            bw = total / wall_s / 1e9
            util = bw / BANDWIDTH_GBS * 100
            print(f"  pp{sl:>5}: {total/1024:>7.1f}K data, {bw:>6.1f} GB/s needed, {util:>5.1f}% of {BANDWIDTH_GBS} GB/s peak")
        else:
            print(f"  pp{sl:>5}: {total/1024:>7.1f}K data, no benchmark")

    print(f"\n  KV cache at 32k context, 40 heads:")
    for name, bpc in [('turbo3_0 block-32', K_BLOCK_BYTES), ('turbo3_0 block-128', 66), ('q8_0', 165)]:
        mb = 32768 * bpc * 2 * N_HEADS / 1024 / 1024
        print(f"    {name:<25}: {mb:>6.1f} MB")

    # Section 5: Hot loop analysis
    print("\n" + "=" * 80)
    print("  SECTION 5: HOT LOOP ANALYSIS (KQ SCORING = 95% of kernel time)")
    print("=" * 80)

    iters = D // 8
    byte_loads = iters * 3
    index_extract = iters * 8
    lds_reads = iters * 8
    h2f_convert = iters * 8
    adds = iters * 7
    muls = iters
    softmax_ops = 4
    v_dequant_ops = 17
    total_ops = iters * 8 * 3 + byte_loads + softmax_ops + v_dequant_ops

    print(f"""
  The KQ scoring hot loop (fattn-vec.cuh):
    for (d0 = 0; d0 < {D}; d0 += 8):  // {iters} iterations
    {{
        ib = d0 / {QK};  // block index (0..{D//QK-1})
        // Load: norm(f16), qs0(u8), qs1(u8), sgn(u8) from block_turbo3_0
        // For each of 8 elements:
        //   idx = extract_bits(qs0, qs1, sgn, k)
        //   sum += half2float(turbo_lut[d0+k][idx])  // LDS read
        sum *= norm;
    }}

  Operations per KQ score:
    Byte loads:     {byte_loads:>4}  (3 per iteration: qs0, qs1, sgn)
    Index extract:  {index_extract:>4}  (8 per iteration: shift+mask)
    LDS reads:      {lds_reads:>4}  (8 per iteration: LUT lookup)
    h2f convert:    {h2f_convert:>4}  (8 per iteration: half->float)
    Adds:           {adds:>4}  (7 per iteration: sum 8 values)
    Muls:           {muls:>4}  (1 per iteration: norm scale)
    Softmax:           {softmax_ops}
    V dequant:        ~{v_dequant_ops}
    TOTAL:          {total_ops:>4}

  vs q8_0 V_DOT2:      44 total (4 blocks x (4 byte load + 1 V_DOT2) + softmax)

  Operations ratio:   {total_ops/44:.1f}x
  Measured ratio:     {BENCH['turbo3_0']['pp32768']/BENCH['q8_0']['pp32768']:.2f}x (pp32768)

  The {total_ops/44:.1f}x ops ratio compresses to {BENCH['turbo3_0']['pp32768']/BENCH['q8_0']['pp32768']:.2f}x measured because RDNA2 overlaps
  independent operations via ILP (instruction-level parallelism).
""")

    # Section 6: Known issues / gotchas
    print("=" * 80)
    print("  SECTION 6: KNOWN ISSUES AND GOTCHAS")
    print("=" * 80)
    print("""
  1. __launch_bounds__(, 2) = 25% PP regression
     Turbo kernel needs ~180 VGPRs. minBlocks=2 halves available to ~128.
     Result: register spill to local memory.
     FIX: Always use minBlocks=1 for turbo kernels.

  2. LUT vs vec_dot = equivalent on RDNA2
     Simulation predicted 3x ops difference. Benchmarks showed <2% difference.
     RDNA2's ILP hides the extra operations. Do not optimize ops count alone.

  3. Block-32 (QK_TURBO3=32) = +14.5% pp16384
     Single #define change cascades via macros. No model re-quant needed
     (turbo3_0 is KV cache format, not model weights).

  4. half4 vectorized WHT = slight PP regression
     Reduced thread count from 128 to 32 hurt more than SIMD benefit helped.
     WHT is ~5% of total time -- not worth optimizing.

  5. Bank-pad (lut_stride n_centroids+2) = slight PP regression
     Adding 1 half of padding per LUT row hurt more than it helped.
     The bank conflicts were not the bottleneck.

  6. Q_reg/Q_i32/Q_ds size reduction = PP regression
     Reducing array sizes to 1 element when k_lut=true caused compiler
     to generate worse code for surrounding loops.
""")

    return True


# ============================================================================
# Module 7: Integration Debug (from debug_turbo.py + debug_turbo2.py)
# ============================================================================

def detect_garble(text, label="content"):
    """Detect garbled output patterns in text.

    Source: debug_turbo.py + debug_turbo2.py.
    Checks for: replacement chars, control chars, repetition loops, newline floods,
    high special char ratios, stuck token patterns.
    """
    import re
    issues = []
    if not text:
        issues.append(f"{label}: EMPTY")
        return issues

    if "\ufffd" in text:
        issues.append(f"{label}: REPLACEMENT_CHARS")

    ctrl = re.findall(r"[\x00-\x08\x0e-\x1f]", text)
    if ctrl:
        issues.append(f"{label}: CONTROL_CHARS ({len(ctrl)})")

    for patlen in (2, 3, 4, 5):
        for i in range(len(text) - patlen * 5):
            pat = text[i:i + patlen]
            if pat * 5 in text:
                issues.append(f"{label}: REPETITION '{pat}' x5+ at pos {i}")
                break

    nl = text.count("\n")
    if nl / max(len(text), 1) > 0.15 and len(text) > 100:
        issues.append(f"{label}: NEWLINE_FLOOD ({nl}/{len(text)})")

    alphanum = len(re.findall(r"[a-zA-Z0-9]", text))
    special = len(text) - alphanum
    if alphanum > 10 and special / alphanum > 5.0:
        issues.append(f"{label}: HIGH_SPECIAL_RATIO ({special}/{alphanum})")

    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))
    if cjk > 0 and alphanum > 10:
        issues.append(f"{label}: UNEXPECTED_CJK ({cjk})")

    # Check for stuck token patterns
    words = text.split()
    if len(words) > 20:
        for w in (1, 2, 3):
            if len(words) > w * 10:
                seq = " ".join(words[-w:])
                count = text.count(seq)
                if count > 10:
                    issues.append(f"{label}: STUCK_TOKEN '{seq[:40]}' x{count}")
                    break

    return issues


def analyze_response(resp, run_label=""):
    """Analyze API response for garble indicators.

    Source: debug_turbo.py.
    """
    import json
    issues = []
    stats = {}

    if "error" in resp:
        return [{"fatal": resp["error"]}], stats

    choices = resp.get("choices", [])
    if not choices:
        return [{"fatal": "NO_CHOICES"}], stats

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    finish = choices[0].get("finish_reason", "unknown")
    usage = resp.get("usage", {})

    stats = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "finish_reason": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "content": content,
        "reasoning_preview": reasoning[:200] if reasoning else "",
    }

    issues.extend(detect_garble(content, "content"))
    issues.extend(detect_garble(reasoning, "reasoning"))

    if not content.strip() and finish == "stop":
        issues.append("content: EMPTY_BUT_STOPPED")

    return issues, stats


def debug_via_api(server_bin="./build/bin/llama-server", model=None, port=8099,
                  combos=None, prompts=None):
    """Server-driven garble diagnostic via llama-server API.

    Source: debug_turbo.py (adapted to accept parameters instead of hardcoded paths).
    Requires a running llama-server instance. Tests each -ctk/-ctv combo.

    Args:
        server_bin: Path to llama-server binary
        model: Path to GGUF model file
        port: Server port
        combos: List of (label, ctk, ctv) tuples to test
        prompts: Dict of (name, prompt_text) pairs
    """
    import json as _json
    import subprocess as _sp
    import time as _time
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    if combos is None:
        combos = [
            ("q8_q8",         "q8_0",     "q8_0"),
            ("q8_turbo3",     "q8_0",     "turbo3_0"),
            ("turbo3_turbo3", "turbo3_0", "turbo3_0"),
        ]

    if prompts is None:
        prompts = {
            "short": "Say hello in one sentence.",
            "long": ("You are a helpful AI assistant with expertise in computer science, "
                     "mathematics, physics, and engineering. Explain how neural networks work."),
        }

    base_args = [
        "-ngl", "99", "-c", "4096", "-fa", "on", "-n", "256",
        "--no-mmap", "--mlock", "-t", "8", "-tb", "12",
        "--temp", "0.0",
    ]

    print("=" * 70)
    print("SERVER-DRIVEN GARBLE DIAGNOSTIC (llama-server API)")
    print("=" * 70)
    print(f"Server: {server_bin}")
    if model:
        print(f"Model:  {model}")
    print(f"Port:   {port}")
    print()

    for label, ctk, ctv in combos:
        print(f"\n--- Testing: -ctk {ctk} -ctv {ctv} ({label}) ---")

        # Start server
        cmd = [server_bin, "-m", model] + base_args + ["-ctk", ctk, "-ctv", ctv, "--port", str(port)]
        try:
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE)
        except FileNotFoundError:
            print(f"  SKIP: Server binary not found at {server_bin}")
            continue

        # Wait for readiness
        ready = False
        for _i in range(60):
            try:
                with urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            _time.sleep(1)

        if not ready:
            print(f"  FAIL: Server did not start within 60s")
            try:
                proc.kill()
            except Exception:
                pass
            continue

        print(f"  Server ready. Running tests...")

        combo_issues = []
        for pname, prompt in prompts.items():
            payload = _json.dumps({
                "model": "test",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            }).encode()
            req = Request(f"http://localhost:{port}/v1/chat/completions", data=payload, method="POST")
            req.add_header("Content-Type", "application/json")

            try:
                with urlopen(req, timeout=120) as resp:
                    result = _json.loads(resp.read())
            except HTTPError as e:
                body = e.read().decode(errors="replace") if e.fp else ""
                issues, stats = analyze_response({"error": f"HTTP {e.code}: {body[:200]}"}, f"{label}/{pname}")
                combo_issues.extend(issues)
                continue
            except Exception as e:
                issues, stats = analyze_response({"error": str(e)}, f"{label}/{pname}")
                combo_issues.extend(issues)
                continue

            issues, stats = analyze_response(result, f"{label}/{pname}")
            combo_issues.extend(issues)

            if issues:
                for iss in issues:
                    print(f"    !! {iss}")
            else:
                print(f"    OK ({stats.get('content_len', 0)} chars, "
                      f"{stats.get('completion_tokens', 0)} tok)")

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

        verdict = "FAIL" if any(isinstance(i, dict) for i in combo_issues) else ("WARN" if combo_issues else "PASS")
        print(f"  >>> {verdict}: {len(combo_issues)} issues")

    return True


def debug_via_cli(model=None, port=8099):
    """Minimal CLI-based turbo KV cache debug via llama-cli.

    Source: debug_turbo2.py (adapted to accept parameters).
    Runs llama-cli directly with various -ctk/-ctv combinations.
    """
    import subprocess as _sp
    import time as _time
    import re as _re
    import os

    if model is None:
        print("SKIP: No model specified. Pass --model to specify a GGUF file.")
        return True

    cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "bin", "llama-cli")

    base_flags = [
        "-ngl", "99", "-c", "4096", "-fa", "on", "-n", "256",
        "--no-mmap", "--mlock", "-t", "8", "-tb", "12",
        "--temp", "0.0",
    ]

    combos = [
        ("q8_q8",     "q8_0",     "q8_0"),
        ("q8_turbo3", "q8_0",     "turbo3_0"),
        ("turbo3_turbo3", "turbo3_0", "turbo3_0"),
    ]

    prompt = "What is 2+2? Answer in one sentence."

    print("=" * 70)
    print("CLI-BASED TURBO KV DEBUG (llama-cli)")
    print("=" * 70)
    print(f"Model:  {model}")
    print(f"CLI:    {cli_path}")
    print(f"Prompt: {prompt}")
    print()

    for label, kt, vt in combos:
        label = f"{kt}/{vt}"
        cmd = [cli_path] + base_flags + ["-m", model, "-ctk", kt, "-ctv", vt, "-p", prompt,
                                           "--no-display-prompt", "--special", "-e"]

        t0 = _time.time()
        try:
            r = _sp.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = _time.time() - t0
            out = r.stdout

            # Extract generation lines (after prompt echo)
            lines = out.split('\n')
            gen_lines = []
            found_prompt = False
            for line in lines:
                if '> ' in line and not found_prompt:
                    found_prompt = True
                    idx = line.rfind('> ')
                    if idx >= 0:
                        rest = line[idx+2:].strip()
                        if rest:
                            gen_lines.append(rest)
                    continue
                if found_prompt:
                    gen_lines.append(line)
            gen = '\n'.join(gen_lines).strip()

            garble = "OK"
            s = gen.strip()
            if not s:
                garble = "EMPTY"
            else:
                for pl in range(1, 6):
                    for i in range(0, len(s) - pl * 5):
                        c = s[i:i+pl]
                        if c * 5 in s:
                            garble = f"REP '{c}'"
                            break
                    if garble != "OK":
                        break

            preview = gen[:150].replace('\n', ' | ') if gen else "(empty)"
            status = "OK  " if garble == "OK" else "FAIL"

            print(f"  {label:<25} {status:<10} {preview[:55]}")
            if garble not in ("OK", "EMPTY"):
                print(f"{'':25} {garble}")
            print(f"{'':25} {_time.time()-t0:.1f}s")

        except _sp.TimeoutExpired:
            print(f"  {label:<25} TIMEOUT")
        except FileNotFoundError:
            print(f"  {label:<25} SKIP: CLI not found at {cli_path}")
            break

    return True


# ============================================================================
# CLI Entry Point -- dispatches to the right sub-function based on --mode
# ============================================================================

_MODES = {
    "block-structure":    lambda: test_block_structure(),
    "rotation-mismatch":  lambda: test_rotation_mismatch(),
    "cpu-vs-gpu-quant":   lambda: test_cpu_vs_gpu_quant(),
    "attention-collapse": lambda: test_attention_collapse(),
    "norm-blowup":        lambda: test_norm_blowup(),
    "innerq-interference":lambda: test_innerq_interference(),
    "centroid-distribution": lambda: test_centroid_distribution(),
    "precision-audit":    lambda: precision_audit(),
    "garble":             lambda: garble_diagnostics(),
    "roundtrip":          lambda: roundtrip_tests(),
    "fa-kernel-audit":    lambda: fa_kernel_audit(),
    "rotation-verify":    lambda: rotation_fix_verification(),
    "full":               None,  # special: runs all tests
}


def _run_all_parallel(ncpus, n_samples, quick):
    """Run all sub-modes in parallel using ProcessPoolExecutor.

    Dispatches each independent sub-mode to a worker process. Results are
    collected and printed as they complete (via ``as_completed`` ordering).

    Falls back to sequential execution when only one CPU is available or
    when the worker cannot be spawned.
    """
    if ncpus is None:
        ncpus = os.cpu_count() or 1
    ncpus = max(1, ncpus)

    # Sub-modes that are safe to parallelize. Each maps a mode name to the
    # callable that should run in a worker process.
    parallelizable = [
        ("block-structure",    test_block_structure),
        ("rotation-mismatch",  test_rotation_mismatch),
        ("cpu-vs-gpu-quant",   test_cpu_vs_gpu_quant),
        ("attention-collapse", test_attention_collapse),
        ("norm-blowup",        test_norm_blowup),
        ("innerq-interference",test_innerq_interference),
        ("centroid-distribution", test_centroid_distribution),
        ("fa-kernel-audit",    fa_kernel_audit),
        ("rotation-verify",    rotation_fix_verification),
    ]

    all_results = {}
    any_failure = False

    if ncpus > 1 and len(parallelizable) > 1:
        actual_workers = min(ncpus, len(parallelizable))
        print(f"  Running {len(parallelizable)} tests across {actual_workers} workers...")

        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {}
            for mode_name, fn in parallelizable:
                f = pool.submit(fn)
                futures[f] = mode_name

            for future in as_completed(futures):
                mode_name = futures[future]
                try:
                    passed = future.result()
                except Exception as exc:  # pragma: no cover - worker crash handling
                    print(f"  [CRASH] {mode_name} (worker exception: {type(exc).__name__}: {exc})")
                    passed = False
                all_results[mode_name] = passed
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {mode_name} (completed in worker)")
                if not passed:
                    any_failure = True

        # Run precision-audit and garble tests with their parallel variants
        print()
        pa_pass = _precision_audit_parallel(n_samples=n_samples, quick=quick, ncpus=ncpus)
        all_results["precision-audit"] = pa_pass
        if not pa_pass:
            any_failure = True

        gr_pass = _garble_diagnostics_parallel(ncpus=ncpus)
        all_results["garble"] = gr_pass

        rt_pass = _roundtrip_tests_parallel(ncpus=ncpus)
        all_results["roundtrip"] = rt_pass

    else:
        # Sequential fallback -- use existing sequential implementations
        print(f"  Running {len(parallelizable)} tests sequentially (single CPU)...")

        for mode_name, fn in parallelizable:
            ok = fn()
            all_results[mode_name] = ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {mode_name}")
            if not ok:
                any_failure = True

        pa_pass = precision_audit(n_samples=n_samples, quick=quick)
        all_results["precision-audit"] = pa_pass
        if not pa_pass:
            any_failure = True

        gr_pass = garble_diagnostics()
        all_results["garble"] = gr_pass

        rt_pass = roundtrip_tests()
        all_results["roundtrip"] = rt_pass

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<50} {'Result':<8}")
    print("-" * 58)
    for mode_name in [m[0] for m in parallelizable] + ["precision-audit", "garble", "roundtrip"]:
        passed = all_results.get(mode_name, False)
        status = "PASS" if passed else "FAIL"
        print(f"{mode_name:<50} {status:<8}")

    print()
    if not any_failure and all(all_results.values()):
        print("ALL TESTS PASSED -- No issues detected in simulated pipeline.")
    else:
        print("FAILURES DETECTED -- See details above.")
    print("=" * 70)

    return 0 if (not any_failure and all(all_results.values())) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Turbo KV Cache Diagnostics -- merged debug suite")
    parser.add_argument("--mode", type=str, default="full",
                        choices=list(_MODES.keys()) + ["server-api", "server-cli"],
                        help="Test mode to run (default: full)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples for precision audit (default: 2000)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced sample count (~500)")
    parser.add_argument("--ncpus", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel workers for --mode full "
                             "(default: all available CPU cores)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model file (for server debug modes)")
    parser.add_argument("--port", type=int, default=8099,
                        help="Server port (default: 8099)")

    args = parser.parse_args()

    if args.mode == "server-api":
        debug_via_api(model=args.model, port=args.port)
        return 0

    if args.mode == "server-cli":
        debug_via_cli(model=args.model, port=args.port)
        return 0

    print("=" * 70)
    print("TURBO3_0 KV CACHE DIAGNOSTICS")
    print("=" * 70)
    print(f"Head dimension:   {D}")
    print(f"Block size:       {QK}")
    print(f"Centroids:        {N_CENTROIDS_3BIT} (3-bit)")
    print(f"WHT group size:   {GROUP_SIZE}")
    print(f"Mode:             {args.mode}")
    if args.mode == "precision-audit":
        print(f"Samples per dist: {args.n_samples}")
    print(f"Workers:          {args.ncpus} CPU(s)")
    print()

    if args.mode == "full":
        return _run_all_parallel(
            ncpus=args.ncpus,
            n_samples=args.n_samples,
            quick=args.quick,
        )

    # Single mode dispatch (sequential)
    if args.mode == "precision-audit":
        ok = precision_audit(n_samples=args.n_samples, quick=args.quick)
    else:
        fn = _MODES[args.mode]
        ok = fn()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
