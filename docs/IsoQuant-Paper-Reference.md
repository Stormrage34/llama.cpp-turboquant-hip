# IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression

**Paper:** https://arxiv.org/abs/2603.28430
**Reference code:** https://github.com/ParaMind2025/isoquant
**llama.cpp commit:** https://github.com/ggml-org/llama.cpp/commit/406bfbbfe7d0a0f3e188282d3ebb92f622143102
**Author:** Zhongping Ji (2026)

---

## Abstract

Orthogonal feature decorrelation is effective for low-bit online vector quantization, but dense random
orthogonal transforms incur prohibitive O(d^2) storage and compute. RotorQuant reduces this cost with
blockwise 3D Clifford rotors, yet the resulting 3D partition is poorly aligned with modern hardware and
offers limited local mixing.

IsoQuant uses blockwise SO(4) rotations based on quaternion algebra and the isoclinic decomposition of
the Lie group SO(4). Each block of four contiguous features is mapped to a quaternion, enabling compact,
closed-form rotational transforms via the "sandwich product" T(v) = q_L * v * conj(q_R).

**Performance (d=128):**
| Method | FMAs | Params | vs RotorQuant |
|---|---|---|---|
| IsoQuant-Full | 1,024 | 256 | 4.49x speedup |
| IsoQuant-Fast | 512 | 128 | 4.66x speedup |
| IsoQuant-2D | ~256 | 128 | 4.66x speedup |
| RotorQuant | ~2,408 | 172 | baseline |
| TurboQuant | 16,384 | 16,384 | — |

**Validation:** Stage-1 quantize–dequantize on synthetic normalized vectors only. End-to-end KV-cache
evaluation on real model activations remains future work as of publication.

---

## The Three Variants

### IsoQuant-Full (6 DoF)

Most expressive variant. Uses the complete double-sided action of SO(4).

```
Forward:  v' = q_L * v * conj(q_R)
Inverse:  v  = conj(q_L) * v' * q_R

Where q_L, q_R are independent unit quaternions on S^3.
Parameters per block: 8 (4 for q_L + 4 for q_R)
FMAs per block: 32 (16 for each quaternion multiply)
Blocks for d=128: 32 (128/4)
```

**Reference centroids (iso3):**
```c
static const float iso_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
```

**Rotation parameter generation:**
- Random fixed (Haar distribution on S^3 = normalize Gaussian sample)
- Can also be learned
- llama.cpp uses precomputed constants from Python torch.manual_seed(42)

**Key architectural invariant:** The FORWARD rotation in set_rows MUST match the INVERSE rotation
in dequant. If forward uses full sandwich, dequant must use full sandwich. If forward uses
left-only, dequant must use left-only.

### IsoQuant-Fast (3 DoF)

Lower-cost variant. Keeps only one isoclinic factor (left isoclinic).

```
Forward:  v' = q_L * v
Inverse:  v  = conj(q_L) * v'

Where q_L is a unit quaternion on S^3.
Parameters per block: 4
FMAs per block: 16
Blocks for d=128: 32
```

**llama.cpp's current implementation** (commit 406bfbb) uses IsoQuant-Fast (left-only),
NOT IsoQuant-Full. The commit comment explicitly states:
> "Uses quaternion sandwich product T(v) = q_L * v for 4D block rotation."

And the dequant comment:
> "Inverse: T^-1(v) = conj(q_L) * v"

### IsoQuant-2D / PlanarQuant (1 DoF per pair)

Lightweight planar special case. 2D Givens rotation per coordinate pair.

```
Forward:  u' = R(theta) * u
Inverse:  u  = R(-theta) * u'

Where R(theta) = [[cos(theta), -sin(theta)],
                  [sin(theta),  cos(theta)]]

Parameters per pair: 2 (cos, sin)
FMAs per pair: ~4
Pairs for d=128: 64
```

**Reference centroids (planar3):**
```c
// Same as iso3 but tuned for 2D Givens rotation
```

---

## Comparison with Other Methods

| Method | Block structure | Params (d=128) | FMAs (d=128) | Status |
|---|---|---|---|---|
| TurboQuant | dense 128x128 WHT | 16,384 | 16,384 | Production |
| RotorQuant | 43 x 3D Clifford | 172 | ~2,408 | Research (Triton) |
| IsoQuant-Full | 32 x 4D quaternion | 256 | 1,024 | Production |
| IsoQuant-Fast | 32 x 4D quaternion | 128 | 512 | Production |
| PlanarQuant/2D | 64 x 2D Givens | 128 | ~256 | Production |

**PPL results:** Simpler rotations (PlanarQuant, IsoQuant) work *better* than complex ones (RotorQuant)
for KV cache decorrelation, despite lower FMAs.

---

## Quaternion Math

### Quaternion Multiplication

```c
// r = p * q
r[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];  // scalar
r[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2];  // i
r[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1];  // j
r[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];  // k
// 16 FMAs per multiply
```

### Quaternion Conjugate

```c
// conj(q) = [w, -x, -y, -z]
conj[0] = q[0];  // w
conj[1] = -q[1]; // -x
conj[2] = -q[2]; // -y
conj[3] = -q[3]; // -z
```

### 4D Rotation via Sandwich Product

```c
// Full (IsoQuant-Full): v' = q_L * v * conj(q_R)
quat_mul(tmp, q_L, v);          // tmp = q_L * v
quat_mul(result, tmp, conj_R);  // result = q_L * v * conj(q_R)

// Fast (IsoQuant-Fast): v' = q_L * v
quat_mul(result, q_L, v);

// Inverse: v = conj(q_L) * v'
quat_mul(result, conj_L, v);
```

### Properties

- |q| = 1 for unit quaternion => |v'| = |v| (norm-preserving)
- conj(q) * q = q * conj(q) = |q|^2 (scalar)
- conj(q_L) * (q_L * v * conj(q_R)) * q_R = v (full inverse)
- conj(q_L) * (q_L * v) = v (fast inverse)

---

## Precomputed Constants

### Rotation Parameters (Python, torch.manual_seed(42))

```python
import torch
import math

torch.manual_seed(42)
d = 128

# Planar: 64 pairs of 2D Givens rotations
planar_cos = torch.empty(64)
planar_sin = torch.empty(64)
for i in range(64):
    angle = torch.rand(1).item() * 2 * math.pi
    planar_cos[i] = math.cos(angle)
    planar_sin[i] = math.sin(angle)

# Iso: 32 blocks × 2 quaternions = 64 unit quaternions
# q_L for each block (32)
iso_qw_L = torch.empty(32)
iso_qx_L = torch.empty(32)
iso_qy_L = torch.empty(32)
iso_qz_L = torch.empty(32)
# q_R for each block (32) — only for IsoQuant-Full
iso_qw_R = torch.empty(32)
iso_qx_R = torch.empty(32)
iso_qy_R = torch.empty(32)
iso_qz_R = torch.empty(32)

for i in range(32):
    # q_L: random unit quaternion on S^3
    u = torch.randn(4)
    u = u / u.norm()
    iso_qw_L[i], iso_qx_L[i], iso_qy_L[i], iso_qz_L[i] = u.tolist()
    
    # q_R: separate random unit quaternion
    u = torch.randn(4)
    u = u / u.norm()
    iso_qw_R[i], iso_qx_R[i], iso_qy_R[i], iso_qz_R[i] = u.tolist()
```

### Centroids (Lloyd-Max optimal for N(0,1))

**3-bit (8 levels):**
```c
// IsoQuant-specific (from paper reference):
const float iso_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// PlanarQuant (same as turbo3):
const float planar3_centroids[8] = {
    -0.174872f, -0.095650f, -0.038645f, -0.006723f,
     0.006723f,  0.038645f,  0.095650f,  0.174872f
};
```

**Note:** The reference commits use DIFFERENT centroids for iso3 vs planar3.
iso3 centroids are more spread out (max = 0.190685 vs 0.174872).
This suggests iso3's quaternion rotation produces rotated values with wider variance
than planar3's Givens rotation, requiring different Lloyd-Max levels.

**4-bit (16 levels, reserved for future planar4/iso4):**
```c
const float centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
```

---

## llama.cpp Implementation Guide

### File Structure

| File | Purpose |
|---|---|
| `planar-iso-constants.cuh` | Precomputed rotation constants (cos/sin, quaternions) |
| `planar-iso-dequant.cuh` | Rotation helpers + standalone dequant functions |
| `set-rows.cu` | Set_rows kernels with forward rotation |
| `set-rows-planar-iso.cuh` | Alternative device quantize functions (from johndpope) |
| `ctrl + v.cu` | Bulk F16→quantized conversion |
| `ctrl + v.cuh` | Header for cpy functions |
| `fattn-common.cuh` | KQ dot + V dequant with inverse rotation |

### Key Architecture: Forward + Inverse Must Match

**WRONG (causes precision loss):**
```
set_rows forward:  q * v * conj(q)   (full sandwich)
dequant inverse:   conj(q) * v        (left only)
```
These DON'T cancel!

**CORRECT:**
```
# IsoQuant-Fast (paper reference, commit 406bfbb):
set_rows forward:  q * v              (left only)
dequant inverse:   conj(q) * v        (left only)

# IsoQuant-Full with q_L = q_R:
set_rows forward:  q * v * conj(q)    (full sandwich)
dequant inverse:   conj(q) * v * q    (full sandwich)
```

### Lloyd-Max Quantization

```
Normalize:    v /= grp_norm
Rotate:       v' = rotate(v)           // decorrelate
Quantize:     idx = nearest_centroid(v')
              recon_sq = sum(centroid[idx]^2)
Store:        qs[idx], signs[idx], norm = grp_norm / sqrt(recon_sq)

Dequantize:
Dequant:      v' = centroid[idx] * norm
Inverse:      v  = inverse_rotate(v')  // undo decorrelation
```

### Key Constraints

1. **Forward and inverse rotation MUST use the same formula** (both IsoQuant-Fast or both IsoQuant-Full)
2. **Centroids should be type-specific** — iso3 uses different centroids than planar3
3. **Rotation preserves L2 norm** of each quaternion block independently
4. **Precomputed constants are compile-time initialized** via `__constant__` arrays
5. **V cache dequant must use ne=4** for quaternion rotation (ne=2 is broken for 4D rotation)

---

---

## Reference Kernel Implementation

Source: https://github.com/ParaMind2025/isoquant/blob/main/isoquant/csrc/isoclinic_fused_kernel.cu

### IsoQuant-Full Fused Kernel

```c
// Input tensors: input, q_left[N_GROUPS*4], q_right[N_GROUPS*4], centroids[LEVELS]
// Output: output (reconstructed after quantize-dequantize)

// Per-block (4 elements):
float ql[4], qr[4], qr_conj[4];
ql[i] = q_left[g * 4 + i];    // q_L: SEPARATE per-block quaternion
qr[i] = q_right[g * 4 + i];   // q_R: SEPARATE per-block quaternion

// Forward: q_L * v * conj(q_R)
quat_conj(qr, qr_conj);        // conj(q_R)  (in NEW variable, qr unchanged)
quat_mul(ql, v, temp);         // temp = q_L * v
quat_mul(temp, qr_conj, rotated); // rotated = q_L * v * conj(q_R)

// Scalar quantization (per-coordinate, Lloyd-Max)
qv[i] = nearest_centroid(rotated[i], centroids, n_levels);

// Inverse: conj(q_L) * qv * q_R
quat_conj(ql, ql_conj);        // conj(q_L)  (in NEW variable, ql unchanged)
quat_mul(ql_conj, qv, temp2);  // temp2 = conj(q_L) * qv
quat_mul(temp2, qr, restored); // restored = conj(q_L) * qv * q_R
// NOTE: qr is ORIGINAL (unconjugated) here
```

### IsoQuant-Fast Fused Kernel

```c
// Input tensors: input, q_left[N_GROUPS*4], centroids[LEVELS]
// (NO q_right! Only left isoclinic factor)

// Forward: q_L * v
quat_mul(ql, v, rotated);

// Quantize
qv[i] = nearest_centroid(rotated[i], centroids, n_levels);

// Inverse: conj(q_L) * qv
quat_mul(ql_conj, qv, restored);
```

### Planar2 Fused Kernel

Source: https://github.com/ParaMind2025/isoquant/blob/main/isoquant/csrc/planar2_fused_kernel.cu

```c
// Input tensors: input, rot2[N_PAIRS*2], centroids[LEVELS]
// rot2 layout: [cos0, sin0, cos1, sin1, ...]

// Forward
rotated[0] = cos * v[0] - sin * v[1];
rotated[1] = sin * v[0] + cos * v[1];

// Inverse
restored[0] = cos * qv[0] + sin * qv[1];
restored[1] = -sin * qv[0] + cos * qv[1];
```

### Key Architecture Differences vs Our llama.cpp Implementation

| Aspect | Reference Fused Kernel | Our llama.cpp Implementation |
|--------|----------------------|---------------------------|
| Purpose | Stage-1 benchmarking (quantize→dequantize fused) | KV cache (set_rows + dequant separate) |
| Norm | No norm separation (synthetic unit-norm data) | corrected_norm: `block_norm / sqrt(recon_sq)` |
| q_L/q_R storage | Two separate tensor params (`q_left`, `q_right`) | Same constant arrays for both |
| Centroids | Runtime param, Python-generated per-type | Static compile-time arrays |
| Output | Full reconstructed values | Indices + norm (zip format) |

---

## Known Issues & Debugging

### Symptom: Garbled output after ~200-400 tokens

**Root cause suspect:** Mismatch between forward and inverse rotation formula.
Verify:
1. set_rows and dequant use the SAME rotation variant (both Fast or both Full)
2. Centroids match the paper reference
3. Norm computation is consistent (corrected_norm = grp_norm / sqrt(recon_sq))
4. V_cache dequant uses ne=4, not ne=2

**Debugging steps:**
1. Test planar3_0 separately — if it works, issue is quaternion-specific
2. Add debug kernel to dump forward→quantize→dequant→inverse roundtrip values
3. Compare RMS error with vs without rotation
4. Verify rotation parameter consistency between set_rows and dequant calls

### Implementation Pitfalls

- **q_L = q_R from same array** — only provides 3 DoF rotation (equivalent to IsoQuant-Fast)
  without the left-only arithmetic savings. For IsoQuant-Full, need SEPARATE q_L and q_R arrays.
- **q_R overwritten by conj()** — set_rows calls `quat_conj(q_R, q_R)` which modifies q_R
  in-place. This is fine since iso3_get_rotation is called fresh each time.
- **Centroids must match rotation type** — iso3 rotation produces different distribution
  than planar3 rotation, requiring different Lloyd-Max levels.
