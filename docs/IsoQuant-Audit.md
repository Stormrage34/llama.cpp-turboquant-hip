# IsoQuant Audit: Our Implementation vs Reference Kernel

**Paper:** https://arxiv.org/abs/2603.28430
**Reference kernels:** https://github.com/ParaMind2025/isoquant/tree/main/isoquant/csrc
**Files:** `isoclinic_fused_kernel.cu`, `planar2_fused_kernel.cu`

---

## Audit Results

| Check | Reference Code | Our Code | Match? |
|---|---|---|---|
| **IsoQuant-Full forward** | `q_L * v * conj(q_R)` | `q_L * v * conj(q_R)` | ✅ YES |
| **IsoQuant-Full inverse** | `conj(q_L) * qv * q_R` | `conj(q_L) * kv * q_R` | ✅ YES |
| **q_L and q_R separate?** | YES — two tensor params `q_left`, `q_right` | NO — same constant reused | ❌ BUG |
| **q_R conjugated in forward?** | YES — `quat_conj(qr, qr_conj)` fresh | YES — `quat_conj(q_R, q_R)` in-place | ✅ YES |
| **q_R uses unconjugated in inverse?** | YES — original `qr` used | YES — fresh `iso3_get_rotation` call | ✅ YES |
| **Norm separation** | NO (fused kernel only) | YES (corrected_norm pattern) | ⚠️ Different, valid |
| **Planar forward** | `c*v[0]-s*v[1], s*v[0]+c*v[1]` | `a*c-b*s, a*s+b*c` | ✅ YES |
| **Planar inverse** | `c*qv[0]+s*qv[1], -s*qv[0]+c*qv[1]` | `a*c+b*s, -a*s+b*c` | ✅ YES |
| **Per-type centroids?** | YES (Python-generated, per-type) | NO (planar3 and iso3 share same) | ❌ BUG |

---

## Bug 1: q_L and q_R Use Same Constants (3 DoF instead of 6 DoF)

### Reference:
```c
__global__ void isoclinic_full_fused_kernel(
    const float* __restrict__ q_left,    // [n_groups * 4] SEPARATE tensor
    const float* __restrict__ q_right,   // [n_groups * 4] SEPARATE tensor
    ...)
{
    // ...
    ql[i] = q_left[g * 4 + i];    // q_L FROM q_left
    qr[i] = q_right[g * 4 + i];   // q_R FROM q_right (DIFFERENT values!)
    // Forward: q_L * v * conj(q_R)
    quat_conj(qr, qr_conj);
    quat_mul(ql, v, temp);
    quat_mul(temp, qr_conj, rotated);
    // Inverse: conj(q_L) * qv * q_R
    quat_conj(ql, ql_conj);
    quat_mul(ql_conj, qv, temp2);
    quat_mul(temp2, qr, restored);
}
```

The reference takes **two separate tensor arguments** — `q_left` and `q_right`. These are independent per-block quaternions. For d=128 with 32 blocks:
```
q_left:  32 blocks × 4 floats = 128 floats  (q_L for each block)
q_right: 32 blocks × 4 floats = 128 floats  (q_R for each block)
Total: 256 floats
```

### Our code:
```c
static __constant__ float PI_QW[32] = { /* 32 values */ };
static __constant__ float PI_QX[32] = { /* 32 values */ };
static __constant__ float PI_QY[32] = { /* 32 values */ };
static __constant__ float PI_QZ[32] = { /* 32 values */ };
// Only 32 quaternions total!

void iso3_get_rotation(int block_idx, float q_L[4], float q_R[4]) {
    int idx = block_idx % 32;
    q_L[0] = PI_QW[idx]; q_L[1] = PI_QX[idx]; q_L[2] = PI_QY[idx]; q_L[3] = PI_QZ[idx];
    q_R[0] = PI_QW[idx]; q_R[1] = PI_QX[idx]; q_R[2] = PI_QY[idx]; q_R[3] = PI_QZ[idx];
    // SAME values for both q_L and q_R!
}
```

We only have 32 quaternions total — we reuse the same values for both q_L and q_R.
Reference has 64 (32 L + 32 R).

**Impact:** Reduces rotation from 6 DoF to 3 DoF. We pay the cost of IsoQuant-Full (32 FMAs/block)
but only get IsoQuant-Fast's expressivity.

### Fix: Add separate q_R constant arrays

Need 32 more quaternions for q_R side. Can be generated from Python:
```python
torch.manual_seed(42)
n_blocks = 32  # for d=128

# q_L for each block
q_L = torch.randn(n_blocks, 4)
q_L = q_L / q_L.norm(dim=1, keepdim=True)

# q_R for each block (separate random!)
q_R = torch.randn(n_blocks, 4)
q_R = q_R / q_R.norm(dim=1, keepdim=True)
```

---

## Bug 2: Shared Centroids Between Iso3 and Planar3

The reference implementation passes centroids as a **runtime parameter**, generated from Python per-type. Our code has:

```c
// PLANAR3 centroids
static __constant__ float PLANAR3_CENTROIDS[8] = {
    -0.174872f, -0.095650f, -0.038645f, -0.006723f,
     0.006723f,  0.038645f,  0.095650f,  0.174872f
};
// ISO3 centroids — IDENTICAL!
static __constant__ float ISO3_CENTROIDS[8] = {
    -0.174872f, -0.095650f, -0.038645f, -0.006723f,
     0.006723f,  0.038645f,  0.095650f,  0.174872f
};
```

Both iso3 and planar3 should have **distinct** Lloyd-Max centroids because the 4D quaternion rotation
produces a different distribution of rotated values than the 2D Givens rotation.

**Reference centroids (from johndpope commit 406bfbb):**
```c
// IsoQuant 3-bit (wider spread):
const float iso_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
```

---

## Note: Norm Separation

The reference fused kernel does NOT separate norm from direction — it rotates raw values directly.
This is because it's a fused quantize–dequantize kernel for benchmarking on synthetic unit-norm data.

Our llama.cpp implementation separates norm (corrected_norm = block_norm / sqrt(recon_sq)).
This is correct for KV cache usage where only indices + norm are stored for memory efficiency.
The paper's Algorithm 1 also separates norm, confirming our approach is valid.

---

## Summary of Fixes Needed

1. **Fix centroids** — Type-specific Lloyd-Max centroids for iso3 vs planar3
2. **Add separate q_R constants** — 32 more quaternions for the right-isoclinic factor
3. **Update iso3_get_rotation** — Use independent q_L and q_R arrays
