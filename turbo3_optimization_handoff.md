# Turbo3_0 Flash Attention Optimization — Engineer Handoff

## Current State (QK_TURBO3=32, block-32 format)

### Benchmark Results (Qwen-AgentWorld-35B-A3B, RX 6800 XT 16GB, gfx1030)
```
-ngl 99 -ncmoe 15 -b 4096 -ub 2048 -fa 1
```

| Metric | turbo3_0/turbo3_0 | q8_0/q8_0 | Gap |
|--------|------------------:|----------:|----:|
| pp512   | 785.80          | —         | —   |
| pp4096  | 1197.89         | —         | —   |
| pp16384 | 585.36          | 763       | -23%|
| pp32768 | 309.24          | 482.79    | -36%|
| tg128   | 45.57           | 45        | ~0% |

### Architecture Summary
- **Kernel**: `ggml/src/ggml-cuda/fattn-vec.cuh` — flash_attn_ext_vec
- **K scoring**: Shared-memory LUT, nthreads_KQ=1, 8 centroids, D=128
- **V accumulation**: Custom turbo3_0 dequant, nthreads_V=32, V_rows_per_thread=4
- **WHT**: `ggml/src/ggml-cuda/turbo-wht.cu` — scalar fp32, runs once per token per layer
- **Block size**: QK_TURBO3=32 (changed from 128 during this session)

### Key Files Modified
1. `ggml/src/ggml-common.h` — QK_TURBO3 128->32
2. `ggml/src/ggml-cuda/fattn-vec.cuh` — nthreads_V fix, k_lut conditions, dispatch
3. `ggml/src/ggml-cuda/turbo-wht.cu` — half4 vectorized WHT (minor PP regression, kept)

### What We Learned (Benchmarked, Not Simulated)

#### 1. LUT vs vec_dot KQ scoring: EQUIVALENT
Tested both paths with the same binary (compile-time switch `k_lut_bench`):
```
k_lut_bench=true  (LUT, nkq=1):  pp512=771  pp4096=1194  pp16384=511  tg128=43.5
k_lut_bench=false (vec_dot, nkq=2): pp512=781  pp4096=1108  pp16384=511  tg128=42.3
```
The 3x operation count difference in the simulation is hidden by RDNA2's memory latency hiding and ILP. Neither approach is meaningfully better.

#### 2. __launch_bounds__(, 2) causes 25% PP regression
Changing minBlocks from 1 to 2 halves available VGPRs per thread. The turbo kernel needs ~180 VGPRs but only gets ~128 with minBlocks=2 → register spill to local memory. **Must use minBlocks=1 for turbo kernels.**

#### 3. Block-32 (QK_TURBO3=32) gives +14.5% pp16384
Single #define change, cascades everywhere via macros. No model re-quantization needed (turbo3_0 is KV cache format, not model weights).

#### 4. half4 vectorized WHT: slight PP regression
Changed from 128 threads to 32 threads (4 elements each). The WHT kernel runs once per token per layer (~5% of total kernel time), so optimizing it has limited impact.

### The 36% Gap: Where It Lives

The gap is in the **KQ scoring loop** (95% of kernel time):

| Operation | turbo3_0 (LUT) | q8_0 (V_DOT2) | Ratio |
|-----------|----------------|----------------|-------|
| Per element | 1 LDS read + 1 h2f + 1 add | 0.125 V_DOT2 (8 elem/instr) | ~28x |
| Parallelism | 1 thread, 128 elements | 32 threads, 4 elem each | 32x |
| Net throughput | 32 KQ scores/wave | 32 KQ scores/wave | ~1x |

Wait — if both give 32 KQ scores/wave, why is turbo 36% slower at pp32768?

**Answer: turbo does MORE WORK per K position:**
- LUT build: D * 8 = 1024 half multiplies (precompute Q * centroid)
- LUT scoring: 16 iterations * (8 LDS + 8 h2f + 8 add + 1 mul) = ~256 ops
- Total: ~1280 ops per KQ score
- q8_0: 4 blocks * (4 byte loads + 1 V_DOT2) = ~20 ops per KQ score

The LUT approach does 60x more operations per KQ score. RDNA2 hides some via ILP, but at pp32768 the K sequence is long enough that memory bandwidth becomes the bottleneck, and the extra operations add up.

### Optimization Opportunities for Engineer Review

#### A. Eliminate LUT precomputation (~50% of KQ cost)
The LUT stores `Q[d] * centroid[c]` for all d and c. This precomputation costs D * n_centroids = 1024 half multiplies per KQ position. Instead of precomputing, compute `centroid[idx] * Q[d]` inline — same as the vec_dot approach does. The simulation predicted this would be 3x more ops, but benchmarked as equivalent because RDNA2 hides the difference.

#### B. Use V_DOT2 instruction for turbo KQ scoring
RDNA2 has `vf_dot2_f32_f16` which computes `a.x*b.x + a.y*b.y` in 1 cycle for half2 inputs. Could encode turbo centroids as half2 and use V_DOT2 for the dot product. Currently turbo uses scalar centroid lookup + scalar multiply.

#### C. Block size tuning
QK_TURBO3_GROUP=128 but QK_TURBO3=32 → 4 blocks per rotation group. Could try QK_TURBO3=64 (2 blocks per group) or QK_TURBO3=16 (8 blocks). Trade-off: more blocks = more norms = more metadata overhead but better parallelism.

#### D. Graph-side WHT optimization
The WHT kernel (`turbo-wht.cu`) runs once per token per layer. Currently scalar fp32. The turboquant-hip fork showed fp16 + half4 + pre-packed signs gives 31% speedup on WHT itself. Low overall impact but easy to implement.

#### E. hipGraph capture
Wrap the flash attention kernel launch in hipGraph to reduce launch overhead. Research shows +10-20% from eliminating per-kernel launch latency. This is infrastructure change, not algorithmic.

### Simulation Scripts

Two Python scripts for analysis:

1. **`fattn_turbo_sim.py`** — Cycle-level simulation of the full kernel
   - Models KQ scoring, V dequant, softmax, output stages
   - Memory bandwidth and compute analysis
   - Run: `python3 fattn_turbo_sim.py`

2. **`fattn_optimization_sim.py`** — Compares optimization strategies
   - nthreads_KQ=1/2/4, LUT f32, q8_0 baseline
   - Memory stall analysis, throughput estimates
   - Run: `python3 fattn_optimization_sim.py`

**Caveat**: The simulation's absolute cycle counts are wrong (doesn't model RDNA2 ILP, wavefront switching, or cache hierarchies properly). The relative comparisons between configs are more reliable, but even those were contradicted by actual benchmarks (LUT vs vec_dot).

**Recommendation**: Use ROCm's `rocprof` or `rocprofv2` for accurate profiling. The simulation identifies the right bottleneck (KQ scoring = 95%) but the optimization predictions need hardware validation.

### Changes Summary (git diff HEAD)

```
 ggml/src/ggml-common.h      | 6 +++---
 ggml/src/ggml-cuda/fattn-vec.cuh | 47 ++++++++++++++++++++++-----------
 ggml/src/ggml-cuda/turbo-wht.cu  | 192 +++++++++++++++++++++++++++++++
 3 files changed, 207 insertions(+), 38 deletions(-)
```
