# Hipfire RDNA2 Research — Validated Results

Source: hipfire PR #298 (merged May 20, 2026), PR #315 (merged May 25, 2026)
Additional: PR #434 dispatch unification (Jun 10), PR #477 spec decode (Jun 28)
Hardware validated on: gfx1030/gfx1031/gfx1131 (RDNA2 wave32, v_dot2_f32_f16 confirmed)
Research author: unverbraucht (hipfire)

## gfx1030/gfx1131 confirmed support
- Has v_dot2_f32_f16: YES (gfx1030/gfx1031/gfx1032 + gfx11xx + gfx12xx)
- GEMV rows default: 1 (single-row optimal)
- All optimizations opt-in behind HIPFIRE_HFQ4_MMQ_RDNA2=1 (default OFF)

---

## PR #298 — MQ3 prefill on RDNA1/RDNA2 (Merged May 20, 2026)

### Commit-by-commit validated results

All numbers on gfx1031 / qwen3.5-9b.mq3 / warm cache unless noted.

| Commit | Phase | What | Result |
|--------|-------|------|--------|
| ab02d57 | 1: scalar batched HFQ3 | 4 new .hip kernels (qkv, qkvza, gate_up, residual). Single-warp, LB(32,8), LDS=0 | 56 -> 137 tok/s (2.45x) |
| 931088e | 2a: launch_bounds(32,6) | Relaxed from 8 waves to 6. Eliminated 3 VGPR spills (128 -> 158 VGPR ceiling) | pf=21: +16% (141->164), pf=191: +10% (138->151) |
| c2734b8 | 2b: dot2 inner loop | v_dot2_f32_f16 with FP16 X via ensure_fp16_x. 98 VGPR, 52 SGPR, 0 spills, 0 LDS | 148 -> 224-249 (+50-67%). Cumulative 4.18x |
| aa5ea6f | bind_thread cleanup | 31 pub fn reordered | No perf change |
| 059e829 | 2c: fp16-packed for gfx1010 | v_pk_fma_f16 for archs without dot2. gfx1010/gfx1013 only | Functional verified, no gfx1010 bench |
| 2cf265a | dp4a wave32 port | v_dot4_i32_i8 + Q8_1 X. 35 VGPR, 0 spills | **NEGATIVE: -15% median vs dot2** |
| e4548ae | dp4a long-prefill probe | Tested pf=30, 240, 1188 | -17% at pf=30, -17% at pf=240, -12% at pf=1188. Gap narrows but never closes |
| adc1558 | 3: MMQ minimal probe | LDS-tiled residual MMQ, mmq_x=32, MMQ_Y=128. 26KB LDS, 110 VGPR, 0 spills | +21% MQ3 (290->350), +22% MQ4 (288->351) |
| ce95ca1 | 3: tile-size family + auto-selector | 3 wrappers (x8/x16/x32) + batch-size gate. VGPR: 89/91/110 | +20-28% end-to-end |
| 052d36d | 3: full HFQ3 MMQ family | qkv + gate_up + residual MMQ. All 3 hot paths | **+89% at pf=240. Cumulative 9.77x** |
| 4f1abbf | 3: qkvza MMQ | LA preamble (4-way). All 4 batched-prefill hot paths | +2pp on short prompts |

### Phase 0 -> Final cumulative on 9B MQ3

| Workload | Phase 0 baseline | Final MMQ | Speedup |
|----------|-----------------|-----------|---------|
| paris (pf=21) | 56 tok/s | 255 tok/s | 4.55x |
| sheep (pf=36) | 56 tok/s | 338 tok/s | 6.04x |
| code (pf=21) | 56 tok/s | 262 tok/s | 4.68x |
| awq (pf=24) | 56 tok/s | 292 tok/s | 5.21x |
| LRU (pf=240) | 56 tok/s | 545 tok/s | **9.73x** |

### KLD validation (issue #302)

| Metric | Phase 1 baseline | Phase 3 MMQ | Verdict |
|--------|-----------------|-------------|---------|
| KLD n=30 KV=Q8 | 0.219 | 0.191 | "No regression" (MISLEADING) |
| KLD n=256 KV=Q8 | 0.219 -> 0.261 | 0.261 | +19% KLD vs pre-branch baseline |
| PPL | 9.93 | 10.44 | +5.1% |

**Correction (May 20):** The +19% was attributed to MMQ but turned out to be a STALE BASELINE. Fresh dot2 baseline at n=256 = 0.261334, MMQ = 0.261245. **MMQ is innocent** — dot2 and MMQ produce equivalent KLD. The 0.219 -> 0.261 jump belongs to Phase 2b (FP16 weight dequant vs Phase 1's FP32 scalar).

**Lesson:** KLD@n=30 is unreliable. Use n=256 minimum for precision changes.

---

## PR #315 — HFQ4 MMQ family + HFQ3 polish (Merged May 25, 2026)

### HFQ3 polish (5 commits)

| Commit | What | Result |
|--------|------|--------|
| 15264c9 | Per-layer MMQ gate for KLD attribution | Instrumentation only |
| d7a7cb3 | Per-layer FP16 gate for KLD attribution | Instrumentation only |
| 9f2359b | qkvza split routing (2+2 when beta/alpha unaligned) | LRU 240: 545 -> 668 tok/s (+22.6%). **Cumulative 11.93x** |
| 01469af | MMQ_Y=64 residual variant (26->15KB LDS, 2->4 WG/CU) | +1.5% on LRU 240 (668->678). VGPR 110->85 |
| 4ee8a90 | gate_up MMQ_Y=64 | **NEGATIVE: -4% (678->641)** |
| eb50436 | gate_up y64 sweep N=64..1024 | y64 ALWAYS slower at N>=128 (3-21% worse) |
| 3256104 | gate_up y96 probe | **NEGATIVE** |
| a9a5658 | residual MMQ_Y=32 | **NEGATIVE: -20-40%** (L2 cache pressure) |

**Key finding:** MMQ_Y is NOT a universal tuning knob. Residual benefits from y=64 (per-WG work is small, occupancy win dominates). gate_up REGRESSES at y=64 (per-WG work is 2x, dispatch overhead eats occupancy win).

### HFQ4 MMQ family on RDNA2 (4 commits)

| Phase | What | pp32 delta | pp128 delta |
|-------|------|-----------|-------------|
| 1 | Residual x16/x32/x32_y64 + auto-selector | 476 -> 509 (+7%) | 531 -> 648 (+22%) |
| 2 | qkv 3-way x16/x32 | 509 -> 528 (+4%) | 648 -> 673 (+4%) |
| 3 | gate_up 2-way x16/x32 | 528 -> 641 (+21%) | 673 -> 995 (+48%) |
| 4 | qkvza 4-way + split routing | 641 -> 645 (+1%) | 995 -> 1276 (+28%) |

### Final numbers from #300

| Model | Workload | Pre-#298 baseline | After #315 final | Delta |
|-------|----------|-------------------|------------------|-------|
| 4B MQ4 | pp32 | 476 | 645 | +35% |
| 4B MQ4 | pp128 | 531 | 1276 | **+140%** |
| 9B MQ4 | pp32 | 277 | 471 | +70% |
| 9B MQ4 | pp128 | 289 | 719 | **+149%** |

### NaN bug fix (commit 4e5fefc)
Root cause: `tid < 128` hardcoded in X-header load. At MMQ_Y=64, threads 64..127 wrote past x_dm into tile_y LDS. Fix: `tid < MMQ_Y`.

---

## Issue #299 — HFQ4 follow-ups (closed by #315)

- HFQ4 residual MMQ probe: +22% (matching HFQ3's +21%)
- Key insight: "LDS tiling is the dominant factor, NOT the unpack cost difference between 3-bit and 4-bit weights"
- Full family expected +30-60%, actual delivered +35-149%

---

## Issue #300 — Remaining levers post-Phase 3

### Status: Prefill largely exhausted

### Open levers (shipped or characterized)

| Lever | Priority | Expected gain | Status |
|-------|----------|--------------|--------|
| F1: hipGraph for prefill | HIGH | +10-20% | Blocked by hipMalloc-during-capture. GPU util fluctuates 60-99% during prefill |
| F2: MMQ epilogue fusion | MEDIUM | +5-15% | Fuse SwiGLU/RMSnorm into GEMM epilogue. Not started |
| F3: Persistent kernels | ARCH | +10-30% | Single WG loops over tiles. Not started |
| F4: VGPR reduction | LOW | +5-15% | 110 -> 88 VGPR for 3 waves/SIMD. Not started |

### Decode lever
- Phase 2 HFQ3 high-occupancy GEMV: ~18% expected
- Low priority: MQ3 decode already 1.11x faster than MQ4

### MMQ default-on gate blockers
- KLD eval at n=256 (done: MMQ is innocent)
- Coherence-gate clean across full model matrix
- Second RDNA2 SKU verification (RX 6800 XT or 6900 XT)

---

## Issue #301 — RDNA1 follow-ups (Open, not RDNA2)

Not relevant to our gfx1030 target. Tracks gfx1010 hardware verification + MMQ-fp16 variant.

---

## Issue #302 — KLD investigation (Closed)

Final resolution: MMQ Q8_1 X quantization does NOT add measurable KLD. The +19% report was from a stale baseline. dot2 and MMQ produce equivalent KLD at n=256 (delta < 0.04%).

---

## Issue #303 — Code cleanup (not RDNA2)
## Issue #304 — License change (not RDNA2)

---

## What transfers to llama.cpp (validated from hipfire)

### 1. LDS-tiled MMQ kernel for Q4K (highest impact)
hipfire's MMQ body is 199 lines of pure HIP C++. The topology:
- `__launch_bounds__(128, 2)`, 4 wave-32 warps
- LDS: x_qs (20KB) + x_dm (1KB) + tile_y (4.5KB) = 26KB
- Inner loop: 8x `__builtin_amdgcn_sdot4()` per sub-block
- Format-specific part: ONLY the X-tile loader (30 lines of bit manipulation)

For llama.cpp's Q4K layout `{d(2B), dm(2B), scales(12B), qs(128B)}`:
- Adapt X-tile loader for Q4K super-block structure
- Keep LDS layout, sdot4, wave-32 unchanged
- Expected: +15-25% prefill on dense models

### 2. MMQ_Y per-kernel tuning
- Residual kernel: y=64 (LDS-bound, smaller per-WG work)
- gate_up / qkv kernels: y=128 (need per-WG work to hide latency)
- NOT universal — requires per-kernel sweep

### 3. KLD validation methodology
- n=30 is unreliable. Use n=256 minimum.
- Cross-validate across code paths (dot2 vs MMQ vs scalar)

### 4. dp4a is negative on RDNA2
- Confirmed at pf=30/240/1188. Structural disadvantage is per-element.
- Do NOT port dp4a to llama.cpp for RDNA2.

### 5. Batch-size auto-selector
- b<=12: scalar/dot2 (MMQ tile granularity wastes compute)
- 13<=b<=127: mmq_x=16
- b>=128: mmq_x=32

### 6. hipGraph for prefill (+10-20%)
- hipfire confirmed GPU util fluctuates 60-99% during prefill
- Blocked by hipMalloc-during-capture (HIP restriction)
- Decode works around it via KernargBlob + ensure_* prequant scratches

---

## Our Baselines (measured on gfx1030)

| Model | Type | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|------|-------------|-------------|
| gemma-4-12B Q4_K_XL | Dense | 6.3 GB | 1333 | 54.6 |
| gemma-4-26B-A4B Q4_K_XL | MoE (4B active) | 14 GB | 2590 | 101.8 |

### MoE n-cpu-moe sweep (all-GPU is optimal)

| n_cpu_moe | pp512 (t/s) | tg128 (t/s) | tg delta vs GPU-all |
|-----------|-------------|-------------|---------------------|
| 0 (all GPU) | 2629 | 102 | baseline |
| 2 | 2084 | 68 | -33% |
| 6 | 1459 | 44 | -57% |
| 12 | 1031 | 28 | -73% |

---

## Bug Fixes Applied to turboquant-rebase

| # | File | Fix | Status |
|---|------|-----|--------|
| 1 | set-rows-planar-iso.cuh | Full q_L * v * conj(q_R) rotation | Verified |
| 2 | rotorquant.cuh:232 | Centroid bsearch inverted return | Verified |
| 3 | rotorquant.cuh:469,527 | % 42 -> % RQ_N_GROUPS | Verified |
| 4 | fattn-vec.cuh:296 | Magic 50 -> sizeof(block_iso3_0) | Verified |
| 5 | rotorquant.cuh:375 | QJL inner product index fix | Verified |
| 6 | set-rows-planar-iso.cuh:208 | static_assert block layout | Verified |
| 7 | turbo-quant.cuh:499 | turbo2_0 missing * norm | Verified |
| 8 | fattn-vec.cuh:102-113 | V_rows_per_thread=8 for planar3/iso3 | Verified (FA_ALL_QUANTS build fix) |

---

## Scripts

| Script | Purpose | Status | Usage |
|--------|---------|--------|-------|
| `scripts/rdna2-diagnostic.py` | Bayesian gap analysis on rocprofv3 profiles | Tested, working | `--profile-dir <dir> --quick` |
| `scripts/optimize_rdna2.py` | Build/bench both models across cmake variants | Tested, working | `--model-dense <path> --model-moe <path>` |
| `scripts/optimize-turboquant.py` | Build/bench turbo/planar/iso quant types | Tested, working | `--model-path <path> --configs baseline,fa_all_quants` |
| `scripts/gen_planar_iso_quaternions.py` | Regenerate unit quaternion constants | Tested, working | No args, writes to stdout |
| `scripts/bench-planar-iso.sh` | Bash build/bench harness (deprecated) | Replaced by Python scripts | |

## Optimization Branches

All from turboquant-rebase HEAD (bf0745cd8):

```
opt/hipgraph-capture    — cmake flag, P(helps)=99.1%
opt/vgpr-reduction      — __launch_bounds__, P=95.8%, LOW effort
opt/wave32-alignment    — rope workgroup fix, P=92.7%, LOW effort
opt/kernel-fusion       — rms_norm+rope+quantize fusion, P=95.0%
opt/persistent-kernels  — warp reuse, P=94.0%
opt/moe-cpu-offload     — smarter MOE routing
```

## Diagnostic Results (Bayesian)

| Optimization | P(helps) | Gain | Effort | Status |
|-------------|----------|------|--------|--------|
| hipGraph_capture | 99.1% | +18.0% | medium | 0% measured gain (different dispatch pattern from hipfire) |
| VGPR_reduction | 95.8% | +11.6% | low | mul_mat_q already has __launch_bounds__, VGPR=232 compiler-limited |
| memory_copy_batching | 98.3% | +9.6% | medium | Would require hipGraph (see above) |
| kernel_fusion | 95.0% | +7.2% | high | Not started |
| wave32_batch_alignment | 92.7% | +6.0% | low | Already wave-32 aligned (256 threads = 8 waves) |
| LDS_tile_resize | 67.8% | +7.0% | medium | Transferable from hipfire MMQ body |
| GEMM_epilogue_fusion | 60.2% | +6.8% | high | Not started |
| persistent_kernels | 94.0% | +18.0% | high | Not started |

## Build Variants Tested (both models)

| Variant | Dense pp512 | Dense tg128 | MoE pp512 | MoE tg128 |
|---------|-------------|-------------|-----------|-----------|
| baseline | 1333 | 54.6 | 2590 | 101.8 |
| hipgraph | 1332 (-0.1%) | 54.8 (+0.4%) | 2609 (+0.8%) | 103 (+1.1%) |
| fa_turbo | 1335 (+0.1%) | 55.0 (+0.8%) | 2606 (+0.6%) | 104 (+1.7%) |
| fa_graphs | 1333 | 54.5 | 2587 | 102 |
