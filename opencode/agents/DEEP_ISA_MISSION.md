# DEEP ISA Mission — RDNA2 Decode Optimization (v0.4.0)

## Overview
Map the 5 research ideas (A–E) to specific RDNA2 ISA instructions. Each idea must target an instruction-level transformation on the hot path (`mul_mat_vec_q` in `mmvq.cu`), maintain VGPR ≤ 38, and pass rocprofv3 telemetry gates.

---

## [ ] Idea A: 128-bit Vector Loads (`BUFFER_LOAD_DWORD4`)

**ISA Target**: `BUFFER_LOAD_DWORD4` (128-bit) replacing scalar `BUFFER_LOAD_DWORD` (32-bit).

- **Goal**: Reduce transactions per warp from 32×32B → 8×128B, aligning to 128B cache line granularity.
- **Implementation**: Replace `get_int_b4()` (32-bit) with `int4` load via `v4i32` type in `vec_dot_q*_K_q8_1()`.
- **VGPR Risk**: +4 VGPRs needed (int4 vs int). Offset by half* fix (~2-4 freed).
- **Gate**: `GL2C_EA_RDREQ_128B / GL2C_EA_RDREQ_32B` ratio ↑. `MemUnitBusy` ↑. VGPR ≤ 38.
- **Fallback**: `#ifdef RDNA2_V128_LOAD`

---

## [ ] Idea B: Software Prefetch (`s_buffer_load_dword` + `v_add_co`)

**ISA Target**: Schedule `s_buffer_load_dword` on the scalar unit to pre-load the next weight row's pointer while vector units compute the current dot product.

- **Goal**: Hide 600-800 cycle VRAM latency by starting the next row's fetch during the current row's dp4a chain.
- **Implementation**: `__builtin_amdgcn_s_buffer_load_dword()` or `__builtin_prefetch()` with lookahead=1 in the `kbx` loop.
- **Measurement**: `WAVE_ISSUE_WAIT` ↓, `WAVE_DEP_WAIT` ↓.
- **Gate**: tg128 regression < 2%. No VGPR change.

---

## [ ] Idea C: MoE Decode Weight Preload (Admin Stream + Async ACE)

**ISA Target**: ACE async copy engine — overlap MoE weight DMA with computation on the main stream.

- **Goal**: Eliminate the remaining ~600µs sync stall for MoE decode by pre-loading `w_{k+1}` via admin stream while computing `w_k × x`.
- **Implementation**: Extend `RDNA2_ASYNC_ROUTING` from prefill (P3) to decode path — preload next expert's weights via `admin_stream`.
- **Measurement**: MoE tg128 decode throughput delta.
- **Gate**: tg128 regression < 2% for non-MoE paths.

---

## [x] Idea D: Compiler Tuning (LLVM `-mllvm` flags) — SHIPPED v0.4.0

**ISA Target**: Force compiler to emit optimal scheduling for the gfx1030 wave32+dp4a pattern.

- **Implementation**: Applied unconditionally in `ggml/src/ggml-hip/CMakeLists.txt`:
  - `-mllvm -amdgpu-early-inline-all=true` — **active**
  - `-mllvm -amdgpu-spill-sgpr-to-vgpr` — **active**
  - `-mllvm -amdgpu-enable-rewrite-out-of-range-value=1` — **commented out** (not supported in ROCm 7.13)
- **These flags are always-on in v0.4.0+** (previously env-gated by `RDNA2_LLVM_OPT=1`).
- **Gate**: No regression expected — all flags are LLVM upstream, safe defaults.
- **2026-05-16**: Re-enabled after false accusation of all-newline bug. Real cause was `-n` flag (count-tokens mode).

---

## [ ] Idea E: Cooperative Warp Shuffle (`DS_SWIZZLE` / `V_DPP`)

**ISA Target**: Use `V_DPP` (Data Parallel Primitives) for warp-level weight distribution instead of individual per-thread global loads.

- **Goal**: 4 threads load 128B each, then `V_DPP` butterfly shuffle distributes across 32 lanes.
- **VGPR Risk**: Uses all ~4-6 VGPR headroom from half* fix. DPP16 needs 2 extra VGPRs for shuffle target.
- **Gate**: VGPR ≤ 38. LDS usage = 0.
- **Fallback**: `#ifdef RDNA2_DPP_SHUFFLE`

---

## [x] Idea F: VGPR_OPT Launch Bounds Tuning — SHIPPED v0.4.2

**ISA Target**: Reduce VGPR pressure by changing `__launch_bounds__` minBlocks from 1→3 in mmvq kernels.

- **Implementation**: `#ifdef RDNA2_VGPR_OPT_V1` in `mmvq.cu:395,615`:
  - `__launch_bounds__(nwarps*warp_size, 3)` instead of `(nwarps*warp_size, 1)`
  - Also applies `#pragma nounroll` to small loops in `vecdotq.cuh:1283`
- **Measured effect**: IQ4_NL 32→24 VGPRs (8→10 waves/CU); IQ4_XS unchanged at 48
- **Throughput impact**: Neutral (±0.5%, within noise)
- **Gate**: No regression; safe to enable by default

---

## Hardware Ceiling — PyTorch Benchmark (2026-05-19)

**Source**: Chief Engineer PyTorch benchmark on RX 6800 XT (gfx1030)

| Metric | Hardware Ceiling | Our Current Usage | Utilization |
|--------|-----------------|-------------------|-------------|
| **FP16 Compute** | 35.7 TFLOPS | ~1.3 TFLOPS (est.) | **3.6%** |
| **Memory Bandwidth** | 469 GB/s | ~17 GB/s | **3.6%** |
| **INT8 (dp4a)** | 92 TOPS (gfx1030) | ~0.234 TOPS | **0.25%** |
| **Infinity Cache** | 128 MB | N/A | N/A |

**Comparison**: RTX 3090 has 284 TOPS @ INT8 (3× more than RX 6800 XT's 92 TOPS).

**Bottleneck Analysis**:
- Decode is **memory-bound** but using only 3.6% of available bandwidth
- **dp4a utilization is 0.25%** — optimizing dp4a pipeline yields <1% overall gain
- Root causes: V_DOT8 instruction stalls, kernel launch overhead, MoE sync stalls
- Not compute-bound — plenty of FP16 headroom (35.7 TFLOPS ceiling)

**Implication**: Optimizations targeting memory bandwidth (prefetch, 128-bit loads) or dp4a pipeline have limited ROI until kernel launch overhead and sync stalls are addressed.

---

## dp4a Utilization Analysis (2026-05-19)

**Chief Engineer Finding**: dp4a pipeline optimization has **near-zero ROI** at 0.25% utilization.

### Hardware Context

| GPU | INT8 TOPS | Relative Performance |
|-----|-----------|---------------------|
| **RTX 3090** | 284 TOPS | 3.0× (baseline) |
| **RX 6900 XT** | 92 TOPS | 1.0× (our target) |
| **Current decode usage** | ~0.234 TOPS | 0.25% utilization |

### Real Bottleneck Rank Order

| Rank | Bottleneck | Impact | Optimization ROI |
|------|------------|--------|------------------|
| **1** | Kernel launch overhead (HIP ~5-10µs × 768 launches/step) | **~40-50%** | **HIGH** — P1 |
| **2** | MTP sync stalls (triple-sync + cross-stream) | **~20-30%** | **HIGH** — P0 |
| **3** | Perm chain in `get_int_from_table_16` (70% of kernel compute) | **~10-15%** | **MEDIUM** — P0 (Gap 3) |
| **4** | dp4a chain (10% of kernel compute × 0.25% GPU util) | **<1%** | **NONE** — P3 (demoted) |

**Key insight**: Even if we achieve 100% dp4a dual-issue efficiency (currently impossible due to dependency chains), the maximum gain is **0.25% of 10% = 0.025% overall improvement** — effectively noise.

### Updated Priority Recommendations

| Idea | Original Priority | New Priority | Rationale |
|------|------------------|--------------|-----------|
| **Idea B (Software Prefetch)** | P2 | **P4 (lowest)** | Infinity Cache + 0.25% dp4a util → no ROI |
| **Idea C (MoE Preload)** | P2 | **P1** | Wiring `LOAD_EXPERT_F32` still worth doing for cache hygiene |
| **Idea E (Warp Shuffle)** | P3 | **P3 (deferred)** | IQ4_NL only, narrow impact — keep deferred |
| **dp4a Split Accumulators** | P0 | **P3 (demoted)** | Targets wrong bottleneck — max 0.025% overall gain |

---

## Gap 3: Wrong Bottleneck — Perm Chain, Not dp4a (2026-05-19)

**Source**: Chief Engineer ISA analysis of `vecdotq.cuh:47-69`

### The Finding

The P0 split-accumulator optimization (CR-006) targets **dp4a accumulation** as the bottleneck. However, ISA analysis reveals the **real bottleneck is `get_int_from_table_16`'s perm chain**.

### Instruction Breakdown (per iteration)

| Function | Instructions | Cycles | % of Total |
|----------|-------------|--------|------------|
| `get_int_from_table_16` | 6× `V_PERM_B32` | ~48 cycles (8 each) | **70%** |
| dp4a accumulation | 2× `V_DOT8` | ~4 cycles | **10%** |
| Memory loads | `BUFFER_LOAD_DWORD` | ~12 cycles | **20%** |
| **Total** | — | ~64 cycles | 100% |

### Key Insight

**P0 split-accumulator optimization targets the wrong bottleneck.** Even if we achieve 100% dp4a dual-issue efficiency, the maximum gain is **10% of 10% = 1% overall improvement**.

The **real optimization opportunity** is reducing the perm chain:
1. **Pre-compute lookup tables** in a format that doesn't require byte-perm (e.g., already-expanded weights)
2. **Use `V_BFE_U32`** (bit-field extract) for nibble extraction if quant layout allows
3. **Fuse lookup + dp4a** into a single kernel that keeps table in LDS

### Updated Priority

**Perm chain optimization is now P0** — higher priority than dp4a splitting. See `docs/RESEARCH_LOG.md` for full analysis.

---

## Execution Order (Updated 2026-05-19 — Chief Engineer Priorities)

```
 Phase 1 (v0.4.0)          Phase 2 (v0.5.0)          Phase 3 (v0.6.0)
 ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐
 │ D: Compiler ✅  │     │ P0: MTP n-max=3 │       │ E: Warp Shuffle │
 │ A: 128-bit load │────▶│ P1: HIP Graph   │       │                 │
 └─────────────────┘     │ P2: MoE Prefetch│       └─────────────────┘
                         └─────────────────┘
```

**Chief Engineer ROI Ranking** (2026-05-19):

| Priority | Item | Est. Time | Expected Gain | Risk |
|----------|------|-----------|---------------|------|
| **P0** | Increase `--spec-draft-n-max` to 3 | 5 min (config) | +20-30% | Zero |
| **P1** | Reduce kernel launch overhead (HIP Graph/fusion) | 2-3 hrs | +10-15% | Low (gated) |
| **P2** | MoE weight prefetch (wire `LOAD_EXPERT_F32` macros) | 1 hr | +5-10% | Zero (gated) |
| **P3** | Demote Idea B (software prefetch) | N/A | N/A | N/A — Infinity Cache may limit ROI |

**Rationale**:
- P0 is a config change with proven gains (MTP acceptance 78.7% can absorb deeper drafts)
- P1 targets kernel launch overhead (identified bottleneck in PyTorch benchmark)
- P2 wires existing macros (zero new code, just connection)
- P3: Idea B demoted because memory bandwidth is not the bottleneck (only 3.6% utilized)

**MTP Optimization Track** (parallel to ISA roadmap):
- [x] **Triple-sync bug fixed** (v0.4.3-beta): Removed redundant `llama_synchronize()` calls
- [x] **load_gtt_slc() wired** (v0.4.3-beta): `LOAD_EXPERT_F32/F32X4` macros in `common.cuh`
- [ ] **MTP n-max=3 experiment** (P0): Test if 78.7% acceptance holds at deeper draft depth
- [ ] **MTP PP Overhead**: D2H transfer barrier — pinned memory solution (v0.4.4)
- [ ] **MTP Parallel Decoding**: Batched verification — shared draft context (v0.4.4)
- [ ] **Double-buffer MTP AR loop**: Overlap draft generation with verification (v0.6.0)

**Updated 2026-05-19**: Added PyTorch hardware ceiling data, re-prioritized roadmap per Chief Engineer ROI analysis. **Gap 3**: Perm chain optimization is new P0 (70% of compute time), dp4a split-accumulators demoted (only 10% of compute time).

## VGPR Budget (per-quantization)
| Quant Type | Baseline VGPR | With VGPR_OPT | Occupancy |
|------------|---------------|---------------|-----------|
| Q8_0 (type 8) | 16 | N/A | 100% (16 VGPRs) |
| Q4_0 (type 2) | 24 | N/A | 100% |
| Q4_1 (type 3) | 24 | N/A | 100% |
| IQ4_NL (type 20) | 32 | 24 | 100% with VGPR_OPT |
| IQ4_XS (type 23) | 48 | 48 (no change) | 50% |
| Q3_K (type 11) | 64 | N/A | 40% |
| IQ1_M (type 29) | 128 | N/A | 20% |
| Idea A (int4 load) | +0 (actual fix uses 32-bit) | — | — |
| Idea E (DPP shuffle) | +2 | — | Feasible: IQ4_NL only |

## Telemetry Gates
All future ideas require:
- [ ] rocprofv3 kernel trace (hot path = `mul_mat_vec_q`)
- [ ] `MemUnitBusy` delta
- [ ] `SQ_INSTS_VALU` delta
- [ ] VGPR ≤ 38 (`llvm-readelf` on compiled kernel)
- [ ] tg128 regression < 2%
- [ ] bit-exact parity (temp=0, seed=42, md5sum)

## v0.4.0 Ship Status
- [x] Idea D: Compiler flags — applied unconditionally in CMakeLists.txt
- [x] vecdotq bug fix (alignment mask) — upstream reverted
- [x] mmq.cuh LDS double-buffer — upstream pipeline restored
- [x] Build isolation (RPATH) — prevents ABI mismatch with other forks
- [x] CLI `--reasoning off` fix — no longer hardcodes DEEPSEEK format
- [x] PEG parser crash defense — try-catch in server-task.cpp
- [x] smoke_rdna2.cpp ROCm 7.13 compat — gcnArch → gcnArchName, half→uint16_t
- [x] Unified build script — interactive ROCm selection + RPATH isolation

---

## Updated Roadmap — 2026-05-19 (Post-PyTorch Benchmark + Gap 3 Analysis)

### P0: Perm Chain Optimization (v0.5.0) — **NEW (Gap 3)**
- [ ] **Reduce `get_int_from_table_16` perm chain**: 6× V_PERM_B32 = 70% of compute time
  - **Why**: Chief Engineer Gap 3 finding — real bottleneck, not dp4a accumulation
  - **Options**:
    1. Pre-compute lookup tables (already-expanded weights, no perm needed)
    2. Use `V_BFE_U32` for nibble extraction (if quant layout allows)
    3. Fuse lookup + dp4a into single kernel with table in LDS
  - **Files**: `ggml/src/ggml-cuda/vecdotq.cuh:47-69`
  - **Gate**: Perm chain instructions ↓ ≥50%, decode t/s ≥45 t/s
  - **Expected gain**: +30-50% (70% of compute time is perm chain)

### P0: MTP Configuration Tuning (v0.5.0) — **DEMOTED from P0**
- [ ] **Increase `--spec-draft-n-max` to 3**: Test if 78.7% acceptance holds at deeper draft depth
  - **Why**: Chief Engineer P0 — config change with +20-30% expected gain, zero risk
  - **Note**: Still valuable, but perm chain optimization has higher ceiling
  - **Files**: CLI flags only (no code change)
  - **Gate**: MTP acceptance ≥70% (current: 78.7%), decode t/s ≥40 t/s

### P1: Kernel Launch Overhead Reduction (v0.5.0) — **NEW**
- [ ] **HIP Graph or kernel fusion**: Reduce launch overhead identified in PyTorch benchmark
  - **Why**: Chief Engineer P1 — targets identified bottleneck (kernel launch overhead)
  - **Files**: `ggml/src/ggml-hip/`, `ggml/src/ggml-cuda/mmvq.cu`
  - **Gate**: Kernel launch time ↓ ≥20%, tg128 regression < 2%

### P2: MoE Weight Prefetch (v0.5.0) — **WIRED**
- [x] **`LOAD_EXPERT_F32/F32X4` macros defined** (`common.cuh:1558-1559`)
- [ ] **Wire into vec_dot kernels**: Connect macros to actual MoE expert weight loads
  - **Why**: Chief Engineer P2 — existing macros, just needs connection
  - **Files**: `ggml/src/ggml-hip/mmvq.cu`, `ggml/src/ggml-cuda/vecdotq.cuh`
  - **Gate**: MoE decode t/s improvement ≥5%

### P3: Demoted Items (v0.6.0 or later)
- [ ] **dp4a split-accumulators**: **DEMOTED** — targets wrong bottleneck (only 10% of compute time)
  - **Note**: Still valid optimization, but max gain is 1% overall (10% of 10%)
  - **Files**: `vecdotq.cuh:1333-1344` (IQ4_XS), `vecdotq.cuh:362-374` (nvfp4)
- [ ] **Idea B (Software Prefetch)**: **DEMOTED** — Infinity Cache may limit ROI (only 3.6% bandwidth utilized)
- [ ] **Idea E (Warp Shuffle)**: Deferred — IQ4_NL only, narrow impact
- [ ] **SDWA Register Packing**: Deferred — requires significant refactoring

### P1: MTP Optimization (v0.4.4)
- [ ] **Pinned memory for MTP D2H transfers**: Eliminate prompt processing overhead via `hipHostMalloc`
  - **Why**: D2H transfer barrier causes MTP prompt processing overhead
  - **Files**: `src/speculative.cpp`, `ggml/src/ggml-hip/hip-common.h`
  - **Gate**: PP overhead reduced by ≥50%

- [ ] **Shared MTP draft context across server slots**: Reduce memory duplication in multi-user scenarios
  - **Why**: Multiple slots duplicate draft model context unnecessarily
  - **Files**: `src/server-context.cpp`, `src/llama-context.h`
  - **Gate**: VRAM usage reduced by ≥20% in multi-slot scenarios

### P2: ISA-Level Optimizations (v0.5.0)
- [x] **128-bit vector loads** (`BUFFER_LOAD_DWORD4`): Implemented for nvfp4 kernel (`RDNA2_V128_LOAD`)
  - **Status**: OFF by default (+4 VGPRs), expansion to other quant types rejected (stride-4 access pattern)
  - **Files**: `ggml/src/ggml-cuda/vecdotq.cuh:37-42,362-375`
- [ ] **MoE decode weight preload** (Admin Stream V2): Extend `RDNA2_ASYNC_ROUTING` from prefill to decode path
  - **Why**: Eliminate remaining ~600µs sync stall for MoE decode
  - **Files**: `ggml/src/ggml-hip/moe_stream.cu`, `ggml/src/ggml-hip/mmvq.cu`
  - **Gate**: MoE tg128 decode throughput delta ≥10%

### P3: Advanced Research (v0.6.0)
- [ ] **Double-buffer MTP AR loop**: Overlap draft generation with verification
  - **Why**: MTP AR loop currently sequential — draft then verify
  - **Files**: `src/speculative.cpp`
  - **Gate**: MTP throughput ↑ ≥15%

- [ ] **Cooperative warp shuffle** (`DS_SWIZZLE` / `V_DPP`): IQ4_NL only, wave-level weight distribution
  - **Why**: 4 threads load 128B each, then DPP butterfly shuffle distributes across 32 lanes
  - **Files**: `ggml/src/ggml-cuda/vecdotq.cuh`
  - **Gate**: VGPR ≤ 38, LDS usage = 0
