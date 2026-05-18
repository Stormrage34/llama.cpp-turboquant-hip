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

## Execution Order

```
 Phase 1 (v0.4.0)          Phase 2 (v0.5.0)          Phase 3 (v0.6.0)
 ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐
 │ D: Compiler ✅  │     │ B: Prefetch     │       │ E: Warp Shuffle │
 │ A: 128-bit load │────▶│ C: MoE Preload  │       │                 │
 └─────────────────┘     └─────────────────┘       └─────────────────┘
```

**MTP Optimization Track** (parallel to ISA roadmap):
- [x] **Triple-sync bug identified** (2026-05-18): Found in `server-context.cpp`, fix needed in `speculative.cpp`
- [ ] **MTP PP Overhead**: D2H transfer barrier — pinned memory solution planned (v0.4.4)
- [ ] **MTP Parallel Decoding**: Batched verification — shared draft context across slots (v0.4.4)
- [ ] **Double-buffer MTP AR loop**: Overlap draft generation with verification (v0.6.0)

**Updated 2026-05-18**: Added P0 triple-sync fix, P1 MTP pinned memory, P2 shared draft context priorities.

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

## Updated Roadmap — 2026-05-18 (Post-Benchmark)

### P0: Critical Fixes (v0.4.3-beta)
- [ ] **Apply triple-sync removal in `speculative.cpp`**: Already identified in `server-context.cpp`, needs implementation in speculative decoding path
  - **Why**: Triple-sync bug causes unnecessary synchronization barriers in MTP speculative decoding
  - **Files**: `src/speculative.cpp`, `ggml/src/ggml-hip/server-context.cpp`
  - **Gate**: No regression in MTP acceptance rate (must maintain 78.7%)

- [ ] **Wire `load_gtt_slc()` into MoE weight fetch path**: Pillar 2 from Librarian research — currently dead code in MoE decode path
  - **Why**: SLC cache-bypass GTT loads not connected to MoE expert weight fetch
  - **Files**: `ggml/src/ggml-hip/moe_stream.cu`, `ggml/src/ggml-hip/mmvq.cu`
  - **Gate**: MoE decode t/s improvement ≥5%

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
- [ ] **128-bit vector loads** (`BUFFER_LOAD_DWORD4`): Replace scalar loads in `vec_dot_q*_K_q8_1()`
  - **Why**: Reduce transactions per warp from 32×32B → 8×128B
  - **Files**: `ggml/src/ggml-cuda/vecdotq.cuh`
  - **Gate**: `MemUnitBusy` ↑, VGPR ≤ 38

- [ ] **Software prefetch** (`s_buffer_load_dword`): Hide VRAM latency in weight fetch loop
  - **Why**: Hide 600-800 cycle VRAM latency by pre-loading next row's pointer
  - **Files**: `ggml/src/ggml-hip/mmvq.cu`
  - **Gate**: `WAVE_ISSUE_WAIT` ↓, tg128 regression < 2%

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

- [ ] **SDWA register packing**: Goal: 32 VGPR sustained for 100% occupancy
  - **Why**: Pack two 16-bit weights into one 32-bit VGPR using SDWA
  - **Files**: `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-cuda/mmvq.cu`
  - **Gate**: VGPR ≤ 32, occupancy = 100%
