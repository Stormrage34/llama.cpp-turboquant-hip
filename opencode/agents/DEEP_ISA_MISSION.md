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
  - `-mllvm -amdgpu-spill-sgpr-to-vgpr`
  - `-mllvm -amdgpu-enable-rewrite-out-of-range-value=1`
  - `-mllvm -amdgpu-early-inline-all=true`
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

## Execution Order

```
 Phase 1 (v0.4.0)          Phase 2 (v0.5.0)          Phase 3 (v0.6.0)
 ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐
 │ D: Compiler ✅  │     │ B: Prefetch     │       │ E: Warp Shuffle │
 │ A: 128-bit load │────▶│ C: MoE Preload  │       │                 │
 └─────────────────┘     └─────────────────┘       └─────────────────┘
```

## VGPR Budget
| Component | VGPRs | Source |
|-----------|-------|--------|
| Baseline decode (gfx1030) | 38 | Hardware limit for 100% occupancy |
| Idea A (int4 load) | +2 to +4 | per-thread load registers |
| Idea E (DPP shuffle) | +2 | shuffle target registers |
| Total with A+E | 42-44 | Exceeds 38 — only one feasible |

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
