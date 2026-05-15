# RDNA2 Decode Optimization Research

## Current Baseline (gfx1030, RX 6800 XT, 16GB)

| Model | tg50 (t/s) | VRAM | Bottleneck |
|-------|-----------|------|-----------|
| 8B Dense Q4_K_M | ~77 | full GPU | `MemUnitBusy`=85%, streaming read-once |
| 27B Dense IQ4_XS | ~27 | near-full GPU | memory BW bound |
| 35B MoE IQ4 (ncmoe=10) | ~62 | partial offload | memory BW + CPU sync |
| 26B MoE Q4_K (ncmoe=10) | ~53 | partial offload | memory BW |

**Established**: `mul_mat_vec_q` is streaming read-once. Each weight row read once per decode step. L2/Infinity Cache hit rate near-zero by design. No data reuse. `MemUnitBusy`=85% → 15% headroom remaining.

---

## Idea A: 128-bit Vector Loads

**Hypothesis**: Current per-thread loads use `get_int_b4` (32-bit scalar loads). Replacing with `int4` (128-bit vector loads) reduces transaction count 4x and aligns with RDNA2's 128B cache line granularity.

**Where**: `mmvq.cu` — weight loading inside `vec_dot_q*_K_q8_1()` and their helper impl functions.

**Risk**: Each `int4` needs 4 VGPRs vs 1 for `int`. The half* VGPR fix (P2.6) freed ~2-4 VGPRs — may need them for this.

**Measurement**: `GL2C_EA_RDREQ_128B / GL2C_EA_RDREQ_32B` ratio via rocprofv3.

**Gate**:
- `FETCH_SIZE` unchanged (same bytes fetched)
- `MemUnitBusy` ↑ (indicates better BW saturation)
- VGPR ≤ 38 (no occupancy regression)

---

## Idea B: Software Prefetch

**Hypothesis**: Insert explicit prefetch hints ahead of the weight load loop to tell L0/L1 to start fetching the next cache line while the current one is consumed.

**Where**: In the `kbx` loop of `mul_mat_vec_q` — prefetch the next weight row before entering the inner `j` loop.

**Mechanism**: `__builtin_prefetch()` or inline `s_buffer_load_dword` via `__builtin_amdgcn_s_buffer_load` intrinsics.

**Risk**: Pipeline stall from incorrect prefetch distance. Need to tune the lookahead distance (1 row? 2 rows?).

**Measurement**:
- `WAVE_ISSUE_WAIT` ↓ (fewer stalls while waiting for data)
- `WAVE_DEP_WAIT` ↓ (less time waiting for load results)
- `FETCH_SIZE` should stay the same (no extra bytes fetched, just earlier)

**Gate**: tg128 decode regression < 2%, VGPR unchanged.

---

## Idea C: MoE Decode Weight Preload (Admin Stream)

**Hypothesis**: For MoE decode (`mul_mat_vec_q_moe`), expert indices are known one token ahead of the current dispatch. Pre-load the next expert's weights via the admin stream while computing the current expert — the decode analog of P3's prefill async routing fix.

**Where**: `mmvq.cu` — the `mul_mat_vec_q_moe` kernel or its dispatch logic in `ggml_cuda_mul_mat_vec_q`.

**Mechanism**: The admin stream (`RDNA2_ASYNC_ROUTING`) is already wired for prefill. Extend its use to start the next expert's weight load while the current expert's dot product finishes.

**Risk**: Expert-to-expert dependency chains may limit overlap. Start with the simplest case: preload `w_{k+1}` while computing `w_k × x`.

**Measurement**: MoE decode throughput (tg128) delta.

**Gate**: tg128 regression < 2% for non-MoE paths.

---

## Idea D: Compiler Tuning

**Hypothesis**: The LLVM codegen for gfx1030 may not be using optimal scheduling heuristics for the mmvq workload pattern (streaming reads + dense dp4a chains).

**Where**: `ggml/src/ggml-hip/CMakeLists.txt` — HIP compiler flags.

**Candidates**:
- `-mllvm -amdgpu-spill-sgpr-to-vgpr` (trade SGPR for VGPR)
- `-mllvm -amdgpu-enable-rewrite-out-of-range-value=1`
- `-mllvm -amdgpu-schedule-lit=...` (adjust scheduling heuristic)
- `-mllvm -amdgpu-early-inline-all=true`
- `-mllvm -amdgpu-dump-kernel-resource-usage` (diagnostic)

**Measurement**: TG128 throughput + SQ_INSTS_VALU change.

**Gate**: No regression on any tested model + quant combo.

---

## Idea E: Cooperative Weight Loading (Warp-level Shuffle)

**Hypothesis**: Threads within a warp cooperate — each thread loads a 128-byte cache-line-sized chunk (16× `int`), then distributes via `__shfl_xor` or DS read. Reduces global memory transactions by sharing across the warp.

**Difference from P2.3**: P2.3 tried `#pragma unroll 2` which caused VGPR explosion (38→62) because the compiler unrolled the entire fused load+compute nest. A shuffle-based approach separates load from compute, avoiding the VGPR spike.

**Where**: `mmvq.cu` — the weight loading loop before the inner `j` (dp4a) loop. Load into registers, then shuffle to distribute across lanes.

**Risk**: Shuffle instructions add ~4 cycles of latency. Need enough independent work to hide it.

**Measurement**: `SQ_INSTS_VALU` delta from shuffle ops. `WAVE_ISSUE_WAIT` delta from reduced global load instructions.

**Gate**: VGPR ≤ 38 (headroom from half* fix is ~4-6). LDS usage = 0 (decode path cannot afford LDS overhead at batch=1).

---

## Summary: Research Priority

| Idea | ROI | Risk | VGPR Impact | Pre-req |
|------|-----|------|-------------|---------|
| A: 128-bit loads | Medium-High | Low | +2-4 VGPR | half* fix frees room |
| B: Software prefetch | Medium | Low | None | — |
| C: MoE preload (admin) | Medium | Medium | None | P3 infrastructure ready |
| D: Compiler tuning | Medium | Low | None | — |
| E: Warp shuffle | Medium | High | Uses all headroom | half* fix frees room |
