# RDNA2 Research Log

> AI-assisted documentation. All claims backed by telemetry data in `benchmarks/`.

## v0.3.2-alpha: P3+DPP Investigation (2026-05-14)

### DPP Scale Broadcast — REVERTED

**Status**: ❌ Reverted. Zero measurable benefit. Path mismatch with benchmark model.

**What was tested**: `RDNA2_EXP_DPP_SCALES` — DPP (`v_mov_dpp` / `__builtin_amdgcn_readlane`) broadcast of block-scale `d` in `dequantize_block_iq4_xs_rdn2`. Thread 0 loads `x->d`, broadcasts to threads 1–7 via intra-wave readlane.

**Why it failed**:

1. **Path mismatch**: The DPP code targets `dequantize_block_iq4_xs_rdn2` (IQ4_XS quant type). The benchmark model (Qwen3.6-27B) uses TurboQuant (`turbo4`/`turbo2`), which dispatches through `dequantize_turbo4_0` / `dequantize_turbo2_0` — completely bypassing the IQ4_XS kernel. DPP code compiled but **never executed**.

2. **Counter methodology flaw**: Baseline vs P3+DPP counters were collected from different builds with different compile flags. `SQ_INSTS_VALU` showed +165% increase, but this was global kernel activity noise, not attributable to the DPP optimization scope. No `--dispatch-filter` or `--kernel-trace` was used to isolate the target kernel.

3. **Theoretical ceiling**: Even if the path matched, IQ4_XS dequant uses only 8/32 threads (25% occupancy). DPP saves 7 half-loads per 256-element block — a 5.1% load reduction on L1-cached data, yielding <1% actual throughput gain, unmeasurable against run-to-run variance.

**Revert scope**:
- `ggml/src/ggml-cuda/iq4_dequant_rdn2.cuh`: Removed `#ifdef RDNA2_EXP_DPP_SCALES` block, kept original `(float)x->d` path
- `ggml/src/ggml-hip/CMakeLists.txt`: Removed `GGML_RDNA2_EXP_DPP_SCALES` option and `RDNA2_EXP_DPP_SCALES` compile definition

**Lesson learned**: Always verify kernel dispatch path with `rocprofv3 --kernel-trace` before attributing counter deltas. Use `--dispatch-filter` for kernel-isolated metrics. Run A/B comparisons from the same build with only the target flag toggled.

---

## BFE Dispatcher — OFF-HOT-PATH (Cold Path Only)

**Status**: ❌ BFE targets standalone dequant path, which is NOT on the inference hot path.

**What it does**: `RDNA2_BFE_DISPATCHER` replaces shift/mask unpack with `v_bfe_u32` (1-cycle) in `dequantize_row_q4_K_cuda` and `dequantize_row_q5_K_cuda` (convert.cu:649-659).

**Why it doesn't matter for inference**:
- Kernel trace on Llama-3.1-8B-Q4_K_M shows Q4_K_M inference uses the **fused `mul_mat_vec_q` path** (stream-k fixup), NOT standalone `dequantize_row_q4_K_cuda`
- The standalone dequant path (`dequantize_row_q4_K_cuda`) is only called for:
  - KV cache type conversion (`GGML_OP_DEQUANTIZE` for `cache_type_k` changes)
  - Tensor copies between devices
  - Debug/inspection operations
- During normal inference, dequantization is **inlined into `mul_mat_vec_q`** — the BFE optimization never executes

**Evidence** (rocprofv3 kernel trace, Llama-3.1-8B-Q4_K_M):
- Observed kernels: `mul_mat_vec_q` (type12, type14), `rms_norm`, `rope`, `flash_attn_tile`
- NOT observed: `dequantize_row_q4_K`, `dequantize_block_q4_K`, or any BFE variant
- This matches upstream llama.cpp architecture: weight dequant is fused into matmul kernels

**Recommendation**: SUNSET BFE dispatcher. Keep code behind `#ifdef RDNA2_BFE_DISPATCHER` for reference, but do NOT promote to ON-by-default. The optimization targets a cold path.

**Additional finding**: Fixed brace bug in `build_attn_kv_iswa` (llama-graph.cpp:2527) that caused SIGSEGV for models using the kv_iswa attention path (Gemma 4, etc.). The `if (inp->self_v_rot)` block was missing its closing brace.

**Validation gates** (for BFE promotion to ON-by-default — NOT MET):

| Metric | Target | Result |
|--------|--------|--------|
| Kernel invoked on hot path | Yes | ❌ Not invoked during inference |
| `SQ_INSTS_VALU` ↓ | ≥10% (kernel-filtered) | N/A — kernel not on hot path |
| Decode (`tg128`) | ≥26.5 t/s | N/A — optimization not exercised |
| Variance | ≤±1.5 t/s | N/A |
| Parity | Zero mismatches @ `temp=0.0` | N/A |

---

## P2.2 SALU Offload (`readfirstlane` broadcast) — SUNSET (2026-05-14)

**Status**: ❌ Sunset. Zero measurable impact. Compiler already SALU-optimal on gfx1030.

**What was tested**: `__builtin_amdgcn_readfirstlane(kbx)` broadcast in `mul_mat_vec_q` and `mul_mat_vec_q_moe` inner loops. The idea: `kbx` is wave-uniform (for K-quants where `qi/vdr >= warp_size`), and broadcasting it from SGPR instead of VGPR would enable `s_add_u32` for address arithmetic, reducing VALU pressure and `SQ_INSTS_VALU`.

**Gate implementation**:
- Compile gate: `#ifdef RDNA2_ISSUE_OPT_V1` (CMakeLists.txt + build script)
- Runtime gate: `getenv("RDNA2_ISSUE_OPT")` → `cudaMemcpyToSymbol` → `__constant__ bool g_rdna2_issue_opt`
- Safety guard: `qi/vdr >= warp_size` prevents use on non-K-quant types (Q4_0, Q8_0, etc.)
- Host-side `rdna2_issue_opt_check_once()` in `ggml_cuda_mul_mat_vec_q()`

**Why it failed** (A/B comparison, same build, 5 runs each):

| Metric | Baseline (gate off) | P2.2 (gate on) | Delta | Gate |
|--------|--------------------|----------------|-------|------|
| pp512 | 1158.54 ± 0.78 t/s | 1158.92 ± 1.15 t/s | +0.03% | N/A |
| tg128 | 77.84 ± 0.16 t/s | 77.82 ± 0.13 t/s | **−0.03%** | ❌ |
| WAVE_ISSUE_WAIT | ~52,560 | ~52,560 (noise) | ~0% | ❌ |
| SQ_INSTS_VALU | 171,807 | 171,850 | ~0% | ❌ |
| VGPR | 40 | 40 | 0 | ✅ |
| Hot-path | ✅ | ✅ | — | ✅ |

**Score: 2/5 → SUNSET** (pass condition was ≥4 gates)

**Root cause**: LLVM's AMDGPU backend already recognizes wave-uniform VGPRs in address computation and routes them through SALU (`s_add_u32`) without `readfirstlane` hints. The `readfirstlane` intrinsic adds a `v_readfirstlane_b32` instruction (consuming a VALU cycle) and a constant memory load (`s_load_dword` for `g_rdna2_issue_opt`), which can offset any theoretical benefit. The real bottleneck is `MemUnitBusy`=85% (memory latency), not instruction issue — `WAVE_ISSUE_WAIT` is a symptom of the memory wall, not the root cause.

**Revert scope**:
- `ggml/src/ggml-cuda/mmvq.cu`: Removed `readfirstlane` blocks, host-side `rdna2_issue_opt_check_once()`, `__constant__ g_rdna2_issue_opt` — all reverted to upstream baseline
- `ggml/src/ggml-hip/CMakeLists.txt`: Removed `add_compile_definitions(RDNA2_ISSUE_OPT_V1)`
- `scripts/build_rdna2_llama.sh`: Removed `-DRDNA2_ISSUE_OPT_V1=1` flag
- Agent docs (`AMD.md`, `KERNEL_ENGINEER.md`): Updated priority tables to SUNSET, added sunset rationale

**Lesson learned**: Trust the compiler for wave-uniform routing on gfx1030. Always run a same-build A/B with `getenv` runtime gate before committing kernel changes. Profile the actual bottleneck before guessing — `MemUnitBusy` telemetry would have revealed the memory-bound nature earlier, saving the implementation effort.

**Pivot**: Cross-fork benchmarking (P2.4) — quantify RDNA2_OPT_V1+MATMUL_OPT_V1 delta vs v0.3.0-stable baseline.

---

## P2.3 Software Pipeline / kbx Loop Unrolling — SUNSET (2026-05-14)

**Status**: ❌ Sunset. VGPR headroom insufficient for meaningful latency hiding on decode path.

**Phase 0 (ISA audit)**:
- VGPR = **38** for `mul_mat_vec_q<Q4_K, 1, 0, 0>` (decode, no fusion) — not 40 as previously assumed
- Serialized load-compute pattern: 12 global_loads → 197 VALU → back-edge branch
- Only 2 VGPRs of headroom before occupancy cliff (VGPR≥48 → 64→56 waves/CU)

**Phase 1 failure** (`#pragma unroll 2` on kbx loop):
- VGPR exploded 38→62 (+63%), occupancy collapsed 64→40 waves/CU (−37.5%)
- Root cause: compiler eagerly unrolls entire kbx×j×i nest with inlined `vec_dot_q_cuda` (~20-25 VGPRs each), doubling live VGPRs
- Reverted immediately

**Phase 2 (manual load hoisting) — NOT ATTEMPTED**: 
- Theoretical analysis: `vec_dot_q_cuda` is a fused load+compute function; internal arrays (`v[]`, `u[]`, `d8[]`) stay live throughout dp4a chain
- Two simultaneous invocations require ~32-36 VGPRs, exceeding 38-VGPR ceiling
- `__builtin_prefetch` ineffective: streaming kernel uses non-temporal loads (L1 bypass), prefetch into L1 never consumed
- Compiler flag `-amdgpu-schedule-ilp=2` cannot overcome control dependency from loop back-edge branch

**Amdahl ceiling**: Decode kernel is 85% memory-bound. With 2 VGPRs headroom, theoretical maximum gain from any instruction-level optimization is <2%.

**Verdict: SUNSET**. Pivot to cross-fork benchmarking (P2.4). The real performance delta to chase is the existing RDNA2_OPT_V1 + MATMUL_OPT_V1 vs v0.3.0-stable baseline comparison, which has never been quantified.

**Lesson learned**: ISA audit must come BEFORE any optimization attempt. The 2-VGPR headroom finding would have ruled out P2.3 at design time, saving the `#pragma unroll 2` implementation and revert effort. All future kernel optimization proposals require a documented VGPR budget analysis as a gating step.

---

## Infrastructure Gaps Blocking MTP/Async V2

| Gap | Impact | Fix |
|-----|--------|-----|
| No kernel-path verifier | Cannot confirm target kernel runs | `scripts/verify_kernel_dispatch.sh` |
| No counter normalizer | Cannot isolate metrics to target kernel | `rocprofv3 --dispatch-filter` |
| No A/B harness | Cannot compare same-build with/without flag | `scripts/run_ab_telemetry.sh` |
| No model/quant matrix | Unknown which flags affect which models | `docs/rdna2-flags.md` |

**Rule**: No new `#ifdef` kernel work until the validation pipeline proves isolation, normalization, and reproducibility.

---

## MTP Optimization Research Notes (2026-05-18)

### MTP Prompt Processing (PP) Overhead — D2H Transfer Barrier

**Status**: ⚠️ Known architectural limitation — future optimization target

**Observation**: Prompt processing (PP) speed takes a negative hit when MTP is enabled, mainly due to Device-To-Host (D2H) embedding transfers.

**Root Cause**: The MTP speculative decoding implementation requires embedding vectors to be transferred from GPU (device) to CPU (host) for draft token verification. This D2H transfer creates a **synchronization barrier** between GPU compute and host reads:

```
GPU: Generate draft tokens → Compute embeddings
     │
     ├─→ [SYNC BARRIER] ← D2H transfer (hipMemcpyAsync, blocking)
     │
CPU: Verify draft tokens → Accept/reject
     │
     └─→ Signal GPU to continue
```

**Impact**:
- During prompt processing phase, the GPU must wait for host-side verification before proceeding
- The synchronization barrier prevents overlap of PP computation with verification
- Measured effect: PP throughput degradation proportional to prompt length

**Files Involved**:
- `tools/server/server-context.cpp` — MTP draft token verification logic
- `tools/server/server-task.cpp` — Speculative decoding orchestration

**Future Work**:
- [ ] Investigate pinned memory for D2H transfers (`hipHostMalloc`)
- [ ] Explore batched verification to amortize sync overhead
- [ ] Consider GPU-side verification kernel (keep verification on device)
- [ ] Profile with `rocprofv3 --dispatch-filter` to quantify exact sync stall duration

**Gate**: PP throughput with MTP enabled should match baseline within ±5%

---

### MTP Parallel Decoding Support Gap

**Status**: ⚠️ Supported but not optimized — future optimization target

**Observation**: Parallel decoding with MTP is supported, but not fully optimized yet.

**Root Cause**: The MTP speculative decoding implementation supports parallel decoding in theory, but the **draft token verification path has not been optimized for parallel execution**. This creates unnecessary serialization:

```
Current (serialized):
  Batch: [prompt_1, prompt_2, ..., prompt_N]
  
  For each prompt in batch:
    1. Generate draft tokens (GPU)
    2. D2H transfer (blocking)
    3. CPU verification
    4. H2D result transfer
    5. Continue generation
  
  → Sequential bottleneck: each request waits for previous verification
```

**Expected (parallel)**:
```
  Batch: [prompt_1, prompt_2, ..., prompt_N]
  
  1. Generate all draft tokens (GPU, parallel)
  2. Batched D2H transfer (single async copy)
  3. Parallel CPU verification (thread pool)
  4. Batched H2D result transfer
  5. Continue all generations
  
  → Amortized sync overhead across batch
```

**Impact**:
- Multi-request scenarios (server with concurrent users) see sub-linear scaling
- Single-request throughput unaffected, but batched throughput limited
- MTP acceptance rate (78.7% measured) remains strong, but verification latency adds up

**Files Involved**:
- `tools/server/server-context.cpp` — Draft token verification (serialization point)
- `tools/server/server-task.cpp` — Request batching logic
- `tools/server/server.cpp` — HTTP request handling

**Future Work**:
- [ ] Profile multi-request scenario with `rocprofv3` + server metrics (`/stats` endpoint)
- [ ] Implement batched D2H/H2D transfers for draft verification
- [ ] Add thread pool for parallel CPU-side verification
- [ ] Consider async streams for overlapping verification with generation
- [ ] Measure scaling: 1, 2, 4, 8 concurrent requests with MTP enabled

**Gate**: Multi-request throughput should scale linearly (N requests → N× throughput)

---

### Cross-Reference: MTP in DEEP_ISA_MISSION.md

The MTP optimization notes complement the existing roadmap in `opencode/agents/DEEP_ISA_MISSION.md`:

- **Idea C: MoE Decode Weight Preload** — Related to MTP async optimization (both involve async ACE streams)
- **Infrastructure Gaps** — MTP PP overhead and parallel decoding gap are now documented as specific optimization targets

**MTP Configuration (validated)**:
```bash
--spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75
```

**Measured Performance** (Qwen3_35BMTPIQ4, IQ4_XS, RX 6800 XT):
- Draft acceptance rate: **78.7%** (3,711/4,716 tokens)
- Decode throughput: **~39 t/s** (with MTP enabled)
- Effective throughput boost: **~21%** reduction in full forward passes

See `opencode/agents/chief_engineer.md` for full benchmark report (2026-05-17).

---

### Deep-Dive Findings — Explorer Code Analysis (2026-05-18)

**Source**: `opencode/reports/mtp_optimization_targets.md` (full structured report)

#### Issue 1: D2H Transfer Barrier — Triple Sync Confirmed

The D2H barrier is worse than initially documented. There are **two distinct D2H paths** and a **triple-sync waterfall** in the decode path.

**Decode path triple sync** (`common/speculative.cpp:698-738`):
1. `llama_synchronize(ctx_tgt)` — L710: Full GPU drain (redundant)
2. `llama_synchronize(ctx_mtp)` — L715: Full GPU drain (redundant)
3. `ggml_backend_tensor_get()` — L721-722: D2H copy + internal `cudaStreamSynchronize`

Each `llama_synchronize()` calls `ggml_backend_sched_synchronize()` — a full pipeline drain. The D2H copy already syncs internally via `cudaStreamSynchronize` after `cudaMemcpyAsync(D2H)`. The pre-syncs are **redundant** — stream ordering guarantees the compute is done before D2H starts.

**PP hook path** (`src/llama-context.cpp:3262-3301`):
- `synchronize()` at L3262 — redundant (same reasoning)
- Two `ggml_backend_tensor_get()` calls at L3281 and L3298 — each with internal sync
- `flush_mtp_data()` at L3314 — another `synchronize()` before staged batch drain

**P0 fixes identified**: Remove 3 redundant sync points → immediate throughput gain, zero risk.

#### Issue 2: Serialized Verification — Three Levels Confirmed

The serialization exists at three levels:

1. **Per-slot** (`server-context.cpp:2375-2387`, L3146): Each server slot's draft generation and verification runs sequentially in `for(auto & slot : slots)` loops. The existing TODO at L369 explicitly targets this: "rework to have a single draft llama_context shared across all slots."

2. **Per-draft-token** (`common/speculative.cpp:698-738`): The MTP AR loop is inherently sequential (each step depends on the previous), but the sync+copy could be pipelined with double-buffered embedding buffers.

3. **Per-batch-flush** (`src/llama-context.cpp:3360-3393`): `flush_mtp_data()` calls `llama_decode()` once per staged batch sequentially.

#### Optimization Priorities

| Priority | Optimization | File:Lines | Risk | Gain |
|----------|-------------|------------|------|------|
| **P0** | Remove redundant `llama_synchronize(ctx_tgt)` | `speculative.cpp:710` | None | ~1 sync/decode step |
| **P0** | Remove redundant `llama_synchronize(ctx_mtp)` | `speculative.cpp:715` | None | ~1 sync/decode step |
| **P0** | Remove redundant `synchronize()` in `collect_mtp_data` | `llama-context.cpp:3262` | Low | ~1 sync/ubatch during PP |
| **P1** | Pinned memory for MTP draft loop `batch.embd` | `speculative.cpp:721` | Low | Async D2H, no blocking |
| **P1** | Shared MTP draft context across server slots | `server-context.cpp:369` | Medium | N× throughput multi-slot |
| **P2** | Double-buffer MTP AR loop embed copies | `speculative.cpp:698-738` | Medium | Overlap D2H + compute |
| **P2** | Batch staged MTP batches | `llama-context.cpp:3360-3393` | Low | Fewer graph allocs |

**Full report**: `opencode/reports/mtp_optimization_targets.md`