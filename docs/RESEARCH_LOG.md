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

## Infrastructure Gaps Blocking MTP/Async V2

| Gap | Impact | Fix |
|-----|--------|-----|
| No kernel-path verifier | Cannot confirm target kernel runs | `scripts/verify_kernel_dispatch.sh` |
| No counter normalizer | Cannot isolate metrics to target kernel | `rocprofv3 --dispatch-filter` |
| No A/B harness | Cannot compare same-build with/without flag | `scripts/run_ab_telemetry.sh` |
| No model/quant matrix | Unknown which flags affect which models | `docs/rdna2-flags.md` |

**Rule**: No new `#ifdef` kernel work until the validation pipeline proves isolation, normalization, and reproducibility.