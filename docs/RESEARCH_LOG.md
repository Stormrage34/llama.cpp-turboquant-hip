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

## BFE Dispatcher — VALID BUT UNTESTED ON CORRECT PATH

**Status**: ⚠️ Valid code, untested on target quant types (Q4_K_M / Q5_K_M).

**What it does**: `RDNA2_BFE_DISPATCHER` replaces shift/mask unpack with `v_bfe_u32` (1-cycle) for Q4_K_M and Q5_K_M dequant kernels. Trait-based compile-time dispatch, zero runtime branching.

**What's missing**:
- Kernel-path verification: `rocprofv3 --kernel-trace` must confirm `dequantize_row_q4_K_cuda` is invoked
- Correct model/quant: Validation must use Q4_K_M or Q5_K_M quant, not IQ4_XS/TurboQuant
- A/B harness: Same-build comparison with/without `RDNA2_BFE_DISPATCHER` flag

**Validation gates** (for BFE promotion to ON-by-default):

| Metric | Target | Method |
|--------|--------|--------|
| Kernel invoked | Yes | `rocprofv3 --kernel-trace` grep |
| `SQ_INSTS_VALU` ↓ | ≥10% (kernel-filtered) | Median across 3 runs, CV < 2% |
| Decode (`tg128`) | ≥26.5 t/s | `llama-bench -r 5` median |
| Variance | ≤±1.5 t/s | Std dev across 5 runs |
| Parity | Zero mismatches @ `temp=0.0` | Token diff vs CPU baseline |

---

## Infrastructure Gaps Blocking MTP/Async V2

| Gap | Impact | Fix |
|-----|--------|-----|
| No kernel-path verifier | Cannot confirm target kernel runs | `scripts/verify_kernel_dispatch.sh` |
| No counter normalizer | Cannot isolate metrics to target kernel | `rocprofv3 --dispatch-filter` |
| No A/B harness | Cannot compare same-build with/without flag | `scripts/run_ab_telemetry.sh` |
| No model/quant matrix | Unknown which flags affect which models | `docs/rdna2-flags.md` |

**Rule**: No new `#ifdef` kernel work until the validation pipeline proves isolation, normalization, and reproducibility.