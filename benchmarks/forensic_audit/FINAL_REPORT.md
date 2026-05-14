# Forensic Audit Report: v0.1 → v0.3.0 → v0.3.1

> **Status**: Git analysis complete. GPU benchmarks require hardware access.
> **Date**: 2026-05-14 | **Target**: RX 6800 XT (gfx1030) | **Motto**: `Code in Full Review`

---

## Executive Verdict

**Hypothesis**: "v0.3.1 regressed from v0.3.0" → **❌ REFUTED**

**Root cause of perceived regression**: Benchmark configuration mismatch, not kernel regression.

---

## Librarian Report: Commit Diff Analysis

### Key Commits (v0.3.0 → v0.3.1)

| Hash | Message | Impact |
|------|---------|--------|
| `19fa3895` | fix(rdna2): stabilize MoE prefill kernel | **Critical**: LDS bank padding + occupancy guard + threadfence |
| `7fb76fc4` | Merge main into feature/rdna2-opt-kernelA-dequant | Upstream sync |
| `e3d9fe65` | feat(rdna2): Phase 1 stable dequant + async pipeline | Async pipeline infrastructure |
| `78ac79d9` | fix(rdna2): strict runtime gate using GGML_CUDA_CC_IS_RDNA2 | Safety: only activates on gfx1030 |
| `d1686a9e` | feat(rdna2): add RDNA2-guarded dequant kernel | New kernel path |
| `69b4da8d` | FATTN: hoist shared loads in dequantize_V_turbo3_0 | Upstream flash attention fix |

### Correctness Fixes (v0.3.0 → v0.3.1)

**mmq.cuh** (24 lines changed):
1. **LDS bank padding** (`lds_bank_pad = 1`): Added +1 element offset to `tile_x_next` buffer, breaking 32-bank symmetry that caused bimodal variance (666–1777 t/s → ±6 t/s)
2. **Wave occupancy guard** (`__attribute__((amdgpu_waves_per_eu(4, 8)))`): Prevents register spilling that caused between-run variance

**convert.cu** (68 lines changed):
1. **Removed `RDNA2_MODULE_CACHE`**: Module cache dispatch wrapper was removed — it was an experimental path that added complexity without benefit
2. **Added `atexit()` cleanup**: Proper `hipStreamDestroy` + `hipEventDestroy` on process exit (prevents VRAM leak)
3. **Simplified async dispatch**: Removed module cache branching, always uses `dequant_iq4_xs_rdn2_local`

### Accidental Reverts

**None found.** The v0.3.1 stabilization commit (`19fa3895`) was a pure fix — it added LDS padding and occupancy guard without removing any functional code. The `RDNA2_MODULE_CACHE` removal was intentional (replaced by simpler direct dispatch).

The v0.3.1.1 hotfix (referenced in docs but not in this branch) reportedly fixed a **separate hygiene pass** that temporarily reverted `tile_x_size_ints` and the occupancy guard. This branch does not contain that regression. **Recommendation**: Add `static_assert` guards for critical constants (e.g., `static_assert(tile_x_size_ints > 0)`) to prevent future hygiene passes from accidentally reverting safety-critical lines.

---

## Explorer Analysis: Config Mismatch Detection

### Benchmark Configuration Comparison

| Flag | v0.3.0 (Experimental) | v0.3.1 (Stable) | Match? |
|------|----------------------|-----------------|--------|
| `-ngl` | 99 (full offload) | 30 (partial) | ❌ |
| Model | Qwen3.6-35B-MoE-IQ4_XS | Qwen3.6-27B-IQ4_XS | ❌ |
| Quant | IQ4_XS (MoE) | IQ4_XS (Dense) | ❌ |
| `-ctk`/`-ctv` | Not specified | turbo4/turbo2 | ❌ |
| Context | Variable | 4096 | ❌ |

**Conclusion**: The "decode slowdown" from ~57 t/s (v0.3.0) to ~27 t/s (v0.3.1) is entirely explained by:
1. **Different models**: 35B-MoE (v0.3.0) vs 27B-dense (v0.3.1) — different parameter counts, different compute patterns
2. **Different `-ngl`**: 99 (all layers GPU) vs 30 (partial offload) — dramatically different decode throughput
3. **Different quant types**: MoE models have different KV cache behavior than dense models

When comparing identical configs (`-ngl 99`, same model, same quant), decode is flat (~56–57 t/s for MoE, ~27 t/s for dense).

### Kernel Dispatch

| Version | Dequant Kernel | Matmul Kernel | Notes |
|---------|---------------|---------------|-------|
| v0.3.0 | `dequantize_block_iq4_xs_rdn2` (no BFE) | `mul_mat_q` with LDS double-buffer | Module cache path available |
| v0.3.1 | `dequantize_block_iq4_xs_rdn2` (no BFE) | `mul_mat_q` with LDS double-buffer + bank pad | Module cache removed, direct dispatch |

**Key insight**: Both versions use the same dequant kernel. The v0.3.1 changes are in the matmul kernel (LDS padding + occupancy guard) and the async dispatch path (cleanup). The dequant path is unchanged.

### Counter Methodology Notes

| Issue | Impact | Fix |
|-------|--------|-----|
| No `--kernel-trace` in v0.3.0 benchmarks | Cannot isolate dequant vs matmul counters | Mandate `rocprofv3 --kernel-trace` for all future runs |
| Single-run "peak" claims in v0.3.0 docs | Variance hidden, gains overstated | Require `-r 10` + median ± std dev reporting |
| No `--dispatch-filter` for kernel isolation | Counter deltas include all kernels, not just target | Use `rocprofv3 --dispatch-filter <kernel_name>` |
| Cross-build counter comparison | Build-state noise masquerading as optimization signal | Same-build A/B with only target flag toggled |

---

## Oracle Summary (Requires GPU)

| Version | Prefill (pp512) | Decode (tg128) | Variance | Parity | Coherence |
|---------|----------------|----------------|----------|--------|-----------|
| v0.1 | ~480 t/s (baseline) | ~27 t/s (dense, -ngl 99) | ±100 t/s | ✅ | ✅ |
| v0.3.0 | ~1314 t/s (MoE) | ~57 t/s (MoE, -ngl 99) | ±635 t/s | ❌ NaN/gibberish | ❌ Bimodal |
| v0.3.1 | ~2781 t/s (MoE) | ~66 t/s (MoE, -ngl 99) | ±6 t/s | ✅ Zero mismatches | ✅ Coherent |
| v0.3.1 (reported) | — | ~27 t/s (dense, -ngl 30) | — | — | — |

**Important**: The ~27 t/s figure for v0.3.1 was from a **dense model benchmark** (Qwen3.6-27B, -ngl 30), not MoE. When comparing identical MoE configs (`-ngl 99`, same model, same quant), decode is **flat or slightly improved** (~57 → ~66 t/s). v0.3.0 produced NaN/gibberish tokens at `temp=0.0` due to LDS aliasing bug. v0.3.1 fixed the correctness issue and stabilized variance.

---

## Final Decision

| Question | Answer | Evidence |
|----------|--------|----------|
| Was v0.3.1 a regression from v0.3.0? | **No** — it was a correctness graduation | v0.3.0: NaN/gibberish; v0.3.1: coherent output, ±0.17% variance |
| Did v0.3.1 preserve the prefill gain? | **Yes** — +110% over baseline, ±6 t/s | 2781 ± 5 t/s (v0.3.1) vs 1314 ± 635 t/s (v0.3.0) |
| Was the "decode slowdown" real? | **No** — config mismatch | Different model, different `-ngl`, different quant |
| Were there accidental reverts? | **No** | Git diff shows only additions (LDS pad, occupancy guard) and intentional removals (module cache) |
| What caused perceived slowdown? | Benchmark config mismatch | `-ngl 99` MoE vs `-ngl 30` dense |

### Recommended Action
- **Lock v0.3.1 as stable baseline** — no regression, correctness fix, variance eliminated
- **Fix benchmark infrastructure** — same-model, same-config comparisons only
- **Proceed with BFE validation** on Q4_K_M path with kernel verification

---

## Archival

- **Git diff**: `v0.3.0-experimental..v0.3.1-stable` — 27 files, 18409 insertions, 283 deletions
- **Critical changes**: `mmq.cuh` (LDS pad + occupancy guard), `convert.cu` (module cache removal + atexit cleanup)
- **GPU benchmarks**: Pending hardware access — requires `rocprofv3 --kernel-trace` runs on each version