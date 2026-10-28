# gfx1030 checkpoint — review manifest

Repo: `/home/stormrage/llama.cpp`, base HEAD `d327bf1f0`. Everything below is uncommitted working-tree state. This file is documentation only; no source was changed to produce it.

## 1. Per-area inventory

Counts are from `git diff --stat` on the files in each area. Files that appear in more than one area (notably `ggml-cuda.cu`) make those counts overlap; the table records the overlap rather than hiding it.

| area | files (actual working tree) | diff stat | what it changes | evidence it is safe |
|---|---|---|---|---|
| prequant arena | `ggml/src/ggml-cuda/common.cuh`, `ggml-cuda.cu`, `unary.cu`, `mmvq.cu`, `tests/test-backend-ops.cpp` | 5 files, +1183 / -25 | Adds a per-context q8_1 prequant arena: fused RMS_NORM+MUL writes quantised blocks once; downstream matvecs skip their own `quantize_q8_1` launch. | `quantize_q8_1` 127.65 → 24.0 launches/token; `UNARY_MUL_PREQUANT` 3/3, `GLU_PREQUANT` 2/2, `RMS_NORM_MUL_PREQUANT` 2/2; `MUL_MAT` 2445/2445; `FLASH_ATTN_EXT` 4046/4046; eval-callback zero diff; perplexity 6.9047 ± 0.37351 identical. |
| pre-norm residual ADD fusion | `ggml/src/ggml-cuda/norm.cu`, `norm.cuh`, `ggml-cuda.cu` | 3 files, +670 / -24 (`ggml-cuda.cu` overlaps area 1) | Fuses the residual ADD into RMS_NORM (`do_preadd`) so the norm reads the residual and writes the sum in one pass. | `RMS_NORM_PREADD_MUL` 3/3; `RMS_NORM_MUL_ADD` 37/37 (pre-existing) still passing; census 1536.2 → 1535.2 (1 launch). |
| RDNA2 MMQ config table | `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh` (`mmq-config-rdna2.cuh` is already tracked and unchanged here) | 2 files, +14 / -1 | Routes HIP RDNA2 (gfx1030) through its own MMQ config and stops the old silent fall-back to that config on other arches (host pass keeps a stub; device pass errors instead). | Needs validation: no dedicated RDNA2 MMQ result was recorded; relies on the shared `MUL_MAT` 2445/2445 pass. |
| prefill kernel work | `ggml/src/ggml-cuda/vecdotq.cuh`, `gated_delta_net.cu`, `mmvf.cu`, `ggml/CMakeLists.txt` | 4 files, +91 / -17 | `ggml_cuda_mmq_scale_sumi` in the K-quant vecdot paths; prefill broadcast on the recurrent (GDN) path; dual-matvec fusion (`fusion.x2`) support behind the SSM alpha/beta projections; adds `q8_0-q5_0` to the FA quant combo. | Measured during the prefill lane: pp512 593.4 → 598.9 t/s; GDN broadcast −12.1 % kernel time; bit-identical results at the time. **Needs validation:** no stored command/output in the tree. |
| gfx1030 GQA decode attention | `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu` (new, 291 lines), `fattn.cu`, `fattn.cuh`, `fattn-tile.cuh` | `fattn.cu` +150, `fattn.cuh` +11, `fattn-tile.cuh` +27, new file 291 lines | Adds a GQA decode flash-attention kernel (one K/V read per KV head), admitted only for gfx1030, plus an RDNA2 flat-pointer fix in the tile loader for unaligned KV lengths. | Multi-sequence correctness fix; 4.4–5× over the generic path at 32k. |
| state-safety patches | `src/llama-batch.h`, `src/llama-graph.cpp`, `src/llama-graph.h`, `src/llama-kv-cache-dsv4.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/llama-memory-hybrid-iswa.cpp`, `src/llama-memory-hybrid.cpp`, `src/llama-memory-recurrent.cpp`, `src/models/qwen35.cpp` | 9 files, +96 / -32 | Centralises the recurrent rollback split rule in `llama_n_keep_tail()` with asserts at every call site; adds a direct single-cell state alias in `build_rs`; routes qwen35 through it; adds dual-matvec fusion for the SSM alpha/beta projections. | Campaign reports the state-safety suite green. Needs validation: no command or output is stored in the tree, so this cannot be re-checked here. |
| scripts | `scripts/gfx1030.py`, `ic_expose.py`, `common.py`, `common.sh`, `quality_ladder.sh`, `bench-fork-vs-upstream.sh`, `bench-exclusive.sh`, `test-graph-census.cpp`; `tests/CMakeLists.txt` (builds the census test) | 8 new files (2818 + 648 + 16 + 9 + 295 + 201 + 50 + 1026 lines); `tests/CMakeLists.txt` +5 | gfx1030 harness, ISA census, quality ladder, benchmark wrappers, graph census test. | Needs validation: no recorded test result beyond the census 1536.2 → 1535.2 launch count cited under area 2. |
| docs | 16 planning `.md` at repo root (ARCHITECTURE, BENCHMARKING_LLAMACPP, FINAL_RECOMMENDATION, GFX1030_MASTERPLAN_ROUND3, GFX1030_QWEN35_MASTERPLAN, HIPFIRE_27B_GFX1030_TEST, HIPFIRE_FINDINGS, HIPFIRE_PORT_ROUND2, HIPFIRE_QWEN35_PORT_BRIEF, HYPOTHESIS_HIP_LONG_CONTEXT_DECODE, MASTER_PLAN, N1_N4_PATCH_SPEC, PLAN_3MS, PLAN_GFX1030_DECODE_V2, PLAN_GQA_DECODE, PLAN_HIPFIRE), `docs/gfx1030-benchmarking.md`, `docs/gfx1030-isa-report.md` | all new/untracked | Planning, ISA, and benchmarking notes for the campaign. | Documentation only; no runtime effect. |

### Unclassified (modified, fits none of the areas above)

- `.gitignore` — ignores machine-local gfx1030 state.
- `AGENTS.md` — replaced with the local agent-instruction set.
- `common/common.cpp` — `LLAMA_FORCE_N_RS_SEQ` diagnostic env var.
- `examples/speculative-simple/speculative-simple.cpp` — token-id dump diagnostic and null-speculator fallback.
- `ggml/src/ggml-cuda/vendors/hip.h` — `cudaEventCreate`/`cudaEventElapsedTime` macros.
- `ggml/src/ggml-hip/CMakeLists.txt` — gfx1030 compile options for the new TUs.
- `tools/cli/cli-context.cpp`, `tools/cli/cli-context.h` — speculative-decode draft/accept timings in the CLI.

## 2. Evidence summary

- prequant arena: `quantize_q8_1` 127.65 → 24.0 launches/token; `UNARY_MUL_PREQUANT` 3/3, `GLU_PREQUANT` 2/2, `RMS_NORM_MUL_PREQUANT` 2/2; `MUL_MAT` 2445/2445; `FLASH_ATTN_EXT` 4046/4046; eval-callback zero diff; perplexity 6.9047 ± 0.37351 identical.
- preadd fusion: `RMS_NORM_PREADD_MUL` 3/3; `RMS_NORM_MUL_ADD` 37/37 pre-existing still passing; census 1536.2 → 1535.2 (1 launch).
- attention: multi-sequence correctness fix; 4.4–5× over the generic path at 32k.
- state-safety: recurrent rollback splits centralised and asserted; build_rs direct alias for single-cell state; qwen35 routed through it; suite reported green (Needs validation: no stored command/output).
- RDNA2 MMQ, scripts, docs: no dedicated recorded result (Needs validation).

## 3. Suggested commit message (draft — owner edits and commits)

```
ggml-cuda : gfx1030 decode optimisations and recurrent state-safety fixes

- prequant arena (q8_1) in the CUDA backend
- pre-norm residual ADD fusion in RMS_NORM
- RDNA2 MMQ config table
- gfx1030 GQA decode attention kernel
- recurrent state-safety patches (rollback splits, build_rs alias, qwen35)
- scripts, benchmarking docs, and planning notes
```

## 4. Not included (must never be committed)

Machine-local state (already ignored by the `.gitignore` change):
- `scripts/.baseline.json`, `scripts/.cost_model.json`, `scripts/.isa_baseline.json`, `scripts/.interventions.jsonl`
- `scripts/.rotcheck/`
- `.rocprofv3/`
- `build*`, `build-hip-*` directories and downloaded `models/*.gguf`.

## 5. Review order

1. Shared CUDA files first — `common.cuh`, `ggml-cuda.cu`, `norm.cu`/`norm.cuh`, `mmq.cuh`, `fattn.cu`/`fattn.cuh`: highest blast radius, used by every backend path.
2. gfx1030-only files — `fattn-gqa-gfx1030.cu`, `mmq-config-rdna2.cuh`: isolated by device admission, lowest risk of affecting other targets.
3. State-safety `src/` changes — centralised rule plus asserts; read `llama-batch.h` first.
4. Scripts and docs last — tooling only, no runtime effect.
