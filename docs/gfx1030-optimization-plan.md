# RDNA 2 (gfx1030) Optimization Plan — Structured

> **Purpose:** Actionable optimization plan for the llama.cpp HIP/ROCm backend on AMD RX 6900 XT (gfx1030).
> Complements `docs/rdna2-optimization-guide.md` (ISA reference) and `docs/rdna2-agent-tasks.md` (task checklist).
> All claims tagged `[ISA]` / `[measured]` / `[supplementary]` / `[hypothesis]`. Untagged = defect.
> **Owner lands commits by hand. Never commit/push.**

---

## 1. Target Identification

### 1.1 Hardware

| Property | Value | Provenance |
|----------|-------|------------|
| GPU | AMD Radeon RX 6900 XT (Navi 21, gfx1030) | `[supplementary]` |
| Architecture | RDNA 2, wave32 native | `[ISA §2.1 p.7]` |
| Compute units | 80 CUs (160 SIMD32, 4 SIMD32/WGP) | `[supplementary]` |
| Memory | 16 GB GDDR6, 512-bit bus | `[supplementary]` |
| Peak bandwidth | 500.6 GB/s read ceiling (measured) | `[measured: gfx1030-directive.md]` |
| L2 cache | 4 MB (Infinity Cache 128 MB above) | `[supplementary]` |
| LDS per WGP / CU | 128 kB / 64 kB (max 64 kB/workgroup) | `[ISA §2.3.1, §10.1]` |
| LDS banks | 64 banks, 2×32-bank halves per WGP | `[ISA §10.1 p.90]` |
| VGPRs per SIMD | 65,536 (16 wave slots/SIMD, 256 VGPRs/wave max) | `[measured: gfx1030-isa-report.md:22]` + `[supplementary: KFD]` |
| GDS | **Absent** (`gds_size_in_kb = 0`) | `[supplementary: KFD topology]` |

### 1.2 Software Stack

| Component | Detail |
|-----------|--------|
| Build system | CMake (not Make — Makefile errors out) |
| Compiler | hipcc / clang with `--offload-arch=gfx1030` |
| Build dir | `build/` (root), `-j12` (verified safe on i7-14700) |
| Build command | `cmake --build build --config Release -j12` |
| Backend | HIP/ROCm, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_NO_VMM=ON` |

### 1.3 Models of Record

| Model | Path | Role |
|-------|------|------|
| Dense (27B IQ3_S, MTP) | `/home/stormrage/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | Primary decode benchmark |
| MoE (35B-A3B IQ3_S) | `/home/stormrage/Downloads/Qwen3.8-35B-A3B-Q4_K_M-exps-IQ3_S-im.gguf` | Expert-width benchmark |

### 1.4 Baselines (numbers of record — do not mix sessions)

| Workload | Fork | Upstream | Provenance |
|----------|------|----------|------------|
| Dense tg | 30.89 t/s | 30.05 t/s | `[measured: AGENTS.md]` |
| MoE tg | 95.19 t/s | 94.11 t/s | `[measured: AGENTS.md]` |
| MTP (enabled) | 47.0 t/s | — | `[measured: AGENTS.md]` |

### 1.5 Target Kernels

| Kernel | File | Role | Current State |
|--------|------|------|---------------|
| MMVQ (`mul_mat_vec_q`) | `ggml/src/ggml-cuda/mmvq.cu` | Decode matvec (ncols_dst=1) | KLONG=0, WIDEN=1, SMALLK=1, 2ROWS=0, LBMIN=1 |
| MMQ (`mul_mat_q`) | `ggml/src/ggml-cuda/mmq.cu` | Prefill matmul (tiled) | Routed selection on by default, never paired-measured |
| Flash Attention | `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu` | Attention | q4_0-q5_0 + q5_0-q5_0 FA quants added |
| vec_dot | `ggml/src/ggml-cuda/vecdotq.cuh` | Decode dot product (IQ3_S sign path still old idiom) | TODO: apply multiply-spread trick |
| IQ3_S loader | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | MMQ tile loading | Sign path rewritten (+19/−4) for IQ3_S; 4 types pending |

### 1.6 Constraints (binding)

1. **GPU access**: only while owner's server is idle. Never launch/stop/query/health-check `llama-server`.
2. **Metric**: t/s only (never ms). PPL/eval-callback are quality anchors, never performance metrics.
3. **One change at a time**: name the full switch vector (§13 of rdna2-optimization-guide.md).
4. **Correctness gate**: `./build/bin/test-backend-ops -b ROCm` 100% pass + model output "Paris" at temp=0.
5. **Measurement**: paired A/B, interleaved, `scripts/bench-exclusive.sh` mandatory. 95% CI must exclude zero.
6. **Never commit/push** — owner lands by hand. Never write PR text, commit messages, or reviewer replies.
7. **No inline eval** for file writes. Use `write`/`edit` tools only.

---

## 2. Parallel Explorer Analysis (4 Roles)

### Role 1: @explorer — Codebase + ISA Census

**Task:** Catalog all RDNA2-relevant code paths, switch states, and ISA mechanism coverage.

**Inputs:**
- `ggml/src/ggml-cuda/mmvq.cu` (calc_nwarps, switch arms, reduction)
- `ggml/src/ggml-cuda/mmq.cu` + `mmq-config-rdna2.cuh` (routed selection, config table)
- `ggml/src/ggml-cuda/fattn-gqa-gfx1030.cu` (FA kernel)
- `ggml/src/ggml-cuda/vecdotq.cuh` (vec_dot, get_int_b2)
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh` (IQ sign paths)
- `docs/rdna2-optimization-guide.md` (ISA reference, §5 checklist)

**Deliverable:** Table of (file:symbol, current state, ISA mechanism applicable, gap/hypothesis).

**Key findings from prior work:**
- MMVQ has NO RDNA2 parameter table — falls through to generic (1 warp) unless a switch arm fires
- 5 IQ types still use old `__vcmpne4`/`__vsub4` idiom in vec_dot and 4 in MMQ loader
- `get_int_b2` does two 16-bit loads (alignment-limited, 2-byte aligned)
- KLONG disabled (Q6_K correctness bug)
- `alloc_failed` is process-global, not per-device
- Non-atomic `static bool logged` in fattn.cu

### Role 2: @explorer — Instrumentation + Measurement

**Task:** Establish measurement baseline and instrument capabilities.

**Inputs:**
- `scripts/gfx1030.py` (probe, bench --ab, profile, isa, audit, census)
- `docs/gfx1030-benchmarking.md` (host rules, paired stats)
- `rocprofv3 -L` (counter availability on gfx1030)

**Deliverable:** 
- Baseline paired A/B numbers (ms/token, sd, CI) for dense and MoE
- Counter availability list (`rocprofv3 -L` output)
- Resource usage from `--save-temps` metadata (VGPR/SGPR/LDS/scratch)

**Key constraints:**
- Dead counters on gfx1030: `SQ_INSTS_VALU`, `SQ_WAIT_ANY`, `SQC_LDS_IDX_ACTIVE`, `FETCH_SIZE`, all `DRAM_*`/`TCC_*`
- `rocprofv3 --stats` does NOT report VGPR/SGPR/LDS — use `--save-temps` + `.s` metadata
- Run-to-run decode sd ≈ 1.56 t/s (~2.3%)

### Role 3: @oracle — Architecture + Risk Review

**Task:** Review optimization candidates for correctness risk and architectural fit.

**Inputs:**
- Current codebase state (switches, landmines, cross-backend symbols)
- ISA reference (rdna2-optimization-guide.md §5 checklist)
- Prior oracle findings (7 audit fixes, KLONG investigation)

**Deliverable:** Prioritized risk assessment for each candidate:
- Correctness risk (will it break output?)
- Performance risk (will it actually help?)
- Reversibility (can we undo cleanly?)
- Interaction effects (does it compose with other switches?)

**Key findings from prior oracle reviews:**
- KLONG has a real correctness bug (not just FP reordering) — produces "333333333"
- `accepted.size()` → `accepted.size() - 1` was a correctness fix in server-context.cpp
- 7 audit fixes applied (W1-W4, I2, I5, I6)
- Remaining debt: I3, I4, I7, I8, missing tests

### Role 4: @fixer — Implementation (after plan approval)

**Task:** Implement approved changes, one at a time, with correctness gates.

**Constraints:**
- One switch/change per diff
- Build with `-j12`, test-backend-ops -b ROCm, "Paris" output check
- Paired A/B measurement before declaring KEEP
- Name full switch vector in every report

---

## 3. Per-Kernel Deliverable Format

Every kernel optimization must produce this structured report before any commit decision:

```markdown
## Task ID: [e.g., L-11]

### Change
- **Description:** [one sentence]
- **Anchor:** [file:line, symbol]
- **ISA citation:** [§section p.page] or "none — code-structure change"
- **Provenance:** [measured: …] / [supplementary] / [hypothesis]

### Implementation
- **Diff:** [files changed, lines added/removed]
- **Switch vector:** [full state of all 5 GGML_RDNA2_MMVQ_* switches]
- **Build:** clean / warnings
- **Correctness:** test-backend-ops -b ROCm [pass/fail], output "Paris"/"333333333"

### Measurement
- **Base artifact / fix artifact:** [path to each .so]
- **Pairs:** [N interleaved pairs]
- **Δ ms/token (paired mean):** [+X.XX ms]
- **sd of per-pair differences:** [X.XX ms]
- **95% CI half-width:** [±X.XX ms]
- **CI decision:** [excludes zero → keep] / [includes zero → investigate / drop]
- **Δ prefill (separate run):** [+X% / N/A]
- **Resource usage:** VGPRs=[N], SGPRs=[N], LDS=[N]B, scratch=[N]B (from .s metadata)

### Decision
- **Result:** KEEP / REVERT / INVESTIGATE
- **Notes:** [tool limitations, dead counters, confounds, anomalies]
```

---

## 4. Prioritized Execution Roadmap

### Tier 1 — High value, low risk (do first)

| Priority | Task | Why | Gate |
|----------|------|-----|------|
| 1 | **L-11** IQ3_S decode sign path (`vecdotq.cuh:1317-1321`) | Decode path for model of record; same transform as L-07 which landed for prefill | Decode A/B, ≥40 pairs |
| 2 | **L-16** MMQ routed selection A/B (`mmq.cu:162`) | On by default, never paired-measured; could be net-negative | `GGML_HIP_MMQ_ROUTED_GFX1030=0` vs `=1`, ≥40 pairs, MoE model |
| 3 | **L-13** `__threadfence()` removal (`mmvq.cu:916-937`) | `__syncthreads()` already orders shared memory; fence is pure per-token cost | A/B, ≥40 pairs |
| 4 | **L-09** H14 wide loads (`vecdotq.cuh:25-35`) | 5,302 `global_load_dword` (4B) vs 932 `dwordx4` (16B); 2-byte alignment is the blocker | Correctness-first: widen only if alignment guaranteed, then A/B |

### Tier 2 — Medium value, medium risk

| Priority | Task | Why | Gate |
|----------|------|-----|------|
| 5 | **L-08** Sign path for 4 remaining IQ types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS in loader + IQ3_S in vec_dot) | ~15% per type (hypothesis); 5 sites total | Per-site A/B, one at a time |
| 6 | **L-06/L-17** FA quant pairs (`q8_0-q5_0` vs `q4_0-q5_0`) | Removes full-KV f16 conversion; per-consumer (different KV types) | A/B each pair against model's actual `-ctk/-ctv` |
| 7 | **L-04** 2ROWS re-enable (`mmvq.cu:31`) | Shares q8_1 loads across 2 rows; traffic claim is hypothesis | A/B; check launch bounds interaction |
| 8 | **L-12** Non-atomic latches in fattn.cu (`fattn.cu:701-703, :721-723`) | Data race as written; benign for logging but unreliable | No perf claim; correctness only |

### Tier 3 — High risk, needs investigation first

| Priority | Task | Why | Gate |
|----------|------|-----|------|
| 9 | **L-01** KLONG fix (Q6_K correctness) | +3.75% non-MTP tg; currently disabled; MTP guard inert | GPU debugging (rocgdb); numerical A/B; "Paris" output |
| 10 | **L-10** MTP conditional KLONG | Depends on L-01; inert while KLONG=0 | After L-01 fixed |
| 11 | **L-14** `alloc_failed` per-device | Process-global state; cross-device contamination | Two-GPU test or moved tensor |
| 12 | **L-02/L-03** WIDEN/SMALLK A/B | Already enabled; need paired measurement to confirm value | A/B two .so's; check interaction matrix |

### Tier 4 — Audit + hygiene (CPU-only, do anytime)

| Task | Description | Agent |
|------|-------------|-------|
| A1 | Wave size audit (all kernels wave32) | @explorer |
| A2 | LDS bank conflict audit | @explorer |
| A3 | VGPR pressure audit (corrected arithmetic) | @explorer |
| A4 | Global memory coalescing audit (load width) | @explorer |
| A5 | Redundant fence/barrier census | @explorer |
| A6 | Instrument policy audit (no `rocprofv3 --stats`) | @explorer |
| A7 | No-go list verification (closed lanes) | @explorer |
| A8 | Cross-backend landmine census | @reviewer |
| I3, I4, I7, I8 | Audit debt (doc ref, q5_0-q5_0 build cost, VK_BIN path, env parsing) | @fixer |

### Dependency chain (from gfx1030-directive.md §1)

```
WP-F (GQA multi-row verify) → SLC weight-stream / Probe D → 3.6 µs floor test
  → WP-B (MoE expert-width gate) → H14 wide decode loads → WP-A (scale pooling)
```

L-series sits underneath: L-01/L-10 and L-15 are prerequisites for any MMVQ measurement claim. L-07/L-08/L-11 are instruction-mix work that can proceed without GPU-bound packages. L-06/L-17 are gated on cache types the model actually runs.

---

## 5. Measurement Protocol

### 5.1 Pre-flight (mandatory)

```bash
# GPU must be idle
pgrep -a -f "llama-server|llama-bench"   # must be empty
rocm-smi --showmemuse                      # ≈ 0
```

### 5.2 Correctness Gate (before any measurement)

```bash
cmake --build build --config Release -j12
./build/bin/test-backend-ops -b ROCm          # 100% pass
# Model output check at temp=0:
./build/bin/llama-cli -m <model> -ngl 99 -c 256 -fa on -ctk q4_0 -ctv q5_0 \
  --reasoning-budget 0 -p "The capital of France is" -n 10 --temp 0 --no-display-prompt
# Expected: "Paris" (not "333333333")
```

### 5.3 Paired A/B Measurement

```bash
# Build two artifacts (base and fix)
# Swap them interleaved, one pair at a time:
scripts/gfx1030.py bench --ab <base.so> <fix.so> --pairs N

# Compute stats offline:
scripts/gfx1030.py bench --ab-stats "d1,d2,..."
```

**Required pairs:** `n = (1.96 * sd_pair / delta)^2`
- At `sd_pair = 0.64 ms`: 0.20 ms needs ~40 pairs, 0.14 ms needs ~80, 0.08 ms needs ~250
- `--min-detectable <ms>` prints the count

**Acceptance:** 95% CI excludes zero. Delta smaller than CI → `INVESTIGATE`, not `KEEP`.

### 5.4 Resource Usage (not from profiler)

```bash
cmake --build build --config Release -j12 -- -save-temps
grep -E "amdhsa_(next_free_vgpr|next_free_sgpr|group_segment_fixed_size|private_segment_fixed_size)" <kernel>.s
```

- `amdhsa_next_free_vgpr` → VGPRs/wave (already rounded to 16-dword granularity)
- `amdhsa_next_free_sgpr` → SGPRs/wave
- `amdhsa_group_segment_fixed_size` → static LDS bytes
- `amdhsa_private_segment_fixed_size` → scratch/spill bytes (0 = no spills)

**Occupancy arithmetic:** `waves/SIMD = min(16, 65536 / (VGPRs × 32))`, `blocks/CU = min(waves/SIMD / warps_per_block, 65536 / LDS_per_block)`

### 5.5 Instrument Selection

| Question | Instrument | Notes |
|----------|-----------|-------|
| VGPR/SGPR/LDS/scratch/occupancy | `--save-temps` + `.s` metadata | NOT `rocprofv3 --stats` |
| Bank conflicts | `rocprofv3 -L` then `--pmc SQ_LDS_BANK_CONFLICT` | Counter may not exist on this part |
| Per-class bandwidth | `scripts/gfx1030.py profile` | Graph-free kernel trace |
| Instruction mix | `scripts/gfx1030.py isa` | Static mix + 5% ALU gate |
| Structural checks | `scripts/gfx1030.py audit` | Mechanical source checks |
| Graph shape | `scripts/gfx1030.py census` / `rotcheck` | CPU-only |
| Backend correctness | `./build/bin/test-backend-ops -b ROCm` | Not `ctest -L main` |
| A/B decision | `scripts/gfx1030.py bench --ab` | Paired, interleaved |

**Dead counters on gfx1030:** `SQ_INSTS_VALU`, `SQ_WAIT_ANY`, `SQC_LDS_IDX_ACTIVE`, `FETCH_SIZE`, all `DRAM_*`/`TCC_*`.

### 5.6 Exclusivity Rules

1. `scripts/bench-exclusive.sh` is mandatory for any measured claim
2. One measurement process at a time — never backgrounded behind another
3. Same model, same prompt, same seed, same `-fa`/`-ctk`/`-ctv`/`-ngl`/`-ncmoe` on both arms
4. Report: ms/token, paired mean delta, sd of per-pair differences, 95% CI half-width

---

## 6. Known Landmines

| # | Landmine | Anchor | Consequence | Action |
|---|----------|--------|-------------|--------|
| 1 | Cross-library symbol: `ggml_cuda_klong_disabled` defined in HIP backend, declared in core | `src/llama-context.cpp:4`, `common.cuh:1670`, `mmvq.cu:10` | Unresolved symbol in non-HIP builds | Move to backend-visible interface |
| 2 | Process-global `alloc_failed` latch | `common.cuh:1705-1712` | One failure on GPU0 disables fast path on all GPUs | Per-device state (L-14) |
| 3 | Non-atomic `static bool logged` in fattn.cu | `fattn.cu:701-703, :721-723` | Data race as written; unreliable log | `std::atomic<bool>` + `exchange` |
| 4 | Env kill switches (undocumented) | `GGML_HIP_MMQ_ROUTED_GFX1030` (`mmq.cu:135`), `GGML_CUDA_MMQ_PREC` (`mmq.cu:99`) | Stray env var = different build | Print env in report; unset before benchmarking |
| 5 | Compile-time switches select different kernels | `GGML_RDNA2_MMVQ_*` (`mmvq.cu:14-31`) | A/B is artifact swap, not flag flip | Two `.so`s, `bench --ab` |
| 6 | Silent fallthrough: no RDNA2 MMVQ table | `mmvq.cu:1277-1290` | "RDNA2 path" is mostly generic + switches | Read §2 before assuming a row exists |
| 7 | Dead counters on this part | `SQ_INSTS_VALU`, `SQ_WAIT_ANY`, etc. | PMC recipe from another card measures nothing | `rocprofv3 -L` first |

---

## 7. Switch Interaction Matrix

All five `GGML_RDNA2_MMVQ_*` switches are read in `calc_nwarps` (`mmvq.cu:619-720`) under mutually exclusive guards. All key off `ncols_dst == 1`.

| Switch | Guard | Composes with | Excludes |
|--------|-------|---------------|----------|
| `KLONG` | `!small_k && halve_iters` | — | `SMALLK`, `WIDEN` |
| `WIDEN` | `!small_k && !halve_iters` | — | `SMALLK`, `KLONG` |
| `SMALLK` | `small_k` (evaluated first) | suppresses others | `KLONG`, `WIDEN` |
| `2ROWS` | separate arm (`:696-712, :742-746`) | orthogonal | — |
| `LBMIN` | `__launch_bounds__` 2nd arg (`:752`) | orthogonal | — |

**Consequences:**
- `SMALLK=1` + `KLONG=1` together measures neither (`small_k` wins)
- `KLONG` requires `halve_iters`, which is why the MTP guard exists
- Any report must name the full switch vector, not just the one changed

---

## 8. No-Go List (closed lanes — do not reopen without contradicting evidence)

| Closed lane | Verdict | Evidence |
|-------------|---------|----------|
| "dot pipes idle — kernels dequantise then scalar FMA" | Refuted: 3,008 `v_dot4c_i32_i8` vs 568 FMA | `gfx1030-directive.md §10` |
| "`ds_bpermute` unharvested" | Refuted: 39,381 occurrences | `gfx1030-directive.md §10` |
| "`S_CLAUSE` unemitted" | Refuted: 18,873 occurrences | `gfx1030-directive.md §10` |
| "`__ldg`/`__ldcg` wrappers give cache control" | Wrappers are no-ops | `gfx1030-directive.md §10` |
| "prefill ceiling 957.6 t/s" | Refuted: dp4a multiply IS the dot op; ceiling ~1,915 t/s | `gfx1030-directive.md §2` |
| 2 blocks/CU on `mul_mat_q` | Closed by LDS (2 × 45.9 KiB > 64 kB); VGPR half contested but irrelevant | `gfx1030-isa-report.md` |
| `ds_bpermute`/`DS_SWIZZLE` "free shuffle" as new lane | Already generated by shuffle reductions | `gfx1030-directive.md §10` |
| SFU pipe | Negligible in profile | `gfx1030-directive.md §10` |
| GDS append/consume/GWS (P-01…P-03) | Hardware absent (`gds_size_in_kb = 0`) | KFD topology |
| "`≤ 64 VGPRs/wave for 4 waves/CU`" | Wrong: pool is 65,536/SIMD, not 128/SIMD | `gfx1030-isa-report.md` |

---

## 9. Open Questions (ISA does not answer these)

| # | Gap | Impact | Mitigation |
|---|-----|--------|------------|
| 1 | No tuning guidance in ISA | Cannot derive optimum from manual | Measure; `gfx1030.py profile` |
| 2 | No V_DOT* throughput spec | Cannot predict quantized speedup | Benchmark each variant |
| 3 | No LDS-vs-DPP latency comparison | Cannot choose a priori | Benchmark or keep compiler's choice |
| 4 | No address→bank mapping in ISA | Bank conflicts unpredictable statically | Structural argument; counter may not exist |
| 5 | No VALU/LDS/VMEM latency tables | Cannot model stalls analytically | Counters only |
| 6 | No VGPR operand-bank conflict rules | Some stalls inexplicable | Out of reach |
| 7 | No CU-vs-WGP LDS mode mechanism | Cannot decide split from ISA | Compiler default (WGP mode) |
| 8 | No guidance on what HIP already emits | "Unused instruction" claims keep being wrong | Disassemble first |
| 9 | No cache sizes in ISA | Cannot size working sets | `[supplementary]`: 4 MB L2, 128 MB Infinity Cache |
| 10 | No bandwidth specification | No theoretical peak from manual | `[measured]`: 500.6 GB/s read ceiling |
| 11 | KLONG Q6_K correctness bug | KLONG off, L-10 inert | GPU debugging (rocgdb); FP associativity suspect |

---

## References

- `docs/rdna2-optimization-guide.md` — full ISA reference (643 lines, §5 checklist, §12 landmines, §13 switch matrix)
- `docs/rdna2-agent-tasks.md` — task checklist (498 lines, 50+ tasks, 6 phases)
- `docs/gfx1030-directive.md` — campaign directive, closed-lane ledger, ISA mechanism inventory
- `docs/gfx1030-isa-report.md` — occupancy measurement, ISA quotes
- `docs/gfx1030-test-plan.md` — queued T0-T7 experiments with pass/fail predicates
- `docs/gfx1030-benchmarking.md` — host rules, paired statistics, resolving power
- `AGENTS.md` — build/test/architecture/conventions (updated with RDNA2 guide as mandatory reading)
- `/tmp/opencode/rdna2-isa.txt` — extracted PDF text (283 pages)
- `/tmp/opencode/klong_test/README.md` — KLONG numerical A/B test harness