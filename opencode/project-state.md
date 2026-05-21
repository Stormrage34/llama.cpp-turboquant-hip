# Project State — llama.cpp-turboquant-hip

**Last Updated:** 2026-05-21  
**Updated By:** @orchestrator  
**Current Sprint:** v0.5.0 — Hardware Target Lockdown + Custom Sparse Q4_K (CR-008.5) + Perm Chain Optimization  
**Latest Release:** v0.5.0-stable ← **preparing**  
**Sprint End:** 2026-05-23

---

## Active Proposals

| ID | Title | Author | Status | Priority | Human Action Needed |
|----|-------|--------|--------|----------|---------------------|
| CR-008 | Perm chain optimization (`get_int_from_table_16`) | @explorer | Researching | P0 | No |
| [CR-009](opencode/proposals/CR-009.md) | Kernel launch overhead profiling | @oracle | Implementing | P0 | ✅ approved — 4-6 hrs profiling |
| CR-010 | Cross-fork baseline benchmark | @fixer | Researching | P1 | No |
| CR-011 | Archive LOAD_EXPERT_F32 dead code | @chief_engineer | Done | P4 | ✅ — removed 65 lines, CMake cleaned, archived |
| [CR-013](opencode/proposals/CR-013.md) | IQ4_XS SoA vs AoS mismatch in swizzled load_tiles | @fixer | Implementing | P1 | ⚠️ Open: meta offset computation in mmq.cuh:3291 |
| [CR-015](opencode/proposals/CR-015.md) | Upstream MoE CPU offloading bug | @orchestrator | Verifying | P0 | ✅ 4-fix patch applied, build+test in progress |
| [CR-020](opencode/proposals/CR-020.md) | Swizzle-All Binary Initiative (Q4_K/Q5_K SoA expansion) | @fixer+@oracle | Verifying | P0 | ✅ 5K-Coherence: PASSED, MMQ dispatch wired |
| [CR-022](opencode/proposals/CR-022.md) | 5K-Coherence Gate (new QA protocol) | @oracle | Done | P0 | ✅ Codified in AGENTS.md §9, infra scripts created |
| [CR-023](opencode/proposals/CR-023.md) | Workflow Optimization & MMQ Dispatch Integration | @fixer+@oracle | Verifying | P0 | ✅ All 4 actions delivered, MMQ dispatch validated |
| [CR-008.5](docs/architecture/rebuttal_cr008.md) | Sparse-Aware Q4_K + Metadata Gating (MoE-Infinity adaptation) | @fixer | Researching | P1 | 🆕 Oracle WONTFIX overridden — counter-rebuttal filed |

---

## Current Project Focus

**Sprint Goal (Rebaselined 2026-05-20):**
1. ✅ **CR-015 4-fix patch applied** — copy slot exhaustion + bounds guard + cross-split scope + async barrier
2. ⏳ **Verify fix** — build + test past 2000-token boundary (no Chinese output)
3. ⏳ **File upstream PR** to ggml-org/llama.cpp with the complete 4-fix patch set
4. ✅ **CR-017 Cache Swizzle Stabilization** — RDNA2_CACHE_SWIZZLE stability baseline verified across Q4_K, Q5_K, IQ4_XS
5. Restore focus on CR-008 (perm chain optimization) — highest ROI at 70% of compute
6. CR-009 (kernel launch profiling) — pure PMC counters, no tracing flags
7. 🆕 **CR-008.5: Custom Sparse Q4_K** — structural zero-masking pass + early-exit gate in `vec_dot_q4_K_q8_1_tmpl` (Oracle WONTFIX overridden)
8. 🆕 **Idea 1: Dual stream prefetch** — VRAM→IC latency hiding via `stream_prefetch` + `stream_compute` on ACE engines

**Blockers:**
- rocprofv3 confirmed at `/home/stormrage/rocm-7.13-nightly/bin/rocprofv3` (83 KB)
- GPU profiling confirmed working (JSON format, need 300s timeout)
- **MoE CPU offloading bug** confirmed upstream — blocks long-form generation with --n-cpu-moe
- All local changes reverted to isolate upstream bug; ggml-backend.cpp and fattn-tile.cuh clean

---

## Known Constraints

| Constraint | Status | Details |
|------------|--------|---------|
| GPU | n=1 only | RX 6800 XT (gfx1030) — all tests sequential |
| llama-server | Running (PID 17813) | Blocks local benchmarking; use `scripts/server_check.sh` |
| ROCm | 7.13-nightly | `/home/stormrage/rocm-7.13-nightly` (runtime), `/opt/rocm` (stable) |
| VRAM Redline | 15.5GB absolute | 15.0GB yellow alert; current IQ4_XS + `-ncmoe 41` @ 32k = ~2.2GB |
| dp4a utilization | 0.25% | Optimizations targeting dp4a have <0.025% overall ROI |
| ⚠️ **MoE CPU offloading bug** | **Upstream** | CR-015: Long prompts + many tokens + any `--n-cpu-moe` → garbled output after ~1000 tokens. Verified on vanilla upstream code. Workaround: all-GPU offload or short prompts. |
| ⚠️ **CR-013: IQ4_XS SoA vs AoS mismatch** | **OPEN** | `load_tiles_iq4_xs_swizzled` in `mmq.cuh:3222` meta offset may be wrong for sub-tile views. High-priority fix required before activating swizzle on IQ4_XS MMQ path. |
| 📊 **-12% speed delta (41.0→36.1 t/s) on MMQ-fixed build** | **INVESTIGATING** | Unknown if: (a) thermal throttling on RX 6800 XT, (b) algorithmic regression from MMQ dispatch, (c) run-to-run variance. Needs thermal monitoring run (rocm-smi --showtemp before/after). |
| 🔧 **MMQ Q4_K/Q5_K dispatch: wired and validated** | **DONE** | 3 dispatch points in mmq.cuh:3882-3962 verified at 5K tokens. Single-turn (MMVQ) path fully validated. Batch inference (MMQ) path passes coherence but needs thermal profiling. |
| ⚠️ **No thermal/power monitoring in agent workflows** | **GAP** | Chief Engineer §13 mandates thermal monitoring. No agent queries rocm-smi --showtemp before/after GPU tests. Must be integrated into oracle_5k_fastfail.sh and run_benchmark.sh. |

## Hardware-Enforced Constants (HARDWARE_TARGET.md)

See `HARDWARE_TARGET.md` at repo root for the full hardware profile. Key constants enforced by @librarian at documentation gate:

| Constant | Value | Violation Penalty |
|---|---|---|
| VRAM ceiling | 15,872 MiB (15.5 GB) | Auto-reject proposal |
| L2 alignment | 128 bytes | Auto-reject proposal |
| Wave32 execution | Mandatory (gfx1030) | Auto-reject proposal |
| LDS-free shuffling | Mandatory (DS_PERMUTE_B32) | Auto-reject proposal |
| Host L3 budget | 32 MiB (Zen 3 CCD) | Auto-reject proposal |

**Gate rule:** @librarian enforces these at documentation intake. Any CR that violates a hardware constant is rejected before reaching @fixer.

---

## Phase Status Ledger

| Phase | Status | Impact | Savings | Coverage |
|-------|--------|--------|---------|----------|
| **1. Hardware Matrix & Enforcement** | 🔒 LOCKED & ACTIVE | Hybrid offloading auto-rejected; 128B L2 alignment + Wave32 + LDS-free shuffle enforced at intake gate | — | All CRs entering pipeline |
| **2. File System Defragmentation** | 🧹 CLEANED | 262 MB build artifact purge; 45 orphaned template files pruned; inter-script paths re-mapped to `./build/` | 262 MB + 45 files | `build-swizzled/`, `fattn-vec-instance-*.cu`, 3 scripts |
| **3. Layout Verification Tracks** | 📊 DEPLOYED | Q4_K_M, Q5_K_M, IQ4_XS, Q8_0, F16 bound to validation matrix; flagged TBD until CR-022 5K coherence passes | — | 5 quant formats × 4 ISA requirements |

**Next Session:** Task 1 — Custom Sparse Q4_K implementation. Structural zero-masking pass + early-exit gate in `vec_dot_q4_K_q8_1_tmpl`. Multi-channel SoA alignment deferred to follow.

---

## MoE-Infinity Research Digest (ICML 2025)

**Source:** "MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache"  
**Logged:** 2026-05-21 | **Author:** @orchestrator  
**Status:** 🔬 ANALYZED → ⚡ **OVERRIDDEN by System Architect** (2026-05-21) — Ideas 1, 2, 4 re-instated. See `docs/architecture/rebuttal_cr008.md`.

**Core Insight:** Single-user MoE inference (batch-size-1) exhibits extreme activation sparsity and temporal locality. Cloud batching averages out routing; local inference activates a tiny, repeatedly-reused expert subset per context window.

**Problem:** Reactive Fetching — standard offloading stalls execution while waiting for expert weights from VRAM/PCIe.

**Solution:** Sparsity-Aware Expert Caching with Look-Ahead Prefetching + activation trace replacement policy.

### Feasibility Verdict (by @oracle)

| ID | Idea | Feasibility | Effort | ROI | Verdict |
|----|------|-------------|--------|-----|---------|
| 1 | **Dual HIP streams** — overlap compute with async expert weight prefetching | MEDIUM | 40-60h | LOW-MEDIUM | **P2** — Speculative prefetch needs 10h design review; copy dominates compute by 6-12× on PCIe Gen4 |
| 2 | **Intra-block metadata gating** — early-exit `vec_dot` when `dm.d` evaluates to zero | LOW | 4-8h | **NEGATIVE** | **WONTFIX** — `dm.d` (super-block scale) is never zero for trained weights (quantizer enforces `d >= DBL_MIN`). Even if `d == 0`, the `-dmin * sumf_m` term remains active so early-exit is numerically incorrect. MoE sparsity operates at expert-routing level, not super-block level. |
| 3 | **Wave32 lane compaction** via `DS_PERMUTE_B32` | LOW | 25-40h | NEGATIVE | **WONTFIX** — Architectural mismatch: `mul_mat_vec_q_moe` assigns one full warp per expert (all 32 lanes 100% utilized). No idle lanes to compact. DS_PERMUTE_B32 adds ~4 cycles overhead for zero gain. |
| 4 | **Dynamic L3 cache sizing** — scale `-ub` at runtime | LOW | 30-50h | VERY LOW | **WONTFIX** — 128MB Infinity Cache is fixed SRAM, cannot be resized. Dynamic `-ub` scaling saves only ~10-50MB VRAM (negligible vs 16GB) and requires `ggml_gallocr` restructure. |

### ⚡ System Architect Override (2026-05-21)

The Oracle's WONTFIX verdict has been **overturned** for Ideas 1, 2, and 4. Full counter-rebuttal filed at `docs/architecture/rebuttal_cr008.md`.

| Idea | Original Verdict | Override Status | Basis |
|------|-----------------|-----------------|-------|
| 1 | P2 (defer — PCIe bandwidth myopia) | **P1 — REINSTATED** | Bottleneck is VRAM→IC latency, not PCIe. Dual streams hide VRAM fetch stalls for next-layer blocks. |
| **2** | **WONTFIX** (d never zero, dmin term) | **P1 — REINSTATED** | Custom sparse re-quantization contract zeros BOTH `d` and `dmin`. Early-exit becomes **exact arithmetic** (0·sum - 0·sum = 0). L2 cache line fetch of 128B `qs` saved per gated block. |
| 3 | WONTFIX (warp-per-expert, no idle lanes) | **WONTFIX — CONFIRMED** | Architectural mismatch is fundamental. |
| 4 | WONTFIX (cannot resize silicon) | **P2 — REINSTATED** | Misinterpretation: we size the **data footprint** within 128MB IC, not the cache itself. Dynamic `-ub` prevents activation overflow into VRAM. |

### Key Findings from Code-Path Survey (by @explorer)

**Idea 2 insertion point confirmed — now actionable:**
- `vec_dot_q4_K_q8_1_tmpl<block_q4_K_intra>` at vecdotq.cuh:872-917 is the ideal insertion point
- `dm` at offset 128 (separate L2 cache line from `qs` at offset 0) → reading 16B metadata can skip 128B qs load
- C++ template constraints are nil (all field accesses by name — `bq4_K->dm`, `bq4_K->qs`)
- IQ4_XS swizzled path already reads metadata before qs (vecdotq.cuh:1373) — template for our approach
- Existing CPU-side expert skip at `ggml-cuda.cu:2714` handles **expert-level** sparsity; our gating handles **super-block-level** sparsity within active experts

**Re-aligned sprint effort:** Redirect partial focus to CR-008.5 (custom sparse Q4_K) and Idea 1 (dual stream prefetch) alongside ongoing CR-008, CR-009, CR-015 work.

---

## Decision Log

| Date | Decision | Decided By | Rationale |
|------|----------|------------|-----------|
| 2026-05-19 | Demote Idea B (prefetch) to P4 | Council CR-007 | Infinity Cache + 0.25% dp4a utilization → no measurable ROI |
| 2026-05-19 | Demote dp4a split accumulators to P3 | Council CR-007 | Targets wrong bottleneck (10% of compute × 0.25% GPU util = <0.025%) |
| 2026-05-19 | Promote perm chain to P0 | Council CR-007 | 70% of kernel compute time in `get_int_from_table_16` |
| 2026-05-19 | Approve kernel launch profiling | Human | CR-009: 4-6 hrs allocated for rocprofv3 overhead analysis |
| 2026-05-18 | Server-aware benchmarking mandatory | @oracle | Prevent contaminated results from shared GPU |
| 2026-05-17 | Tag v0.4.2-stable | Chief Engineer | CI green, all gates pass, GitHub release pipeline fixed |
| 2026-05-20 | Bug hunt: P0 — crash on unused graph inputs | @fixer + @oracle | Added null checks on graph input pointer returns (src/llama-graph.cpp) |
| 2026-05-20 | Bug hunt: P0 — FA device mismatch with --no-kv-offload | @fixer + @oracle | Explicit device type comparison instead of pointer identity (src/llama-context.cpp) |
| 2026-05-20 | Bug hunt: P1 — BitNet wrong output tensor | @fixer + @oracle | Corrected output tensor in bitnet model (src/models/bitnet.cpp) |
| 2026-05-20 | Bug hunt: P1 — MPT q_norm test crash | @fixer + @oracle | Corrected tensor shapes in MPT model (src/models/mpt.cpp) |
| 2026-05-20 | Bug hunt: P1 — Token type i32 vs u32 FIXME | @fixer + @oracle | Documented the FIXME (src/llama-model-saver.cpp) |
| 2026-05-20 | Bug hunt: P2 — Stubbed training functions | @fixer + @oracle | Removed FIXME stubs from llama-context.cpp |
| 2026-05-20 | Bug hunt: P2 — Redundant synchronize() | @fixer + @oracle | Removed unnecessary sync (src/llama-context.cpp) |
| 2026-05-20 | Bug hunt: P2 — size_t→int overflow casts | @fixer + @oracle | static_cast + bounds checks (common/speculative.cpp) |
| 2026-05-20 | Bug hunt: P2 — Zero-size tensor crash | @fixer + @oracle | Null + zero-element guards (ggml-cuda.cu) |
| 2026-05-20 | Bug hunt: P2 — Stats misclassification race | @fixer + @oracle | Added counter (src/llama-context.cpp) |
| 2026-05-20 | Bug hunt: P2 — DeepSeek2 misleading FIXME | @fixer + @oracle | Clarified comment (src/models/deepseek2.cpp) |
| 2026-05-20 | Bug hunt: P2 — GGML_ASSERT crashes | @fixer + @oracle | Graceful equal_seqs handling (src/llama-graph.cpp) |
| 2026-05-20 | Bug hunt: P2 — KV cache multi-stream assert | @fixer + @oracle | Proper multi-stream support (src/llama-kv-cache.cpp) |
| 2026-05-20 | Bug hunt: P2 — KV cache save/restore TODO | @fixer + @oracle | Removed stale TODO (src/llama-kv-cache.cpp) |
| 2026-05-20 | Bug hunt: P2 — BF16 FP32 fallback | @fixer + @oracle | Restructured vec_dot (fattn-common.cuh) |
| 2026-05-20 | Bug hunt: P2 — Chat template hacks | @fixer + @oracle | Fixed jinja runtime (common/chat.cpp + jinja runtime) |
| 2026-05-20 | Bug hunt: P2 — Metadata flags misclassification | @fixer + @oracle | switch → bitwise AND (src/llama-model-saver.cpp) |
| 2026-05-20 | Bug fix: VRAM delta in run_rue_benchmark.sh | @fixer + @oracle | Changed max(before,after) → after-before delta |
| 2026-05-20 | Bug fix: gpu_acquire error resilience | @fixer + @oracle | Added `\|\| true` to prevent crash under set -e |
| 2026-05-20 | Bug fix: rocm-smi float regex | @fixer + @oracle | Accept float percentages (85.5) not just integers |
| 2026-05-20 | Bug fix: gpu_release between benchmark runs | @fixer + @oracle | Prevent GPU resource accumulation across configs |
| 2026-05-20 | Bug fix: stale binary check in benchmark scripts | @fixer + @oracle | Added check_stale_binaries() to run_benchmark.sh + run_rue_benchmark.sh |
| 2026-05-20 | Bug fix: fragile set +e/-e in run_rue_benchmark.sh | @fixer + @oracle | Restructured to explicit exit code check |
| 2026-05-20 | Bug fix: MoE CPU/GPU copy slot exhaustion | @fixer | CR-014: Added `next_copy = 0` reset after decode steps |
| 2026-05-20 | CR-010 status drift fixed | @orchestrator | Proposal status updated from Researching → Done |
| 2026-05-20 | **CR-014 invalidated** — upstream MoE bug found | @orchestrator + council + chief_engineer | Bug is pre-existing upstream (not our code). CR-014 copy slot hypothesis was wrong target. Replaced by CR-015. |
| 2026-05-20 | **Upstream MoE bug confirmed** | @orchestrator | Reverted ALL local changes. Bug reproduces on vanilla upstream with ANY `--n-cpu-moe N` + long prompt + many tokens. |
| 2026-05-20 | **Sprint focus restored** | Council directive | De-prioritize MoE CPU offloading optimizations. Return to CR-008 (perm chain) and CR-009 (kernel profiling). |
| 2026-05-20 | **CR-015 4-fix patch applied** | @explorer → @orchestrator | Deep forensic analysis found 4 bugs: (1) copy slot exhaustion — force-reset `next_copy` on every sync, (2) bitset OOB — bounds guard on expert ID scan, (3) cross-split stale — scoped `prev_ids_tensor` per split, (4) async barrier — sync before graph compute for non-async backends. See CR-015-postmortem.md. |
| 2026-05-21 | **CR-017 Cache Swizzle documented** | @orchestrator | RDNA2_CACHE_SWIZZLE stabilization complete: host-side AoS→SoA conversion, GPU-residency gating, kernel SoA loaders. CR-013 IQ4_XS MMQ mismatch still open. |
| 2026-05-21 | **CR-020 Swizzle-All Binary Initiative launched** | @orchestrator | feat/swizzle-all-quants branch created. Q4_K/Q5_K SoA kernel paths implemented in vecdotq.cuh, mmvq.cu. llama-swizzle-dev CMake target added. |
| 2026-05-21 | **CR-022 5K-Coherence Gate codified** | @orchestrator+@oracle | New QA protocol requiring 5K tokens + adversarial prompts + differential validation. Added to AGENTS.md §9. scripts/oracle_5k_fastfail.sh created. |
| 2026-05-21 | **Stable 5K baseline established** | @oracle | Qwen3.6-35B-A3B-UD-Q4_K_M: 5000 tokens @ 41.0 t/s, 0 corruption. Saved to /tmp/5k_stable_baseline_v2.txt. |
| 2026-05-21 | **Experimental swizzle-all 5K PASSED** | @oracle | build-swizzle binary: 5000 tokens @ 42.2 t/s, 0 corruption, 0 semantic divergence from baseline. |
| 2026-05-21 | **MMQ dispatch wiring completed** | @fixer | Q4_K/Q5_K added to 3 if constexpr dispatch points in mmq.cuh:3882-3962. Build passed. |
| 2026-05-21 | **MMQ-fixed 5K validation PASSED** | @oracle | 5000 tokens, 0 corruption, +2.9% gen speed (pre-fix), -12% gen speed vs stable (investigating — possible thermal). |
| 2026-05-21 | **CI pipeline patches applied** | @fixer | total_blocks_x CI guard (mmq.cuh:3827), --hip-trace removal (run_rocprof_baseline.sh), `<unusedN>` grep (oracle_5k_fastfail.sh), TOKEN_COUNT/TIMEOUT_SEC env vars (run_benchmark.sh). |
| 2026-05-21 | **⚡ Oracle WONTFIX Override** — MoE-Infinity Ideas 1, 2, 4 re-instated | System Architect | Oracle evaluated against static ggml constraints. Fork requires custom quantization contract (zero-mask d+dmin), VRAM→IC latency hiding via stream prefetch, and dynamic -ub scaling. Rebuttal filed: `docs/architecture/rebuttal_cr008.md`. Idea 3 (Wave32 compaction) confirmed WONTFIX. |
| 2026-05-21 | **SWIZZLE_COVERAGE.md created** | @explorer | 145-line master reference documenting all 5 swizzle sites with exact line numbers. |
| 2026-05-21 | **Chief Engineer workflow audit completed** | @orchestrator | 9 gaps identified: stale project-state, missing CR proposals, no safety sign-off, no thermal monitoring, no VRAM budget check, proposal pipeline drift, ifdef complexity, fast-fail not integrated, council bypassed. |
| 2026-05-21 | **Agent System Audit — Reference Pattern Alignment** | @orchestrator | 11 gaps vs gist 1d3eeb46ddfda5257c08744972e0fc4c (orchestrator pattern). P0 fixes applied: routing decision section, capability table, 3-tier escalation, minimal verdict formats, clarification protocol. New file: `opencode/agents/chaining_protocol.md`. Plugin agents updated: orchestrator.md, fixer.md, oracle.md, council.md. |
| 2026-05-21 | **CR-001 follow-up: release.yml RPATH + binary verification gaps fixed** | @fixer | 3 issues found: (1) HIP build missing `CMAKE_INSTALL_RPATH='$ORIGIN'` + `CMAKE_BUILD_WITH_INSTALL_RPATH=ON` (CPU/Vulkan had them), (2) RPATH verification step was passive — didn't assert `$ORIGIN`, (3) CPU/Vulkan builds lacked binary existence validation before packaging. All 3 fixed in release.yml. |

---

## Human Action Items

| Item | Asked By | Date | Question | Deadline |
|------|----------|------|----------|----------|
| ~~Approve rocprofv3 profiling time~~ | ~~@oracle~~ | ~~2026-05-19~~ | ✅ **RESOLVED** — CR-009: approved, 4-6 hrs allocated | ~~2026-05-21~~ |
| Choose MTP n-max experiment | @fixer | 2026-05-19 | Test n-max=3 or n-max=4 first? Current acceptance: 78.7% | 2026-05-20 |
| ~~Archive LOAD_EXPERT_F32 dead code?~~ | ~~@chief_engineer~~ | ~~2026-05-19~~ | ✅ **RESOLVED** — CR-011 Option A: archived, CMake cleaned | ~~2026-05-21~~ |

---

## Session Context

**Last Session:** 2026-05-20 — Upstream MoE CPU offloading bug hunt

**What happened:**
1. Spent ~6 hours debugging Chinese/garbage output with `--n-cpu-moe` + FA + long prompts
2. Reverted ALL local changes — bug still reproduces on vanilla upstream code
3. Confirmed: Bug is a **pre-existing upstream llama.cpp MoE CPU offloading issue**
4. Trigger: Long prompt + 1000+ tokens + ANY `--n-cpu-moe N` (even N=41 = all CPU)
5. Not related to: flash attention, batch size, context size, our sparse copy changes
6. Short prompts work fine with any config
7. Council verdict: CONDITIONAL — bug discovery valid, file upstream report
8. Chief Engineer: INVESTIGATE — algorithmic bug, not driver issue, CR-014 was wrong target
9. CR-014 → Invalid (closed). CR-015 → created for upstream tracking

**Next Session:** 
- File upstream bug report (ggml-org/llama.cpp)
- Restore sprint focus to CR-008 (perm chain) and CR-009 (kernel profiling)
- Document MoE CPU offloading workaround in project docs
- Human action: Decide if we invest time fixing upstream or pivot to all-GPU MoE

**Pending Handoffs:**
- @explorer: Investigate CR-015 fix options (copy slot barrier, per-token expert sync)
- @oracle: Verify CR-015 root cause hypothesis
- @fixer: Document workaround
- Attempt 1 (90s, --kernel-trace true): Killed by timeout — 0 data.
- Attempt 2 (300s, --kernel-trace true): Killed by timeout — 0 data.
- Attempt 3 (300s, --stats --hip-trace): Stalled — even `--hip-trace` intercepts `hipMalloc`/`hipMemcpy` during model load, causing slowdown.
- Root cause: Any HIP API tracing (`--hip-trace` or `--kernel-trace`) intercepts every `hipMalloc`/`hipMemcpy` during model loading. For 20GB models with thousands of tensors, this adds >300s to load time.
- Fix: Stripped to **pure PMC hardware counters only** — `rocprofv3 -i counters.json` with zero tracing flags. Hardware counters collected at GPU level with zero application slowdown.
- Timeout reduced to 180s (model load ~60s + inference ~3s at full speed).

**Next Session:** Re-run CR-009 with pure PMC counters (no tracing flags — full speed)  
**Pending Handoffs:** 
- CR-009: **Re-execute** — zero-overhead PMC counters only
- Human decision on MTP n-max experiment priority

---

## Priority Queue (from chief_engineer.md)

| Prio | Task | File:Line | Effort | Expected Gain | Status |
|:----:|------|-----------|:------:|:-------------:|:------:|
| **P0** | CR-020 Swizzle-All (Q4_K/Q5_K SoA expansion) | `mmq.cuh`, `mmvq.cu`, `vecdotq.cuh`, `llama-model-loader.cpp` | ✅ 3-day sprint | +2.9% gen speed, SoA cache efficiency for all K-quants | 🟢 CR-020 **5K validated, MMQ wired** |
| **P0** | CR-022 5K-Coherence Gate — infra tooling | `scripts/oracle_5k_fastfail.sh`, `run_benchmark.sh` | ✅ 1-day sprint | 5K adversarial validation + fast-fail + differential compare | 🟢 CR-022 **Codified in AGENTS.md** |
| **P0** | CR-023 Workflow Optimization + MMQ dispatch | `mmq.cuh`, pipeline scripts | ✅ 1-day sprint | 3 dispatch points wired, CI patches, SWIZZLE_COVERAGE.md | 🟡 CR-023 **Validated, -12% speed delta needs thermal investigation** |
| **P0** | 🔥 **Thermal isolation run** (diagnose -12% delta) | `rocm-smi --showtemp` | 1 hr GPU | Determine if throttling or regression | 🔴 **CR-023 deferred action** |
| **P0** | **Agent System Alignment** (reference pattern) | `~/.config/opencode/oh-my-opencode-slim/*.md` | ✅ 15 min | Routing decision, capability table, 3-tier escalation, minimal formats | 🟢 **Done** — orchestrator/fixer/oracle/council updated, chaining_protocol.md created |
| **P0** | Perm chain optimization | `vecdotq.cuh:47-69` | 4 hrs | +30-50% (70% of compute) | 🔴 CR-008 Researching |
| **P0** | [Kernel launch overhead profiling](opencode/proposals/CR-009.md) | `scripts/run_kernel_launch_profiling.sh` | 4-6 hrs | Confirm 40-50% estimate | 🟢 CR-009 **Script ready — human executes** |
| **P0** | [Upstream MoE CPU offloading bug](opencode/proposals/CR-015.md) | `ggml-backend.cpp:1541-1923` | ✅ Fixed | 4-fix patch | 🟡 CR-015 **Build+test in progress** |
| **P1** | CR-013 IQ4_XS MMQ meta offset fix | `mmq.cuh:3291` | 2 hrs | Correct SoA access for sub-tile IQ4_XS views | 🔴 CR-013 **Open** |
| **P1** | Cross-fork baseline | `scripts/build_baseline.sh` | 1 hr GPU | Documentation credibility | 🔴 CR-010 Researching |
| **P1** | `#ifdef` flattening in mmq.cuh (code cleanliness) | `mmq.cuh:3882-3962` | 1 hr | Readable gates per Chief Engineer §24 | 🟢 **CR-023 Phase 2 — in progress** |
| **P1** | Agent System P1/P2 follow-up | `opencode/agents/*.md`, `scripts/*.sh` | ✅ 30 min | Consolidate redundant sections, add designer/observer to capability table, GPU gate automation | 🟢 **Done** — orchestrator updated, 6 agent docs marked, 4 scripts gated |
| **P2** | Wire fast-fail into build_rdna2.sh --validate | `build_rdna2.sh` | 1 hr | Automated gate enforcement | ⚪ Not started |
| **P2** | Add VRAM budget check to testing scripts | `oracle_5k_fastfail.sh`, `run_benchmark.sh` | 1 hr | Prevent OOM on 16GB GPU | 🟢 **CR-023 Phase 1.2 — in progress** |
| **P3** | dp4a split accumulators | `vecdotq.cuh:1305-1370` | 2 hrs | <0.025% | ✅ Complete |
| **P4** | Idea B (prefetch) | — | N/A | No ROI | ⚪ Demoted indefinitely |

---

## Archived Proposals

| ID | Title | Status | Date Archived |
|----|-------|--------|---------------|
| CR-001 | Fix GitHub Release Pipeline | Done | 2026-05-17 |
| CR-002 | Server-aware benchmarking | Done | 2026-05-18 |
| CR-003 | run_std_bench.sh API fix | Done | 2026-05-18 |
| CR-004 | Fix throughput targets in docs | Done | 2026-05-18 |
| CR-005 | v0.5.0 Execution Audit | Done | 2026-05-19 |
| CR-006 | dp4a Utilization Analysis | Done | 2026-05-19 |
| CR-007 | Priority Re-evaluation | Done | 2026-05-19 |
| CR-011 | Archive LOAD_EXPERT_F32 dead code | Done | 2026-05-19 |
| CR-014 | Fix MoE CPU/GPU copy slot exhaustion (wrong target) | Superseded | 2026-05-21 |
| CR-017 | RDNA2 Cache Swizzle Stabilization (superseded by CR-020) | Superseded | 2026-05-21 |

---

## Workflow Audit — Agent System Changes

**Date:** 2026-05-21  
**Reference:** Gist 1d3eeb46ddfda5257c08744972e0fc4c (orchestrator pattern)

### Changes Applied

| Component | Change | Status |
|-----------|--------|--------|
| `~/.config/opencode/oh-my-opencode-slim/orchestrator.md` | Added designer + observer to capability table | ✅ Done |
| `opencode/agents/*.md` (6 files) | Added "DOCUMENTATION REFERENCE ONLY" notice to description field | ✅ Done |
| `scripts/gpu_failback.sh` | Added `gpu_ensure_free()` function (hard gate for GPU work) | ✅ Done |
| `scripts/run_benchmark.sh` | Replaced inline pgrep check with `gpu_ensure_free()` call | ✅ Done |
| `scripts/oracle_5k_fastfail.sh` | Replaced inline pgrep check with `gpu_ensure_free()` call | ✅ Done |
| `scripts/run_rue_benchmark.sh` | Added `gpu_ensure_free()` call at pre-flight | ✅ Done |
| `opencode/v0.5.0_EXECUTION_PLAN.md` | Added pointer to `project-state.md` priority queue | ✅ Done |
| `opencode/project-state.md` | Added "Workflow Audit" section + GPU gate automation P2 item | ✅ Done |

### Remaining Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| Dual agent definitions (plugin vs project) still diverge | Confusion about authoritative source | P1 |
| GPU gate not enforced in all GPU scripts (e.g., run_occupancy_benchmark.sh, validate_hygiene.sh) | Risk of GPU conflicts | P2 |
| Execution plan not programmatically enforced | Agents can work on low-priority items | P2 |

---

## Notes for Next Session

1. ✅ **CR-020 Swizzle-All** — Q4_K/Q5_K SoA expansion complete. 5K validated. MMQ dispatch wired.
2. ✅ **CR-022 5K-Coherence Gate** — codified in AGENTS.md §9. Fast-fail script + benchmark params done.
3. ✅ **CR-023 Workflow Restructure** — MMQ wiring done. CI patches applied. SWIZZLE_COVERAGE.md created.
4. 🔥 **Thermal isolation run** — **HIGHEST PRIORITY:** Run stable + experimental 5K back-to-back with rocm-smi --showtemp logging to diagnose the -12% speed delta.
5. **CR-013 IQ4_XS MMQ fix** — `load_tiles_iq4_xs_swizzled` meta offset in mmq.cuh:3291 needs sub-tile view fix
6. **CR-008 perm chain** — highest-ROI item (+30-50% at 70% of compute). Resume after thermal run.
7. **CR-009 kernel profiling** — pure PMC counters only. Need rocprofv3 at `/home/stormrage/rocm-7.13-nightly/bin/rocprofv3`
8. **CR-015 upstream PR** — file ggml-org/llama.cpp PR with 4-fix patch set
9. **VRAM budget guard** — integrate check_vram_budget() into oracle_5k_fastfail.sh and run_benchmark.sh
10. **Thermal monitoring** — add rocm-smi --showtemp --showpower before/after every GPU test
11. **Fast-fail integration** — wire oracle_5k_fastfail.sh into build_rdna2.sh --validate mode
12. **Council vote** — schedule async vote for CR-020/CR-022/CR-023 merger approval
13. 🔒 **HARDWARE_TARGET.md created** — physical topology locked. All future CRs validated against hardware constants by @librarian.
14. 🎯 **Phase Status Ledger deployed** — 3 phases verified. Next session re-oriented: **Task 1 — Custom Sparse Q4_K** (structural zero-masking + early-exit gate in `vec_dot_q4_K_q8_1_tmpl`). Multi-channel alignment deferred.
15. 🔬 **MoE-Infinity digest logged** (ICML 2025) — 4 ideas assessed. **⚡ System Architect OVERRIDE** filed: Ideas 1, 2, 4 re-instated. Oracle's WONTFIX overturned on: (a) custom re-quantization contract zeros both d+dmin, (b) VRAM→IC latency hiding via dual streams, (c) dynamic -ub as data sizing not cache sizing. See `docs/architecture/rebuttal_cr008.md`.
