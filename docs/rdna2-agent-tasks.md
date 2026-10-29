# RDNA 2 Agent Task Checklist

**Purpose**: actionable, one-task-per-row checklist for agents optimising RDNA 2 (gfx1030) GPU
performance in llama.cpp.

**Supersedes**: the 2026-10-28 draft of this file and of `docs/rdna2-optimization-guide.md`.
Corrected here: the `≤ 64 VGPRs/wave for 4 waves/CU` occupancy target (wrong), `§12.19.1` cited as
the DPP description (it is a prohibition list), `rocprofv3 --stats` prescribed for resource usage
(it does not provide it), and the 3-run / 1 % acceptance rule (this host cannot resolve at that
level).

**Companions** (read first):
`docs/gfx1030-directive.md` (priorities, binding measurement protocol, closed lanes) ·
`docs/gfx1030-isa-report.md` (the occupancy measurement and the ISA quotes behind it) ·
`docs/gfx1030-test-plan.md` (queued T0–T7 experiments with pass/fail predicates) ·
`docs/gfx1030-benchmarking.md` (host rules of record, paired statistics) ·
`docs/rdna2-optimization-guide.md` (full derivation of every claim below).

**Provenance tags** — every row carries one: `[ISA §x p.y]` · `[measured: path]` ·
`[supplementary]` · `[hypothesis]`. Untagged text is a defect, not permission.

**Target**: AMD RX 6900 XT (gfx1030), llama.cpp HIP/ROCm backend.

**Findings inventory**: **16** code-level findings (L-01…L-14, L-16, L-17) plus one documentation
row (L-15, the switch interaction matrix). Earlier drafts counted 15 because the IQ3_S *decode*
sign path (now L-11) was folded into L-07 by mistake — it is a different function in a different
file.

---

## Scope and baseline

| Item | Detail | Provenance |
|------|--------|------------|
| Primary source | AMD "RDNA 2" ISA Reference Guide (30-Nov-2020, 283 pp.) | `[supplementary]` |
| Target hardware | RX 6900 XT (gfx1030), wave32 native | `[supplementary]` |
| Application | llama.cpp compute kernels: `mmvq.cu`, `mmq*.cu`, `fattn*.cu`, `vecdotq.cuh` | `[measured]` |
| Dense baseline | **30.89 t/s** fork vs **30.05 t/s** upstream | `[measured: AGENTS.md]` |
| MoE baseline | **95.19 t/s** fork vs **94.11 t/s** upstream | `[measured: AGENTS.md]` |
| MTP baseline | **47.0 t/s** | `[measured: AGENTS.md]` |

Do **not** mix these with the `31.1 / 598.9 / 46.40` figures in `gfx1030-directive.md`: those are a
different build, model file and graphs-on production shape. Earlier drafts also quoted
`31.83 → 33.07 t/s` for a hypothetical KLONG fix; that pair is `[unverified]` and is not a baseline.

**Current state — verify, do not re-derive** `[measured: ggml/src/ggml-cuda/mmvq.cu]`:

| Switch | Value | Anchor | Meaning |
|--------|-------|--------|---------|
| `GGML_RDNA2_MMVQ_LBMIN` | 1 | `mmvq.cu:14-16` | `__launch_bounds__` **minBlocksPerMultiprocessor** (not a block size) |
| `GGML_RDNA2_MMVQ_SMALLK` | 1 | `mmvq.cu:19` | borrow the GENERIC block width for `ncols_dst == 1` |
| `GGML_RDNA2_MMVQ_KLONG` | **0** | `mmvq.cu:23` | 2-warp K reduction **disabled** (Q6_K correctness) |
| `GGML_RDNA2_MMVQ_WIDEN` | 1 (worktree) | `mmvq.cu:27` | 8 warps for single-token non-routed matvec |
| `GGML_RDNA2_MMVQ_2ROWS` | **0** | `mmvq.cu:31` | two weight rows sharing one q8_1 load — off |

MMVQ has **no RDNA2 parameter table**: `calc_nwarps` returns 1 unless one of the switch arms fires
(`mmvq.cu:619-720`). MMQ *does* have an RDNA2 config table (`mmq-config-rdna2.cuh`) and a routed
expert selection (`mmq.cu:133-170`) that is on by default.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| GPU | AMD RX 6900 XT (gfx1030) |
| Driver | ROCm 6.x+ |
| Build | `cmake --build build --config Release -j12` (never `-j$(nproc)`) |
| Arch flag | `GPU_TARGETS=gfx1030` through CMake, or `--offload-arch=gfx1030` to hipcc. `-mcpu=gfx1030` is the LLVM/assembler form, not the build recipe `[supplementary]` |
| Exclusivity | `scripts/bench-exclusive.sh` — mandatory for any measured claim |
| Instruments | `scripts/gfx1030.py` (`probe`, `bench --ab`, `profile`, `trace`, `isa`, `audit`, `census`, `rotcheck`, `predict`, `score`, `report`) |
| Correctness | `./build/bin/test-backend-ops -b ROCm` |
| Resource usage | `--save-temps` + `.s` metadata, or `llvm-readelf -n` on the `.hsaco` — **not** `rocprofv3 --stats` |
| GPU access | Only while the owner's server is idle. Never launch/stop/query `llama-server` |
| Metric | **ms/token, paired A/B, 95% CI** (§ Phase 4) |
| Knowledge | ISA opcodes, EXEC mask, SGPR/VGPR allocation, LDS semantics, and the split between compiler-emitted and hand-authorable mechanism |

---

## Phase 0: Baseline Measurement

| Step | Command | Record |
|------|---------|--------|
| 0.1 | `scripts/bench-exclusive.sh ./build/bin/llama-bench -m <dense> -ngl 99 -fa on` | t/s (pp512, tg128) |
| 0.2 | `scripts/bench-exclusive.sh ./build/bin/llama-bench -m <moe> -ngl 99 -fa on` | t/s (pp512, tg16) |
| 0.3 | `./build/bin/test-backend-ops -b ROCm` | pass/fail — this, not `ctest -L main`, covers backend ops |
| 0.4 | `./build/bin/llama-perplexity` on wikitext-2 | PPL (quality anchor, never a perf metric) |
| 0.5 | `scripts/gfx1030.py bench --ab <base.so> <base-copy.so>` | The instrument's own noise floor for today's session; record it before judging anything |

**Acceptance**: all tests pass; paired baseline recorded with pair count and CI half-width.

---

## Phase 1: Audit Tasks (CPU-only unless stated)

### A1: Wave Size Audit
- **Description**: every RDNA2 compute kernel runs wave32.
- **Source**: `[ISA §2.1 p.7]` — hardware is primarily wave32; a wave64 VALU instruction issues as
  two wave32 issues and the empty half is *skipped*.
- **Files**: `ggml/src/ggml-cuda/*.cu`, `*.cuh`
- **Check**: `ggml_cuda_get_physical_warp_size()` returns 32; no wave64-only arithmetic.
- **Verification**: `grep -rn "WARP_SIZE" ggml/src/ggml-cuda/ | grep -v 32`
- **Agent**: @explorer

### A2: LDS Bank Conflict Audit
- **Description**: find LDS access patterns that serialise.
- **Source**: `[ISA §10.1 p.90]` — 64 banks, 512 × 4 B each, two 32-bank halves; same-bank accesses in
  one cycle serialise.
- **Files**: `fattn-gqa-gfx1030.cu`, `mmvq.cu`, `mmq.cu`, `common.cuh`
- **Check**: a wave32 reading 32 consecutive dwords is conflict-free by construction. Look for
  strided/transposed tile reads instead.
- **Verification**: `rocprofv3 -L` to confirm the counter exists on this part, then
  `rocprofv3 --pmc SQ_LDS_BANK_CONFLICT`. If the counter is absent, the lane is unmeasurable —
  stop and record that.
- **Agent**: @explorer

### A3: VGPR Pressure Audit *(rewritten — the old target was wrong)*
- **Description**: identify kernels whose VGPR count limits waves/SIMD. VGPRs allocate in
  **16-dword groups for wave32**.
- **Source**: `[ISA §3.6.4 p.18]`; register pool **65,536 per SIMD**, **16 wave slots per SIMD**,
  **256 VGPRs/wave max** `[measured: docs/gfx1030-isa-report.md:22]` + `[supplementary: KFD topology]`.
- **The old rule is void**: "target ≤ 64 VGPRs per wave for 4 waves/CU" implied a 128-VGPR/SIMD
  pool that does not exist. The correct arithmetic is
  `waves/SIMD = min(16, 65536 / (VGPRs x 32))`, so **128 VGPRs/wave already buys full occupancy**,
  and 240 VGPRs (the IQ3_S `mul_mat_q` figure, rounded up from 234) buys 8 waves/SIMD = 2 blocks/CU
  on the register side — the second block is blocked by LDS, not registers.
- **Files**: `mmvq.cu`, `mmq.cu`, `fattn-gqa-gfx1030.cu`
- **Check**: read the emitted count; never assume `__launch_bounds__` was honoured. A zero-spill
  kernel cannot be improved by "reclaiming registers".
- **Verification**: `--save-temps`, then
  `grep amdhsa_next_free_vgpr <kernel>.s` (also `_sgpr`, `group_segment_fixed_size`,
  `private_segment_fixed_size`); or `llvm-readelf -n <kernel>.hsaco`; or `hipFuncGetAttributes`.
- **Note**: the RDNA2 MMQ table's `occupancy = 2` request is architecturally impossible at J=64
  (2 × 45.9 KiB > 64 kB per CU) — this lane is closed for those tiles `[measured: gfx1030-isa-report.md]`.
- **Agent**: @explorer

### A4: Global Memory Coalescing Audit
- **Description**: verify the work-item→address mapping merges into full sectors.
- **Source**: `[ISA §4.1.1 p.26-27]` — the clause/coalescing rule lives in the S_CLAUSE discussion,
  not in §2.4 or §8.1.5.
- **Files**: `mmvq.cu`, `mmq.cu`, `fattn-gqa-gfx1030.cu`, `mmq-load-tiles.cuh`
- **Check**: no scatter in hot paths; SoA over AoS. The surviving real gap is **load width**, not
  coalescing: 5,302 `global_load_dword` (4 B) vs 932 `dwordx4` (16 B) in the decode weight loads
  `[measured: gfx1030-directive.md §2]`.
- **Verification**: `scripts/gfx1030.py profile` (per-class bandwidth table), plus an instruction
  census (`gfx1030.py isa`). The `DRAM_*` and `FETCH_SIZE` counters are dead/absent on this part.
- **Agent**: @explorer

### A5: Redundant Fence / Barrier Census *(replaces the S_WAITCNT audit)*
- **Description**: the previous version of this task asked agents to "verify S_WAITCNT". That is not
  actionable: `S_WAITCNT` is emitted by the compiler and cannot be authored from HIP C++. What *is*
  actionable is finding memory ordering the source adds on top of what the hardware and the existing
  barriers already guarantee.
- **Source**: `[ISA §4.4 p.29-30]` — four counters (`VM_CNT`, `LGKM_CNT`, `EXPCNT`, `VS_CNT`); each
  instruction class completes out of order and the compiler inserts the waits. `[ISA §4.3 p.29]` for
  barriers.
- **Files**: `mmvq.cu`, `fattn*.cu`, `common.cuh`, `ggml-cuda.cpp`
- **Check**: every `__threadfence*()` / `__syncthreads()` / `atomic_thread_fence` in a per-token hot
  path, and whether a preceding barrier already orders the data.
- **Known target**: `mmvq.cu:916-937` — `__threadfence()` before `__syncthreads()` in the cross-warp
  reduction. `__syncthreads()` already orders shared memory; A/B the fence (L-13).
- **Verification**: source census (`grep -rn "__threadfence\|__syncthreads\|memory_order_" ggml/src/ggml-cuda/`),
  then paired A/B for anything removed.
- **Agent**: @explorer

### A6: Instrument Policy Audit *(new)*
- **Description**: stop using `rocprofv3 --stats`. It does not report VGPR/SGPR/LDS/scratch/occupancy
  on this stack, and the earlier drafts built three verification columns on it.
- **Mapping**:

  | Question | Instrument |
  |---|---|
  | VGPR / SGPR / LDS / scratch / occupancy | `--save-temps` + `.s` metadata, or `llvm-readelf -n <hsaco>` |
  | Bank conflicts | `rocprofv3 -L`, then `--pmc SQ_LDS_BANK_CONFLICT` (if listed) |
  | Per-class bandwidth | `scripts/gfx1030.py profile` |
  | Instruction mix | `scripts/gfx1030.py isa` |
  | Structural/source checks | `scripts/gfx1030.py audit` |
  | Graph shape | `scripts/gfx1030.py census` / `rotcheck` |
  | Backend correctness | `./build/bin/test-backend-ops -b ROCm` |
  | A/B decision | `scripts/gfx1030.py bench --ab` |
- **Check**: every row of every report names a real instrument and, where applicable, the counter
  it used.
- **Verification**: dead counters on gfx1030 are `SQ_INSTS_VALU`, `SQ_WAIT_ANY`,
  `SQC_LDS_IDX_ACTIVE`, `FETCH_SIZE`; all `DRAM_*`/`TCC_*` are absent `[measured: gfx1030-directive.md §2]`.

### A7: No-Go List *(new)*
- **Description**: mechanisms already refuted on this tree. Do not reopen without a new measurement.
- **Source**: `docs/gfx1030-directive.md §10` (closed-lane ledger) — this table is a copy, and the
  directive wins any conflict.

  | Closed lane | Verdict |
  |---|---|
  | "dot pipes idle — kernels dequantise then scalar FMA" | refuted: 3,008 `v_dot4c_i32_i8` vs 568 FMA |
  | "`ds_bpermute` unharvested" | refuted: 39,381 occurrences |
  | "`S_CLAUSE` unemitted" | refuted: 18,873 occurrences |
  | "`__ldg`/`__ldcg` wrappers give cache control" | wrappers are no-ops |
  | "prefill ceiling 957.6 t/s" | refuted: the dp4a multiply *is* the dot op; ceiling ~1,915 t/s |
  | 2 blocks/CU on `mul_mat_q` | closed by LDS (2 × 45.9 KiB > 64 kB); the VGPR half is contested (234 vs 225) but irrelevant once LDS blocks it |
  | `ds_bpermute`/`DS_SWIZZLE` "free shuffle" as a new lane | already generated by shuffle reductions |
  | SFU pipe | negligible in the profile |
  | GDS append/consume/GWS (P-01…P-03) | **hardware absent** on this part (`gds_size_in_kb = 0`) |
- **Verification**: cite the ledger entry in the report; a reopened lane needs a number that
  contradicts it.

### A8: Cross-Backend Landmine Census *(new)*
- **Description**: find code where a backend-specific symbol leaks into core, or process-global
  state stands in for per-device state. These break non-HIP builds or silently cross-contaminate.
- **Known sites**: `src/llama-context.cpp:4` + `:1533-1545` (`extern std::atomic<bool>
  ggml_cuda_klong_disabled;` declared in core, defined in `mmvq.cu:10`);
  `common.cuh:1705-1712` (`alloc_failed` is a function-local `static` → process-global);
  `fattn.cu:701-703`, `:721-723` (non-atomic `static bool logged` latches).
- **Check**: `grep -rn "ggml_cuda_\|GGML_USE_HIP\|GGML_USE_CUDA" src/` — every hit is either guarded
  or a defect.
- **Verification**: a build with `-DGGML_CUDA=OFF` must link.
- **Agent**: @reviewer

---

## Phase 2: Implementation Tasks

`Applies?` filters for a HIP compute kernel. `N/A` means the mechanism is the compiler's business or
belongs to a pipeline llama.cpp does not have — do not spend GPU time there.

### S-Series: Compute / Shader Core

| ID | Description | Applies? | Source | Expected Impact | Verification |
|----|-------------|----------|--------|-----------------|--------------|
| S-01 | Prefer Wave32 | yes | §2.1 (p.7) | Scheduling granularity, **not** instruction count | `WARP_SIZE` census |
| S-02 | Subvector loop | **N/A** — not expressible from HIP | §4.2.1 | — | — |
| S-03 | Packed math (VOP3P) for dot products | yes | §6.7, §12.10 (p.162-164) | 2× f16/int throughput where applicable | Instruction count |
| S-04 | SDWA for 16/8-bit ops | yes (usually already emitted) | §6.8 (p.49) | Fewer moves | `.s` inspection |
| S-05 | **DPP** as a VALU source-0 modifier (VOP1/VOP2 **only**) | yes | §6.9 (p.49) | Cheaper intra-wave exchange | `.s` shows `dpp:` operand |
| S-06 | DPP8/DPP16 formats | yes | §13.3.9/§13.3.10 | Flexible swizzle | Row mask + `bound_ctrl` + `fi` present |
| S-07 | `DS_SWIZZLE_B32` for LDS-side permutation | yes | §12.13.1 (p.207) | Avoids a read/write round trip | Before/after LDS count |
| S-08 | VOP3A over VOP1/VOP2 | **N/A** — compiler chooses the encoding | §6.1 | — | — |

> **Two corrections agents must not forget.** (1) DPP is *not* `ds_permute_b32`/`ds_bpermute_b32` —
> those are DS instructions. (2) §12.19.1 (p.234) is a **prohibition list** for DPP (it says what DPP
> may not combine with), not a description of it. The mechanism is §6.9 p.49; the encodings are
> §13.3.9/§13.3.10.

### M-Series: Memory Access

| ID | Description | Applies? | Source | Expected Impact | Verification |
|----|-------------|----------|--------|-----------------|--------------|
| M-01 | LDS allocation alignment | partial | §3.6.6 (p.18) | Avoids misaligned group segments | `amdhsa_group_segment_fixed_size` |
| M-02 | SGPR load alignment | yes | §3.6.3 (p.18) | Avoids split scalar loads | `s_load_dwordx*` operand alignment |
| M-03 | GLC=0 for read-only buffers | yes | §8.1.10 (p.66) | L0 hits persist | `.s` `glc:0` |
| M-04 | GLC=1 only where required | yes | §8.1.10 (p.66) | Correctness for atomics-with-return | Atomic tests |
| M-05 | DLC=1 only for device coherency | yes | §8.1.10 (p.66) | L1 bypass | `.s` `dlc:1` |
| M-06 | SLC=1 for stream policy | yes | §8.1.10 (p.66) | Protects LRU lines from the weight stream | Source confirmed; SASS **unverified on this box** |
| M-07 | Buffer swizzled addressing | **N/A** | §8.1.5 (p.59) | — | — |
| M-08 | Add-TID addressing | **N/A** — pre-RDNA, not emitted | §8.1.2 | — | — |
| M-09 | MUBUF `LDS=1` | **N/A** — not reachable from HIP C++ | §8.1.9 (p.65) | — | — |
| M-10 | GLOBAL/SCRATCH over FLAT | partial | §9.2.2 (p.86), Table 51 (p.83) | Avoids the aperture check | `SEG` field |
| M-11 | Manual buffer stride+offset | **N/A** | §8.1.5 (p.59) | — | — |

Cache-bit effects, precisely `[ISA §8.1.10 p.66]`: the bits affect **loads**; stores always bypass L1
and their L0 is always miss-evict. `GLC=0` allows an L0 hit, `GLC=1` bypasses L0. `DLC=1` is an L1
miss-evict. `SLC=1` is L2 stream.

### C-Series: Compiler / Code Generation

| ID | Description | Applies? | Source | Expected Impact | Verification |
|----|-------------|----------|--------|-----------------|--------------|
| C-01 | `S_VERSION` | **N/A** — code-object metadata | §2.5 (p.11), §12.2.1 | — | — |
| C-02 | `S_CODE_END` padding | **N/A** — assembler emits 5× padding; §2.5 p.11 is complete (64 dwords / 256 B, prefetch 1–3 cachelines) | §2.5 (p.11) | — | — |
| C-03 | `S_CLAUSE` grouping | **N/A** — already emitted 18,873× | §4.1.1 (p.26-27) | — | — |
| C-04 | `S_WAITCNT` correctness | yes (compiler-emitted; audit only) | §4.4 (p.29-30) | Prevents out-of-order hazards | Counter values in `.s`; see A5 for the actionable form |
| C-05 | `S_MOVRELS/D_B32` | **N/A** | §5.8.1 | — | — |
| C-06 | `V_MOVRELSD_B32` | **N/A** | §6.6 | — | — |
| C-07 | `V_CVT_PK_*` conversions | partial | §12.12 | Fewer conversion instructions | Instruction count |

### L-Series: llama.cpp-specific (gfx1030)

**Every row has a file:line anchor.** Navigate by symbol in practice; the anchors are here so the
work is not re-derived.

| ID | Description | State | Anchor | Source | Provenance | Verification |
|----|-------------|-------|--------|--------|------------|--------------|
| L-01 | **KLONG** 2-warp K-loop reduction for `ncols_dst == 1` | disabled (`KLONG 0`) — Q6_K breaks; MTP guard also disables at run time | `mmvq.cu:23`, `:619-641` (arm), `:634` (Q6_K), `:1331-1335` (MTP guard) | §4.4, §10.1 | `[measured: mmvq.cu]`; gain `[unverified]` | Fix Q6_K → "Paris" at temp=0, then ≥40 pairs |
| L-02 | **WIDEN** 8 warps for single-token non-routed matvec | enabled (worktree) | `mmvq.cu:27`, `:643-666` | §3.6.4, §10.1 | `[measured: mmvq.cu]` | A/B two `.so`s |
| L-03 | **SMALLK** borrow GENERIC block width | enabled | `mmvq.cu:19`, `:668-677`, `:1277-1284` | §3.6.4 | `[measured: mmvq.cu]` | A/B; it suppresses L-01/L-02 (§ matrix) |
| L-04 | **2ROWS** share q8_1 loads across two weight rows | disabled | `mmvq.cu:31`, `:696-712`, `:742-746` | §10.1 | `[measured: mmvq.cu]`; traffic claim `[hypothesis]` | A/B |
| L-05 | **LBMIN** — `__launch_bounds__` **minBlocksPerMultiprocessor**, *not* a block size | enabled | `mmvq.cu:14-16`, `:752` | §3.6.4 | `[measured: mmvq.cu]` | Read the emitted VGPR count; raise only if registers bind |
| L-06 | **FA quants**: `q5_0-q5_0` (committed) and `q4_0-q5_0` (worktree) | split | `ggml/CMakeLists.txt:208` | §8.1.10 | `[measured: ggml/CMakeLists.txt]`; "+2.15% pp512, 597.39→610.25" **`[supplementary]`** / `[unverified]` | A/B each pair against the model's real `-ctk/-ctv` |
| L-07 | IQ3_S sign expansion in the **MMQ tile loader** (nibble→byte-MSB multiply + `v_perm_b32` + fused xor-add) | landed | `mmq-load-tiles.cuh:1423-1442` (+19/−4) — **misplaced in earlier drafts as a `vec_dot` change** | §12.10, §12.12 | `[measured: mmq-load-tiles.cuh]`; "578→258, −55.4%, 1,048,576 cases" **`[unverified]`** | `gfx1030.py isa` mix, then A/B |
| L-08 | Same transform for the other IQ types — **5 sites** (the 5th is L-11) | TODO | `mmq-load-tiles.cuh:1157-1161` (IQ2_XXS), `:1220-1224` (IQ2_XS), `:1288-1292` (IQ2_S), `:1354-1358` (IQ3_XXS), **plus** `vecdotq.cuh:1317-1321` (IQ3_S decode) | §12.10, §12.12 | `[measured: both files]`; "~15% per type" `[hypothesis]` | Per-site A/B |
| L-09 | **H14 wide loads**: `get_int_b2` does two 16-bit loads | audit only | `vecdotq.cuh:25-35`; `block_iq3_s.signs` at `ggml-common.h:414-421` (offset 74, 2-byte aligned) | §8.1.9, §8.1.6 | `[measured: source]`; "2–4× bandwidth" `[hypothesis]` | **Measurement required** — widening is a correctness change on a 2-byte-aligned pointer |
| L-10 | **MTP conditional KLONG** | implemented | `src/llama-context.cpp:1533-1545`, `common.cuh:1663-1670`, `mmvq.cu:10`, `:1334-1336` | §4.4 | `[measured: those files]` | **INERT while KLONG=0** — it flips a compiled-out switch. Depends on L-01 |
| L-11 | **IQ3_S decode `vec_dot` sign path** — same transform as L-07, decode side | TODO (highest value of the family) | `vecdotq.cuh:1296` (function), `:1317-1321` (old `__vcmpne4`/`__vsub4` idiom) | §12.10, §12.12 | `[measured: vecdotq.cuh]` | Decode A/B — this is the model-of-record's decode path |
| L-12 | Non-atomic latches in `fattn.cu` | defect | `fattn.cu:701-703`, `:721-723` | §4.3 | `[measured: fattn.cu]` | `std::atomic<bool>` + `exchange`, or `call_once`; no perf claim |
| L-13 | `__threadfence()` before the MMVQ cross-warp reduction | candidate removal | `mmvq.cu:916-937` | §4.3, §4.4 | `[measured: mmvq.cu]` | A/B; `__syncthreads()` already orders shared memory |
| L-14 | `alloc_failed` is process-global, not per-device | defect | `common.cuh:1705-1712` | — | `[measured: common.cuh]` | Make it per-device; test with two GPUs or a moved tensor |
| L-15 | **Switch interaction matrix** (SMALLK/WIDEN/KLONG mutually exclusive) | documentation | `mmvq.cu:619-720` | — | `[measured: mmvq.cu]` | Every report names the full switch vector (see below) |
| L-16 | **MMQ routed selection A/B** | on by default, never paired-measured | `mmq.cu:133-139` (env), `:141-152` (types), `:156-158` (J), `:162` (predicate), `:283-292` (stream-K rejection) | §8.1 | `[measured: mmq.cu]` | `GGML_HIP_MMQ_ROUTED_GFX1030=0` vs `=1`, ≥40 pairs, MoE |
| L-17 | **FA quant pairs are per-consumer** (`q8_0-q5_0` ≠ `q4_0-q5_0`) | clarification | `ggml/CMakeLists.txt:208` | §8.1.10 | `[measured: CMakeLists]` | A/B each pair against the cache types actually used |

**Switch interaction matrix** (from `mmvq.cu:619-720`; all guards require `table_id == MMVQ_PARAMETERS_RDNA2`
and `ncols_dst == 1`):

| Switch | Guard | Excludes |
|--------|-------|----------|
| `KLONG` | `!small_k && halve_iters` | `SMALLK`, `WIDEN` |
| `WIDEN` | `!small_k && !halve_iters` | `SMALLK`, `KLONG` |
| `SMALLK` | `small_k` (evaluated first, suppresses the others) | `KLONG`, `WIDEN` |
| `2ROWS` | separate arm (`:696-712`, `:742-746`) | orthogonal |
| `LBMIN` | `__launch_bounds__` second arg (`:752`) | orthogonal |

### H-Series: Special Instructions

| ID | Description | Applies? | Source | Note |
|----|-------------|----------|--------|------|
| H-01 | Dot products + accumulate forms | yes | §12.10 (p.162-164) | Six forms on p.2-3: `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4` and their `*C*` accumulate variants; that page pair also lists the FMA instructions removed in RDNA. Already harvested (3,008 `v_dot4c_i32_i8`) |
| H-02 | `V_MBCNT_LO/HI_U32_B32` | partial | §12.12 | Rarely the bottleneck |
| H-03 | `V_READLANE_B32` / `V_WRITELANE_B32` | yes | §12.12 | VGPR↔SGPR bridge |
| H-04 | `V_PERMLANE16_B32` / `V_PERMLANEX16_B32` | yes | §12.12 (p.186) | Half-wave permutation, no LDS |
| H-05 | `V_PERM_B32` | yes | §12.12 (p.180) | **VOP3A only**, opcode 836 — not a VOP2/VOP3 op |
| H-06 | `V_CUBE*` | **N/A** | §12.12 | Graphics |
| H-07 | `V_DIV_SCALE/FMAS_F32` | partial | §12.12 | Softmax epilogue only |
| H-08 | `IMAGE_BVH_INTERSECT_RAY` | **N/A** | §8.2.10, §12.16 | Ray tracing |
| H-09 | `S_GETREG_B32` / `S_SETREG_B32` | partial | §5.8 (p.37) | `SHADER_CYCLES` and `IB_STS` are both §5.8 p.37; `S_MEMTIME`/`S_MEMREALTIME` are p.3 / §7.2.3 p.53 |

### P-Series: Synchronization

| ID | Description | Applies? | Source |
|----|-------------|----------|--------|
| P-01 | `DS_APPEND` / `DS_CONSUME` | **N/A** — GDS ops; this part has `gds_size_in_kb = 0` | §10.4.3 |
| P-02 | `DS_ORDERED_COUNT` | **N/A** — GDS absent | §12.13.63 |
| P-03 | `DS_GWS_BARRIER` / `DS_GWS_SEMA_*` | **N/A** — GWS lives in GDS; absent | §12.13.24-29 |
| P-04 | `S_BARRIER` within a workgroup | yes | §4.3 (p.29) |
| P-05 | `S_CBRANCH_EXECNZ` | **N/A** — compiler-emitted | §12.5.9 |

---

## Phase 3: Step-by-Step Workflow

### Phase A: Baseline
1. Build: `cmake --build build --config Release -j12`
2. Correctness: `./build/bin/test-backend-ops -b ROCm`
3. Benchmark through `scripts/bench-exclusive.sh`; record paired ms/token, pair count, CI

### Phase B: Single Change
4. Pick ONE task (order: L → M → S → C-04/H; never R or P-01…P-03)
5. Implement the smallest possible diff; change exactly one switch (§ matrix)
6. Build; `test-backend-ops -b ROCm`; output check "Paris" not "333333333" at temp=0
7. Resource check from the artifact metadata (`--save-temps` → `amdhsa_next_free_vgpr`), not a profiler

### Phase C: Measure
8. `scripts/gfx1030.py bench --ab <base.so> <fix.so> --pairs N` (interleaved)
9. Accept only if the 95% CI excludes zero; otherwise `INVESTIGATE`, never `KEEP`
10. Correctness failure → revert immediately, no measurement

### Phase D: Iterate
11. One change per diff; report the full switch vector every time

---

## Phase 4: Measurement and Validation

### Metrics and instruments

| Metric | Instrument | Command |
|--------|------------|---------|
| Decode / prefill rate | `gfx1030.py bench --ab` | paired artifact swap, interleaved |
| Per-class bandwidth | `gfx1030.py profile` | graph-free kernel trace → scored class table |
| VGPR / SGPR / LDS / scratch | build metadata | `--save-temps` then `grep amdhsa_* <kernel>.s`; or `llvm-readelf -n <hsaco>` |
| Occupancy | arithmetic | `min(16, 65536/(VGPRs*32))` waves/SIMD; LDS from the launch parameter |
| Bank conflicts | `rocprofv3` | `-L` first, then `--pmc SQ_LDS_BANK_CONFLICT` |
| Instruction mix | `gfx1030.py isa` | static mix + 5 % ALU gate |
| MTP acceptance | `llama-cli` | `grep "accepted"` |
| PPL | `llama-perplexity` | quality anchor only |

### Comparison rules (supersedes "3 runs / ≥1 %")

- **Paired A/B**, interleaved: `scripts/gfx1030.py bench --ab`, `--ab-stats` to recompute offline
- Report **ms/token**, paired mean delta, sd of per-pair differences, **95 % CI half-width**
- Required pairs: `n = (1.96 * sd_pair / delta)^2`; at `sd_pair = 0.64 ms`, 40 pairs resolve ~0.20 ms,
  80 → ~0.14 ms, 250 → ~0.08 ms `[measured: docs/gfx1030-benchmarking.md]`
- Same model, prompt, seed, and every `-fa/-ctk/-ctv/-ngl/-ncmoe` flag on both arms
- **One measurement process at a time**, always via `scripts/bench-exclusive.sh`
- Preflight: `pgrep -a -f "llama-server|llama-bench"` empty; `rocm-smi --showmemuse` ≈ 0

### Verification checklist

- [ ] `./build/bin/test-backend-ops -b ROCm` 100 % pass
- [ ] Model output "Paris" at temp=0
- [ ] PPL not regressed (>0.1 % is suspicious, not automatically fatal)
- [ ] No memory violations (`TRAPSTS.MEM_VIOL`)
- [ ] No exception traps (`TRAPSTS.EXCP`)
- [ ] Resource numbers read from artifact metadata, not estimated
- [ ] Paired A/B with the required pair count; CI reported
- [ ] Exactly one switch changed; full switch vector named

---

## Phase 5: Reporting Template

| Task ID | Change | Anchor | ISA citation | Before (ms/tok) | After (ms/tok) | Δ ms/tok | Pairs | sd(pair) | 95 % CI | CI decision | Correctness | Result | Notes |
|---------|--------|--------|--------------|-----------------|----------------|----------|-------|----------|---------|-------------|-------------|--------|-------|
| L-01 | KLONG fix | `mmvq.cu:619-641` | §4.4 | — | — | — | — | — | — | — | — | — | inert until Q6_K fixed |
| L-11 | IQ3_S decode sign path | `vecdotq.cuh:1317-1321` | §12.12 | — | — | — | — | — | — | — | — | — | highest value of the family |
| L-16 | MMQ routed A/B | `mmq.cu:162` | §8.1 | — | — | — | — | — | — | — | — | — | env kill switch |
| … | … | … | … | … | … | … | … | … | … | … | … | … | … |

**Result values**: `KEEP` (CI excludes zero, correct), `REVERT` (regressive or incorrect),
`INVESTIGATE` (CI includes zero — usually needs a different instrument, not more pairs).

---

## Phase 6: Open Questions and Gaps

| # | Gap | Impact | Mitigation |
|---|-----|--------|------------|
| 1 | No tuning guidance in the ISA | Cannot derive an optimum | Measure; `gfx1030.py profile` |
| 2 | No `V_DOT*` throughput spec | Cannot predict quantized speedup | Benchmark each variant |
| 3 | No LDS-vs-DPP latency comparison | Cannot choose a priori | Benchmark, or keep the compiler's choice |
| 4 | **No address→bank mapping** in the ISA | Conflicts unpredictable statically | Structural argument only; counter may not exist on this part |
| 5 | **No VALU/LDS/VMEM latency tables** | Cannot model stalls analytically | Counters only |
| 6 | **No VGPR operand-bank conflict rules** | Some stalls inexplicable | Out of reach |
| 7 | **No CU-vs-WGP LDS mode mechanism** | Cannot decide the split from the ISA | Compiler default is WGP mode; bit in `COMPUTE_PGM_RSRC` `[supplementary]` |
| 8 | **No guidance on what HIP already emits** | "Unused instruction" claims keep being wrong | Disassemble first; three such claims already refuted (A7) |
| 9 | No cache sizes in the ISA | Cannot size working sets | `[supplementary]`: 4 MB L2, 128 MB Infinity Cache above it; `GL2C_*` sees L2 only |
| 10 | No bandwidth specification | No theoretical peak from the manual | `[measured]` 500.6 GB/s read ceiling |
| 11 | KLONG Q6_K bug | KLONG off, L-10 inert | Needs GPU debugging (rocgdb); FP associativity is the suspect `[hypothesis]` |

---

## Appendix A: Key ISA Instructions

| Instruction | Format | Description | Use |
|---|---|---|---|
| `V_FMAC_F32` | VOP2 | Fused multiply-add (replaces `V_FMA_F32`) | Dot products, reductions |
| `V_DOT4C_I32_I8` | VOP3P | 4× int8 dot, accumulate | Quantized matmul (in use) |
| `V_DOT8_I32_I4` | VOP3P | 8× int4 dot | Low-bit matmul |
| `V_DOT2_F32_F16` | VOP3P | 2× f16 dot | Half-precision matmul |
| `V_DOT4_U32_U8` | VOP3P | unsigned 4× u8 dot | Q8_0 activations |
| `DS_READ_B32` / `DS_WRITE_B32` | DS | LDS read/write | Shared memory |
| `DS_READ2_B32` / `DS_WRITE2_B32` | DS | Two consecutive dwords | LDS tiles |
| `DS_BPERMUTE_B32` | DS | Cross-lane byte permute (**not** DPP) | Shuffle reductions |
| `DS_SWIZZLE_B32` | DS | LDS-side lane permutation | Tile re-layout |
| `V_PERMLANE16_B32` / `V_PERMLANEX16_B32` | VOP1/VOP2-class | Half-wave permutation | Warp reduce, broadcast |
| `S_WAITCNT` | SOPP | Wait on `VM/LGKM/EXP/VS` | Dependency resolution |
| `V_PERM_B32` | **VOP3A** (opcode 836) | Byte permute | Sign-path work |
| `V_LSHLREV_B32` / `V_AND_B32` / `V_OR_B32` / `V_XOR_B32` | VOP2 | Bit manipulation | Sign/mask work |
| `S_LOAD_DWORD` / `X2` / `X4` / `X8` / `X16` | SMEM | Scalar load | Metadata in SGPRs |
| `BUFFER_LOAD_DWORD` / `_DWORDX4` | MUBUF | Vector load | H14 width question |
| `BUFFER_LOAD_*_LDS` | MUBUF | Load direct to LDS (`TFE=0`, M0[15:0]) | Bypass VGPRs |
| `s_getreg_b32` | SOP1 | `SHADER_CYCLES`, `IB_STS` (both §5.8 p.37) | Timing, ifetch stalls |
| `S_MEMTIME` / `S_MEMREALTIME` | SOP1 | 64-bit clock (p.3 / §7.2.3 p.53) | Bracketing |

## Appendix B: LDS Organisation

- **64 banks**, as **two 32-bank halves**, one per SIMD of the pair `[ISA §10.1 p.90]`
- 512 entries × 4 B per bank = 2048 B; total 128 kB per WGP `[ISA §10.1 p.90]`
- Max **64 kB per workgroup**, and 64 kB per CU (half the WGP) `[ISA §2.3.1]`
- LDS bandwidth ≈ **10×** global for the same access pattern `[ISA §10.1]`
- `read2`/`write2` touch two consecutive dwords; `_B128` touches four `[ISA §12.13 p.205-206]`
- **No address→bank mapping is specified in the ISA.** The earlier `(addr/4) % 64` formula was not
  from the manual. What holds: a wave32 reading 32 consecutive dwords hits 32 distinct banks in its
  own half → conflict-free by construction `[measured: gfx1030-isa-report.md S4]`
- Same-bank accesses in one cycle serialise `[ISA §10.1 p.90]`

## Appendix C: Hardware Constants

| Constant | Value | Provenance |
|----------|-------|------------|
| Wave32 / Wave64 lanes | 32 / 64 (wave64 issues as 2× wave32, empty half skipped) | `[ISA §2.1 p.7]` |
| VGPR group (wave32 / wave64) | 16 / 8 dwords | `[ISA §3.6.4 p.18]` |
| Registers per SIMD | 65,536 | `[measured: gfx1030-isa-report.md:22]` |
| Wave slots per SIMD / per CU | 16 / 32 | `[supplementary: KFD topology]` |
| Max VGPRs per wave | 256 | `[supplementary]` |
| VGPRs/wave for full occupancy | 128 | arithmetic |
| SGPRs | 106 + 2 (VCC) + 16 (trap) | `[ISA §3.6.2 p.17]` |
| LDS banks / entries / bytes per bank | 64 / 512 / 2048 | `[ISA §10.1 p.90]` |
| LDS per WGP / per CU / per workgroup | 128 kB / 64 kB / 64 kB | `[ISA §2.3.1, §10.1]` |
| SIMD32 per WGP / per CU | 4 / 2 | `[ISA §10.3 p.91, §1.1 p.6]` |
| GDS | **absent** (`gds_size_in_kb = 0`) | `[supplementary: KFD topology]` |
| GLC / DLC / SLC | see M-series note | `[ISA §8.1.10 p.66]` |
| Infinity Cache | 128 MB above a 4 MB L2 | `[supplementary]` |

---

## Constraints

- Base all recommendations strictly on the AMD RDNA 2 ISA Reference Guide; tag everything
  `[ISA]` / `[measured]` / `[supplementary]` / `[hypothesis]`.
- Never commit or push (owner lands by hand). Never write PR text, commit messages, or reviewer replies.
- Never launch/stop/query/health-check `llama-server`.
- GPU work only while the owner's server is idle; one measurement process at a time via
  `scripts/bench-exclusive.sh`.
- Report **ms/token with a 95 % CI** from paired A/B — never a single unpaired run.
- Every claim needs a citation or a repro command.
- One switch / one change at a time; name the full switch vector.
- Correctness: `./build/bin/test-backend-ops -b ROCm` 100 % pass **and** model output "Paris" at temp=0.
- Agents with no prior context must be able to execute a task unambiguously from its row alone.
- Rendering-pipeline rows (R-series) and GDS rows (P-01…P-03) are **not applicable** to llama.cpp
  compute kernels on this part.
