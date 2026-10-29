# RDNA 2 GPU Performance Optimization Playbook

> **Purpose:** Structured, step-by-step workflow for agents to identify, apply, measure, and validate
> RDNA 2 GPU performance optimizations in llama.cpp. Instruction-level claims are cited to the AMD
> *"RDNA 2" Instruction Set Architecture: Reference Guide* (30-Nov-2020, 283 pages) by section **and**
> page. The ISA specifies mechanisms; it is **not** a tuning guide, and nothing here should be read
> as a promised speedup.

> **Supersedes:** the 2026-10-28 drafts of this file and `docs/rdna2-agent-tasks.md`. Those drafts
> carried four blocking errors, all corrected below: an invented `≤ 64 VGPRs per wave for 4 waves/CU`
> occupancy target, `§12.19.1` cited as the DPP description, `rocprofv3 --stats` prescribed as a
> verification tool, and a 3-run / 1% measurement protocol.
> **Superseded by:** nothing — this is the current text.

> **Companion documents (read before starting work):**
> - `docs/gfx1030-directive.md` — campaign directive: work-package priority, binding measurement
>   protocol, closed-lane ledger, ISA mechanism inventory for this part.
> - `docs/gfx1030-isa-report.md` — the measurement that closed the 2-blocks/CU occupancy lane, with
>   the ISA quotes that do it.
> - `docs/gfx1030-test-plan.md` — queued experiments T0–T7, each with a pass/fail predicate.
> - `docs/gfx1030-benchmarking.md` — **host rules of record**: one process at a time, paired A/B
>   statistics, resolving power of this box.

> **Provenance convention.** Every factual row carries one of four tags and nothing else may be
> asserted:
> - `[ISA §x p.y]` — stated in the RDNA 2 ISA manual, section and page.
> - `[measured: path]` — measured on this host; the path is the evidence file or the source anchor.
> - `[supplementary]` — outside the ISA (vendor docs, compiler behaviour, community reports).
>   Not ISA-grounded; may not be used as the sole basis for a change.
> - `[hypothesis]` — plausible, unmeasured. Requires a falsifier before it may gate work.
>
> Anything without a tag is a defect in this document, not permission.

> **Source document:** `/home/stormrage/Downloads/rdna2-shader-instruction-set-architecture.pdf`
> (text extraction at `/tmp/opencode/rdna2-isa.txt` via `pdftotext`).
> **Target hardware:** AMD Radeon RX 6900 XT (gfx1030), RDNA 2, wave32 native.
> **Application context:** llama.cpp HIP/ROCm backend — compute kernels for LLM inference
> (`mmvq.cu`, `mmq*.cu`, `fattn*.cu`, `vecdotq.cuh`).

---

## 1. Scope

| Item | Detail | Provenance |
|------|--------|------------|
| **Target hardware** | RDNA 2 GPU (gfx1030; RX 6900 XT) | `[supplementary]` |
| **Primary source** | AMD "RDNA 2" ISA Reference Guide, 30-Nov-2020, 283 pp. | `[supplementary]` |
| **In scope** | Shader-level mechanism selection, memory access patterns, occupancy arithmetic, cache-policy bits, instruction scheduling, coherency controls, measurement methodology | — |
| **Out of scope** | CPU-side driver tuning, power management, PCIe/NUMA topology, RGP/driver settings. The ISA is silent on all of these; agents **shall not** infer them. | — |
| **Project context** | llama.cpp fork, HIP backend. See §2 for the numbers of record. | — |

---

## 2. Current state (verify before optimising anything)

Read this block first. Several L-series tasks in §6 are inert or already routed, and re-deriving
that costs a build.

| Fact | Value | Provenance |
|------|-------|------------|
| Warp size in the HIP build | `WARP_SIZE == 32`; RDNA2 path selected by a `__gfx1030__`-derived macro, not by a runtime probe | `[measured: ggml/src/ggml-cuda/common.cuh]` |
| KLONG (2-warp K-loop reduction) | **default 0 (disabled)** — `GGML_RDNA2_MMVQ_KLONG 0` | `[measured: ggml/src/ggml-cuda/mmvq.cu:23]` |
| WIDEN (8 warps, single-token matvec) | **1 in the working tree** — `GGML_RDNA2_MMVQ_WIDEN 1` | `[measured: ggml/src/ggml-cuda/mmvq.cu:27]` |
| SMALLK | **1** — `GGML_RDNA2_MMVQ_SMALLK 1` | `[measured: mmvq.cu:19]` |
| 2ROWS (share q8_1 loads across two weight rows) | **0 (disabled)** — `GGML_RDNA2_MMVQ_2ROWS 0` | `[measured: mmvq.cu:31]` |
| LBMIN (`__launch_bounds__` **minBlocksPerMultiprocessor**) | **1** — `GGML_RDNA2_MMVQ_LBMIN 1` | `[measured: mmvq.cu:14-16]` |
| MMVQ RDNA2 parameter table | Falls through to the generic single-warp case; every RDNA2 arm in `calc_nwarps` is a build switch, not table data | `[measured: mmvq.cu:619-720, 1277-1290]` |
| MMQ RDNA2 config table | Already present and **already routed** (routed/expert selection is separate, see L-16) | `[measured: ggml/src/ggml-cuda/mmq-config-rdna2.cuh, mmq.cu:133-170]` |
| Cross-library KLONG latch | `ggml_cuda_klong_disabled` is **defined in the HIP backend** and referenced from **core** — see §12 | `[measured: src/llama-context.cpp:4,1533-1545; common.cuh:1670; mmvq.cu:10]` |

**Numbers of record for baselines** (do not mix measurement sessions — see §10):

| Workload | Fork | Upstream | Provenance |
|----------|------|----------|------------|
| Dense (27B IQ3_S), tg | 30.89 t/s | 30.05 t/s | `[measured: AGENTS.md]` |
| MoE (35B-A3B), tg | 95.19 t/s | 94.11 t/s | `[measured: AGENTS.md]` |
| MTP enabled | 47.0 t/s | — | `[measured: AGENTS.md]` |

The `31.1 t/s decode / 598.9 t/s prefill / 46.40 t/s MTP` figures in `gfx1030-directive.md` come from
a different build, a different model file and graphs-on production shape. They are not
interchangeable with the AGENTS.md numbers and must never be differenced against them.
Earlier drafts of this file quoted `31.83 → 33.07 t/s` for a hypothetical KLONG fix; that pair is
`[unverified]` and is retained only as an order-of-magnitude note.

---

## 3. Prerequisites

### 3.1 Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `hipcc` / `clang` | Build gfx1030 | The arch flag for HIP builds is `--offload-arch=gfx1030` (or `GPU_TARGETS=gfx1030` through CMake). `-mcpu=gfx1030` is the **assembler/LLVM** form; do not cite it as the build recipe. `[supplementary]` |
| `scripts/bench-exclusive.sh` | **Mandatory** for any measured claim | Enforces no-server + single-process. Failure invalidates every number taken under it. `[measured: docs/gfx1030-benchmarking.md]` |
| `scripts/gfx1030.py` | Consolidated instrument: `probe`, `bench` (incl. `--ab`), `profile`, `trace`, `isa`, `audit`, `census`, `rotcheck`, `predict`, `score`, `report` | Replaces the raw `rocprofv3` invocations in the earlier drafts. `[measured: scripts/gfx1030.py]` |
| `rocprofv3` | PMC counters only | `-L` lists the counter names available on this part; not every documented counter exists. `[measured: docs/gfx1030-directive.md §2]` |
| `llvm-readelf -n`, `--save-temps` | Static resource usage (VGPR/SGPR/LDS/scratch) | Preferred over any profiler for these quantities — see §10.2 |
| `test-backend-ops` | Correctness | `./build/bin/test-backend-ops -b ROCm`. Not `ctest -L main`, which does not run backend-op coverage. `[measured: AGENTS.md]` |

### 3.2 Knowledge

- Instruction formats: SOP1/SOP2/SOPK/SOPC/SOPP, VOP1/VOP2/VOPC/VOP3A/VOP3B/VOP3P, DS, MUBUF, MTBUF, MIMG, FLAT/GLOBAL/SCRATCH, EXPORT.
- EXEC mask, VCC, SGPR/VGPR allocation, LDS semantics.
- The distinction between **compiler-emitted** code (what HIP produces) and **hand-authorable**
  mechanism. A large share of the ISA checklist is unreachable from HIP/CUDA C++ (§5, "Applies?").
- llama.cpp layout: `ggml/src/ggml-cuda/` holds every kernel; `src/` is core and must not depend on
  backend symbols.

### 3.3 Baseline discipline

- Identify a **measurable** workload and a **ceiling** for it before changing code.
- Every change ships with a correctness gate, or it does not ship.
- Report **ms/token with a 95% CI** from a paired A/B (§10.3). A single unpaired run proves nothing
  on this host: run-to-run decode sd is ≈1.56 t/s (`[measured: docs/gfx1030-benchmarking.md]`).

---

## 4. Hardware facts (corrected)

The occupancy numbers in the earlier drafts were wrong and caused two whole task lanes (A3, and the
`I4` draft finding) to be built on a false premise. These are the numbers to use.

| Quantity | Value | Provenance |
|----------|-------|------------|
| Vector registers per SIMD | **65,536** | `[measured: docs/gfx1030-isa-report.md:22]` / KFD topology |
| Max wave slots per SIMD | 16 | KFD topology `[supplementary]` |
| Max wave slots per CU | **32** (2 SIMD32 per CU) | derived, same |
| Max VGPRs per wave | **256** | `[supplementary]` |
| VGPRs/wave for **full** occupancy | **128** (65536 / (128 × 32 lanes) = 16 waves/SIMD) | arithmetic on the two rows above |
| VGPR allocation granularity | **16 dwords (wave32)**, 8 dwords (wave64) | `[ISA §3.6.4 p.18]` |
| SGPRs | 106 architectural + 2 (VCC) + 16 (trap) | `[ISA §3.6.2 p.17]` |
| LDS per CU | **64 kB** (half of the WGP's 128 kB) | `[ISA §2.3.1 p.9]`, `[measured: gfx1030-isa-report.md]` |
| LDS per WGP | 128 kB, 64 banks × 512 × 4 B | `[ISA §2.3.1, §10.1 p.90]` |
| Max LDS per workgroup | **64 kB** | `[ISA §2.3.1]` |
| LDS bank presentation | 64 banks arranged as **two 32-bank halves**, one per SIMD of the pair | `[ISA §10.1 p.90]`, `[measured: gfx1030-isa-report.md S4]` |
| Address → bank mapping | **Not specified by the ISA.** No formula is given anywhere in the manual. | `[ISA — absent]` |

**Consequence (the one that matters).** `ceil(240 / 16) = 240` VGPRs/wave → 240 × 32 = 7,680
registers → `65536 / 7680 = 8` waves/SIMD = 16 waves/CU = **two 256-thread blocks per CU** on the
register side. The second block is then blocked by LDS: `mul_mat_q` IQ3_S at J=64 allocates 45.9 KiB,
and 2 × 45.9 KiB > 64 kB per CU. Both halves are independent, and each alone blocks the second block.
`[measured: docs/gfx1030-isa-report.md §1–§2]`

Corollary: **"target ≤ 64 VGPRs/wave" is not a real target.** At 64 VGPRs/wave a wave would consume
2,048 registers; the machine would be occupancy-capped at 16 waves/SIMD by the *wave-slot* limit
long before registers mattered. Do not carry that number into any task.

---

## 5. Optimization checklist

`Applies?` is the honest filter for a HIP compute kernel. `N/A` means the mechanism is either
authoring-tool territory (the HIP compiler picks the encoding, clause grouping, or export form) or
belongs to a pipeline llama.cpp does not have. Do not spend GPU time on `N/A` rows.

### 5.1 Compute / shader core (S)

| ID | Mechanism | Applies? | Source | Provenance | Verification |
|----|-----------|----------|--------|------------|--------------|
| S-01 | **Wave32** preferred over wave64 | **yes** | §2.1 (p.7) | `[ISA]` | `WARP_SIZE` is 32; `grep -rn "WARP_SIZE" ggml/src/ggml-cuda/` |
| S-02 | Subvector loop (`S_SUBVECTOR_LOOP_BEGIN/END`) | **N/A** — compiler-scheduled, not expressible from HIP | §4.2.1 | `[ISA]` | — |
| S-03 | **Packed math** (`VOP3P`) for dot products | yes | §6.7, §12.10 (p.162-164) | `[ISA]` | Instruction count in `--save-temps` `.s` |
| S-04 | **SDWA** for 16/8-bit ops | yes (usually already emitted) | §6.8 (p.49) | `[ISA]` | Look for redundant `v_mov` around sub-dword ops |
| S-05 | **DPP** for intra-wave exchange — see §7 | yes | §6.9 (p.49) | `[ISA]` | `.s` shows `v_add_f32 ... dpp:` (VOP1/VOP2 only) |
| S-06 | **DPP8/DPP16** swizzle formats | yes | §13.3.9 / §13.3.10 | `[ISA]` | Row/bank mask + `bound_ctrl` + `fi` fields present |
| S-07 | **DS_SWIZZLE_B32** for intra-LDS lane permutation | yes | §12.13.1 (p.207) | `[ISA]` | Note: the compiler already emits `ds_bpermute` 39,381× — verify before "discovering" it `[measured: gfx1030-directive.md]` |
| S-08 | Prefer VOP3A encoding over VOP1/VOP2 | **N/A** — encoding is the compiler's choice | §6.1 | `[ISA]` | — |

> **Wave-size correction.** RDNA 2 VALU issues a wave64 instruction as two wave32 issues, and the
> half with an empty EXEC mask is **skipped** — it does not execute. Per-work-item instruction count
> is therefore *identical* between wave32 and wave64. The benefit of wave32 here is scheduling and
> occupancy granularity, not "eliminating 50% wasted execution". `[ISA §2.1 p.7]`

### 5.2 Memory access (M)

| ID | Mechanism | Applies? | Source | Provenance | Verification |
|----|-----------|----------|--------|------------|--------------|
| M-01 | LDS allocation alignment | partial (HIP dynamic LDS is dword-aligned; larger alignment is a source-level choice) | §3.6.6 (p.18) | `[ISA]` | `.amdhsa_group_segment_fixed_size` in the `.s` metadata |
| M-02 | SGPR load alignment (even / 4-dword) | yes | §3.6.3 (p.18) | `[ISA]` | `s_load_dwordx*` operand alignment in `.s` |
| M-03 | **GLC=0** for read-only buffers | yes | §8.1.10 (p.66) | `[ISA]` | `.s` shows `glc:0` |
| M-04 | **GLC=1** only where required | yes | §8.1.10 (p.66) | `[ISA]` | Atomic correctness tests |
| M-05 | **DLC=1** only for device coherency | yes | §8.1.10 (p.66) | `[ISA]` | `.s` shows `dlc:1` |
| M-06 | **SLC=1** for stream/system policy | yes | §8.1.10 (p.66) | `[ISA]` | Source uses `__builtin_nontemporal_load`; SASS emission is **unverified on this box** `[measured: gfx1030-directive.md §5]` |
| M-07 | Buffer swizzled addressing (`const_swizzle_enable`) | **N/A** — no HIP knob; not emitted for these kernels | §8.1.5 (p.59) | `[ISA]` | — |
| M-08 | Add-TID addressing (`*_ADDTID`) | **N/A** — predates RDNA; not emitted | §8.1.2, §12.13.176-177 | `[ISA]` | — |
| M-09 | MUBUF `LDS=1` (buffer load direct to LDS) | **N/A** — not reachable from HIP C++; buffer-load-to-LDS is already used where the compiler chooses it | §8.1.9 (p.65) | `[ISA]` | — |
| M-10 | GLOBAL/SCRATCH over FLAT when the address space is known | partial | §9.2.2 (p.86), Table 51 (p.83) | `[ISA]` | `SEG` field in `.s` |
| M-11 | Manual buffer stride+offset | **N/A** — index arithmetic already in source | §8.1.5 (p.59) | `[ISA]` | — |

**Cache-bit effects, stated precisely** `[ISA §8.1.10 p.66]`:
- Applies to **loads**. Stores always bypass L1; the L0 for stores is always miss-evict.
- `GLC=0`: L0 hit allowed. `GLC=1`: L0 bypass, read from L2.
- `DLC=0`: L1 LRU. `DLC=1`: L1 miss-evict.
- `SLC=0`: L2 LRU. `SLC=1`: L2 stream.
- `SLC=1, DLC=1` is the "hit-no-allocate" combination for one-shot streams.

### 5.3 Rendering pipeline (R) — **all N/A**

| ID | Mechanism | Applies? | Source |
|----|-----------|----------|--------|
| R-01 | `V_INTERP_P1LL_F16` / `V_INTERP_P2_F16` | **N/A** | §12.12 |
| R-02 | `V_INTERP_P1_F32` / `V_INTERP_P2_F32` | **N/A** | §10.4.2 (p.93) |
| R-03 | Minimise `EXPORT` count | **N/A** — no graphics exports | §11.2.2 |
| R-04 | `DONE=1` on the final export | **N/A** | §11.2.2 |
| R-05 | `VM=1` once per pixel-shader export | **N/A** | §11.2.1 |
| R-06 | `S_SUBVECTOR_LOOP_BEGIN` for vertex shaders | **N/A** | §4.2.1 |

llama.cpp has no raster/export/texture pipeline. The whole of §11 of the ISA is out of scope.

### 5.4 Compiler / code generation (C)

| ID | Mechanism | Applies? | Source | Provenance | Verification |
|----|-----------|----------|--------|------------|--------------|
| C-01 | `S_VERSION` at block top | **N/A** — code-object metadata, not source | §2.5 (p.11), §12.2.1 | `[ISA]` | — |
| C-02 | `S_CODE_END` padding | **N/A** — the assembler emits the required 5× padding | §2.5 (p.11) | `[ISA]` | — |
| C-03 | `S_CLAUSE` grouping | **N/A** — already emitted 18,873× `[measured: gfx1030-directive.md]` | §4.1.1 (p.26-27) | `[ISA]` | — |
| C-04 | **`S_WAITCNT`** correctness | **yes** | §4.4 (p.29-30) | `[ISA]` | Four counters: `VM_CNT`, `LGKM_CNT`, `EXPCNT`, `VS_CNT`; each instruction class completes out of order |
| C-05 | `S_MOVRELS/D_B32` indexed SGPR | **N/A** — not emitted for these kernels | §5.8.1 | `[ISA]` | — |
| C-06 | `V_MOVRELSD_B32` VGPR indexing | **N/A** | §6.6 | `[ISA]` | — |
| C-07 | `V_CVT_PK_*` packed conversion | partial — check the epilogue before claiming a win | §12.12 | `[ISA]` | Instruction count |

### 5.5 Synchronization (P)

| ID | Mechanism | Applies? | Source | Provenance |
|----|-----------|----------|--------|------------|
| P-01 | `DS_APPEND` / `DS_CONSUME` | **N/A** — these are **GDS** ops; this part reports `gds_size_in_kb = 0` `[supplementary: KFD topology]` | §10.4.3 | `[ISA]` |
| P-02 | `DS_ORDERED_COUNT` | **N/A** — GDS absent on this part | §12.13.63 | `[ISA]` |
| P-03 | `DS_GWS_BARRIER` / `DS_GWS_SEMA_*` | **N/A** — GWS lives in GDS; absent | §12.13.24-29 | `[ISA]` |
| P-04 | `S_BARRIER` within a workgroup | yes | §4.3 (p.29) | `[ISA]` |
| P-05 | `S_CBRANCH_EXECNZ` | **N/A** — compiler emits it | §12.5.9 | `[ISA]` |

> 1024 work-items = **32 wave32** per workgroup, which is the wave-slot limit of a CU. Any kernel
> asking for 1024 threads with non-trivial VGPRs cannot co-schedule a second block, regardless of
> LDS. `[ISA §4.3 p.29]`

### 5.6 Special instructions (H)

| ID | Mechanism | Applies? | Source | Provenance | Note |
|----|-----------|----------|--------|------------|------|
| H-01 | **Dot products** | yes | §12.10 VOP3P, opcodes 19-25 (p.162-164) | `[ISA]` | The six forms on p.2-3: `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4` and their accumulate-in-destination `V_DOT*C_*` variants. Also on those pages: the list of FMA instructions removed/replaced in RDNA (`V_FMA_F32` → `V_FMAC_F32`). Already harvested: 3,008 `v_dot4c_i32_i8` vs 568 scalar FMAs `[measured: gfx1030-directive.md]` |
| H-02 | `V_MBCNT_LO/HI_U32_B32` | partial — rarely the bottleneck | §12.12 | `[ISA]` | — |
| H-03 | `V_READLANE_B32` / `V_WRITELANE_B32` | yes | §12.12 | `[ISA]` | VGPR↔SGPR bridge for scalar epilogues |
| H-04 | `V_PERMLANE16_B32` / `V_PERMLANEX16_B32` | yes | §12.12 (p.186) | `[ISA]` | Half-wave lane permutation, no LDS round trip |
| H-05 | `V_PERM_B32` | yes | §12.12 (p.180) | `[ISA]` | **VOP3A only**, opcode 836. Not a VOP2/VOP3 instruction. Byte permute; already used via `__byte_perm` in `vecdotq.cuh` |
| H-06 | `V_CUBE*` | **N/A** | §12.12 | `[ISA]` | Graphics only |
| H-07 | `V_DIV_SCALE_F32` / `V_DIV_FMAS_F32` | partial — no division in hot paths | §12.12 | `[ISA]` | Softmax epilogue only |
| H-08 | `IMAGE_BVH_INTERSECT_RAY` | **N/A** | §8.2.10, §12.16 | `[ISA]` | Ray tracing only |
| H-09 | `S_GETREG_B32` / `S_SETREG_B32` | partial — `SHADER_CYCLES` at §5.8 p.37; `S_MEMTIME` / `S_MEMREALTIME` at p.3 | `[ISA]` | In-kernel timing. `ADD_TID_ENABLE` is also a p.3 status-register bit |

---

## 6. llama.cpp-specific tasks (L-series, gfx1030)

Every row carries the **file:line anchor in the current tree**. Line numbers drift; the directive's
rule is to navigate by symbol. Anchors are given because agents keep re-deriving work that the tree
already answers.

| ID | Task | State | Anchor | Source | Provenance | Verification |
|----|------|-------|--------|--------|------------|--------------|
| **L-01** | **KLONG** — 2-warp K-loop reduction for `ncols_dst == 1` | **disabled** (`KLONG 0`). KLONG changes the FP accumulation order; Q6_K breaks under it. An MTP guard also disables it at run time | `mmvq.cu:23` (switch), `:619-641` (arm), `:634` (Q6_K in the promoted type list), `:1331-1335` (MTP guard) | §4.4, §10.1 | `[measured: mmvq.cu]`; gain `[unverified]` | Fix Q6_K → output "Paris" at temp=0, then paired A/B (expect ≥40 pairs) |
| **L-02** | **WIDEN** — 8 warps for single-token, non-routed matvecs | **enabled in the working tree** (`WIDEN 1`) | `mmvq.cu:27`, `:643-666` | §3.6.4, §10.1 | `[measured: mmvq.cu]` | A/B the switch; it is one compile-time flag, so `--ab` two `.so`s |
| **L-03** | **SMALLK** — borrow the GENERIC table's block width when `ncols_dst == 1` | **enabled** (`SMALLK 1`) | `mmvq.cu:19`, `:668-677` (arm), `:1277-1284` (`should_use_small_k`) | §3.6.4 | `[measured: mmvq.cu]` | A/B; also check it does not starve WIDEN (both are gated on `ncols_dst == 1`) |
| **L-04** | **2ROWS** — one q8_1 activation load shared by two weight rows | **disabled** (`2ROWS 0`) | `mmvq.cu:31`, `:696-712`, `:742-746` | §10.1 | `[measured: mmvq.cu]`; "2.1–3.5× weight traffic" `[hypothesis]` | A/B; the claim is traffic *per row*, not speed |
| **L-05** | **LBMIN** — second argument of `__launch_bounds__`, i.e. **minBlocksPerMultiprocessor**, *not* a block size | **enabled** (`LBMIN 1`) | `mmvq.cu:14-16`, `:752` | §3.6.4 | `[measured: mmvq.cu]` | Read the emitted VGPR count (§10.2); raising LBMIN only helps if registers are the binding limit |
| **L-06** | **FA quant pairs** — add `q5_0-q5_0` and `q4_0-q5_0` to `GGML_CUDA_FA_QUANTS`, removing a full-KV f16 conversion | `q5_0-q5_0` is **committed**; `q4_0-q5_0` is **worktree only** | `ggml/CMakeLists.txt:208` | §8.1.10 | `[measured: ggml/CMakeLists.txt]`; "+2.15% pp512 (597.39→610.25)" is **`[supplementary]`** — the CMake list change cannot by itself be credited, and those t/s values are `[unverified]` | A/B each pair separately; they convert different KV types |
| **L-07** | **IQ3_S sign expansion, MMQ tile loader** — nibble→byte-MSB multiply + `v_perm_b32` + fused xor-add replacing `__vcmpne4`/`__vsub4` | landed in the tile loader. **Misplaced in the earlier drafts**: this is `ggml_cuda_mmq_load_tiles_iq3_s`, *not* a `vec_dot` | `mmq-load-tiles.cuh:1423-1442` (+19/−4). The old idiom still stands in `vecdotq.cuh:1317-1321` → see L-11 | §12.10, §12.12 | `[measured: mmq-load-tiles.cuh]`; "578→258, −55.4%, 1,048,576 cases" are **`[unverified]`** | Instruction count from `gfx1030.py isa` on the prefill path, then A/B |
| **L-08** | **Same sign expansion for the other four IQ types** | **5 sites, not 4** — the fifth is the decode `vec_dot` (L-11), which is the highest-value one and was missing from the earlier list | `mmq-load-tiles.cuh:1157-1161` (IQ2_XXS), `:1220-1224` (IQ2_XS), `:1288-1292` (IQ2_S), `:1354-1358` (IQ3_XXS), **plus** `vecdotq.cuh:1317-1321` (IQ3_S decode) | §12.10, §12.12 | `[measured: same files]`; "~15% per type" `[hypothesis]` | Per-site `isa` mix + A/B; do them one at a time |
| **L-09** | **H14 wide loads** — `get_int_b2` performs two 16-bit loads; widen to 4/8-byte | audit, **not** implemented | `vecdotq.cuh:25-35` (`get_int_b2`); `block_iq3_s.signs` at `ggml-common.h:414-421` (offset 74, 2-byte aligned) | §8.1.9, §8.1.6 | `[measured: vecdotq.cuh, ggml-common.h]`; "2–4× bandwidth" `[hypothesis]` | **Needs measurement, not static analysis**: the pointer is only guaranteed 2-byte aligned, so a wider load is a correctness change. A/B after a `$`-count of `global_load_dword` vs `dwordx4` |
| **L-10** | **MTP conditional KLONG** — disable KLONG for draft graphs | implemented | `src/llama-context.cpp:1533-1545`, `common.cuh:1663-1670`, `mmvq.cu:10`, `:1334-1336` | §4.4 | `[measured: those files]` | **INERT while KLONG defaults to 0** — the latch flips a switch that is compiled out. Verify L-01 first |

Additional L-series work created by the review (see also `docs/rdna2-agent-tasks.md` for the
one-task-per-row form):

| ID | Task | Anchor | Why |
|----|------|--------|-----|
| **L-11** | **IQ3_S decode `vec_dot` sign path** — the same transform as L-07, applied to `vec_dot_iq3_s_q8_1` | `vecdotq.cuh:1296`, `:1317-1321` | This is the *decode* path for the model of record; L-07 only touched prefill tiles. Highest expected value of the sign-path family |
| **L-12** | **Non-atomic latches in `fattn.cu`** | `fattn.cu:701-703`, `:721-723` | Leftover of I1: `static bool logged` guards are read/modified without atomics across threads. Benign for logging, a data race as written; make them `std::atomic<bool>` or `call_once` |
| **L-13** | **Drop or A/B the `__threadfence()`** before the MMVQ cross-warp reduction | `mmvq.cu:916-937` | `__syncthreads()` already orders shared memory. The fence may be pure cost on a path that runs per token |
| **L-14** | **`alloc_failed` scope** | `common.cuh:1705-1712` | It is a function-local `static` — process-global, not per-device. One failed prequant allocation on GPU0 disables the fast path on every device. Make it per-device state |
| **L-15** | **Switch interaction matrix** | `mmvq.cu:619-720` | SMALLK/WIDEN/KLONG are mutually exclusive by construction (`small_k`, `widen`, `halve_iters` guards). Documented in §13; any A/B must set exactly one |
| **L-16** | **MMQ routed selection A/B** | `mmq.cu:133-139` (env kill switch), `:141-152` (type list), `:156-158` (J gate), `:162` (predicate), `:283-292` (stream-K rejection) | The routed gfx1030 selection is on by default; its value has never been paired-measured. `GGML_HIP_MMQ_ROUTED_GFX1030=0` disables it |
| **L-17** | **FA quant pairs are per-consumer** | `ggml/CMakeLists.txt:208` | `q8_0-q5_0` and `q4_0-q5_0` are instantiated for different KV type pairs; adding one does not help a model using the other. A/B them separately, against the cache types the model actually runs (`-ctk/-ctv`) |

---

## 7. DPP, lane exchange, and LDS permutation (corrected)

The earlier draft of this section was wrong in three ways. The corrected picture:

**DPP is a VALU operand modifier, not a data-movement instruction.** `[ISA §6.9 p.49]`
- DPP modifies **source operand 0** of a **VOP1 or VOP2** instruction: the incoming lane value is
  fetched from another lane inside the wave before the operation executes.
- DPP is **not available on VOP3 / VOP3P**. A VOP3-encoded instruction cannot take a DPP modifier.
  This is the point of the §12.19.1 (p.234) table: it is a **prohibition list** (what DPP may not be
  combined with), not a description of DPP. Cite §6.9 for the mechanism and §13.3.9/§13.3.10 for the
  encodings; do not cite §12.19.1 as the DPP reference.
- To use DPP properly you must control three fields: the **row mask** (`dpp_ctrl`, per-lane source),
  **`bound_ctrl`** (whether out-of-range lanes read zero or keep their own value), and **`fi`**
  (fetch-invalid). Omitting `bound_ctrl`/`fi` is the usual reason a hand-written DPP shuffle reads
  garbage at the wave edge.

**`ds_permute_b32` / `ds_bpermute_b32` are DS instructions, not DPP.** `[ISA §12.13]`
- `DS_BPERMUTE_B32` permutes bytes across lanes through the LDS crossbar without an LDS round trip
  in the usual sense; `DS_PERMUTE_B32` does the reverse gather.
- The compiler already emits `ds_bpermute` heavily on this tree (39,381 occurrences)
  `[measured: gfx1030-directive.md §5]`. Treat it as harvested; verify coverage, do not "discover" it.

**Three more lane-exchange mechanisms that are under-used:**

| Mechanism | What it does | Source | Use in llama.cpp |
|---|---|---|---|
| `V_PERMLANE16_B32` / `V_PERMLANEX16_B32` | Permutes lanes within one half of the wave using a 4-bit-per-lane selector, no LDS | §12.12 (p.186) | Warp reductions, scale broadcast — cheaper than `ds_bpermute` + wait |
| `DS_SWIZZLE_B32` | LDS-side lane permutation with the address swizzled in hardware | §12.13.1 (p.207) | Tile re-layout before a dot, avoiding a bank-conflict-prone read pattern |
| **LDS direct read** (LDS as a VALU source) | VALU reads one operand straight from LDS, no `ds_read` + `v_mov` | §10.4.1 (p.92-93) | Tile reads in `mul_mat_q`; saves both an instruction and a VGPR |

Also available and largely unused: `DS_READ2_B32` / `DS_WRITE2_B32` (two dwords in one
instruction, consecutive banks) and `DS_READ_B128` / `DS_WRITE_B128`, plus the LDS atomics
(`DS_ADD_U32`, `DS_MIN/MAX`, ...) in §12.13 (p.205-206).

**Retarget of I6.** The earlier draft pointed I6 at `common.cuh:456-540`. That range is the
`warp_reduce_*` template family, and it is a legitimate target — those reductions currently go
through `__shfl_xor_sync` (which lowers to `ds_bpermute`) — but the win, if any, is in replacing a
full-wave xor-shift tree with `V_PERMLANE16_B32`-based widening plus a final cross-half step, and
that must be measured, not assumed. `[measured: ggml/src/ggml-cuda/common.cuh:456-540]`

---

## 8. ISA mechanisms not yet in the checklist

Added by the ISA review. None of these are promises; each is a candidate with a citation.

| Mechanism | Source | Relevance to a HIP compute kernel |
|---|---|---|
| `S_CLAUSE` / instruction clauses | §4.1.1 (p.26-27) | Groups same-type memory ops so their requests merge. The rule quoted for coalescing lives here, not in §2.4 or §8.1.5. Already emitted 18,873× `[measured]` |
| `FLAT` vs `GLOBAL`/`MUBUF` | §9.2.2 (p.86), Table 51 (p.83) | `FLAT` does an aperture check and can contend with LDS. Where the address space is known, `GLOBAL`/`SCRATCH` is strictly cheaper. Compiler-chosen; verify in `.s` |
| `D16` and alignment modes | §8.1.6 (p.64), §8.1.7 (p.64) | 16-bit data loads/stores; relevant to f16 and sub-dword quant codes |
| Scratch / spills | §9.8 (p.88) | `private_segment_fixed_size` in the metadata. A zero-spill kernel cannot be improved by "reclaiming registers", and a spilling one is a different problem |
| SDWA | §6.8 (p.49) | Sub-dword addressing: operate on a byte/word inside a VGPR without a separate move |
| Packed math + VOP3P modifiers | §6.7, §12.10, p.264 | `V_PK_*` dual-issue-adjacent ops; VOP3P carries `NEG`/`OPSEL` modifiers, so sign handling can be free |
| All dot variants | §12.10, p.162-164 | `V_DOT4_U32_U8`, `V_DOT2_U32_U16`, and the `C` (accumulate) forms — the int8 path uses `V_DOT4C_I32_I8`; the unsigned forms matter for Q8_0 activations |
| `OMOD` / `CLAMP` | §6.2 (p.44), §6.5 (p.48) | Output modifiers: multiply-by-2/4/half and saturate, on the instruction, no extra op |
| `FP_DENORM` / rounding modes | Table 4 (p.15), §6.4 (p.47) | Per-instruction denorm and round control; matters where a reduction's associativity is being changed (i.e. KLONG) |
| `S_LOAD_DWORDX2/X4/X8/X16` | §3.6.3 (p.18) | Wide scalar loads: block scales and row pointers can live in SGPRs instead of VGPRs |
| `S_MEMREALTIME` | §7.2.3 (p.53) | Higher-resolution timer than `S_MEMTIME` for in-kernel bracketing |
| `s_getreg_b32` with `IB_STS` | §5.8 (p.37) | Instruction-buffer status: tells you whether the kernel is ifetch-stalled rather than data-stalled |
| `S_BARRIER` with 1024 work-items | §4.3 (p.29) | 32 wave32 per workgroup = the CU's whole wave-slot budget |
| `ADD_TID_ENABLE` | p.3 | Status bit that makes the hardware add the work-item ID to a base address |
| GDS | §10.4.3 | **Not present on this part** — KFD reports `gds_size_in_kb = 0`. The whole append/consume/GWS family is unavailable `[supplementary: KFD topology]` |

---

## 9. Agent workflow

### Phase 0 — Environment
1. Confirm the GPU is idle and no `llama-server` is running (`pgrep -a -f "llama-server|llama-bench"` must be empty; `rocm-smi --showmemuse` ≈ 0).
2. Use the source of record in §2; do not re-derive the switch states.
3. Pick the workload **and its ceiling** from `docs/gfx1030-directive.md`.

### Phase 1 — Baseline (paired, CI)
1. Build: `cmake --build build --config Release -j12`.
2. Correctness: `./build/bin/test-backend-ops -b ROCm`.
3. Baseline via `scripts/gfx1030.py bench --ab <base.so> <fix.so>` — see §10.3. Store the paired
   ms/token, the sd of the per-pair differences, and the pair count.

### Phase 2 — One change
1. Choose one task ID. Recommended order: **L → M(partial) → S → C-04/H → never R/P**.
2. Implement the smallest possible diff. One switch at a time (§13).
3. Build, `test-backend-ops -b ROCm`, then model output at temp=0 ("Paris", not "333333333").
4. Measure per §10.3 — paired, interleaved, with the process-exclusivity guard.

### Phase 3 — Decide
- Kept only if the **95% CI excludes zero** and correctness passed.
- Below the resolving power of the instrument → `INVESTIGATE`, not `KEEP`. Switching instruments
  (per-class bandwidth from `gfx1030.py profile`) is legitimate; switching to an unpaired run is not.
- Never apply two task IDs in one diff unless the dependency is stated and the interaction matrix
  (§13) says the switches compose.

### Phase 4 — Report
Fill §11 in full. Every row carries its ISA citation, pair count, and 95% CI.

---

## 10. Measurement and validation

### 10.1 Metrics and instruments

| Metric | Instrument | Recipe |
|--------|------------|---------|
| Decode / prefill rate | `gfx1030.py bench --ab` | Paired artifact swap, interleaved; §10.3 |
| Per-class bandwidth | `gfx1030.py profile` | Graph-free kernel trace → scored class table; resolves a geometry change on the class it touches |
| MTP acceptance | `llama-cli` output | `grep "accepted"` |
| Perplexity | `llama-perplexity` | Quality anchor, never a performance metric |
| Instruction mix | `gfx1030.py isa` | Static mix + the 5% ALU gate |
| Structural checks | `gfx1030.py audit` | Mechanical source checks |
| Graph shape | `gfx1030.py census` / `rotcheck` | CPU-only |

`rocprofv3 --stats` is **not** a resource-usage instrument on this build. It does not report
VGPR/SGPR/LDS/occupancy in a form this project can use, and the earlier drafts were wrong to
prescribe it. Use the recipes below.

### 10.2 Resource usage without a profiler

**Preferred: read the compiler's own metadata.**

```
cmake --build build --config Release -j12 -- -save-temps   # or pass --save-temps to hipcc
# then, in the generated .s for the kernel of interest:
grep -E "amdhsa_(next_free_vgpr|next_free_sgpr|group_segment_fixed_size|private_segment_fixed_size)" <kernel>.s
```

- `amdhsa_next_free_vgpr` → VGPRs/wave (already rounded to the 16-dword granularity).
- `amdhsa_next_free_sgpr` → SGPRs/wave.
- `amdhsa_group_segment_fixed_size` → static LDS bytes (dynamic LDS is a launch parameter).
- `amdhsa_private_segment_fixed_size` → scratch/spill bytes. `0` means no spills.

**Alternative:** `llvm-readelf -n <kernel>.hsaco` prints the same fields from the code object's
metadata notes, if the `.s` is not kept. A runtime alternative is `hipFuncGetAttributes`
(`numRegs`, `sharedSizeBytes`, `localSizeBytes`) on the instantiated kernel.

**Occupancy arithmetic** is then done by hand with §4's numbers:
`waves_per_simd = min(16, 65536 / (vgprs * 32))`, `blocks_per_cu = min(waves_per_simd / warps_per_block,
65536 / lds_per_block)`. Note that a `__launch_bounds__` min-blocks hint (LBMIN) is a *request*; the
compiler may ignore it, and the metadata tells you what actually happened.

**Bank conflicts.** `rocprofv3 -L` first — this part does not expose every documented counter
(`SQ_INSTS_VALU` and `SQ_WAIT_ANY` are known dead on gfx1030 `[measured: gfx1030-directive.md §2]`).
If `SQ_LDS_BANK_CONFLICT` is listed, then:

```
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -o <out> -- ./build/bin/llama-bench -m <model> -ngl 99 ...
```

If it is not listed, the lane is unmeasurable on this card and the only remaining argument is
structural: a wave32 reading 32 consecutive dwords hits 32 distinct banks by construction.

### 10.3 Comparison methodology (paired, not 3-run)

**Superseded rule:** "minimum 3 runs, accept ≥1% on t/s". Unpaired repeats cannot resolve the
effects this campaign chases — decode sd is ≈1.56 t/s, i.e. ~2.3%, larger than every win on the
board. `[measured: docs/gfx1030-benchmarking.md]`

**Current rule:**

1. Build two artifacts (base and fix) and swap them **interleaved**, one pair at a time:
   `scripts/gfx1030.py bench --ab <base.so> <fix.so> --pairs N`.
2. Report **ms/token**, the paired mean delta, the **sd of the per-pair differences**, and the
   **95% CI half-width**. `bench --ab-stats "d1,d2,..."` does the arithmetic offline.
3. Size the run in advance: `n = (1.96 * sd_pair / delta)^2`. At the measured `sd_pair = 0.64 ms`,
   0.20 ms needs ~40 pairs, 0.14 ms ~80, 0.08 ms ~250. `--min-detectable <ms>` prints the count.
4. Accept only when the CI excludes zero. A delta smaller than the CI is `INVESTIGATE`.
5. Exclusivity: `scripts/bench-exclusive.sh` is mandatory; **one measurement process at a time**,
   never backgrounded behind another run.
6. Same model, same prompt, same seed, same `-fa` / `-ctk` / `-ctv` / `-ngl` / `-ncmoe` on both arms.

Full host rules: `docs/gfx1030-benchmarking.md` (this section is a summary and loses any conflict).

### 10.4 Verification checklist

- [ ] `./build/bin/test-backend-ops -b ROCm` 100% pass
- [ ] Model output "Paris" at temp=0 (not "333333333")
- [ ] Perplexity not regressed (a >0.1% move is suspicious, not automatically fatal)
- [ ] No memory violation (`TRAPSTS.MEM_VIOL`, §3.11)
- [ ] No exception trap (`TRAPSTS.EXCP`, §3.10.1)
- [ ] VGPR/SGPR/LDS/scratch read from the artifact metadata (§10.2), not estimated
- [ ] Paired A/B with ≥ the required pair count, CI reported
- [ ] Exactly one interaction-matrix switch changed (§13)

---

## 11. Reporting template

```markdown
## Task ID: [e.g., L-11]

| Field | Value |
|-------|-------|
| **Change description** | [what was modified, one sentence] |
| **Anchor** | [file:line, symbol] |
| **ISA citation** | [§section p.page] or "none — this is a code-structure change" |
| **Provenance of the claim** | [measured: …] / [supplementary] / [hypothesis] |
| **Base artifact / fix artifact** | [path to each .so] |
| **Pairs** | [N interleaved pairs] |
| **Δ ms/token (paired mean)** | [+X.XX ms] |
| **sd of per-pair differences** | [X.XX ms] |
| **95% CI half-width** | [±X.XX ms] |
| **CI decision** | [excludes zero → keep] / [includes zero → investigate / drop] |
| **Δ prefill (separate run)** | [+X% / N/A] |
| **Correctness** | ✅ / ❌ / ⚠️ (with the failing test name) |
| **Switches changed** | [exactly one, named] |
| **Notes / anomalies** | [tool limitations, dead counters, confounds] |
| **Decision** | commit / revert / investigate |
```

---

## 12. Known landmines

These are defects and hazards that exist in the tree now. They are not optimizations; they change
what "verified" means.

| # | Landmine | Anchor | Consequence | Action |
|---|----------|--------|-------------|--------|
| 1 | **Cross-library symbol**: `ggml_cuda_klong_disabled` is defined in `mmvq.cu` (HIP backend) and declared `extern` in **core** | `src/llama-context.cpp:4`, `common.cuh:1670`, `mmvq.cu:10` | A build without the HIP/CUDA backend has an unresolved symbol; core now knows a backend's name | Move the state to a backend-visible interface (a callback or a field the backend reads), not a raw global |
| 2 | **Process-global allocation latch** | `common.cuh:1705-1712` (`static std::atomic<bool> alloc_failed`) | One failure on one device disables the prequant fast path on every device in the process | Per-device state (L-14) |
| 3 | **Non-atomic logging latches** | `fattn.cu:701-703`, `:721-723` | Benign in practice, a data race as written; also makes the one-shot log line unreliable under concurrency | `std::atomic<bool>` + `exchange`, or `call_once` (L-12) |
| 4 | **Env kill switches** (runtime, undocumented in the code path they gate) | `GGML_HIP_MMQ_ROUTED_GFX1030` (`mmq.cu:135`), `GGML_CUDA_MMQ_PREC` (`mmq.cu:99`) | A measurement taken with a stray env var is a measurement of a different build | Print the environment in the report; unset before benchmarking |
| 5 | **Compile-time switches** | `GGML_RDNA2_MMVQ_{LBMIN,SMALLK,KLONG,WIDEN,2ROWS}` (`mmvq.cu:14-31`) | They select *different kernels*, so an "A/B" is an artifact swap, not a flag flip | Two `.so`s, `bench --ab` |
| 6 | **Silent fallthrough** | MMVQ has no RDNA2 table row; `calc_nwarps` returns 1 unless a switch arm fires (`mmvq.cu:1277-1290`) | "The RDNA2 path" is mostly the generic path plus build switches | Read §2 before assuming a row exists |
| 7 | **Dead counters on this part** | `SQ_INSTS_VALU`, `SQ_WAIT_ANY`, `SQC_LDS_IDX_ACTIVE`, `FETCH_SIZE`; all `DRAM_*`/`TCC_*` absent | A PMC recipe copied from another card measures nothing | `rocprofv3 -L` first |

---

## 13. Switch interaction matrix

The five `GGML_RDNA2_MMVQ_*` switches are **not independent**. They are read in `calc_nwarps`
(`mmvq.cu:619-720`) under mutually exclusive guards, and they all key off `ncols_dst == 1`.

| Switch | Guard | Composes with | Excludes |
|--------|-------|---------------|----------|
| `KLONG` | `table_id == RDNA2 && ncols_dst == 1 && !small_k && halve_iters` | — | `SMALLK` (`!small_k`), and `WIDEN` (which requires `!halve_iters`) |
| `WIDEN` | `table_id == RDNA2 && widen && ncols_dst == 1 && !small_k && !halve_iters` | — | `SMALLK`, `KLONG` |
| `SMALLK` | `table_id == RDNA2 && small_k && ncols_dst == 1` | suppresses the other two | `KLONG`, `WIDEN` |
| `2ROWS` | separate arm at `mmvq.cu:696-712` / `:742-746` | orthogonal; check the launch bounds | — |
| `LBMIN` | `__launch_bounds__` second argument (`mmvq.cu:752`) | orthogonal | — |

Consequences for experiments:
- Turning `SMALLK=1` and `KLONG=1` together measures neither. `small_k` wins the `if` ordering.
- `KLONG` requires `halve_iters`, which is why the MTP guard (`mmvq.cu:1331-1335`) exists and why
  L-10 is inert while `KLONG` is 0.
- Any report must name the full switch vector, not just the one changed.

---

## 14. Dependencies and order

Taken from `docs/gfx1030-directive.md` §1 (standing order), not invented here:

```
WP-F (GQA multi-row verify)  →  SLC weight-stream / Probe D  →  3.6 µs floor test
      →  WP-B (MoE expert-width gate)  →  H14 wide decode loads  →  WP-A (scale pooling)
```

The L-series here sits underneath that order: L-01/L-10 (KLONG) and L-15 (switch matrix) are
prerequisites for any MMVQ measurement claim; L-07/L-08/L-11 (sign path) are prefill and decode
instruction-mix work that can proceed without the GPU-bound packages; L-06/L-17 (FA pairs) are
gated on the cache types the model actually runs.

---

## 15. Open questions and gaps

Items the ISA does **not** answer. Flagged, not invented.

| # | Gap | Impact | Mitigation / status |
|---|-----|--------|---------------------|
| 1 | No performance tuning guidance — the ISA is a reference | Cannot derive an optimum from the manual | Measure; use `gfx1030.py profile` |
| 2 | No V_DOT* throughput (ops/cycle) | Cannot predict a quantized-matmul speedup | Benchmark each variant; the dp4a ceiling was **re-derived** to ~1,915 t/s from `[measured: gfx1030-directive.md §2]`, not from the ISA |
| 3 | No LDS-vs-DPP latency comparison | Cannot choose between them a priori | Benchmark or stay with what the compiler emits |
| 4 | **No address→bank mapping** | Bank conflicts cannot be predicted statically | Structural argument only (32 consecutive dwords → 32 banks) `[measured: gfx1030-isa-report.md S4]`; no counter on this part |
| 5 | **No VALU/LDS/VMEM latency tables** | Cannot model a stall analytically | Counters only |
| 6 | **No VGPR operand-bank conflict rules** | Cannot explain some stalls | Out of reach |
| 7 | **No mechanism for CU-vs-WGP LDS mode** | Cannot decide the per-CU vs per-WGP split from the ISA | Compiler default is WGP mode; the mode bit lives in `COMPUTE_PGM_RSRC` `[supplementary]` |
| 8 | **No guidance on what the HIP compiler already emits** | "Unused instruction" claims are frequently wrong | Disassemble before claiming; three such claims were already refuted (ds_bpermute, S_CLAUSE, dot pipes) `[measured: gfx1030-directive.md §10]` |
| 9 | No cache sizes in the ISA | Cannot size working sets from the manual | `[supplementary]`: 4 MB L2, 128 MB Infinity Cache above it; `GL2C_*` sees L2 only |
| 10 | No memory-bandwidth specification | Cannot compute a peak from the manual | `[measured: gfx1030-directive.md]`: 500.6 GB/s read ceiling, read-only float4 stream, 2 GiB |
| 11 | KLONG Q6_K correctness bug | KLONG is off; L-10 is inert | Needs GPU debugging (rocgdb); the FP-associativity change is the suspect `[hypothesis]` |

Shader padding (§2.5 p.11) is **not** a gap: the manual specifies 64 dwords / 256 B of padding,
terminated by five `S_CODE_END`, with prefetch limited to 1–3 cache lines, and the assembler emits
it. `[ISA §2.5 p.11]`

---

## Appendix A — Key ISA instructions

| Instruction | Format | Description | Use | Provenance |
|---|---|---|---|---|
| `V_FMAC_F32` | VOP2 | Fused multiply-add (replaces `V_FMA_F32` in RDNA) | Dot products, reductions | `[ISA p.2-3]` |
| `V_DOT4C_I32_I8` | VOP3P | 4× int8 dot, accumulate in dest | Quantized matmul (already used) | `[ISA §12.10 p.162-164]` |
| `V_DOT8_I32_I4` | VOP3P | 8× int4 dot | Low-bit quantized matmul | `[ISA §12.10]` |
| `V_DOT2_F32_F16` | VOP3P | 2× f16 dot | Half-precision matmul | `[ISA §12.10]` |
| `V_DOT4_U32_U8` | VOP3P | unsigned 4× u8 dot | Q8_0 activations | `[ISA §12.10]` |
| `DS_READ_B32` / `DS_WRITE_B32` | DS | LDS read/write 32-bit | Shared memory | `[ISA §12.13]` |
| `DS_READ2_B32` / `DS_WRITE2_B32` | DS | Two consecutive dwords, one instruction | LDS tiles | `[ISA §12.13 p.205-206]` |
| `DS_BPERMUTE_B32` | DS | Cross-lane byte permutation (not DPP) | Reductions, shuffles | `[ISA §12.13]` |
| `DS_SWIZZLE_B32` | DS | LDS-side lane permutation | LDS re-layout | `[ISA §12.13.1 p.207]` |
| `V_PERMLANE16_B32` / `V_PERMLANEX16_B32` | VOP1/VOP2-class | Permute within a half-wave | Warp reduce, broadcast | `[ISA §12.12 p.186]` |
| `S_WAITCNT` | SOPP | Wait on `VM/LGKM/EXP/VS` counters | Dependency resolution | `[ISA §4.4 p.29-30]` |
| `S_BARRIER` | SOPP | Workgroup barrier | Phase boundaries | `[ISA §4.3 p.29]` |
| `V_PERM_B32` | **VOP3A** (opcode 836) | Byte permutation | Sign-path optimisation | `[ISA §12.12 p.180]` |
| `V_LSHLREV_B32` / `V_AND_B32` / `V_OR_B32` / `V_XOR_B32` | VOP2 | Bit manipulation | Sign/mask work | `[ISA §12.7 p.130]` |
| `S_LOAD_DWORD` / `X2` / `X4` / `X8` / `X16` | SMEM | Scalar load, 1..16 dwords | Metadata in SGPRs | `[ISA §3.6.3 p.18]` |
| `BUFFER_LOAD_DWORD` / `_DWORDX4` | MUBUF | Vector load | Global reads (narrow vs wide is the H14 question) | `[ISA §8.1]` |
| `BUFFER_LOAD_*_LDS` | MUBUF | Load direct to LDS; `TFE=0`; M0[15:0] carries the LDS offset | Bypass VGPRs | `[ISA §8.1.9 p.65]` |
| `s_getreg_b32` | SOP1 | Read status (`SHADER_CYCLES` §5.8 p.37, `IB_STS` §5.8 p.37, `ADD_TID_ENABLE` p.3) | In-kernel timing, ifetch stalls | `[ISA]` |
| `S_MEMTIME` / `S_MEMREALTIME` | SOP1 | 64-bit clock | Bracketing | `[ISA p.3, §7.2.3 p.53]` |

## Appendix B — LDS organisation

- **64 banks**, presented as **two 32-bank halves**, one per SIMD of the pair. `[ISA §10.1 p.90]`
- Each bank: **512 entries × 4 B = 2048 B**; total **64 × 2048 = 131,072 B = 128 kB** per WGP.
  `[ISA §10.1 p.90]`
- **Max 64 kB** per workgroup (one CU's half). `[ISA §2.3.1]`
- LDS bandwidth is roughly **10×** global-memory bandwidth for the same access. `[ISA §10.1]`
- `read2`/`write2` access two consecutive dwords; `_B128` accesses four. `[ISA §12.13 p.205-206]`
- **The ISA specifies no address→bank mapping.** The `(addr/4) % 64` formula in the earlier drafts
  is not in the manual. What the manual does support is the *shape*: a wave32 reading 32
  consecutive dwords covers 32 distinct banks in its own half, hence conflict-free by construction.
  `[ISA — absent]`, `[measured: gfx1030-isa-report.md S4]`
- Conflict cost: two accesses to the same bank in one cycle serialise. `[ISA §10.1 p.90]`

## Appendix C — Constants

| Constant | Value | Provenance |
|----------|-------|------------|
| Wave32 lanes | 32 | `[ISA §2.1 p.7]` |
| Wave64 lanes | 64 (issues as 2× wave32; empty half skipped) | `[ISA §2.1 p.7]` |
| VGPR group (wave32) | 16 dwords | `[ISA §3.6.4 p.18]` |
| VGPR group (wave64) | 8 dwords | `[ISA §3.6.4 p.18]` |
| Registers per SIMD | 65,536 | `[measured: gfx1030-isa-report.md:22]` |
| Wave slots per SIMD / per CU | 16 / 32 | `[supplementary: KFD topology]` |
| Max VGPRs per wave | 256 | `[supplementary]` |
| VGPRs/wave for full occupancy | 128 | arithmetic |
| SGPRs | 106 + 2 (VCC) + 16 (trap) | `[ISA §3.6.2 p.17]` |
| LDS banks / entries / bytes per bank | 64 / 512 / 2048 | `[ISA §10.1 p.90]` |
| LDS per WGP / per CU / per workgroup | 128 kB / 64 kB / 64 kB | `[ISA §2.3.1, §10.1]` |
| SIMD32 per WGP / per CU | 4 / 2 | `[ISA §10.3 p.91, §1.1 p.6]` |
| GDS present? | **No** (`gds_size_in_kb = 0`) | `[supplementary: KFD topology]` |
| GLC / DLC / SLC effects | see §5.2 | `[ISA §8.1.10 p.66]` |
| Infinity Cache | 128 MB, above a 4 MB L2 | `[supplementary]` |

---

## 16. Constraints

- Ground ISA claims in the manual; tag everything `[ISA]` / `[measured]` / `[supplementary]` / `[hypothesis]`.
- Never commit or push — the owner lands by hand. Never write PR text, commit messages, or reviewer replies.
- Never launch, stop, query, or health-check `llama-server`.
- GPU work only while the owner's server is idle; one measurement process at a time, through `scripts/bench-exclusive.sh`.
- Report **ms/token with a 95% CI** from paired A/B; never a single unpaired run.
- Every claim needs a citation and a repro command.
- One switch / one change at a time; name the full switch vector (§13).
- Correctness: `./build/bin/test-backend-ops -b ROCm` 100% pass **and** model output "Paris" at temp=0.
