# gfx1030 Optimization Campaign — single directive

*(Consolidated: agent directive + MTP/batched-MTP/IC hypotheses + capability map. Other `gfx1030-*.md` files superseded.)*

**Repository:** `Stormrage34/llama.cpp-turboquant-hip` (branch `turboquant-rebase`, HEAD `d327bf1f0`)
**Hardware:** AMD RX 6900 XT, gfx1030, wave32, 80 CUs × 2 SIMD32 = 160 SIMDs, 128 MB Infinity Cache. Sustained clock 2525 MHz. **DRAM read ceiling 500.6 GB/s measured** (read-only float4 stream, 2 GiB ≫ 128 MB IC). Never derive a ceiling from `hipMemcpy` — a D2D copy reads+writes and understates by ~15%.
**Models:** dense `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (11.35 GB/token weights) · MoE `Qwen3.8-35B-A3B-Q4_K_M-exps-IQ3_S-im.gguf` (16.05 GB). **Both are targets.**
**Operating points (confirmed):** dense ≤112k ctx, no MTP (second KV can't fit) · MoE 256k ctx, MTP on.
**At these context sizes KV is tens of GB — a second weight-scale stream, never IC-resident.** The GQA/attention path and KV layout are first-class targets.
**Production state:** decode 31.1 t/s graphs-on (**Vulkan 32.6 — 1.5 t/s ahead**, a real unexplained deficit vs a driver stack we do not control) · prefill 598.9 t/s = **31.3% of the ~1,915 t/s dp4a ceiling** (upstream parity is not the bar) · MoE pp512 2,323 t/s · MoE decode 90.75 t/s · MTP n_max2 46.40 t/s. **Idle 14.7% graphs-on** (median gap 3.60 µs, uniform across boundaries — the earlier 37% was a graphs-off artifact).
**⚠ MoE stream caveat:** the 16.05 GB MoE file is 14.95 GiB — it does not fit the ~15.98 GiB usable VRAM at zero context, and at 256k the KV cache alone is 7,168 MiB, leaving ~8.8 GiB for the weight set → **~6 GiB of experts must sit on host behind PCIe 4.0 x16.** The declared operating point runs a different `-ncmoe`, a different byte mix (PCIe, not DRAM), and different tile geometry (fewer resident expert layers = more host round-trips per token). **Every MoE number in this doc (pp512 2,323 / tg16 90.75) was measured at `-ncmoe 0` — all experts in VRAM — a configuration unreachable at 256k.** No measurement exists at the declared MoE operating point; WP-G prices it.
**Landed (uncommitted — see §8):** prequant arena (quantize 127.65→24.0 launches/token), {ADD,RMS_NORM,MUL} preadd fusion, 22-type RDNA2 MMQ table, gfx1030 GQA decode attention (4.4–5× @ 32k ctx), dispatch 1522.65→1417.65/token.
**Repo policy:** agents produce analysis and diffs; a human reviews and lands everything (AGENTS.md). Navigate code by **symbol, not line number** — the working tree is ~62 files ahead of pushed `f70563a`.

---

## 1. State of the campaign

**Decode is open — WP-0 measured it; WP-E then measured the gaps.** The matvec saturates DRAM *inside its own window* (508 GB/s vs 500.6 peak). But the "37% idle" was a graphs-off artifact: **graphs-on idle is 14.7%**, the gaps are a uniform ~3.6 µs launch floor (median, 19,639/19,642 <20 µs, no boundary pattern, 1,404 dispatches/token) — device-side dispatch turnaround, not a fixable boundary. Realistic WP-E yield: ~0.2 t/s (≈1 t/s per 278 launches removed; ~50 foldable dispatches left). **The launch floor itself is now the sharpest open number** — see the floor test in §3.

**Standing order (re-derived from evidence):** **0. checkpoint commit → 1. WP-F (GQA multi-row verify — IN FLIGHT, ≈56 t/s upside) → 2. SLC weight-stream / Probe D (in flight) → 3. 3.6 µs floor test → 4. WP-B expert-width gate → 5. H14 wide decode loads → 6. WP-A scale pooling.** WP-D residual / MT lanes / IC lanes: see closed lanes and Annex C gates.

## 2. Measured evidence

| Claim | Number | Source |
|---|---|---|
| Read ceiling | 500.6 GB/s | WP-0 read-only float4 stream |
| Matvec inside window | 508 GB/s, 13.18 ms/token | WP-0 trace, 36,251 dispatches |
| Per-token idle (graphs-off trace) | 12.37 ms, 37% — **artifact of graphs-off capture** | WP-0 trace |
| Per-token idle (**graphs-on, production**) | **14.7%**, median gap 3.60 µs, 19,639/19,642 <20 µs uniform, 1,404 dispatches/token, ceiling ~36.5 t/s | gap census |
| Dispatch floor | ~3.6 µs/launch ≈ 1 t/s per 278 launches; ~50 foldable left → ~0.2 t/s. **Self-consistent:** prequant arena removed 103 launches → predicted +0.37 t/s, tripwire measured +0.21 | floor test pending (1000-kernel graph chain) |
| Gap flatness | 3 of 19,642 gaps ≥20 µs; correlation(gap, preceding kernel duration) = 0.02 — flat at every boundary, no pair to blame | gap census |
| MTP n_max 2 | 46.40 t/s, 6.28 GB/tok, 12.56 GB/pass, L2 34.62% | user rocprofv3 — 1.90 of 2.0 reuses real |
| MTP n_max ≥3 | strictly worse (44.70 / 39.70) | same |
| MoE over-tiling | **hypothesis, not measurement** — arithmetic ours (2,097,152 wg vs ~2,048 routed rows ≈ 4×), gains borrowed (+15.3%/+42.4% other fork/model); falsifier stands at MoE pp512 2,323 t/s | user delta |
| MTP bandwidth gap — **mechanism found** | Custom GQA kernel rejects Q rows >1 → every MTP verify (2–5 Q rows) falls back to the VEC path, which **reads+dequantises K/V once per Q head** — 6× KV traffic at gqa_ratio 6. Eff. BW 353→291 GB/s = the re-read's shape | oracle review, source-verified |
| L2 hit trajectory | 24.3% decode → 34.6% MTP → 59.2% pp512 | measured — **L2 only** (see §5 hierarchy note) |
| Prefill compute ceiling | **corrected ×2:** 51.7 TMAC/s ÷ ~27e9 MACs/token ≈ **1,915 t/s** → at 598.9 = **31.3% of ceiling** (~3.2× headroom, not 1.6×). The dp4a multiply IS the dot operation — 2-FLOP-per-multiply arithmetic invented the fake 957.6 | Stefan's re-derivation + explorer cross-check (dense pp = 32.34 TFLOPS-eq = 125% of FP32 wall → 4 mults/instr confirmed) |
| Dot pipes | **already in use** — 3,008 `v_dot4c_i32_i8` vs 568 scalar FMAs in the build. "dequantise then scalar FMA" premise dead | disassembly count |
| Decode load width | **the surviving real gap** — 5,302 narrow `global_load_dword` (4B) vs only 932 `global_load_dwordx4` (16B) in decode weight loads: the bandwidth-bound matvec issues narrow loads | disassembly count |
| SLC SASS status | explorer's "0 slc" verdict **void** — it grepped a .s 3 days older than the change. Hot-path sites verified in source: mmq-load-tiles.cuh:1412/1438/1439, vecdotq.cuh:1311/1336/1338 (**bq3 dequant = this model's weight type**). Re-verification blocked on box tooling | Stefan's timestamp audit |
| Weights/token | 11.35 GB (production arithmetic; the 6.7 GiB alloc trace was a partial-load capture) | WP-0 correction |

**Counters on this part:** working — `GRBM_*_BUSY`, `GL2C_MC_{RDREQ,WRREQ}[_sum,_STALL]`, `EA_UTIL`. Dead — `SQ_INSTS_VALU` (undercounts), `SQ_WAIT_ANY`, `SQC_LDS_IDX_ACTIVE`, `FETCH_SIZE` (err 38). Absent — all `DRAM_*`, `TCC_*` (MALL/IC stays dark). Bytes = `GL2C_MC_RDREQ × 64 B`.

## 3. Work packages (priority order)

### WP-F — GQA multi-row verify *(priority 1 — IN FLIGHT: ses_e7f36e7a9ffep7innr4jdJRnCu)*

Relaxing the `ne[1] != 1` gate + adding a query-row loop that keeps one K/V read per KV head. Gate: `FLASH_ATTN_EXT` 4046/4046 correctness + 20-pair MTP measurement against 46.40 t/s baseline.

**Finding:** our custom GQA attention kernel is decode-only — it rejects any pass with Q rows >1 (`Q->ne[1] != 1 → return false`, fattn.cu ~:591 in the working tree). Single-token decode qualifies; an MTP verify pass (2–5 query rows) does not — so **every MTP verify falls back to the generic VEC path, which reads and dequantises K/V once per Q head.** Six query heads per KV head = 6× KV traffic on the pass that produces most tokens — exactly the 353→291 GB/s effective-BW shape (WP-D). Not the draft, not accept logic, not split-K.

**Work:** extend the custom GQA kernel to `ncols > 1` so verify passes stay on the fast path. Same file family as the already-landed GQA work — self-contained, gfx1030-owned.
**Measure:** verify-pass effective BW recovers toward ~353 GB/s; the verify-window kernel name in the trace changes from VEC to the custom kernel; MTP t/s rises above 46.40 on the same n_max 2 — recovering to plain-decode efficiency ≈ **~56 t/s**, touching no speculator logic.

### WP-B — MoE expert-width gate *(priority 4 — hypothesis, one-line)*

**Claim (hypothesis — over-tiling arithmetic is ours, gains are borrowed):** RDNA2 is missing from the expert-width tile condition — `ggml_cuda_should_use_mmq` (mmq.cu, RDNA3/4 arm ~line 349-366; dossier cited :202). The ~4× over-tiling (2,097,152 wg for ~2,048 routed rows) is our arithmetic; +15.3%/+42.4% come from a different fork on a different model. **Falsifier:** MoE pp512 must move from 2,323 t/s — **but measured at both `-ncmoe 0` and the operating-point `-ncmoe`** (WP-G step 0 decides which is real); a gain at -ncmoe 0 may not survive 256k. Land **after** the checkpoint commit (it modifies mmq.cu).

### WP-C — MTP config *(resolved)*

Off on dense ≤112k; **on for MoE at 256k ctx, n_max = 2.** n_max ≥3 is strictly worse. Only remaining state: WP-D.

### WP-D — MTP bandwidth gap *(mechanism identified → WP-F)*

Bytes fall 1.90×, speed rises 1.50×, eff. BW 353→291 GB/s. Root cause found: the VEC fallback's per-Q-head K/V re-dequant on every verify pass (WP-F). Residual decomposition (acceptance waste, draft stream) is a minor term — re-measure after WP-F lands before spending on it.

### WP-G — MoE PCIe ceiling *(new open measurement — nobody has priced it)*

At 256k the MoE runs with ~6 GiB of experts behind PCIe 4.0 x16 (~32 GB/s theoretical, ~25 GB/s practical). **Step 0 (one run, decides whether the concern is real):** record which `-ncmoe` the 256k operating point actually uses, then measure pp512/tg16 *there* — if experts turn out DRAM-resident the concern evaporates. If it stands: measure the MoE path's effective GB/token and achieved GB/s in a production-shape trace (`GL2C_MC_*` can't see PCIe — derive it from bytes-transferred vs wall-time in trace windows). **WP-B consequence:** the 2,323 t/s falsifier is anchored at unreachable `-ncmoe 0` — WP-B must be measured at BOTH `-ncmoe 0` and the operating-point `-ncmoe`; a gain at 0 may not survive the move.

### WP-H — IC cliff map (Stefan's work order, reworked for this card)

**Objective (accepted):** map the prompt-size/context-size points where working sets overflow the cache tiers and TTFT/decode rate breaks non-linearly; deliver context-cap + slot-boundary rules.

**Corrections to the work order (binding):**
- **`mall_hit`/`mall_request` counters do not exist on gfx1030** — MALL has no programmer-visible counters; `GL2C_*` sees only the 4 MB L2. `umr` gives register dumps, not hit rates; RCD is Windows-only. **Method: empirical boundary-derivative testing** — `achieved_bandwidth = tokens/s × active working-set bytes`; cross-check against `GL2C_MC_RDREQ × 64 B`: **a spike in this counter's growth rate relative to working-set growth = MALL eviction cliff** (requests routed to VRAM).
- **Cliff positions are predictable — verify, don't scan.** KV at 256k = 7,168 MiB ⇒ ~28 MB per 1k tokens. The KV working set crosses tiers at: **L2 4 MB ≈ ~140 tokens; IC 128 MB ≈ ~4.5k tokens.** High-resolution sweeps target the knees: **100–200 tokens** (L2) and **4,000–5,000 tokens** (IC) — linear 512-steps cancelled.
- Weights are a pure stream (11.35–16 GB ≫ IC) — the prefill cliff is in the **KV/intermediate working set**, not the weights.

**Stress matrix (executable):** T1 targeted knee sweeps — prefill t/s + TTFT slope breaks at 100–200 and 4,000–5,000 tokens (temp 0 / top_p 1 static). T2 slot thrash — N parallel slots until aggregate KV crosses tiers / PCIe offload engages; detect via wall-time + `GL2C_MC` traffic surge. T3 context-swap cost — forced context swap = KV host round-trip over PCIe; all unified-memory-bus timeline telemetry feeds WP-G.

**Deliverable:** cliff locations in tokens, per-tier bandwidth numbers, context-cap + parallel-slot rules for the declared operating points.

### WP-A — MMQ scale pooling *(priority 6 — largest measured pool, largest blast radius)*

Re-priced by the corrected ceiling: prefill has ~3.2× headroom to ~1,915 t/s — the largest single pool of unclaimed work in the programme. Mutates a data layout shared with every CUDA backend — deferred until the checkpoint lands and WP-B/the SLC run are measured. **Gate if funded: pp512 ≥ 650 t/s + KLD n≥256.** If funded later: the alternative is the **dot2-f16 restructure** (pre-scaled f16 activations → `V_DOT2_F32_F16`, the +50–67% form hipfire validated) which removes scales from the inner loop entirely without touching `block_q8_1_mmq` — slot-count both before choosing. Anchors: `block_q8_1_mmq` (mmq.cuh:28), `quantize_mmq_q8_1` (quantize.cu:277), `vec_dot_q8_1_q8_1_impl` (vecdotq.cuh), `MMQ_Q8_1_DS_LAYOUT_D2S6` (existing per-64-scale arm = nearest template). Quality gate if touched: backend-ops + perplexity + KLD n≥256.

### WP-E — Launch floor *(collapsed — was graphs-off artifact; one experiment remains)*

**Falsified as a lane:** the 37%/12.37-ms numbers were graphs-off. Graphs-on: idle 14.7%, 19,639/19,642 gaps <20 µs, median 3.60 µs, uniform across all boundaries, 1,404 dispatches/token, ceiling ~36.5 t/s. Already decomposed: device-side dispatch turnaround — 0 of 3 large gaps contain an in-flight graph launch; host blocks 31.6/35.4 ms in `hipStreamSynchronize` but that is not the gap mechanism. **No per-boundary fix exists.**

**What remains — the 3.6 µs dispatch-floor test (one experiment, not a lane):** a synthetic chain of ~1,000 trivial kernels inside one graph. If it also shows 3.6 µs → the floor is hardware, every dispatch-removal plan is capped ~1 t/s per 278 launches. If it shows ~1.5 µs → ~2/3 of the floor is ours ≈ 2.8 ms/token ≈ ~3 t/s — more than WP-E ever offered. Cheapest open experiment in the program.

## 4. Measurement protocol (mandatory)

No finding ships without a measured number + its ceiling + a falsifier, else tagged `Needs validation`.

- **Peak:** read-only float4 stream at 2 GiB (WP-0's proven method — never `hipMemcpy`).
- **Bytes:** `GL2C_MC_RDREQ × 64 B`; never `FETCH_SIZE`.
- **Timeline:** rocprofv3 kernel/API/copy trace matching `profile_baseline/Storm/` CSVs; scoped PMC pass per-kernel.
- **Idle:** `GRBM_*_BUSY` counter set for gap attribution.
- **SASS:** `llvm-objdump` — confirm what the compiler actually emitted (cache bits, clause, dot form). ⚠ Device disassembly is currently blocked on this box (no `--offloading`, roc-obj-ls broken) — mark SASS claims "artefact-unverified" until tooling works.
- **Hygiene:** `bench-exclusive.sh` is the only file whose failure invalidates everything (md5 e04b59f1…) — never touch it. Shared config lives in `common.py`/`common.sh` (model path, `-fa`, cache types). Interleaved A/B ≥20 pairs; noise floor ±0.15–0.20 t/s.
- **Harnesses in-tree:** `engines/run_diagnostics.py`, `rdna2_occupancy_solver.py`; `gfx1030.py` (9 subcommands), `ic_expose.py --dry-run`. Calibration-FAILED simulators are untrusted.
- **`gfx1030_audit` — merged into `scripts/gfx1030.py`** (audit rules + `trace --steady` + `score` + `report`; standalone deleted). Its priors were corrected when WP-E was falsified (14.7% / ~0.2 t/s realistic). **Graphs-off lesson now a rule: never run gap analysis on a graphs-off trace.**

## 5. ISA annex — gfx1030 mechanism inventory

Per-load/store cache-policy bits (ISA §8.1.10, Tables 38–39) — every load carries them:

| SLC | DLC | Policy | Apply to |
|---|---|---|---|
| 0 | 0 | LRU | KV (when short enough), activations, draft tensors, router |
| 1 | 0 | L2 Stream (evict-early) | weight stream, KV at our contexts — defends the LRU lines |
| 1 | 1 | Hit-No-Allocate | pure one-shot streams |

**SLC status — source-confirmed, artefact-unverified:** `__builtin_nontemporal_load` is on the hot path in source (mmq-load-tiles.cuh:1412/1438/1439, vecdotq.cuh:1311/1336/1338 — the **bq3 dequant path, this model's weight type**). At SASS level it is **unproven**: llvm-objdump here has no `--offloading`, roc-obj-ls is broken, plain disassembly yields host x86 only, and the one .s showing zero slc predates the edit by three days. Do not cite it either way — it is neither confirmed nor refuted at artefact level. The closed `__ldg` lane only proved wrappers are no-ops. Remaining question is the *effect*: Probe D — does stream-marked weights raise token rate.

**Cache hierarchy correction (binding):** the Infinity Cache is a separate **128 MB L3 above a 4 MB L2** — rocminfo reports both, and `GL2C_*` counters see **only the L2**, so IC residency is unmeasurable on this card and every IC claim must rest on timing, never on a hit-rate number. The decode weight stream (11.35 GB) is **81× oversubscribed** against the IC — the ~1.5+ TB/s "hit bandwidth" is an ISA spec, unmeasured here. The 24.3→34.6→59.2% trajectory is an **L2** number — reuse engineering demonstrably works at L2; whether it reaches the IC tier is what the timing probes answer.

**IC as bandwidth multiplier — four mechanisms:**

1. **Token reuse per weight pass** — MTP's measured hit rise; same mechanism for batch decode and expert-grouped tokens.
2. **Temporal co-scheduling** — prefill workgroups sharing a weight tile hit if dispatched adjacent (the 59.2% pp512 hit rate is this already working); grid ordering alone pushes it.
3. **`S_ATC_PROBE` ahead of the consumer** — MoE router top-k predicts which expert pages the next GEMM needs: probe during preceding compute. Exact prefetch, not speculative.
4. **LRU residency for small hot sets** — draft-head/NextN, router, quantizer buffers stay SLC=0 while giant streams go Stream.

At 112k/256k ctx KV never fits — the IC's job is protecting hot sets + token-reuse hits, not holding KV.

**Instruction inventory (verified in the ISA):**

| Mechanism | Use |
|---|---|
| `__builtin_nontemporal_load` → `... slc` | **source-confirmed on hot path; SASS artefact-unverified** |
| `V_DOT4C_I32_I8` (accumulate-in-dest) | **already harvested** — 3,008 in build vs 568 FMA |
| `V_DOT8_I32_I4` / `V_DOT2_F32_F16` | **unused dot forms** — the real remaining multiplier (quant-format gated); dot2 = the WP-A alternative |
| LDS-Direct operand read | under-used LDS→VALU path for tile reads |
| `DS_BPERMUTE_B32`/`DS_SWIZZLE_B32` | ~~unharvested~~ — **refuted: generated 39,381× by shuffle reductions** |
| `V_READLANE`/`V_WRITELANE`/`V_READFIRSTLANE` | VGPR↔SGPR bridge — scalar-side epilogue math (H2 toolbox) |
| `V_CVT_PK*` | packed f32→f16 converts — cheap pre-scaled f16 tiles |
| `DS_GWS_*` | hardware workgroup sync on GDS — pairs with append/consume circuits for expert-token compaction |
| `V_BFE_U32/I32`/`V_ALIGNBIT_B32` | bit-field extract/align for IQ3_S 3-bit dequant — audit whether mmvq uses shift+mask chains |
| `V_PERM_B32`/`__byte_perm` | **already exploited** in `vecdotq.cuh` — verify coverage, don't rediscover |
| `S_CLAUSE` | ~~group metadata loads~~ — **refuted: already emitted 18,873×** |
| `S_ATC_PROBE[_BUFFER]` | scalar-path prefetch (downgraded by filter — see Annex A MT-H3) |
| `S_INST_PREFETCH` | 1–3 I-cache lines — only if SQ counters show ifetch stalls |
| `SHADER_CYCLES` | in-kernel timer |
| `S_SENDMSG`, subvector/shared VGPRs | no compute use — don't re-mine |

**Architecture facts:** WGP = 2 CUs / 4 SIMD32s, 128 KiB LDS shared; WGP mode is the compiler default on this toolchain. VGPRs allocate in 16-Dword wave32 groups. ⚠ **Contested number — do not cite:** `mul_mat_q` was counted at 234 VGPRs (→240 = 8 waves = one block) but the same IQ3_S kernel later measured **225** in a trace table (4–5 occupancy). 225 and 234 sit on opposite sides of a wave boundary after 16-dword rounding — the "2 blocks/CU closed" verdict is **unresolved** until a register count is read from the live artifact (`hipFuncGetAttributes` or a working device disasm). `COMPUTE_PGM_RSRC` holds the WGP-mode bit. Latency hiding = hundreds of in-flight work-items (§1.1). SFU pipe: separate, **negligible in the profile** (refuted as a lane).

**Machine ceilings (post-refutation):** FP32 FMA ~25.8 TFLOPS · INT8 dot4 ~51.7 TOPS · INT4 dot8 ~103 TOPS · prefill wall **~1,915 t/s** (at 31.3%) · decode wall **~44.1 t/s weight-bound** (at 71%) · dispatch floor 3.60 µs.

**Hypotheses (all `Needs validation` — each has a verification path):**

- **H1** — WGP-mode 2-blocks on `mul_mat_q` → **CONTESTED, reopened**: WGP mode is already the compiler default and two 45.9 KiB workgroups fit the 128 KiB WGP LDS — but the VGPR count is contested (234 vs 225, opposite sides of the rounding boundary). Resolve by reading the register count from the live artifact before any occupancy claim.
- **H2** — migrate wave-uniform metadata (scales, block addresses) to `S_LOAD_DWORD*`+SGPRs — frees VGPRs and VALU slots in `mul_mat_vec_q`.
- **H3** — SLC stream-marking on weight loads (see IC section).
- **H4** — SALU epilogue offload runs parallel with VALU — WP-A's slot accounting may double-count.
- **H5** — batch N loads per `S_WAITCNT` in mmvq — decode latency hiding.
- **H6/H7** — `S_CLAUSE` on scale/metadata loads; LDS bank-layout notes for WP-A.
- **H8** — SLC=Stream weights (folded into IC section).
- **H9** — KV = second weight stream at our contexts (folded into header).
- **H10–H12** — ATC_PROBE, clause coalescing, inst-prefetch (folded into table).
- **H13** — **split-K GEMV**: decode matvec is small-M — fan the K-reduction across workgroups → more in-flight misses past the 8-wave ceiling; merge via L2 atomics or a reduce pass. Check the grid shape first; port candidate from stew675/llama-cpp-rdna-boosts.
- **H14** — **widen the decode weight loads**: 5,302× `global_load_dword` (4B) vs 932× `dwordx4` (16B) — the matvec spends issue slots on narrow loads while bandwidth-bound. Wider loads raise bytes/instruction and outstanding-miss bytes in flight — same wall, fewer slots. *This is the surviving gap after the refutation pass.*

## 5a. IC characterization workflow (agent-run, MALL counters are dark — measure by timing)

Goal: prove the IC's usable capacity, hit bandwidth, and whether SLC bits reach SASS — before any production kernel depends on them. Reuse WP-0's read-only float4 stream harness; `hipcc -O3 --offload-arch=gfx1030`; every run under `bench-exclusive.sh`; outputs to CSV.

1. **Capacity knee (Probe A):** read-only stream over working sets 8→256 MiB in ~16 MiB steps, ≥10 passes each, steady-state bandwidth vs size. The knee where BW collapses toward ~500 GB/s = usable IC capacity (expect somewhere ≤128 MiB — hit fraction decides where).
2. **Latency map (Probe B):** pointer-chase (stride ≥ 64 B, pseudo-random order) over the same size sweep, avg cycles/access. Latency plateaus reveal the levels; the jump past the IC edge confirms Probe A's knee.
3. **SLC bit A/B (Probe C):** ~~repeat Probe A with `slc=1` loads and confirm the bit survives the compiler~~ — **half-answered**: the builtin is on the hot path in source; SASS emission is unverified (no working device disassembler on the box). Remaining: the *effect* side — identical curves on re-read data would mean the bit is dead in policy regardless of emission.
4. **Resident-vs-thrash contention (Probe D — the production question):** two streams — a ~64 MiB LRU "hot set" re-read continuously + a ~1 GiB thrashing stream. A/B the thrasher's SLC policy: if Stream-marking protects the hot set, the hot set's bandwidth rises measurably. This is the direct test of the weight-stream policy hypothesis.
5. **Reuse-distance curve (Probe E — the MTP question):** cyclic pass over W bytes re-read k times vs W sweep — maps "how much reuse within how large a set" to hit rate; explains the measured 24.3→34.6% MTP trajectory and predicts what batch-N would buy.

Deliverables: knee point (MiB), IC hit bandwidth (GB/s), SLC-alive verdict + SASS excerpt, contention delta, reuse-distance curve. Cross-check any "hit" inference against `GL2C_MC_RDREQ` drop where measurable.

## 6. External evidence base

- **`stew675/llama-cpp-rdna-boosts`** — closest existing RDNA2-tuned llama.cpp body; diff its mmvq/mmq/fattn before writing new code (if someone already solved multi-row GQA verify or dot-native mmvq on RDNA2, WP-F is a diff-read not an invention).
- **`warpfront/hipfire`** PRs — port concepts, not code: capability admission gating, KV/expert host-offload, split-K, f16 dot2 (+50–67%), MTP enable rules.
- **arxiv 2603.10031** (8×MI325X vLLM study — mined; **`Needs validation`: claims unverified by Stefan** — "active params ≈ decode speed" is a cross-model correlation used here as a design heuristic, not a law):
  - **KV offload to host RAM works for GQA models** (vLLM `--kv-offloading-backend native`, 64 GiB CPU buffer). Our MoE is MoE+GQA → KV offloading is a validated path; MLA models can't, but ours aren't MLA. Independently corroborates hipfire #793/#858.
  - **KV quantization halves footprint — GQA-compatible**: FP8 KV `≈50%` KV memory cut, orthogonal to weight quant. At 256k ctx where KV is tens of GB, llama.cpp's `--cache-type-k/v` quants are the analogous lever.
  - **Active params/token ≈ decode speed** (paper's cross-model law): our MoE A3B activates ~3B/token vs the dense 27B — the measured 90.75 vs 31.1 t/s gap is this law operating; MoE decode is the high-value lane.
  - **Kernel wins concentrate in MoE/expert paths** (AITER ablation: 2–3× on MoE/MLA kernels vs 3–5% + 2–16× variance on dense GQA): confirms WP-B/MoE-path kernel work outranks generic attention tuning.
  - **Chunked-prefill knob**: vLLM always-on chunked prefill — small chunks → better ITL, large → better TTFT. Our analog is `-ub`/`--ubatch-size`; for 256k ctx prefill.
  - **Concurrency saturates at the BW bound** (~500 short / ~100–200 long seqs) — corroborates the KV-stream limiter model.
  - **Fused-MoE correctness gate**: on CDNA3 a fused MoE kernel needed `AITER_ENABLE_VSKIP=0` to prevent `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` — our fused-MoE work needs an explicit correctness gate, not just a speed check.
  - **NUMA hygiene**: `kernel.numa_balancing=0` + pinned host memory — matters when host RAM is a real tier (see expert-prefetch below).
  - **Not transferable**: TP=4/8 configs, 256 GB HBM, FP8 weights path (CDNA hardware), vLLM scheduling internals.
- **r/LocalLLaMA Infinity-Cache thread** (RDNA3 discussion — our card has 128 MB not 96 MB; **`Needs validation`: community-level, unmeasured here**):
  - **The claim that matches our measurements:** past ~960 GB/s (RDNA3's DRAM) toward ~3.5 TB/s only happens when data stays inside the IC — identical to our IC-as-multiplier framing, and our L2 trajectory (24.3→59.2%) is the same mechanism one tier down.
  - **The concrete knob — prefill `ubatch`:** the thread argues prompt-prefill intermediates must fit the IC or the GPU "swaps" to VRAM; suggested `-ub 128` (dense) / `-ub 64` (MoE). Same knob as the arxiv chunked-prefill finding — **cheap to test: A/B `--ubatch-size` against pp512 baseline before any kernel work.** It's a runtime flag, so it can't be the lead answer — but it prices the IC's prefill benefit in one run.
  - **The architectural argument (frames Annex B):** "inner loops in cache are cheap" — a model that refines output over multiple passes on the same weight block gets extra compute almost free relative to the outer VRAM-bound loop. Nobody trains models that way, so the payoff is structural, not a flag — our version is **batched adaptive MTP (Annex B): more verified tokens per weight pass IS the inner-loop-in-cache pattern applied to a model that exists.**
- **Ibrahim et al., "Analyzing and Leveraging Shared L1 Caches in GPUs" (PACT'20)** — AMD research paper; a hardware redesign study (we cannot reconfigure our L1s), so the mechanism doesn't transfer, but **two principles do**:
  - **Replication, not bytes, sets effective capacity.** Private L1s duplicate every shared line; they measured up to 79% reduction in collective miss rate and ~39% avg speedup (≤83%) by slicing the address range so each line is cached once. Our decode weight stream is the extreme replication case — every CU reads the same 11.35 GB — so **the private-L1 tier is a near-zero-capacity cache for decode; all reuse capture must happen at the shared tiers (L2→IC) or in LDS.** Formalizes why decode can't be fixed at L1.
  - **Their sensitivity classifier is our triage tool:** shared-org wins only when (a) replication ratio >10%, (b) L1 miss rate >50%, (c) speedup with 16× capacity >5%; apps tolerating added latency pay for it. Our bandwidth-bound decode is the most latency-tolerant case they describe. **The software analog of their address-slicing: schedule same-tile workgroups co-resident on one WGP (they share the 128 KiB LDS) — IC-4 co-scheduling at the LDS tier**, and the same argument applies to WP-A's LDS layout choices (put shared/replicated data in LDS, keep per-workgroup data in L1/VMEM).
- **Design law:** "prove it, then admit it" — exact gfx1030 capability+shape gating with generic fallback.

## 7. Build & run

- Build: `GPU_TARGETS=gfx1030` HIP build; run flags: `-fa on|off|auto` (flash-attention enable; navigate by symbol, line ref drifts).
- Dispatch path: `ggml_cuda_dp4a`→`__builtin_amdgcn_sdot4` (common.cuh:695); `ggml_cuda_should_use_mmq` (mmq.cu) selects MMQ vs mmvq.
- One GPU process at a time — `bench-exclusive.sh` is mandatory for any measured claim.

## 8. Agent tasking — chief-architect directives

**Standing challenge to every agent:** your job is not to confirm this document — it is to find where it is wrong. Every belief below has already been wrong at least once (460.8 was invented; 37% idle was a graphs-off artifact; "decode is closed" died twice; the prefill ceiling was off ×2; my own acceptance arithmetic disagreed with itself). Bring measurements that break things, not stories that comfort them. A theory without a counter is a rumor. A trace without production flags is a fiction.

**Rules of engagement — binding, not advisory:**

1. **No claim without a number, a ceiling, and a falsifier.** "Should be faster" ships nothing.
2. **Production shape or it doesn't count.** Graphs-on, real model, `-fa`, real context — synthetic stands only when the experiment is declared synthetic (the floor test is).
3. **Assume the previous belief is wrong until measured otherwise.** That is how WP-E collapsed and WP-D got solved — by checking, not trusting.
4. **If your number contradicts this document, the document loses.** Report the contradiction first, loudly — the WP-0 500.6 and the WP-E 14.7% corrections are the campaign's most valuable outputs.
5. **Borrowed numbers are hypotheses, not evidence.** Other forks' gains mark where to look, never what we'll get.

| Agent | Directive — the belief to attack |
|---|---|
| **Performance Analyst** | **Attack the 3.6 µs floor.** Run the ~1,000-trivial-kernel graph chain and tell me whether this card's inter-kernel toll is silicon or self-inflicted. If it's ~1.5 µs, I owe this programme ~3 t/s and every "dispatch floor" statement in this doc is wrong. Then instrument the WP-F verify window: prove the kernel name actually flips VEC→GQA and eff. BW actually recovers toward 353 GB/s — do not report a fix you didn't watch happen in the trace. |
| **Kernel Auditor** | **Attack WP-B and the VEC-fallback claim.** For WP-B: the one-line expert-width gate is a hypothesis built on our arithmetic and *borrowed* gains — falsify it on pp512 2,323 t/s and kill it without ceremony if the number doesn't move. For WP-F: verify the oracle's claim against the working tree (the ~:591 gate lives in the uncommitted files) — confirm VEC actually re-dequantises K/V per Q head before anyone extends the kernel for ncols>1. Then the H13/H14 grid-shape and load-width checks. |
| **Bug Hunter** | **Attack the SLC bet.** The builtin is on the hot path in source — SASS emission itself is unproven on this box's tooling, and even proven emission says nothing about effect. Probe D decides: does Stream-marked weights move token rate? If the curves are identical, the bit is dead in policy and the whole SLC lane closes — say so fast, that saves a lane of work. |
| **Research** | **Attack our novelty assumption.** Diff `stew675/llama-cpp-rdna-boosts` mmvq/mmq/fattn — if someone already solved multi-row GQA verify or dot-native mmvq on RDNA2, we're porting, not inventing, and the WP-F estimate shrinks to a diff-read. Report what they did *worse* too — their mistakes are our data. |

**What a finished directive looks like:** a measured number, the ceiling it was compared against, the falsifier's verdict, and a SASS/trace excerpt proving the machine did what you claim. Anything less goes back unmerged.

## 9. Checkpoint landing guide (Stefan lands; agents never commit)

1. `git status --short` + `git log --oneline -3` — freeze the base at `d327bf1f0`.
2. Group by area: **prequant arena** (quantize.cu — 127.65→24.0) · **preadd fusion** (dispatch 1522.65→1417.65) · **RDNA2 MMQ table** (mmq.cu/h — pp512 599) · **GQA attention** (4.4–5× @ 32k) · **state-safety + MTP plumbing** (llama-context.cpp — 46.40 t/s) · **scripts** (gfx1030.py, ic_expose.py, common.py/sh — **never touch `bench-exclusive.sh`**).
3. One commit, human-written message with the measured deltas: e.g. `gfx1030: prequant arena + preadd fusion + RDNA2 MMQ table + GQA attn + MTP plumbing (decode 31.1, prefill 598.9, MTP n_max2 46.40 t/s)`.
4. Re-verify one decode + one pp512 after landing — provenance changes, behavior doesn't.
5. **Push immediately** — a checkpoint only exists off-box.
6. Then WP-B lands as a small clean diff.

## 10. Closed-lane ledger (do not reopen without new measurements)

- ~~Decode is closed~~ → **reopened by WP-0** (WP-E owns the open lane).
- ~~"dot pipes are idle — kernels dequantise then scalar FMA"~~ — **refuted**: 3,008 `v_dot4c_i32_i8` vs 568 FMAs; the dot path is the build's normal mode.
- ~~"prefill ceiling 957.6 t/s / 62.5% reached"~~ — **refuted**: dp4a multiply = the dot operation; honest ceiling ~1,915 t/s, we're at 31.3%.
- ~~`ds_bpermute` unharvested~~ — already generated 39,381× by shuffle reductions.
- ~~`s_clause` unemitted~~ — already emitted 18,873×.
- SFU-pipe hypothesis — **refuted**: negligible in the profile.
- "decode uses 16-byte weight loads" — **refuted by count**: 5,302 dword vs 932 dwordx4 → H14 is the surviving gap.
- ~~"slc confirmed in SASS"~~ — **corrected to source-confirmed, artefact-unverified** (box tooling can't disassemble device code).
- Copy-stall chase — the 745 ms copy→gap→copy cluster is model-load phase, not the timed pass.
- `__ldg`/`__ldcg` wrappers — no-ops (says nothing about ISA cache bits).
- Q4_0 format change — conditional on WP-0's successor measurements; DOT8 measured −0.71%.
- Launch-count cuts — **the gap problem IS the launch floor**: removing launches is the only thing that touches it (~1 t/s per 278). ~50 foldable dispatches remain → ~0.2 t/s realistic. (Earlier "≈0.1 t/s below noise, ≠ the gap problem" framing was backwards — superseded.)
- 2-blocks/CU on `mul_mat_q` — **contested, reopened**: VGPR count 234 vs 225 unresolved (see §5).
- WP-E "dispatch bubbles" as a lane — **closed**: graphs-on idle is 14.7%, uniform ~3.6 µs launch floor, no boundary pattern; ~0.2 t/s realistic. The launch floor remains as one synthetic-graph experiment (§3).
- ~~"launch-count cuts ≈ 0.1 t/s below noise (≠ the WP-E gap problem)"~~ — that distinction was backwards: the gap problem **is** the launch floor; removing launches is the only thing that touches it.
- `S_SENDMSG`, subvector exec — no compute use.
- ~~MT-H5 "fuse accept chain"~~ — right conclusion, wrong argument (oracle): the 3.6 µs floor is **per graph node**, and accept/resample is **host-side** — not among the 1,404 dispatches. Still don't spend there.
- ~~MT-H7 wider first-step~~ — dead by arithmetic: width-4 at step 1 tops out at 2.0 tokens/pass — can only tie today.
- Acceptance-vs-depth table — **needs re-derivation**: 60→50→43→32% is conditional on survival (draft degrades with position); marginals give 1+0.6+0.3+0.129 = 2.03, not the reported 2.29.
- ~~Tree-verify via mask parameter~~ — it is a **speculator rewrite**, not a mask: one result vector per sequence (speculative.cpp:211), scalar accept (:213), rollback = prefix truncation (:1807), target mask built only from (seq_id, pos) — siblings cannot be excluded.
- ~~MT-H4 split-K verify~~ — **superseded by WP-F**, not merely doubtful: the WP-D mechanism is the VEC fallback's 6× K/V re-read. Revisit only if a residual gap survives WP-F's landing measurement.
- ~~MT-H6 draft/verify stream overlap~~ — **dropped**: chunk k+1's draft depends on chunk k's accepted tokens — true serial dependency; only prefetching overlaps, not execution.
- ~~Annex B batched adaptive MTP as a kernel project~~ — **demoted**: batching is already implemented (speculative.cpp:182-230 n_seq; server drafts all slots :3802-3806, verifies one shared batch :4648, accepts per slot :4808+; per-slot n_max via common_speculative_draft_params :3755/:3784-3794). The remaining work is ~20 lines in `get_n_draft_max()`, not a kernel. AND it doesn't fit at 256k: ~11 GiB KV + 14.95 GiB weights on a 15.98 GiB card → a **≤55k-ctx multi-slot feature**, not the operating point. Measured amortisation: 4 token-rows cost 1.20× one row.
- ~~IC-1 pinned hot experts~~ — **blocked on its own first step**: the routing-frequency histogram does not exist (repo, /tmp/opencode, bench-results checked; `llama_moe_cache` exports hits/misses/fills only). Producing one is a code change, not a profiler flag. And at 256k the expert stream is PCIe anyway (WP-G).
- ~~IC-2 router prefetch at 256k~~ — **aimed at the wrong stream**: experts are host-resident behind PCIe at the operating point; IC policy targets L2/DRAM. Re-evaluate only after WP-G shows what the MoE stream actually is.
- ~~IC-3..IC-6~~ — subsumed by Probe D (IC-3) or prefill-only at the operating point (IC-4 is a knob A/B, IC-5/IC-6 minor).

## 11. Glossary

- **dp4a/`V_DOT4C_I32_I8`** — 4-element int dot instruction (`__builtin_amdgcn_sdot4`); the multiply IS the dot op (the prefill-ceiling correction).
- **MMQ** — tiled GEMM for prefill; **mmvq** — GEMV for single-token decode.
- **`block_q8_1_mmq`** — LDS tile format: 128 int8 + scale metadata; WP-A target.
- **WGP** — work-group processor = 2 CUs / 4 SIMD32s / 128 KiB LDS.
- **SLC/DLC/GLC** — per-load cache-policy bits (L2 pool / L1 / L0).
- **IC** — Infinity Cache, 128 MB L3 *above* the 4 MB L2 (rocminfo shows both; `GL2C_*` counts L2 only — IC residency is unmeasurable, timing-only).
- **MTP** — multi-token prediction (chunked speculative decode); n_max = tokens per pass.

---

*Demoted hypothesis annexes (MTP lanes, batched adaptive MTP, IC hypotheses, capability map) moved to `gfx1030-annexes.md` — reference material, not the issued directive.*
