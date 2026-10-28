# MASTER PLAN — gfx1030 (RX 6900 XT) HIP decode performance

Status: living document, uncommitted. Owner: kernel/perf track.
Model of record: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (qwen35, 866 tensors, 11.277 GiB, 3.55 bpw).
Reference stack: HIP `build-hip-gfx1030` vs Vulkan `build/`. Measurement: `llama-bench` only.

---

## 1. Scope and hard constraints

- **gfx1030 only.** LDS 64 KB/CU, 1024 VGPR/SIMD, wave32, `v_dot4_i32_i8` + `v_dot2_f32_f16` both present (only gfx10 subset with sdot4). No RDNA1/RDNA3 inference.
- **`llama-bench` and `test-backend-ops` only.** No `llama-server`, no curl/health, no port 9931, no `pkill` of the production server.
- **Nothing committed.** All experiments stay in the working tree, env-gated and default-off.
- **One attempt per idea.** A null result closes the idea; no parameter variants afterwards.
- **Report ratios and ablation deltas, never absolute microarchitectural claims.** This host has no rocprof; DRAM row/bank, stall reasons and queue occupancy are unmeasurable by construction.

---

## 2. Corrected accounting (supersedes every earlier figure)

Per-token weight stream is **11.353 GB**, not 12.11 GB:

| excluded | size | why |
|---|---:|---|
| `token_embd.weight` (IQ2_S) | 0.379 GiB | `get_rows` reads one 5120-wide row per token |
| `blk.64.*` (MTP block) | 0.325 GiB | not executed in the main graph |

12.109 GB − 0.407 − 0.349 = **11.353 GB/token**. Also: `output.weight` is Q4_K at 682 MiB = **6.0 % of the token as a single lm_head matvec**.

| measurement | HIP | Vulkan | ratio |
|---|---:|---:|---:|
| decode d0 | 31.00–31.19 t/s (32.26 ms) | 34.24–34.41 t/s (29.06 ms) | 0.90 |
| whole-token bandwidth | **352 GB/s** (69 % of peak) | **391 GB/s** (76 %) | 0.90 |
| matvec-only bandwidth | **373 raw / 386 corrected** | **410–433** (by subtraction) | 0.89–0.94 |
| prefill pp512 | 609–625 t/s | 487–508 t/s | **1.23 (HIP wins)** |
| depth: d0 / 32k / 112k | 3.19 / 2.63 / 2.08 ms ΔT | — | gap narrows with depth |

Two independent error corrections to keep straight:
1. Every figure built on 377/417 GB/s was ~6 % optimistic.
2. The matvec gate must compare **matvec to matvec**. Comparing HIP's matvec-only rate to Vulkan's whole-token rate (391) understated Vulkan by 5–10 %.

**The depth data cannot separate "per-token-fixed cost" from "per-byte deficit + HIP attention advantage".** Both fit the three points exactly (3.19 / 2.63 / 2.08 ms). The sign of the pairwise slopes is the only evidence and it is weak. Treat as an unresolved identification problem, not a premise.

---

## 3. Closed branches (do not reopen)

| branch | killing measurement |
|---|---|
| ALU / instruction mix in `vec_dot_iq3_s_q8_1` | full unpack ablation, identical memory traffic: **+0.38 %** e2e (inside sd) |
| Weight load width / request count | Q4_K VDR=8: **−34 % isolated**, **0.0 % e2e**; IQ3_S VDR=4: **−3.2 %** e2e at equal occupancy |
| Occupancy / block geometry | 7-config sweep **28.13–30.80** t/s, all worse |
| "16 waves is gold" VGPR cliff | wave arithmetic does not reconcile with the register file (512/R ⇒ 8 or 7 waves/SIMD, not 16/12); retire `MMVQ_LB_MIN` as a tuning axis |
| VMM | **+0.04 %**, and `-p 512` aborts in `cuMemSetAccess` (`ggml-cuda.cu:705`) |
| Weight alignment | row-boundary sector waste **≤ 2.072 %** upper bound; Q4_K **0.000 %** |
| hipGraphs for prefill | **0.0 %** (HIP/Vulkan both; already active) |
| "Backends run different programs" | HIP and Vulkan both **3991 nodes**, identical op histogram |
| Port Vulkan's IQ matvec structure | kernels are isomorphic: VK = 32-thread WG, `NUM_ROWS=4`, K split 4×14, grid staged in LDS, 4× activation bytes and 4× multiplies vs HIP |
| Requantization | 4-bit-ward +18.1 % bytes and VRAM-infeasible; 3-bit-ward lossy-on-lossy (no BF16/FP16 source on disk) |
| Isolated-kernel µs as an e2e predictor | the Q4_K result is the standing counter-example |
| Surrogate / ISA gate as a decision instrument | 9 rows, R²≈0. Keep as a static regression guard only |

---

## 4. Open branches (the whole remaining budget)

**A. Matvec memory-path efficiency — 6–11 % of matvec, ≈4–7 % e2e.**
HIP 386 vs Vulkan 410–433 GB/s. Not ALU, not width, not geometry. The only untested mechanism is **load scheduling / queue depth**: how many outstanding cache lines the kernel keeps in flight. The four prior failures all attacked bytes-per-instruction instead.

**B. Non-matvec node mass — ~3.3 ms across ~560 small nodes.**
Corrected from E1: `RMS_NORM` 305 nodes ≈ 1.75 ms, `UNARY` 160 ≈ 0.74 ms, `GET_ROWS` 98 ≈ 0.84 ms. This is the largest single identified target and it is **not** the GDN block that was predicted. A 2× backend advantage here is worth 1–2 ms (3–6 %) and is cheaper to obtain than any matvec change.

**C. Long-context KV — 0 at d0, ~+1 t/s at 112k.**
K row per (layer, head) = 272 B = 5 sectors = **85 % sector efficiency**. `-ctk q5_0` → 176 B = 3 sectors (91.7 %), KV 28.0 → 22.5 KiB/token (−19.6 %), 2.99 → 2.40 GiB at 112k. Needs a perplexity check. **Long-context item only — must not absorb d0 budget.**

---

## 5. Execution order

### Step −1 — Bisect the 3 % regression (BLOCKING, ~1 h)
With `HIP_NODE_TIMING` unset, decode currently reads **30.10–30.45** t/s against the 31.00–31.19 baseline. Either the new outer graph-capture gate changed the production path, or it is thermal drift. Every comparison below is against a possibly-shifted floor, so this comes first: `git stash` the instrumentation, rebuild, re-measure, restore. Record the true floor.

### Step 0 — DRAM-regime per-op A/B (0.5 d, mandatory)
Add to `tests/test-backend-ops.cpp:12702` (local, uncommitted):
`test_mul_mat(type_a, F32, 4096, 1, k)` for `type_a ∈ {IQ3_S, IQ4_XS, IQ2_S, IQ3_XXS, Q4_K, F16}`, `k ∈ {14336, 131072}` (WSS 179–285 MB for IQ types; `k=65536` → 537 MB for F16). Run `perf -o MUL_MAT -b ROCm0|Vulkan0 --min-time 2000`, 3 repeats, interleaved. Plus the 30-minute bookkeeping: sum `ggml_nbytes` of every tensor consumed by `MUL_MAT` — it must equal 11.353 GB.

**Control that decides everything: F16.** If HIP F16 is ≥5 % below Vulkan F16, the deficit is generic HIP matvec codegen and the whole in-kernel program stops immediately.

Pre-registered:
- IC control reproduces the per-type table within 5 %.
- HIP IQ3_S DRAM **380–400** (central 390); Vulkan **405–430** (central 420); ratio **1.05–1.10**.
- All four IQ types converge to a **≤7 % band** on each backend (in cache they span 2.57×). If they do not converge, the two-regime model is wrong — re-derive.
- No-decode ablation at `k=131072` within 2 % of stock.

Decision: **stop** if HIP/VK ≥ 0.95. **One attempt** if 0.89–0.95. If HIP ≥ 1.0, the gap is entirely non-matvec → go to Step 3.

### Step 1 — Geometry check (2 h, zero code)
Confirm the old 7-config sweep ever contained `(nwarps=1, rows_per_block=4)` and `(2,4)` for IQ3_S at `ncols=1, !small_k` — Vulkan's exact geometry, expressible in HIP only through the tune header. If it was never tested, test it before writing any kernel. Prediction: 385–400 GB/s DRAM, e2e null.

### Step 2 — Load pipeline (1 d, only if Step 0/1 pass)
~20 lines: software-pipeline the stock `kbx` loop in `mmvq.cu` — prefetch the next superblock's ~14 B/lane (8 B `qs`, `qh`, `signs`, `d`/scale) into registers before decoding the current one. Keep lane mapping, dp4a, 1-warp geometry, ≤64 VGPR.

Acceptance: **≥405 GB/s DRAM matvec (+5 %)** *and* **≥31.8 t/s e2e (+2.5 %)**, ≥5 repeats, lower bound above the true Step −1 floor, reproduced in a second process. Kill at <+1.5 % e2e or >64 VGPR / 12 waves.

### Step 3 — Non-matvec decomposition (0.5 d, parallel to Step 1)
Measure `RMS_NORM`, `UNARY`, `GET_ROWS`, `FLASH_ATTN_EXT` at batch 1 on both backends. If Vulkan is ≥2× better on any of them, that is 1–2 ms (3–6 %) and beats any remaining matvec work. Candidates in order of measured size: RMS_NORM (1.75 ms), GET_ROWS (0.84 ms), UNARY (0.74 ms).

### Step 4 — KV at long context (only after Steps 0–3 resolve)
`-ctk q5_0`, perplexity check, then measure at 32k and 112k. Expected +1 t/s at 112k, 0 at d0.

---

## 6. Budget and stop conditions

- **Cap: 2 engineering days** after Step −1 (Step 0 0.5 d, Step 1 0.25 d, Step 2 1 d, Step 3 0.5 d).
- **≤ 4 build-measure cycles**, then stop regardless of state.
- **STOP immediately** if: HIP/VK DRAM ratio ≥ 0.95; or the F16 control reproduces the gap; or Step 1/2 is null.
- No re-tuning of `nwarps`, no VMM/alignment/graph work, no second kernel variant — all closed above.

---

## 7. If the program stops: the deployment decision

| workload | choice | why |
|---|---|---|
| generation-heavy | **Vulkan for decode**, HIP for prefill | +10.4 % decode at d0, +~6 % at 112k, at the cost of −19 % prefill |
| prefill-heavy | HIP for both | +23 % prefill, −10 % decode |
| single-backend requirement | accept ~3 % residual, spend remaining time on Step 3 only if it shows ≥2× | — |

This is a deployment choice, not a patch.

---

## 8. Closing document requirements

The result is only defensible with all seven:

1. Corrected accounting with evidence (11.353 GB, the two exclusions, lm_head 6.0 %, 352/391 whole-token, 386/410–433 matvec-only, 69/76 % of peak).
2. **A location, not a bound** — the per-op-class table with Σmatvec, Σidle, `C_max`, per-class GB/s.
3. A falsification table: mechanism → killing measurement → number. Narrative is not admissible ("we tried occupancy" is not a result; "7 configs, 28.13–30.80 vs 31.00" is).
4. Both adversarial fits of the depth data (3.19 / 2.63 / 2.08 ms), stated as unresolved.
5. A scope statement of what was not measurable, bounded by construction.
6. Reproduction block: exact commands, env, `-r 8`, model path and size, build id.
7. Tree hygiene: `vec_dot_iq3_s_q8_1_ablate` and its dispatch arm deleted; `GGML_ABORT` on the null dispatch arm (`mmvq.cu`, `get_vec_dot_q_cuda` default).

---

## 9. Known measurement hazards

- **Infinity Cache illusion.** Working sets under 128 MB are cache-resident; a kernel can be 34 % faster there and 0 % faster in situ. Every isolated number must be paired with a DRAM-regime number at k ≥ 131072.
- **Event overhead.** Per-node `hipEvent` pairs cost ~3.5 µs; with 3991 nodes that is ~14 ms, inflating wall by 44 %. The "idle 6.257 ms" reading is CPU launch latency created by forcing graphs off, not GPU bubbles.
- **Forced-graphs-off changes the thing being measured.** Idle must be inferred from node counts and fusion deltas, not from a graphs-off run.
- **Reporting only ratios across backends.** Absolute GB/s from a cache-resident harness is not a bandwidth claim.
- **Thermal/clock drift.** Vulkan drifted 33.3 → 34.4 within one session. Always interleave, always record which run was first.

---

## 10. Outcome (measured, supersedes sections 5–7)

The programme is closed. Sections 5–7 describe the plan as written before the profiler existed; this section records what actually happened.

### 10.1 The instrument

`rocprofv3` (v1.1.0, ROCm 7.2.4) works and was the decisive tool. Kernel tracing **requires `GGML_CUDA_DISABLE_GRAPHS=1`** — hipGraph nodes are not HSA dispatch packets, so a graph-captured run traces nothing but model load. Traced wall is unusable; per-kernel durations and dispatch order are sound.

**Divide per-token counts by 17, not 16.** llama-bench runs a 1-token warmup generation (`tools/llama-bench/llama-bench.cpp:2434`) before the measured tokens. Every earlier per-token figure in this document is 6.25 % high.

### 10.2 The measured split (corrected, ÷17)

```
wall 32.26 ms = matvec 25.39 + small ops 4.585 + non-kernel ~2.3
matvec rate 446.9 GB/s = 87 % of the card's 512 GB/s nominal (92-95 % of any realistic read stream)
1,865 dispatches/token; ~1,376 of them small ops averaging 2.5-5.8 us
```

**The weight stream is not the problem.** It was never the problem.

### 10.3 Dispatch economics (calibrated, not assumed)

A tiny op costs **~3 µs**; deleting a dispatch saves **~1.2 µs of gap plus its kernel time**. Proven by the bf16 merge experiment: removing 48 bf16 dispatches saved 0.230 ms of kernel time, adding 96 `ggml_cont` dispatches cost ~0.29 ms, netting a measured **−0.063 ms** — the model predicted −0.06 ms.

Consequence: **any transformation that adds a view, copy, cont or concat node must save ≥3 µs per node added, or it loses.** This is why every "clever" fusion attempted here came back negative or flat.

### 10.4 Branches opened and killed after the profiler

| branch | verdict | killing measurement |
|---|---|---|
| A — matvec memory path | **closed** | 446.9 GB/s in situ vs Vulkan's 391 whole-token; isolated DRAM ratios VK/HIP: IQ3_S 1.036, IQ2_S 1.082, IQ3_XXS 1.115, F16 1.018, IQ4_XS 0.962, Q4_K 0.908 |
| attention chunk bound | **dead** | predicted 11 chunks where 1 needed; measured 36.25 vs 35.72 µs/call, e2e slower in all 3 pairs; `grid.y` clamp alone +2.58 % at `-d 32768` |
| bf16 `ssm_alpha/beta` merge | **dead** | dispatches 96 → 48 and kernel 0.489 → 0.259 ms, but the `ggml_cont` views it required cost more than it saved (−0.063 ms); retry without conts is blocked — this tree has no load-time tensor-concat infrastructure and the GGUF stores the two tensors separately |
| quantize dedupe (377 calls) | **skipped as unsafe** | needs aliasing-correct invalidation of a cached activation; not expressible in <40 defensible lines, and a stale cache produces silently wrong logits that throughput cannot detect |
| quantize inside the matvec | **withdrawn** | the q8_1 block scale needs a per-32 reduction; per-row on-the-fly quantization multiplies work rather than removing it |
| `get_rows` (98/token) | **irreducible** | per-layer recurrent-state gathers for scattered sequence slots; only 2 final output crops are skippable |
| `copy` (83/token) | **irreducible** | with default `n_rs_seq=0` the GDN cache fusion is inapplicable by construction (`keep=false`, K=1, no snapshot tail); these are conv-state updates and SSM-state stores |
| narrow `rms_norm` (176 calls, 0.51 ms) | **untried** | the only remaining item with a measured number; requires a fused kernel inside the GDN path |

### 10.5 Final standing

| depth | HIP | Vulkan | delta |
|---|---:|---:|---:|
| d0 | 31.15 t/s | 34.4 | −10 % |
| 32 k | 28.10 | 30.34 | −7.4 % |
| 112 k | 22.16 | 23.23 | −4.6 % |
| prefill 512 | 625 t/s | 507 | **+23 %** |

The decode deficit is per-dispatch overhead in the ROCm path (1,865 dispatches, single stream, fully serialized), not bandwidth, not ALU, not occupancy, not alignment, not VMM, and not graph capture. Vulkan's advantage is bounded at **~1 ms of cheaper small-op submission**; the rest of the gap remains unattributed and is not addressable with the instruments on this host, because RADV is not an HSA client and its per-op cost cannot be traced.

### 10.6 If the program is revived

Do not start from a kernel hypothesis. Start from `scripts/gfx1030.py trace`, and re-derive the ÷17 numbers before anything else. The tooling is one CLI (`probe`, `build`, `bench`, `trace`, `isa`, `audit`) with the exclusive-access guard enforced by `scripts/bench-exclusive.sh`.

**Deployment:** HIP for prefill (+23 %), Vulkan for decode (−10 % at d0, −4.6 % at 112 k). At long context the two are close enough that a single-backend deployment on HIP is defensible.
