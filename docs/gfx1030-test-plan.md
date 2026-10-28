# gfx1030 — queued test plan T0–T7

Status: **queued, not started.** Runs strictly after the lanes currently holding
`build-hip-gfx1030` / the GPU have finished. One GPU process at a time, always
through `scripts/bench-exclusive.sh`.

Every test states: what it decides, the command, the baseline it is judged
against, and the result that makes it fail. A clean negative is a success.

---

## T0 — Dispatch-floor probe (no model, minutes)

**Decides:** whether the 3.6 µs per-launch toll is hardware or ours. This is the
largest single number in the directive — 14.7 % of a token.

**Method:** throwaway HIP C++ under `/tmp/opencode/floorprobe/` (the pattern that
produced the 500.6 GB/s DRAM figure), two arms in one binary:

1. a `hipGraph` holding ~1,000 trivial kernels (one float add each), replayed;
2. the same 1,000 kernels launched directly, no graph.

Report µs per launch for both. Python/`torch.cuda.CUDAGraph` is **not** an
option here: `torch` is not installed and Python is 3.14 (see the install note
at the bottom).

**Pass/fail:**
- ≈3.6 µs both arms → the floor is hardware. The launch lane closes permanently
  and no further dispatch-removal work is funded.
- ≈1.5 µs in-graph → ~2 µs per launch is ours, ≈2.8 ms/token ≈ **+3 t/s**. The
  gap becomes findable and worth a lane.

---

## T1 — MoE operating point (WP-G, ~15 min)

**Decides:** which `-ncmoe` the declared 256k operating point actually needs.
Every MoE number in the directive was taken at `-ncmoe 0`, which cannot hold the
7,168 MiB KV cache plus a 14.95 GiB weight set on a 15.98 GiB card.

**Method:** MoE model, `-c 262144`, KV `q8_0`/`q5_0`, sweep `-ncmoe {0, 8, 16, 24}`;
record VRAM peak per arm (`rocm-smi --showmeminfo vram`) and both pp512 and tg16.

**Baselines to beat / match:** pp512 **2,323 t/s**, tg16 **90.75 t/s**, both at
`-ncmoe 0` (also the falsifier anchor for T5).

**Pass/fail:** if the operating point requires host-resident experts, the byte
stream is PCIe, not DRAM/L2/Infinity-Cache — and every MoE cache claim in the
annexes is re-scoped to "GPU-resident layers only". If experts all fit, the
PCIe concern evaporates and no re-scoping is needed.

---

## T2 — WP-F, re-aimed (geometry gate FAILED, 2026-10-28)

**Geometry gate result — this changes the target.** The custom kernel is compiled
with `GQA_GFX1030_GROUP 6` (`fattn-gqa-gfx1030.cu:23`), a compile-time constant that
sizes `sQq[GROUP][D]` and fixes the block→head mapping
(`sequence = blockIdx.z / (ne02 / GROUP)`, `head0 = blockIdx.z*GROUP - sequence*ne02`).
Admission requires `Q->ne[2] == 6*K->ne[2]` (`fattn.cu:596`).

| model | head_count | head_count_kv | ratio | head_dim | admitted? |
|---|---:|---:|---:|---:|---|
| dense `Swift-1.5-Qwen3.8-27B` | 24 | 4 | **6** | 256 | yes |
| MoE `Qwen3.8-35B-A3B` | 16 | 2 | **8** | 256 | **no** |

MTP runs **only on the MoE** (dense is capped at 112k with MTP off). So WP-F as
originally scoped — multi-row verify at ratio 6 — would improve the one
configuration the owner has switched off. On the MoE, **plain decode also falls
back to VEC**, which reads and dequantises K/V once per Q head: 8× the KV traffic
at every context, worst at 256k.

**Corrected target, in this order:**
1. **Templatise `GROUP` and instantiate 6 and 8** (both ratios pass the same
   admission checks otherwise). This is the change that reaches the declared
   operating point; it removes a hardcoded assumption rather than adding a feature.
2. **Then** multi-row `Q->ne[1]` 2–4 for verify passes.

**Method:** verbose run to capture the route-selection line (which attention kernel
fired), plus `n_max 2` throughput, plus a kernel trace apportioning attention time
at depth.

**Pass/fail:** at 32k+ on the MoE, the route line names the gfx1030 GQA kernel in
**both** plain decode and the verify window. Then multi-row: verify-window kernel
is the custom one and decode beats the contemporaneous n_max 2 baseline by more
than the paired noise floor. A gain without the KV request drop
(`GL2C_MC_RDREQ_sum`) is not this fix.

**Do not inherit 46.40 t/s as the baseline** — it records no model, context,
`-ncmoe` or command, and `llama-bench` has no speculative options at all. Re-measure
the base arm in the same session.

---

## T3 — H14 wide loads (audit + one build)

**Decides:** the only surviving mechanical gap. The decode matvec emits
**5,302 `global_load_dword`** against **932 `dwordx4`** — 4-byte loads on a
bandwidth-bound kernel.

**Method:** audit the weight-load call sites in `vecdotq.cuh` / `mmvq.cu`; decide
whether the narrow pattern is a missed vectorisation or inherent to the IQ3_S
block layout. If fixable, widen the weight side only (never KV or activations).

**Pass/fail:** bit-identical (`MUL_MAT` 2445/2445, `FLASH_ATTN_EXT` 4046/4046,
eval-callback zero diff, perplexity unchanged) **and** a paired-A/B decode gain
above the ±0.40 t/s floor. If the layout forbids it, close H14 with the reason.

---

## T4 — SLC decision (depends on the A/B in flight, minutes)

**Decides:** whether a flat weight-stream result refutes the cache hypothesis or
merely measured an unmodified kernel.

**Method:** if the A/B is flat, prove whether `slc` reached SASS. The `.s` files
on disk predate the edit; `roc-obj-ls` is broken on this box and `llvm-objdump`
has no `--offloading` flag, so this needs a `-save-temps` rebuild of **one**
translation unit in `build-hip-metrics` (not `build-hip-gfx1030`). Count `slc`
modifiers with `v_dot4c` as the control.

**Pass/fail:** zero `slc` in a fresh `.s` → the change never reached the kernel,
the A/B is void, re-run after fixing. `slc` present and flat → the hypothesis is
genuinely refuted; close it.

---

## T5 — WP-B expert-width MMQ gate

**Decides:** the borrowed +15.3 % (ub128) / +42.4 % (ub64) claim, on our model.

**Method:** add the RDNA2 arm to the expert-width condition (mirroring the
RDNA3/4 arm; the RDNA2 path currently falls through to the generic return), then
measure MoE pp512 at **both** `-ncmoe 0` and T1's operating setting.

**Pass/fail:** falsified if MoE pp512 does not move at `-ub 128` in either
setting. Gains that do not survive the move to the operating point are not gains.

---

## T6 — MTP `n_max` sweep at the real operating point

**Decides:** whether `n_max 2` is still optimal once KV dominates. At long
context a verify pass amortises far more than at short context, so the optimum
may move up.

**Method:** `n_max {1,2,3,4}` at 32k / 112k / 256k (dense at ≤112k with MTP off,
per the confirmed operating points; MoE at 256k with MTP on).

**Pass/fail:** if 2 remains optimal at 256k, the MTP configuration is settled and
documented. If a higher `n_max` wins at long context, the operating recommendation
changes and must be re-measured against KV residency.

---

## T7 — WP-A pooled activation scales (deferred, needs funding)

**Decides:** the one large prefill lever. ~768 epilogue slots per tile collapse to
~192; estimated **+20 %**, high cost, shared-CUDA blast radius, quantiser becomes
approximate.

**Falsifier:** pp512 **< 650 t/s**.

Not started without explicit approval; it is last in the build order for a reason.

---

## Ordering rule

`T0 → T1 → T4 → T3 → T5 → T6`, with T2 verified whenever its lane reports and
T7 held. Rationale: T0 and T1 are minutes-to-quarter-hour and each can invalidate
a headline claim; T4 bounds whether the cache work measured anything; T3 is the
last mechanical decode lever; T5 and T6 both need T1's operating point settled
first.

---

## Note — `torch` is not needed for any of this

The annex's floor test uses `torch.cuda.CUDAGraph`; `torch` is absent and is not
required, since T0 is written directly against HIP. If it is wanted anyway:

- Verified present: `torch 2.12.0+rocm7.2` and `2.11.0+rocm7.2` wheels for
  **cp314** on `https://download.pytorch.org/whl/rocm7.2` — so Python 3.14 and
  ROCm 7.2.4 are both covered.
- System pip is blocked by PEP 668 (`/usr/lib/python3.14/EXTERNALLY-MANAGED`), so
  it must go in a virtual environment:

```sh
python3 -m venv ~/venvs/torch-rocm
~/venvs/torch-rocm/bin/pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
```

- **Unverified:** whether the PyTorch index alone satisfies the pure-Python
  dependencies (`sympy`, `networkx`, `filelock`, `jinja2`, `fsspec`). A dry-run
  resolve did not finish inside a 2-minute timeout. If it fails, add
  `--extra-index-url https://pypi.org/simple`.
- Expect ~3 GB. gfx1030 is not a first-class wheel target on every release; if
  `torch.cuda.is_available()` returns False, that is the likely reason, and
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` is the usual workaround.

---

# Delivery queue (one GPU, one build)

Physical constraint: **exactly one GPU process at a time, exactly one build at a
time**, and a build's CPU load corrupts any timing run in flight. The queue is
therefore strictly sequential; each item is dispatched only when the previous one
has released the card. `scripts/bench-exclusive.sh` enforces this and exits 4 on
a concurrent bench — it must never be bypassed.

| # | item | needs | releases on | gate |
|---|---|---|---|---|
| 1 | **T0** dispatch floor | GPU only | — | 1,000-node graph reports 1,000 nodes; floor ≈3.6 µs = hardware, ≈1.5 µs = ~2 µs ours (~3 t/s) |
| 2 | **T1** MoE operating point | GPU only | T0 | `-c 262144`; `-ncmoe {0,8,16,24}`; VRAM peak + pp512 + tg16 all recorded |
| 3 | **T2/WP-F** GQA ratio-8 first, then multi-row | build + GPU | T1 | MoE route line must name the custom kernel in plain decode *and* verify; dense (ratio 6) already admitted |
| 4 | **T3** H14 wide loads | build + GPU | T2 | bit-identical + paired-A/B gain above ±0.40 t/s, else revert |
| 5 | **T5** WP-B expert-width gate | build + GPU | T3 | MoE pp512 must move at `-ub 128`, measured at both `-ncmoe 0` and T1's setting |
| 6 | **T6** MTP `n_max` sweep | GPU only | T5 | `n_max {1,2,3,4}` at 32k/112k/256k; n_max 2 is the incumbent to beat |
| 7 | **T7** WP-A pooled scales | build + GPU | funded | deferred; falsifier pp512 < 650 t/s |

**T4 is closed and will not be scheduled** — SLC weight-stream marking reached
SASS (315 `slc` device-wide), raised L2 hit 24.53 % → 55.67 %, and lost 3.86 % on
decode and ~2–3 % on prefill, 5/5 interleaved pairs. Reverted byte-exact,
library `d15211baa69e705e`. Cache policy is closed in both directions.

## Standing corrections every lane must be given

1. **Baselines drift.** Today's default-library figures are MoE pp512 **2,396.2**
   and dense pp512 **592.53** — not the 2,323 / 598.9 in the directive. Every
   falsifier uses a contemporaneous interleaved baseline.
2. **Never run gap analysis on a graphs-off trace.** Production is graphs-on;
   the 37 % idle figure came from a graphs-off capture and is falsified. The real
   figure is 14.7 % as a uniform 3.60 µs dispatch floor over 1,404 dispatches.
3. **Dead counters on this part:** `SQ_INSTS_VALU` (undercounts),
   `SQ_WAIT_ANY` and `SQC_LDS_IDX_ACTIVE` (read zero), `FETCH_SIZE` (aborts,
   error 38), all `MALL`/`TCC_*` (the Infinity Cache is a 128 MB L3 with no
   counters; `GL2C_*` is the 4 MB L2). Working: `GL2C_HIT_sum`,
   `GL2C_MISS_sum`, `GL2C_MC_RDREQ_sum` (bytes = requests × 64 B), `GRBM_*_BUSY`.
4. **20 interleaved pairs** resolve ±0.15–0.20 t/s; five pairs only ±0.40.
   Anything below the floor is reported as unmeasurable, never as a win.
5. **One lane per GPU.** The WP-F lane stalled because it was dispatched while
   another lane held both the card and `build-hip-gfx1030`; its harness was
   refused by the guard on every attempt and no kernel code was ever written.
