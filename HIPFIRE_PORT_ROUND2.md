# HIPFIRE → llama.cpp, Round 2: Qwen3.5 dense / MoE / TP / RDNA2

Status: uncommitted review artifact. No code changed by this round.
Host: gfx1030, decode 31.414 ms / 31.833 t/s (Vulkan reference 29.06 ms / 34.4 t/s).
Tree: `HEAD = d327bf1f0`, 18 modified / 20 untracked files.
Upstream: `/home/stormrage/.hipfire/src` @ `FETCH_HEAD = 56367a75a` (2026-10-24).

---

## 0. Verdicts

| PR | Exists | Merged | What it actually is | Portable? |
|----|--------|--------|---------------------|-----------|
| #935 | yes | **yes** 10-22 | +31 lines: 3 restored `memcpy` into the GDN tape in `batch_chunk_delta_net_moe` | **No** — bug class not expressible in ggml. Structural immunity. |
| #897 | yes | **yes** 10-22 | arch predicate `gfx1100 \|\| gfx1201` + per-layer dtype/shape gate; gfx1100 runs per-request, unbatched | Predicate: no (we already match the style). The *double-rotate marker* hazard: **yes, as a guard**. |
| #690 | yes | **yes** 09-13 | collapses two drifted copies of one admit rule; **llama path now refuses MQ-V2 on every arch** | **Yes — highest-value artifact.** Rule, not format. |
| #793 | yes | **NO — closed unmerged** | +14,904: host-RAM spill for dense | **No.** Re-implements what llama.cpp already ships (`-ngl`, `-ot`, mmap). |
| #858 | yes | **NO — closed unmerged** | +32,418: MoE expert spill, MTP over a spill | **No.** Same, plus MoE has no path in our model. |
| #915 | yes | yes 10-22 | `Step`/`OpKind` tape replaces hand dispatch | No — ggml builds a `cgraph`. |
| #919 | yes | yes 10-22 | MTP/DFlash/VL-mrope onto the step program | No. **It is a measurement we can already run.** |
| #933 | yes | yes 10-22 | `gated_mlp.rs` +1111, `program.rs` +562, prefill −11,809 | No — fusion wrapper + parity discipline we already practise. |
| #934 | yes | yes 10-22 | `mesh_tp.rs` +1517, `mesh_split.rs` +655 | **Partly — see §1.3. My earlier "no substrate" ruling was wrong.** |
| #936 | yes | yes 10-22 | `head_mixer_admission` / `gated_mlp_admission` split rules | **Partly — the rule maps onto an enum we already have.** |

### Corrections to the round-1 ruling

1. **"Mesh TP has no substrate in llama.cpp (RPC + row-split, no all-reduce primitive)" — FALSIFIED by our own tree.**
   `ggml/src/ggml-backend-meta.cpp` (2547 lines, in `HEAD`, tracked) implements the mesh model directly:
   `enum ggml_backend_meta_split_axis` at `ggml/include/ggml-backend.h:387-400` has
   `AXIS_0..3`, `MIRRORED` (*"all values on all backends"*), `PARTIAL` (*"each backend has a partial sum"*),
   and `ggml_backend_comm_allreduce_tensor` is implemented for CUDA/HIP at
   `ggml/src/ggml-cuda/ggml-cuda.cu:1320`, registered at `:6515`, with a three-step ladder
   `nccl → internal → butterfly` (`:1243`, `:1229`, `:1225`).
   hipfire's `ShardPolicy { Replicate, ColumnShard{axis}, RowShard{axis}, HeadSharded, FusedQkv }`
   maps 1:1 onto `MIRRORED / AXIS_0 / AXIS_1 / (segment layout) / (segment layout)`.
   My earlier ruling described the *old* RPC picture; the meta backend supersedes it.

2. The rest of the round-1 ruling stands and is not re-derived here: `Step`/`OpKind` has no ggml
   counterpart (ggml builds a `cgraph`); the Product ladder is a fusion-wrapper shape;
   `ArchCaps` duplicates `ggml_cuda_device_info`; the eleven Qwen3.5 fusion gates key on
   gfx1100/gfx1151/gfx1201 and admit no gfx1030.

### The gfx1030 question, answered from the predicate

`#897`'s admission, verbatim (`exact.rs`, `supports()`):

```rust
if !gpu.arch_caps.is_gfx1201() && !gpu.arch_caps.is_gfx1100() {
    return Err(format!("exact VMM route: arch {} not admitted (gfx1100/gfx1201 only)", gpu.arch));
}
// then, for every layer:
LayerWeights::DeltaNet(..) | LayerWeights::FullAttn(..) => all projections are DType::MQ4G256V2
_ => false                                   // rejects MoE layers outright
```

**gfx1030 is not a policy exclusion — it is a dtype exclusion.** The route requires *uniform
MQ4G256V2* for every projection. `#690` pins that `MQ4G256V2` is admitted **only** on
gfx1100/1101/1102/1150/1151/1200/1201 and explicitly:

```rust
assert!(!is_batchable_la(DType::MQ4G256V2, "gfx1030"));
assert!(!llama::is_batchable_la(DType::MQ4G256V2, "gfx1201"));   // llama path: no V2 arms at all
```

gfx1030 cannot satisfy the precondition, so the arch check is never the binding constraint.
Every *positive* gfx1030 admission in the entire repo is one family:

```rust
// prefill.rs:4003
let mq3_uniform_with_gfx10_scalar = matches!(dt, DType::MQ3G256)
    && matches!(arch, "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032");
```

→ **Upstream's gfx1030 story is "scalar HFQ3/K-quant prefill, no WMMA."** That is already what
llama.cpp's CUDA backend does on RDNA2 (`mmq-config-rdna2.cuh` + the vec/GEMV kernels).
There is nothing to import; it confirms we are on the same footing, and it tells us not to chase
a WMMA-shaped decode path on this card.

---

## A. Relevant upstream changes, by group

### A.1 Qwen dense execution — **nothing portable**

#915, #919, #933, #934, #936 are one refactor in five slices: a typed `Step` program replaces
hand-written layer bodies. Evidence that this is internal restructuring, not new math:
`#933` deletes **11,809** lines from `prefill.rs` while adding 562 in `program.rs`; `#933`'s own
stated goal is *"generic dense TP matches the hand route"* — a parity claim, i.e. the output is
unchanged. The one substantive routing fact inside #897:

> *"gfx1100's fused projections and FA epilogue have not been replaced by row-batched twins.
> Keep one binding alive for the entire layer: in particular, Attend can mark FA output as already
> rotated for ResidualGemv. Splitting that binding loses the marker and rotates the output twice."*

This is the only *correctness* content in the dense group: a **fusion-boundary double-rotation**
hazard introduced by re-scheduling a fused op into separately-dispatched pieces. It is worth a
guard in our tree (§E, P2) because our graph has the same class of hazard at build time.

### A.2 Qwen MoE execution — one real bug, not ours

`#935`, `+31/-0`, single site `prefill.rs`, fn `batch_chunk_delta_net_moe`:
restores three `memcpy_dtod_at_auto` calls copying `dn_qkv_batch` / `dn_alpha_batch` /
`dn_beta_batch` into `tape.{qkv,alpha,beta}_bufs[delta_layer_idx]` at `tape_offset`.
Regression since `420065b05`; the batched MoE DeltaNet body had stopped writing the rollback tape,
so a partially-accepted draft replayed the **previous window's** q/k/v/α/β into every MoE DeltaNet
layer. Upstream changelog: τ 7.47 → 2.5 on A3B, reply cut off at 15 tokens, *"Dense models were
not affected."*

**Verdict: OMIT for us, record as upstream knowledge.** Two independent reasons:
1. Our model is dense Qwen3.5/3.8 27B, 65 blocks, **no experts** — the body that broke does not exist.
2. The mechanism cannot occur in ggml. See §D.1.

MoE *routing* work named in the mission (expert schedule, host spill, expert activation) appears
only in **#858, which is closed unmerged** (+32,418). There is no merged MoE-execution change in
this PR set to port. **Zero portable MoE content.**

### A.3 TP / mesh / dense-FFN — one real rule, one wrong premise of mine

`#936`'s `head_mixer_admission` states the fused-QKV+out_proj TP rule:

- all `Replicate`, **or** `out_proj` = `RowShard{axis:1}` **and** every other placed weight is a
  head split (`HeadSharded` / `FusedQkv` / `ColumnShard{axis:0}`);
- `heads` and `kv_heads` must divide evenly over `parts`;
- *"Every weight the op reads (including `out_proj`) must be planned, else `UnknownWeight`"* —
  comment: *"A split rank would otherwise run a head projection it holds no slice of."*

`gated_mlp_admission` states the FFN rule: gate/up `ColumnShard{axis:0}`, down `RowShard{axis:1}`,
all over one intermediate extent, all-or-none, `intermediate % (parts * group)` on down,
and at `parts == 1` the output is local (`Whole`), so **no one-device reduction is emitted**.
`SplitError::{UnshardableOp, LayoutMismatch, UnsupportedPolicy, UnknownWeight}` — a typed refusal
per failure mode rather than a boolean.

The "parts == 1 ⇒ no reduction" clause is the same idea as our
`n_rs == 1 && kv_state->get_size() == 1` fast path in `build_rs` (`llama-graph.cpp:3978-3982`):
at one participant the mesh collapses and the shuffle is the identity.

### A.4 MTP / DFlash / speculative — a measurement, not a port

`#919` puts MTP/DFlash/VL-mrope on the step program. We already have all of it:
`llama_model_qwen35::graph_mtp` (`src/models/qwen35.cpp:503`), loader `:96-126`,
`common/speculative.cpp:1489+`, complete 15-tensor GGUF MTP block, gated behind
`--spec-type draft-mtp`. **#919 changes nothing we lack.** See §E, P4 for the measurement.

### A.5 Consumer RDNA / kernel gating — the transferable discipline

`#690` is the only PR in the set whose *bug* is a routing bug, and it is a good one:

`llama::is_batchable_la` and `qwen35::is_batchable_la` were two hand-maintained copies of one rule.
They drifted. V2 was admitted on gfx12 by both, but llama's `forward_prefill_chunk` has **no V2
dispatch arms** — its per-layer dtype matchers (`qkv_is_mq` ~:2570, `wo_is_mq` ~:3025,
`ffn_is_mq` ~:3117, `w_down_is_mq` ~:3248) list only `MQ4G256|MQ6G256|MQ3G256|MFP4G32`. An admitted
V2 model therefore **skipped the FWHT rotate and ran V1 `hfq4g256` launchers on V2 blobs —
silently incoherent prefill.**

The fix has two halves, and only the first is obvious:
1. hoist into one function `mqv2_wmma_batchable(dtype, env, arch)` with a documented kill-switch
   (`HIPFIRE_MQV2_GFX11_WMMA=0` disables gfx11 only; gfx12 unaffected);
2. **change the semantics from "admit where the kernel exists" to "refuse everywhere the caller has
   no arms"** — llama now refuses V2 on *every* arch, *including* gfx12 where the kernels exist;
3. pin it with a lockstep test that asserts both callers **and** the absolute values
   (`mqv2_admit_llama_qwen35_lockstep`), so the test fails on rule regression, not only on drift.

**The transferable rule: a dispatch predicate must be co-located with, and validated against, the
set of kernel arms its caller actually has. Where two callers share one rule, one owns it and a test
pins both.** This is dtype-independent and applies directly to our tree (§E, P1/P3).

Secondary but real: `load.rs:2601 warn_rdna2_unvalidated_dtypes()` — a load-time scan of per-tensor
quant_type bytes that, on RDNA2 only, prints which RDNA3+-only formats are present and that they are
unvalidated, ending *"RDNA2 (gfx1030): uniform .mq4 is the validated SKU."* Refusal expressed as an
observable diagnostic instead of a silent attempt.

### A.6 Offload / spill — **zero content**

#793 and #858 are **closed, unmerged**, from external contributors (16 GB gfx1201 / RX 7900 XT),
+14,904 and +32,418. They are proposals, not upstream-proven code. And their content is a
re-implementation of what llama.cpp ships natively and more maturely:

| hipfire | llama.cpp, already present |
|---|---|
| layer offload | `-ngl` |
| expert offload to host | `--n-cpu-moe` / `-ncmoe` — `common/arg.cpp:2779` |
| per-tensor placement | `-ot` / `--override-tensor` — `common/arg.cpp:2766` |
| "auto must not default to no-offload" (#858 body) | `-ngl auto` heuristic |
| MTP over a spill | n/a — our MTP is dense |

**Zero port content.** Two notes recorded for completeness only: (i) the UX argument in #858 that an
"auto" offload mode must not silently default to *no* offload; (ii) `host_link.rs` /
`slowlink_admission.rs` in qwen4 show a "slow link" admission concept — interesting, out of scope,
and unrelated to a single-GPU host.

---

## B. Port map to llama.cpp

| Upstream idea | Closest C++ location | Fit |
|---|---|---|
| #935: tape write must exist for every layer the rollback can reach | `src/models/delta-net-base.cpp:494-519`; `src/llama-graph.cpp:3887-3928` | **Structural immunity.** Writes are graph nodes emitted by one shared builder. |
| #935/#690: one rule, N copies, drift = silent corruption | `n_rs_seq > 0 ? n_rs_seq + 1 : 0` — **5 hand-written copies**: `llama-memory-hybrid.cpp:89`, `llama-memory-hybrid-idx.cpp:99`, `llama-memory-hybrid-iswa.cpp:88`, `llama-memory-recurrent.cpp:461`, `llama-kv-cache-dsv4.cpp:1417` | **Direct hit. This is our instance of #690's defect.** |
| #897: double-rotate marker lost when a fused op is re-scheduled | build-time in `src/models/qwen35.cpp` (GDN norm / `q_conv`,`k_conv` views); detector: `scripts/test-graph-census.cpp` | Adapt — guard, not runtime state. |
| #897: exact-arch + all-or-nothing dtype gate, `Err` not `bool` | `ggml/src/ggml-cuda/fattn.cu:529-586` (`ggml_cuda_fattn_gqa_gfx1030_supported`) | **Already matches.** ~30 fail-closed predicates, exact arch, shape, dtype. |
| #690: predicate co-located with the arms | `ggml/src/ggml-cuda/mmq.cu:358-487` (`ggml_cuda_should_use_mmq`) | Single source on the host side; **twin on the device side drifts** (§D.4). |
| #936: `HeadSharded` / `ColumnShard` / `RowShard` / `Partial` | `ggml/include/ggml-backend.h:387-422` `ggml_backend_meta_split_axis` + `ggml_backend_meta_split_state` | **1:1 vocabulary exists.** Validation is the missing piece. |
| #936: `parts == 1 ⇒ no reduction` | `llama-graph.cpp:3978-3982` (`direct` fast path) | Same idea already present. |
| #934: all-reduce ladder | `ggml-cuda.cu:1144-1189` internal (TP2, F32/F16/BF16 only), `:1225-1240` init ladder `nccl → internal → butterfly` | **Exists.** Missing: arch/capability admission and a TP4 story. |
| #690: loud load-time refusal | `llama_model_loader` / `llama_model_quantize` type checks; `ggml_cuda_should_use_mmq` default `false` | Adapt the *diagnostic*; llama.cpp already fails closed by dequant/CPU fallback. |
| #793/#858 offload | `-ngl`, `--n-cpu-moe`, `-ot` | Already superior. Refuse. |

---

## C. Implementation plan

### Port directly
Nothing, verbatim. No hipfire code is C++-shaped.

### Adapt
1. **`n_keep_tail` single-source** (from #690's mechanism). Add one helper and use it at all five
   sites. Pure deduplication; zero behaviour change; converts a 5-way silent-drift surface into a
   1-way edit.
2. **Snapshot-invariant guard** (from #935's class). `llama-batch.cpp:718` already asserts
   `n_ubatch > n_keep_tail`; add the missing half — when `n_rs_seq > 0`, the value passed *must*
   be `n_rs_seq + 1`, asserted at the five call sites via the helper, not by hand.
3. **Graph census for double rotation** (from #897's marker). Extend
   `scripts/test-graph-census.cpp` to count rotate/norm-family nodes per `(layer, tensor-role)`.
4. **Split-rule validation** (from #936), *only if* TP becomes a goal: a validator over
   `ggml_backend_meta_split_state` encoding "fused-QKV ⇒ every segment split on the same axis,
   heads divide evenly over parts, all-or-none". Typed refusal, not a bool.

### Drop as Rust-only / not applicable
- `Step` / `OpKind` / `program.rs` / `run_layer_program` — ggml builds a `cgraph`; scheduling is
  `ggml_backend_sched`. No equivalent, no gap.
- `ArchCaps` — `ggml_cuda_device_info` (`devices[].cc`, `smpbo`) already carries it.
- `DispatchCtx`, `Qwen35Bindings`, `GdnTape`, `RowTwins` — hipfire executor internals.
- MQ-V2/MQ4CG256 container and every `mq*` kernel — GGUF cannot carry these formats.
- #793/#858 in full — closed, unmerged, and behind llama.cpp already.
- Infinity Cache and other AMD cache-policy knobs — not implicated by any target PR.

---

## D. Risks and compatibility

### D.1 Why #935 cannot happen here (and what that does *not* buy us)
hipfire's tape is **imperative state written by a hand-written batched body**; the bug is a body
that stopped writing it. In llama.cpp the same state is written by **graph nodes**:

- `build_conv_state` (`src/models/delta-net-base.cpp:446-519`) emits `ggml_cpy` in *both* branches —
  the `n_rs_seq == 0` branch at `:493`, the rollback branch at `:517` inside a `t = 1..=K` loop;
- `build_rs` (`src/llama-graph.cpp:3887-3928`) emits the gather at `:3913`, the main write at
  `:3918`, and the extra-planes copy at `:3922-3925`;
- dense and MoE call the **same** builders: `qwen35.cpp:405/407` vs `qwen35moe.cpp:407/409`.

A layer cannot skip the write, because there is no per-layer body to skip it. **This is a
structural property, not a passing test.** It also means the MoE variant is not a second code path
here the way `batch_chunk_delta_net_moe` was upstream.

Caveat worth stating: immunity applies to *missing writes*, not to *wrong offsets*. The offset
arithmetic at `delta-net-base.cpp:502` (`s_idx = max(0, n_seq_tokens - K + t)`) is only correct
because `split_equal(n_keep_tail = n_rs_seq + 1)` guarantees the trailing `K` tokens share a ubatch
— which is why P1/P2 matter.

### D.2 Cross-file invariant is the real exposure
Five hand-written copies of `n_rs_seq > 0 ? n_rs_seq + 1 : 0` (§B). Passing `0` is *legal and
silent*: `GGML_ASSERT(n_ubatch > n_keep_tail)` at `llama-batch.cpp:718` passes for `n_keep_tail == 0`.
A sixth memory type, or one edited site, yields a split that violates the snapshot invariant →
rollback restores a state that never corresponded to a prefix → **silent recurrent-state corruption
on partial draft acceptance**, i.e. #935's exact user-visible symptom, reached by a different route.

### D.3 Kernel admission — our gate is correct, but wider than the validated set
`ggml_cuda_fattn_gqa_gfx1030_supported` (`fattn.cu:529`) admits on `GGML_CUDA_CC_IS_RDNA2(cc)`,
i.e. gfx1030–gfx1032+ up to the RDNA3 floor, while only gfx1030 has been validated here.
This is **ISA-defensible, not sloppy**: the kernel's only arch-specific primitive is
`__builtin_amdgcn_sdot4` (`fattn-gqa-gfx1030.cu:212`), and `common.cuh:82` records
`GGML_CUDA_CC_RDNA2 … // RX 6000, minimum for dp4a`. So the family check *is* the instruction floor.
Do **not** tighten to `== gfx1030` — that would break gfx1031/1032 users for a claim we believe.
Do make the distinction observable (§E, P3).

### D.4 Host/device predicate twins in MMQ (drift risk, gfx1030 unaffected)
`ggml_cuda_mmq_get_config` exists twice in `mmq.cuh`:
- host, runtime `cc` switch, cm. `:200-250` — terminal `return ggml_cuda_mmq_get_config_rdna2(...)` at `:246`;
- device, compile-time `#ifdef` ladder, `:264-278` — terminal `#else return …_rdna2(...)` at `:277`.

Every HIP arch not in `{GCN, CDNA, RDNA4, RDNA3_5, RDNA3}` silently receives the **RDNA2** config.
For gfx1030 that is correct. But on AMD, `ggml_cuda_highest_compiled_arch(cc) { return arch; }`
(`common.cuh`) and `cc = 0x1000000 + gfx`, so the pre-DP4A guard at `mmq.cu:415` can never fire and
the tail `return (!GGML_CUDA_CC_IS_CDNA(cc)) || …` at `:489` returns **true** for RDNA1
(gfx1010/1013) — which would then compile the dp4a-shaped RDNA2 config on a device without dp4a.
**Hypothesis only, not proven, and out of scope for gfx1030.** Answered by `test-backend-ops -o MUL_MAT`
on a gfx1010/gfx1013 if one is ever available. Not proposed as a patch.

### D.5 TP / mesh risks (unmeasurable on this host)
- `ggml_backend_cuda_comm_allreduce_internal` asserts `n_backends == 2` (`ggml-cuda.cu:1149`) and
  accepts only F32/F16/BF16 (`:1155`), contiguously allocated (`:1174`), 16-byte aligned (`:1180`),
  returning `false` otherwise → graceful fall to butterfly. **TP2 is the proven shape; TP4 has no
  fast path.**
- `ggml_backend_cuda_comm_init_none` sets `try_allreduce = butterfly`, which is a stub returning
  `false` (`:1209-1212`) — the meta backend's generic implementation covers that call. So there is
  no hard failure, but there is also no NVML/RCCL-free fast path beyond TP2.
- On gfx1030 the internal path depends on `cudaMemcpyPeerAsync`-class P2P; RDNA2 has no XGMI, so
  large-BAR behaviour and host-staging cost are unverified. **Do not claim gfx1030 TP works.**

### D.6 MTP / DFlash / speculative interactions
Already live in our tree, already gated. The interactions that matter are already accounted:
`n_rs_seq > 0` disables `build_rs`'s `direct` alias path (`llama-graph.cpp:3980`), which costs the
O1 gather (−0.6 ms) — consistent with the round-1 accounting: ~348 MB Q6_K + ~471 MB snapshots,
break-even α > 0.164, 46.5 t/s at α = 0.7. #935 does not change this; it removes a MoE-only
corruption mode we never had.

---

## E. Prioritized patch order

Verification class: **(a)** correctness-only, **(b)** measurable on this host, **(c)** unmeasurable
here, with the reason. Every patch must clear the standing gate: **bit-identical numerics plus a
paired A/B** — two of our landed fusions passed correctness tests while producing corrupted output.

### P1 — `n_keep_tail` single-source  · (a)  · from #690
- **Target:** new helper (e.g. `llama-batch.h`); replace the expression at
  `llama-memory-hybrid.cpp:89`, `llama-memory-hybrid-idx.cpp:99`,
  `llama-memory-hybrid-iswa.cpp:88`, `llama-memory-recurrent.cpp:461`,
  `llama-kv-cache-dsv4.cpp:1417`.
- **Change:** one `uint32_t llama_keep_tail(uint32_t n_rs_seq)`; comment carries the
  `[TAG_RECURRENT_ROLLBACK_SPLITS]` explanation once.
- **Verify:** bit-identical by construction (same expression, same value). Paired A/B on a 32k
  decode run; diff token hashes over 512 tokens. Grep proof: `grep -rc "n_rs_seq > 0 ? n_rs_seq + 1 : 0" src/` → 1.
- **Refuse-if:** any site turns out to need a *different* value. Then the drift is semantic, and it
  needs a design question, not a dedupe.

### P2 — assert the snapshot invariant  · (a)  · from #935's class
- **Target:** the five call sites + `llama-batch.cpp:718`.
- **Change:** `GGML_ASSERT(keep_tail == 0 || keep_tail == n_rs_seq + 1)` where `n_rs_seq` is the
  recurrent memory's, so "passed 0 by mistake" is impossible; keep the existing
  `n_ubatch > n_keep_tail` assert.
- **Verify:** (i) 1-token prompt, `n_rs_seq = 0` → no assert; (ii) `n_rs_seq = 4`, pad/ubatch regime
  that forces short sequences → no assert; (iii) `--spec-type draft-mtp` at α > 0 → no assert;
  (iv) all of the above bit-identical vs current build.
- **Refuse-if:** any legitimate configuration trips it — that would mean a real invariant gap and the
  patch should become a fix, not a guard.

### P3 — make the gfx1030 GQA admission *observable*  · (a) + (b)  · from #897
- **Target:** `ggml/src/ggml-cuda/fattn.cu:529-586`.
- **Change:** no predicate change. Record in the comment that `IS_RDNA2` is the **ISA floor**
  (`sdot4` / dp4a, `common.cuh:82`) and that gfx1030 is the **validated shape**; emit a
  `GGML_LOG_DEBUG` when the route is selected on RDNA2 ≠ gfx1030, saying shape-unvalidated but
  ISA-compatible.
- **Verify:** (b) dispatch count and kernel sum unchanged — the route must still hit at 32k
  (FLASH_ATTN_EXT 4044/4044 baseline), decode within noise of 31.414 ms. Log line present on a
  simulated non-1030 cc (or by inspection only, if no second card).
- **Refuse-if:** anyone proposes tightening to `== gfx1030`. That breaks gfx1031/1032 without
  evidence of a defect.

### P4 — the #919 measurement (not a patch)  · (b)
- **Target:** existing `--spec-type draft-mtp`; no code.
- **Run:** `scripts/bench-exclusive.sh <llama-bench> -m <27B-mtp.gguf> -p 0 -n 128,512 -r 5`
  with and without `--spec-type draft-mtp`, plus a token-hash comparison for acceptance parity.
- **Decision rule:** adopt only if measured t/s > 31.833 × (1 + margin) **and** α clears 0.164 on
  the real prompt mix; otherwise record as measured-and-rejected.
- **Why:** #919 is not portable; it is a question our tree can already answer. Cost ~819 MB + 0.6 ms.

### P5 — double-rotation census  · (a)  · from #897's marker
- **Target:** `scripts/test-graph-census.cpp`.
- **Change:** count rotate/norm-family nodes per `(layer, tensor-role)`; fail on duplicates.
- **Verify:** deterministic output, diffable between builds. Run against the 27B graph: clean today
  == no live bug, and it becomes a regression guard for any future fusion. Also requires
  `--spec-type draft-mtp` and a prefill graph, since the MTP graph is a separate builder
  (`qwen35.cpp:503`).

### P6 — MMQ host/device predicate parity  · (a)/(c)  · from #690, **optional, low value for us**
- **Target:** `ggml/src/ggml-cuda/mmq.cuh:265-281`.
- **Change:** replace the terminal `#else → rdna2` with explicit arms and `#else #error "unhandled HIP
  arch: add an MMQ config"`. Build-only; no runtime change on gfx1030 (it still takes the RDNA2 arm).
- **Verify:** (a) compiles for gfx1030; `test-backend-ops -o MUL_MAT` bit-identical; deliberately
  omitting an arch now fails the build where it previously passed silently.
- **(c):** the thing this would actually fix (RDNA1, §D.4) cannot be measured — no gfx1010/1013 on
  this host. This is an upstream-shaped change to shared code; keep it out of scope unless the fork
  needs RDNA1.

### P7 — split-rule validation  · (c)  · from #936
- Only if TP becomes a goal. No TP hardware on this host; every claim would be unverified, and
  §D.5 says TP4 has no fast path. **Do not start this before P1–P5 are closed.**

**Not in the order, deliberately:** anything MoE (#935 as code, MoE routing, expert spill),
anything offload (#793/#858), and any widening of gfx1030 kernel admission.

---

## Open questions (each with the artifact that answers it)

1. **Does our dense 27B actually take the gfx1030 GQA route on every decode layer, or only at long
   context?** → `scripts/gfx1030.py` dispatch census; expect `FLASH_ATTN_EXT` 4044/4044 at 32k.
2. **Does any legitimate config pass `n_keep_tail == 0` with `n_rs_seq > 0`?** → instrument the five
   sites, run the full `tests/` matrix plus `--spec-type draft-mtp`; answers whether P2 is a guard
   or a bug fix.
3. **Is MTP worth it on gfx1030 at realistic α?** → P4's paired A/B; break-even α = 0.164 is already
   computed, the measurement supplies α.
4. **Would 2-rows-per-block GEMV help our decode?** → **Upstream already answered no for RDNA2:**
   `gemv.rs` pins `use_wide = … && !(is_rdna2() || is_rdna3_dgpu())` at three sites (11889, 13167,
   13974), comment *"RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel."* Treat as a
   negative result; only revisit with `scripts/.isa_baseline.json` showing headroom in VGPR/LDS.
5. **Does the meta backend's internal all-reduce work on 2× gfx1030 without RCCL?** →
   `GGML_CUDA_CC_IS_RDNA2` in `ggml-cuda.cu:1144-1189`; needs a second card, otherwise unanswerable.
   Also check whether `ggml_cuda_ar_pipeline_init` rejects `n != 2`, which determines whether the
   `GGML_ASSERT(n_backends == 2)` at `:1149` is reachable or dead.
6. **Is §D.4 (RDNA1 → RDNA2 MMQ config) real?** → `test-backend-ops -o MUL_MAT` on gfx1010/gfx1013.
   No such device here; explicitly unanswered.

---

## What I refuse

- **Porting the Step/program executor.** It is hipfire's scheduler, and ggml's `cgraph` +
  `ggml_backend_sched` already cover the responsibility. Porting it would be a rewrite of our
  dispatch layer to change nothing observable.
- **Treating #793 / #858 as upstream evidence.** Both are **closed unmerged**. Citing a 32k-line
  unmerged proposal as "proven upstream" is exactly the error this brief exists to prevent.
- **Any MoE patch.** No expert path exists in a 65-block dense 27B. MoE items here are upstream
  correctness knowledge, not wins for this workload.
- **Widening gfx1030 kernel admission on the strength of gfx1100/gfx1201 results.** The predicate is
  per-kernel and per-shape; §0 shows gfx1030 is excluded from every WMMA/V2 route by *dtype*, and the
  one place it is admitted upstream (`mq3_uniform_with_gfx10_scalar`, HFQ3 scalar) is a route
  llama.cpp already has.
- **Tightening `IS_RDNA2` → `== gfx1030`.** ISA floor is defensible (§D.3); tightening would be a
  guess dressed as caution.
- **Writing patches against this brief without the bit-identical + paired-A/B gate.** Two of our
  landed fusions passed correctness tests while corrupting output. The gate is the deliverable.
- **Touching `llama-server` in any way** while measuring, and **more than one `llama-bench` at a
  time**. All measurements via `scripts/bench-exclusive.sh`.
