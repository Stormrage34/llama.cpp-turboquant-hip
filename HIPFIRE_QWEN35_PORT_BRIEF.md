# hipfire → llama.cpp porting brief: Qwen 3.5 family and RDNA2/gfx1030

Evidence base: `warpfront/hipfire` at `56367a75a` (2026-10-24), the five target PRs
(#915, #919, #933, #934, #936, all merged 2026-10-22), read from source with citations.
Every claim below is either quoted from the code or marked as a gap.

---

## 0. Summary — and the one finding that changes the mission

**hipfire's Qwen 3.5 work is not a set of model optimisations. It is a refactor of
hipfire's own dispatch architecture.** #915/#919 move decode onto a typed `Step` tape;
#933/#934/#936 move dense FFN and tensor-parallel execution onto that tape. The
performance claims attached to them are *parity* claims — "the refactor did not change
the numbers" — not "this is faster than what llama.cpp does".

Consequences for this port:

1. **The tape has no llama.cpp counterpart.** llama.cpp builds a `ggml_cgraph` and
   executes it; hipfire executes a hand-rolled `&[Step]` interpreter with a frozen
   `OpKind` discriminant table. Porting the tape would mean replacing ggml's graph
   executor. Omit.
2. **The "Product ladder" is a fusion-table wrapper**, not a new algorithm: norm →
   fused gate/up GEMV → fused SiLU·mul + down GEMV. Our tree already does this shape of
   fusion through `ggml_cuda_can_fuse` and has landed three such changes this session
   (O1, O3, O11′). The portable idea is the *gate table discipline*, not the code.
3. **gfx1030 gets none of it.** `gfx1030` appears in `arch_caps.rs` only as an atom
   (`is_gfx1030`, with `is_rdna2 = is_gfx1030 || is_gfx1031 || is_gfx1032`). **Every**
   Qwen 3.5 decode fusion gate keys on `gfx1100` / `gfx1151` / `gfx1201` (or the
   rdna3/rdna4 molecules). There is no gfx1030 Qwen path upstream to port. This is a
   stated gap, not a deficiency of the search.
4. **Tensor parallel has no llama.cpp substrate.** llama.cpp ships RPC and row-split
   sharding, not TP with all-reduce. The `HeadMixer`/`GatedMlp` `SplitRule` machinery
   maps to nothing we can execute.

What genuinely transfers is smaller, and worth the trip:
**the `ArchCaps` capability-struct discipline**, **the one-predicate-per-fusion gate
table**, **the `GdnTape` rollback rule** (pre-conv1d `qkv`, not post-conv1d), and
**the MTP head as a step sub-program** — which maps to a real feature gap in our tree
(`blk.64.*` tensors are present in the GGUF and never executed).

---

## A. Relevant upstream PRs

### A1. Core Qwen 3.5 execution
| PR | what it does | why it matters here | portability |
|---|---|---|---|
| **#915** | Decode layers as typed hybrid `Step`s. Adds `pipeline/hybrid.rs` (+1533) with `DeltaNetMixerOp` / `GatedAttentionOp` / `SwigluFfnOp`, `kv_attention`, `normed_projection`, `swiglu_down_residual`; extends `Step` and `OpKind 28..30`; adds `qwen35/program.rs` (+244) binding weights/state/KV/scratch; **deletes the gate and handler bodies from `forward.rs`** (+212/−1875). | The fusion-gate table with exact predicates is the only genuinely reusable artefact. | **Adapt** (predicates only) |
| **#919** | MTP layer, DFlash capture and VL mrope onto the step program (HIPF-120). | MTP head is a real feature we do not have; see §C3. | **Adapt** |

`Step` is a tagged enum over borrowed GPU tensors (`steps.rs:40-230`); the interpreter
`launch_op` matches exhaustively with **no catch-all**, so the compiler forces every op
to have an arm. `OpKind` (`step_graph.rs:38`) is a frozen `#[repr(u16)]` discriminant
with a `op_kind_ids_frozen` test; retirements are permanent (`18` DenseFfn and `20`
SandwichMlp are both retired into `GatedMlp`; `31` SwigluFfn retired by #933).

Layer routing is a `match` on `LayerWeights` (`program.rs:120-195`): `DeltaNet |
DeltaNetMoe → Step::DeltaNetMixer`, `FullAttn | FullAttnMoe → Step::GatedAttention`.
The hybrid rule itself (`full_attention_interval`) comes from config, not this crate —
the same metadata our loader already reads for this model.

### A2. TP / mesh / dense FFN
| PR | what it does | why it matters here | portability |
|---|---|---|---|
| **#933** | Dense FFN on `GatedMlp`'s Product ladder; freezes `OpKind` (GatedMlp 28, DeltaNetMixer 29, GatedAttention 30, SwigluFfn deleted); `program.rs` +562. | Product ladder = the fusion shape we already implement. | **Omit code, adopt the gate discipline** |
| **#934** | Dense prefill and decode TP on the generic mesh executor. | Requires an all-reduce substrate we do not have. | **Omit** |
| **#936** | `HeadMixer` split rule; generic dense TP matches the hand route. Adds `qwen35/mesh_tp.rs` (+1517). | Only the divisibility rule is transferable, and only if TP ever arrives. | **Omit** (record the rule) |

`MlpLadder::Product` (`gated_mlp.rs:66-72`): `project()` = `normed_projection`
(RmsnormAutomatic + fused gate/up GEMV via the fusion table); `down()` is either
`Residual` (`swiglu_down_residual` = SiLU·mul+rotate + residual GEMV) or `Write`
(`write_partial` = silu_mul + plain GEMV into the output, the TP partial).
Admission: `rows>=1`, pre_norm Some, Silu, F32 storage, output Write|Residual
(`gated_mlp.rs:155-162`).

`HeadMixer` split rule (`step_graph.rs`, `mesh_split::head_mixer_admission`): either all
weights Replicate (op stays whole), or `out_proj` is `RowShard{axis:1}` and every other
weight is a head split (`HeadSharded`/`FusedQkv`/`ColumnShard{axis:0}`) — **then
`heads % parts == 0` and `kv_heads % parts == 0`**. Attention splits on query/value
heads, DeltaNet on value/key heads, which is why the two differ.
Partial reductions: each mixer/FFN writes a Partial into `mix_out`/`ffn_out`; the
following `Accumulate` becomes a single rank-ordered `AllReduceAdd`.

Parity mechanism: bit-exact byte comparison of `mq4_prefill_state_oracle` dumps
(prefill logits, Q8 KV, GDN state+EF, conv state, every decode step's logits) on
9B/27B-xt/A3B, ≥2 chunk boundaries plus an odd tail. The digests quoted in the PR bodies
(`a786a131…`, `89693fb7…`) are test artefacts and are **not in the source** — treat them
as claims, not evidence we can reproduce.

### A3. MTP / DFlash / speculative
| item | content | portability |
|---|---|---|
| MTP head (`mtp_head.rs:2440-2490`) | A decoder layer expressed as trunk steps `[GatedAttention, GatedMlp(Product)]`, private KV, slot shape from `seq_len_hint-1`, RoPE/KV position from `pos_buf`, flash policy from the trunk. | **Adapt** — maps onto a second graph we could build from `blk.64.*` |
| DFlash capture | Hidden row written to a ring buffer once the layer is final; deferred until after the next layer's fused combine+norm when `moe_combine_next_rms_enabled`. | **Adapt** |
| **GdnTape** (`speculative.rs:2054-2160`) | Captures **pre-conv1d `qkv`** and post-sigmoid `(α,β)` per block position during batched verify; on rollback replays conv1d+QK-norm+repeat-interleave+GDN for `accept_len+1` steps against a pre-verify DN snapshot, advancing both S-state and conv_state. | **Direct design port** — see §C2 |

The `GdnTape` rationale is the single most valuable correctness statement in the set:
*"pre-conv1d qkv instead of post-conv1d (q,k,v): conv_state is a recurrent buffer
advanced by conv1d_silu_split. If we skipped conv1d on replay the next verify would see
a stale conv_state … small numerical drift that empirically halves τ."* Replay requires
multi-layer tables armed at provision; an unarmed tape plus an admitted replay is an
**error, never a silent switch**.

### A4. RDNA2 / consumer-GPU compatibility
| capability | gfx1030 | gfx1100 | gfx1151 | gfx1201 | enforced at |
|---|---|---|---|---|---|
| WMMA | no | yes | yes | yes | `arch_caps.has_wmma` |
| IU4 WMMA | no | yes | yes | yes | `arch_caps.has_iu4_wmma` |
| MQ4 Lloyd mb4 | no | yes | yes | no | `supports_mq4_lloyd_mb4` |
| DFlash f16 residual/projection fusions | no | yes | no | yes | `supports_dflash_f16_*` |
| DFlash hidden scatter fusion | no | yes | yes (copy only) | yes | `supports_dflash_hidden_scatter_fusion` |
| exact VMM route | **no** | yes (singleton/row) | no | yes (batched) | `exact::supports` |
| native FP8 KV / FA2 | no | no | no | yes | `FP8_KV_QUALIFIED` |
| GDN compact2/3 | **no** | yes | yes | yes | `hybrid.rs gdn_compact2/3` |
| fused FA prep/epilogue | **no** | yes | yes (A3B) | yes (A3B) | `hybrid.rs fused_attention_*` |
| gated norm MQ rotate | **no** | yes | yes (A3B) | yes (27B) | `hybrid.rs gated_norm_mq_rotate` |

**gfx1030 has no Qwen 3.5 path upstream.** It is an atom in `ArchCaps`; no hybrid gate
admits it; the rdna2 molecule does not appear in any Qwen3.5 predicate. Nearest analogs:
**gfx1100** (Radiowave campaign) and **gfx1151** (exact certified bundle via
`gfx1151_radiowave`).

MQ-V2 gating: `MQ4G256V2` (qt=44) is not arch-gated as a *format*; `mq4v2_symmetric` is
a weight property set at load (`shared_weights.rs:304`) and required by the gfx1201 IU4
symfold path. Typed-only formats are refused on generic batched routes by
`reject_typed_only_format` (`batched.rs:174-190`): `MQ4G128V2` (qt=53) and MQ-I4 /
E8P-I4 (qt=131-136) error with *"only Qwen4 (Flash-Next) models can run"*.

### A5. Kernel admission
`tables/` exposes a `KernelRegistry` (`register`/`resolve`/`eval`/`eval_arch`/
`variants_for`) over family tables — `attention_table`, `fused_qkv_table`, `gemm_table`,
`gemv_table`, `moe_table`, `rotation_table`. Key = `KernelKey` (dtype × variant × shape
predicate); arch arrives via `DispatchCtx`. Refusal normally **keeps the incumbent
path** (fallback, not panic); `MissingImpl { key }` is the hard error for an unresolved
fused key.

---

## B. Porting map

| upstream concept | closest llama.cpp subsystem | mapping quality |
|---|---|---|
| `Step` / `OpKind` tape | `ggml_cgraph` + `ggml_backend_sched` | **No mapping.** Different execution architecture. |
| Fusion gates (one predicate per gate) | `ggml_cuda_can_fuse` predicate chain | **Good** — same job, ours is a chain of `if`s rather than a table |
| `MlpLadder::Product` | `ggml_cuda_can_fuse` `{MUL_MAT, MUL_MAT, GLU}` + our landed O3 dual-matvec | **Good** — already implemented on our side |
| `SplitRule` / `HeadMixer` / `Accumulate` | *none* (llama.cpp has RPC + row split, no TP all-reduce) | **No substrate** |
| `ArchCaps` | *scattered*: `#if defined(RDNA2)`, `#if defined(GGML_USE_HIP)`, `getenv()` | **Good, and currently a gap on our side** |
| `KernelRegistry` | `get_vec_dot_q_cuda` / `get_vdr_mmvq` / `get_mmq_q8_1` switches | **Partial** — ours are compile-time switches, not a table |
| `GdnTape` | nothing (we have no speculative decode) | **Greenfield, and the design is the deliverable** |
| MTP head as trunk sub-program | nothing (`blk.64.*` loaded, never executed) | **Greenfield feature** |
| `reject_typed_only_format` | our `GGML_ABORT` on a null dispatch arm | **Already aligned** |

---

## C. Implementation recommendations

### C1. Port directly (small, no new kernels)
**(a) `ArchCaps`-style capability struct.** Our tree decides capability by preprocessor
symbol and `getenv` — `#if defined(RDNA2)`, `rdna2_tune_variant()`, the
`GGML_HIPFIRE_Q8_0_MIN_N` cutoff. That is how a gitignored header ended up deciding
which production kernel compiles. A tiny `struct ggml_cuda_arch_caps { bool has_wmma,
has_iu4_wmma, has_dot2_f32_f16, has_sdot4, is_rdna2, is_gfx1030; }` built once at device
init, with every gate reading it, converts untestable compile-time state into a value we
can assert on and log. No perf claim; pure hygiene, and it is the precondition for
anything else being reviewable.

**(b) The refusal discipline** — already done here (the `GGML_ABORT` on the null
dispatch arm, the cost model's exit-4 contract). hipfire's `reject_typed_only_format`
is the same pattern with a better error message; adopt the message style.

### C2. Adapt
**(a) `GdnTape` rollback rule** — if speculative decode is ever attempted, the tape must
capture **pre-conv1d `qkv`**, not `q, k, v`, because `conv_state` is recurrent and a
replay that skips conv1d leaves it stale; and an unarmed tape must **error, not fall
back silently**. This is a correctness constraint we would otherwise discover the hard
way, and it costs nothing to record now.

**(b) MTP head structure** — the GGUF we run carries `blk.64.*` and never executes it.
hipfire expresses the MTP layer as an ordinary decoder layer with a private KV and a
different slot shape. When we build that path, the shape rule (`seq_len_hint-1`) and
the "flash policy from the trunk" rule are the two things to copy.

**(c) Fusion-gate table** — our fusions (O1 state alias, O3 dual matvec, O11′
prequantised activations) each carry an admission predicate scattered across
`ggml-cuda.cu`, `qwen35.cpp` and `mmvq.cu`. Collecting them into one table keyed by
(arch, shape, dtype) is the hipfire pattern that would most improve our maintainability
— but the *conditions* are ours, not theirs, because theirs exclude gfx1030 wholesale.

### C3. Omit, and why
- **The `Step`/`OpKind` tape and the frozen-ID discipline** — a rewrite of ggml's
  executor. The one idea worth stealing is the frozen-discriminant *test*.
- **`Product ladder`, `mesh_tp.rs`, `HeadMixer` split rules, `Accumulate`→`AllReduceAdd`**
  — no TP substrate. Recording `heads % parts == 0 && kv_heads % parts == 0` is
  sufficient; the 1,517-line program is not.
- **All gfx1100/gfx1151/gfx1201 decode fusions** (`gdn_compact2/3`,
  `gated_norm_mq_rotate`, `fused_attention_prep/epilogue`, `conv_scalar_prep`,
  `qkvza_scalar_prep`) — these are arch-gated *away* from gfx1030, so importing them
  would either be dead code or a claim of support no upstream evidence backs. This is
  exactly the failure mode the mission's decision rule warns about.
- **exact-VMM routing** — `exact::supports` admits gfx1100 (singleton) and gfx1201
  (batched) only; gfx1030 is not admitted, and we independently measured VMM as neutral
  here with a hard abort under `-p 512`.
- **MoE / offload / host-memory paths** — no MoE model in scope, and our measurement
  base is a dense 27B.

---

## D. Risks and compatibility notes

1. **The gfx1030 gap is real and unavoidable.** Zero upstream Qwen 3.5 gates admit it.
   Any gfx1030 kernel work derived from hipfire would be *our* engineering informed by
   their gfx1100 shapes, not a port — and should be labelled as such in any write-up.
2. **Their optimisations are shape- and dtype-specific.** Every gate pins exact
   dimensions (`linear_value_dim==128`, `head_dim==256`, `n_rot==64`, `w_qkv.k==2048`,
   16Q/4KV, 24Q/4KV). Our model hits `head_dim 256` and `n_rot 64` but the head counts
   differ; predicates do not transfer by inspection.
3. **MQ4/HFQ containers do not exist in GGUF.** The `gated_norm_mq_rotate` fuse requires
   `is_mq4(wo.dtype)`; that dtype cannot appear in a GGUF. Half the decode fusions are
   therefore unreachable for us regardless of architecture.
4. **Exact-vs-approximate parity.** hipfire's parity is bit-exact digest comparison
   across prefill logits, KV, GDN state and every decode step. Ours has been
   `test-backend-ops` plus eval-callback string equality plus PPL — adequate, but we
   should state which of the three guards each future fusion.
5. **Host-memory / offload assumptions.** `swiglu_down_residual` includes a CPU-offload
   splice path; irrelevant here (weights are fully VRAM-resident) and a source of
   divergence if copied.
6. **MTP/DFlash gating difference.** Native MTP is *explicit-only* upstream
   (`MTP_MEASURED_AR_FLOOR=false`, so `auto` keeps plain AR). Their own default does not
   enable MTP; neither should ours without a measurement.
7. **Their default is "refuse and keep the incumbent".** Ours has been "enable and
   measure". Both are defensible, but a port should not silently invert the default.

---

## E. Proposed patch plan

**P0 — capability struct + gate consolidation (correctness/hygiene, no perf claim).**
Add `ggml_cuda_arch_caps` populated at device init; replace `#if defined(RDNA2)` /
`getenv` decision points with reads from it; log the caps once at startup. Delete the
gitignored tuning header's role in production selection in favour of a CMake option.
*Acceptance:* identical decode t/s within noise; `test-backend-ops -o MUL_MAT` 2/2; the
caps line printed. Effort: 1 day.

**P0b — record the `GdnTape` rule and the MTP shape rules** in the design docs, so a
future speculative-decode patch cannot get them wrong. Effort: 1 hour.

**P1 — MTP execution path (`blk.64.*`).** Build the MTP layer as a second graph reusing
the trunk's attention and GatedMlp steps, private KV, slot shape `seq_len_hint-1`.
Gate: explicit opt-in only, never `auto`; measure at ≤26k where batching fits.
This is the only item on the list that adds a capability we do not have.
Effort: 3–5 days, and it needs the batching work first to be worth anything.

**P2 — fusion-gate table in our tree.** Move the admission predicates for O1/O3/O11′
and the existing `can_fuse` patterns into one table keyed by (arch, shape, dtype), with
a unit test asserting each gate rejects the wrong arch. Effort: 1–2 days.

**P3 — gfx1030 kernel work: explicitly NOT a port.** If pursued, it is new engineering
against the measured facts in `ARCHITECTURE_GFX1030.md` (element-rate model; batching is
the lever; single-stream ceiling 21.5 ms). Do not cite hipfire as evidence for it.

**Do not do:** port the tape; port mesh TP; port arch-gated fusions; port MQ4/HFQ
container logic; enable MTP by default.

---

## Open questions for the fork

1. Does our loader expose `layer_types` / `full_attention_interval` identically to
   hipfire's config read? (Ours reads it — confirm the same semantics for `LinearAttention`
   vs `FullAttention`.)
2. Is `blk.64.*` complete in our GGUF (all MTP tensors present), or partially stripped?
   Determines whether P1 is a port or a conversion problem.
3. Does our `ggml_cuda_can_fuse` already cover the `normed_projection` shape, or is that
   our O11′ (separate, landed)?
4. Which of hipfire's bit-exact parity guards map onto our three existing checks
   (`test-backend-ops`, eval-callback string equality, PPL)? A per-fusion statement is
   needed before P2.
5. For gfx1030 specifically: is there any hipfire evidence at all, or is the honest
   position "none, and our numbers stand alone"? (Current evidence: none.)
