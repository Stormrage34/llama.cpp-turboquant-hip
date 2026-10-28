# Plan: recover 3.06 ms on gfx1030 decode

Target: 32.26 ms → 29.20 ms (31.15 → 34.25 t/s). Vulkan measures 29.06 ms / 34.4 t/s,
so 3.06 ms is **parity**, not a win. Beating Vulkan needs 3.20+ ms.

Uncommitted planning document. Nothing here is committed or pushed.

---

## 1. Where 3.06 ms can possibly come from

Measured split (÷17, trace duration sum 30.12 ms, wall 32.26 ms):

| component | ms | share | addressable? |
|---|---:|---:|---|
| matvec (quant + bf16) | 25.39 | 78.7 % | **≤1.5**, theoretical only |
| small ops (1,376 dispatches) | 4.585 | 14.2 % | **≤1.5** by structural fusion |
| non-kernel gap (1,865 dispatches) | 2.14 | 6.6 % | **0.58** if 480 dispatches go |
| **total theoretical slack** | **~3.2** | | |

Kernel-time slack in the matvec is real but thin: every large shape already runs at
88–94 % of the best measured rate (ffn_down/attn_output 442 GB/s, qkv 438, lm_head 475
vs 446.9 overall). Closing that 1.5 ms requires every matvec to hit lm_head's rate,
which no geometry experiment has achieved.

**The arithmetic only closes if the small-op dispatches are removed in bulk.** A change
that deletes one dispatch saves ~1.2 µs of gap plus its kernel time; deleting 480 of
them is 0.58 ms of gap alone. Subset-fusion programmes have historically captured
~0.2 ms each because they add nodes (a `cont` costs 3 µs) and the accounting nets out.

---

## 2. Why an incremental programme does not reach 3.06

Oracle's opportunity map, nominal vs expected value (probability-weighted):

| id | change | nominal ms | P | EV ms |
|---|---|---:|---:|---:|
| O1 | eliminate 48 SSM-state `GET_ROWS` (identity gather at `n_rs_seq=0`) | 0.49 | 55 % | 0.27 |
| O3 | bf16 alpha/beta dual-output matvec (no `cont`) | 0.29 | 70 % | 0.20 |
| O2 | fuse 144 narrow norms into GDN | 0.58 | 40 % | 0.23 |
| O11 | 129 wide norms write `q8_1` as a second output | 0.37 | 30 % | 0.11 |
| O6 | fuse 64 ffn norms into gate/up `q8_1` pass | 0.30 | 35 % | 0.11 |
| O4 | FFN gate/up dual matvec + fused GLU | 0.65 | 35 % | 0.23 |
| O5 | fuse conv-state into `SSM_CONV` | 0.56 | 30 % | 0.17 |
| O8 | fuse 32 `rope_multi` into attention Q/K load | 0.17 | 25 % | 0.04 |
| O9 | multi-stream overlap of independent branches | 0.42 | 25 % | 0.11 |
| O10 | persistent/fused GDN layer chain (~480 dispatches) | 1.85 | 50 % | 0.93 |
| | **sum** | **5.68** | | **2.40** |

The nominal column exceeds 3.06; the expected-value column does not reach it. Overlap
between O2/O6/O11/O5 and the shared node populations makes the nominal column doubly
optimistic. **Stated honestly: this plan reaches 3.06 ms only if four of the five
phases land in their upper half.** The median outcome is ~1.9–2.2 ms, i.e. 31.8–32.2 t/s.

Two items could change those odds and are therefore front-loaded as measurement, not
as implementation: the Vulkan per-node trace (is the gap actually small-op submission?)
and the two-stream synthetic test (is single-stream serialization a hardware property
or a backend choice?). If the second comes back "backend choice", O9 is worth several
times its current estimate and the plan's ceiling moves.

---

## 3. Phases

### Phase 0 — Decide before building (1 day, no kernel code)

**0a. Vulkan per-node trace.** `GGML_VK_PERF_LOGGER=1` is an in-tree per-node logger
(`ggml-vulkan.cpp:5271`, `vkCmdWriteTimestamp` at `:14743-14818`, aggregate `:16564`).
Zero build. Run HIP and Vulkan at d0 with pinned sha256 of both shared libraries,
≥5 interleaved repeats, produce a per-op-class VK/HIP delta table.
**Gate:** the table must account for ≥94 % of the 3.19 ms (residual ≤0.2 ms).
Fail → the gap is not attributable with our instruments; stop and ship the split.

**0b. Two-stream synthetic test.** One HIP graph, two independent small kernels, run
as one stream vs two streams vs Vulkan. Decides whether the 1,865-dispatch
serialization is a hardware property.
**Gate:** if two streams beat one by ≥0.3 ms on the synthetic, O9 is promoted to
Phase 2 and its estimate is revised upward before any code is written.

**0c. Re-pin the baseline.** Record `libggml-hip.so.0` sha256, ROCm version, sclk, and
re-measure d0 with 5 interleaved pairs. Every later phase is judged against this number.

### Phase 1 — Highest-confidence pair (1.5 days)

**O1 — eliminate 48 SSM-state `GET_ROWS`.** Fuse the `build_rs` gather into
`GATED_DELTA_NET` (`llama-graph.cpp:3887-3927`, `delta-net-base.cpp:460/552`; precedent
`ggml_cuda_try_gdn_cache_fusion`, `ggml-cuda.cu:2947`). The state *write* is already
fused; this is the read half. Numerically neutral at `n_rs_seq == 0`.
**O3 — bf16 alpha/beta dual-output matvec.** One kernel, two weight pointers, two
outputs. Must not `ggml_cont` — the 96 conts are exactly why the previous attempt lost
(+0.063 ms measured).
**Gate:** paired A/B, ≥5 repeats, combined **≥0.40 ms** (≥31.5 t/s). Below → O1's model
is wrong; stop and re-measure rather than proceed.

### Phase 2 — Bulk small-op removal (3 days)

This is the phase that has to carry the plan. Target: remove ~300 dispatches without
adding a single node.

**2a. Producer-epilogue pass.** The existing fusion machinery proves the right
mechanism is absorbing a producer into its consumer, never inserting a view. Systematically:

- 144 narrow norms (96 q/k prologue + 48 output epilogue) absorbed into the GDN kernel
- 129 wide norms emit `q8_1` as a second output, feeding the following matvec directly
- 64 ffn norms absorbed into the gate/up `q8_1` pass

**Gate:** ≥0.70 ms combined (≥31.9 t/s) and zero new `cont`/`view` nodes in the census
diff. The census `diff` is the guard: if ΔN_dispatch ≥ 0, the design is wrong regardless
of the wall-clock result.

### Phase 3 — FFN dual matvec (2 days)

**O4.** One kernel, gate/up two weights, GLU in the epilogue, shared `q8_1`. Upper bound
0.65 ms; the per-launch figure is inferred, not measured, so treat it as a ceiling.
**Gate:** ≥0.35 ms.

### Phase 4 — Remaining dispatches (2 days)

O8 rope fuse (0.17) + O5 conv-state (0.56) + O9 multi-stream if Phase 0b licensed it.
**Gate:** ≥0.35 ms combined.

### Phase 5 — Structural, only if Phases 1–4 fall short (5–10 days)

**O10 — fused/persistent GDN layer chain.** ~10 small ops × 48 layers absorbed into one
kernel. This is the only single change that removes ~480 dispatches *and* 1.0–1.5 ms of
kernel time, i.e. the only one whose arithmetic can still reach 3.06 if the incremental
phases underdeliver.

**Entry condition:** total recovered < 2.5 ms after Phase 4. **Exit:** ≥1.2 ms or abandon.

---

## 4. Measurement protocol (every phase)

- `scripts/bench-exclusive.sh` for every run; one `llama-bench` at a time.
- Paired A/B via `scripts/gfx1030.py bench --ab BASE.so FIX.so`, ≥5 interleaved pairs,
  `cmp` verifying the restored library before the series is trusted.
- MDE at d0 is **0.30 ms / 0.30 t/s**; no verdict below that, regardless of sign.
- Record `libggml-hip.so.0` sha256, sclk, and run order with every result.
- Kernel-level rocprof evidence (`scripts/gfx1030.py trace --ledger`) accompanies every
  end-to-end claim, because isolated-kernel wins have repeatedly not transferred.
- `scripts/gfx1030.py predict` may be used as a **sanity check only** until the cost
  model meets its five evidence criteria; it currently inverts the O1/O3 ranking.

---

## 5. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| Node-adding requirement defeats a fusion (the bf16 precedent) | high for any `cont`-based design | census `diff` must show ΔN_dispatch < 0 before building |
| Isolated win, flat end-to-end (Q4_K precedent: −34 % isolated, 0.0 % e2e) | high | every gate is end-to-end t/s, never kernel µs |
| Vulkan trace fails to attribute ≥94 % | medium | Phase 0 gate; if it fails, the programme stops and the split ships |
| Cost model misleads phase ordering | high until fixed | explicitly downgraded to sanity check |
| Thermal drift across a 10-day programme | certain | re-pin baseline at each phase entry |

---

## 6. Stop conditions

Stop and ship the split (Vulkan decode / HIP prefill) if any of:

1. Phase 0a attributes <94 % of the gap.
2. Phase 1 recovers <0.40 ms — the two highest-confidence items failing means the
   dispatch model is wrong and every later estimate is unfounded.
3. Phases 1–4 total <1.5 ms and Phase 5's entry condition is not met.
4. Any phase consumes its build-cycle cap without a paired-A/B verdict above MDE.

---

## 7. Honest summary

- **3.06 ms requires phases 1–4 to land in their upper half, or Phase 5 to succeed.**
- Median expectation: **1.9–2.2 ms → 31.8–32.2 t/s**, i.e. still behind Vulkan at d0
  but ahead at 112 k (where HIP is already only 4.6 % behind) and ahead on prefill (+23 %).
- The plan is worth running because Phases 1–2 are cheap, independently valuable in
  every configuration, and would land HIP near Vulkan parity at long context.
- It is not worth running if Phase 0a fails: without a per-class attribution, every
  subsequent phase is a guess, and this project has already spent three weeks on guesses.

---

## Results log

### O1 — eliminate 48 SSM-state `GET_ROWS` (LANDED)

Fuse `build_rs` so a single-cell, single-sequence batch aliases the cache row instead of
materialising a gather. Gate: `allow_direct && n_seq_tokens == 1 && n_seqs == 1 &&
n_rs_seq == 0 && n_rs == 1 && get_size() == 1`. Three files, no kernel changes.

| metric | predicted | measured |
|---|---|---|
| Δdispatches | −96 | **−96** |
| Δnodes | −240 (wrong arithmetic) | **−288** (6/layer: GET_ROWS, zero-size CPY, RESHAPE, 3×VIEW) |
| recoverable time | 0.43–0.68 ms | **0.712 ms** |
| decode | ≥31.5 t/s | **31.604 t/s** (30.909 → 31.604, +2.25 %) |

Numerics: bit-identical. `llama-eval-callback -ub 1` (16 sequential one-token decodes,
covering fresh cell then steady state) gives identical `name||sum` streams over 4,864
lines; default `-ub` fallback path identical over 304 lines; CPU PPL identical
(4.4122 ± 0.64159 both). `test-backend-ops -o MUL_MAT` 2445/2445, 2/2 backends.

The direct path is visible in the dump: `state_predelta-0` becomes
`RESHAPE(cache_s_l0{786432,1,1,1})` instead of `RESHAPE((view){786432,1,1,1})`, the
freshly cleared cell reads 0.0 in both, and the steady-state read is bit-identical.

**Methodology correction:** the A/B must swap `libllama.so.0.7.0`, not
`libggml-hip.so.0` — the latter is byte-identical before and after (the change is graph
construction, not a kernel). Swapping the HIP library would have measured a no-op.

Not exercised: `n_seqs > 1` (explicitly gated off), other `build_rs` callers
(RWKV/Mamba-style, `allow_direct` defaults false), and `delta-net-base.cpp:460`
(conv-state path, deliberately out of scope).

### O3 — dual-output bf16 alpha/beta matvec (LANDED)

One launch computes both weights via an extra grid row (`grid.y = 2`, second row carries
`channel_dst = 0` and swaps `x`/`dst`). Node count unchanged.

| metric | O1 | O1+O3 | delta |
|---|---|---|---|
| `mul_mat_vec_f` launches / token | 96 | **48** | −48 |
| `mul_mat_vec_f` kernel sum | 0.4526 ms | **0.2839 ms** | −0.169 ms |
| total dispatches / token | 1835.8 | 1787.8 | −48 |
| census nodes / real | 3513 / 2201 | **3513 / 2201** | **0 / 0** |
| decode (5 pairs) | 31.607 t/s | **31.833 t/s** | +0.226 t/s, −0.225 ms |

Numerics bit-identical: eval-callback `-ub 1` diff 0 lines over 662,640; `-ub 512 -n 16`
diff 0 lines over 92,920; PPL 4.4122 both; `llama-cli --temp 0 -s 42` identical token
stream. `test-backend-ops -o MUL_MAT` 2445/2445.

Two things the design did not anticipate, both load-bearing:

1. **Node order is a post-order DFS over `src[]`, not source order.** Reordering the
   `qwen35.cpp` calls changed nothing (alpha at 33, beta at 39 in both builds). Skipping
   the 6-node range would have skipped `reshape/add/softplus/mul/reshape`, i.e. `gate`
   itself. Replaced with a mark-and-drop executor path: the fused launch marks both
   outputs and the executor consumes each mark when it reaches that node.
2. **`ggml-alloc` gave both outputs the same buffer** (`dst == dst2 == 0x…9000` for every
   layer) because alpha's lifetime ends before beta is allocated. That silently corrupted
   output (cli divergence; eval-callback happened to match). Fixed with an explicit
   output-vs-output overlap test plus `ggml_set_output(alpha_mm)`, which keeps the buffer
   out of the free list (cost 48 × 512 B ≈ 24 KB).

Also: `libggml-hip` is **not** byte-identical for this change (kernel + matcher live
there) — both libraries must be swapped in the A/B.

FFN swiglu fusion untouched: `has_fusion=true` mmvq launches 2567 → 2567.

## Re-trace after O1+O3 (31.414 ms / 31.833 t/s)

÷17 basis, 30,393 trace rows, join coverage 100 %, 0 unmatched. Token-boundary ledger
(driver blits removed, 16 fully delimited tokens): **1,764.0 dispatches / 29.172 ms**.
The 23.8/token difference is driver blits plus a leading fragment.

| class | disp/token | ms/token | µs/call | % kernel |
|---|---:|---:|---:|---:|
| matvec | 425.0 | 24.785 | 58.32 | 84.6 % |
| norm | 305.0 | 0.964 | 3.16 | 3.3 % |
| ssm (conv/gdn/fwht) | 160.0 | 0.783 | 4.89 | 2.7 % |
| attention | 16.0 | 0.737 | 46.06 | 2.5 % |
| quantize_q8_1 | 377.0 | 0.667 | 1.77 | 2.3 % |
| activation | 200.0 | 0.386 | 1.93 | 1.3 % |
| copy | 64.0 | 0.195 | 3.04 | 0.7 % |
| elementwise | 60.3 | 0.188 | 3.12 | 0.6 % |
| get_rows | 50.0 | 0.149 | 2.98 | 0.5 % |
| set_rows | 32.0 | 0.138 | 4.31 | 0.5 % |
| rope | 32.0 | 0.123 | 3.84 | 0.4 % |
| concat | 48.0 | 0.101 | 2.10 | 0.3 % |
| driver blit | 18.5 | 0.075 | 4.03 | 0.3 % |
| **total** | **1,787.8** | **29.289** | — | 100 % |

Gap = wall − Σkernel = **2.125 ms/token ≈ 1.19 µs/dispatch**. Measured bare-kernel floor
on this host is **1.67 µs**, so "dispatch-bound" means ≤ 2.5 µs/call.

Addressable small-op mass: **1,298.6 dispatches / 3.195 ms** (µs/call ≤ 10), dominated by
`quantize_q8_1` 377 @ 1.77 µs, `rms_norm<32>` 144 @ 2.57 µs, `rms_norm<1024>` 129 @
3.85 µs, `unary_gated_op<silu>` 88 @ 1.82 µs, `concat_cont` 48 @ 2.10 µs,
`ssm_conv` 48 @ 2.25 µs, `unary_gated_op<softplus>` 48 @ 2.34 µs, `unary_op<sigmoid>`
48 @ 1.67 µs. The earlier "1,021 ms rms_norm" figure was wrong on two counts: norm is
**0.964 ms/token**, and the class is 305 dispatches.

### Matvec batch sweep (n = 1/2/4/8, HIP, `--test-file` shapes)

All 13 shapes gave n8/n1 throughput ratio **0.25–0.39** → no launch/tail amortisation,
i.e. adding work per launch never improves throughput. **But 12 of 13 are Infinity-Cache
resident (19–47 MB)** and therefore not predictive of decode: `iq4_xs 5120×17408` runs
33.5 µs isolated against 98.5 µs in situ (2.9×), `iq4_xs 17408×5120` 37.0 vs 112.2 (3.0×).

Only `q4_K 248320×5120` (715 MB weights) is DRAM-resident, and its isolated 1393.6 µs
reproduces the in-situ 1378.8 µs to 1 % — **513 GB/s, 100 % of peak**. The lm_head is done.

In-situ per-shape rates over 10.414 GB/token of real weight traffic: all-matvec
**420.6 GB/s**; `iq4_xs 5120×17408` 480, `iq3_s 17408×5120` 385, `iq3_xxs 17408×5120`
**322 (63 % of peak, worst)**. The 446.9 GB/s quoted earlier divided the 11.353 GB file
constant, which includes tensors the matvecs never read.

### O11′ — wide `rms_norm` emits its `q8_1` as a second output (LANDED)

Mechanism: the 5120-wide norm computes the `q8_1` blocks for its output row in a second
phase and writes them to a context-owned arena outside `ggml-alloc` (deliberately — a
graph tensor re-enters the allocator's free list and re-creates the O3 aliasing corruption).
`mmvq` looks the blocks up and skips its own `quantize_q8_1`.

| gate | result |
|---|---|
| byte test (`memcmp` vs the CUDA quantizer, no tolerance) | **PASS**, and fails when the lane mapping is rotated by one element |
| census Δ | **0 nodes** (3513/2201 both) |
| `quantize_q8_1` calls/token | 377 → **128** (−249, −0.436 ms) |
| `rms_norm` calls/token | 305 → 305 ✓ |
| `rms_norm` k=5120 µs/call | 3.84 → **6.21 (+2.37)** ✗ |
| paired A/B (10 interleaved) | **median −0.353 ms**, 10/10 negative, stdev 0.214 |
| PPL / greedy stream | identical / byte-identical |

**Only ~47 % of the theoretical −0.65 ms survives.** The +2.37 µs is concentration, not a
code defect: the standalone quantizer spreads 160 block-quantisations over ~20 CUs,
while phase 2 must do it inside a single block (the row has to be normalised by one block
to keep the f32 output bit-identical), i.e. 5 sequential rounds on 32 warps of one CU.
Three fixes were tried and reverted — 4-way load prefetch, mask-free `__shfl_xor`
(935 → 489 instructions, verified in SASS), and dropping the fence. All neutral. The cost
is the intrinsic single-CU throughput of the reduction pattern.

Ceiling was **249, not 273**: the census counts 273 quantised consumers, but the baseline
trace only had 249 `k=5120` quantises for 376 mmvq — the other 24 already share one
quantise via an existing matvec fusion. All 249 that exist were eliminated.

Cumulative: **32.26 → 30.97 ms/token (31.15 → ~32.3 t/s)**, Vulkan 29.06 ms / 34.4 t/s.
Remaining gap ~1.9 ms.

---

## MTP speculative decode — measured, +53.5 % (bigger than everything else combined)

The MTP head was already implemented; this was a measurement, not a port. It runs:
`creating MTP draft context`, 37 tensors of block 64 loaded including 8 `blk.64.nextn.*`,
ROCm buffer 10827 → 11160 MiB, driver reports `n_drafted = 146` for 128 tokens.

| mode | t/s | ms/token | α | RS buffer |
|---|---:|---:|---:|---:|
| plain AR | 30.57 | 32.72 | — | 149.6 MiB |
| MTP, `n_max=1` | 44.4 | 22.52 | 0.817 | 299.3 MiB |
| **MTP, `n_max=2`** | **50.9** | **19.65** | 0.661 | 448.9 MiB |
| MTP, `n_max=3` (default) | 46.9 | 21.31 | 0.562 | 598.5 MiB |

Break-even was α > 0.164; measured α 0.497–0.773, so 3× margin at the worst prompt.
`n_max=2` beats the default by 8 % **and** uses 150 MiB less — the third draft costs a
full MTP decode to buy ~0.37 tokens.

**O1 regression measured, not assumed: +1.33 ms/token**, and it is already inside the
21.31 ms figure. Any speculative type sets `n_rs_seq > 0`, which breaks O1's
`n_rs_seq == 0` precondition. Gross gain if the fast path were kept would be ≈ −12.7 ms.

### The clamp I specified was wrong
I briefed `need_n_rs_seq()` be clamped to the MTP head count (one head → one draft).
**False premise.** qwen35's driver chains the *same* head across positions
(`common/speculative.cpp:1767+`); only `chain_heads` is clamped to the layer count
(`:1584`). Log proof: `call_count = 1, gen = 3`, candidates at pos 0/1/2. After partial
acceptance the rollback distance is `n_draft − n_accepted ∈ {1,2,3}`, and `seq_rm` refuses
any rollback `> n_rs_seq` (`llama-memory-recurrent.cpp:214`). Applied → decode dies on
round one with an inconsistent-sequence-position error. **Reverted.** The only correct
memory lever is lowering `n_max`, which is also faster here.

### Token-exactness is unobservable at `-fa 1`
At the production flags, MTP diverged from plain AR on 2 of 5 prompts (index 39 and 38).
Isolated cause: **flash attention alone flips greedy tokens** — plain AR with `-fa 1`
≠ plain AR with `-fa 0` on the same prompt. At `-fa 0 -ctk f16 -ctv f16`, MTP is
**token-exact**: 3 prompts × 129 tokens × 2 reps × 2 modes, including the prompt that
fails at `-fa 1`. So verification is lossless; the divergence is FA kernel selection
(1-row vec vs multi-row tile near-ties), not the feature. Pre-existing issue worth filing
separately: FA is not bit-stable against non-FA on this model.

**Rule for future speculative testing on this host:** token-exactness must be checked at
`-fa 0 -ctk f16 -ctv f16`, otherwise the test measures FA kernel selection.

### VRAM is the binding constraint, not speed
Peak at `-c 4096`: spec off **12.88 GB**, spec on **13.84 GB** (of a 15.98 GiB card).
Adding ≈ +2.88 GB of KV for 112k would put spec-on at ≈ 16.7 GB — **over the card**.
So MTP fits at short-to-mid context and does not fit at 112k; the crossover needs
measuring at 32k before any deployment claim.
