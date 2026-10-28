# N1 + N4 — implementable patch specification

Derived from `GFX1030_QWEN35_MASTERPLAN.md` §C5, §5.2 row 03, §9 (N1, N4), §10.2–§10.4.
Written 2026-10-27. Nothing here is committed.

**Tagging.** Every claim carries one tag:

| tag | meaning |
|---|---|
| **[P]** | proven in this session, on this host, with the instrument named in the same sentence |
| **[U]** | read from an upstream source, quoted with the URL and the byte-identical artifact |
| **[H]** | hypothesis with a stated discriminator; not to be treated as a result |

No untagged assertions. Where the masterplan and the code disagree, the code wins and the
disagreement is recorded (§0.3).

**Environment for every [P] below.** gfx1030 (RX 6900 XT), ROCm 7.2.5 / HIP 7.2.53211-3d9ef42,
clang 22.0.0git, base commit `d327bf1f0`, model of record
`/home/stormrage/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`.
Pre-flight on every run: `pgrep -a -f "llama-server|llama-bench"` empty, one GPU process at a time.

---

## 0. Facts this spec stands on

### 0.1 Geometry as it is today

**[P]** `mmvq.cu:463-591` `calc_nwarps()`: the branches are `GENERIC`, `GCN`, `RDNA4`, `RDNA3_0`,
`TURING`, `GB10`. `MMVQ_PARAMETERS_RDNA2` matches none of them and falls to `return 1;` at
`mmvq.cu:590`.

**[P]** `mmvq.cu:607-632` `calc_rows_per_block()`: the branch at `:608` lists only
`GENERIC || GCN || TURING || GB10`; `MMVQ_PARAMETERS_RDNA2` falls to `return 1;` at `:631`.

⇒ every MMVQ dispatch on this device is 1 wave32 × 1 row, at every width. The masterplan's
statement is correct.

### 0.2 The table id is shared with gfx115x

**[P]** `mmvq.cu:122-123` (`__device__`) and `:142-143` (host) map **both** `RDNA2` (gfx1030–1037)
**and** `RDNA3_5` (gfx1150–1153) to `MMVQ_PARAMETERS_RDNA2`. Any behaviour added to that table id
would silently change Strix Halo, which cannot be tested here. The table must be **split**, not
extended in place.

### 0.3 The trap that makes a naive implementation produce garbage silently

**[P]** On this toolchain the **host pass of a HIP translation unit does not define `RDNA2`**
(equivalently `__gfx1030__`). Measured with a 3-line probe compiled by the same hipcc:

```
$ hipcc --offload-arch=gfx1030 -std=c++17 rdna2_hostpass.cpp -o probe && ./probe
host_sees_RDNA2=0
$ hipcc --offload-arch=gfx1030 --offload-arch=gfx1100 ... && ./probe   # multi-arch
host_sees_RDNA2=0
```

The tree already knows this: `mmq.cuh:279-284` says *"The host pass of the HIP toolchain does not
define the per-arch macros, so it lands here too"*.

Consequence, and it is the single most important line in this document:
**`#if defined(RDNA2)` must not appear inside `calc_nwarps()` / `calc_rows_per_block()`.** Those
are `__host__ __device__ constexpr` and are evaluated **twice**: once in the device pass, where they
set `__launch_bounds__` (`mmvq.cu:635`) and the `constexpr int nwarps` inside the kernel
(`mmvq.cu:652`), and once in the host pass, where `calc_launch_params()` (`mmvq.cu:1051-1055`) sets
`dim3 block_dims(warp_size, nwarps, 1)`. An arch-macro guard returns 2 on the device and 1 on the
host, so the kernel is **compiled for 2 warps and launched with 1**. `threadIdx.y > 0` is then never
true, `tmp_shared[l][j][i][threadIdx.x]` at `mmvq.cu:830` is read never having been written, and the
uninitialised LDS value is added to **every** output element. See failure mode N1-F1.

The correct mechanism is the one upstream used: a **new table id** returned by both
`get_device_table_id()` overloads — `#elif defined(RDNA2)` for the device pass,
`GGML_CUDA_CC_IS_RDNA2(cc)` for the host pass. `#if defined(RDNA2)` *is* safe in a
`__device__`-only inline function (that is why N4's helper may use it).

### 0.4 Current geometry is width-invariant, and that is measurable

**[P]** Because the geometry ignores `ncols_dst` today, row 0 of a width-N `MUL_MAT` is
**byte-identical** to a width-1 `MUL_MAT`. Measured with the harness in Appendix A, k=5120,
m=1024/5120, deterministic input, run twice to confirm determinism:

| dtype | n=3 vs n=1, row 0 | n=6 vs n=1, row 0 |
|---|---|---|
| `q4_K` (m=5120) | byte-identical | byte-identical |
| `iq3_s` | byte-identical | byte-identical |
| `q6_K` | byte-identical | byte-identical |
| `iq4_xs` | byte-identical | byte-identical |

This is the pre-registered baseline for N1, and it is also the falsifier for the "rows_per_block is
a free tiling change" claim in §1.4.

### 0.5 Which width our deployment actually uses

**[P]** `--spec-draft-n-max 2` submits `n_max + 1 = 3` tokens to the verify pass (spec chain in
`common/speculative.cpp`), and `GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE)` with
`MMVQ_MAX_BATCH_SIZE 8` (`mmvq.cuh:3`) is the dispatch clamp at `mmvq.cu:1135`. **Width 3 is the
only width the recommended configuration exercises.** Widths 5–8 require `n_max ≥ 4`, which §C3
rejects.

### 0.6 dtype census of the model of record (from the GGUF itself)

**[P]** Parsed the 866 tensor infos out of the file (script kept at `/tmp/opencode/gguf_dtypes.py`):

| dtype | tensors | Gelem | share | where |
|---|---:|---:|---:|---|
| `iq3_s` | 144 | 8.478 | 31.0 % | main blocks |
| `iq4_xs` | 96 | 5.689 | 20.8 % | main blocks |
| `iq3_xxs` | 78 | 5.421 | 19.8 % | main blocks |
| **`q4_K`** | 39 | **2.813** | 10.3 % | 1.541 in main blocks + `output.weight` 1.271 |
| `iq2_s` | 17 | 2.498 | 9.1 % | main blocks |
| `iq2_xs` | 9 | 0.765 | 2.8 % | main blocks |
| `q2_K` | 13 | 0.734 | 2.7 % | main blocks |
| **`q6_K`** | 8 | **0.425** | 1.6 % | **100 % in the MTP block** (`blk.64.*`) |
| `iq2_xxs` | 5 | 0.383 | 1.4 % | main blocks |
| `iq1_m` | 1 | 0.089 | 0.3 % | main block |
| `bf16` / `f32` | 456 | 0.027 | 0.1 % | `ssm_alpha`/`ssm_beta`/`ssm_conv1d` |

No `Q5_K` anywhere. **Two corrections to the masterplan's §13 census follow**: `q4_K` is 10.3 % of
the model by elements, not just `lm_head`; and the MTP block is entirely `q6_K`. §2.1 uses this.

**[P]** All 554 tensors with ≥2 dims have `ne01 % 4 == 0`, i.e. the `rows_per_block > 1` tail
(`nrows_x % rows_per_block != 0`) is **unreachable on this model**. That does not make it
impossible — see N1-F4.

**[P]** The quantised 2-D tensors have `ne0` (K) ∈ {5120 ×279, 6144 ×65, 10240 ×1, 17408 ×65} ⇒
`blocks_per_row_x = ne0/256 ∈ {20, 24, 40, 68}`, and `ne1` (rows) ∈ {1024, 5120, 6144, 10240, 12288,
17408, 248320}. This matters for §1.9: the win depends on how cleanly `blocks_per_row_x` divides by
the new `blocks_per_iter`, and the four K values do not divide equally well.

### 0.7 Instrument facts that change the gates

**[P]** `llama-bench` **rejects** `--spec-type` (`error: invalid parameter for argument:
--spec-type`), because the option is registered with
`.set_examples({SPECULATIVE, SERVER, CLI})` only (`common/arg.cpp:4323-4331`). Therefore
`scripts/gfx1030.py bench --ab BASE.so FIX.so` — which shells out to `llama-bench`
(`scripts/gfx1030.py:1065`) — **cannot** run the N1 speed gate. §1.7 gives the replacement.

**[P]** `llama-eval-callback` is **not byte-level**. `common/debug.cpp:112` prints each sampled
element with `%12.4f` and `:122` prints `sum = %f` over a host-side sequential float accumulation.
A 1-ULP change in one element of a 5120-element tensor is invisible at that precision. It is a
high-sensitivity aggregate, not a bit comparison. §1.6(b) uses a real byte comparison instead and
keeps eval-callback only as the whole-model cross-check.

**[P]** `test-backend-ops -o MUL_MAT -b ROCm0` on this tree, today: **2445/2445 passed**. This is
the baseline that gate (a) in §1.5 refers to. Note that the MUL_MAT suite only registers `m=16`,
`m=32`, `m=4096`, `m=900000`, `m=1700000` (`tests/test-backend-ops.cpp:11209-11239`) — all
multiples of 4, so it would **not** catch a rows=4 tail overrun (N1-F4).

---

## 1. N1 — per-width MMVQ geometry for the verify band

### 1.1 The table, entry by entry, with provenance

Width = `ncols_dst`. `[U]` values are verbatim from the upstream table
(`stew675/llama-cpp-rdna-boosts`, `patches/0007-rdna-boosts-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM.patch`,
blob `ee09952ca1add9fc9b21c97aa3a77cb384e79359`, 86 229 B, fetched 2026-10-27; the file's own
attribution is *"llama-navi21-furnace 0aa20a02b, b5f65d215, 05512bb1d"*). The `sixvolts/llama-halo-hybrid`
PR #165 URL returns 404 as of today — **[P]** it was not retrievable; the content was obtained from
stew675's copy of it, which is the same patch the masterplan cites.

| width | upstream `nwarps` | upstream `rows_per_block` | provenance | **ship?** | **our `nwarps`** | **our `rows`** |
|---:|---:|---:|---|---|---:|---:|
| 1 | 4 | `small_k ? nwarps : 1` | [U] | **no** — R1, §1.1 | 1 | 1 |
| 2 | 2 | 2 | [U] | yes | 2 | 2 |
| 3 | 2 | 2 | [U] | yes — *the* width for us | 2 | 2 |
| 4 | 2 | 2 | [U] | yes | 2 | 2 |
| 5 | 1 | 4 | [U] | no — R2, §1.1 | 1 | 1 |
| 6 | 1 | 4 | [U] | no — R2, §1.1 | 1 | 1 |
| 7 | 1 | 4 | [U] | no — R2, §1.1 | 1 | 1 |
| 8 | 1 | 4 | [U] | no — R2, §1.1 | 1 | 1 |
| >8 | `default: 1` | `default: 1` | [U] | unreachable (`MMVQ_MAX_BATCH_SIZE 8`) | 1 | 1 |

Upstream's reported deltas, from the same patch's comment block and the masterplan §5.2:
**+6.3 % at widths 2–4** (2 waves + 2 rows) and **+7.6 % at widths 5–8** (1 wave + 4 rows), on a
V620 (gfx1030) with Qwen3.8-27B. **[U]**, and per §5.6 those are an *expectation*, never a result.

There is no `[H]` entry in the shipped table. Everything shipped is `[U]`; the only `[H]` is the
*transfer* of the delta to our dtype mix (§1.4), which the A/B decides. This is deliberate: a table
half interpolated from a 3-data-point sweep would be indistinguishable from an invention, and the
sweep to recover it costs more than the A/B that would replace it.

**R2 — widths 5–8 are dropped.** (i) Our deployment never issues them (§0.5); (ii) `rows_per_block =
4` exists **nowhere** in the tree today — every other table id uses 1 or 2 — so it is the only
genuinely new geometry, and it is the one that widens the tail read of N1-F4 from 1 row to 3;
(iii) their +7.6 % is measured at `n_max ≈ 7`, a configuration §C3 rejects. Dropping them removes
an untested code path from a change whose value is at width 3. If `n_max ≥ 4` is ever adopted,
re-open 5–8 as a separate measured change (add the table rows, run the same gate).

**R1 — width 1 is dropped.** Upstream sets `nwarps = 4` at width 1. That is a decode
change, it re-partitions the K loop at width 1 (5 → 2 iterations for `iq3_s` at k=5120), and it
would break the only exactness invariant we can *prove* for this patch (§1.4: width 1
byte-identical). Our own record has multi-warp decode measurements coming back flat-to-worse
(§C5, §11), and upstream keeps `rows = 1` there — so the source itself only claims the *rows* stay
at 1. The masterplan's C5 wording ("leaves `ncols_dst == 1` at one row per block") is right about
rows and imprecise about warps; §5.2 row 03 already states "width 1 → 4 waves". Both are recorded
here so the next reader does not have to re-derive it.

### 1.2 Patch, part 1 of 2 — split the table id (5 hunks, `ggml/src/ggml-cuda/mmvq.cu`)

**Hunk 1** — `mmvq.cu:107-115`, append one enumerator. Appending (not inserting after `RDNA2`)
avoids renumbering; nothing persists the integer (all uses are symbolic — verified by grep),
but there is no reason to move the others.

```c
 enum mmvq_parameter_table_id {
     MMVQ_PARAMETERS_GENERIC = 0,
     MMVQ_PARAMETERS_TURING,
     MMVQ_PARAMETERS_GCN,
     MMVQ_PARAMETERS_RDNA2,
     MMVQ_PARAMETERS_RDNA3_0,
     MMVQ_PARAMETERS_RDNA4,
-    MMVQ_PARAMETERS_GB10
+    MMVQ_PARAMETERS_GB10,
+    // gfx103x only (RDNA2 proper). MMVQ_PARAMETERS_RDNA2 above keeps serving gfx115x unchanged;
+    // that is why this is a new id rather than an arch macro inside calc_nwarps(): the host pass of
+    // the HIP toolchain does not define the arch macros, so a macro guard there would compile the
+    // kernel for one warp count and launch it with another (see mmq.cuh: "host pass ... lands here").
+    // Band-scoped on purpose: only ncols_dst 2..4 differ from today (V620, Qwen3.8-27B: +6.3 %).
+    MMVQ_PARAMETERS_RDNA2_BAND
 };
```

**Hunk 2** — `mmvq.cu:122-123`, device-side selection.

```c
-#elif defined(RDNA2) || defined(RDNA3_5)
-    return MMVQ_PARAMETERS_RDNA2;
+#elif defined(RDNA3_5)
+    return MMVQ_PARAMETERS_RDNA2;      // gfx115x: unchanged, untestable here
+#elif defined(RDNA2)
+    return MMVQ_PARAMETERS_RDNA2_BAND; // gfx103x: verify band only
```

**Hunk 3** — `mmvq.cu:142-144`, host-side selection. This is the half that keeps
`calc_launch_params()` and `__launch_bounds__` in agreement.

```c
-    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
-        return MMVQ_PARAMETERS_RDNA2;
-    }
+    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
+        return MMVQ_PARAMETERS_RDNA2;      // gfx115x: unchanged
+    }
+    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
+        return MMVQ_PARAMETERS_RDNA2_BAND; // gfx103x: verify band only
+    }
```

**Hunk 4** — `mmvq.cu:588-590`, new branch in `calc_nwarps()`, after the `GB10` block and before
the final `return 1;`.

```c
         return generic;
     }
+    // gfx103x verify band. ncols_dst == 1 is deliberately absent: multi-warp decode is measured
+    // flat-to-worse here and upstream leaves rows at 1 there, so the band stops at 2..4. At these
+    // widths each wave already carries ncols_dst accumulators, so the K loop is worth splitting in
+    // two exactly once the block still covers K blocks per trip (k = 5120 -> 20 blocks per row).
+    if (table_id == MMVQ_PARAMETERS_RDNA2_BAND) {
+        switch (ncols_dst) {
+            case 2:
+            case 3:
+            case 4:
+                return 2;
+            case 1:
+            default:
+                return 1;
+        }
+    }
 
     return 1;
 }
```

**Hunk 5** — `mmvq.cu:607-608`, new branch in `calc_rows_per_block()`. Inserted at the top so the
`GENERIC || GCN || TURING || GB10` list below is untouched.

```c
 static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
+    // Two rows per block share the q8_1 activation loads across the verify columns. Rows are a pure
+    // tiling change over the dst dimension: every row keeps its own accumulator and its own lane
+    // assignment, so this cannot alter a row's reduction order (see the bit-identity gate). Widths
+    // 5..8 stay at 1 row: their 4-row entry needs n_max >= 4, which this deployment does not run.
+    if (table_id == MMVQ_PARAMETERS_RDNA2_BAND) {
+        switch (ncols_dst) {
+            case 2:
+            case 3:
+            case 4:
+                return 2;
+            case 1:
+            default:
+                return small_k ? nwarps : 1;
+        }
+    }
     if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
```

`small_k` is always false on gfx1030 — `should_use_small_k()` bails out for every RDNA device at
`mmvq.cu:1185`. **[P]**

**Hunks 2 and 3 together *are* the `defined(RDNA2)` guard**, and both are required: the device pass
needs the macro (there is no `cc` in a kernel), the host pass can only use the runtime predicate.
Adding only Hunk 2 is failure mode N1-F1; adding only Hunk 3 is a launch error (block dims larger
than `__launch_bounds__`). The width scope is the pair of `switch` statements in Hunks 4 and 5 —
nothing else in either function changes, so widths 1 and 5–8 take exactly today's path through
`default:` → `1` / `small_k ? nwarps : 1`.

**The `ncols_dst` clamps are untouched and must stay untouched.** There are three:
`GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE)` (`mmvq.cu:1135`); `ggml_cuda_should_use_mmvq()`'s
`ne11 <= MMVQ_MAX_BATCH_SIZE` on the AMD path (`mmvq.cu:329-450`, reached from
`ggml-cuda.cu:2045`); and `get_mmvq_mmid_max_batch_*()` for `MUL_MAT_ID`, whose RDNA1/RDNA2 arm is
`get_mmvq_mmid_max_batch_rdna1_rdna2` (`mmvq.cu:316-317`, `:448-449`). The third one narrows the
*MoE/MUL_MAT_ID* band, is keyed on `cc` rather than on `table_id`, and would be a separate admission:
a band change there needs its own A/B and this model has no expert path (§11). Leave it.

**Nothing else in the file needs to change.** Checked, and this is the complete list of `table_id`
consumers: `calc_nwarps` (`:464-590`), `calc_rows_per_block` (`:608-631`),
`calc_launch_params` (`:1051-1052`), `should_use_small_k` (`:1158`), `should_halve_iters`
(`:1194`, only true for `GB10`). None of them has an `RDNA2` arm that would now miss gfx103x, and
the MoE path (`mul_mat_vec_q_moe`, `:893-1045`) does not consult `table_id` at all.

### 1.3 What the dispatch does with the result

For width 3, `calc_launch_params<type>(3, ...)` now returns
`block_dims = (warp_size, 2, 1) = (32,2,1)` and `block_nums.x = ceil(nrows_x/2)`; the kernel
instantiation `mul_mat_vec_q<type, 3, false, false, false>` carries `nwarps = 2`,
`rows_per_cuda_block = 2`, `blocks_per_iter = vdr * 64 / qi`, and
`__launch_bounds__(64, 1)` (`MMVQ_LB_MIN` = 1, `mmvq.cu:9`). LDS: `tmp_shared[1][3][2][32]` =
768 B, `tmp_shared_gate[1][...]` unused (`has_fusion` is width-1 only, `mmvq.cu:1071`).

### 1.4 Numerics: the reduction order, stated plainly

Two knobs, two different answers. **This distinction is the whole correctness argument, and it is
testable in one run (§1.6(b)).**

**`rows_per_block` (1 → 2): cannot change a row's numerics. [P-by-construction]**

Each output row has its own accumulator (`float tmp[ncols_dst][rows]`, `mmvq.cu:750`) and its own
K-loop trip set: the loop bound `kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter`
(`mmvq.cu:756`) depends on `tid`, `vdr`, `qi`, `nwarps`, `ncols_x` — **not** on
`rows_per_cuda_block`. The per-row index appears only as a pointer offset
(`kbx_offset + i*stride_row_x + kbx`, `:788`). The reduction (`tmp_shared` write at `:807`, the
`l < nwarps-1` sequential add at `:829-830`, `warp_reduce_sum` at `:837`) is executed once per
`(j,i)` with the same operand order for every `i`. So row `i` in a 2-row block sums exactly the
terms row 0 would sum in a 1-row block, in the same association.

Residual risk, compiler level: FMA contraction. `tmp[j][i] += vec_dot_q_cuda(...)` is unchanged
textually; `rows` appears only in the array dimension. A contraction difference would show up in the
gate. **Prediction to pre-register: with `nwarps` unchanged, a rows-only change must leave bytes
identical.** If it does not, the claim above is falsified and rows must be re-classified as
numerics-changing before shipping.

**`nwarps` (1 → 2): DOES change the reduction order. This is a correctness risk, not a perf knob.**

Proof from the code, three independent changes at once:

1. `blocks_per_iter = vdr * nwarps * warp_size / qi` (`mmvq.cu:661`) doubles, so the K blocks are
   re-partitioned: today lane group `g = tid/(qi/vdr)` sums blocks `g, g+4, g+8, g+12, g+16` for
   `iq3_s` at k=5120 (5 terms, `blocks_per_iter = 4`); with 2 warps the stride becomes 8 and it sums
   `g, g+8, g+16` (3 terms). Different terms per lane.
2. Warp 1's partials are written to `tmp_shared[0][j][i][lane]` (`:807`) and added into warp 0's
   accumulator **sequentially and before the lane reduction** (`:830`), i.e. one extra addition node
   is spliced into the tree, at a different level than the `__shfl_xor` tree.
3. `warp_reduce_sum` (`common.cuh:469-475`) is a `__shfl_xor` butterfly over `warp_size`; unchanged
   itself, but it now receives differently-valued lanes.

FP addition is not associative ⇒ widths 2–4 are **not bit-identical** to today. The masterplan §10.2
already says this; this section is the code proof, and it means:

- the gate for widths 2–4 is **not** "identical", it is **(i) a bounded difference, and (ii) an
  identical token stream**;
- expected magnitude: a re-association of a 20–24 term sum in fp32 gives a relative error of order
  `log2(24)·ε ≈ 2.9e-7`; for outputs of magnitude O(1) expect `≤ 1e-5` absolute. **[H]** — but the
  discriminant is sharp: a per-element difference `≥ 1e-2`, or a nonzero difference on a *rows-only*
  width, is a bug, not rounding (N1-F2, N1-F3).
- the gate for width 1 is "identical, or stop" — width 1 still takes the same instantiation with
  `nwarps = 1`, `rows = 1`, `blocks_per_iter = vdr*32/qi`, so its code and its reduction tree are
  unchanged. **[P]** any diff there means the patch leaked into the decode path.

### 1.5 Gate (a) — the regression suite

```bash
pgrep -a -f "llama-server|llama-bench"          # must be empty
cd /home/stormrage/llama.cpp
./build/bin/test-backend-ops -o MUL_MAT -b ROCm0 2>&1 | tail -3
```

Pass: `2445/2445 tests passed` (baseline measured today, §0.7). This **does** exercise widths 1–9 for
every quantised `type_a` (`tests/test-backend-ops.cpp:11220-11234`: `m=16, n=1..9, k=256`, plus
`m=16, n=1/8, k=4096`), so it is the only pre-existing automated test that reaches the changed
kernels — and `k=4096` (16 blocks per row) is what actually exercises the K split and the
`tmp_shared` reduction rather than the single-trip case. It would catch N1-F1 (values are compared
against a CPU reference) but not N1-F4.

### 1.6 Gate (b) — width 1 byte-identity, widths 2–4 bounded and token-identical

**Deviation from the brief, stated up front.** The brief asks for "byte-level, not tolerance" at
widths 2–8 via `llama-eval-callback`. That is impossible for two independent reasons: (1) the tool
prints `%12.4f` and a 6-decimal aggregate (§0.7), so it cannot resolve a last-bit change; (2) widths
2–4 **cannot** be byte-identical by construction, because `nwarps` re-associates the K sum (§1.4) —
an equality gate there would reject a correct patch. So the byte comparison is applied where identity
is a real invariant (width 1, widths 6/8, and all of N4), and widths 2–4 get a bound plus the token
stream. If a reviewer insists on byte-identity at width 3, the answer is that the change cannot
deliver it and should be refused on those terms rather than measured dishonestly.

**b1. Byte comparison, width 1 and width 3/4, per dtype.** Build once against
`BASE = build-hip-gfx1030/bin/libggml-hip.so.0` (save its sha256 first) and once against the patched
library; the harness in Appendix A writes raw `f32` output. Save it as
`/tmp/opencode/mmvq_bitexact.cpp`, build with the command in Appendix A.

```bash
for t in iq3_s iq3_xxs iq4_xs q4_K q6_K iq2_s; do
  for n in 1 2 3 4 6 8; do
    ./mmvq_bitexact ROCm0 $t 1024 5120 $n base_${t}_n${n}.bin   # BASE library
    ./mmvq_bitexact ROCm0 $t 1024 5120 $n fix_${t}_n${n}.bin    # FIX  library
  done; done
```

| width | required result | why |
|---:|---|---|
| 1 | `cmp` identical, every dtype | geometry untouched (§1.4) |
| 2,3,4 | **not** identical; and `max abs diff` on an O(1) output `< 1e-4` | `nwarps` re-associates (§1.4) |
| 6,8 | **identical** (rows stays 1 at these widths) | this patch does not touch them |

A `cmp` failure at width 1, or at widths 6/8, is a stop. A `max|diff| ≥ 1e-2` anywhere is a stop.

**b2. Token stream, spec on and spec off.**

```bash
# spec OFF: pins the decode path and every prefill-side kernel
LC_ALL=C ./build-hip-gfx1030/bin/llama-cli -m <model> -ngl 99 -fa 0 -ctk f16 -ctv f16 \
  -p "<fixed 512-token prompt>" -n 512 --temp 0 -s 42 -st --no-warmup > base_dec.txt

# spec ON at width 3: the only path that exercises the patch
LC_ALL=C ./build-hip-gfx1030/bin/llama-cli -m <model> -ngl 99 -fa 0 -ctk f16 -ctv f16 \
  -p "<same prompt>" -n 512 --temp 0 -s 42 --spec-type draft-mtp --spec-draft-n-max 2 \
  -st --no-warmup > base_spec.txt
```

`-fa 0 -ctk f16 -ctv f16` is mandatory (§10.3: `-fa 1` flips greedy tokens). Compare the generated
text with `diff` after stripping the `[ Prompt: ... ]` line. Both files must match the patched run
**exactly**. `--temp 0` + fixed seed makes this a function of the model math alone.

Whole-model cross-check with the existing instrument (accepts a tolerance-free *aggregate*, not
bytes — §0.7):

```bash
LC_ALL=C ./build-hip-gfx1030/bin/llama-eval-callback -m <model> -ngl 99 -fa 0 -ctk f16 -ctv f16 \
  -p "<fixed prompt>" -ub 1 -n 8 --temp 0 -s 42 > base_cb.txt
```
Diff the `sum = ` lines only. Widths 2–4 may show 6th-decimal movement; **width 1 must show none**.

### 1.7 Gate (c) — the actual speed gate: speculative decode at n_max = 2

**Pre-registered expectation, so the outcome is falsifiable rather than narratable.** The MTP
`n_max = 2` verify pass is 19.65 ms per round for 3 tokens (§6.3). **[H]** if the width-3 MMVQ
dispatches carry most of that pass, +6.3 % on them is `0.063 · 19.65 = 1.24 ms` per round, i.e.
`0.41 ms` per emitted token; against the 50.9 t/s decode baseline (19.65 ms/token) that is **+2.1 %,
≈ 52 t/s**. Anything above 54 t/s means the ledger moved for another reason — check the `.so` before
believing it. MDE is 0.30 t/s (§10.3), so a real +2 % is ~3.4× MDE: detectable with 5 pairs, not
with 3.

`llama-bench` cannot take `--spec-type` (§0.7), so `scripts/gfx1030.py bench --ab` is out. Use
`llama-cli -st`, which prints exactly one parseable line, and swap the library between runs with a
sha256 check both ways (this is the O1/O3 lesson: swapping the wrong `.so` measures a no-op).

```bash
# pre-flight
pgrep -a -f "llama-server|llama-bench"     # empty
sha256sum build-hip-gfx1030/bin/libggml-hip.so.0   # record BASE
cp build-hip-gfx1030/bin/libggml-hip.so.0 /tmp/opencode/base.so
# FIX is the same file rebuilt from the patched tree; record its sha256 too.

# one sample, of five interleaved pairs, alternating order
run() { LC_ALL=C ./build-hip-gfx1030/bin/llama-cli -m <model> -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 \
        -fitt 128 -p "<fixed prompt, 32-64 tokens>" -n 128 --temp 0 -s 42 \
        --spec-type draft-mtp --spec-draft-n-max 2 -st --no-warmup 2>&1 \
        | sed -n 's/.*Generation: \([0-9.]*\) t\/s.*/\1/p'; }   # LC_ALL=C: the decimal is a dot
```

Per pair: `cp /tmp/opencode/base.so build-hip-gfx1030/bin/libggml-hip.so.0` → run → `cp` the FIX
`.so` → run (reverse the order on odd pairs), `cmp` the restored file against the backup after the
series. Record sclk and DPM state with every sample.

Pass: **median FIX/BASE ≥ +1.0 %** with **≥4/5 pairs the same sign**, *and* no pair below the
0.30 t/s MDE (§10.3). Also record the two other numbers the same run produces: prompt t/s must not
move beyond noise, and the 512-token greedy stream from b2 must be identical (a faster run that
changes tokens is a rejection, not a win).

Reporting α: **[P]** this tree exposes no draft-acceptance counter (`llama_perf_context_print` has
none, and the spec chain has no `SPC_CNT`/`SPC_INF` on the accept path). If the token stream is
identical, α is identical by construction; if the stream differs, do not report a t/s win — report
the differing token and stop.

### 1.8 Gate (d) — plain decode must not regress

Width 1 never enters the new branch, but `__launch_bounds__`/register allocation of a shared
translation unit can move, so re-measure:

```bash
scripts/bench-exclusive.sh build-hip-gfx1030/bin/llama-bench \
  -m <model> -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -p 0 -n 128 -r 5 -o jsonl
```
Interleaved A/B against BASE, ≥5 pairs, same protocol as (c). Pass: `tg128` within the MDE of
BASE (no regression), and `pp512` unchanged.

### 1.9 Predictions to check, and what a miss means

K-loop trip counts, `blocks_per_iter = vdr·nwarps·32/qi` (`mmvq.cu:661`), `warp_size = 32`.
**[P]** arithmetic on the constants (`vdr`, `qi` read from `vecdotq.cuh` / `ggml-common.h`); the
*effect* is [H].

| dtype | `vdr` | `qi` | blocks/iter @1 → @2 | trips, K=5120 (20 blk) | K=6144 (24) | K=17408 (68) |
|---|---:|---:|---:|---|---|---|
| `iq3_s` | 2 | 16 | 4 → 8 | 5 → 3 (8,8,4) | 6 → 3 | 17 → 9 (8·8+4) |
| `iq3_xxs` | 2 | 16 | 4 → 8 | 5 → 3 | 6 → 3 | 17 → 9 |
| `iq4_xs` | 4 | 32 | 4 → 8 | 5 → 3 | 6 → 3 | 17 → 9 |
| `q4_K` | 2 | 32 | 2 → 4 | 10 → 5 | 12 → 6 | 34 → 17 |
| `q6_K` | 2 | 32 | 2 → 4 | 10 → 5 | 12 → 6 | 34 → 17 |
| `iq2_s` | 2 | 16 | 4 → 8 | 5 → 3 | 6 → 3 | 17 → 9 |

Three consequences worth pre-registering:

- `q4_K`/`q6_K` halve **exactly** at every K in the model (all four `blocks_per_row_x` values are
  even). They should show the cleanest gain, and `q6_K` is 100 % of the MTP block (§0.6).
- the `iq*` types, which are 85.2 % of the model by elements, waste 20 % of the last trip at K=5120 and
  5 % at K=17408, so their gain is diluted by `(B/bpi1) / (2·ceil(B/bpi2))` — 0.83 at K=5120,
  0.94 at K=17408.
- if the A/B is flat while the ledger shows the width-3 dispatches, the cause is that dilution on the
  dominant dtypes, **not** a broken patch. If `q4_K`/`q6_K`-shaped work gains and the aggregate does
  not, that is the same story with numbers. **[H]**

### 1.10 N1 failure modes

| id | symptom | cause | how it is caught |
|---|---|---|---|
| **N1-F1** | `MUL_MAT` output wrong everywhere at widths 2–4, magnitude arbitrary, sometimes correct (shared memory happens to be zero); `test-backend-ops` fails at `n=2..4` | `#if defined(RDNA2)` used inside `calc_nwarps`/`calc_rows_per_block`: device compiles for 2 warps, host launches 1, `tmp_shared` read uninitialised (`mmvq.cu:830`) | §0.3 probe; gate (a); if (a) passes but b1 shows garbage at width 3, check this first |
| **N1-F2** | width 3 diff is `≥1e-2` or the token stream diverges | geometry applied to the wrong band, or `rows` wired to the wrong accumulator index | b1 magnitude bound; b2 `diff` |
| **N1-F3** | widths 6/8 are no longer byte-identical | the `default:` arm did not fall through to `small_k ? nwarps : 1` / `1` | b1 `cmp` |
| **N1-F4** | device fault (700/701) during a verify pass with `nrows_x % rows_per_block != 0` | the K loop reads row `row0+i` with no bound (`mmvq.cu:788`); only the store is guarded (`:844`). `rows=2` reads 1 row past the end of `src0` for odd `nrows_x` | **[P]** unreachable on this model (§0.6: all `ne01 % 4 == 0`); **[H]** latent for any other GGUF. If a new model is used, add `const int rowi = row0 + i < (int) stride_col_dst ? row0 + i : (int) stride_col_dst - 1;` and dereference with `rowi` |
| **N1-F5** | gfx115x behaviour changes | `MMVQ_PARAMETERS_RDNA2` was extended instead of a new id added | Hunk 1/2/3; `git diff` shows no arm that both RDNA3_5 and RDNA2 reach |
| **N1-F6** | MUL_MAT_ID (MoE/lm_head gather) changes | patched the MoE kernel or `get_mmvq_mmid_max_batch_*` | **[P]** this patch touches neither; diff review |
| **N1-F7** | t/s moves but the width-3 dispatch count does not | measured the wrong `.so`, or `--spec-type` silently ignored | sha256 both libraries; `pgrep`; the `llama-bench` reject in §0.7 |
| **N1-F8** | speed win at width 3, decode regression | `nwarps=4` or a width-1 arm slipped in | gate (d) |

### 1.11 What would make me refuse to ship N1

1. Width 1 is not byte-identical on any dtype (§1.6 b1). No exception, no "small diff".
2. Any per-element diff `≥ 1e-4` at widths 2–4 (my bound is 1e-5; 1e-4 is already generous).
3. The token stream differs at `-fa 0 -ctk f16` in either b2 run. A perf win with a different token
   stream is the exact failure the project has already paid for twice.
4. `test-backend-ops -o MUL_MAT` below 2445/2445.
5. Median gain `< +1.0 %` or fewer than 4/5 pairs of the same sign — then the band is admitted as
   *not transferable to this dtype mix* and the patch is withdrawn, not kept "because upstream
   measured +6.3 %" (§10.3: cross-implementation numbers do not cross).
6. `rows_per_block = 4` entries reappearing in the diff without their own A/B.
7. Any change to `RDNA3_5`'s effective table.

---

## 2. N4 — `ggml_cuda_mmq_scale_sumi`

### 2.1 Exposure on this model, before spending any GPU time

The three sites are `vecdotq.cuh:552` (`q4_K` MMQ), `:615` (`q5_K` MMQ), `:678` (`q6_K` MMQ).
**[P]** they are reached only from `mmq-vec-dot.cuh:932` and its siblings, i.e. the **MMQ** kernels,
which on RDNA2 run only when `ne11 > MMVQ_MAX_BATCH_SIZE` (`ggml-cuda.cu:2045-2052`,
`ggml_cuda_should_use_mmvq`) — prefill. **[P]** decode (width 1) and the verify band (widths 2–8)
never reach them.

**[P]** per §0.6 the affected tensors are `q4_K` (2.813 Gelem, of which 1.271 is `output.weight`) and
`q6_K` (0.425 Gelem, all inside `blk.64.*`), no `q5_K`. Two consequences the masterplan's `[H] 1–3 %`
does not state:

- `output.weight` probably does not contribute at all: llama.cpp gathers the output rows before the
  projection (`build_inp_out_ids`, `llama-graph.cpp:2830-2846`), so that `MUL_MAT` runs with
  `ncols = n_outputs`, and `n_outputs` is 1 whenever only the last token's logits are requested —
  which is what `llama-bench` and generation both do. **[H]**, because I did not run the ledger to
  confirm it. Discriminator, one cheap run: `scripts/gfx1030.py trace --ledger` on `pp512`; if
  `output.weight` shows up as an **MMVQ** dispatch at width 1, the claim holds and the largest single
  `q4_K` tensor is out of N4's exposure. If it shows as MMQ, add 1.271 Gelem to the numerator below
  and double the expectation.
- what remains is `q4_K` 1.541 Gelem + `q6_K` 0.425 Gelem = **1.966 of 25.624 Gelem = 7.7 %** of the
  per-token matvec work (§13 census), ≈ 9–11 % of prefill bytes.

So the honest expectation is **≤ 2 % pp512 and quite possibly < 1 %**, not 1–3 %. It is still worth
doing *if* it is bit-identical, because it is 7 lines and it removes a quarter-rate op — but it must
be ranked and reported as a hygiene/portability fix, not as a prefill lever.

### 2.2 Patch — `ggml/src/ggml-cuda/vecdotq.cuh`, 4 hunks

**Hunk 1** — insert the helper after `vec_dot_q3_K_q8_1_impl_mmq` (`:502`) and before
`#define VDR_Q4_K_Q8_1_MMVQ 2` (`:505`).

```c
     return d3*d8 * sumi;
 }
 
+// scale * dot-sum for the K-quant MMQ paths. v_mul_lo_u32 is quarter rate on RDNA2 while a float
+// multiply is full rate, so do the product in float there; every other target keeps the integer
+// multiply so its codegen is unchanged. Both forms round the same exact integer to nearest fp32,
+// so the result is bit-identical (see the argument in the patch description):
+//   q4_K/q5_K: |sc| <= 255, |sumi| < 2^17  -> |sc*sumi| < 2^25  (one RN rounding of the same value)
+//   q6_K:      |sc| <= 128, |sumi| <= 2^16 -> |sc*sumi| <= 2^23  (exact, no rounding at all)
+// No int32 overflow is possible on either path, so the integer form is not relying on wrap-around.
+static __device__ __forceinline__ float ggml_cuda_mmq_scale_sumi(const int sc, const int sumi) {
+#if defined(RDNA2)
+    return (float) sc * (float) sumi;
+#else
+    return (float) (sc * sumi);
+#endif // defined(RDNA2)
+}
+
 #define VDR_Q4_K_Q8_1_MMVQ 2
```

`#if defined(RDNA2)` is correct **here** and only here: this function is `__device__`-only, so the
host pass's version is dead code and no launch geometry depends on it (§0.3).

**Hunk 2** — `vecdotq.cuh:552` (`vec_dot_q4_K_q8_1_impl_mmq`).

```c
         const float2 ds8f = __half22float2(ds8[i]);
 
-        sumf_d += ds8f.x * (sc[i] * sumi_d);
+        sumf_d += ds8f.x * ggml_cuda_mmq_scale_sumi(sc[i], sumi_d);
         sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
```

**Hunk 3** — `vecdotq.cuh:615` (`vec_dot_q5_K_q8_1_impl_mmq`). Same text, different function:

```c
         const float2 ds8f = __half22float2(ds8[i]);
 
-        sumf_d += ds8f.x * (sc[i] * sumi_d);
+        sumf_d += ds8f.x * ggml_cuda_mmq_scale_sumi(sc[i], sumi_d);
         sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
```

**Hunk 4** — `vecdotq.cuh:678` (`vec_dot_q6_K_q8_1_impl_mmq`). Note the **int add is kept inside the
parentheses in the upstream form** — the two products are still summed as integers before the single
conversion *if* the compiler keeps them integer, which it does not after this change. That is why the
q6_K bound in Hunk 1 is the tight one (`2^23` each, sum `≤ 2^24`, exactly representable):

```c
-        sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
+        sumf_d += d8[i0/4] * (ggml_cuda_mmq_scale_sumi(sc_reg[i0/2+0], sumi_d.x) +
+                              ggml_cuda_mmq_scale_sumi(sc_reg[i0/2+1], sumi_d.y));
```

### 2.3 Why it is bit-identical — the argument, not the slogan

The claim is not "float multiply equals integer multiply". It is: **both forms round the same exact
real number to fp32 with round-to-nearest-even, and in the q6_K case neither rounds at all.**

**(i) The exponent ranges make every single rounding benign.**
`sc` is a `uint8_t` for q4_K/q5_K (`const uint8_t * sc`) and an `int8_t` for q6_K
(`sc_reg` is `(const int8_t *) &sc_packed`, `:662-663`). Both promote to an `int` that fp32
represents **exactly** (≤ 2^8). `sumi_d` is the dp4a accumulator:

- q4_K/q5_K: 4 outer iterations × `QI8_1 = 4` dp4a (`ggml-common.h:124`), each 4 byte-products,
  value range [0,15] × [-128,127] ⇒ `|sumi_d| ≤ 16·4·15·128 = 122 880 < 2^17`. **[P]** arithmetic on
  the constants at `vecdotq.cuh:542-548`.
- q6_K: 4 dp4a per accumulator (2 per `i`, 2 `i` per `i0`), values `v-32 ∈ [-32,31]` ⇒
  `|sumi_d| ≤ 16·32·128 = 65 536 = 2^16`. **[P]**

**(ii) The integer form cannot overflow, so there is no UB to compare against.**
max `|sc·sumi|` = `255 · 122 880 = 31 334 400 < 2^31`. The integer path is a well-defined exact
product. Same for q6_K (`128 · 65 536 = 2^23`).

**(iii) Both paths are one RN rounding of that same exact value.**
Integer path: `cvt.rn.f32.s32` of the exact product — IEEE round-to-nearest-even of the real value.
Float path: `(float)sc` and `(float)sumi` are exact conversions of exactly representable integers,
then `v_mul_f32` — IEEE round-to-nearest-even of the *same* real value. Same input, same rounding
mode ⇒ same bits. For q6_K both products are `≤ 2^23` and their sum `≤ 2^24`, all integers `≤ 2^24`
are exactly representable ⇒ **no rounding at all** on either path.

**(iv) What could break it, checked one by one.**

| threat | status |
|---|---|
| fast-math / `-ffast-math` | not in the ROCm defaults and would not matter: fp32 multiply is always RN; products are ≥ 1 in magnitude, so no denormal flush |
| `-ffp-contract=fast` fusing `sumf_d += a*b` | the fusion is identical on both paths: the multiplicand is bit-identical, so the FMA is too |
| int32 overflow | bounded at 3.1e7 (§ii) |
| signedness: `uint8_t` vs `int8_t` | the helper takes `int`; both promote identically to what the original expression did |
| NaN/Inf weights | `sumi` is an integer, so no NaN can enter before the outer multiply, which is unchanged |

**[H]** the *speed* claim (that removing one quarter-rate `V_MUL_LO_U32` beats the extra full-rate
`V_CVT_F32_*` it adds) is not established and is not needed for correctness. The cheap way to settle
it before a benchmark: `llvm-objdump -d --mcpu=gfx1030` the built `libggml-hip.so.0`, find the MMQ
`q4_K` kernel inner loop, and count `v_mul_lo_u32` vs `v_cvt_f32_i32` in both builds. If the count of
quarter-rate ops does not drop, expect a wash and say so instead of measuring noise.

### 2.4 Gate — prefill, byte-level, then t/s

**a. Byte comparison at MMQ widths.** The same harness, `n = 512` (MMQ territory), `q4_K` and
`q6_K`, m=1024 and m=5120:

```bash
for t in q4_K q6_K; do for m in 1024 5120; do
  ./mmvq_bitexact ROCm0 $t $m 5120 512 base_${t}_${m}.bin   # BASE library
  ./mmvq_bitexact ROCm0 $t $m 5120 512 fix_${t}_${m}.bin    # FIX  library
  cmp base_${t}_${m}.bin fix_${t}_${m}.bin || echo "DIFF $t $m"
done; done
```
Pass: **every `cmp` identical.** This is the check that the masterplan's §10.2 asks for
("verify, do not assume") — and unlike `test-backend-ops`, which compares against a CPU reference
with a tolerance, it compares *two GPU builds* bit for bit.

**b. `test-backend-ops -o MUL_MAT -b ROCm0`**: 2445/2445 (unchanged; it is the only suite that
drives the MMQ K-quant tiles through real shapes).

**c. Prefill A/B** (this is a prefill change; decode is not the gate):
```bash
scripts/bench-exclusive.sh build-hip-gfx1030/bin/llama-bench \
  -m <model> -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 -p 512 -n 0 -r 5 -o jsonl
```
Interleaved, ≥5 pairs, sha256 both libraries, sclk recorded. Pass: `pp512` median above the MDE
(`pp512` MDE is not 0.30 ms — take it from the spread of the five BASE samples, per §10.3) and no
`tg128` movement (`-p 0 -n 128` in the same series, since the translation unit is shared).

**d. Whole-model exactness**: the b2 commands from §1.6 with `-n 512` and, additionally,
`-ub 512 -n 16` (the batched/prefill path of masterplan §10.2 check 2). Token stream identical.

### 2.5 N4 failure modes

| id | symptom | cause | caught by |
|---|---|---|---|
| **N4-F1** | the last bit differs on many/all `q4_K`/`q5_K` outputs at prefill widths only | rounding mode differs — i.e. the int path was re-expressed with `cvt.rz` or a multiply-add was introduced | 2.4a `cmp` |
| **N4-F2** | large errors, sign flips in the q6_K path | the `sc_reg` sign extension was lost (helper took `unsigned`, or the call passes a `uint8_t` view) | 2.4a; q6_K `cmp` at m=1024 |
| **N4-F3** | decode or widths 2–8 changed | the helper was used in a `_vmmq` function or in `mmvq.cu` | 2.4d token stream; grep the diff for `_vmmq` |
| **N4-F4** | non-RDNA2 (gfx11/gfx12/BLAS builds) change | the `#else` arm was dropped, or the guard was written as `#if !defined(GGML_USE_HIP)` | diff review: the `#else` must be verbatim `(float) (sc * sumi)` |
| **N4-F5** | compiles but the helper is a no-op on gfx1030 | `RDNA2` not defined in the *device* pass — impossible unless `vendors/hip.h` is also touched | `llvm-objdump` count in §2.3; sha256 of `hip.h` unchanged |
| **N4-F6** | perf moves ≥5 % | something else changed; 7.7 % of the matvec cannot produce that | expected-value bound in §2.1; check `git diff --stat` |

### 2.6 What would make me refuse to ship N4

1. Any `cmp` failure in 2.4a — the whole point is that it is free.
2. A `pp512` gain claimed without 2.4a; or a gain `> 2 %` (outside the arithmetic bound of §2.1 ⇒ the
   measurement moved, not the kernel).
3. Any change to the `#else` arm's expression (it must stay the integer multiply, so no other target's
   codegen moves).
4. Any use of the helper outside the three `_mmq` functions.
5. If someone wants to justify it as a "prefill lever": refuse the framing. By §2.1's exposure
   arithmetic it is worth at most ~2 % of pp512 and most likely under 1 %, on one pass, for one
   tensor population; it is a hygiene fix and should stay ranked where §9 has it (a side-quest),
   never above N0/N1.

---

## 3. Shared requirements for both patches

**L0.** `cmake --build build-hip-gfx1030 -j$(nproc)` and the CUDA path must still configure (the
`#else` arms are CUDA-reachable). `clang-format` on the changed hunks (120 columns,
`.clang-format`); the Hunk-1 comment block is written to that width.

**L1/L3.** For each patch: run the gate with the change *forced off* (revert the table-id hunks /
the `#if defined(RDNA2)` arm) and diff the artifacts. The artifacts are identical by construction for
N4 and for N1-at-width-1; anything else is a bug in the gate.

**L5.** The `NAVI21`/band path must be *taken*: `test-backend-ops` passing does not prove that, since
it would also pass with the table never selected. Prove it by the b1 width-3 difference (it is nonzero
**only** if the band is active) — that is a cheap and sufficient dispatch witness for N1. For N4 the
witness is `llvm-objdump` (§2.3) plus 2.4a.

**Ordering.** N1 first. N4 is independent and can land in any gap, but do not run them in the same
A/B series — a shared translation unit makes `libggml-hip.so.0` change for both.

**Do not** touch `llama-server`, run two GPU processes, or benchmark above the card's pinned DPM state
without recording it.

---

## Appendix A — the byte-exactness harness

Kept at `/tmp/opencode/mmvq_bitexact.cpp` (read-only w.r.t. the repo; ~90 lines, public API only, one
GPU, no model file, no change to model code — it is a `MUL_MAT` written by hand, not a llama graph).
It is the only instrument available here that compares two *GPU builds* bit for bit; every
tolerance-based instrument (`test-backend-ops` vs CPU, `llama-eval-callback`'s 4-dp printout) cannot
settle N4 or N1-at-width-1. Build:

```bash
g++ -O2 -std=c++17 mmvq_bitexact.cpp -I/home/stormrage/llama.cpp/ggml/include \
    -L/home/stormrage/llama.cpp/build-hip-gfx1030/bin -lggml -lggml-base \
    -Wl,-rpath,/home/stormrage/llama.cpp/build-hip-gfx1030/bin -o mmvq_bitexact
# the rpath decides which libggml-hip.so.0 is loaded - it MUST be the build dir whose
# backend you are swapping, or the A/B silently compares a library against itself
```

The §0.4 numbers above were produced through the `build/bin` copy of the backend (same source, the
HIP build dir was not touched); before using b1 as a gate, re-run the *BASE* half against
`build-hip-gfx1030/bin` and re-derive the baseline files there.

It creates one `MUL_MAT` (`w[type, k×m] × x[f32, k×n] → y[f32, m×n]`), fills `w` from an LCG through
`ggml_quantize_chunk`, runs `ggml_backend_graph_compute`, and writes raw `f32` bytes.
**Verified today**: deterministic across runs (`cmp` identical), and it reproduces the width-invariance
baseline of §0.4 for `q4_K`, `iq3_s`, `q6_K`, `iq4_xs`. Type names are `ggml_type_name()` spellings —
lowercase for the `iq*` types (`iq3_s`), but `q4_K`/`q5_K`/`q6_K` keep the capital K.

## Appendix B — upstream source, verbatim

Fetched 2026-10-27 from
`https://raw.githubusercontent.com/stew675/llama-cpp-rdna-boosts/HEAD/patches/0007-rdna-boosts-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM.patch`
(blob `ee09952ca1add9fc9b21c97aa3a77cb384e79359`, 86 229 B). The relevant fragments, unedited:

```c
+    if (table_id == MMVQ_PARAMETERS_NAVI21) {
+        if (ncols_dst == 1) {
+            return 4;
+        }
+        if (ncols_dst <= 4) {
+            return 2;
+        }
+        return 1;
+    }
```
```c
+    if (table_id == MMVQ_PARAMETERS_NAVI21) {
+        switch (ncols_dst) {
+            case 1:
+                return small_k ? nwarps : 1;
+            case 2:
+            case 3:
+            case 4:
+                return 2;
+            // 5..8 (block-drafter verify): 4 rows amortise the 8 y-columns each wave holds; 8 rows regress
+            case 5:
+            case 6:
+            case 7:
+            case 8:
+                return 4;
+            default:
+                return 1;
+        }
+    }
```
```c
+static __device__ __forceinline__ float ggml_cuda_mmq_scale_sumi(const int sc, const int sumi) {
+#if defined(RDNA2)
+    return (float) sc * (float) sumi;
+#else
+    return (float) (sc * sumi);
+#endif // defined(RDNA2)
+}
```

Their own comment on the delta: *"ncols_dst == 1 wants 4 waves to keep enough loads in flight; the
verify shapes carry ncols_dst accumulators per wave, so wider blocks stop paying (V620,
Qwen3.8-27B: 2..4 at 2 waves +6.3%, 5..8 at 1 wave with 4 rows per block +7.6%)"*.

**Differences between this spec and upstream, deliberate and complete:**

| upstream | this spec | why |
|---|---|---|
| `MMVQ_PARAMETERS_NAVI21`, inserted mid-enum; `RDNA3_5` also split out with its own new behaviour | new id appended; `RDNA3_5` keeps `MMVQ_PARAMETERS_RDNA2` unchanged | gfx115x is untestable here and its +0.6 % is not ours to claim |
| width 1 → `nwarps = 4` | width 1 untouched | decode, re-associates the K loop, breaks the only provable invariant; our decode data says no |
| widths 5–8 → 4 rows | untouched | needs `n_max ≥ 4`, which §C3 rejects |
| applied to every MMQ quant type | same | no change |

## Appendix C — the run sheet, in order

1. Save `sha256sum` of `libggml-hip.so.0`; copy it to `/tmp/opencode/base.so`. Run §1.5 baseline
   (expect 2445/2445) and §1.6 b1 with the BASE library → these are the pre-registered base files.
2. Apply N1 hunks 1–5. Build. `test-backend-ops` (§1.5) → 2445/2445.
3. b1 byte comparison: width 1 identical, 2/3/4 differ and `< 1e-4`, 6/8 identical.
4. b2 token streams: spec off identical, spec on identical.
5. Gate (c): 5 interleaved pairs at `--spec-type draft-mtp --spec-draft-n-max 2`. Verdict.
6. Gate (d): 5 interleaved pairs at plain decode. Verdict.
7. If N1 passes: apply N4 hunks 1–4. Build. §2.4a `cmp` → identical; §2.4b → 2445/2445;
   §2.4c prefill A/B; §2.4d token stream.
8. Only then update the masterplan with the two results, and only with the numbers this run produced.
