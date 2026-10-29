# CJK / server-only corruption audit — ranked suspects (in-line audit, 2026-10-28)

Owner symptom: Chinese characters appearing in `llama-server` responses on
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` with `--reasoning-budget 32000`
`--reasoning-preserve` `--chat-template-kwargs '{"reasoning_effort":"xhigh"}'`
`--spec-type draft-mtp --spec-draft-n-max 2` `-c 60000 -fitt 128`. All 14
kernel-side `llama-cli` repros were clean; F3 (prequant arena) exonerated.

## Verdict on the budget sampler: no smoking gun

The `--reasoning-budget` implementation was audited end-to-end at the code level:

- `common/reasoning-budget.cpp:74-186` — state machine (IDLE → COUNTING →
  WAITING_UTF8 → FORCING → DONE) is coherent. Forcing sets all logits but the
  forced end-tag token to `-INFINITY` (`reasoning-budget.cpp:181-185`); UTF-8
  boundary check `common_utf8_is_complete` (`reasoning-budget.cpp:104-108`)
  prevents truncating a multi-byte character — the exact guard CJK would need.
- `id_rng`: `common/sampling.cpp:476-508` — `rbudget` only advances on
  `is_generated` tokens; grammar replay of the end sequence on DONE is correct.
- Server side: `tools/server/server-context.cpp:3064` (`force` control action)
  and `:2017` (per-prompt `common_sampler_init` rebuild) are consistent;
  `server-common.cpp:1516-1526` maps CLI `--reasoning-budget` into per-request
  fields with model auto-detected thinking tags.
- **No mechanism in this path can select a token outside the forced end-tag**
  during FORCING, and outside forcing it is pure passthrough.

## Ranked residual suspects

1. **Fork-novel rejection sampler, missing EOG guard (server-only path)**
   `common/sampling.cpp:736-840` `common_sampler_sample_and_accept_n_rejection`
   is fork-specific (rejection-sampling with draft distribution `q`) and is used
   by `llama-server` only (`server-context.cpp:4842-4844`). Unlike
   `common_sampler_sample_and_accept_n` (`:706-708`) and the synthetic variant
   (`server-context.cpp:117`), it has **no `llama_vocab_is_eog` guard on
   accepted draft tokens**: an accepted draft EOG stays in the KV/context
   without being output and generation continues in the same loop. This is
   context pollution, not CJK by itself — flag as a correctness nit to fix.
2. **Natural domain drift, not a bug (most likely actual cause (hypothesis))**
   The model of record is Qwen-family, whose pretraining and vocabulary are
   massively bilingual/Chinese-heavy. With `--temp 1.0 --top-p 0.95 --top-k 0
   --min-p 0.00` at reasoning effort xhigh, permissive sampling can legitimately
   surface CJK in long reasoning sessions. A/B-relevant: the sister run on the
   exact same model+flags that never shows CJK in 14 CLI repros is at temp 0? If
   CLI repros were run with different sampling temperature, "server-only"
   attribution weakens — a temperature-matched CLI repro at temp 1.0/top-p 0.95
   with the same long-context regime is the single cheapest decisive test.
3. **Chat-template / prompt rendering** (`--chat-template-kwargs` xhigh):
   unchecked in detail in this pass. The template changes the reasoning-mode
   instruction embedded in the prompt; xhigh instructions are long and the
   model's Chinese-lean could be amplified by longer prompts. Possible but
   unlikely to inject literal CJK text.
4. **Pipeline sampler deltas**: server applies penalties/top-k/p in the same
   chain code as CLI (same `common/sampling.cpp`); per-slot samplers are rebuilt
   per prompt (`server-context.cpp:2017`). No fork-specific divergence spotted.

## Recommended next probes (owner-actionable, zero code)

- Run one *temperature-matched* long CLI generation (`--temp 1.0 --top-p 0.95
  --top-k 0 --min-p 0.00` + same xhigh kwargs, same -c 60000, ~4k-token prompt,
  n≥2048, several seeds). Any CJK here = model behavior, not server bug.
- If the owner wants deterministic English output regardless: lower temp /
  raise min-p / turn reasoning budget off per request — but these are flag
  changes and stay the owner's call.

## Fork-novel code audit summary (this fork, uncommitted tree)

From earlier oracle pass plus this lane: fork-only speculative machinery in
`common/sampling.cpp` (`..._rejection`, `:736`) and server context is the one
place where behavior diverges structurally from stock llama.cpp sampling. The
missing EOG guard (item 1) is the only concrete defect found. Fix proposal
(small, code-only): replicate the `:708` EOG guard in `_rejection` — after an
**accepted** draft token, break the loop if it is EOG and more drafted tokens
remain. Gate coverage identical to other sampling-only diffs (eval-callback
0-diff at default flags unaffected; the change only fires when EOG-draft
interaction occurs).

## Second-pass follow-up (same session, later)

Three refinements after checking the remaining open areas:

**Rank 3 (chat-template kwargs) — CLEARED.** The server renders the prompt from
the template as *text*, then tokenizes it (`tools/server/server-context.cpp`,
`out_session.prompt().str()` → `common_tokenize`). `reasoning_effort` reaches
the template only as a variable adding fixed ASCII instruction strings; an
invalid value raises an exception instead of anything token-shaped. There is
no path by which template kwargs inject tokens, so no CJK injection source.

**Rank 1 EOG-guard defect — UPGRADED (real consequence, not a nit).** Verified
the server consumption path:

- The emit loop stops output at the first accepted EOG
  (`tools/server/server-context.cpp:4939` → `process_token` returns false at
  `:2208` → final response + `slot.release()`), so post-EOG tokens are never
  streamed to the client.
- BUT the retained context is polluted anyway: `:4918`
  (`slot.prompt.tokens.insert({ids.begin(), ids.end() - 1})`) inserts **all**
  accepted tokens — including the post-EOG ones the guardless `_rejection`
  sampler kept sampling — into `slot.prompt.tokens` **before** the emit loop
  runs, and `:4889` (`common_speculative_accept`) commits the full accepted
  count to the MTP/draft state. Net effect: every long multi-turn session
  accumulates phantom tokens the turn never emitted, plus draft-model state
  desync. This is a genuine degradation vector for long server sessions; a
  direct CJK *source* remains unproven (see rank 2), but the bug makes long
  sessions drift.
- The batch loop itself breaks on EOG (`:4939`), so the earlier "post-EOS
  garbage streamed" concern is not what happens — the damage is the retained
  prompt + MTP count, which is worse for the owner's symptom because it
  persists across turns invisibly.

**Rank 5 (sibling fork diff) — the comparison is NOT apples-to-apples.**
Sibling HEAD `e194cbddc` (2026-09-06) has the qwen3 arch support and an
identical `common/reasoning-budget.cpp` (diff-0), but its sampler set lacks
`common_sampler_sample_and_accept_n_rejection` entirely (predates it), and no
`draft-mtp` path exists in its server. The sibling therefore cannot run the
owner's server configuration at all; "the sibling server was clean" carries no
weight. The structural divergence is exactly the fork's new server-only
`_rejection` path — the same code under the rank-1 defect.

**Fix dispatched** (code-only, flags untouched): (A) primary — replicate the
`:706-708` EOG break inside `common_sampler_sample_and_accept_n_rejection`;
(B) defensive — truncate the `:4918` prompt insert at the first EOG with
rollback/replay-coherent bookkeeping (`n_rollback`, `slot.sampled`,
`common_speculative_accept` count). Implementation is edit-only at dispatch
time; build + gates serialize behind the currently-running fix lane.
