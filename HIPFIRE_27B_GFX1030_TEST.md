# hipfire v0.4.3.1 — Qwen 3.8 27B on gfx1030 (RX 6900 XT): build + test report

Date: 2026-10-25. GPU: gfx1030, 16 GB, ROCm 7.2.4. Card otherwise idle (496 MB used).

## 1. Build

- hipfire `v0.4.3.1` (commit `c24dd36d5`) was already installed at `~/.hipfire/bin/`
  (`hipfire`, `daemon`, `hipfire-tui`), built from the managed checkout.
- `hipfire doctor`: all PASS, `kernel_pack: no kernel pack for gfx1030: kernels JIT-compile on first use`.
  JIT works (`hipcc compiled (gfx1030, key …)` lines during first run); no prebuilt pack exists for RDNA2.
- Local source rebuild (`cargo build --release -p hipfire-cli` at tag `v0.4.3.1`) failed for an
  **environment** reason, not a hipfire one: the cargo registry cache on this host is corrupted
  (`clap_builder-4.6.0/src/util/id.rs` stored `Id::from(c\u{5}w)` instead of `Id::from(cow)`;
  earlier `futures-util` had a stray control byte and rustc SIGSEGV'd on two crates). Cache wiped with
  `rm -rf ~/.cargo/registry/{cache,src}`; the installed v0.4.3.1 binary is the same commit, all tests
  below use it.

## 2. Model

`hipfire fit` on an idle 16 GB card (VRAM free 15.5 GiB):

| tier | file | weights | + prefill scratch | headroom |
|---|---:|---:|---:|---|
| `qwen3.8:27b` (MQ4V2, default) | 15.66 GB | — | — | does not fit |
| `qwen3.8:27b-mq3` (MQ3V2) | 12.62 GB | 13.7 GiB | 1.4 GiB | 9 684 tok |
| `qwen3.8:27b-mq3-xt` | 11.78 GB | 12.8 GiB | 1.4 GiB | 36 865 tok |

Pulled `qwen3.8:27b-mq3` — the only 27B tier that fits at 8192 ctx, and the closest bit-width analogue
to our IQ3_S GGUF (3.44 bpw). Sidecars pulled automatically: `qwen3.8-27b.mtp` (226 MB),
`qwen3.8-27b-vision.hfq` (928 MB), `qwen38-27b-dflash-mq3.hfq` (985 MB). Models dir: 14 GB.

`verdict: UNDETERMINED` at every ctx — on gfx1030 hipfire uses the legacy KV backend and performs no
context admission check; the token headroom above is advisory only.

Arch from the container: `qwen3_5`, 5120 hidden, 64 layers, vocab 248320,
`full_attention_interval=4` (16 full-attention + 48 linear-attention/DeltaNet layers),
4 KV heads, head_dim 256, 1 MTP layer.

## 2b. What is and is not comparable

Both containers are the same **architecture and size**, but **not the same weights**:

| | our GGUF (`Swift-1.5-…-IQ3_S-mtp.gguf`) | hipfire `qwen3.8:27b-mq3` |
|---|---|---|
| arch string | `qwen35` | `qwen3_5` |
| blocks | 65 (64 + 1 MTP) | 64 + 1 MTP |
| hidden / ff | 5120 / 17408 | 5120 / 17408 |
| heads / kv / head_dim | 24 / 4 / 256 | 24 / 4 / 256 |
| layer mix | 48 linear (SSM) + 17 full-attn | 48 linear + 16 full-attn (`full_attention_interval=4`) |
| vocab / ctx | 248320 / 262144 | 248320 / 262144 |
| weights | **Swift-1.5 fine-tune** (GSQ-V1MIX imatrix) | **base qwen3.8:27b** |
| quant | IQ3_S, 3.546 bpw (3.594 over decode-relevant tensors) | MQ3V2, 104 B / 256 = **3.25 bpw** |
| params | 27.321 B (26.049 B read per token) | ~27.3 B |
| bytes streamed per decode step | **11.70 GB** | lower (~3.25 bpw bulk; embed/`output` at higher precision) |

Consequences:

- Shapes are identical, so FLOPs and parameter count match and **throughput** is comparable.
- Weights and quantization scheme differ, so **quality/coherence is not comparable**, and the
  decode numbers carry a systematic offset: we read ~5 % more bytes, hipfire pays a per-token
  activation rotation (`mq_rotate_x`) that IQ3_S does not. Decode is therefore **parity within
  quant-scheme differences** and must not be read as a kernel-quality ranking.
- Prefill (7.1×) is far outside anything a 0.34 bpw difference can explain; it is structural.

## 3. Results

`sanity`: `hipfire run qwen3.8:27b-mq3 --spec off "Say hello in one short sentence."` → `Hello there!`

`sanity with spec decode (default)`: **fails** —
`[validation retryable=false rolled_back=true] spec_step: HipError(0): qt=49 gemm_mq3g256v2_batched_lmhead: scalar fallback no v2 source`.
This is exactly hipfire issue **#771** (`gfx1030 - qwen3.8 + dflash issue`, closed) and it is still
present in v0.4.3.1 for MQ3V2. Workaround: `--spec off`, i.e. MTP/DFlash speculative decoding — which
the README advertises as default-on — is unusable on this card with the V2 tiers.

`hipfire bench --spec off`:

| shape | decode | prefill | TTFT | wall |
|---|---:|---:|---:|---:|
| ctx 128, 24-token prompt | **28.70 t/s** | (n/a, launch-bound) | 829 ms | 24.2 s |
| ctx 4096, 920-token prompt | **25.75 t/s** | **27.50 t/s** | **29 934 ms** | 3.7 s |

Reference, same model family, our tree (`build-hip-gfx1030`, GQA-6 decode kernel, IQ3_S 27.3B):

| metric | hipfire MQ3V2 | llama.cpp HIP | llama.cpp Vulkan |
|---|---:|---:|---:|
| decode @ d≈0 | 28.70 | 30.9 | 33.15 |
| decode @ 4k | 25.75 | ~28 | — |
| decode @ 32k | not runnable (VRAM) | 27.3 | 29.24 |
| decode @ 112k | not runnable (VRAM) | 22.4 | 23.23 |
| prefill @ 15 428-token prompt | not runnable (16 GB) | **556 t/s** | 425 t/s |
| prefill @ their 920-token prompt | 27.50 | n/a | 474 (measured at 4 501 tok) |
| TTFT @ ~920 tok | **29.9 s** | n/a (different prompt length) | n/a |

Decode is at parity (28.70 vs 30.9 t/s at d≈0) once the quantization-scheme offset above is
accounted for — hipfire streams fewer bytes but pays a per-token rotation, we stream ~5 % more
bytes. Neither number is evidence about kernel quality.

**Prefill: hipfire is ~20× slower** (27.50 vs 556 t/s) and that gap is structural (per-token GEMV
fallback, §4). Against Vulkan on identical flags and an identical 15 428-token prompt, **HIP is
~30 % faster** (556 vs 425 t/s; two runs each, order inverted, 547–554 t/s on the hotter re-run).
The 195.6 t/s / "2.4× behind Vulkan" figure that appeared in earlier drafts came from an
unrepresentative `llama-cli` run and is retracted — see `PLAN_GFX1030_DECODE_V2.md` §1.5.

## 4. Why prefill collapses — verified in source

`prefill` ≈ `decode` is the signature of the per-token `forward_scratch` GEMV path, i.e. no batched
GEMM at all. Two arch gates in `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` explain it:

1. `is_batchable_la()` admits `MQ3G256` on gfx10 through the scalar HFQ3 GEMM family
   (`gemm_qkv_hfq3g256`, `gemm_gate_up_hfq3g256`, `gemm_hfq3g256_residual`, …) — the arch list
   explicitly contains `gfx1030`.
2. But the V2 dtypes are gated to WMMA arches, and the unit test
   `qwen35_is_batchable_la_v2_family_gfx11_and_gfx12` asserts it:

```rust
for arch in ["gfx942", "gfx1010", "gfx1030", "gfx1103", "gfx1152"] {
    assert!(!is_batchable_la(DType::MQ3G256V2, arch), "MQ3V2 not on {arch}");
}
```

Every 27B registry tier that fits 16 GB is a **V2** tier (`MQ3V2` / `MQ4V2`), and V2 batched GEMMs
are gfx11/gfx12-only because they use WMMA. So on gfx1030 the fitting 27B model has no batched prefill
by design — the 30 s TTFT is not a misconfiguration or a `--spec off` artifact.

README expectation for `qwen3.5:27b` is "45 / 489 tok/s" (prefill/decode). Measured on gfx1030:
27.5 / 28.7. The README numbers presuppose WMMA (RDNA3/4) hardware.

## 5. Relevance for our gfx1030 work

- Attention is *not* where we lose: our GQA-6 kernel gives 22.4 t/s at 112k, a depth hipfire cannot
  even load on 16 GB, and decode is at parity with hipfire at shallow depth despite a different
  quantization scheme.
- hipfire's gfx1030 GEMV design (`kernels/src/gemv_hfq4g256.gfx1030.v2.hip`) uses
  `__launch_bounds__(32, 20)` — one warp per output row, 20 waves per SIMD, explicitly sized to
  ~51 VGPRs. Our `mul_mat_vec_q` in `ggml/src/ggml-cuda/mmvq.cu` declares
  `__launch_bounds__(nwarps*warp_size, 1)` and, for RDNA2, falls through to `calc_nwarps()=1` and
  `calc_rows_per_block()=1`. That is the opposite occupancy posture, and it matches the measured
  per-type MMVQ deficits against the Vulkan backend (Q4_K 1.6×, IQ4_XS 1.29×, IQ2_S 1.24×),
  where Q4_K is also the type of the entire 1.27 B-parameter `output.weight` lm_head.
- hipfire's own GEMV conclusion (issue #300: v1–v5 tie, decode is DRAM-bound) applies to *their*
  RDNA2 kernels run at 20-wave occupancy — not to ggml's untuned RDNA2 MMVQ, which is still below
  that ceiling.
- hipfire issue #533 (MQ2/MQ3 prefill) and #558 (`perf/gfx1030` branch) confirm upstream accepts
  these gfx1030 gaps as open work; our port is far ahead on prefill there.

## 6. Artifacts / housekeeping

- Models: `~/.hipfire/models/` (14 GB): `qwen3.8-27b.mq3`, `.mtp`, `-vision.hfq`, `qwen38-27b-dflash-mq3.hfq`.
- Bench source worktree: `~/.hipfire/build-v043` (tag `v0.4.3.1`, detached).
- Logs: `~/.hipfire/logs/{pull,build,run1,run2,bench1,bench2,llama-server-restore}.log`.
- Production llama.cpp server restored on `127.0.0.1:9931` (health 200, ctx 112128, `q8_0/q5_0` FA),
  log `~/.hipfire/logs/llama-server-restore.log`.
- `build/bin/llama-server` has the port baked in (9931); it does not listen on the default 8080.
