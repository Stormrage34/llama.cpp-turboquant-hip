# Benchmarking llama.cpp — Reference Guide

Compiled from the official `llama-bench` README, `docs/development/token_generation_performance_tips.md`,
`tools/server/bench/README.md`, `tools/server/bench/speed-bench/README.md`,
`tools/batched-bench/README.md`, and community practice.

## 1. The tools

| Tool | Purpose | When to use |
|---|---|---|
| `llama-bench` | Micro/macro benchmark of pp (prompt processing) and tg (text generation) | **Default for comparing backends/builds** |
| `llama-cli` | End-to-end single run (incl. sampling/tokenization) | Quick sanity checks only |
| `llama-server` + k6 script | Server throughput under concurrent load | Production serving scenarios |
| `speed-bench.py` | Per-category eval, speculative-decoding acceptance | Spec decode comparison |
| `llama-batched-bench` | Batched decoding (multi-slot) | Multi-user / continuous batching |
| `test-backend-ops` (perf mode) | Single-op microbench (µs/run) | Kernel-level tuning (our main tool) |

`llama-bench` does **not** include tokenization or sampling time — only the model forward pass.

## 2. `llama-bench` parameters

### Test-shape parameters
| Flag | Default | Meaning |
|---|---|---|
| `-p, --n-prompt` | 512 | Prompt tokens for pp test |
| `-n, --n-gen` | 128 | Generated tokens for tg test |
| `-pg <pp,tg>` | — | Combined pp+tg test |
| `-d, --n-depth` | 0 | Prefill KV cache with N tokens before measuring (long-context decode) |
| `-b, --batch-size` | 2048 | Logical batch size |
| `-ub, --ubatch-size` | 512 | Physical micro-batch (≤ batch-size) |

### Model / offload
| Flag | Default | Meaning |
|---|---|---|
| `-m, --model` | — | GGUF path |
| `-ngl, --n-gpu-layers` | -1 | Layers offloaded to GPU (-1 = all) |
| `-ctk, --cache-type-k` | f16 | KV cache K type (q8_0, q5_0, …) |
| `-ctv, --cache-type-v` | f16 | KV cache V type |
| `-fa, --flash-attn` | auto | Flash attention on/off/auto |
| `-fitt, --fit-target` | off | Fit model to device with N MiB margin (auto ngl) |
| `-fitc, --fit-ctx` | 0 | Min ctx for --fit-target |
| `-ncmoe, --n-cpu-moe` | 0 | MoE layers on CPU |
| `-ot, --override-tensor` | — | Per-tensor buffer overrides |

### Runtime
| Flag | Default | Meaning |
|---|---|---|
| `-t, --threads` | system | CPU threads (critical — see §4) |
| `-r, --repetitions` | 5 | Repeats per test (averaged) |
| `--delay` | 0 | Seconds between tests |
| `--no-warmup` | off | Skip warmup runs |
| `--prio` | 0 | Process priority |
| `--numa` | off | NUMA mode |
| `--poll` | 50 | Polling % |
| `--repack` | 1 | Repack tensors |
| `-lm, --load-mode` | auto | mmap/mlock/dio |

### Output
| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | md | md / csv / json / jsonl / sql |
| `-oe, --output-err` | none | Stderr format |
| `-v, --verbose` | off | Verbose |
| `--progress` | off | Progress indicators |

### Multi-value syntax
Every parameter accepts comma lists, repeated flags, and ranges:
```
-n 128,256,512          # comma list
-n 128 -n 256           # repeated
-ngl 10-35+5            # range with step
-d 0,4096,16384         # multiple depths
```
All combinations of multi-valued params are run (cartesian product).

## 3. Recommended benchmark setups

### 3.1 Quick decode check (our default)
```sh
llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 \
  -p 0 -n 128 -r 3 --min-time 300   # (min-time is our test-backend-ops addition)
```
- `-p 0` = pure decode (no pp).
- `-n 128` (not 64 — too short, dominated by warmup).
- `-r 3` minimum for stable numbers; 5 for publication.

### 3.2 Long-context decode sweep (our key scenario)
```sh
llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -fitt 128 \
  -p 0 -n 128 -d 0,8192,32768,112000 -r 2 -o jsonl
```
- `-d` prefills KV cache → measures decode at that depth.
- Use `-o jsonl` for machine parsing.

### 3.3 Prefill measurement
```sh
llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 \
  -p 512,2048,8192 -n 0 -r 3
```
- `-n 0` = pure prompt processing.

### 3.4 Combined (pp + tg in one run)
```sh
llama-bench -m model.gguf -ngl 99 -fa 1 \
  -pg 512,128 -r 3
```

### 3.5 CPU thread sweep (find optimal -t)
```sh
llama-bench -m model.gguf -n 0 -p 64 -t 1,2,4,8,16 -r 3
```
Start at 1 thread, double until throughput plateaus, then back off one step.

### 3.6 GPU layer sweep (find optimal -ngl)
```sh
llama-bench -m model.gguf -ngl 10,20,30,35,40 -p 512 -n 128 -r 2
```
Useful when VRAM is tight; find the knee where offloading stops helping.

### 3.7 Kernel-level microbench (test-backend-ops)
```sh
# perf mode: single-op timing
test-backend-ops -b ROCm0 -o MUL_MAT --perf --min-time 300

# filter to one type/shape
test-backend-ops -b ROCm0 -o MUL_MAT --perf --min-time 200 \
  2>&1 | grep -i "q4_K"
```
- `--min-time <ms>` controls the wall-clock floor (default 1000 ms).
- Use for isolating a single kernel (MMVQ, MMQ, FA) without model load overhead.

## 4. Best practices

### 4.1 Warmup
- llama-bench runs warmup by default. **Do not use `--no-warmup`** unless you specifically want cold-start numbers.
- First run after a build is often slower (kernel JIT, cache cold). Discard it.

### 4.2 Repetitions and stability
- Minimum `-r 3` for any conclusion; `-r 5` for publication.
- Report mean ± stddev. If stddev > 2 % of mean, something is wrong (thermal, background load, or too few tokens).
- Use `-o jsonl` and parse `avg_ts`/`stddev_ts` programmatically.

### 4.3 Thermal management
- GPU thermal throttling is the #1 source of noise on consumer cards.
- Let the card cool between runs (we saw 78 °C junction → 5 % drop).
- Monitor with `rocm-smi` / `nvidia-smi` during runs.
- For multi-run comparisons, interleave builds (A, B, A) to cancel thermal drift.

### 4.4 CPU thread count (critical for GPU users)
- Even with `-ngl 99`, CPU threads matter for sampling, graph scheduling, and CPU-fallback ops.
- Too many threads → oversaturation → **slower** (documented: 7 threads → 1.7 t/s, 1 thread → 5.5 t/s on a 7-core CPU).
- Rule: set `-t` to physical core count (not logical). If unsure, sweep `-t 1,2,4,8`.

### 4.5 Context depth (`-d`)
- Decode speed degrades with context (KV cache traffic). A single `-d 0` number is misleading.
- Always report at least two depths: 0 (short) and your target working depth.
- For 112k-class models, include `-d 0,8192,32768,112000`.

### 4.6 KV cache quantization
- `-ctk q8_0 -ctv q5_0` is a common production combo (quality/speed tradeoff).
- Benchmark with the **same** KV types you deploy with; mismatched types give misleading numbers.
- q5_0 V saves ~36 % V bandwidth vs f16 with negligible quality loss.

### 4.7 Flash attention
- `-fa 1` (on) is almost always faster on modern GPUs. `-fa auto` may pick off on some backends.
- Always specify explicitly for reproducibility.

### 4.8 Model loading
- Use `--repack 1` (default) for quantized models — repacking improves decode.
- `--no-mmap` can help on slow storage but costs RAM; `--mlock` prevents swapping.
- `-fitt` auto-fits to VRAM; verify the resulting ngl in the output.

### 4.9 Avoiding measurement traps
- **Don't benchmark with the server running** (VRAM contention) — stop it first.
- **Don't run other GPU workloads** during measurement.
- **Use `pkill -x llama-server`**, never `pkill -f` (matches the invoking shell).
- **Pin CPU frequency** if possible (`cpupower frequency-set -g performance`).
- **Disable CPU turbo** for reproducibility if needed.

### 4.10 Output format for analysis
- `md` — human-readable, good for reports.
- `csv` / `jsonl` — machine-parseable, good for scripting and CI.
- `sql` — import into SQLite for historical tracking.
- Always save raw output to a file for later re-analysis.

## 5. Server benchmarking (concurrent load)

### 5.1 k6-based (tools/server/bench/)
- Requires k6 built with xk6-sse extension.
- Uses ShareGPT dataset; measures pp/tg under concurrent users.
- Key env vars: `SERVER_BENCH_N_PROMPTS`, `SERVER_BENCH_MAX_TOKENS`, `SERVER_BENCH_MAX_PROMPT_TOKENS`, `SERVER_BENCH_MAX_CONTEXT`.
- Start server with `--parallel N` (slots) and match `--vus` to slot count.

### 5.2 speed-bench (speculative decoding eval)
- `tools/server/bench/speed-bench/speed_bench.py`
- Per-category throughput, latency, draft acceptance rate.
- Match `--concurrency` to server `--np`.
- Use `--output` + `speed_bench_compare.py` for A/B comparison.

### 5.3 batched-bench
- `llama-batched-bench` for multi-slot batched decoding.
- Two modes: shared prompt vs separate prompts.
- Measures `S_PP`, `S_TG`, `S` (total) at various batch sizes.

## 6. Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| CPU oversaturation | GPU bench slower than expected | Set `-t` to physical cores, or 1 |
| Thermal throttling | Numbers degrade over time | Cool between runs, interleave A/B |
| Too few tokens (`-n 64`) | High variance, warmup-dominated | Use `-n 128` or `-n 256` |
| Mismatched KV types | Incomparable numbers | Match `-ctk`/`-ctv` to deployment |
| Server running during bench | VRAM contention, OOM | Stop server first (`pkill -x`) |
| `--no-warmup` | First-run JIT/cache effects | Keep warmup on |
| Single depth only | Misses long-context degradation | Sweep `-d` |
| Comparing different builds without rebuild | Stale binary | Rebuild both, verify commit hash |
| Ignoring stddev | False conclusions | Require stddev < 2 % of mean |

## 7. Example: our gfx1030 workflow

```sh
# Full comparison (HIP vs Vulkan), ~2-3 min
scripts/gfx1030.py bench

# Quick HIP-only check, ~45 s
scripts/gfx1030.py bench --backend hip --depths "0"

# Manual equivalent
build-hip-gfx1030/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q5_0 -fitt 128 -p 0 -n 128 -d 0,32768 -r 3 -o jsonl
```

## 8. Interpreting results

- **pp (prompt processing)**: higher = better. Bottlenecked by compute (MMQ/GEMM). Our HIP is ~30 % faster than Vulkan.
- **tg (text generation)**: higher = better. Bottlenecked by memory bandwidth (MMVQ/GEMV). Our HIP is ~7 % behind Vulkan.
- **t/s at depth d**: decode speed at that context. Slope = µs/KV-token (memory traffic); intercept = fixed weight cost.
- **stddev**: if > 2 % of mean, re-run or investigate (thermal, background load).
- **avg_ns / stddev_ns** (jsonl): nanoseconds per token; convert to t/s = 1e9/avg_ns.
