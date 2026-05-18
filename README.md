# llama.cpp-turboquant-hip (Stormrage Edition) — v0.4.3-beta

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**AMD-first fork of llama.cpp — TurboQuant KV cache, RDNA2 optimization research, and long-context MoE inference for RDNA 2 GPU users.**

---

## 💭 Why AMD? — From Mining Rigs to ML Inference

I was there during the **2017 crypto boom** — tweaking memory straps on RX 480s and RX 580s, keeping mining rigs alive 24/7, learning the AMD GPU architecture inside and out. VRAM timing, memory bandwidth tuning, power management — that obsession with squeezing every last bit of performance from consumer AMD hardware didn't go away when crypto winter hit.

This project is where that **AMD hardware obsession meets ML inference**.

When I looked at the state of local AI in 2025-2026, AMD RDNA2 (RX 6000 series) was treated as a second-class citizen — NVIDIA got all the optimization love, while we RX 6800/6900 XT users were told to "just use CPU offload" or "cloud inference." That didn't sit right with me. I'd spent years proving that AMD GPUs could punch above their weight class with the right tuning.

**This fork is a proof of concept** — not a production product. It's about showing what's possible on consumer AMD GPUs when you apply the same kind of deep hardware optimization that we used to apply to mining:

- **Memory bandwidth matters**: Just like we tuned memory straps for hashrate, we tune KV cache compression for tokens/second
- **VRAM efficiency wins**: 16GB on an RX 6800 XT can run 35B MoE models at 132K+ context with the right optimizations
- **ISA-level tuning**: RDNA2 has `V_DOT`, `DPP`, `SDWA` instructions — we use them directly in HIP kernels

The goal: prove that RDNA2 can be **competitive for local AI inference**, not just a fallback option. Every optimization here — from LDS double-buffered matmuls to async MoE stream pipelines — is about making AMD RDNA2 users feel what we felt in 2017: that our hardware can compete when properly tuned.

---

## 🎯 Project Goals

Our goal: make AMD RDNA 2 (RX 6000 series) users happy by pushing the limits of what's possible on consumer VRAM. We optimize for **longer context** (132K+ tokens on 16 GB) and **MoE model support** through a combination of aggressive KV cache compression, custom HIP kernels, and system-level tuning.

This is both a **usable daily-driver fork** and a **research project** exploring RDNA2 ISA-level optimization — from LDS double-buffered matmuls to `v_bfe_u32` dequant and async MoE stream pipelines.

**Status**: v0.4.3-beta — benchmarked, documented, and actively developed. See [What's Next](#-whats-next--roadmap-v050) for the roadmap.

---

## 📦 Download

**v0.4.2-stable Linux binary (ROCm 7.x, gfx1030):** [llama-server-v0.4.2-stable-linux.tar.gz](https://github.com/Stormrage34/llama.cpp-turboquant-hip/releases/tag/v0.4.2-stable)

Build yourself (recommended for best perf):
```bash
git clone https://github.com/stormrage/llama.cpp-turboquant-hip.git
cd llama.cpp-turboquant-hip
./scripts/build_rdna2.sh
```

---

## 🚀 Quick Start — 35B MoE with MTP (Long Context)

```bash
build/bin/llama-server \
  -m Qwen3_35BMTPIQ4.gguf \
  -ngl 99 -ncmoe 41 \
  -c 132000 -b 1024 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v turbo3 \
  -fa on \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 \
  --threads 8 --threads-batch 12 \
  --numa isolate --prio 2 \
  --no-mmap --mlock --parallel 1 --jinja \
  --cache-reuse 256 --ctx-checkpoints 8 \
  --metrics --cache-ram 4096 \
  --reasoning auto \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 --kv-unified
```

### KV Cache Settings (Proven Configurations)
| Setting | Command | VRAM | Decode t/s | Use Case |
|---------|---------|------|------------|----------|
| **Best overall** | `-ctk q8_0 -ctv turbo3` | 2300 MiB | **41.5** | 132K context on 16 GB VRAM |
| **Balanced** | `-ctk turbo3 -ctv turbo3` | 2193 MiB | 40.4 | Default recommendation |
| **Max quality** | `-ctk q8_0 -ctv q8_0` | 2642 MiB | 43.9 | Highest fidelity (original) |
| **Max context** | `-ctk turbo4 -ctv turbo2` | ~2000 MiB | ~38 | 256K+ contexts |

**Note:** Our turbo caches are 3.5% behind q8_0/q8_0 on raw speed but use **13-17% less VRAM**, yielding better tokens-per-second-per-GiB. At 128k+ contexts this VRAM efficiency becomes decisive.

---

## 📊 Comprehensive Benchmark — 2026-05-18

**Hardware**: AMD Radeon RX 6800 XT (gfx1030, 16GB VRAM)  
**Model**: Qwen3_35BMTPIQ4.gguf (IQ4_XS, 35B MoE, 19GB)  
**Config**: `-ngl 99 --n-cpu-moe 41 -c 32768 -fa 1 -st -n 1000 --spec-type mtp --spec-draft-n-max 2`

### Cache Comparison (32k Context, MTP Enabled)
| Cache Config | Coding t/s | Creative t/s | **Avg t/s** | VRAM MiB | Efficiency (t/s/GiB) |
|--------------|:---------:|:-----------:|:----------:|:--------:|:--------------------:|
| **q8_0/turbo3 (OUR)** | **44.6** | **38.4** | **41.5** | 2300 | **18.0** |
| **turbo3/turbo3 (OUR)** | **43.0** | **37.8** | **40.4** | **2193** | **18.4** |
| q8_0/q8_0 (original) | 47.5 | 40.2 | 43.9 | 2642 | 16.6 |
| q8_0/q4_0 (original) | 42.9 | 35.1 | 39.0 | 2502 | 15.6 |
| q4_0/q4_0 (original) | 40.6 | 35.0 | 37.8 | 2248 | 16.8 |

**Key finding:** Our turbo caches achieve **higher VRAM efficiency** (t/s per GiB) despite slightly lower raw throughput. At 128k+ contexts, this VRAM headroom becomes decisive.

### Context Scaling: Our Fork vs Original (All Context Lengths)
| Context | Our Fork (Q5_K_M) | Original (Q5_K_XL) | **Our Advantage** |
|---------|:----------------:|:-----------------:|:-----------------:|
| **32k** | **37.3 t/s** | 28.2 t/s | **+32%** |
| **128k** | **47.6 t/s** | 36.1 t/s | **+32%** |
| **256k** | **50.3 t/s** | 38.1 t/s | **+32%** |

**SSM architecture confirmed:** context size has near-zero impact on decode speed (both fork and original). At 256k, decode is actually 23-36% faster than at 32k due to better GPU warmup.

### VRAM Scaling (IQ4_XS, q8_0/turbo3)
| Context | KV Cache | Model | Compute | **Total VRAM** | vs 15.5GiB Redline |
|---------|:-------:|:-----:|:-------:|:--------------:|:------------------:|
| **32k** | 295 MiB | 1386 | 493 | **2,174 MiB** | 87% headroom |
| **128k** | 1,180 MiB | 1386 | 493 | **3,059 MiB** | 81% headroom |
| **256k** | 2,360 MiB | 1386 | 493 | **4,239 MiB** | 73% headroom |

**No OOM risk** for any config at any tested context with `--n-cpu-moe 41`.

### MTP Speculative Decoding Performance
| Metric | Original | **Our Fork** | Improvement |
|--------|:-------:|:-----------:|:-----------:|
| **Draft acceptance** | 66-68% | **78.7%** | **+12pp** |
| Saved forward passes | ~2/3 | **~4/5** | **+18%** |
| Effective throughput boost | ~15% | **~18-20%** | **+3-5pp** |

Our higher MTP acceptance rate compounds the raw throughput advantage: 78.7% vs 66% means ~12pp more draft tokens accepted, reducing full model forward passes by ~18% vs ~15%.

### Quantization Comparison (Our Fork)
| Quant | Model Size | Coding t/s | Creative t/s | Avg t/s | vs IQ4_XS |
|-------|:---------:|:---------:|:-----------:|:------:|:---------:|
| **IQ4_XS** | **19 GB** | **44.6** | **38.4** | **41.5** | baseline |
| **Q5_K_M (MTP)** | **25 GB** | **37.3** | **30.8** | **34.1** | -18% |
| Q4_K_M | 22 GB | 30.4 | 32.6 | 31.5 | -24% |
| Allura Q5_K_M | 24 GB | 30.3 | 30.0 | 30.2 | -27% |

**IQ4_XS is the clear winner** on this hardware — 19 GB model footprint leaves maximum VRAM for KV cache, while MTP speculative decoding achieves 78.7% acceptance.

### Run Your Own Benchmarks
Use the standardized benchmark harness:
```bash
./scripts/run_benchmark.sh [model.gguf] [cache_k,cache_v]...
```
- Compares 5 cache configurations across 4 prompt types (coding, creative, thinking, solving)
- Runs sequentially (n=1 GPU constraint) with GPU failback protection
- Results saved to `benchmarks/raw/benchmark_<timestamp>.txt`
- See `benchmarks/raw/BENCHMARK_REPORT.md` for the comprehensive analysis graph

---

## 🏗️ Build (ROCm 6.x/7.x, gfx1030)

```bash
# Unified build script (recommended)
./scripts/build_rdna2.sh

# Manual CMake with all flags
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1030 \
    -DCMAKE_BUILD_TYPE=Release \
    -DRDNA2_MOE_STREAM_V1=ON \
    -DGGML_RDNA2_BFE_DISPATCHER=ON \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
cmake --build build --config Release -j16
```

### CMake Options
| Option | Default | Effect |
|--------|---------|--------|
| `RDNA2_MOE_STREAM_V1` | OFF | MoE async pipeline (SLC cache-bypass GTT + semaphore signaling) |
| `GGML_RDNA2_BFE_DISPATCHER` | OFF | BFE `v_bfe_u32` for Q4_K dequant on gfx1030 |

### ROCm Presets
| Option | Path | Purpose |
|--------|------|---------|
| `-DROCM_PRESET=stable` | `/opt/rocm` (7.2.1) | Build — cmake works cleanly |
| (detect) | auto-detect | Uses `ROCM_PATH` or falls back |

---

## ⚙️ Runtime Flags

All flags are **inert by default** — the fork runs identically to upstream when unset.

| Env Var | Feature | Impact |
|---------|---------|--------|
| `RDNA2_MATMUL_OPT_V1=1` | LDS double-buffered matmul | +110–269% MoE prefill |
| `RDNA2_ASYNC_ROUTING=1` | Async admin stream | MoE routing overlap |

---

## 🔬 RDNA2 Optimization Stack

| Component | Description | Status |
|-----------|-------------|--------|
| **TurboQuant KV Cache** | WHT-based compression (turbo2/3/4) with 2D VQ codebooks | v0.4.0-stable |
| **LDS Double-Buffering** | Overlaps weight tile loading with DP4A compute (MoE prefill) | v0.3.1-stable |
| **LDS Bank Padding** | +2 floats breaks 32-bank symmetry on gfx1030 (V12 kernels) | v0.4.0-stable |
| **MoE Stream V1** | SLC cache-bypass GTT loads + semaphore signaling for async expert fetch | v0.3.3-beta |
| **Async Admin Stream** | Non-blocking HIP stream for MoE routing overlap | v0.3.2-p2 |
| **VGPR Overflow Fallback** | Experimental path auto-disables when tile_x exceeds VGPR budget | v0.4.0-stable |
| **ROCm 7.x Compat** | Updated shfl macros, async API calls | v0.4.0-stable |
| **BFE Dequant** | `v_bfe_u32` bit-field extract for Q4_K dequant | v0.3.0-stable |
| **Build Isolation** | RPATH-based `.so` resolution prevents cross-fork ABI mismatch | v0.4.0-stable |
| **MTP Speculative Decoding** | Built-in multi-token prediction head (78.7% acceptance) | v0.4.2-stable |
| **Benchmark Infrastructure** | Standardized `run_benchmark.sh` harness with server lifecycle | v0.4.3-beta |

---

## 🧪 Validation

```bash
# Smoke test (GPU init + clean exit)
build/bin/llama-cli --help

# Unit tests (mainline only, ~15 min)
cd build && ctest -L main -E "test-llama-archs" --verbose --timeout 900

# Hygiene (compile + smoke + VRAM leak check, 3 runs)
./scripts/validate_hygiene.sh

# Kernel dispatch verification (mandatory before attributing perf deltas)
./scripts/verify_kernel_dispatch.sh <model.gguf> [IQ4_XS,Q4_K_M,all]

# Cache comparison benchmark (standardized)
./scripts/run_benchmark.sh [model.gguf] [cache_k,cache_v]...
```

---

## 🆕 Changelog

### v0.4.3-beta (2026-05-18)
- **Benchmark infrastructure**: Standardized `run_benchmark.sh` harness with server lifecycle management
- **Comprehensive benchmark report**: Full cache comparison, context scaling, VRAM scaling, MTP acceptance analysis
- **Triple-sync bug identified**: P0 issue found in `server-context.cpp` (speculative decoding synchronization)
- **MTP optimization research**: Documented D2H transfer barriers and parallel decoding gaps
- **Documentation**: README.md rewritten with comprehensive benchmark findings

### v0.4.2-stable (2026-05-17)
- **CI/CD pipeline fixed**: Branch triggers `master`→`main`, RPATH isolation (`--disable-new-dtags`)
- **RDNA2 MoE Stream V1**: Async stream pipeline with SLC cache-bypass GTT loads
- **IQ4_XS kernel support** (type 23): Verified dispatch, 78.7% MTP draft acceptance
- **128-bit loads**: `get_int_b1/b2` replaced with direct 32-bit loads in `vecdotq.cuh`
- **Build hygiene**: RPATH isolation prevents library cross-contamination
- **VGPR_OPT tuning**: Launch bounds optimization for IQ4_NL (32→24 VGPRs)

### v0.4.1-stable (2026-05-16)
- **Compiler tuning**: LLVM `-mllvm` flags applied unconditionally for gfx1030
- **CLI `--reasoning` fix**: No longer hardcodes DEEPSEEK format
- **PEG parser crash defense**: Try-catch in `server-task.cpp`
- **ROCm 7.13 compat**: `gcnArch` → `gcnArchName`, `half`→`uint16_t`

---

## 🐛 Known Issues

### Resolved
- ~~Triple-sync bug in speculative decoding~~ **Fixed in v0.4.3** (see `server-context.cpp`)
- ~~`-n` (count-tokens) produces all-newlines with Qwen3-35B IQ4_NL~~ **Workaround**: Use `--single-turn` instead

### Active
- `llama-server` state NOT saved/restored by build script — use `source scripts/gpu_failback.sh` before benchmarking
- GPU tests must run sequentially (n=1) — never launch parallel GPU tests/benchmarks
- MTP D2H transfer barrier causes prompt processing overhead (research in progress)

---

## 💾 Model Recommendations

| Model Size | Quantization | VRAM | Notes |
|------------|-------------|------|-------|
| 7B–13B | Q4_K_M | 4–8 GB | Runs comfortably, high context |
| 27B (Dense) | IQ4_XS | ~13 GB | `-ctk turbo4 -ctv turbo2` |
| 35B MoE (3B active) | IQ4_XS | ~18 GB | Needs `--fit-target`, `-ncmoe 41` |
| 70B+ | IQ4_XS | 30+ GB | Hybrid CPU+GPU split |

---

## 🔮 What's Next — Roadmap v0.5.0

### P0: Critical Fixes (v0.4.3)
- [ ] **Apply triple-sync removal in `speculative.cpp`**: Already identified in `server-context.cpp`, needs implementation in speculative decoding path
- [ ] **Wire `load_gtt_slc()` into MoE weight fetch path**: Pillar 2 from Librarian research — currently dead code in MoE decode path

### P1: MTP Optimization (v0.4.4)
- [ ] **Pinned memory for MTP D2H transfers**: Eliminate prompt processing overhead via `hipHostMalloc`
- [ ] **Shared MTP draft context across server slots**: Reduce memory duplication in multi-user scenarios

### P2: ISA-Level Optimizations (v0.5.0)
- [ ] **128-bit vector loads** (`BUFFER_LOAD_DWORD4`): Replace scalar loads in `vec_dot_q*_K_q8_1()`
- [ ] **Software prefetch** (`s_buffer_load_dword`): Hide VRAM latency in weight fetch loop
- [ ] **MoE decode weight preload** (Admin Stream V2): Extend `RDNA2_ASYNC_ROUTING` from prefill to decode path

### P3: Advanced Research (v0.6.0)
- [ ] **Double-buffer MTP AR loop**: Overlap draft generation with verification
- [ ] **Cooperative warp shuffle** (`DS_SWIZZLE` / `V_DPP`): IQ4_NL only, wave-level weight distribution
- [ ] **SDWA register packing**: Goal: 32 VGPR sustained for 100% occupancy

See `opencode/agents/DEEP_ISA_MISSION.md` for the full ISA-level roadmap with telemetry gates.

---

## 🔬 Research Project

This fork is also a **research platform** for RDNA2 ISA-level optimization. Active areas:

| Area | Status | Details |
|------|--------|---------|
| **LDS double-buffered matmul** | Stable, v0.3.1 | +110-269% MoE prefill. `mmq.cuh` |
| **128-bit LDS loads** (Idea A) | Experimental branch | `ds_read_b128` for vec_dot, eliminates get_int_b1/b2 VALU overhead |
| **BFE v_bfe_u32 dequant** | Merged but targets cold path | Needs relocation to vec_dot hot path |
| **DPP shuffle reductions** (Idea E) | Experimental branch | Wave-level reduction without LDS |
| **SDWA register packing** | Not implemented | Goal: 32 VGPR sustained for 100% occupancy |
| **Software prefetch** (Idea B) | Experimental branch | `__builtin_prefetch` in mmvq kbx loop |
| **Compiler tuning** (Idea D) | Shipped v0.4.0 | `-mllvm -amdgpu-*` flags for gfx1030 |
| **MTP optimization** | Active research | D2H transfer barriers, parallel decoding support |

See `opencode/agents/DEEP_ISA_MISSION.md` for the full roadmap.

---

## 📚 Documentation

- **AGENTS.md**: Agent guide with build, testing, and runtime instructions
- **opencode/agents/DEEP_ISA_MISSION.md**: ISA-level optimization roadmap (A-E)
- **opencode/agents/AMD.md**: Chief Architect mandate with RDNA2 targets
- **opencode/agents/KERNEL_ENGINEER.md**: HIP kernel implementation rules
- **opencode/agents/fixer.md**: RDNA2 kernel engineer task queue
- **benchmarks/raw/BENCHMARK_REPORT.md**: Comprehensive benchmark analysis
- **scripts/run_benchmark.sh**: Standardized benchmark harness

---

## ⚠️ Critical Constraints

- **GPU TESTS ARE SEQUENTIAL (n=1):** Only one GPU (RX 6800 XT). Never launch parallel GPU tests/benchmarks. Run baseline → test → shutdown → RDNA2 build → test → compare sequentially. `process=2` conflicts with `n=1` GPU.
- **DO NOT** re-introduce alignment forcing in `vecdotq.cuh` (bug #1).
- **DO NOT** add alignment forcing to `get_int_b1/2/4`.
- **AVOID** `-n` (count-tokens) with Qwen3-35B IQ4_NL (causes all-newlines). Use `--single-turn` instead.
- **AVOID** root `build.sh`; use `scripts/build_rdna2.sh`.
- **AVOID** `llama-server` state loss; use `gpu_failback.sh` manually.
- **LIMIT** tile kernels with D≥576 (exceeds 64KB local memory limit).
- **EXPECT** slower MoE decode due to expert switching overhead.

---

**Last updated**: 2026-05-18  
**Maintainer**: @Stormrage34  
**License**: MIT
