# llama.cpp-turboquant-hip (Stormrage Edition) — v0.4.2-stable

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**AMD-first fork of llama.cpp — TurboQuant KV cache, RDNA2 optimization research, and long-context MoE inference for RDNA 2 GPU users.**

Our goal: make AMD RDNA 2 (RX 6000 series) users happy by pushing the limits of what's possible on consumer VRAM. We optimize for **longer context** (132K+ tokens on 16 GB) and **MoE model support** through a combination of aggressive KV cache compression, custom HIP kernels, and system-level tuning.

This is both a **usable daily-driver fork** and a **research project** exploring RDNA2 ISA-level optimization — from LDS double-buffered matmuls to `v_bfe_u32` dequant and async MoE stream pipelines.

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
  -ngl 99 -ncmoe 32 \
  -c 132000 -b 1024 -ub 2048 \
  --cache-type-k turbo4 --cache-type-v turbo2 \
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

### KV Cache Settings
| Setting | Command | Use Case |
|---------|---------|----------|
| Best overall | `-ctk turbo4 -ctv turbo2` | 132K context on 16 GB VRAM |
| Balanced | `-ctk turbo3 -ctv turbo2` | Default recommendation |
| Max quality | `-ctk turbo3 -ctv turbo3` | Highest fidelity |

---

## 📊 Benchmark — v0.4.2-stable

**Hardware**: RX 6800 XT (16 GB VRAM) · **Model**: Qwen3-35B IQ4_XS + MTP
**Server flags**: `-ngl 99 -ncmoe 39 -c 128000 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75`

| Metric | Value | Notes |
|--------|-------|-------|
| Prefill | 295–405 t/s | Depends on prompt complexity |
| **Decode (MTP)** | **~39 t/s** | Consistent across runs |
| MTP draft accept | **78.7%** | 3,711/4,716 accepted |
| VRAM usage | <15.5 GB | Within 15.5GB redline |

**IQ4_XS kernels verified** (type 23) — dispatch confirmed via `verify_kernel_dispatch.sh`.

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
```

---

## 🆕 v0.4.2-stable Changelog

- **Benchmark infrastructure**: Automated `benchmark_qwen3.py` script with server lifecycle management
- **Chief Engineer report**: Performance validation, hardware safety checks, VRAM monitoring
- **CI fixes**: Branch triggers `master`→`main`, RPATH isolation (`--disable-new-dtags`)
- **RDNA2 MoE Stream V1**: Async stream pipeline with SLC cache-bypass GTT loads
- **IQ4_XS kernel support** (type 23): Verified dispatch, 78.7% MTP draft acceptance
- **128-bit loads**: `get_int_b1/b2` replaced with direct 32-bit loads in `vecdotq.cuh`
- **Build hygiene**: RPATH isolation prevents library cross-contamination
- **Development plan**: `opencode/plan.md` with benchmark targets and safety constraints

---

## 🐛 Known Issues

- `-n` (count-tokens) produces all-newlines with Qwen3-35B IQ4_NL — omit `-n`, use `--no-display-prompt` instead
- `llama-server` state NOT saved/restored by build script — use `source scripts/gpu_failback.sh` before benchmarking

---

## 💾 Model Recommendations

| Model Size | Quantization | VRAM | Notes |
|------------|-------------|------|-------|
| 7B–13B | Q4_K_M | 4–8 GB | Runs comfortably, high context |
| 27B (Dense) | IQ4_XS | ~13 GB | `-ctk turbo4 -ctv turbo2` |
| 35B MoE (3B active) | IQ4_XS | ~18 GB | Needs `--fit-target`, `-ncmoe` |
| 70B+ | IQ4_XS | 30+ GB | Hybrid CPU+GPU split |

---

## 🔬 Research Project

This fork is also a **research platform** for RDNA2 ISA-level optimization. Active areas:

| Area | Status | Details |
|------|--------|---------|
| **LDS double-buffered matmul** | Stable, v0.3.1 | +110-269% MoE prefill. `mmq.cuh` |
| **128-bit LDS loads** (idea-a) | Experimental branch | `ds_read_b128` for vec_dot, eliminates get_int_b1/b2 VALU overhead |
| **BFE v_bfe_u32 dequant** | Merged but targets cold path | Needs relocation to vec_dot hot path |
| **DPP shuffle reductions** (idea-e) | Experimental branch | Wave-level reduction without LDS |
| **SDWA register packing** | Not implemented | Goal: 32 VGPR sustained for 100% occupancy |
| **Software prefetch** (idea-b) | Experimental branch | `__builtin_prefetch` in mmvq kbx loop |
| **Compiler tuning** (idea-d) | Experimental branch | `-mllvm -amdgpu-*` flags for gfx1030 |

See `opencode/agents/DEEP_ISA_MISSION.md` for the full roadmap.
