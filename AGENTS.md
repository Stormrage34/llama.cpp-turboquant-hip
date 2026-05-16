# llama.cpp-turboquant-hip Agent Guide

## 🔑 Critical Setup (Easy to Miss)
- **ROCm 6.1+ required** (CI uses 7.2.1 container: `rocm/dev-ubuntu-22.04:7.2.1`)
- **Default ROCm paths** (auto-detected in order): `/opt/rocm` → `/home/stormrage/rocm-7.13-nightly`
- **Override ROCm version**: `export ROCM_PATH=/path/to/rocm` before any build
- **CMake-first build**: Run `cmake -B build` before any build scripts or GPU work.
- **GPU prep**: Source `scripts/gpu_failback.sh` before any GPU work — auto-saves/restores llama-server state, kills conflicting processes, waits for VRAM to free.
- **Model location**: Default is `/home/stormrage/models/` (not `$HOME/models/`).
- **PATH order matters**: System PATH has `rocm-7.13-nightly/bin` before `/opt/rocm/bin`. For cmake, ensure `/opt/rocm/bin` is first:
  `export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"`
- **Build isolation (RPATH > RUNPATH)**: Always use `--disable-new-dtags` + `CMAKE_BUILD_RPATH_USE_ORIGIN` to prevent library cross-contamination from other llama forks in `LD_LIBRARY_PATH`.

## ⚙️ Build & Run

### Unified Build Script (recommended)
```bash
./scripts/build_rdna2.sh                      # Interactive ROCm selection + build all targets
./scripts/build_rdna2.sh stable               # Production-safe, no experimental flags
./scripts/build_rdna2.sh baseline             # No RDNA2 optimizations
./scripts/build_rdna2.sh --clean --benchmark  # Clean rebuild + benchmark binary
./scripts/build_rdna2.sh --no-interactive     # Skip ROCm prompt, use default
```
- Prompts which ROCm to use (stable 7.2.1 vs nightly 7.13)
- Applies build isolation (RPATH) automatically
- Builds `llama-cli`, `llama-server`, `llama-bench` by default
- Use `--benchmark` to also build `llama-bench-rdna2` (hipcc-based)

### Manual CMake (gfx1030) — With Build Isolation
```bash
# Force RPATH (searched before LD_LIBRARY_PATH) to avoid ABI mismatch with other forks:
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1030 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
cmake --build build --config Release -- -j 16
```

### Binaries
- `build/bin/llama-cli`, `llama-server`, `llama-bench` — CMake-built
- `build/bin/llama-bench-rdna2` — built by `build_rdna2_llama.sh`

## 🐛 Known Bugs & Fixes
### 1. Garbled Output: `vecdotq.cuh` Alignment Mask (FIXED 2026-05-16)
**File:** `ggml/src/ggml-cuda/vecdotq.cuh`

The `RDNA2_FORCE_LDS_ALIGNMENT(addr) & ~0xF` macro in `get_int_b1`/`get_int_b2`/`get_int_b4`
forced 16-byte alignment on quantized weight reads. For indices 0-3 within a 16-byte chunk,
**all returned the same 4 bytes** instead of distinct rows, producing mixed-script garbage
output (e.g. `evelปุевичFTWARE全屏查看шта技术在`).

**DO NOT RE-INTRODUCE** alignment forcing on these accessors. The original byte-by-byte
reads handle unaligned loads safely on all GPU architectures.

### 2. `mmq.cuh` LDS Double-Buffer Loop Bug (FIXED 2026-05-16)
**File:** `ggml/src/ggml-cuda/mmq.cuh`

The LDS double-buffer path (`RDNA2_MATMUL_OPT_V1=1`) had `load_tiles` hardcoded to
`offset_x + kb0_start` inside the loop instead of `offset_x + kb0`. Reverted to upstream
pipeline pattern (prefetch before loop, swap at end).

### 3. Build Cross-Contamination (FIXED 2026-05-16)
**Problem:** `LD_LIBRARY_PATH` with `llama-mtp/build/bin` first causes turboquant binaries
to load mtp's `.so` files. Turboquant adds fields (`reasoning_format`, `enable_reasoning`)
to `common_params` — mtp's struct doesn't have them → ABI mismatch → segfault/garbled
output.

**Prevention:** The cmake flags above (`--disable-new-dtags`) force `RPATH` which the
loader searches before `LD_LIBRARY_PATH`. Verify with:
```bash
readelf -d build/bin/llama-cli | grep RPATH
# Should show: 0x000000000000000f (RPATH)  Library rpath: [$ORIGIN:]
ldd build/bin/llama-cli | grep llama
# All should resolve to build/bin/, never to mtp/
```

### 4. `-n` (count-tokens) Flag Produces All-Newlines (OPEN — 2026-05-16)
**Problem:** `llama-cli -n N -p "prompt"` outputs only `\n` characters instead of actual
tokens. Only observed with the Qwen3-35B IQ4_NL model. Workaround: omit `-n` and let the
model generate naturally (use `--no-display-prompt` to suppress prompt echo).

**Code path:** `llama-cli` uses `server_context` internally (cli.cpp line 57). The `-n` flag
sets `task_params.n_predict` which is processed by the server task loop. Suspect: interaction
between `n_predict` bound and the reasoning/MTP format that causes the generation loop to emit
only newline tokens. Needs debug logging on a real GPU run.

### 5. Idea D Compiler Flags Falsely Accused (CLEARED — 2026-05-16)
**Problem:** Previously suspected that `-mllvm -amdgpu-*` flags caused all-newline output.
Tested and cleared — output is correct with flags re-enabled. Real all-newline cause is the
`-n` flag (see #4). Flags re-enabled in `ggml/src/ggml-hip/CMakeLists.txt`.

## 🚩 RDNA2 Runtime Flags
| Env Var | Feature | Notes |
|---------|---------|-------|
| `RDNA2_ASYNC_ROUTING=1` | Async admin stream (MoE routing) | Experimental |

## 🧪 Testing & Validation
- **Smoke test (no model)**: `build/bin/llama-cli --help` — should init GPU and exit cleanly
- **Unit tests**: `cd build && ctest -L main -E "test-llama-archs" --verbose --timeout 900`
- **Hygiene check**: `./scripts/validate_hygiene.sh` — compile + smoke test + VRAM leak check (3 runs, >100MB delta = fail)
  - **Known issue:** `tests/smoke_rdna2.cpp` broken on ROCm 7.13 (uses removed `gcnArch`/`half` types)
- **Qwen3 reasoning check**: `./scripts/validate_qwen3_reasoning.sh` — validates RDNA2 flags don't break sampling/reasoning
- **Kernel dispatch verification** (mandatory before attributing perf deltas): `./scripts/verify_kernel_dispatch.sh <model.gguf> [IQ4_XS,Q4_K_M,all]`

## 💾 KV Cache (TurboQuant) Settings
- **Best overall** (high context, low VRAM): `-ctk turbo4 -ctv turbo2`
- **Balanced**: `-ctk turbo3 -ctv turbo2`
- **Max quality**: `-ctk turbo3 -ctv turbo3`

## ⚙️ Key CLI Flags (35B MoE Context)
- `-ngl 99` + `--ncmoe <N>`: Required for 35B MoE on 16GB VRAM (offloads N MoE expert layers)
- `--reasoning [on|off|auto]`: Qwen3 defaults to `auto` — starts interactive mode by default with `-p`
- `--chat-template none`: Raw completion mode (avoids template application issues)
- `--no-display-prompt`: Suppress prompt echo in interactive mode
- `--repeat-penalty 1.1`: Prevent repetition loops
- `-fitt <MiB>`, `-fitc <tokens>`: Fit target margin and minimum context

## 📦 Model VRAM Guide (RX 6800 XT, 16GB)
| Model | Quant | VRAM | Notes |
|-------|-------|------|-------|
| 7B–13B | Q4_K_M | 4–8 GB | Comfortable fit |
| 27B Dense | IQ4_XS | ~13 GB | Use `-ctk turbo4 -ctv turbo2` |
| 35B MoE (3B active) | IQ4_XS | ~18 GB | Requires `--ncmoe` or `-fitt/-fitc` for offload |
| 70B+ | IQ4_XS | 30+ GB | Hybrid CPU+GPU split recommended |

## 🔬 Active Research
See `opencode/agents/DEEP_ISA_MISSION.md` for ISA-level optimization roadmap (A-E):
- **A**: 128-bit vector loads (`BUFFER_LOAD_DWORD4`)
- **B**: Software prefetch (`s_buffer_load_dword`)
- **C**: MoE decode weight preload (admin stream + ACE)
- **D**: Compiler tuning (LLVM `-mllvm` flags)
- **E**: Cooperative warp shuffle (`V_DPP`/`DS_SWIZZLE`)

Execution order: Phase 1 (D→A) → Phase 2 (B→C) → Phase 3 (E). VGPR budget ≤38.

## ⚠️ Gotchas
- `llama-server` state not saved/restored by build script — use `gpu_failback.sh` manually.
- **DO NOT** add alignment forcing to `get_int_b1`/`get_int_b2`/`get_int_b4` — see Known Bugs above.
- Qwen3-35B (IQ4_NL) measured: **58.4 t/s prefill, 34.2 t/s decode** at `-ngl 99 -ncmoe 33` on RX 6800 XT.
- MoE decode slower than dense decode due to expert switching overhead.
- Tile kernels with D≥576 are excluded from HIP builds (exceed 64KB local memory limit).
- `build.sh` (root) is a generic CMake wrapper — prefer `scripts/build_rdna2_llama.sh`.
- `llama-cli -p <prompt>` enters interactive mode by default; for non-interactive use
  stdin pipe or `--interactive` mode.
