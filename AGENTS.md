# llama.cpp-turboquant-hip Agent Guide

## 🔑 Critical Setup
- **ROCm 6.1+ required** (CI uses `rocm/dev-ubuntu-22.04:7.2.1`)
- **Default ROCm paths** (auto-detected): `/opt/rocm` → `/home/stormrage/rocm-7.13-nightly`
- **Override**: `export ROCM_PATH=/path/to/rocm` before any build
- **GPU prep**: `source scripts/gpu_failback.sh` — saves/restores llama-server state, waits for VRAM to free
- **Model location**: `/home/stormrage/models/` (not `$HOME/models/`)
- **PATH order**: `rocm-7.13-nightly/bin` is before `/opt/rocm/bin` in system PATH. For cmake, ensure `/opt/rocm/bin` first:
  `export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"`
- **Build isolation (RPATH > RUNPATH)**: Always use `--disable-new-dtags` + `CMAKE_BUILD_RPATH_USE_ORIGIN`. Without these, `LD_LIBRARY_PATH` with other llama forks causes ABI mismatch → segfault/garbled output.

## ⚙️ Build

### Unified build script (recommended)
```bash
./scripts/build_rdna2.sh                      # Interactive ROCm selection + all optimizations
./scripts/build_rdna2.sh stable               # Production-safe, no experimental flags
./scripts/build_rdna2.sh baseline             # No RDNA2 optimizations
./scripts/build_rdna2.sh --clean --benchmark  # Clean + build llama-bench-rdna2 (hipcc)
./scripts/build_rdna2.sh --no-interactive     # Skip ROCm prompt
```
- Builds `llama-cli`, `llama-server`, `llama-bench` by default
- `--benchmark` also builds `llama-bench-rdna2` (standalone hipcc binary)
- Applies RPATH isolation automatically

### Manual CMake (gfx1030)
```bash
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1030 \
    -DCMAKE_BUILD_TYPE=Release \
    -DRDNA2_MOE_STREAM_V1=ON \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
cmake --build build --config Release -- -j 16
```

### ROCm presets (CMake options)
| Option | Path | Purpose |
|--------|------|---------|
| `-DROCM_PRESET=stable` | `/opt/rocm` (7.2.1) | Build — cmake works cleanly |
| `-DROCM_PRESET=nightly` | `/home/stormrage/rocm-7.13-nightly` | Runtime — newer hipblas/rocblas |
| (default) | auto-detect | Checks both paths |

Both use same LLVM/clang 23.0.0 — generated GPU code is identical. Build with stable, `LD_LIBRARY_PATH` to nightly at runtime.

### RDNA2 CMake options
| Option | Default | Effect |
|--------|---------|--------|
| `-DRDNA2_MOE_STREAM_V1` | OFF | MoE async stream pipeline (V1) — SLC cache-bypass GTT loads + semaphore signaling ✅ **IMPLEMENTED** |
| `-DGGML_RDNA2_BFE_DISPATCHER` | OFF | BFE `v_bfe_u32` for Q4_K dequant |

SLC=1 GTT optimization status:
- ✅ **Implemented**: `load_gtt_slc()`, `load_gtt_slc4()` with `global_load_dword slc` modifier
- ✅ **Verified**: 128 SLC-modified instructions emitted in assembly
- ✅ **Fixed**: `hipHostMalloc` + `hipHostMallocMapped` for semaphore memory
- ⏳ **Runtime validation**: Pending MoE model testing with rocprofv3 counters

Always-on: `RDNA2_OPT_V1` (compile definition for dequant kernel), gfx1030 LLVM tuning flags (`-mllvm -amdgpu-*`).

### Verification
```bash
readelf -d build/bin/llama-cli | grep RPATH   # Should show: $ORIGIN
ldd build/bin/llama-cli | grep llama          # All should resolve to build/bin/
grep "RDNA2_MOE_STREAM_V1" build/CMakeCache.txt
```

## 🧪 Testing & Validation
| Command | What it checks |
|---------|----------------|
| `build/bin/llama-cli --help` | Smoke test — should init GPU and exit cleanly |
| `cd build && ctest -L main -E "test-llama-archs" --verbose --timeout 900` | Unit tests |
| `./scripts/validate_hygiene.sh` | Compile + smoke test + VRAM leak (3 runs, >100MB delta = fail) |
| `./scripts/validate_qwen3_reasoning.sh` | RDNA2 flags don't break sampling/reasoning (needs Qwen3-35B model) |
| `./scripts/verify_kernel_dispatch.sh <model.gguf> [IQ4_XS,Q4_K_M,all]` | **Mandatory** before attributing perf deltas — verifies target kernel is actually dispatched |

**Known issue**: `tests/smoke_rdna2.cpp` is broken on ROCm 7.13 (uses removed `gcnArch`/`half` types). Fixed in 7.2.1.

## 🚩 Runtime Flags
| Env Var | Feature | Notes |
|---------|---------|-------|
| `RDNA2_MATMUL_OPT_V1=1` | LDS double-buffered matmul (MoE prefill) | Stabilized v0.3.1, +110–269% prefill |
| `RDNA2_ASYNC_ROUTING=1` | Async admin stream (MoE routing) | Experimental |

All flags are inert by default — fork runs identically to upstream when unset.

## 💾 KV Cache (TurboQuant) Settings
| Setting | Command | Use case |
|---------|---------|----------|
| Best overall | `-ctk turbo4 -ctv turbo2` | High context, low VRAM |
| Balanced | `-ctk turbo3 -ctv turbo2` | Default recommendation |
| Max quality | `-ctk turbo3 -ctv turbo3` | Highest fidelity |

## ⚙️ Key CLI Flags (35B MoE on 16GB)
- `-ngl 99` + `--ncmoe <N>`: Required for 35B MoE (offloads N expert layers)
- `--reasoning [on|off|auto]`: Qwen3 defaults to `auto` — interactive mode with `-p`
- `--chat-template none`: Raw completion mode
- `--no-display-prompt`: Suppress prompt echo in interactive mode
- `--repeat-penalty 1.1`: Prevent repetition loops
- `-fitt <MiB>`, `-fitc <tokens>`: Fit target margin and minimum context

## 🐛 Known Bugs & Gotchas
### Fixed
1. **Garbled output** (`vecdotq.cuh`): `RDNA2_FORCE_LDS_ALIGNMENT(addr) & ~0xF` macro forced 16-byte alignment on `get_int_b1/2/4`, causing all indices 0-3 to return same bytes. **DO NOT RE-INTRODUCE** alignment forcing.
2. **LDS double-buffer loop** (`mmq.cuh`): `load_tiles` hardcoded to `offset_x + kb0_start` instead of `offset_x + kb0`. Reverted to upstream pipeline.
3. **Build cross-contamination**: `LD_LIBRARY_PATH` with other llama forks loads wrong `.so` files. Fixed by RPATH isolation (see Build section).
4. **Idea D compiler flags falsely accused**: `-mllvm -amdgpu-*` flags were blamed for all-newline output; real cause was the `-n` flag.

### Open
5. **`-n` (count-tokens) produces all-newlines** with Qwen3-35B IQ4_NL. Workaround: omit `-n`, let model generate naturally; use `--no-display-prompt` to suppress echo.

### Gotchas
- `llama-server` state NOT saved/restored by build script — use `gpu_failback.sh` manually.
- **DO NOT** add alignment forcing to `get_int_b1/2/4` — see bug #1.
- Tile kernels with D≥576 are excluded from HIP builds (exceed 64KB local memory limit).
- `llama-cli -p <prompt>` enters interactive mode by default.
- `build.sh` (root) is a generic CMake wrapper — prefer `scripts/build_rdna2.sh`.
- MoE decode is slower than dense decode due to expert switching overhead.
- Benchmark baseline (RX 6800 XT, 35B MoE IQ4_XS, `-ngl 99 -ncmoe 33`): ~58 t/s prefill, ~34 t/s decode.

## 🔬 Active Research
See `opencode/agents/DEEP_ISA_MISSION.md` for ISA-level optimization roadmap (A-E).

Execution order: Phase 1 (D→A) → Phase 2 (B→C) → Phase 3 (E). VGPR budget ≤ 38.
