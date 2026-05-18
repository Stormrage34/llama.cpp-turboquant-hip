# llama.cpp-turboquant-hip Agent Guide

## 🔑 Critical Setup
- **ROCm 6.1+ required** (CI: `rocm/dev-ubuntu-22.04:7.2.1`)
- **Default ROCm paths**: `/opt/rocm` (stable) → `/home/stormrage/rocm-7.13-nightly` (runtime)
- **Override**: `export ROCM_PATH=/path/to/rocm` before build
- **GPU prep**: `source scripts/gpu_failback.sh` (saves/restores llama-server state)
- **Model location**: `/home/stormrage/models/`
- **PATH order**: For cmake, ensure `/opt/rocm/bin` is first: `export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"`
- **Build isolation (RPATH > RUNPATH)**: ALWAYS use `--disable-new-dtags` + `CMAKE_BUILD_RPATH_USE_ORIGIN` to prevent ABI mismatch/segfaults from other llama forks.

## ⚙️ Build
### Unified build script (Recommended)
`./scripts/build_rdna2.sh [stable|baseline|--clean --benchmark|--no-interactive]`
- Builds `llama-cli`, `llama-server`, `llama-bench`
- `--benchmark` also builds `llama-bench-rdna2` (standalone hipcc)

### Manual CMake (gfx1030)
```bash
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
    -DRDNA2_MOE_STREAM_V1=ON -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
cmake --build build --config Release -- -j 16
```

## 🧪 Verification & Testing
### Smoke & Hygiene
- `build/bin/llama-cli --help` (GPU init check)
- `./scripts/validate_hygiene.sh` (Compile + VRAM leak check)
- `./scripts/verify_kernel_dispatch.sh <model.gguf> [IQ4_XS,Q4_K_M,all]` (**Mandatory** for perf validation)

### Unit Tests
- `cd build && ctest -L main -E "test-llama-archs" --verbose --timeout 900`

## 🚀 Runtime & CLI
### RDNA2 Optimization Flags
- `RDNA2_MATMUL_OPT_V1=1`: LDS double-buffered matmul (MoE prefill)
- `RDNA2_ASYNC_ROUTING=1`: Async admin stream (MoE routing) - *Experimental*

### Key CLI (35B MoE on 16GB)
- `-ngl 99 --ncmoe <N>`: Required for 35B MoE offloading
- `--reasoning [on|off|auto]`: Qwen3 defaults to `auto`
- `--no-display-prompt`: Suppress echo (use if `-n` causes issues)
- `-fitt <MiB> -fitc <tokens>`: Target margin/context

## ⚠️ Critical Constraints & Gotchas
- **DO NOT** re-introduce alignment forcing in `vecdotq.cuh` (bug #1).
- **DO NOT** add alignment forcing to `get_int_b1/2/4`.
- **AVOID** `-n` (count-tokens) with Qwen3-35B IQ4_NL (causes all-newlines).
- **AVOID** root `build.sh`; use `scripts/build_rdna2.sh`.
- **AVOID** `llama-server` state loss; use `gpu_failback.sh` manually.
- **LIMIT** tile kernels with D≥576 (exceeds 64KB local memory limit).
- **EXPECT** slower MoE decode due to expert switching overhead.

## 🔬 Active Research
See `opencode/agents/DEEP_ISA_MISSION.md` for ISA-level optimization roadmap (A-E).
