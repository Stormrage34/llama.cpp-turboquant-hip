# llama.cpp-turboquant-hip Agent Guide

## 🔑 Setup & Environment
- **ROCm**: `/opt/rocm` (stable/building) or `/home/stormrage/rocm-7.13-nightly` (runtime). Override: `export ROCM_PATH=...` or use `-DROCM_PRESET=stable|nightly|auto` (CMake only).
- **GPU prep**: `source scripts/gpu_failback.sh && gpu_acquire` — saves/restores llama-server state and waits for VRAM to free. Always call before GPU work.
- **PATH**: Put `/opt/rocm/bin` first for cmake: `export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"`
- **Models**: `/home/stormrage/models/`
- **Build isolation (RPATH > RUNPATH)**: Always use `--disable-new-dtags` + `CMAKE_BUILD_RPATH_USE_ORIGIN` to prevent ABI mismatch/segfaults from other llama forks on the same system.

## ⚙️ Build
### Unified script (the one true way)
```bash
./scripts/build_rdna2.sh [stable|baseline|--clean --benchmark|--no-interactive|--fast]
```
- Builds `llama-cli`, `llama-server`, `llama-bench`
- **`--fast`**: incremental rebuild (skip clean). Auto-detects stale binaries — warns if a shared lib is newer than its dependent binary.
- **`--benchmark`**: also builds `llama-bench-rdna2` (standalone hipcc, for rocprofv3 profiling)
- **`--no-interactive`**: auto-selects ROCm path; use in scripts
- **Modes**: `all` (LDS double-buffered matmul + cache swizzle, default), `stable` (no experimental features), `baseline` (no RDNA2 opts)

### 🔴 Stale binary rule
If you rebuild libraries (e.g., `libggml-hip`), ALL binaries linking them MUST also be rebuilt. Partial rebuilds cause SIGSEGV on startup. The `--fast` flag's stale-binary check catches this. If you see "GPU init OK then crash immediately", stale binaries are the #1 suspect.

### RPATH isolation flags (P0 for dev machines with multiple llama forks)
```
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${ROCM_PATH}/lib"
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${ROCM_PATH}/lib"
```
Without `--disable-new-dtags`, ld.so checks `LD_LIBRARY_PATH` before RPATH, pulling in incompatible libraries from other llama.cpp builds.

### RDNA2 CMake options
| Option | Default | Effect |
|--------|---------|--------|
| `RDNA2_MOE_STREAM_V1` | ON (in build script) | MoE async pipeline (SLC cache-bypass GTT + semaphore signaling) |
| `GGML_RDNA2_BFE_DISPATCHER` | OFF | BFE `v_bfe_u32` for Q4_K dequant on gfx1030 (cold path only) |
| `GGML_HIP_ROCWMMA_FATTN` | OFF | ROCm WMMA fused attention |
| `RDNA2_CACHE_SWIZZLE` | OFF (experimental) | IQ4_XS AoS→SoA swizzling for 128B cache line alignment |

## 🧪 Testing Pipeline
Order matters when doing a full verification:

### 1. Compile & smoke
```bash
build/bin/llama-cli --help           # GPU init check (HIP backend loads here)
./scripts/validate_hygiene.sh         # compile + VRAM leak check
```

### 2. Kernel dispatch verification
```bash
./scripts/verify_kernel_dispatch.sh <model.gguf> [IQ4_XS,Q4_K_M,all]
```
**Mandatory** before any perf claim. Confirms the expected dequant kernel path was actually invoked (catches cases where model quant doesn't match optimization target).

### 3. Inference parity
```bash
timeout 90 build/bin/llama-cli -m model.gguf -ngl 99 --n-cpu-moe 15 -p "Hello" --single-turn -n 50
```
Always use `--single-turn` + `timeout 90` to prevent interactive mode flood (all-newlines bug with Qwen3-35B).

### 4. Unit tests
```bash
cd build && ctest -L main -E "test-llama-archs" --verbose --timeout 900
```

### 5. Standardized benchmark
```bash
./scripts/run_benchmark.sh [model.gguf] [-r RUNS] [cache_k,cache_v...]
```
Compares turbo vs standard KV cache across 4 prompt types (coding, creative, thinking, solving). Default 5 runs per config. Config: `-ngl 99 --n-cpu-moe 41 -c 32768 -fa 1 --spec-type mtp --spec-draft-n-max 2 --single-turn -n 1000`. Results → `benchmarks/raw/`.

### 6. HIP smoke tests (ROCm sanity suite)
```bash
./scripts/test_therock_smoke.sh [--rocm-path PATH]
```
6 tests: hipcc kernel, Python suite, GPU init, IQ4_XS dispatch, VRAM leak, numerical parity. Requires `therock/` cloned + `.venv`.

### 7. Pre-commit hooks
Minimal: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8 (with flake8-no-print). Run via `pre-commit run --all-files`.

### 8. CR-018 1K-Stable Coherence Gate (Production Baseline)
**Note:** CR-022 supersedes CR-018 for all experimental branch validation. CR-018 remains the minimum gate for production/non-experimental builds.

**Authority:** @Oracle is the sole agent authorized to run this validation.
**Orchestrator is PROHIBITED from executing benchmarks or inference tests.**

**Protocol:**
- All model tests must generate **at least 1,000 tokens** (`-n 1000`).
- **Success:** Zero `<unused8>`/`<unused24>`/`<unusedN>` tokens, zero raw binary output, zero memory dumps.
- **Failure:** If any corruption occurs, abort immediately (do not complete 1000 tokens) and log the token index.
- **Pre-merge gate:** Any build that fails the 1K-stable check is forbidden from merging.

**Canonical test command:**
```bash
timeout 120 build/bin/llama-cli -m <model.gguf> -ngl 99 -ub 512 --single-turn \
    -p "Write a short story about a robot learning to paint." -n 1000 2>&1
```

**Config baseline:** `-ngl 99, -ub 512, --single-turn, -c 4096`

**Rollback:** If `n=1000` fails, revert to `n=512`. Only escalate after 5 clean `n=512` runs.

### 9. CR-022 5K-Coherence Gate (Mandatory)
**Authority:** @Oracle is the sole agent authorized to run this validation.
**Orchestrator is PROHIBITED from executing benchmarks or inference tests.**

**Protocol:**
- All experimental branch tests must generate at least **5,000 tokens** (`-n 5000`).
- Use **adversarial prompts** designed to stress KV cache saturation and expert routing (long-form narrative, multi-turn reasoning, code generation).
- **Differential validation:** Run the exact same prompt on the stable baseline branch first. Compare token-for-token output. Flag semantic divergence after tokens 500, 1000, 2000, 3000, 4000.
- **Fast-fail gate:** Run a 100-token pre-check before any full 5K run. If `<unused8>`/`<unused24>`/`<unusedN>` appears, abort immediately and log the token index.
- **Success:** Zero `<unused8>`/`<unused24>`/`<unusedN>` tokens across all 5000 tokens, zero raw binary output, zero semantic drift from stable baseline.
- **Failure:** Any corruption or semantic divergence fails the gate. Abort and report the token index of first divergence.

**Canonical test command:**
```bash
timeout 600 build/bin/llama-cli -m <model.gguf> -ngl 99 -ub 512 --single-turn \
    -p "<adversarial_prompt>" -n 5000 2>&1
```

**Config baseline:** `-ngl 99, -ub 512, --single-turn, -c 4096, -n 5000`

**5K Stable Coherence Gate:** Same as 5K, but against the stable production branch as ground truth anchor.

### Telemetry Requirement
Every `n=1000` stability run MUST be paired with a `rocprofv3` memory-alignment trace (PMC counters only, no `--hip-trace`). Counter configs are in `counters_*.json` at repo root.

## 📊 Profiling Infrastructure
### rocprofv3 (pure PMC counters only)
- Binary: `/home/stormrage/rocm-7.13-nightly/bin/rocprofv3` (83 KB)
- **Never use `--hip-trace` or `--kernel-trace`** — intercepts every `hipMalloc`/`hipMemcpy`, adding >300s for 20GB model loads.
- Use `rocprofv3 -i counters.json` with zero tracing flags.
- Multiple counter configs in repo root: `counters_*.json`, `counters_*.yaml`, `counters_*.txt`
- Need 300s timeout for model load + inference.
- For kernel dispatch verification: `rocprofv3 --kernel-trace` with `sqlite3` query (script handles this).

### GPU occupancy benchmarks
- `scripts/run_occupancy_benchmark.sh`
- `scripts/run_kernel_launch_profiling.sh` — CR-009
- Compare VGPR spills: `scripts/compare_vgpr_runs.sh`

## 🚀 Runtime
### Key CLI flags (35B MoE on 16GB)
- `-ngl 99 --n-cpu-moe <N>` — required for 35B MoE offloading (41 = all CPU experts)
- `--reasoning [on|off|auto]` — Qwen3 defaults to `auto`
- `--single-turn` — run one turn then exit (prevents interactive mode flood)
- `--spec-type mtp --spec-draft-n-max 2` — Multi-Token Prediction (built-in head, no separate draft model)
- `-fitt <MiB> -fitc <tokens>` — target VRAM margin/context

### Proven KV cache configs (35B MoE, 16GB VRAM)
| Setting | VRAM | Decode t/s | Use case |
|---------|------|------------|----------|
| `-ctk q8_0 -ctv turbo3` | 2300 MiB | 41.5 | **Best overall** — 132K context |
| `-ctk turbo3 -ctv turbo3` | 2193 MiB | 40.4 | Balanced (default) |
| `-ctk q8_0 -ctv q8_0` | 2642 MiB | 43.9 | Max quality |
| `-ctk turbo4 -ctv turbo2` | ~2000 MiB | ~38 | Max context (256K+) |

### RDNA2 env flags
| Env var | Feature | Notes |
|---------|---------|-------|
| `RDNA2_MATMUL_OPT_V1=1` | LDS double-buffered matmul (DEPRECATED) | +25% on Q5_K was from lds_bank_pad=2, not double-buffer. Double-buffer blocked by 64KB LDS on gfx1030. Now trait-gated: mmq_get_lds_bank_pad<type>() provides the gain. |
| `RDNA2_ASYNC_ROUTING=1` | Async admin stream (MoE routing) | Experimental |
| `RDNA2_V128_LOAD=1` | 128-bit int4 loads in vec_dot | ON by default (+4 VGPRs) |

## 🤖 Multi-Agent System
Proposal pipeline (opencode/):
```
Researching → Debating → Approved → Implementing → Verifying → Done → Archived
                REJECT ↕                   ↕ FAIL
```
- **Agent files** (docs): `opencode/agents/{fixer,explorer,oracle,council,...}.md` — documentation reference only
- **Agent definitions (active)**: `~/.config/opencode/oh-my-opencode-slim/*.md` — actual agent definitions loaded by plugin
- **Project state**: `opencode/project-state.md` (read first)
- **Execution plan**: `opencode/project-state.md` (see priority queue)
- **ISA roadmap**: `opencode/agents/DEEP_ISA_MISSION.md`

### 🔴 Council subagent workaround
Do NOT use built-in `councillor` type (dead provider). Use any working type (e.g., `fixer`) — the council.md frontmatter supplies the correct model/prompt:
```python
task(subagent_type="fixer", prompt=council_prompt)   # ✅
task(subagent_type="councillor", ...)                 # ❌ dead provider
```

### GPU gate
Always check before GPU work:
```bash
for proc in llama-server llama-cli llama-bench; do
    pgrep -x "$proc" >/dev/null && echo "BLOCKED: $proc running" && exit 1
done
```

### Subtask rules
- Never parallel GPU, GGUF >1GB, or HIP tasks (n=1 GPU)
- Every bash command in subtask prompt must include `timeout N` (N≤30 reads, N≤60 builds)
- Keep prompts under 30 lines
- Start with `task_id = null` for new threads; reuse only for same investigation

### 🔴 timeout grandchild propagation rule
When `timeout` is used with commands that spawn subprocesses (e.g., `rocprofv3` which spawns `llama-cli`), `timeout` only kills its direct child — grandchildren survive as orphans. **Always wrap the terminal process, not the wrapper:**

```bash
# WRONG — orphans survive when rocprofv3 is killed
timeout 120 rocprofv3 ... -- llama-cli ... | tail -40

# RIGHT — timeout kills the GPU-holding process directly
rocprofv3 ... -- timeout 120 llama-cli ... 2>&1 | tail -40
```

This is especially critical for rocprofv3, cmake, and any pipe-connected commands.

## ⚠️ Known Gotchas
- **GPU is n=1**: RX 6800 XT only. Never parallel GPU tests. `process=2` conflicts.
- **VRAM redline**: 15.5GB absolute max. 15.0GB yellow alert. IQ4_XS + `-ncmoe 41` @ 32k ≈ 2.2GB, @ 256k ≈ 4.2GB.
- **CR-016 Batch ceiling**: `n_ubatch > 1024` causes OOB memory reads into GGUF tensor padding, producing `<unused8>` token hallucinations. Clamped at kernel launch (`launch_mul_mat_q`) to 1024. To avoid truncation, set `--n-ub 1024` or lower. Verified stable at n_ubatch=512-1024 on gfx1030.
- **CR-018 Sanity Gate**: Builds failing the 1K-stable coherence test are BLOCKED from merge. The Oracle's test report is the single source of truth for pass/fail status.
- **MoE CPU offloading upstream bug** (CR-015): Long prompts + many tokens + any `--n-cpu-moe` → garbled output after ~1000 tokens. Verified on vanilla upstream. Workaround: all-GPU offload or short prompts.
- **DO NOT** re-introduce alignment forcing in `vecdotq.cuh` or `get_int_b1/2/4` (bug #1 history).
- **AVOID `-n` (count-tokens)** with Qwen3-35B IQ4_NL — causes all-newlines. Use `--single-turn`.
- **AVOID root `build.sh`** — use `scripts/build_rdna2.sh`.
- **LIMIT tile kernels**: D≥576 exceeds 64KB local memory limit.
- **GGML_ASSERT**: Not for runtime-recoverable conditions. Use graceful fallback.
- **LLAMA_FATAL_WARNINGS**: Enabled in CI release builds (`-Werror`).

## 📁 Directory Map
| Path | What |
|------|------|
| `ggml/src/ggml-cuda/` | HIP kernels: `vecdotq.cuh` (dequant+dot), `mmvq.cu` (matmul vec), `mmq.cuh` (LDS matmul), `common.cuh` (macros) |
| `src/` | llama.cpp core: `llama-context.cpp` (tensor sync), `speculative.cpp` (MTP), `llama-graph.cpp` (graph ops) |
| `scripts/` | Build (`build_rdna2.sh`), benchmark (`run_benchmark.sh`), GPU gate (`gpu_failback.sh`), dispatch verification, profiling |
| `opencode/` | AI agent config, proposals (CR-*), project-state, execution plans |
| `benchmarks/raw/` | Benchmark output, kernel dispatch traces |
| `.github/workflows/` | CI: `release.yml` (artifact build + GH release), `hip-quality-check.yml` (VGPR spill + Werror) |
| `tests/` | Test binaries (built with `LLAMA_BUILD_TESTS=ON`) |
| `counters_*.json/yaml/txt` | rocprofv3 counter profiles (many variants) |

## 🔬 Active Research (v0.5.0)
- **CR-008**: Perm chain optimization (`get_int_from_table_16`) — highest ROI at 70% of compute
- **CR-009**: Kernel launch overhead profiling (pure PMC counters)
- **CR-015**: Upstream MoE CPU offloading bug (4-fix patch applied, upstream PR pending)
- **ISA roadmap**: `opencode/agents/DEEP_ISA_MISSION.md` (verticals A-E)

### Closed: LDS Double-Buffered Matmul Optimization Phase
- **Discovery:** Verified that RDNA2_MATMUL_OPT_V1=1 speedups on heavy quants (Q5_K_XL) stemmed entirely from the lds_bank_pad=2 offset, which decoupled threads from 32-way LDS bank serialization.
- **Hardware Constraint:** True double-buffering is bounded by the 64KB physical LDS ceiling on gfx1030 architectures during heavy quantization passes.
- **Resolution:** Implemented automated static configuration traits via mmq_get_lds_bank_pad<type>(). Deprecated the environment flag entirely.
- **Open Tracking Item (CR-013):** load_tiles_iq4_xs_swizzled exhibits an Array-of-Structures (AoS) vs Structure-of-Arrays (SoA) layout mismatch when RDNA2_CACHE_SWIZZLE=ON. High-priority fix required before activating swizzling layers on IQ4_XS models.

## 🏗️ Backend Schedule Architecture Domain (CR-015)
The backend scheduler (`ggml/src/ggml-backend.cpp`) is a first-class monitoring domain:

| Responsibility | Owner | File |
|---------------|-------|------|
| Copy slot allocation drift during long-context generation (>2000 tokens) | @oracle | `ggml/src/ggml-backend.cpp:1912-1923` |
| MoE expert copy path correctness | @fixer | `ggml/src/ggml-backend.cpp:1576-1673` |
| Upstream sync tracking (ggml-org/llama.cpp PRs) | @librarian | `opencode/bug-analysis/CR-015-POSTMORTEM.md` |
| **4-fix patch applied** | @orchestrator | `opencode/proposals/CR-015.md` |

**Key files:** `ggml/src/ggml-backend.cpp` (scheduler core), `opencode/bug-analysis/CR-015-POSTMORTEM.md` (postmortem), `opencode/proposals/CR-015.md` (fix tracking), `opencode/proposals/CR-014.md` (superseded).

## CI/CD Notes
- **HIP build is non-blocking**: `continue-on-error: true` in `release.yml` line 198. CPU/Vulkan artifacts ship even if HIP fails.
- **Binary verification**: CI checks `llama-server`, `llama-cli`, `llama-bench` all exist before packaging.
- **RPATH enforced in CI**: release.yml builds with `-DCMAKE_INSTALL_RPATH='$ORIGIN'` for portable tarballs.
- **ccache**: Used in all CI jobs, keyed by workflow + platform.

## Language Policy
English only in all communication, comments, commit messages, and documentation.

## Repository Map
Full codemap at `codemap.md` (root). For deep work on a folder, also read that folder's `codemap.md`.
