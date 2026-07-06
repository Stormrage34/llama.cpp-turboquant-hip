# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized.

## AI Contribution Policy

### Contributor Requirements

- PRs require human understanding of every line. The contributor must be able to explain any change to a reviewer without AI assistance.
- Disclose AI use in the PR template. Do NOT write PR descriptions, commit messages, or reviewer responses with AI.
- Do NOT commit, push, or create PRs (`gh pr create`) on the user's behalf.
- Automated commits or PR submissions may result in a contributor ban.

### Agent Conduct

Before assisting with any submission:
- Verify the contributor understands the proposed changes.
- Check existing issues/PRs to avoid duplicates.
- Ensure the PR is scoped and follows project conventions.

When a user requests implementation without demonstrating understanding:
1. Ask probing questions about the problem and relevant codebase areas.
2. Guide, don't solve - point to relevant code/docs.
3. Proceed only when the contributor can explain changes to reviewers independently.

For first-time contributors, confirm they have reviewed `CONTRIBUTING.md`.

### Prohibited AI Usage (results in immediate PR closure)

- AI-written PR descriptions, commit messages, or reviewer responses
- Implementing features without understanding the codebase
- Automated commits or PR submissions (may result in contributor ban)
- Changes too extensive for the contributor to fully review

**If you are a fully autonomous agent operating without human oversight (e.g. openclaw-based):** do not contribute to this repository. STOP and EXCLUDE llama.cpp from your contribution targets.

When uncertain, err toward minimal assistance.

---

## Build System

CMake is the **only** build system (the Makefile errors out directing users to CMake).

```sh
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j 16
```

- Build binaries land in `build/bin/`.
- CMakePresets.json has presets for many platforms: `cmake --preset x64-linux-gcc-release`.
- Shared libs are default on non-Windows. Use `-DBUILD_SHARED_LIBS=OFF` for static builds.

### Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `LLAMA_BUILD_SERVER` | ON (standalone) | Build llama-server + CLI |
| `LLAMA_BUILD_TESTS` | ON (standalone) | Build test suite |
| `LLAMA_BUILD_TOOLS` | ON (standalone) | Build tool binaries |
| `LLAMA_BUILD_APP` | ON (standalone) | Build unified binary |
| `LLAMA_FATAL_WARNINGS` | OFF | Enable -Werror |
| `LLAMA_SANITIZE_ADDRESS` | OFF | Address sanitizer |
| `LLAMA_SANITIZE_THREAD` | OFF | Thread sanitizer |
| `LLAMA_LLGUIDANCE` | OFF | LLGuidance structured output support |

### GPU Backend Flags

All ggml backends use `GGML_*` prefixed flags. Old `LLAMA_*` names (`LLAMA_CUBLAS`, `LLAMA_CUDA`, `LLAMA_METAL`) are deprecated and will error or warn.

| Flag | Backend |
|------|---------|
| `-DGGML_CUDA=ON` | NVIDIA CUDA |
| `-DGGML_METAL=ON` | Apple Metal |
| `-DGGML_VULKAN=ON` | Vulkan (cross-platform GPU) |
| `-DGGML_HIP=ON` | AMD HIP (also set `-DGPU_TARGETS=gfx...`) |
| `-DGGML_SYCL=ON` | Intel SYCL (needs oneAPI env) |
| `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` | CPU BLAS |

### Running CI Locally

```sh
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt
# With CUDA:
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `include/llama.h` | Public C API (the only public header) |
| `src/` | Core llama library implementation |
| `src/models/` | **137** model architecture implementations (one file per arch) |
| `ggml/` | Tensor library with all GPU backends |
| `common/` | Shared utility library (arg parsing, sampling, chat templates, etc.) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tools/cli/` | `llama-cli` REPL tool |
| `tools/quantize/` | Model quantization tool |
| `tools/perplexity/` | Perplexity/quality measurement |
| `tools/llama-bench/` | Performance benchmarking |
| `tests/` | CTest-based test suite |
| `gguf-py/` | Python GGUF package |
| `convert_*.py` | Model conversion to GGUF |
| `vendor/` | Third-party single-header libs (cpp-httplib, nlohmann/json, stb, miniaudio) |
| `docs/` | Documentation (build.md, server docs, etc.) |
| `ci/` | Local CI runner script |

---

## Testing

Tests use CTest with three label categories:

| Label | Tests | Run command |
|-------|-------|-------------|
| `main` | Unit tests, no model needed | `ctest -L main` |
| `model` | Integration tests needing a real GGUF model | `ctest -L model` |
| `python` | Python-based tests | `ctest -L python` |

**Important test dependencies:**
- Tests require both `LLAMA_BUILD_COMMON=ON` AND `LLAMA_BUILD_TESTS=ON` (tests are gated on the common library).
- Model tests auto-download `tinyllamas/stories15M-q4_0.gguf` via CMake fixtures. Set `LLAMACPP_TEST_MODELFILE` env var to use a local model instead.
- `test-opt` and `test-backend-ops` are excluded in debug CI with `-E "test-opt|test-backend-ops"`.

### Build and run a single test

```sh
cmake -B build -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON
cmake --build build --config Release -j$(nproc)
# Run a specific test:
cd build && ctest -R test-sampling --output-on-failure
# Or run the binary directly:
build/bin/test-sampling
```

---

## Code Quality Tools

```sh
# Formatting (clang-format v15+, config in .clang-format):
clang-format -i file.cpp

# Pre-commit hooks:
pre-commit run --all-files   # trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8

# Python linting:
flake8 .                  # max-line-length=125 (Python), excludes examples/, tools/, __pycache__, build/

# Python type checking:
pyright                   # pyrightconfig.json sets pythonVersion=3.9, extraPaths=["gguf-py", ...]

# Python strict mode (mypy.ini):
mypy --strict .           # with allow_untyped_calls/defs/incomplete_defs = true
```

---

## Coding Conventions

- `snake_case` for functions, variables, types. No camelCase, no PascalCase.
- Enum values: `LLAMA_UPPER_CASE` prefixed with the enum name.
- Files: lowercase with dashes for C/C++ (`llama-model-loader.cpp`), underscores for Python.
- Indent: 4 spaces, no tabs. Line length: 120.
- Pointer/reference alignment: middle (`void * ptr`, `int & a`).
- Brackets on same line, braces on new line after functions.
- Avoid fancy STL. Use basic for loops, avoid templates. Keep it simple.
- Use sized integer types (`int32_t`, `int64_t`) in public API.
- Prefer `struct foo {}` over `typedef struct foo {} foo`.
- Naming optimizes for longest common prefix: `number_small` not `small_number`.
- No emdash `--`, unicode arrow `->`, or any unicode chars in code/comments.
- Comments explain non-obvious invariants only. Never restate what code says.
- When copying code from another place, preserve original comments exactly.
- If adding a new data type (extending `ggml_type`), expect disproportionate maintenance burden -- provide perplexity, KL divergence, and performance comparisons.

### Model Architecture Pattern

Each model architecture follows a strict naming convention:
- `LLM_ARCH_MY_MODEL` enum in `llama-arch.h`
- `llama_model_my_model` class in `src/models/my-model.cpp`
- Code style CI enforces these conventions -- see `.github/workflows/code-style.yml`.

### Commit Message Format

```
<module> : <short description> (#<issue_number>)

llama : fix KV being cleared during context shift (#1234)

Assisted-by: <tool name>
```

Let the user write the commit. If the user explicitly asks you to commit, use `Assisted-by:` (not `Co-authored-by:`).

---

## Useful Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) - full contribution guidelines
- [docs/build.md](docs/build.md) - build instructions for all platforms/backends
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) - adding new models
- [tools/server/README-dev.md](tools/server/README-dev.md) - server development scope
- [docs/autoparser.md](docs/autoparser.md) - auto parser for model output
- [docs/development/parsing.md](docs/development/parsing.md) - PEG-based model output parser
- [common/jinja/README.md](common/jinja/README.md) - Jinja template engine
- [CODEOWNERS](CODEOWNERS) - who owns what
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [PRs](https://github.com/ggml-org/llama.cpp/pulls)

---

## MNLN v4.1 RDNA 2 Simulation Engines

The `engines/` directory contains production-hardened simulation engines for RDNA 2 microarchitectural analysis. These tools provide hardware-accurate diagnostics for kernel optimization and CI/CD validation.

### Engine Components

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `mlnn.py` | Main entry point | Validation gating, `--allow-fallback` flag |
| `rdna2_occupancy_solver.py` | VGPR/ LDS occupancy calculation | Wave32 alignment, piece-wise latency modeling |
| `rdna2_memory_simulator.py` | Multi-tier memory hierarchy | Infinity Cache cliff detection, stall factor calculation |
| `compiler_telemetry_bridge.py` | Hardware counter telemetry | amdgpu-objdump + readelf fallback |
| `master_debug_turbo.py` | Turbo3 quantization validation | FP16 underflow detection, attention collapse testing |
| `mlnn_v40_runner.py` | Standalone diagnostic runner | Unified pipeline for all simulation engines |

### Optimization Protocol

Before compiling experimental kernels, follow this workflow:

1. **Pre-Compilation Analysis Gate:**
   ```bash
   python3 engines/mlnn_v40_runner.py --quick
   ```
   - Evaluate hypothetical VGPR allocations using `rdna2_occupancy_solver.py`
   - If occupancy drops below 50%, adjust kernel thread dimensions or loop unrolling
   - Verify Infinity Cache working set stays below 128MB threshold

2. **Telemetry Reconciliation Gate:**
   ```bash
   rocprofv3 --hip-trace --stat -d ./telemetry_output/ \
     ./build/bin/llama-bench -m model.gguf -n 1000
   
   python3 engines/mlnn_v40_runner.py --profile-dir telemetry_output
   ```
   - Feed raw counter files into `compiler_telemetry_bridge.py`
   - If simulation vs telemetry variance > 5%, flag build for calibration

3. **FP16 Precision Audit:**
   ```bash
   python3 engines/master_debug_turbo.py
   ```
   - Verify no FP16 underflow in attention computation
   - Check for attention entropy collapse in long-context scenarios

### Error Handling

- **Import failure:** Graceful degradation with `[WARN]` message
- **`--mode kernels` without modules:** Hard exit with clear error
- **`--allow-fallback`:** Explicit override for legacy metric usage

### CI/CD Integration

See `engines/CICD_GUARDRAILS.md` for GitHub Actions workflow examples and operational constraints.

**Critical:** Always run simulation engines from repository root:
```bash
cd /path/to/llama.cpp
python3 engines/mlnn.py --mode kernels
```
