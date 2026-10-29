# AGENTS.md

llama.cpp: LLM inference in C/C++. Build system is CMake (not Make - the Makefile only errors out).

## Build

- CPU build:
  - `cmake -B build`
  - `cmake --build build --config Release -j12`
- HIP/ROCm build (gfx1030 / RX 6900 XT):
  - `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON`
  - `cmake --build build --config Release -j12`
  - Binaries land in `build/bin/`. Test with `./build/bin/llama-bench`.
  - Use `-j8` to `-j12`; `-j$(nproc)` (28 on 14700) crashes (OOM + ROCm clang segfault on E-cores).
- Toolchain presets (build dir becomes `build-<preset>`):
  - `cmake --preset x64-linux-gcc-release`
  - `cmake --build build-x64-linux-gcc-release`
- Standalone builds enable tests/tools/examples by default. Disable with `-DLLAMA_BUILD_TESTS=OFF`, `-DLLAMA_BUILD_TOOLS=OFF`, etc.
- CI-style flags: `-DLLAMA_FATAL_WARNINGS=ON -DGGML_NATIVE=OFF`.
- Docs: `docs/build.md`. Local heavy CI: `bash ./ci/run.sh ./tmp/results ./tmp/mnt` (see `ci/README.md`).

## Test

- C/C++ tests: `cd build && ctest -L main --verbose --timeout 900`.
  - Labels: `main`, `behavior`, `model`, `backend`, `python`. Run subsets with `ctest -L behavior`.
- Single test: `cd build && ctest -R '^test-chat$' --verbose`, or run the binary directly: `./build/bin/test-chat`.
- `test-backend-ops` supports `--min-time <ms>` for perf cases.
- Model/network tests are under `model` and may need downloads.
- ggml changes: run `test-backend-ops` with at least two backends; update `docs/ops.md` and `docs/ops/*.csv` for new/changed ops.
- Server tests: build `llama-server`, then:
  - `cd tools/server/tests && pip install -r requirements.txt && ./tests.sh`
  - Single: `./tests.sh unit/test_chat_completion.py::test_name`
  - Slow: `SLOW_TESTS=1 ./tests.sh`; debug: `DEBUG=1 ./tests.sh -s -v -x`; external server: `DEBUG_EXTERNAL=1`.
- Python lint: `flake8` (config `.flake8`, max line 125, excludes tools/examples). Uses `flake8-no-print`.
- Type checkers: configs exist for pyright (`pyrightconfig.json`), mypy (`mypy.ini`), ty (`ty.toml`).
- Pre-commit: `pre-commit run --all-files` (trailing whitespace, yaml, large files, flake8).
- Format C/C++ with clang-format using `.clang-format` (120 columns). Format changed code: `git clang-format` or `clang-format -i`.

## Architecture

- `include/llama.h`: stable public C API. `include/llama-cpp.h`: thin C++ convenience. `src/llama-ext.h`: experimental/WIP API where breaking changes are allowed.
- `src/`: library implementation; `src/models/`: per-architecture model code.
- `common/`: shared utilities, arg parsing, chat/Jinja templates, peg parsers. Chat parsers are enumerated in `common/parsers/sources.cmake`.
- `ggml/`: tensor library plus backends (cpu, cuda, metal, vulkan, hip, sycl, etc.).
- `tools/`: cli, server, quantize, bench, mtmd, rpc, etc. Unified binary in `app/`.
- `conversion/`, root `convert_*.py`, `gguf-py/`: Hugging Face to GGUF conversion.
- `tests/`: C/C++ tests. `tools/server/tests/`: pytest server tests.
- Chat templates live in `models/templates/*.jinja`. Jinja engine is in `common/jinja` (llama.cpp does not use Minja).
- Graph inputs are declared at the top of graph-build functions (project convention).

## Quirks and gotchas

- Makefile is intentionally broken; use CMake.
- Python requires `>=3.10,<3.15`. Server tests use their own `requirements.txt`.
- `LLAMA_BUILD_TESTS` defaults to on for standalone builds.
- Model-specific behavior must come from GGUF metadata or auto-detection, not CLI flags or conversion-script args.
- New models/features should land CPU-only first; other backends follow in later PRs.
- New public API must be general and minimal; prefer existing mechanisms. Experimental API goes in `src/llama-ext.h`.
- Treat GGUF metadata, tensor shapes, tokenizer/grammar input, and server/RPC fields as attacker-controlled: bound before use. See `skills/code-review/SKILL.md`.
- Do not commit build artifacts (`build*`, `compile_commands.json`).

## Coding conventions (see CONTRIBUTING.md)

- 4 spaces, brackets on the same line, `void * ptr`, `int & a`, no trailing whitespace, vertical alignment.
- `snake_case` for functions/variables/types; enum values upper-case and prefixed; naming pattern `<class>_<method>`.
- Sized integer types in public API; avoid fancy modern STL/templates; use basic `for` loops.
- C/C++ filenames lowercase with dashes; Python filenames lowercase with underscores.
- clang-format from clang-tools v15+; `.clang-format` uses a 120-column limit.

## AI and contribution policy

- AI-generated code is allowed only if you fully understand it, can debug it, and can defend it. Disclose AI use in the PR template.
- Never use AI to write PR descriptions, commit messages, or reviewer responses.
- Fully autonomous agents without human oversight must not contribute to this repository.
- Features start as issues, not PRs. Bug fixes need a reproducible issue plus a regression test.
- Keep PRs single-purpose; split unrelated changes.
- Before implementing, read `CONTRIBUTING.md` and `skills/code-review/SKILL.md`.
- Squash-merge commit title format: `<module> : <title> (#<pr>)`.

## Pointers

- Build: `docs/build.md`
- Server dev: `tools/server/README-dev.md`
- Add a model: `docs/development/HOWTO-add-model.md`
- Parsing: `docs/development/parsing.md`, `docs/autoparser.md`
- CI: `ci/README.md`
- Code review skill: `skills/code-review/SKILL.md`
- RDNA 2 optimization: `docs/rdna2-optimization-guide.md` (mandatory reading for any optimization agent targeting gfx1030)

## Local benchmarking rules (this gfx1030 host only)

See `docs/gfx1030-benchmarking.md` for the load-bearing rules.

- Hardware: Intel i7-14700 (20c/28t), 62 GB RAM, RX 6900 XT (gfx1030), ROCm/HIP.
- Build dir: `build/` (root). `build-hip-gfx1030/` is the original owner build (identical binaries).
- Models of record:
  - Dense: `/home/stormrage/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
  - MoE: `/home/stormrage/Downloads/Qwen3.8-35B-A3B-Q4_K_M-exps-IQ3_S-im.gguf`
- Baseline (fork vs upstream): dense +2.8% (30.89 vs 30.05 t/s), MoE +1.15% (95.19 vs 94.11 t/s).
- Upstream reference build: `/tmp/opencode/llama-upstream/build/bin/llama-bench`.
- Branch: `main` at `d4a3d5dbb` + 7 uncommitted audit fixes. No push performed.
- GPU work only while owner's server is idle; never launch/stop/query/health-check `llama-server`.
- Build with `-j12` (not `-j$(nproc)` — OOM + clang segfault on E-cores).
