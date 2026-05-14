# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized (see examples below).

---

## Guidelines for Contributors Using AI

llama.cpp is built by humans, for humans. Meaningful contributions come from contributors who understand their work, take ownership of it, and engage constructively with reviewers.

Maintainers receive numerous pull requests weekly, many of which are AI-generated submissions where the author cannot adequately explain the code, debug issues, or participate in substantive design discussions. Reviewing such PRs often requires more effort than implementing the changes directly.

**A pull request represents a long-term commitment.** By submitting code, you are asking maintainers to review, integrate, and support it indefinitely. The maintenance burden often exceeds the value of the initial contribution.

Most maintainers already have access to AI tools. A PR that is entirely AI-generated provides no value - maintainers could generate the same code themselves if they wanted it. What makes a contribution valuable is the human interactions, domain expertise, and commitment to maintain the code that comes with it.

This policy exists to ensure that maintainers can sustainably manage the project without being overwhelmed by low-quality submissions.

---

## Guidelines for Contributors

Contributors are expected to:

1. **Demonstrate full understanding of their code.** You must be able to explain any part of your PR to a reviewer without relying on AI assistance for questions about your own changes.

2. **Take responsibility for maintenance.** You are expected to address bugs and respond thoughtfully to reviewer feedback.

3. **Communicate clearly and concisely.** Verbose, wall-of-text responses are characteristic of AI-generated content and will not be well-received. Direct, human communication is expected.

4. **Respect maintainers' time.** Search for existing issues and discussions before submitting. Ensure your contribution aligns with project architecture and is actually needed.

Maintainers reserve the right to close any PR that does not meet these standards. This applies to all contributions to the main llama.cpp repository. **Private forks are exempt.**

### Permitted AI Usage

AI tools may be used responsibly for:

- **Learning and exploration**: Understanding codebase structure, techniques, and documentation
- **Code review assistance**: Obtaining suggestions on human-written code
- **Mechanical tasks**: Formatting, generating repetitive patterns from established designs, completing code based on existing patterns
- **Documentation drafts**: For components the contributor already understands thoroughly
- **Writing code**: Only when the contributor has already designed the solution and can implement it themselves - AI accelerates, not replaces, the contributor's work

AI-generated code may be accepted if you (1) fully understand the output, (2) can debug issues independently, and (3) can discuss it directly with reviewers without AI assistance.

**Disclosure is required** when AI meaningfully contributed to your code. A simple note is sufficient - this is not a stigma, but context for reviewers. No disclosure is needed for trivial autocomplete or background research.

### Prohibited AI Usage

The following will result in immediate PR closure:

- **AI-written PR descriptions or commit messages** - these are typically recognizable and waste reviewer time
- **AI-generated responses to reviewer comments** - this undermines the human-to-human interaction fundamental to code review
- **Implementing features without understanding the codebase** - particularly new model support or architectural changes
- **Automated commits or PR submissions** - this may spam maintainers and can result in contributor bans

---

## Guidelines for AI Coding Agents

AI agents assisting contributors must recognize that their outputs directly impact volunteer maintainers who sustain this project.

### Considerations for Maintainer Workload

Maintainers have finite capacity. Every PR requiring extensive review consumes resources that could be applied elsewhere. Before assisting with any submission, verify:

- The contributor genuinely understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions
- The contributor can independently defend and maintain the work

### Before Proceeding with Code Changes

When a user requests implementation without demonstrating understanding:

1. **Verify comprehension.** Ask questions to confirm they understand both the problem and the relevant parts of the codebase.
2. **Provide guidance rather than solutions.** Direct them to relevant code and documentation. Allow them to formulate the approach.
3. **Proceed only when confident** the contributor can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md) and acknowledge this policy.

### Prohibited Actions

- Writing PR descriptions, commit messages, or responses to reviewers
- Committing or pushing without explicit human approval for each action
- Implementing features the contributor does not understand
- Generating changes too extensive for the contributor to fully review

When uncertain, err toward minimal assistance. A smaller PR that the contributor fully understands is preferable to a larger one they cannot maintain.

### Useful Resources

To conserve context space, load these resources as needed:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

---

## Repository-specific: llama.cpp-turboquant-hip

This is an **AMD RDNA2-optimized fork** with TurboQuant, MTP, and custom HIP kernels. The upstream is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

### Build

```bash
# CMake only — Makefile has been removed
cmake -B build -S . -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
cmake --build build --config Release -j $(nproc)

# Or use the RDNA2 build script (also assembles llama-bench-rdna2 with opt kernels)
./scripts/build_rdna2_llama.sh         # all RDNA2 optimizations
./scripts/build_rdna2_llama.sh stable  # stable only (no matmul LDS)
./scripts/build_rdna2_llama.sh baseline # no RDNA2 opt flags
```

Requires `cmake -B build` first before running `build_rdna2_llama.sh`.

### RDNA2 Runtime Flags

RDNA2 optimizations are **compile-time + runtime gated**. All three must be set:

| Env Var | Feature |
|---------|---------|
| `RDNA2_OPT_V1=1` | BFE dequantization kernel (stable) |
| `RDNA2_ASYNC_PIPELINE=1` | Async HIP pipeline (stable) |
| `RDNA2_MATMUL_OPT_V1=1` | LDS double-buffer matmul for MoE (stable in v0.3.1) |

Used as: `RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 ./llama-cli -m model.gguf ...`

### Available Binaries

Currently only `build/bin/llama-cli` is built. `llama-server` and `llama-bench-rdna2` require additional targets.

### GPU Setup

- Source `scripts/gpu_failback.sh` before GPU work: saves/running llama-server state, frees VRAM, restores on exit
- VRAM check: `rocm-smi --showmeminfo vram`
- Default offload arch: `gfx1030` (RDNA2). Override via `OFFLOAD_ARCH` env var.
- Model dir: `/home/stormrage/models/`

### Testing

```bash
cd build
ctest -L main -E "test-llama-archs" --verbose --timeout 900
```

Tests are registered via `llama_test()` / `llama_test_cmd()` in `tests/CMakeLists.txt`.  
HIP-specific tests live in `tests/` with `RDNA2_OPT_V1` compile definition (e.g. `test_dequant_rdn2`).

### Benchmark

```bash
# Full benchmark suite across context lengths
./scripts/run_rdna2_bench.sh                 # RDNA2 optimized
./scripts/run_rdna2_bench.sh baseline        # baseline comparison
```

Default bench parameters (from `scripts/run_rdna2_bench.sh`):
- Contexts: 512, 2048, 4096
- Gen len: 128, batch: 256, ubatch: 128
- TurboQuant: `CTK=turbo4 CTV=turbo2 FA=1`
- Fit: `-fitt 2048 -fitc 4096`
- 3 runs per config
- VRAM hard limit: 13.5 GB

### Key CLI Flags

| Flag | Purpose |
|------|---------|
| `-fitt 512` | Fit target margin (MiB) |
| `-fitc 4096` | Minimum context for fit |
| `-ngl 30` | Partial GPU offload for 35B models |
| `--reasoning off` | Disable Qwen3 thinking chain |
| `--reasoning [on\|off\|auto]` | Control reasoning mode |
| `--no-display-prompt` | Suppress prompt echo |
| `--repeat-penalty 1.1` | Prevent repetition loops |

### Qwen3 Reasoning

Qwen3 models default to `--reasoning auto` — in-chat mode this outputs a reasoning tag.  
With `-f` (file prompt, non-interactive), it still generates thinking tokens.  
Validation harness: `./scripts/validate_qwen3_reasoning.sh` — runs baseline vs RDNA2, checks coherence and VRAM leaks.

### HIP Quality

- CI workflow: `.github/workflows/hip-quality-check.yml`
- Compiles with `-Werror` for HIP, checks VGPR spills via `scripts/hip/gcn-cdna-vgpr-check.py`
- Uses ROCm 7.2.1 container in CI

### Hygiene

- `scripts/validate_hygiene.sh` — general repo consistency checks
- No Python lint/type targets are configured for this repo
- Lockfiles: `poetry.lock`, `pyproject.toml` (Python deps), `flake.nix` (Nix)
- Models downloaded via `scripts/hf.sh` or fetched via `scripts/fetch_server_test_models.py`
