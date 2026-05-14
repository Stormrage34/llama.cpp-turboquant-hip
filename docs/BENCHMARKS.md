# Benchmarks

## ⚠️ Qwen3 Reasoning Mode Note

Qwen3 models enable internal reasoning (thinking chain) by default in `llama.cpp`.  
This is **not a hang** — the model is generating a reasoning trace before producing its final answer.

To control this behavior:

| Scenario | Flag | Effect |
|----------|------|--------|
| Standard generation (no thinking) | `--reasoning off` | Outputs only the final response, no thought chain |
| Show thinking + response | *(default / `--reasoning auto`)* | Outputs both reasoning and final answer; requires `-n 1024+` token budget |
| Extract reasoning only | `--reasoning-format deepseek` | Puts thought content into `message.reasoning_content` (API mode) |

**Minimum token budget for reasoning models:**
- `-n 64` — ❌ Too small; thinking alone exceeds this → apparent "hang"
- `-n 256` — ⚠️ May fit short thinking but not full response
- `-n 1024` — ✅ Comfortable for thinking + concise response
- `-n 2048` — ✅ Recommended for long-form generation

**Note:** RDNA2 flags (`RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1`) accelerate tensor math only.  
They do **not** modify sampling, chat templates, or reasoning behavior.

### Validation

A regression harness is available at `scripts/validate_qwen3_reasoning.sh`.  
It runs baseline vs RDNA2-optimized inference on a ~6000-token prompt and verifies:
- Coherent output (garbled/NaN = fail)
- Non-zero generation speed
- VRAM leak check (< 2 GB post-run)
- Both `--reasoning off` and default reasoning modes

Run: `bash scripts/validate_qwen3_reasoning.sh`
