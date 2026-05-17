# Development Plan: llama.cpp-turboquant-hip

## Current Status (2026-05-17)
- ✅ IQ4_XS kernels verified (type 23)
- ✅ MTP speculative decoding: 78.7% acceptance rate
- ✅ Decode stable at ~39 t/s
- ✅ VRAM within 15.5GB redline
- ✅ No TDR resets, no memory leaks

## Benchmark Infrastructure

### Automated Benchmark Script
**File:** `agents/benchmark/benchmark_qwen3.py`

**Purpose:** Standardized benchmarking of Qwen3-35B MT PIQ4 with automated server lifecycle management

**Usage:**
```bash
python3 agents/benchmark/benchmark_qwen3.py
```

**Features:**
- Auto-starts/stops llama-server with exact production parameters
- Runs 4 standardized test prompts (Algorithmic, Knowledge, Needle, Code)
- Captures timings from API response
- Saves structured results to `agents/benchmark/logs.md`
- Handles server health checks and graceful shutdown

**Test Prompts:**
1. `creativeMTP.txt` — Algorithmic Logic & Pathfinding
2. `researchmtp.txt` — Cross-Disciplinary Knowledge
3. `problemsolvingmtp.txt` — Needle-in-Haystack
4. `codingmtp.txt` — Code Generation with Strict Syntax

**Output Format:** Markdown table with per-test metrics + detailed timings

## Next Steps (Tomorrow)

### 1. Run Automated Benchmark
```bash
cd /home/stormrage/llama.cpp-turboquant-hip
python3 agents/benchmark/benchmark_qwen3.py
```

### 2. Compare Results
Review `agents/benchmark/logs.md` for:
- Prefill t/s consistency across runs
- Decode t/s stability
- Draft acceptance rate trends
- VRAM usage patterns

### 3. Optimization Experiments
Based on benchmark results, test:
- `--spec-draft-n-max 3` (increase MTP depth)
- `--cache-type-k q4_0 --cache-type-v q4_0` (VRAM savings)
- Different `-ncmoe` values (expert layer offload tuning)

## Performance Baseline

| Metric | Current | Target |
|--------|---------|--------|
| Prefill t/s | 295-405 | >400 |
| Decode t/s | 38.9-39.0 | >40 |
| Draft acceptance | 78.7% | >80% |
| VRAM usage | <15.5GB | <15GB |
| Stability | No crashes | No crashes |

## Safety Constraints
- **BLOCK** if VRAM >15.5GB
- **REJECT** if decode variance >±5%
- **INVESTIGATE** if draft acceptance <70%
- **MONITOR** thermal throttling on runs >30min

## Files
- `agents/benchmark/benchmark_qwen3.py` — Automated benchmark runner
- `agents/benchmark/logs.md` — Benchmark results (auto-generated)
- `opencode/agents/chief_engineer.md` — Performance report & safety validation
- `run_benchmark.sh` — Legacy bash benchmark (use Python version)
