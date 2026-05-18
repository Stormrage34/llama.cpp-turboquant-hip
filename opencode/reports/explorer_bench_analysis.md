# Explorer Report: Benchmark Script Analysis & Optimization Hints

## Observation: Current Benchmark Scripts Lack Server Awareness

All 5 benchmark scripts attempt GPU access without checking if llama-server is already running:

| Script | Server Check | gpu_failback.sh | Risk Level |
|--------|-------------|-----------------|------------|
| `run_std_bench.sh` | **NONE** | Not used | **HIGH** — Contaminated results, no warning |
| `run_rdna2_bench.sh` | **NONE** | **YES** (line 16-18) | **CRITICAL** — `gpu_acquire()` waits for VRAM free, will hang or produce garbage |
| `bench-models.sh` | **NONE** | Not used | **HIGH** — Contaminated results |
| `run_rocprof_baseline.sh` | **NONE** | Not used | **CRITICAL** — rocprofv3 on active server may corrupt profiling data |
| `benchscript.sh` | N/A (launcher) | N/A | OK — this IS the server launcher |

## Critical Issues Found

### Issue 1: `run_rdna2_bench.sh` Calls `gpu_acquire()` (Line 18)
```bash
source "$(cd "$(dirname "$0")" && pwd)/gpu_failback.sh"
gpu_failback_trap
gpu_acquire  # ← This waits for VRAM to drop below 1GB
```
**Problem**: If llama-server is running, `gpu_acquire()` calls `_gpu_wait_vram_below()` which loops for 30 seconds waiting for VRAM to drop. Since server holds ~14GB VRAM, this **always times out** and proceeds with contaminated GPU state.

**Fix**: Add server check BEFORE `gpu_acquire()`:
```bash
if gpu_is_busy; then
    echo "ERROR: GPU is busy (llama-server running). Cannot benchmark."
    echo "Options: stop server, use cloud benchmark, or check /stats endpoint"
    exit 1
fi
gpu_acquire
```

### Issue 2: No Script Checks `pgrep llama-server` Before Benchmarking
None of the 4 benchmark scripts (excluding benchscript.sh launcher) check for running server. This means:
- Benchmark runs produce **meaningless results** when GPU is shared
- rocprofv3 profiling on active server produces **corrupted counters**
- User gets no warning that results are invalid

### Issue 3: `bench-models.sh` Downloads Models While Server May Be Running
```bash
./bin/llama-bench -m "${HFF}" ...  # Line 51-53
```
Downloads from HuggingFace and benchmarks without checking GPU state. If server is running, VRAM contention causes slow downloads AND corrupted bench results.

### Issue 4: No Metrics Endpoint Fallback
When llama-server is running with `--metrics`, the scripts don't check its `/stats` or `/metrics` endpoints for performance data. This wastes an opportunity to get live performance data without interrupting the server.

## Optimization Hints for RDNA2 Kernels

### Hint 1: Prefill-Heavy Workloads (bench-models.sh pattern)
- `bench-models.sh` tests large contexts (270336 tokens) with `--npp 512,4096,8192`
- These are **prefill-dominated** workloads
- Optimization target: `mmq.cuh` double-buffered matmul (RDNA2_MATMUL_OPT_V1)
- Expected gain: 15-30% prefill throughput on RDNA2 with LDS bank padding fix (P2)

### Hint 2: Decode-Heavy Workloads (run_std_bench.sh pattern)
- `run_std_bench.sh` tests `tg128` (128-token decode) with 10 runs
- This is **decode-dominated** workload
- Optimization target: `vecdotq.cuh` quantized matvec (P1: 128-bit loads)
- Expected gain: 5-10% decode throughput from reduced VALU instructions

### Hint 3: rocprofv3 Profiling Should Only Run on Isolated GPU
- `run_rocprof_baseline.sh` runs rocprofv3 with `--hip-trace --kernel-trace`
- **Must NOT run while llama-server is active** — profiling adds overhead and contaminates counters
- Recommended: run rocprofv3 only in CI or on dedicated benchmark machine

### Hint 4: Cloud Benchmarking for Regression Testing
Since local GPU may be in use:
- Store baseline results in `benchmarks/std_bench/` (already done)
- For regression testing, use cloud GPU instance with identical config
- Compare cloud results to local baselines using `scripts/analyze_counters.sh`

## Recommended Architecture Change

### New Server-Aware Benchmark Flow
```
User runs: ./scripts/run_std_bench.sh <model> moe-99
    │
    ├─→ Check: pgrep llama-server?
    │   ├─ YES → Report SERVER_RUNNING, suggest cloud/manual check, EXIT
    │   └─ NO  → Continue
    │
    ├─→ gpu_acquire (if needed)
    ├─→ Run benchmark
    ├─→ Optionally run rocprofv3 (if available AND no server)
    └─→ Save results to benchmarks/std_bench/<timestamp>_<config>/
```

### New Shared Library: `scripts/server_check.sh`
Create a shared utility that all benchmark scripts source:
```bash
# scripts/server_check.sh
# Usage: source this, then check SERVER_AVAILABLE

check_server_available() {
    if pgrep -x llama-server >/dev/null 2>&1; then
        SERVER_PID=$(pgrep -x llama-server)
        echo "SERVER_BUSY:$SERVER_PID"
        return 1
    fi
    echo "SERVER_FREE"
    return 0
}

get_server_stats() {
    curl -s http://localhost:8080/stats 2>/dev/null | jq '.'
}
```

## Exploration Verdict

| Area | Status | Recommendation |
|------|--------|----------------|
| Server detection | **MISSING** | Add `pgrep llama-server` check to ALL benchmark scripts |
| gpu_acquire safety | **DANGEROUS** | Check server availability BEFORE calling `gpu_acquire()` |
| rocprofv3 safety | **DANGEROUS** | Never run rocprofv3 while server is active |
| Metrics fallback | **MISSING** | Add `/stats` endpoint check when server is running |
| Cloud benchmark guide | **MISSING** | Document cloud benchmark protocol in README |
| Shared server check | **MISSING** | Create `scripts/server_check.sh` utility |
