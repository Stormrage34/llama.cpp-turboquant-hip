# Development Plan: llama.cpp-turboquant-hip

## Current Status (2026-05-17)
- ✅ IQ4_XS kernels verified (type 23)
- ✅ MTP speculative decoding: 78.7% acceptance rate
- ✅ Decode stable at ~39 t/s
- ✅ VRAM within 15.5GB redline
- ✅ No TDR resets, no memory leaks

Benchmarking: use scripts/run_benchmark.sh (see AGENTS.md for details)
