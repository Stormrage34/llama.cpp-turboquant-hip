# Session Summary: 2026-05-17 — AGENTS.md Cleanup & RDNA2_MOE_STREAM_V1 Wiring

## AGENTS.md Consolidation

**Goal:** Reduce 175 lines → 120 lines by removing low-signal content.

**Removed:**
- Benchmark (llama-server) section — not agent-actionable
- Model VRAM guide table — hardware-specific, not workflow-guiding
- Binaries listing — redundant with build section
- Performance summary numbers — verbose
- `build_rdna2_llama.sh` references — unified to `build_rdna2.sh`
- "CMake-first build" line — build script handles this

**Updated:**
- Added RDNA2 CMake options table (new `RDNA2_MOE_STREAM_V1` and `GGML_RDNA2_BFE_DISPATCHER`)
- Merged gotchas into "Known Bugs & Gotchas" section

## RDNA2_MOE_STREAM_V1 Compile Definition Wiring

**Problem:** `RDNA2_MOE_STREAM_V1` was defined as a CMake option in `CMakeLists.txt` (line 195) but was never propagated as a compile definition in `ggml/src/ggml-hip/CMakeLists.txt`. The build script also had it removed from cmake flags.

**Fixes:**
1. `ggml/src/ggml-hip/CMakeLists.txt:169-171` — Added compile definition propagation:
   ```cmake
   if (RDNA2_MOE_STREAM_V1)
       add_compile_definitions(RDNA2_MOE_STREAM_V1)
   endif()
   ```
2. `scripts/build_rdna2.sh:200` — Restored `-DRDNA2_MOE_STREAM_V1=ON` to cmake flags.

**Verified:**
- Build succeeds
- `-DRDNA2_MOE_STREAM_V1` appears in compile flags (`build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/flags.make`)
- `RDNA2_MOE_STREAM_V1:BOOL=ON` in `build/CMakeCache.txt`

## Session from 2026-05-16 Read

Read `opencode/session-summaries/2026-05-16-build-isolation-and-vecdotq-fix.md`. All key findings already covered in AGENTS.md — no gaps identified.
