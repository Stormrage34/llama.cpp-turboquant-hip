# Changelog

## v0.4.4-beta (2026-05-20) — Bug Hunt Release (17 fixes)

### P0 — Critical Fixes
- Crash on unused graph inputs — null checks added to graph lookup pointer returns (`src/llama-graph.cpp`)
- FA device mismatch with `--no-kv-offload` — correct device type comparison (`src/llama-context.cpp`)

### P1 — High Priority Fixes
- BitNet wrong output tensor — corrected to proper output tensor (`src/models/bitnet.cpp`)
- MPT q_norm test crash — corrected tensor shapes (`src/models/mpt.cpp`)
- Token type i32 vs u32 FIXME — documented (`src/llama-model-saver.cpp`)

### P2 — Standard Fixes
- Stubbed training functions — removed FIXME stubs (`src/llama-context.cpp`)
- Redundant synchronize() — removed unnecessary sync call (`src/llama-context.cpp`)
- size_t→int overflow casts — static_cast with bounds checks (`common/speculative.cpp`)
- Zero-size tensor crash — null + zero-element guards (`ggml-cuda.cu`)
- Stats misclassification race — added counter (`src/llama-context.cpp`)
- DeepSeek2 misleading FIXME — clarified comment (`src/models/deepseek2.cpp`)
- GGML_ASSERT crashes — graceful equal_seqs handling (`src/llama-graph.cpp`)
- KV cache multi-stream assert — proper multi-stream support (`src/llama-kv-cache.cpp`)
- KV cache save/restore TODO — removed stale TODO (`src/llama-kv-cache.cpp`)
- BF16 FP32 fallback — restructured vec_dot (`fattn-common.cuh`)
- Chat template hacks — fixed jinja runtime (`common/chat.cpp` + jinja runtime)
- Metadata flags misclassification — switch → bitwise AND (`src/llama-model-saver.cpp`)

---

## v0.4.3-beta (2026-05-18)
- **Benchmark infrastructure**: Standardized `run_benchmark.sh` harness with server lifecycle management
- **Comprehensive benchmark report**: Full cache comparison, context scaling, VRAM scaling, MTP acceptance analysis
- **Triple-sync bug fixed**: P0 fix — removed redundant `llama_synchronize()` calls in speculative decoding
- **load_gtt_slc wired**: `LOAD_EXPERT_F32`/`LOAD_EXPERT_F32X4` macros added in `common.cuh`
- **Idea A (128-bit V128_LOAD)**: Experimental `RDNA2_V128_LOAD` gate in `vecdotq.cuh`
- **Documentation**: README.md rewritten with comprehensive benchmark findings, roadmap updated

## v0.4.2-stable (2026-05-17)
- CI/CD pipeline fixed (branch triggers `master`→`main`, RPATH isolation)
- RDNA2 MoE Stream V1: async stream pipeline with SLC cache-bypass GTT loads
- IQ4_XS kernel support (type 23): verified dispatch, 78.7% MTP draft acceptance
- 128-bit loads: `get_int_b1/b2` replaced with direct 32-bit loads
- Build hygiene: RPATH isolation prevents library cross-contamination
- VGPR_OPT tuning: launch bounds optimization for IQ4_NL (32→24 VGPRs)

## v0.4.1-stable (2026-05-16)
- Compiler tuning: LLVM `-mllvm` flags applied unconditionally for gfx1030
- CLI `--reasoning` fix: no longer hardcodes DEEPSEEK format
- PEG parser crash defense: try-catch in `server-task.cpp`
- ROCm 7.13 compat: `gcnArch` → `gcnArchName`, `half`→`uint16_t`
