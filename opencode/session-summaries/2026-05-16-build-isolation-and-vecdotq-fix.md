# Session Summary: 2026-05-16 — Build Isolation & RDNA2 vecdotq Bug Fix

## Garbled Output Root Cause: `vecdotq.cuh` Alignment Mask Bug

**File:** `ggml/src/ggml-cuda/vecdotq.cuh`

The `RDNA2_FORCE_LDS_ALIGNMENT(addr) & ~0xF` macro was forcing 16-byte alignment on all
quantized weight reads via `get_int_b1`/`get_int_b2`/`get_int_b4`. This corrupted the
address calculation: for indices 0-3 within any 16-byte chunk, **all reads returned the
same 4 bytes** (from `base & ~0xF`), meaning every row of quantized weights was
duplicated 4× instead of reading distinct rows. This produced the mixed-script garbage
output (e.g. `evelปุевичFTWARE全屏查看шта技术在`).

**Fix:** Reverted to upstream byte-by-byte/2-byte/4-byte accessors. The original code
handles unaligned loads correctly via individual byte reads, which is safe on all GPU
architectures including RDNA2.

## Secondary Bug: `mmq.cuh` LDS Double-Buffer Prefetch

**File:** `ggml/src/ggml-cuda/mmq.cuh`

The LDS double-buffer path (`lds_double_buffer == true`, gated by
`RDNA2_MATMUL_OPT_V1=1`) had `load_tiles` hardcoded to `offset_x + kb0_start` inside the
loop body instead of `offset_x + kb0`. This loaded the first block's weights on every
iteration, and the prefetch-from-next-iteration logic was also moved inside the loop
breaking the pipeline.

**Fix:** Reverted the LDS double-buffer path to upstream pattern (prefetch before loop,
use within loop, prefetch-next during compute, swap at end). All `s_waitcnt` fence
additions removed as redundant with `__syncthreads`.

## Build Isolation Fix: Library Cross-Contamination (P0)

**Problem:** `ldd` showed ALL turboquant binaries resolving `.so` dependencies to
`llama-mtp/build/bin/` because `LD_LIBRARY_PATH` included the mtp path before the
turboquant path. The RUNPATH (cmake default) is searched AFTER `LD_LIBRARY_PATH` on
glibc, so mtp libraries won. This caused ABI mismatch crashes (`common_params` struct
differs between forks: turboquant adds `reasoning_format`, `enable_reasoning` fields).

**Fix:** Configured CMake with:
```
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags"
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags"
```
This forces `RPATH` (searched before `LD_LIBRARY_PATH`). Binary verification:
```
$ readelf -d build/bin/llama-cli | grep RPATH
0x000000000000000f (RPATH)  Library rpath: [$ORIGIN:]
```
All libraries now resolve to turboquant build dir even with polluted LD_LIBRARY_PATH.

**Build caveat:** The system PATH has `/home/stormrage/rocm-7.13-nightly/bin` BEFORE
`/opt/rocm/bin`, so `which hipcc` finds the nightly's `hipcc` which sets up `.dll`
library paths. Must set PATH with `/opt/rocm/bin` first during cmake configure:
```bash
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
```

## Verified

- `/tmp/llama-server-state.sh` preserved from prior gpu_failback usage
- `llama-cli --help` exits cleanly with GPU init (no segfault)
- Model loads with `-ngl 99 -ncmoe 33` on 16GB RX 6800 XT (~5.5 GB VRAM used)
- `llama-server` health endpoint returns `{"status":"ok"}`

## Remaining Issues

1. **Hygiene test broken:** `tests/smoke_rdna2.cpp` uses `gcnArch` and `half` types
   removed in ROCm 7.13 nightly. Needs updating for new HIP API.
2. **Full inference untested:** `llama-cli -p` enters interactive mode by default;
   non-interactive completion path needs different flags or stdin piping.
3. **PEG parser still crashes on garbled input** (defense-in-depth fix pending).
