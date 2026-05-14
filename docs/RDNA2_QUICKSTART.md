# RDNA2 Quickstart: RX 6800 XT (gfx1030)

> **Time-to-first-benchmark**: ~15 minutes from clone to results.

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| ROCm | 6.1+ (7.13 nightly recommended) | `hipcc --version` |
| CMake | ≥3.24 | `cmake --version` |
| GPU | RDNA2 (gfx1030) | `rocm-smi --showproductname` |
| VRAM | ≥16 GB | `rocm-smi --showmeminfo vram` |
| OS | Linux (CachyOS/Ubuntu 22.04+) | `uname -a` |

## One-Command Build

```bash
# Clone + build (RDNA2-optimized)
git clone https://github.com/Stormrage34/llama.cpp-turboquant-hip
cd llama.cpp-turboquant-hip

# CMake build (required first)
cmake -B build -S . -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
cmake --build build --config Release -j $(nproc)

# Or use the build script (also builds llama-bench-rdna2)
./scripts/build_rdna2_llama.sh optimized
```

## Run Your First Benchmark

```bash
# MoE prefill test (Qwen3_35BMTPIQ4)
RDNA2_OPT_V1=1 RDNA2_ASYNC_PIPELINE=1 RDNA2_MATMUL_OPT_V1=1 \
  ./build/bin/llama-bench \
    -m /home/stormrage/models/Qwen3_35BMTPIQ4.gguf \
    -ngl 99 -c 4096 -p 512 -n 128 -b 256 -ub 256 \
    -ctk turbo4 -ctv turbo2 --flash-attn on --no-mmap -r 10

# Expected: ~2780 t/s prefill, ±6 t/s variance
```

## RDNA2 Runtime Flags

All three flags must be set for RDNA2 optimizations:

```bash
export RDNA2_OPT_V1=1          # BFE dequant kernel (stable)
export RDNA2_ASYNC_PIPELINE=1  # Async HIP pipeline (stable)
export RDNA2_MATMUL_OPT_V1=1   # LDS double-buffer matmul for MoE (stable)
```

| Flag | Purpose | Models | Status |
|------|---------|--------|--------|
| `RDNA2_OPT_V1=1` | Enable RDNA2 dequant + matmul kernels | All | Stable |
| `RDNA2_ASYNC_PIPELINE=1` | Overlap dequant + compute | All | Stable |
| `RDNA2_MATMUL_OPT_V1=1` | LDS double-buffer matmul | MoE only | Stable |
| `RDNA2_BFE_DISPATCHER=1` | `v_bfe_u32` for K-quant unpack | Q4_K_M, Q5_K_M | **Experimental** — validate first |

> ⚠️ `RDNA2_BFE_DISPATCHER` requires kernel-path verification before enabling. See [docs/RDNA2_FLAGS.md](RDNA2_FLAGS.md).

## Recommended `-ngl` by Model

| Model | Quant | `-ngl` | Expected Prefill | Expected Decode |
|-------|-------|--------|-----------------|----------------|
| 7B | Q4_K_M | 99 | ~850 t/s | ~85 t/s |
| 13B | Q4_K_M | 99 | ~540 t/s | ~55 t/s |
| 27B | Q4_K_M | 55 | ~540 t/s | ~27 t/s |
| 35B MoE | IQ4_XS | 99 | **~2780 t/s** | ~66 t/s |

> **Config matters**: The 27B dense model at `-ngl 99` will OOM on 16 GB VRAM. Use `-ngl 55` or reduce context. MoE models can use `-ngl 99` because only active experts are loaded.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `hipcc not found` | `export PATH=/opt/rocm/bin:$PATH` |
| `GPU_TARGETS not set` | Add `-DGPU_TARGETS=gfx1030` to cmake |
| `VRAM OOM` | Reduce `-ngl` (e.g., `-ngl 55` for 27B dense) |
| `NaN/garbage tokens` | Ensure `RDNA2_OPT_V1=1` + `--no-mmap` |
| `Slow decode (<20 t/s)` | Check `-ngl` matches model size; add `-ctk turbo4 -ctv turbo2` |
| `High variance (>±50 t/s)` | Ensure `RDNA2_MATMUL_OPT_V1=1` for MoE models |
| `rocprofv3 not found` | Install `rocm-profiler` package or use `/opt/rocm/bin/rocprofv3` |

## Validation Scripts

```bash
# Verify which dequant kernels are actually dispatched
./scripts/verify_kernel_dispatch.sh <model.gguf> Q4_K_M

# A/B telemetry: compare with/without a flag (same build)
./scripts/run_ab_telemetry.sh <model.gguf> RDNA2_BFE_DISPATCHER 5

# Standardized benchmark (identical configs across versions)
./scripts/run_std_bench.sh <model.gguf> moe-99
```

## Docker

```bash
# Build multi-stage image (~3 GB runtime)
docker buildx build -t llama-cpp-rdna2:v0.3.2-p3-bfe-stable -f docker/Dockerfile.rdna2 .

# Run with GPU passthrough
docker compose -f docker/docker-compose.rdna2.yml up
```

## Next Steps

- [RDNA2 Flags Reference](RDNA2_FLAGS.md) — detailed flag documentation
- [RDNA2 Experimental Docs](rdna2-experimental.md) — MoE prefill accelerator details
- [Research Log](RESEARCH_LOG.md) — DPP revert rationale, BFE validation gates
- [Forensic Audit](../benchmarks/forensic_audit/FINAL_REPORT.md) — v0.3.0→v0.3.1 regression analysis