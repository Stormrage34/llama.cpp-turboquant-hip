# HARDWARE TARGET — llama.cpp-turboquant-hip

**Target Platform:** AMD RX 6800 XT (gfx1030 / Navi 21) + AMD Ryzen 7 5700X (Zen 3)  
**Role:** Core configuration file — physical constraints are non-negotiable optimization rules  
**Maintainer:** @librarian  
**Last Updated:** 2026-05-21

---

## Section A: Hardware Topology Matrix

```
+--------------------------------------------------------------------+
|                     SYSTEM TOPOLOGY (RX 6800 XT + Ryzen 7 5700X)   |
|                                                                     |
|   HOST (Zen 3)                          DEVICE (RDNA2)             |
|   +---------------------------+         +------------------------+  |
|   |   Ryzen 7 5700X          |         |   AMD RX 6800 XT       |  |
|   |   8C/16T, 1 CCD          |         |   Navi 21 / gfx1030    |  |
|   |   +--------+--------+    |         |   +------------------+  |  |
|   |   | L1$    | L1$    |    |         |   | 72 Compute Units |  |  |
|   |   | 32KB   | 32KB   |    |         |   | (72 CU × 64 ALU) |  |  |
|   |   +--------+--------+    |         |   | = 4608 SP ALUs   |  |  |
|   |   +--------+--------+    |         |   | Wave32 native    |  |  |
|   |   | L2$    | L2$    |    |         |   +--------+---------+  |  |
|   |   | 512KB  | 512KB  |    |         |            |            |  |
|   |   +--------+--------+    |         |   +--------v---------+  |  |
|   |   +------------------+   |         |   | L0/L1$: 64B line |  |  |
|   |   | L3$: 32MB        |   |         |   | L2 TCC: 128B     |  |  |
|   |   | (unified, 1 CCD) |   |         |   |    line          |  |  |
|   |   +--------+---------+   |         |   +--------+---------+  |  |
|   |            |             |         |            |            |  |
|   |   DDR4     |             |         |   +--------v---------+  |  |
|   |   DRAM     |             |         |   | Infinity Cache   |  |  |
|   |   (host)   |             |         |   | 128 MB (L3 GPU)  |  |  |
|   +-----+------+             |         |   +--------+---------+  |  |
|         |                    |         |            |            |  |
|         |     PCIe Gen4      |         |   +--------v---------+  |  |
|         +─────x16────────────+─────────+──▶│ GDDR6 VRAM       |  |  |
|                                              | 16 GB @ 16 Gbps |  |  |
|                                              +------------------+  |  |
|                                                                     |  |
|   Data Path:                                                        |  |
|   NVMe ──(mmap)──▶ DRAM(host) ──(pcie)──▶ VRAM ──(inf.cache)──▶ CU |  |
|   (zero-copy, no host staging buffer)                               |  |
+--------------------------------------------------------------------+
```

**Key:**
- `L1$/L2$`: CPU cache hierarchy
- `L0/L1/L2 TCC`: GPU cache hierarchy
- `Infinity Cache`: GPU L3, 128 MB on-die SRAM
- `CU`: Compute Unit (64 ALUs, 64KB VGPR, 64KB LDS)
- Wave32: Native execution mode for gfx1030 (vs Wave64 on older GCN)

---

## Section B: VRAM Fence & Host Offloading Rules (15.5 GB)

### Absolute VRAM Ceiling
| Parameter | Value | Notes |
|-----------|-------|-------|
| VRAM total | 16,384 MiB | RX 6800 XT GDDR6 |
| Driver/system headroom | ~512 MiB | Reserved for display, kernel, DMA |
| **VRAM_ABSOLUTE_MAX** | **15,872 MiB (15.5 GiB)** | Hard ceiling for all allocations |
| VRAM_YELLOW_ALERT | 15,360 MiB (15.0 GiB) | Warning threshold, requires justification |

### Offloading Policy
- **All hybrid CPU/GPU offloading proposals are classified as INFEASIBLE.**
- 100% GPU residency mandatory for:
  - Model weights (all layers, all experts)
  - KV-cache (key + value tensors)
  - Compute scratchpads (attention, MoE routing, temp buffers)
- **Rationale:** Confirmed upstream MoE CPU offloading bug (CR-015) causes garbage output after ~1000 tokens with any `--n-cpu-moe N`. RX 6800 XT has sufficient VRAM (15.5 GB usable) for the target model class (35B MoE @ IQ4_XS ≈ 9 GB + KV-cache ~2-4 GB at 32K-128K context).

### Zero-Copy mmap Configuration
- Model file on NVMe → `mmap()` into page-locked host memory → GPU DMA directly from mmap'd pages
- No host staging buffer (eliminates redundant DRAM allocation + copy)
- Enforced via: `llama.cpp` mmap backend with `MADV_SEQUENTIAL` + `MADV_WILLNEED`
- Benefits: Reduced peak host memory, faster model load, lower TLB pressure

---

## Section C: RDNA2 ISA Execution Rules

### Rule 1: L2 Cache Line Alignment (128-byte)
All weight matrices, quantization scales, and metadata tables **must** be aligned to 128-byte boundaries.

| Component | Alignment Requirement | Rationale |
|-----------|----------------------|-----------|
| Weight block columns | Multiple of 128B | L2 TCC cache line is 128B; misaligned access wastes 50% bandwidth |
| Quantization scales | 128B-aligned per group | Scales loaded concurrently with weights; must occupy same cache line |
| Metadata/header tables | 128B-aligned start + size | Prevents L2 line splits for async prefetch |
| Multi-channel data | Padded to clean 128B multiple | SoA interleave must respect cache line boundaries |

**Reference:** RDNA2 TCC (L2) cache line = 128 bytes. RDNA2 L0/L1 cache line = 64 bytes.
**Violation penalty:** Wastes up to 50% of L2 bandwidth due to sector-masked partial line fills.

### Rule 2: Wave32 Execution (Mandatory for gfx1030)
| Aspect | Requirement |
|--------|-------------|
| Execution path | Wave32 (32 lanes per wave) |
| Granularity | All compute kernels on gfx1030 |
| VGPR allocation | Half of Wave64 → higher occupancy per CU |
| CU occupancy | Up to 2× vs Wave64 for register-bound kernels |

**Why:** gfx1030 is a Wave32-native architecture. Wave64 support requires instruction pairing (every 2 Wave32 instructions → 1 Wave64 instruction), reducing effective throughput. All GPU kernel launch configurations must specify `__launch_bounds__` or implicit Wave32 via the compiler backend.

### Rule 3: LDS-Free Within-Wave Data Shuffling
**Hard restriction:** `__shared__` / LDS must NOT be used for pure within-wave data permutation or shuffle operations.

| Allowed | Forbidden |
|---------|-----------|
| `__shfl_sync()` (VGPR-to-VGPR) | `__shared__` for data reordering within a wave |
| `__shfl_down_sync()` | `__syncthreads()` for intra-wave synchronization |
| `__shfl_up_sync()` | LDS as staging buffer for permute operations |
| `__shfl_xor_sync()` | |
| `DS_PERMUTE_B32` (ISA intrinsic) | |
| `DS_BPERMUTE_B32` (ISA intrinsic) | |

**Rationale:** LDS is a scarce resource on gfx1030 (64 KB per CU, shared by all waves on a CU). Using LDS for within-wave shuffling starves large-tile tensor kernels that need LDS for block tile layouts. VGPR-based shuffling via `__shfl_sync` or DS permute intrinsics is zero-overhead (same cycle) and conserves LDS bandwidth.

**LDS is reserved for:** Large-tile matmul accumulators (`mmq.cuh`), flash attention tile layouts, and any cross-wave cooperative load.

---

## Section D: Zen 3 Host Alignment (32 MB L3)

### Cache Topology
| Level | Size | Associativity | Line Size | Sharing |
|-------|------|---------------|-----------|---------|
| L1 | 32 KB (data) + 32 KB (instruction) | 8-way | 64B | Per core |
| L2 | 512 KB | 8-way | 64B | Per core |
| **L3** | **32 MB (unified)** | **16-way** | **64B** | **All cores (1 CCD)** |

### Scheduling Data Budget
The Ryzen 7 5700X is a single-CCD part with unified 32 MB L3. The following host-side structures **must** collectively fit within this budget:

| Structure | Target Size | Type | Notes |
|-----------|-------------|------|-------|
| MoE expert routing tables | ≤ 512 KB | `uint16_t[]` | Per-layer, flat index maps |
| Token index arrays | ≤ 256 KB | `uint32_t[]` | Sequence positions, batch slots |
| Graph compilation state | ≤ 1 MB | bitfields + offsets | Node availability, edge traversal |
| Copy slot descriptors | ≤ 128 KB | struct | backend.cpp copy pipeline state |
| Async barrier flags | ≤ 64 KB | `uint32_t[]` | Per-stream semaphores |

### Data Type Mandates
| Element | Required Type | Rationale |
|---------|---------------|-----------|
| MoE expert indices | `uint16_t` | 65,535 expert max; fits in half the cache lines of `uint32_t` |
| Token positions | `uint16_t` or packed | Sequence length ≤ 65535 for target use cases |
| Graph node IDs | `uint16_t` | Flat index layout avoids pointer chasing |
| Routing scores | `float` (required) | Precision needed for MoE gating — no quantization |

**Enforcement:** Host-side scheduling structures exceeding 32 MB total L3 capacity must be restructured or tiled. Performance degradation above the L3 budget is proportional to DRAM latency penalty (~100-150 ns vs L3 ~10-15 ns).

---

## Section E: Data Layout vs Execution Path Validation Matrix

| Quant Format | L2 Alignment (128B) | Wave32 Kernel | LDS-Free Shuffle | SoA Layout | Verification Status |
|---|---|---|---|---|---|
| Q4_K_M | Required | Required | Required | Required | TBD |
| Q5_K_M | Required | Required | Required | Required | TBD |
| IQ4_XS | Required | Required | Required | Required | TBD* |
| Q8_0 | Required | Required | Required | N/A (symmetric) | TBD |
| F16 | Required | Required | Required | N/A | TBD |

**`*` CR-013 (open):** IQ4_XS has a confirmed Array-of-Structures (AoS) vs Structure-of-Arrays (SoA) layout mismatch in `load_tiles_iq4_xs_swizzled` at `mmq.cuh:3291`. The meta offset computation for sub-tile views may produce incorrect addresses when swizzling is enabled. This blocks full IQ4_XS swizzle activation on the MMQ path. See `opencode/proposals/CR-013.md`.

### Legend
- **Required:** Must be satisfied for correct and performant execution on target hardware
- **N/A:** Format does not use this layout (e.g., Q8_0 is symmetric, no SoA conversion needed)
- **TBD:** Verification not yet performed (gate: CR-022 5K-Coherence pass)
- **TBD*:** Known issue; blocked on CR-013 resolution

---

## Section F: Hardware Constants Registry

| Constant | Value | Unit | Source |
|---|---|---|---|
| VRAM_ABSOLUTE_MAX | 15872 | MiB | RX 6800 XT driver headroom (16GB - 512MB) |
| VRAM_YELLOW_ALERT | 15360 | MiB | 15 GB warning threshold |
| L2_CACHE_LINE | 128 | bytes | RDNA2 TCC (Last-Level Cache) |
| L1_CACHE_LINE | 64 | bytes | RDNA2 L0/L1 data cache |
| INFINITY_CACHE | 128 | MiB | RX 6800 XT on-die SRAM (L3 equivalent) |
| WAVE_SIZE | 32 | lanes | RDNA2 native wavefront size |
| L3_CACHE_HOST | 32 | MiB | Ryzen 7 5700X unified L3 (single CCD) |
| PCIE_GEN | 4 | — | CPU ↔ GPU link generation |
| GPU_COMPUTE_UNITS | 72 | CUs | gfx1030 (Navi 21, RX 6800 XT) |
| GPU_MAX_CLOCK | 2250 | MHz | RX 6800 XT game clock (typical) |
| GPU_MEM_CLOCK | 2000 | MHz | GDDR6 effective (16 Gbps) |
| GPU_MEM_BUS | 256 | bit | GDDR6 memory bus width |
| GPU_MEM_BANDWIDTH | 512 | GB/s | Peak theoretical (16 Gbps × 256-bit / 8) |
| INFINITY_CACHE_BANDWIDTH | 2048 | GB/s | Estimated effective bandwidth with IC |
| HOST_MEM_BANDWIDTH | ~45 | GB/s | DDR4-3200 dual-channel typical |
| TDP | 300 | W | RX 6800 XT board power |
| TDP_CPU | 65 | W | Ryzen 7 5700X TDP |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-21 | Initial creation — hardware topology, VRAM fence, ISA rules, Zen 3 alignment, validation matrix, constants registry | @librarian |

---

## Related Documents

- `opencode/project-state.md` — Project state and active proposals
- `opencode/proposals/CR-013.md` — IQ4_XS SoA/AoS mismatch
- `opencode/proposals/CR-015.md` — Upstream MoE CPU offloading bug
- `opencode/proposals/CR-020.md` — Swizzle-All Binary Initiative
- `opencode/agents/DEEP_ISA_MISSION.md` — ISA roadmap
- `AGENTS.md` — Agent workflow guide
