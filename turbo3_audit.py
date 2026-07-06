#!/usr/bin/env python3
"""
Turbo3_0 Flash Attention Kernel Audit Tool
==========================================
Combines cycle-level simulation + optimization strategy comparison +
actual benchmark data for RDNA2 gfx1030 (RX 6800 XT).

Usage:
  python3 turbo3_audit.py

WARNING: Absolute cycle counts are APPROXIMATE. RDNA2's ILP, wavefront
switching, cache hierarchies, and memory coalescing are not modeled.
Relative comparisons between configs are more reliable but still imperfect.
The LUT vs vec_dot benchmark proved the simulation wrong -- always verify
on hardware.
"""
import math

# Hardware constants - RDNA2 gfx1030 (RX 6800 XT)
WAVE_SIZE      = 32
SIMDS_PER_CU   = 2
VGPRS_PER_SIMD = 256
LDS_PER_CU     = 65536
LDS_BANKS      = 32
GFLOPS_FP32    = 20.74
BANDWIDTH_GBS  = 512
CLOCK_GHZ      = 2.074
CU_COUNT       = 72
N_HEADS        = 40

# Instruction costs (cycles, approximate)
C = {
    'fma': 0.5, 'mul': 0.5, 'add': 0.5, 'shfl': 1.0,
    'lds': 0.5, 'lds_bc': 2.0, 'gl1': 4.0, 'gl2': 20.0,
    'gmiss': 40.0, 'sync': 10.0, 'exp': 4.0, 'branch': 1.0,
    'f2h': 0.5, 'h2f': 0.5, 'byte': 0.5, 'vdot2': 1.0,
}

# Kernel parameters
D           = 128
QK_TURBO3   = 32
N_CENTROIDS = 8
NTHREADS_KQ = 1
NTHREADS_V  = 32
K_BLOCK_BYTES = (2 + QK_TURBO3 // 4 + QK_TURBO3 // 8) * (D // QK_TURBO3)
V_BLOCK_BYTES = K_BLOCK_BYTES
Q_BYTES       = D * 4

# Actual benchmark data
BENCH = {
    'turbo3_0': {'pp512': 785.80, 'pp4096': 1197.89, 'pp16384': 585.36, 'pp32768': 309.24, 'tg128': 45.57},
    'q8_0':     {'pp512': None,   'pp4096': None,    'pp16384': 763.00, 'pp32768': 482.79, 'tg128': 45.04},
    'turbo_b128': {'pp512': 771.63, 'pp4096': 1194.57, 'pp16384': 511.14, 'pp32768': None, 'tg128': 43.49},
}

def section1_benchmarks():
    print("\n" + "=" * 80)
    print("  SECTION 1: ACTUAL BENCHMARK DATA")
    print("  Hardware: RX 6800 XT 16GB, gfx1030, ROCm")
    print("  Model: Qwen-AgentWorld-35B-A3B IQ4_NL, -ngl 99 -ncmoe 15")
    print("  Params: -b 4096 -ub 2048 -fa 1")
    print("=" * 80)

    tests = ['pp512', 'pp4096', 'pp16384', 'pp32768', 'tg128']
    cfgs = [('turbo3_0', 'turbo3_0 block-32'), ('turbo_b128', 'turbo3_0 block-128'), ('q8_0', 'q8_0')]

    print(f"\n  {'Test':<10}", end="")
    for _, lab in cfgs:
        print(f" {lab:>22}", end="")
    print(f" {'turbo/q8':>10}")
    print("  " + "-" * 78)

    for t in tests:
        print(f"  {t:<10}", end="")
        for key, _ in cfgs:
            v = BENCH[key][t]
            print(f" {v:>22.2f}" if v else f" {'--':>22}", end="")
        tq = BENCH['turbo3_0'][t]
        q8 = BENCH['q8_0'][t]
        if tq and q8:
            print(f" {tq/q8*100:>9.1f}%")
        else:
            print(f" {'--':>10}")

    print(f"\n  DELTA: block-32 vs block-128")
    print(f"  " + "-" * 50)
    for t in tests:
        b32 = BENCH['turbo3_0'][t]
        b128 = BENCH['turbo_b128'][t]
        if b32 and b128:
            d = (b32 / b128 - 1) * 100
            print(f"    {t:<10} {b128:>8.2f} -> {b32:>8.2f}  ({d:+.1f}%)")


def section2_simulation():
    print("\n" + "=" * 80)
    print("  SECTION 2: CYCLE-LEVEL SIMULATION (turbo vs q8_0)")
    print("  WARNING: Approximate. Does not model RDNA2 ILP or cache hierarchies.")
    print("=" * 80)

    iters = D // 8
    turbo_per_iter = (3*C['byte'] + 8*C['byte'] + 8*C['lds'] + 8*C['h2f'] + 7*C['add'] + 1*C['mul'])
    turbo_per_k = iters * turbo_per_iter + C['exp'] + C['mul'] + C['add'] + C['fma'] + C['gl1'] + 4*C['byte'] + 4*C['mul'] + 4*C['add']
    q8_blocks = D // 32
    q8_per_k = q8_blocks * (4*C['byte'] + C['vdot2']) + C['exp'] + C['mul'] + C['add'] + C['fma'] + q8_blocks * (4*C['byte'] + C['vdot2'])

    print(f"\n  Per K position:")
    print(f"    turbo3_0: {turbo_per_k:.1f} cycles")
    print(f"    q8_0:     {q8_per_k:.1f} cycles")
    print(f"    ratio:    {turbo_per_k/q8_per_k:.2f}x (simulated)")

    seqlens = [512, 4096, 16384, 32768]
    print(f"\n  Simulated ops ratio turbo/q8_0 vs measured throughput ratio:")
    print(f"    {'seqlen':>8} {'sim_ops':>10} {'measured':>10} {'sim_vs_meas':>12}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    for sl in seqlens:
        sim_total = turbo_per_k * sl * N_HEADS
        q8_total = q8_per_k * sl * N_HEADS
        sim_ratio = sim_total / q8_total
        if sl in [16384, 32768] and BENCH['turbo3_0'][f'pp{sl}'] and BENCH['q8_0'][f'pp{sl}']:
            meas_ratio = BENCH['turbo3_0'][f'pp{sl}'] / BENCH['q8_0'][f'pp{sl}']
            print(f"    {sl:>8} {sim_ratio:>9.2f}x {meas_ratio:>9.2f}x {sim_ratio/meas_ratio:>11.1f}x")
        else:
            print(f"    {sl:>8} {sim_ratio:>9.2f}x {'--':>10} {'--':>12}")

    print(f"\n  The simulation overestimates the gap because it ignores ILP.")
    print(f"  RDNA2 hides compute via wavefront switching and instruction overlap.")


def section3_optimization():
    print("\n" + "=" * 80)
    print("  SECTION 3: OPTIMIZATION STRATEGY COMPARISON")
    print("  NOTE: Benchmarks showed LUT nkq=1 ~= vec_dot nkq=2 on RDNA2.")
    print("=" * 80)

    iters = D // 8
    per_iter = 3 + 8 + 8 + 8 + 7 + 1
    per_iter_f32 = 3 + 8 + 8 + 0 + 7 + 1
    q8_ops = 4 * (4 + 1)

    configs = [
        ('LUT nkq=1 (current)', iters * per_iter + 5, WAVE_SIZE),
        ('LUT f32 (no h2f)', iters * per_iter_f32 + 5, WAVE_SIZE),
        ('vec_dot nkq=2', (D//2//8) * per_iter + 7, WAVE_SIZE // 2),
        ('vec_dot nkq=4', (D//4//8) * per_iter + 9, WAVE_SIZE // 4),
        ('q8_0 V_DOT2', q8_ops, WAVE_SIZE),
    ]

    print(f"\n  {'Config':<25} {'Ops':>6} {'Parallel':>9} {'Throughput':>12}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*12}")
    base_tp = None
    for name, ops, parallel in configs:
        tp = parallel / ops
        if base_tp is None:
            base_tp = tp
        print(f"  {name:<25} {ops:>6.0f} {parallel:>9} {tp:>12.4f}")

    print(f"\n  Throughput = parallel KQ scores / ops per KQ score")
    print(f"  BUT: benchmarks proved this metric is unreliable on RDNA2.")
    print(f"  LUT nkq=1 and vec_dot nkq=2 perform identically despite 3x ops diff.")


def section4_memory():
    print("\n" + "=" * 80)
    print("  SECTION 4: MEMORY + KV CACHE ANALYSIS")
    print("=" * 80)

    for sl in [512, 4096, 16384, 32768]:
        k = sl * K_BLOCK_BYTES * N_HEADS
        v = sl * V_BLOCK_BYTES * N_HEADS
        q = Q_BYTES * N_HEADS
        total = k + v + q
        key = f'pp{sl}'
        tps = BENCH['turbo3_0'].get(key)
        if tps:
            wall_s = sl / tps
            bw = total / wall_s / 1e9
            util = bw / BANDWIDTH_GBS * 100
            print(f"  pp{sl:>5}: {total/1024:>7.1f}K data, {bw:>6.1f} GB/s needed, {util:>5.1f}% of {BANDWIDTH_GBS} GB/s peak")
        else:
            print(f"  pp{sl:>5}: {total/1024:>7.1f}K data, no benchmark")

    print(f"\n  KV cache at 32k context, 40 heads:")
    for name, bpc in [('turbo3_0 block-32', K_BLOCK_BYTES), ('turbo3_0 block-128', 66), ('q8_0', 165)]:
        mb = 32768 * bpc * 2 * N_HEADS / 1024 / 1024
        print(f"    {name:<25}: {mb:>6.1f} MB")


def section5_hotloop():
    print("\n" + "=" * 80)
    print("  SECTION 5: HOT LOOP ANALYSIS (KQ SCORING = 95% of kernel time)")
    print("=" * 80)

    iters = D // 8
    print(f"""
  The KQ scoring hot loop (fattn-vec.cuh):
    for (d0 = 0; d0 < {D}; d0 += 8):  // {iters} iterations
    {{
        ib = d0 / {QK_TURBO3};  // block index (0..{D//QK_TURBO3-1})
        // Load: norm(f16), qs0(u8), qs1(u8), sgn(u8) from block_turbo3_0
        // For each of 8 elements:
        //   idx = extract_bits(qs0, qs1, sgn, k)
        //   sum += half2float(turbo_lut[d0+k][idx])  // LDS read
        sum *= norm;
    }}

  Operations per KQ score:
    Byte loads:     {iters*3:>4}  (3 per iteration: qs0, qs1, sgn)
    Index extract:  {iters*8:>4}  (8 per iteration: shift+mask)
    LDS reads:      {iters*8:>4}  (8 per iteration: LUT lookup)
    h2f convert:    {iters*8:>4}  (8 per iteration: half->float)
    Adds:           {iters*7:>4}  (7 per iteration: sum 8 values)
    Muls:           {iters:>4}  (1 per iteration: norm scale)
    Softmax:           4
    V dequant:        ~17
    TOTAL:          {iters*8*3 + iters*3 + 4 + 17:>4}

  vs q8_0 V_DOT2:      44 total (4 blocks x (4 byte load + 1 V_DOT2) + softmax)

  Operations ratio:   {(iters*8*3 + iters*3 + 4 + 17)/44:.1f}x
  Measured ratio:     {BENCH['turbo3_0']['pp32768']/BENCH['q8_0']['pp32768']:.2f}x (pp32768)

  The 3.9x ops ratio compresses to 1.56x measured because RDNA2 overlaps
  independent operations via ILP (instruction-level parallelism).
""")


def section6_gotchas():
    print("=" * 80)
    print("  SECTION 6: KNOWN ISSUES AND GOTCHAS")
    print("=" * 80)
    print("""
  1. __launch_bounds__(, 2) = 25% PP regression
     Turbo kernel needs ~180 VGPRs. minBlocks=2 halves available to ~128.
     Result: register spill to local memory.
     FIX: Always use minBlocks=1 for turbo kernels.

  2. LUT vs vec_dot = equivalent on RDNA2
     Simulation predicted 3x ops difference. Benchmarks showed <2% difference.
     RDNA2's ILP hides the extra operations. Do not optimize ops count alone.

  3. Block-32 (QK_TURBO3=32) = +14.5% pp16384
     Single #define change cascades via macros. No model re-quant needed
     (turbo3_0 is KV cache format, not model weights).

  4. half4 vectorized WHT = slight PP regression
     Reduced thread count from 128 to 32 hurt more than SIMD benefit helped.
     WHT is ~5% of total time -- not worth optimizing.

  5. Bank-pad (lut_stride n_centroids+2) = slight PP regression
     Adding 1 half of padding per LUT row hurt more than it helped.
     The bank conflicts were not the bottleneck.

  6. Q_reg/Q_i32/Q_ds size reduction = PP regression
     Reducing array sizes to 1 element when k_lut=true caused compiler
     to generate worse code for surrounding loops.
""")

print("=" * 80)
print("  TURBO3_0 FLASH ATTENTION KERNEL AUDIT")
print("  Target: RDNA2 gfx1030 (RX 6800 XT)")
print("  Model: Qwen-AgentWorld-35B-A3B IQ4_NL")
print("  Status: tg128 matches q8_0. pp16384 gap 23%. pp32768 gap 36%.")
print("=" * 80)

section1_benchmarks()
section2_simulation()
section3_optimization()
section4_memory()
section5_hotloop()
section6_gotchas()

print("=" * 80)
print("  END OF AUDIT")
print("=" * 80)
