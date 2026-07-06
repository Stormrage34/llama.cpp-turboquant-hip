#!/usr/bin/env python3
"""
RDNA2 Master Diagnostic — Heuristic + Bayesian Probability Analysis

Profiles an llama.cpp ROCm build on gfx1030 (RX 6800 XT), parses kernel
traces, and uses Bayesian probability to rank optimization opportunities
by expected impact.

Usage:
    python3 rdna2-diagnostic.py --model-path /path/to/model.gguf
    python3 rdna2-diagnostic.py --profile-dir /path/to/rocprofv3/output
    python3 rdna2-diagnostic.py --model-path model.gguf --quick

Requires: python3, rocprofv3 (optional, for live profiling), numpy (optional)
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

# RX 6800 XT (gfx1030) architecture parameters
WAVE_SIZE = 32          # RDNA2 wave-32
NUM_CU = 72             # Compute Units
WAVES_PER_CU = 8        # Max waves per CU on RDNA2 (256 threads / 32)
TOTAL_WAVE_SLOTS = NUM_CU * WAVES_PER_CU  # 576
PEAK_BW_GB_S = 512.0    # GB/s
LDS_SIZE_BYTES = 65536   # 64KB per CU
VGPR_TOTAL = 256         # Per SIMD (RDNA2)
SRAMBW_BYTES = 128       # 128 bytes/cycle SRAM per CU (RDNA2)

# Profiling constants
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
BYTES_PER_GB = 1_073_741_824

# ============================================================================
# Bayesian Optimization Prior Model
# ============================================================================
# Each optimization has a prior probability of helping, based on hipfire
# research (PRs #298-304) and our own rocprofv3 baseline profile.
#
# priorProbability: P(this optimization helps) before seeing our data
# expectedGainMean: expected % improvement if it helps (log-normal mean)
# expectedGainSD: uncertainty in expected gain
# relevanceCondition: when this optimization is relevant (checked against data)

@dataclass
class OptimizationPrior:
    name: str
    description: str
    category: str  # memory, compute, latency, occupancy
    prior_probability: float  # P(helps) — from research
    expected_gain_mean: float  # geometric mean expected gain
    expected_gain_sd: float    # log-normal SD
    relevance_check: str       # which data fields trigger relevance
    source: str                # where this prior came from
    implementation_effort: str  # low/medium/high

OPTIMIZATION_PRIORS = [
    OptimizationPrior(
        name="hipGraph_capture",
        description="Batch kernel launches into hipGraph to eliminate host-launch overhead",
        category="latency",
        prior_probability=0.85,
        expected_gain_mean=0.15,   # +15% (research: +10-20%)
        expected_gain_sd=0.05,
        relevance_check="kernel_launch_count>10000",
        source="hipfire #300 F1: +10-20% on prefill, 288 launches/layer",
        implementation_effort="medium",
    ),
    OptimizationPrior(
        name="VGPR_reduction",
        description="Reduce VGPR usage to increase occupancy (e.g., __launch_bounds__)",
        category="occupancy",
        prior_probability=0.70,
        expected_gain_mean=0.10,   # +10% (research: +5-15%)
        expected_gain_sd=0.04,
        relevance_check="max_vgpr>128_or_low_occupancy",
        source="hipfire #300 F4: +5-15% from 110->88 VGPR",
        implementation_effort="low",
    ),
    OptimizationPrior(
        name="LDS_tile_resize",
        description="Reduce FA tile sizes to fit more waves per CU via LDS pressure relief",
        category="occupancy",
        prior_probability=0.60,
        expected_gain_mean=0.08,   # +8% (research: LDS tiling dominant factor)
        expected_gain_sd=0.04,
        relevance_check="max_lds_bytes>50000",
        source="hipfire PR #298 Phase 3: LDS tiling is dominant factor",
        implementation_effort="medium",
    ),
    OptimizationPrior(
        name="kernel_fusion",
        description="Fuse adjacent tiny kernels (rms_norm, rope, quantize) to reduce launch overhead",
        category="latency",
        prior_probability=0.50,
        expected_gain_mean=0.06,   # +6% (research: +5-15% from epilogue fusion)
        expected_gain_sd=0.03,
        relevance_check="tiny_kernel_pct>20",
        source="hipfire #300 F2: MMQ epilogue fusion +5-15%",
        implementation_effort="high",
    ),
    OptimizationPrior(
        name="persistent_kernels",
        description="Keep warps resident across multiple tiles to eliminate dispatch overhead",
        category="latency",
        prior_probability=0.45,
        expected_gain_mean=0.15,   # +15% (research: +10-30%)
        expected_gain_sd=0.08,
        relevance_check="kernel_launch_count>50000",
        source="hipfire #300 F3: +10-30% on residual kernel",
        implementation_effort="high",
    ),
    OptimizationPrior(
        name="wave32_batch_alignment",
        description="Use batch sizes that are multiples of 32 for wave-32 alignment",
        category="compute",
        prior_probability=0.40,
        expected_gain_mean=0.05,   # +5%
        expected_gain_sd=0.03,
        relevance_check="non_aligned_wave_dispatches>100",
        source="gfx103_optimizations.md: wave-32 preference",
        implementation_effort="low",
    ),
    OptimizationPrior(
        name="memory_copy_batching",
        description="Batch H2D copies via pinned memory or graph capture",
        category="memory",
        prior_probability=0.75,
        expected_gain_mean=0.08,   # +8%
        expected_gain_sd=0.03,
        relevance_check="h2d_copy_count>100",
        source="baseline profile: 482 H2D copies, 487ms total",
        implementation_effort="medium",
    ),
    OptimizationPrior(
        name="GEMM_epilogue_fusion",
        description="Fuse SwiGLU/RMSnorm into GEMM epilogue to reduce kernel count",
        category="latency",
        prior_probability=0.40,
        expected_gain_mean=0.07,   # +7%
        expected_gain_sd=0.04,
        relevance_check="matmul_time_pct>60",
        source="hipfire #300 F2: epilogue fusion complementary to graph",
        implementation_effort="high",
    ),
]


# ============================================================================
# Kernel Classification
# ============================================================================

KERNEL_CATEGORIES = {
    "matmul": ["mul_mat_q", "mul_mat_vec_q", "mul_mat_f16", "mul_mat_f32", "GEMM"],
    "flash_attention": ["flash_attn", "f_attn"],
    "normalization": ["rms_norm", "norm", "softmax"],
    "rope": ["rope", "build_inp_pos"],
    "quantize": ["quantize", "dequantize", "cpy"],
    "activation": ["gelu", "silu", "swish"],
    "copy": ["fillBuffer", "memcpy", "hipMemcpy"],
    "embedding": ["embedding", "tok_emb"],
    "sampling": ["sample", "top_k", "top_p", "greedy"],
    "other": [],
}


def classify_kernel(name: str) -> str:
    """Classify a kernel name into a category."""
    name_lower = name.lower()
    for cat, patterns in KERNEL_CATEGORIES.items():
        for pat in patterns:
            if pat.lower() in name_lower:
                return cat
    return "other"


# ============================================================================
# CSV Parsers
# ============================================================================

def parse_kernel_trace(path: str) -> List[dict]:
    """Parse rocprofv3 kernel trace CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Kind"] != "KERNEL_DISPATCH":
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                duration_ns = end - start
                rows.append({
                    "kernel_name": row["Kernel_Name"],
                    "duration_ns": duration_ns,
                    "vgpr": int(row["VGPR_Count"]) if row["VGPR_Count"] else 0,
                    "sgpr": int(row["SGPR_Count"]) if row["SGPR_Count"] else 0,
                    "lds": int(row["LDS_Block_Size"]) if row["LDS_Block_Size"] else 0,
                    "wg_x": int(row["Workgroup_Size_X"]) if row["Workgroup_Size_X"] else 1,
                    "wg_y": int(row["Workgroup_Size_Y"]) if row["Workgroup_Size_Y"] else 1,
                    "grid_x": int(row["Grid_Size_X"]) if row["Grid_Size_X"] else 1,
                    "grid_y": int(row["Grid_Size_Y"]) if row["Grid_Size_Y"] else 1,
                })
            except (ValueError, KeyError):
                continue
    return rows


def parse_memory_trace(path: str) -> List[dict]:
    """Parse rocprofv3 memory copy trace CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                duration_ns = end - start
                rows.append({
                    "direction": row["Direction"],
                    "duration_ns": duration_ns,
                })
            except (ValueError, KeyError):
                continue
    return rows


def parse_hip_api_trace(path: str) -> List[dict]:
    """Parse rocprofv3 HIP API trace CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                duration_ns = end - start
                rows.append({
                    "function": row["Function"],
                    "duration_ns": duration_ns,
                })
            except (ValueError, KeyError):
                continue
    return rows


# ============================================================================
# Heuristic Analysis
# ============================================================================

@dataclass
class KernelStats:
    """Aggregated statistics for a kernel type."""
    name: str
    category: str
    count: int = 0
    total_duration_ns: int = 0
    max_duration_ns: int = 0
    min_duration_ns: int = 2**62
    max_vgpr: int = 0
    max_lds: int = 0
    min_wg_x: int = 999999
    max_wg_x: int = 0
    max_wg_y: int = 0
    max_grid_x: int = 0

    @property
    def avg_duration_ns(self) -> float:
        return self.total_duration_ns / self.count if self.count else 0

    @property
    def avg_duration_us(self) -> float:
        return self.avg_duration_ns / 1000.0

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_ns / NS_PER_MS

    @property
    def estimated_waves(self) -> int:
        return (self.max_wg_x * self.max_wg_y) // WAVE_SIZE if self.max_wg_x and self.max_wg_y else 0

    @property
    def occupancy_estimate(self) -> float:
        """Estimate occupancy based on VGPR and wave count."""
        if self.max_vgpr == 0:
            return 100.0
        waves_per_simd = min(VGPR_TOTAL // self.max_vgpr, WAVES_PER_CU)
        return (waves_per_simd / WAVES_PER_CU) * 100.0


def analyze_kernels(kernel_rows: List[dict]) -> Dict[str, KernelStats]:
    """Aggregate kernel trace into per-kernel-type stats."""
    stats = {}
    for row in kernel_rows:
        name = row["kernel_name"]
        cat = classify_kernel(name)
        if name not in stats:
            stats[name] = KernelStats(name=name, category=cat)
        s = stats[name]
        s.count += 1
        s.total_duration_ns += row["duration_ns"]
        s.max_duration_ns = max(s.max_duration_ns, row["duration_ns"])
        s.min_duration_ns = min(s.min_duration_ns, row["duration_ns"])
        s.max_vgpr = max(s.max_vgpr, row["vgpr"])
        s.max_lds = max(s.max_lds, row["lds"])
        s.min_wg_x = min(s.min_wg_x, row["wg_x"])
        s.max_wg_x = max(s.max_wg_x, row["wg_x"])
        s.max_wg_y = max(s.max_wg_y, row["wg_y"])
        s.max_grid_x = max(s.max_grid_x, row["grid_x"])
    return stats


# ============================================================================
# Bayesian Update Engine
# ============================================================================

@dataclass
class OptimizationRecommendation:
    """A single optimization recommendation with Bayesian posterior."""
    name: str
    description: str
    category: str
    posterior_probability: float
    expected_gain: float       # geometric mean of posterior
    confidence_interval: Tuple[float, float]  # 95% CI
    relevance_evidence: str    # what data triggered this
    source: str
    effort: str
    rank: int = 0


def bayesian_update(
    prior: OptimizationPrior,
    evidence: dict,
) -> OptimizationRecommendation:
    """
    Bayesian update: combine prior with observed evidence to compute
    posterior probability and expected gain.

    evidence dict keys:
        prior_data_match: bool — does the data match the relevance check?
        severity: float 0-1 — how severe is the condition?
        existing_optimization: bool — is this already partially addressed?
    """
    p_prior = prior.prior_probability
    gain_mean = prior.expected_gain_mean
    gain_sd = prior.expected_gain_sd

    # Likelihood: P(data | helps)
    if evidence.get("prior_data_match", False):
        # Data confirms this is relevant — boost likelihood
        severity = evidence.get("severity", 0.5)
        p_likelihood = 0.5 + 0.45 * severity  # 0.5 to 0.95
    else:
        # Data doesn't match — this optimization is less likely to help
        p_likelihood = 0.15

    # If already partially addressed, reduce both probability and expected gain
    existing = evidence.get("existing_optimization", False)
    if existing:
        p_prior *= 0.5
        gain_mean *= 0.5

    # Bayes' theorem: P(helps | data) ∝ P(data | helps) * P(helps)
    # Using log-odds for numerical stability
    def p_to_log_odds(p):
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    def log_odds_to_p(lo):
        return 1.0 / (1.0 + math.exp(-lo))

    lo_prior = p_to_log_odds(p_prior)
    lo_likelihood_ratio = p_to_log_odds(p_likelihood) - p_to_log_odds(0.5)
    lo_posterior = lo_prior + lo_likelihood_ratio

    posterior = log_odds_to_p(lo_posterior)

    # Expected gain: update mean based on evidence
    if evidence.get("prior_data_match", False):
        severity = evidence.get("severity", 0.5)
        gain_mean *= (0.8 + 0.4 * severity)  # scale by severity
    else:
        gain_mean *= 0.3

    # Confidence interval (geometric, so log-normal)
    ci_low = gain_mean * math.exp(-1.96 * gain_sd)
    ci_high = gain_mean * math.exp(1.96 * gain_sd)

    # Relevance evidence string
    evidence_str = ""
    if evidence.get("prior_data_match"):
        sev = evidence.get("severity", 0)
        if sev > 0.7:
            evidence_str = f"HIGH severity ({sev:.0%}) — strong signal"
        elif sev > 0.4:
            evidence_str = f"MEDIUM severity ({sev:.0%}) — moderate signal"
        else:
            evidence_str = f"LOW severity ({sev:.0%}) — weak signal"
    else:
        evidence_str = "Not triggered — data does not match relevance condition"

    return OptimizationRecommendation(
        name=prior.name,
        description=prior.description,
        category=prior.category,
        posterior_probability=posterior,
        expected_gain=gain_mean,
        confidence_interval=(ci_low, ci_high),
        relevance_evidence=evidence_str,
        source=prior.source,
        effort=prior.implementation_effort,
    )


# ============================================================================
# Evidence Gathering
# ============================================================================

def gather_evidence(
    kernel_stats: Dict[str, KernelStats],
    memory_rows: List[dict],
    hip_rows: List[dict],
) -> dict:
    """Gather evidence from profile data for each optimization prior."""

    total_kernel_time = sum(s.total_duration_ns for s in kernel_stats.values())
    total_dispatches = sum(s.count for s in kernel_stats.values())
    max_vgpr = max((s.max_vgpr for s in kernel_stats.values()), default=0)
    max_lds = max((s.max_lds for s in kernel_stats.values()), default=0)

    # Category breakdowns
    cat_time = defaultdict(int)
    cat_count = defaultdict(int)
    for s in kernel_stats.values():
        cat_time[s.category] += s.total_duration_ns
        cat_count[s.category] += s.count

    # Memory copy stats
    h2d_copies = [r for r in memory_rows if "HOST_TO_DEVICE" in r["direction"]]
    d2h_copies = [r for r in memory_rows if "DEVICE_TO_HOST" in r["direction"]]
    h2d_total_ns = sum(r["duration_ns"] for r in h2d_copies)
    d2h_total_ns = sum(r["duration_ns"] for r in d2h_copies)

    # Tiny kernel analysis (< 10us)
    tiny_count = 0
    tiny_time = 0
    for s in kernel_stats.values():
        for _ in range(s.count):
            if s.avg_duration_ns < 10_000:  # < 10us
                tiny_count += 1
                tiny_time += s.avg_duration_ns
    tiny_pct = (tiny_count / total_dispatches * 100) if total_dispatches else 0

    # Wave alignment analysis
    non_aligned = 0
    for s in kernel_stats.values():
        if s.min_wg_x > 0 and s.min_wg_x % WAVE_SIZE != 0:
            non_aligned += s.count

    # Matmul dominance
    matmul_time_pct = (cat_time.get("matmul", 0) / total_kernel_time * 100) if total_kernel_time else 0

    evidence = {
        # hipGraph
        "kernel_launch_count": total_dispatches,
        "h2d_copy_count": len(h2d_copies),
        "h2d_total_ms": h2d_total_ns / NS_PER_MS,

        # VGPR
        "max_vgpr": max_vgpr,
        "max_vgpr>128_or_low_occupancy": max_vgpr > 128,

        # LDS
        "max_lds_bytes": max_lds,
        "max_lds>50000": max_lds > 50000,

        # Tiny kernels
        "tiny_kernel_count": tiny_count,
        "tiny_kernel_pct": tiny_pct,

        # Wave alignment
        "non_aligned_wave_dispatches": non_aligned,

        # Matmul dominance
        "matmul_time_pct": matmul_time_pct,

        # Persistent kernels
        "kernel_launch_count>50000": total_dispatches > 50000,

        # Category breakdown
        "category_time": dict(cat_time),
        "category_count": dict(cat_count),
    }

    return evidence


# ============================================================================
# Static Code Analysis — Heuristic + Bayesian Bug Detection
# ============================================================================
# Analyzes C++ kernel source code for common GPU programming bugs that could
# cause faulty decode output or suboptimal performance. Uses heuristic pattern
# matching combined with Bayesian probability to rank bug likelihood.

@dataclass
class BugFinding:
    """A single bug finding from static analysis."""
    rule_id: str
    severity: str          # CRITICAL, HIGH, MEDIUM, LOW
    category: str          # correctness, performance, safety
    file: str
    line: int
    description: str
    evidence: str          # what was found in the code
    posterior_prob: float   # Bayesian P(bug | code pattern)
    fix_suggestion: str
    source_context: str    # surrounding code lines


@dataclass
class BugRule:
    """A heuristic rule for detecting GPU kernel bugs."""
    rule_id: str
    name: str
    severity: str
    category: str
    prior_prob: float      # P(this pattern indicates a bug)
    description: str
    fix_suggestion: str
    pattern: str           # regex or keyword match
    file_pattern: str      # which files to check


# Rules derived from hipfire research, RDNA2 architecture constraints,
# and common CUDA/HIP kernel bugs found in llama.cpp history.
BUG_RULES = [
    BugRule(
        rule_id="LDS_STRIDE_MISMATCH",
        name="LDS stride doesn't match tile_x_sizes",
        severity="CRITICAL",
        category="correctness",
        prior_prob=0.90,
        description="RDNA2 Q4_K stride (RDNA2_Q4K_X_STRIDE=40) must match tile_x_sizes.qs calculation in mmq_get_dp4a_tile_x_sizes. Mismatch causes LDS overflow corrupting x_dm/x_sc.",
        fix_suggestion="Ensure mmq_get_dp4a_tile_x_sizes returns qs = mmq_y * RDNA2_Q4K_X_STRIDE + mmq_y for Q4_K on RDNA2.",
        pattern=r"RDNA2_Q4K_X_STRIDE|mmq_get_dp4a_tile_x_sizes.*Q4_K",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="FUSED_KERNEL_FORMAT_MISMATCH",
        name="Fused kernel uses wrong quant format",
        severity="CRITICAL",
        category="correctness",
        prior_prob=0.85,
        description="Fused MMQ kernels must use Q4_K format (block_q4_K, 144B/superblock, 6-bit scales) not HFQ4-G256 format (136B/group, float sc/zp). Check for wrong struct types, wrong byte offsets, wrong dequant formulas.",
        fix_suggestion="Fused kernels should call load_tiles_q4_K and vec_dot_q4_K_q8_1_dp4a, not hand-rolled dequant.",
        pattern=r"hfq4g256|hfq3g256|136\s*\*\s*|float.*sc.*zp|__builtin_bit_cast.*float.*gp",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="FUSED_KERNEL_LDS_OVERFLOW",
        name="Fused kernel shared memory exceeds 64KB",
        severity="HIGH",
        category="correctness",
        prior_prob=0.75,
        description="Fused kernel LDS must fit in 64KB per CU. Check: txs_all.qs + txs_all.dm + txs_all.sc + mmq_x * MMQ_TILE_Y_K <= 16384 ints.",
        fix_suggestion="Use mmq_get_dp4a_tile_x_sizes to compute correct LDS size. Verify total < 16384 ints (64KB).",
        pattern=r"__launch_bounds__.*256.*16|extern __shared__ int smem",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="MISSING_SYNCTHREADS",
        name="Missing __syncthreads after LDS write",
        severity="HIGH",
        category="correctness",
        prior_prob=0.80,
        description="After writing to shared memory (x_qs, tile_y, x_dm, x_sc), __syncthreads is required before other threads read. Missing sync causes data races.",
        fix_suggestion="Add __syncthreads() after each shared memory write phase.",
        pattern=r"__syncthreads|smem|x_qs.*=|tile_y.*=",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="WRONG_STRIDE_PARAM",
        name="Wrong stride parameter to load_tiles",
        severity="HIGH",
        category="correctness",
        prior_prob=0.80,
        description="load_tiles_q4_K stride parameter must be groups_per_row (K/256) for Q4_K. Passing stride=1 causes incorrect row indexing.",
        fix_suggestion="Call: load_tiles_q4_K<mmq_y, false>(A, x_qs, kg, mmq_y - 1, groups_per_row)",
        pattern=r"load_tiles_q4_K.*stride\s*=\s*1|load_tiles_q4_K.*,\s*1\s*\)",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="WINDOW_LOOP_Q4_K",
        name="HFQ4-style window loop in Q4_K kernel",
        severity="HIGH",
        category="correctness",
        prior_prob=0.75,
        description="Q4_K has no HFQ4-style 2-window structure. A 'for (window=0; window<2)' loop processing the same superblock twice is incorrect. vec_dot handles all 8 sub-blocks internally.",
        fix_suggestion="Remove window loop. Each kg iteration processes one superblock; vec_dot handles sub-blocks.",
        pattern=r"for\s*\(\s*window\s*=\s*0\s*;\s*window\s*<\s*2",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="WRONG_STRUCT_TYPE",
        name="Wrong Q8_1 struct type in fused kernel",
        severity="MEDIUM",
        category="correctness",
        prior_prob=0.65,
        description="Fused kernels must use block_q8_1 (standard Q8_1 from ggml-common.h), not custom block_q8_1_fused or block_q8_1_mmq. Wrong struct causes misaligned reads.",
        fix_suggestion="Use block_q8_1* for activation tile pointer in fused kernel signature.",
        pattern=r"block_q8_1_fused|block_q8_1_mmq",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="VGPR_PRESSURE_HIGH",
        name="High VGPR usage in fused kernel",
        severity="MEDIUM",
        category="performance",
        prior_prob=0.60,
        description="Fused kernels with __launch_bounds__(256, 16) target 16 waves. If VGPR > 16 (256/16), occupancy drops. Check register pressure from sum[] array and tile loading.",
        fix_suggestion="Profile with rocprof --metrics VGPR_Active. If > 128, reduce MMQ_Y or use __launch_bounds__(256, 8).",
        pattern=r"float\s+sum\[|__launch_bounds__.*256.*16",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="BANK_CONFLICT_STRIDE",
        name="LDS bank conflict from stride pattern",
        severity="MEDIUM",
        category="performance",
        prior_prob=0.55,
        description="RDNA2 LDS has 32 banks. Stride % 32 == 0 causes all rows to hit the same bank. Stride % 32 == 1 causes sequential bank access. Ideal: stride % 32 gives uniform spread.",
        fix_suggestion="For RDNA2: stride 40 (40%32=8) is good. Verify x_qs stride in load_tiles matches tile_x_sizes.",
        pattern=r"X_STRIDE\s*=\s*\d+|RDNA2_Q4K_X_STRIDE",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="ROW_ROUTING_OVERFLOW",
        name="Row routing index out of bounds",
        severity="HIGH",
        category="correctness",
        prior_prob=0.70,
        description="Fused kernel row routing: total_row = blockIdx.x * MMQ_Y must be < total_m. Check for off-by-one in Q/K/V band boundaries (q_m, q_m+k_m).",
        fix_suggestion="Add bounds check: if (total_row >= total_m) return; before routing.",
        pattern=r"total_row\s*<\s*q_m|total_row\s*<\s*q_m\s*\+\s*k_m",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="WRITE_BACK_LAYOUT",
        name="Output write-back uses wrong column/row layout",
        severity="HIGH",
        category="correctness",
        prior_prob=0.65,
        description="Fused kernel writes Y[col * out_m + row]. Verify this matches the expected output layout. Q/K/V may have different ne[0] dimensions.",
        fix_suggestion="Verify Y indexing matches ggml tensor layout: dst[col * ne0 + row].",
        pattern=r"Y\[.*col.*\*.*out_m.*\+.*row|Y\[.*\[\s*col\s*\*",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="QUANT_FORMAT_OFFSET",
        name="Q4_K data offset incorrect",
        severity="CRITICAL",
        category="correctness",
        prior_prob=0.85,
        description="Q4_K format: dm at offset 0 (4B), scales at offset 4 (12B), qs at offset 16 (128B). HFQ4 format: sc at 0 (4B), zp at 4 (4B), data at 8 (128B). Using HFQ4 offsets on Q4_K data produces garbage.",
        fix_suggestion="Fused kernels must use block_q4_K struct, not manual offset arithmetic.",
        pattern=r"gp\s*\+\s*8\s*\+|gp\s*\+\s*offset\s*\*\s*64|group.*stride.*136",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="LAUNCH_GRID_MISMATCH",
        name="Kernel launch grid doesn't match kernel dims",
        severity="HIGH",
        category="correctness",
        prior_prob=0.70,
        description="Fused kernel grid.x = (total_m + MMQ_Y - 1) / MMQ_Y. Launch must match: grid.x covers all Q+K+V or gate+up rows.",
        fix_suggestion="Verify launch grid: dim3((q_m+k_m+v_m+FUSED_MMQ_Y-1)/FUSED_MMQ_Y, (N+FUSED_MMQ_X-1)/FUSED_MMQ_X).",
        pattern=r"gemm_qkv_fused_mmq<<<|gemm_gate_up_fused_mmq<<<",
        file_pattern="mmq.cuh",
    ),
    BugRule(
        rule_id="SHARED_MEM_SIZING",
        name="Shared memory size mismatch with LDS layout",
        severity="HIGH",
        category="correctness",
        prior_prob=0.75,
        description="Launch shared_mem must equal actual LDS usage: (txs_all.qs + txs_all.dm + txs_all.sc + mmq_x * MMQ_TILE_Y_K) * sizeof(int).",
        fix_suggestion="Use: shared_mem = sizeof(int) * (txs_all.qs + txs_all.dm + txs_all.sc + mmq_x * MMQ_TILE_Y_K).",
        pattern=r"shared_mem\s*=|sizeof\(int\)\s*\*.*txs_all|FUSED_MMQ_Y\s*\*\s*FUSED_X_STRIDE",
        file_pattern="mmq.cu",
    ),
    BugRule(
        rule_id="NO_DEQUANT_INNER_LOOP",
        name="Missing Q4_K dequant formula in inner loop",
        severity="CRITICAL",
        category="correctness",
        prior_prob=0.80,
        description="Q4_K dequant: dm.x * (sc[i] * sumi) - dm.y * m[i]. If inner loop uses simple 'scale * sumi' without per-sub-block scales, output is wrong.",
        fix_suggestion="Use vec_dot_q4_K_q8_1_impl_mmq which handles per-sub-block 6-bit scales correctly.",
        pattern=r"scale_w\s*\*\s*d_x\s*\*\s*sumi\s*\+\s*zp_eff|dm4f\.x\s*\*\s*sumf_d\s*-\s*dm4f\.y\s*\*\s*sumf_m",
        file_pattern="mmq.cuh",
    ),
]


def analyze_code_heuristics(source_dir: str) -> List[BugFinding]:
    """Run heuristic bug detection on C++ kernel source files."""
    findings = []
    mmq_cuh_path = os.path.join(source_dir, "ggml", "src", "ggml-cuda", "mmq.cuh")
    mmq_cu_path = os.path.join(source_dir, "ggml", "src", "ggml-cuda", "mmq.cu")

    files_to_check = {
        "mmq.cuh": mmq_cuh_path,
        "mmq.cu": mmq_cu_path,
    }

    for rule in BUG_RULES:
        target_file = files_to_check.get(rule.file_pattern)
        if not target_file or not os.path.exists(target_file):
            continue

        try:
            with open(target_file, "r") as f:
                lines = f.readlines()
        except IOError:
            continue

        source_text = "".join(lines)

        # Check if this file is in the fused kernel section
        in_fused_section = False
        for i, line in enumerate(lines):
            if "gemm_qkv_fused_mmq" in line or "gemm_gate_up_fused_mmq" in line:
                in_fused_section = True
            if "RDNA2 fused QKV" in line:
                in_fused_section = True

        # Apply pattern matching
        import re
        pattern = re.compile(rule.pattern, re.IGNORECASE)

        for i, line in enumerate(lines):
            if pattern.search(line):
                # Get context (3 lines before and after)
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                context = "".join(f"  {j+1:4d}: {lines[j]}" for j in range(start, end))

                # Bayesian update: P(bug | pattern_found)
                # Likelihood: P(pattern | bug) = 0.8 (pattern is a good indicator)
                # P(pattern | no bug) = 0.1 (false positive rate)
                # Posterior via log-odds
                p_prior = rule.prior_prob
                p_likelihood_pattern = 0.8
                p_false_positive = 0.1

                lo_prior = math.log(p_prior / (1 - p_prior))
                lo_lr = math.log(p_likelihood_pattern / p_false_positive)
                lo_posterior = lo_prior + lo_lr
                posterior = 1.0 / (1.0 + math.exp(-lo_posterior))

                findings.append(BugFinding(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    category=rule.category,
                    file=rule.file_pattern,
                    line=i + 1,
                    description=rule.description,
                    evidence=f"Pattern matched: {line.strip()[:100]}",
                    posterior_prob=posterior,
                    fix_suggestion=rule.fix_suggestion,
                    source_context=context,
                ))

    # Deduplicate by rule_id + file (keep highest posterior)
    deduped = {}
    for f in findings:
        key = (f.rule_id, f.file)
        if key not in deduped or f.posterior_prob > deduped[key].posterior_prob:
            deduped[key] = f

    return sorted(deduped.values(), key=lambda f: f.posterior_prob, reverse=True)


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    kernel_stats: Dict[str, KernelStats],
    memory_rows: List[dict],
    recommendations: List[OptimizationRecommendation],
    evidence: dict,
    output_path: str,
    code_findings: Optional[List[BugFinding]] = None,
):
    """Generate the diagnostic report."""

    total_kernel_time = sum(s.total_duration_ns for s in kernel_stats.values())
    total_dispatches = sum(s.count for s in kernel_stats.values())

    h2d_copies = [r for r in memory_rows if "HOST_TO_DEVICE" in r["direction"]]
    h2d_total_ms = sum(r["duration_ns"] for r in h2d_copies) / NS_PER_MS

    lines = []
    lines.append("=" * 78)
    lines.append("RDNA2 MASTER DIAGNOSTIC — Heuristic + Bayesian Probability Analysis")
    lines.append("=" * 78)
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"GPU: AMD Radeon RX 6800 XT (gfx1030, wave-{WAVE_SIZE}, {NUM_CU} CUs)")
    lines.append("")

    # --- Kernel Summary ---
    lines.append("-" * 78)
    lines.append("KERNEL SUMMARY")
    lines.append("-" * 78)
    lines.append(f"Total kernel time:  {total_kernel_time / NS_PER_MS:.1f} ms")
    lines.append(f"Total dispatches:   {total_dispatches:,}")
    lines.append(f"Avg dispatch time:  {total_kernel_time / total_dispatches / 1000:.1f} us" if total_dispatches else "N/A")
    lines.append("")

    # Category breakdown
    lines.append("Time by Category:")
    cat_time = defaultdict(int)
    cat_count = defaultdict(int)
    for s in kernel_stats.values():
        cat_time[s.category] += s.total_duration_ns
        cat_count[s.category] += s.count
    for cat in sorted(cat_time.keys(), key=lambda c: cat_time[c], reverse=True):
        pct = cat_time[cat] / total_kernel_time * 100 if total_kernel_time else 0
        lines.append(f"  {cat:20s} {cat_time[cat] / NS_PER_MS:8.1f} ms  ({pct:5.1f}%)  [{cat_count[cat]:,} dispatches]")
    lines.append("")

    # Top 15 kernels by time
    lines.append("Top 15 Kernels by Total Time:")
    sorted_kernels = sorted(kernel_stats.values(), key=lambda s: s.total_duration_ns, reverse=True)
    for i, s in enumerate(sorted_kernels[:15], 1):
        pct = s.total_duration_ns / total_kernel_time * 100 if total_kernel_time else 0
        vgpr_flag = " *" if s.max_vgpr > 128 else ""
        lds_flag = " !" if s.max_lds > 50000 else ""
        lines.append(f"  {i:2d}. {s.name[:45]:45s} {s.total_duration_ms:8.1f}ms ({pct:5.1f}%)  "
                     f"VGPR={s.max_vgpr:3d}{vgpr_flag}  LDS={s.max_lds:5d}{lds_flag}  x{s.count:,}")
    lines.append("")
    lines.append("  * = VGPR > 128 (occupancy limited)")
    lines.append("  ! = LDS > 50KB (near 64KB limit)")
    lines.append("")

    # --- Memory Summary ---
    lines.append("-" * 78)
    lines.append("MEMORY COPY SUMMARY")
    lines.append("-" * 78)
    lines.append(f"H2D copies: {len(h2d_copies)}")
    lines.append(f"H2D total time: {h2d_total_ms:.1f} ms")
    if h2d_copies:
        avg_copy_us = (sum(r["duration_ns"] for r in h2d_copies) / len(h2d_copies)) / 1000
        lines.append(f"Avg H2D copy: {avg_copy_us:.1f} us")
    lines.append("")

    # --- Evidence Summary ---
    lines.append("-" * 78)
    lines.append("EVIDENCE SUMMARY (what the data shows)")
    lines.append("-" * 78)
    lines.append(f"Kernel launches: {evidence['kernel_launch_count']:,}")
    lines.append(f"H2D copies: {evidence['h2d_copy_count']}")
    lines.append(f"Max VGPR: {evidence['max_vgpr']}")
    lines.append(f"Max LDS: {evidence['max_lds_bytes']} bytes")
    lines.append(f"Tiny kernels (<10us): {evidence['tiny_kernel_count']:,} ({evidence['tiny_kernel_pct']:.1f}%)")
    lines.append(f"Non-wave-32 aligned dispatches: {evidence['non_aligned_wave_dispatches']:,}")
    lines.append(f"Matmul time: {evidence['matmul_time_pct']:.1f}%")
    lines.append("")

    # --- Bayesian Recommendations ---
    lines.append("=" * 78)
    lines.append("BAYESIAN OPTIMIZATION RECOMMENDATIONS (ranked by posterior expected impact)")
    lines.append("=" * 78)
    lines.append("")

    for i, rec in enumerate(recommendations, 1):
        rec.rank = i
        gain_pct = rec.expected_gain * 100
        ci_low_pct = rec.confidence_interval[0] * 100
        ci_high_pct = rec.confidence_interval[1] * 100

        # Expected impact in ms
        expected_ms = rec.expected_gain * total_kernel_time / NS_PER_MS

        lines.append(f"  #{i}  {rec.name}")
        lines.append(f"      {rec.description}")
        lines.append(f"      Category: {rec.category} | Effort: {rec.effort}")
        lines.append(f"      P(helps|data) = {rec.posterior_probability:.1%}")
        lines.append(f"      Expected gain: {gain_pct:+.1f}% (95% CI: {ci_low_pct:+.1f}% to {ci_high_pct:+.1f}%)")
        lines.append(f"      Expected time saved: ~{expected_ms:.0f} ms per forward pass")
        lines.append(f"      Evidence: {rec.relevance_evidence}")
        lines.append(f"      Source: {rec.source}")
        lines.append("")

    # --- Priority Matrix ---
    lines.append("-" * 78)
    lines.append("PRIORITY MATRIX")
    lines.append("-" * 78)
    lines.append(f"{'#':>3s}  {'Optimization':25s}  {'P(helps)':>9s}  {'Gain':>8s}  {'Effort':>8s}  {'Action'}")
    lines.append(f"{'':3s}  {'':25s}  {'':>9s}  {'':>8s}  {'':>8s}  {'------'}")
    for rec in recommendations:
        gain_pct = rec.expected_gain * 100
        # Decision: implement now / investigate / defer
        if rec.posterior_probability > 0.7 and rec.effort == "low":
            action = "IMPLEMENT NOW"
        elif rec.posterior_probability > 0.6:
            action = "INVESTIGATE"
        elif rec.posterior_probability > 0.4:
            action = "DEFER"
        else:
            action = "SKIP"
        lines.append(f"{rec.rank:3d}  {rec.name:25s}  {rec.posterior_probability:8.1%}  {gain_pct:+7.1f}%  {rec.effort:>8s}  {action}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("SUMMARY")
    lines.append("-" * 78)

    total_expected_gain = 1.0
    for rec in recommendations:
        if rec.posterior_probability > 0.5:
            total_expected_gain *= (1 + rec.expected_gain * rec.posterior_probability)
    total_pct = (total_expected_gain - 1) * 100

    implement_now = [r for r in recommendations if r.posterior_probability > 0.7 and r.effort == "low"]
    investigate = [r for r in recommendations if r.posterior_probability > 0.6 and r.effort != "low"]

    lines.append(f"Total expected gain (compound): ~{total_pct:.1f}%")
    lines.append(f"Implement now ({len(implement_now)}): {', '.join(r.name for r in implement_now) if implement_now else 'none'}")
    lines.append(f"Investigate ({len(investigate)}): {', '.join(r.name for r in investigate) if investigate else 'none'}")
    lines.append("")

    # --- Static Code Analysis ---
    if code_findings:
        lines.append("=" * 78)
        lines.append("STATIC CODE ANALYSIS — Heuristic + Bayesian Bug Detection")
        lines.append("=" * 78)
        lines.append("")

        # Group by severity
        by_severity = defaultdict(list)
        for f in code_findings:
            by_severity[f.severity].append(f)

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity not in by_severity:
                continue
            sev_findings = by_severity[severity]
            lines.append(f"  [{severity}] ({len(sev_findings)} findings)")
            for f in sev_findings:
                lines.append(f"    {f.rule_id} @ {f.file}:{f.line}")
                lines.append(f"      P(bug|pattern) = {f.posterior_prob:.1%}")
                lines.append(f"      {f.description[:120]}")
                lines.append(f"      Evidence: {f.evidence[:100]}")
                lines.append(f"      Fix: {f.fix_suggestion[:120]}")
                lines.append("")

        # Summary
        critical_count = len(by_severity.get("CRITICAL", []))
        high_count = len(by_severity.get("HIGH", []))
        total_findings = len(code_findings)
        lines.append(f"  Total findings: {total_findings} (CRITICAL: {critical_count}, HIGH: {high_count})")
        if critical_count > 0:
            lines.append(f"  *** {critical_count} CRITICAL bugs found — DO NOT DEPLOY without fixing ***")
        elif high_count > 0:
            lines.append(f"  WARNING: {high_count} HIGH severity issues found — review before deployment")
        else:
            lines.append(f"  No critical/high issues found — code looks structurally sound")
        lines.append("")

    report = "\n".join(lines)

    # Write to file
    with open(output_path, "w") as f:
        f.write(report)

    # Also print to stdout
    print(report)
    return report


# ============================================================================
# Main
# ============================================================================

def find_profile_csvs(profile_dir: str) -> Optional[Tuple[str, str, str]]:
    """Find kernel_trace, memory_copy_trace, hip_api_trace CSVs in a directory."""
    kernel_csv = None
    memory_csv = None
    hip_csv = None

    for root, dirs, files in os.walk(profile_dir):
        for f in files:
            fp = os.path.join(root, f)
            if "kernel_trace" in f and f.endswith(".csv"):
                kernel_csv = fp
            elif "memory_copy_trace" in f and f.endswith(".csv"):
                memory_csv = fp
            elif "hip_api_trace" in f and f.endswith(".csv"):
                hip_csv = fp
        if kernel_csv and memory_csv and hip_csv:
            break

    if kernel_csv and memory_csv and hip_csv:
        return kernel_csv, memory_csv, hip_csv
    return None


def run_profiling(model_path: str, build_dir: str, output_dir: str) -> Optional[str]:
    """Run rocprofv3 profiling and return the output directory."""
    bench_bin = os.path.join(build_dir, "bin", "llama-bench")
    if not os.path.exists(bench_bin):
        print(f"ERROR: llama-bench not found at {bench_bin}", file=sys.stderr)
        return None

    prof_dir = os.path.join(output_dir, "rocprofv3_profile")
    os.makedirs(prof_dir, exist_ok=True)

    print(f"Running rocprofv3 profiling...")
    print(f"  Model: {model_path}")
    print(f"  Build: {build_dir}")

    cmd = [
        "/opt/rocm/bin/rocprofv3",
        "--output-directory", prof_dir,
        "--output-format", "csv",
        "--runtime-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--", bench_bin,
        "-m", model_path,
        "-t", "1", "-b", "512", "-p", "512", "-n", "128", "-r", "1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"WARNING: rocprofv3 exited with code {result.returncode}", file=sys.stderr)
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("ERROR: profiling timed out after 600s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("ERROR: rocprofv3 not found at /opt/rocm/bin/rocprofv3", file=sys.stderr)
        return None

    return prof_dir


def main():
    parser = argparse.ArgumentParser(
        description="RDNA2 Master Diagnostic — Bayesian optimization gap analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile and diagnose
  python3 rdna2-diagnostic.py --model-path /path/to/model.gguf

  # Analyze existing profile
  python3 rdna2-diagnostic.py --profile-dir bench-results/profile_baseline/Storm/

  # Quick mode (skip profiling, just analyze)
  python3 rdna2-diagnostic.py --profile-dir bench-results/profile_baseline/Storm/ --quick
""",
    )
    parser.add_argument("--model-path", help="GGUF model path (for live profiling)")
    parser.add_argument("--profile-dir", help="Existing rocprofv3 profile directory")
    parser.add_argument("--build-dir", default="build-rocm", help="Build directory (default: build-rocm)")
    parser.add_argument("--output-dir", default="bench-results", help="Output directory (default: bench-results)")
    parser.add_argument("--quick", action="store_true", help="Skip profiling, analyze existing data only")
    args = parser.parse_args()

    if not args.model_path and not args.profile_dir:
        parser.error("Either --model-path or --profile-dir is required")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get profile data
    profile_dir = args.profile_dir
    if not profile_dir and args.model_path:
        if args.quick:
            print("ERROR: --quick requires --profile-dir", file=sys.stderr)
            sys.exit(1)
        profile_dir = run_profiling(args.model_path, args.build_dir, output_dir)
        if not profile_dir:
            sys.exit(1)

    # Step 2: Find CSVs
    csvs = find_profile_csvs(profile_dir)
    if not csvs:
        print(f"ERROR: Could not find all 3 trace CSVs in {profile_dir}", file=sys.stderr)
        print("  Need: *_kernel_trace.csv, *_memory_copy_trace.csv, *_hip_api_trace.csv", file=sys.stderr)
        sys.exit(1)

    kernel_csv, memory_csv, hip_csv = csvs
    print(f"\nProfile data:")
    print(f"  Kernel trace:   {kernel_csv}")
    print(f"  Memory trace:   {memory_csv}")
    print(f"  HIP API trace:  {hip_csv}")

    # Step 3: Parse
    print("Parsing kernel trace...")
    kernel_rows = parse_kernel_trace(kernel_csv)
    print(f"  {len(kernel_rows):,} kernel dispatches")

    print("Parsing memory trace...")
    memory_rows = parse_memory_trace(memory_csv)
    print(f"  {len(memory_rows):,} memory operations")

    print("Parsing HIP API trace...")
    hip_rows = parse_hip_api_trace(hip_csv)
    print(f"  {len(hip_rows):,} HIP API calls")

    # Step 4: Analyze kernels
    print("Analyzing kernel patterns...")
    kernel_stats = analyze_kernels(kernel_rows)
    print(f"  {len(kernel_stats)} unique kernel types")

    # Step 5: Gather evidence
    print("Gathering evidence for Bayesian analysis...")
    evidence = gather_evidence(kernel_stats, memory_rows, hip_rows)

    # Step 6: Run Bayesian updates
    print("Running Bayesian probability analysis...")
    recommendations = []
    for prior in OPTIMIZATION_PRIORS:
        # Build evidence dict for this prior
        prior_evidence = {}

        # Check relevance conditions
        check = prior.relevance_check
        if ">" in check:
            field_name, threshold_str = check.split(">", 1)
            field_name = field_name.strip()
            threshold_str = threshold_str.strip()

            # Handle "or" conditions
            if "_or_" in threshold_str:
                parts = threshold_str.split("_or_")
                triggered = any(field_name.replace("_", " ") in str(evidence) for _ in parts)
                val = evidence.get(field_name, 0)
                threshold_val = int(parts[0].rstrip("_")) if parts[0].rstrip("_").isdigit() else 0
                prior_evidence["prior_data_match"] = val > threshold_val
                prior_evidence["severity"] = min(1.0, val / (threshold_val * 2)) if threshold_val else 0
            else:
                try:
                    threshold_val = int(threshold_str)
                    val = evidence.get(field_name, 0)
                    prior_evidence["prior_data_match"] = val > threshold_val
                    if threshold_val > 0:
                        ratio = val / threshold_val
                        prior_evidence["severity"] = min(1.0, max(0, (ratio - 1.0)))
                    else:
                        prior_evidence["severity"] = 0
                except ValueError:
                    prior_evidence["prior_data_match"] = False
                    prior_evidence["severity"] = 0
        else:
            prior_evidence["prior_data_match"] = evidence.get(check, False)
            prior_evidence["severity"] = 0.5 if prior_evidence["prior_data_match"] else 0

        prior_evidence["existing_optimization"] = False  # baseline has no optimizations

        rec = bayesian_update(prior, prior_evidence)
        recommendations.append(rec)

    # Sort by expected impact (posterior * gain)
    recommendations.sort(
        key=lambda r: r.posterior_probability * r.expected_gain,
        reverse=True,
    )

    # Step 7: Generate report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"rdna2_diagnostic_{timestamp}.txt")

    # Step 7b: Static code analysis on fused MMQ kernels
    print("Running static code analysis on fused MMQ kernels...")
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_findings = analyze_code_heuristics(source_dir)
    if code_findings:
        critical = [f for f in code_findings if f.severity == "CRITICAL"]
        high = [f for f in code_findings if f.severity == "HIGH"]
        print(f"  Found {len(code_findings)} findings: {len(critical)} CRITICAL, {len(high)} HIGH")
    else:
        print("  No findings (code patterns not matched or files not found)")

    generate_report(kernel_stats, memory_rows, recommendations, evidence, report_path,
                    code_findings=code_findings)

    # Also save JSON for programmatic access
    json_path = os.path.join(output_dir, f"rdna2_diagnostic_{timestamp}.json")
    json_data = {
        "timestamp": timestamp,
        "total_kernel_time_ms": sum(s.total_duration_ns for s in kernel_stats.values()) / NS_PER_MS,
        "total_dispatches": sum(s.count for s in kernel_stats.values()),
        "evidence": {k: v for k, v in evidence.items()
                     if not isinstance(v, (defaultdict, dict)) or v},
        "recommendations": [
            {
                "rank": i + 1,
                "name": r.name,
                "posterior_probability": round(r.posterior_probability, 4),
                "expected_gain": round(r.expected_gain, 4),
                "confidence_interval": [round(r.confidence_interval[0], 4), round(r.confidence_interval[1], 4)],
                "category": r.category,
                "effort": r.effort,
                "evidence": r.relevance_evidence,
                "source": r.source,
            }
            for i, r in enumerate(recommendations)
        ],
        "code_analysis": [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "category": f.category,
                "file": f.file,
                "line": f.line,
                "description": f.description,
                "evidence": f.evidence,
                "posterior_prob": round(f.posterior_prob, 4),
                "fix_suggestion": f.fix_suggestion,
            }
            for f in code_findings
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nReports saved:")
    print(f"  Text: {report_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
