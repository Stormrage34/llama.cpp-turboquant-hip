#!/usr/bin/env python3
"""
Asymmetric TurboQuant Fault Diagnostic Tool
============================================
Heuristic + Bayesian analysis of faulty KV cache quantization precision.

Detects which attention heads produce garbled output when K uses q8_0
(8-bit block-wise) and V uses turbo2/3/4 (Lloyd-Max centroid quantization).

Two modes:
  Mode 1 (no C++ hooks): Parse llama-perplexity output + existing log data
  Mode 2 (with C++ hooks): Full per-head attention weight analysis

Usage:
  # Mode 1: Simple perplexity comparison
  python3 diag-turbo-fault.py --baseline baseline.perplexity --asymmetric test.perplexity

  # Mode 2: Full analysis with FA_TRACE_KQ dumps
  python3 diag-turbo-fault.py --kq-traces kq_traces.log --model llama --output report.json

  # Real-time analysis via pipe
  llama-cli -m model.gguf -ctk q8_0 -ctv turbo3_0 -p "prompt" 2>&1 | python3 diag-turbo-fault.py --pipe
"""

import argparse
import io
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    HAS_SCIPY = False
    print("[WARN] scipy not installed. Bayesian CI will use normal approximation.", file=sys.stderr)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HeadStats:
    """Per-head attention statistics for one inference step."""
    layer: int
    head: int
    n_ctx: int
    bit_width_K: int = 8   # q8_0
    bit_width_V: int = 3   # turbo3_0 default
    entropy: float = 0.0
    sparsity: float = 0.0
    max_weight: float = 0.0
    tail_mass: float = 0.0
    n_nonzero: int = 0
    weight_samples: List[float] = field(default_factory=list)


@dataclass
class TokenStats:
    """Per-token statistics from perplexity/log output."""
    token_idx: int
    token_id: int
    log_prob: float
    layer_stats: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Parsers for llama.cpp Output
# ============================================================================

# Regex patterns for llama.cpp log output
RE_KQ_TRACE = re.compile(
    r'FA_KQ_TILE:\s*hd=(?P<head>\d+)\s+col=(?P<col>\d+)\s+'
    r'tile=(?P<tile>\d+)\s+k=(?P<k>\d+)\s+sum=(?P<sum>[-.\d]+)\s+'
    r'max=(?P<max>[-.\d]+)'
)

RE_SOFTMAX_TRACE = re.compile(
    r'FA_SOFTMAX:\s*seq=(?P<seq>\d+)\s+hd=(?P<head>\d+)\s+'
    r'col=(?P<col>\d+)\s+n_tiles=(?P<ntiles>\d+)\s+'
    r'sum=(?P<sum>[-.\d,]+)\s+max=(?P<max>[-.\d,]+)'
)

RE_PERPLEXITY_TOKEN = re.compile(
    r'\[(?P<idx>\d+)\]\s+token\s+(?P<token>\d+):\s+log_prob\s*=\s*(?P<logprob>[-.\d]+)'
)

RE_PERPLEXITY_SUMMARY = re.compile(
    r'perplexity:\s+(?P<ppl>[\d.]+).*std:\s+(?P<std>[\d.]+)'
)


def parse_kq_traces(
    source: 'str | TextIO', n_heads_per_layer: int = 32
) -> Dict[str, HeadStats]:
    """
    Parse FA_TRACE_KQ output from compile-time debug flag.

    Accepts a file path (str) or an already-open text stream (for pipe mode).

    NOTE: Raw KQ traces are pre-softmax logits, not attention weights.
    The softmax normalization below (exp, sum-divide) approximates attention
    distributions but is NOT exact — it treats the dumped KQ logit sequence
    as if it were a complete attention row. This means:
      - "Entropy" measures logit concentration, not true attention entropy
      - Cross-seqlen comparisons are invalid (more tokens -> more logits ->
        higher spread regardless of fault)
      - Only use relative comparisons within same-seqlen runs (baseline vs test)
    """
    from typing import TextIO

    heads: Dict[str, HeadStats] = {}

    def _parse_lines(fh):
        for line in fh:
            m = RE_KQ_TRACE.search(line)
            if not m:
                continue
            d = m.groupdict()
            head = int(d['head'])
            layer = head // n_heads_per_layer
            key = f"L{layer}H{head}"

            if key not in heads:
                heads[key] = HeadStats(layer=layer, head=head, n_ctx=0)

            kq_sum = float(d['sum'])
            heads[key].weight_samples.append(kq_sum)

    if isinstance(source, TextIO):
        _parse_lines(source)
    else:
        with open(source) as f:
            _parse_lines(f)

    # Post-process: approximate softmax from raw KQ logits.
    # This is valid only for relative (same-seqlen) comparison.
    for key, h in heads.items():
        ws = np.array(h.weight_samples)
        if len(ws) > 1:
            ws = ws - ws.max()
            ws = np.exp(ws)
            ws = ws / (ws.sum() + 1e-12)
            h.entropy = float(-np.sum(ws * np.log(ws + 1e-12)))
            h.sparsity = float(np.sum(ws < 1e-4) / len(ws))
            h.max_weight = float(ws.max())
            h.tail_mass = float(np.sum(np.sort(ws)[:len(ws)//2]))
            h.n_nonzero = int(np.sum(ws > 1e-4))

    return heads


def parse_softmax_traces(
    source: 'str | TextIO', n_heads_per_layer: int = 32
) -> Dict[str, HeadStats]:
    """
    Parse FA_SOFTMAX trace lines from compile-time flag -DFA_TRACE_SOFTMAX.

    Dumps per-column softmax sum and max after all tiles complete for each head.
    KQ_sum ~= 1.0 indicates healthy online softmax normalization; drift from 1.0
    indicates numerical issues from faulty quantization.
    """
    heads: Dict[str, HeadStats] = {}

    def _parse_lines(fh):
        for line in fh:
            m = RE_SOFTMAX_TRACE.search(line)
            if not m:
                continue
            d = m.groupdict()
            head = int(d['head'])
            layer = head // n_heads_per_layer
            key = f"L{layer}H{head}"

            if key not in heads:
                heads[key] = HeadStats(layer=layer, head=head, n_ctx=0)

            # Accumulate trace lines per head (1 per query col)
            s = float(d['sum'].replace(',', '.'))
            heads[key].weight_samples.append(s)
            heads[key].max_weight = max(heads[key].max_weight, float(d['max'].replace(',', '.')))
            heads[key].n_nonzero = int(d['ntiles'])

    if isinstance(source, TextIO):
        _parse_lines(source)
    else:
        with open(source) as f:
            _parse_lines(f)

    # Compute per-head aggregate stats from per-col traces
    for key, h in heads.items():
        ws = np.array(h.weight_samples)
        if len(ws) > 1:
            h.entropy = float(-np.mean(np.log(ws + 1e-12)))  # higher = more dispersed
            h.sparsity = float(np.sum(ws < 0.9) / len(ws))   # fraction of cols with sum drift
            h.tail_mass = float(np.sum(np.sort(ws)[:len(ws)//4]))  # bottom-quartile sum
        elif len(ws) == 1:
            h.entropy = 0.0
            h.sparsity = 1.0 if ws[0] < 0.9 else 0.0
            h.tail_mass = ws[0]

    return heads


def parse_perplexity_file(filepath: str) -> List[TokenStats]:
    """Parse llama-perplexity output to extract per-token log probabilities."""
    tokens = []
    with open(filepath) as f:
        for line in f:
            m = RE_PERPLEXITY_TOKEN.search(line)
            if m:
                tokens.append(TokenStats(
                    token_idx=int(m.group('idx')),
                    token_id=int(m.group('token')),
                    log_prob=float(m.group('logprob')),
                ))
    return tokens


# ============================================================================
# Heuristic Analysis
# ============================================================================

class HeuristicAnalyzer:
    """
    Heuristic analysis of quantization fault based on:
    - Attention entropy (higher → less focused → potential fault)
    - Sparsity (lower → more distributed → potential fault)
    - Max weight (lower → peak suppressed → potential fault)
    - Tail mass (higher → more energy in tail → potential fault)
    - Bit-width (lower → higher quantization error → higher prior)
    """
    
    def __init__(self, 
                 entropy_threshold: float = 8.0,
                 sparsity_threshold: float = 0.3,
                 max_weight_threshold: float = 0.3,
                 tail_mass_threshold: float = 0.15):
        self.entropy_th = entropy_threshold
        self.sparsity_th = sparsity_threshold
        self.max_weight_th = max_weight_threshold
        self.tail_mass_th = tail_mass_threshold
    
    def score_head(self, h: HeadStats, baseline: Optional[HeadStats] = None) -> float:
        """
        Heuristic risk score for a single head.
        Higher score → more likely to be faulty.
        Range: [0, 1]
        """
        score = 0.0
        n_factors = 0
        
        # 1. Bit-width factor (lower bits = higher risk)
        if h.bit_width_V == 2:
            score += 0.4
        elif h.bit_width_V == 3:
            score += 0.25
        elif h.bit_width_V == 4:
            score += 0.1
        else:
            score += 0.05
        n_factors += 1
        
        if baseline:
            # 2. Entropy deviation (increase = fault indicator)
            if h.entropy > baseline.entropy * 1.1 and baseline.entropy > 0:
                ent_ratio = h.entropy / baseline.entropy
                score += min(0.25, (ent_ratio - 1.0) * 0.5)
                n_factors += 1
            
            # 3. Sparsity deviation (decrease = fault indicator)
            if h.sparsity < baseline.sparsity * 0.9:
                sp_ratio = h.sparsity / max(baseline.sparsity, 1e-10)
                score += min(0.2, (1.0 - sp_ratio) * 0.5)
                n_factors += 1
            
            # 4. Max weight deviation (decrease = fault indicator)
            if h.max_weight < baseline.max_weight * 0.9:
                peak_ratio = h.max_weight / max(baseline.max_weight, 1e-10)
                score += min(0.15, (1.0 - peak_ratio) * 0.3)
                n_factors += 1
        else:
            # Without baseline, use absolute thresholds
            if h.entropy > self.entropy_th:
                score += 0.2
                n_factors += 1
            if h.sparsity < self.sparsity_th:
                score += 0.15
                n_factors += 1
            if h.max_weight < self.max_weight_th:
                score += 0.1
                n_factors += 1
        
        return score / max(n_factors, 1)


# ============================================================================
# Bayesian Analysis
# ============================================================================

class BayesianFaultDiagnostic:
    """
    Bayesian hierarchical model for per-head fault detection.
    
    Model:
      P(faulty_h | data_h) ∝ P(data_h | faulty) · P(faulty_h)
    
    Priors:
      P(faulty_h) ~ Beta(α_bw, β_bw)  # bit-width dependent
        + Beta(α_ent, β_ent)          # attention entropy dependent
    
    Likelihood:
      P(stat_h | faulty) ~ Normal(μ_faulty, σ²_faulty)
      P(stat_h | clean)  ~ Normal(μ_clean,  σ²_clean)
    
    Posterior:
      Beta-Binomial conjugate update yields closed-form Beta posterior
    """
    
    def __init__(self):
        # Prior hyperparameters
        self.alpha_base = 2.0
        self.beta_base = 6.0
        self.alpha_ent = 3.0
        self.beta_ent = 5.0
        self.alpha_bw_factor = 4.0  # multiplier for bit-width effect
        self.entropy_evidence_factor = 3.0
        self.sparsity_evidence_factor = 3.0
        self.peak_evidence_factor = 2.0
        
        # Likelihood hyperparameters (estimated from calibration)
        self.mu_clean_entropy = 0.0
        self.sigma_clean_entropy = 1.0
        self.mu_faulty_entropy = 0.3
        self.sigma_faulty_entropy = 0.15
        self.calibrated = False
    
    def calibrate(self, baseline_heads: Dict[str, HeadStats]):
        """Estimate likelihood parameters from baseline (clean) run."""
        entropies = [h.entropy for h in baseline_heads.values()]
        if entropies:
            self.mu_clean_entropy = float(np.mean(entropies))
            self.sigma_clean_entropy = float(np.std(entropies)) + 1e-6
            self.calibrated = True
    
    def _prior_from_bitwidth(self, bw: int) -> Tuple[float, float]:
        """Lower bit-width → higher prior probability of being faulty."""
        alpha = self.alpha_base + self.alpha_bw_factor * (8.0 / max(bw, 1))
        return alpha, self.beta_base
    
    def _update_from_deviation(self, prior_alpha: float, prior_beta: float,
                                observed: float, baseline: float,
                                evidence_factor: float,
                                worse_is_higher: bool = True) -> Tuple[float, float]:
        """
        Update Beta posterior with evidence from a deviation statistic.
        Only deviations toward "worse" count as evidence (directional).
        """
        if baseline <= 0:
            return prior_alpha, prior_beta

        ratio = observed / baseline
        if worse_is_higher:
            # Higher is worse: entropy increase, perplexity increase
            evidence = max(0.0, ratio - 1.0 - 0.1) * evidence_factor
        else:
            # Lower is worse: sparsity decrease, max_weight decrease
            evidence = max(0.0, 1.0 - ratio - 0.1) * evidence_factor
        return prior_alpha + evidence, prior_beta
    
    def compute_posterior(self, h: HeadStats,
                          baseline: Optional[HeadStats] = None
                          ) -> Dict:
        """
        Compute P(faulty | data) with 95% credible interval.
        
        Returns:
            dict with layer, head, P_faulty_mean, CI_95, evidence_strength
        """
        # Start with bit-width prior
        p_alpha, p_beta = self._prior_from_bitwidth(h.bit_width_V)
        
        if baseline and self.calibrated:
            # Update with entropy deviation (higher entropy = worse)
            p_alpha, p_beta = self._update_from_deviation(
                p_alpha, p_beta, h.entropy, baseline.entropy,
                self.entropy_evidence_factor, worse_is_higher=True)
            
            # Update with sparsity deviation (lower sparsity = worse)
            p_alpha, p_beta = self._update_from_deviation(
                p_alpha, p_beta, h.sparsity, baseline.sparsity,
                self.sparsity_evidence_factor, worse_is_higher=False)
            
            # Update with max weight deviation (lower max weight = worse)
            p_alpha, p_beta = self._update_from_deviation(
                p_alpha, p_beta, h.max_weight, baseline.max_weight,
                self.peak_evidence_factor, worse_is_higher=False)
        
        # Posterior statistics
        mean_p = p_alpha / (p_alpha + p_beta)
        
        if HAS_SCIPY:
            assert sp_stats is not None
            ci_low = float(sp_stats.beta.ppf(0.025, p_alpha, p_beta))
            ci_high = float(sp_stats.beta.ppf(0.975, p_alpha, p_beta))
        else:
            # Normal approximation for Beta
            std_p = math.sqrt(mean_p * (1 - mean_p) / (p_alpha + p_beta + 1))
            ci_low = max(0.0, mean_p - 1.96 * std_p)
            ci_high = min(1.0, mean_p + 1.96 * std_p)
        
        return {
            "layer": h.layer,
            "head": h.head,
            "P_faulty_mean": round(mean_p, 4),
            "P_faulty_CI_95": [round(ci_low, 4), round(ci_high, 4)],
            "evidence_strength": round(p_alpha + p_beta - self.alpha_base - self.beta_base, 2),
            "bit_width_V": h.bit_width_V,
        }


# ============================================================================
# Perplexity-Based Analysis (No C++ hooks needed)
# ============================================================================

class PerplexityChangeDetector:
    """
    Bayesian change-point detection on per-token log probabilities.
    
    Identifies where in the sequence the asymmetric quantization begins
    to cause statistically significant degradation.
    """
    
    def __init__(self, window_size: int = 64):
        self.window_size = window_size
    
    def detect_degradation(self, baseline_tokens: List[TokenStats],
                           test_tokens: List[TokenStats]) -> Dict:
        """
        Compare per-token log probabilities and identify degraded regions.
        
        Returns dict with:
        - degradation_regions: [(start, end, severity), ...]
        - mean_delta: average log-prob difference
        - significant: bool if degradation is statistically significant
        """
        if not baseline_tokens or not test_tokens:
            return {"error": "No token data"}
        
        n = min(len(baseline_tokens), len(test_tokens))
        baseline_lps = np.array([t.log_prob for t in baseline_tokens[:n]])
        test_lps = np.array([t.log_prob for t in test_tokens[:n]])
        
        deltas = test_lps - baseline_lps  # negative = worse
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        
        # Sliding window change detection
        regions = []
        for i in range(0, n - self.window_size, self.window_size // 2):
            window = deltas[i:i + self.window_size]
            win_mean = float(np.mean(window))
            win_std = float(np.std(window))
            
            if win_mean < -0.05 and abs(win_mean) > 2 * win_std / math.sqrt(len(window)):
                # Significant degradation in this window
                severity = min(1.0, abs(win_mean) * 5.0)
                regions.append({
                    "start_token": i,
                    "end_token": min(i + self.window_size, n),
                    "mean_logprob_delta": round(win_mean, 4),
                    "severity": round(severity, 3),
                })
        
        # t-test for significance
        if HAS_SCIPY:
            assert sp_stats is not None
            _, p_value = sp_stats.ttest_rel(baseline_lps, test_lps)
            significant = p_value < 0.05
        else:
            z_score = abs(mean_delta) / (std_delta / math.sqrt(n) + 1e-10)
            significant = z_score > 1.96
        
        return {
            "mean_logprob_delta": round(mean_delta, 4),
            "std_logprob_delta": round(std_delta, 4),
            "degraded_regions": regions,
            "significant": bool(significant),
            "n_tokens_compared": n,
        }


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    heads: Dict[str, HeadStats],
    baseline_heads: Optional[Dict[str, HeadStats]] = None,
    perplexity_analysis: Optional[Dict] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Generate comprehensive fault report.
    """
    analyzer = HeuristicAnalyzer()
    bayes = BayesianFaultDiagnostic()
    
    if baseline_heads:
        bayes.calibrate(baseline_heads)
    
    # Analyze each head
    results = []
    for key, h in heads.items():
        bl = baseline_heads.get(key) if baseline_heads else None
        
        risk = analyzer.score_head(h, bl)
        posterior = bayes.compute_posterior(h, bl)
        posterior["risk_score"] = round(risk, 4)
        
        results.append(posterior)
    
    # Sort by P_faulty (descending)
    results.sort(key=lambda x: x["P_faulty_mean"], reverse=True)
    
    flagged = [r for r in results if r["P_faulty_mean"] >= threshold]
    
    # Per-layer summary
    by_layer = defaultdict(list)
    for r in results:
        by_layer[r["layer"]].append(r)
    
    layer_summary = {}
    for layer, items in sorted(by_layer.items()):
        p_vals = [i["P_faulty_mean"] for i in items]
        layer_summary[layer] = {
            "n_heads": len(items),
            "mean_P_faulty": round(float(np.mean(p_vals)), 4),
            "max_P_faulty": round(float(max(p_vals)), 4),
            "n_flagged": sum(1 for i in items if i["P_faulty_mean"] >= threshold),
        }
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "type_K": "q8_0",
            "type_V_analyzed": list(set(h.bit_width_V for h in heads.values())),
            "total_heads": len(heads),
        },
        "summary": {
            "n_flagged": len(flagged),
            "pct_flagged": round(len(flagged) / max(len(results), 1) * 100, 1),
            "mean_P_faulty": round(float(np.mean([r["P_faulty_mean"] for r in results])), 4),
            "max_P_faulty": round(max(r["P_faulty_mean"] for r in results), 4),
        },
        "flagged_heads": flagged[:50],  # top 50
        "per_layer": layer_summary,
        "threshold_used": threshold,
    }
    
    if perplexity_analysis:
        report["perplexity_analysis"] = perplexity_analysis
    
    return report


def print_report(report: Dict):
    """Print a human-readable summary of the fault report."""
    summary = report["summary"]
    
    print("=" * 65)
    print(f"  TurboQuant Fault Diagnostic Report")
    print(f"  Generated: {report['timestamp']}")
    print("=" * 65)
    print()
    print(f"  Total heads analyzed: {report['config']['total_heads']}")
    print(f"  KV cache: K={report['config']['type_K']}, V={report['config']['type_V_analyzed']}")
    print(f"  Flagged heads: {summary['n_flagged']} ({summary['pct_flagged']}%)")
    print(f"  Mean P(faulty): {summary['mean_P_faulty']}")
    print(f"  Max P(faulty):  {summary['max_P_faulty']}")
    print()
    
    if report.get("perplexity_analysis"):
        pa = report["perplexity_analysis"]
        if "error" not in pa:
            print(f"  Perplexity delta: {pa['mean_logprob_delta']} ± {pa['std_logprob_delta']}")
            print(f"  Significant degradation: {pa['significant']}")
            if pa.get('degraded_regions'):
                print(f"  Degraded regions: {len(pa['degraded_regions'])}")
                for r in pa['degraded_regions'][:5]:
                    print(f"    tokens {r['start_token']}-{r['end_token']}: "
                          f"delta={r['mean_logprob_delta']}, severity={r['severity']}")
        print()
    
    print("  Top-10 Most Likely Faulty Heads:")
    print("  " + "-" * 55)
    for r in report.get("flagged_heads", [])[:10]:
        ci = r.get("P_faulty_CI_95", [0, 1])
        print(f"    L{r['layer']:3d} H{r['head']:4d}  "
              f"P(faulty)={r['P_faulty_mean']:.3f}  "
              f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"bw={r['bit_width_V']}bit")
    
    print()
    print("  Per-Layer Summary:")
    print("  " + "-" * 55)
    for layer, ls in sorted(report.get("per_layer", {}).items()):
        bar = "#" * int(ls['mean_P_faulty'] * 20)
        print(f"    Layer {layer:3d}: mean P={ls['mean_P_faulty']:.3f} "
              f"max={ls['max_P_faulty']:.3f} "
              f"flagged={ls['n_flagged']}/{ls['n_heads']}  {bar}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Asymmetric TurboQuant Fault Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze FA_TRACE_KQ output with baseline:
  python3 diag-turbo-fault.py --kq-traces kq_asymmetric.log \\
      --baseline-kq kq_symmetric.log --output report.json

  # Analyze FA_SOFTMAX traces (requires -DFA_TRACE_SOFTMAX):
  python3 diag-turbo-fault.py --softmax-traces softmax_asymmetric.log \\
      --baseline-softmax softmax_symmetric.log --output report.json

  # Perplexity comparison (no C++ hooks needed):
  python3 diag-turbo-fault.py --baseline baseline.ppl \\
      --asymmetric test.ppl --output report.json

  # Pipe mode (real-time):
  llama-cli ... -ctk q8_0 -ctv turbo3_0 2>&1 | python3 diag-turbo-fault.py --pipe
        """
    )
    
    # Input sources
    parser.add_argument("--kq-traces", help="FA_TRACE_KQ log file (asymmetric run)")
    parser.add_argument("--baseline-kq", help="FA_TRACE_KQ log file (symmetric q8_0/q8_0 baseline)")
    parser.add_argument("--softmax-traces", help="FA_SOFTMAX trace file (asymmetric, requires -DFA_TRACE_SOFTMAX)")
    parser.add_argument("--baseline-softmax", help="FA_SOFTMAX trace file (symmetric baseline)")
    parser.add_argument("--baseline", help="Baseline perplexity file (symmetric run)")
    parser.add_argument("--asymmetric", help="Asymmetric perplexity file (q8_0 K / turbo V)")
    
    # Configuration
    parser.add_argument("--model", default="llama", choices=["llama", "qwen", "gemma"],
                        help="Model architecture (affects head count heuristic)")
    parser.add_argument("--n-layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=32, help="Heads per layer")
    parser.add_argument("--bit-width-v", type=int, default=3, choices=[2, 3, 4],
                        help="Turbo V bit-width")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(faulty) threshold for flagging")
    parser.add_argument("--output", default=None, help="Output JSON path")
    
    # Mode
    parser.add_argument("--pipe", action="store_true",
                        help="Read from stdin (pipe from llama-cli)")
    
    args = parser.parse_args()
    
    heads: Dict[str, HeadStats] = {}
    baseline_heads: Dict[str, HeadStats] = {}
    perplexity_analysis = None
    
    # Mode 1: Pipe mode — read FA_TRACE_KQ or FA_SOFTMAX lines from stdin
    if args.pipe:
        print("[DIAG] Reading from pipe...", file=sys.stderr)
        pipe_data = list(sys.stdin)
        # Detect format from first line
        if pipe_data and "FA_SOFTMAX" in pipe_data[0]:
            args.softmax_traces = io.StringIO(''.join(pipe_data))
        else:
            args.kq_traces = io.StringIO(''.join(pipe_data))
    
    # Mode 2: Parse FA_SOFTMAX traces (preferred over KQ traces)
    if args.softmax_traces:
        print(f"[DIAG] Parsing FA_SOFTMAX traces from: {args.softmax_traces}", file=sys.stderr)
        heads = parse_softmax_traces(args.softmax_traces, args.n_heads)
        for h in heads.values():
            h.bit_width_V = args.bit_width_v
        print(f"[DIAG] Found {len(heads)} heads with FA_SOFTMAX data", file=sys.stderr)

    if args.baseline_softmax:
        print(f"[DIAG] Parsing baseline FA_SOFTMAX traces from: {args.baseline_softmax}", file=sys.stderr)
        baseline_heads = parse_softmax_traces(args.baseline_softmax, args.n_heads)
        for h in baseline_heads.values():
            h.bit_width_V = args.bit_width_v
        print(f"[DIAG] Found {len(baseline_heads)} baseline heads", file=sys.stderr)

    # Mode 3: Parse FA_TRACE_KQ output (fallback if no FA_SOFTMAX traces)
    if args.kq_traces:
        print(f"[DIAG] Parsing KQ traces from: {args.kq_traces}", file=sys.stderr)
        heads = parse_kq_traces(args.kq_traces, args.n_heads)
        for h in heads.values():
            h.bit_width_V = args.bit_width_v
        print(f"[DIAG] Found {len(heads)} heads with KQ data", file=sys.stderr)

    if args.baseline_kq:
        print(f"[DIAG] Parsing baseline KQ traces from: {args.baseline_kq}", file=sys.stderr)
        baseline_heads = parse_kq_traces(args.baseline_kq, args.n_heads)
        print(f"[DIAG] Found {len(baseline_heads)} baseline heads", file=sys.stderr)
    
    # Mode 3: Perplexity-based analysis
    if args.baseline and args.asymmetric:
        print(f"[DIAG] Comparing perplexity: {args.baseline} vs {args.asymmetric}", file=sys.stderr)
        baseline_tokens = parse_perplexity_file(args.baseline)
        test_tokens = parse_perplexity_file(args.asymmetric)
        
        detector = PerplexityChangeDetector()
        perplexity_analysis = detector.detect_degradation(baseline_tokens, test_tokens)
        
        print(f"[DIAG] Mean log-prob delta: {perplexity_analysis['mean_logprob_delta']}",
              file=sys.stderr)
    
    # If we have KQ traces, generate full report
    if heads:
        report = generate_report(heads, baseline_heads if baseline_heads else None,
                                  perplexity_analysis, threshold=args.threshold)
    elif perplexity_analysis:
        # Degraded report with only perplexity data
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {"type_K": "q8_0", "type_V_analyzed": [args.bit_width_v]},
            "perplexity_analysis": perplexity_analysis,
            "summary": {
                "n_flagged": 0,
                "pct_flagged": 0,
                "mean_P_faulty": 0,
                "max_P_faulty": 0,
            },
            "flagged_heads": [],
            "per_layer": {},
        }
    else:
        print("ERROR: No input data. Provide --kq-traces, --baseline+--asymmetric, or --pipe",
              file=sys.stderr)
        sys.exit(1)
    
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[DIAG] Report saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
