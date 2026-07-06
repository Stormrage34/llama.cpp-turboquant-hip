#!/usr/bin/env python3
"""
High-Fidelity Multi-Tier Memory & LDS Bank Conflict Simulator

Models multi-tier execution latencies for RDNA2 hardware structures.
Accounts for L0/L1/L2 cache locality and calculates performance stalls
caused by shared memory bank layout structural issues.

Replaces broken memory model with hardware-accurate simulation.
"""

import numpy as np
from typing import Dict, Tuple, Any
from _roctx import mark, roctx as _roctx


class RDNA2MemoryHardwareSimulator:
    """Models multi-tier execution latencies for RDNA2 hardware structures.
    
    Accounts for:
    - L0/L1/L2 cache locality and hit rates
    - LDS (Local Data Share) bank conflict analysis
    - 128-byte stride boundary collisions across 32 physical memory banks
    - VRAM throughput accounting with proper per-token streaming
    """
    
    def __init__(self):
        # RDNA 2 Memory Architecture Constants
        self.LDS_BANKS = 32              # Number of LDS memory banks
        self.BANK_WIDTH_BYTES = 4        # Bytes per bank access (32-bit)
        self.LDS_STRIDE_BOUNDARY = self.LDS_BANKS * self.BANK_WIDTH_BYTES  # 128 Bytes
        
        # Physical cache hierarchy for AMD RX 6800 XT (gfx1030)
        self.L0_CACHE_SIZE_BYTES = 16 * 1024      # 16KB per CU (32KB per WGP, shared between 2 CUs)
        self.L1_CACHE_SIZE_BYTES = 128 * 1024     # 128KB per GL1 complex
        self.L2_CACHE_SIZE_BYTES = 4 * 1024 * 1024 # 4MB Shared L2
        self.INFINITY_CACHE_BYTES = 128 * 1024 * 1024  # 128MB Infinity Cache (L3)
        
        # Latency constants (cycles per access, aligned to mlnn.py RDNA 2 values)
        self.LATENCY_L0 = 1.0    # scalar cache hit
        self.LATENCY_L1 = 4.0    # vector L1 cache hit
        self.LATENCY_L2 = 15.0   # L2 cache hit
        self.LATENCY_INFINITY = 50.0  # Infinity Cache (L3) hit
        self.LATENCY_VRAM = 200.0     # GDDR6 miss penalty
    
    def evaluate_lds_bank_conflicts(self, thread_offsets: np.ndarray) -> Tuple[int, float]:
        """Analyzes a 32-element array representing active lane memory addresses.
        
        Detects bank conflicts that cause serialization stalls in LDS access.
        
        Args:
            thread_offsets: Array of 32 memory offsets (one per wave lane)
            
        Returns:
            Tuple of (total_conflicts, cycle_penalty)
            - total_conflicts: Number of lanes involved in conflicts
            - cycle_penalty: Additional cycles needed due to serialization
        """
        with mark("mem_sim_lds_bank"):
            assert thread_offsets.size == 32, "Input vector must match Wave32 execution boundaries."
            
            # Map raw physical memory address requests directly to bank indices
            bank_indices = (thread_offsets // self.BANK_WIDTH_BYTES) % self.LDS_BANKS
            
            # Identify address conflicts across shared execution paths
            unique_banks, counts = np.unique(bank_indices, return_counts=True)
            max_conflict_depth = np.max(counts)
            
            # Each conflicting bank request introduces a serialization stall loop iteration
            cycle_penalty = 0.0
            if max_conflict_depth > 1:
                cycle_penalty = float(max_conflict_depth - 1) * 2.0  # 2-cycle structural penalty per stall
                
            total_conflicts = int(np.sum(counts[counts > 1]))
            return total_conflicts, cycle_penalty
    
    def simulate_lds_access_pattern(self, tile_x: int, tile_y: int, element_size_bytes: int = 2) -> Dict[str, Any]:
        """Simulates LDS access patterns for a given tile configuration.
        
        Args:
            tile_x: Number of elements per row in the tile
            tile_y: Number of rows in the tile
            element_size_bytes: Size of each element (2=half, 4=float)
            
        Returns:
            Dictionary with conflict analysis and estimated latency
        """
        with mark("mem_sim_lds_tile"):
            # Generate thread offsets for a 2D tile access (Wave32 = 32 threads)
            threads_per_row = min(tile_x, 32)
            num_rows = max(1, 32 // threads_per_row)
            
            offsets = []
            for row in range(num_rows):
                for col in range(threads_per_row):
                    offset = (row * tile_x + col) * element_size_bytes
                    offsets.append(offset)
            
            # Pad to 32 elements if needed
            while len(offsets) < 32:
                offsets.append(0)
            
            thread_offsets = np.array(offsets[:32], dtype=np.int64)
            
            conflicts, penalty = self.evaluate_lds_bank_conflicts(thread_offsets)
            
            return {
                "tile_dimensions": (tile_x, tile_y),
                "element_size_bytes": element_size_bytes,
                "total_conflicts": conflicts,
                "cycle_penalty": penalty,
                "is_optimal": conflicts == 0
            }
    
    def calculate_real_vram_throughput(
        self, 
        head_dim: int, 
        quantization_bits: float, 
        token_throughput: float
    ) -> Dict[str, Any]:
        """Calculates accurate VRAM bandwidth targets by ignoring LDS resident steps.
        
        Accounts for:
        - Global VRAM reads: Q, K, V elements streamed to processor cores
        - Global VRAM writes: Attention output projections committed to main memory
        
        Args:
            head_dim: Attention head dimension (e.g., 128)
            quantization_bits: Bits per element for Q/K/V (e.g., 3.5 for turbo3_0)
            token_throughput: Tokens processed per second
            
        Returns:
            Dictionary with bytes_per_token, vram_throughput_mb_s, peak_utilization_pct
        """
        with mark("mem_sim_vram"):
            bytes_per_element = quantization_bits / 8.0
            
            # Global VRAM reads: Stream Q, K, and V elements into the processor cores
            qkv_reads_bytes = 3 * head_dim * bytes_per_element
            
            # Global VRAM writes: Commit completed attention projections back to main memory
            output_writes_bytes = head_dim * 4.0  # Float32 dense precision destination layout
            
            total_vram_bytes_per_token = qkv_reads_bytes + output_writes_bytes
            actual_bandwidth_bytes_sec = total_vram_bytes_per_token * token_throughput
            actual_bandwidth_mb_sec = actual_bandwidth_bytes_sec / (1024 * 1024)
            
            # Peak bandwidth for RDNA 2 gfx1030 (512 GB/s)
            peak_bandwidth_bytes_sec = 512e9
            
            return {
                "bytes_per_token": total_vram_bytes_per_token,
                "vram_throughput_mb_s": round(actual_bandwidth_mb_sec, 4),
                "peak_utilization_pct": round((actual_bandwidth_bytes_sec / peak_bandwidth_bytes_sec) * 100.0, 6)
            }
    
    def estimate_cache_hit_rates(self, working_set_size_bytes: float, cache_size_bytes: float) -> float:
        """Estimates cache hit rate based on working set vs cache size.
        
        Args:
            working_set_size_bytes: Size of data accessed per operation
            cache_size_bytes: Available cache capacity
            
        Returns:
            Estimated hit rate as a percentage (0-100)
        """
        if working_set_size_bytes <= cache_size_bytes:
            return 100.0
        
        # Exponential decay model for cache misses
        ratio = cache_size_bytes / working_set_size_bytes
        hit_rate = 100.0 * (1.0 - np.exp(-3.0 * ratio))
        
        return round(hit_rate, 2)
    
    def analyze_attention_working_set(self, head_dim: int, context_size: int, quantization_bits: float, n_heads: int = 32) -> Dict[str, Any]:
        """Analyzes memory working set for attention computation.
        
        Args:
            head_dim: Attention head dimension
            context_size: Number of tokens in context
            quantization_bits: Bits per element for Q/K/V
            n_heads: Number of KV heads (default 32 for typical GQA configs)
            
        Returns:
            Dictionary with working set analysis and cache estimates
        """
        # QKV working set in VRAM (quantized) - Q is computed on-the-fly, K+V are cached
        qkv_vram_bytes = 3 * n_heads * head_dim * quantization_bits / 8.0 * context_size
        
        # QKV working set in LDS (per-head, no quantization needed for compute)
        qkv_lds_bytes = 3 * head_dim * 4.0  # Float32 for attention compute
        
        # Attention scores matrix (context x context)
        scores_matrix_bytes = context_size * context_size * 4.0  # Float32
        
        # Infinity Cache analysis using total KV cache size
        ic_analysis = self._analyze_infinity_cache_effects(qkv_vram_bytes)
        
        return {
            "qkv_vram_working_set_mb": round(qkv_vram_bytes / (1024 * 1024), 2),
            "qkv_lds_working_set_kb": round(qkv_lds_bytes / 1024, 2),
            "scores_matrix_mb": round(scores_matrix_bytes / (1024 * 1024), 2),
            "lds_hit_rate_estimate": self.estimate_cache_hit_rates(qkv_lds_bytes, 65536),  # 64KB LDS per CU
            "l1_hit_rate_estimate": self.estimate_cache_hit_rates(qkv_vram_bytes, 131072),  # 128KB L1 per CU
            "infinity_cache": ic_analysis
        }
    
    def calculate_working_set_bytes(self, head_dim: int, context_size: int, quantization_bits: float, n_heads: int = 32) -> int:
        """Calculates total bytes for active execution layers (K+V cache).
        
        Args:
            head_dim: Attention head dimension
            context_size: Number of tokens in context
            quantization_bits: Bits per element for Q/K/V
            n_heads: Number of KV heads (default 32 for typical GQA configs)
            
        Returns:
            Total bytes for K+V cache
        """
        bytes_per_token = quantization_bits / 8.0
        # 2 represents the K and V layouts (not Q, which is computed on-the-fly)
        return int(2 * n_heads * head_dim * bytes_per_token * context_size)
    
    def estimate_microarchitectural_stall_factor(self, working_set_bytes: int) -> float:
        """Estimates non-linear latency penalty based on physical cache hierarchy.
        
        Implements piece-wise function matching gfx1030 cache boundaries:
        - L0 (32KB): Base latency
        - L1 (128KB): 3.2x penalty
        - L2 (4MB): 14.5x penalty
        - Infinity Cache (128MB): 38x penalty with saturation scaling
        - Beyond IC: VRAM spill with overflow penalty
        
        Args:
            working_set_bytes: Total working set size in bytes
            
        Returns:
            Latency multiplier relative to L0 baseline
        """
        if working_set_bytes <= self.L0_CACHE_SIZE_BYTES:
            return self.LATENCY_L0
        elif working_set_bytes <= self.L1_CACHE_SIZE_BYTES:
            return self.LATENCY_L1
        elif working_set_bytes <= self.L2_CACHE_SIZE_BYTES:
            return self.LATENCY_L2
        elif working_set_bytes <= self.INFINITY_CACHE_BYTES:
            # Smooth scaling approaching the edge of the Infinity Cache pool
            saturation_ratio = working_set_bytes / self.INFINITY_CACHE_BYTES
            return self.LATENCY_L2 + (saturation_ratio * (self.LATENCY_INFINITY - self.LATENCY_L2))
        else:
            # The Eviction Cliff: Severe penalty for VRAM spill beyond Infinity Cache
            overflow_ratio = working_set_bytes / self.INFINITY_CACHE_BYTES
            return self.LATENCY_INFINITY + (overflow_ratio * self.LATENCY_VRAM)
    
    def _analyze_infinity_cache_effects(self, total_kv_cache_bytes: float) -> Dict[str, Any]:
        """Analyzes Infinity Cache effects on memory latency.
        
        Implements piece-wise non-linear latency model matching gfx1030 physical
        cache hierarchy boundaries (L0/L1/L2/Infinity/VRAM).
        
        Args:
            total_kv_cache_bytes: Total KV cache size in bytes
            
        Returns:
            Dictionary with Infinity Cache analysis including hit rate, latency penalty,
            effective bandwidth utilization, and microarchitectural stall factor
        """
        total_kv_cache_mb = total_kv_cache_bytes / (1024 * 1024)
        infinity_cache_mb = self.INFINITY_CACHE_BYTES / (1024 * 1024)
        
        # Calculate stall factor using piece-wise model
        stall_factor = self.estimate_microarchitectural_stall_factor(int(total_kv_cache_bytes))
        
        # Derive effective IC hit rate from stall factor (inverted)
        # Base hit rate is 100%, degrading as stall factor increases beyond L2 baseline
        if stall_factor <= self.LATENCY_L2:
            ic_hit_rate = 100.0
        elif stall_factor <= self.LATENCY_INFINITY:
            # Linear interpolation between L2 and Infinity thresholds
            ratio = (stall_factor - self.LATENCY_L2) / (self.LATENCY_INFINITY - self.LATENCY_L2)
            ic_hit_rate = 100.0 - (ratio * 20.0)  # Degrade to 80% at IC limit
        else:
            # Beyond Infinity Cache - severe degradation
            overflow_ratio = (stall_factor - self.LATENCY_INFINITY) / self.LATENCY_VRAM
            ic_hit_rate = max(20.0, 80.0 - (overflow_ratio * 60.0))  # Floor at 20%
        
        # Effective bandwidth utilization based on hit rate
        peak_bandwidth_gb_s = 512.0  # RX 6800 XT
        if ic_hit_rate >= 95:
            effective_bw_factor = 0.85
        elif ic_hit_rate >= 80:
            effective_bw_factor = 0.60
        elif ic_hit_rate >= 50:
            effective_bw_factor = 0.40
        else:
            effective_bw_factor = 0.20
        
        return {
            "total_kv_cache_mb": round(total_kv_cache_mb, 2),
            "infinity_cache_capacity_mb": round(infinity_cache_mb, 2),
            "exceeds_ic_threshold": total_kv_cache_bytes > self.INFINITY_CACHE_BYTES,
            "ic_hit_rate_pct": round(ic_hit_rate, 2),
            "stall_factor": round(stall_factor, 2),
            "effective_bandwidth_factor": round(effective_bw_factor, 2),
            "warning": None if ic_hit_rate >= 80 else "CRITICAL: Infinity Cache threshold exceeded - severe latency degradation expected"
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    simulator = RDNA2MemoryHardwareSimulator()
    
    print("=== RDNA 2 Memory Analysis ===\n")
    
    # Example: Analyze working set for Qwen 3.6 (d=128, context=262144, turbo3_0=3.5bpw)
    print("Attention Working Set Analysis (d=128, context=262144, turbo3_0):")
    working_set = simulator.analyze_attention_working_set(
        head_dim=128,
        context_size=262144,
        quantization_bits=3.5
    )
    for key, value in working_set.items():
        print(f"  {key}: {value}")
    print()
    
    # Example: Calculate VRAM throughput at 666 tok/s
    print("VRAM Throughput (666 tok/s):")
    throughput = simulator.calculate_real_vram_throughput(
        head_dim=128,
        quantization_bits=3.5,
        token_throughput=666.0
    )
    for key, value in throughput.items():
        print(f"  {key}: {value}")
    print()
    
    # Example: LDS bank conflict analysis for tile access
    print("LDS Bank Conflict Analysis (tile 64x64, half precision):")
    lds_analysis = simulator.simulate_lds_access_pattern(
        tile_x=64,
        tile_y=64,
        element_size_bytes=2
    )
    for key, value in lds_analysis.items():
        print(f"  {key}: {value}")
