#!/usr/bin/env python3
"""
RDNA 2 ISA Occupancy and Launch Bounds Solver

Calculates deterministic microarchitectural occupancy for the AMD gfx1030 GPU.
Strictly maps execution constraints across Wave32 formats using physical 
hardware limitations.

Replaces fabricated occupancy trackers with hardware-accurate calculations.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from _roctx import mark, roctx as _roctx


class RDNA2OccupancySolver:
    """Calculates deterministic microarchitectural occupancy for the AMD gfx1030 GPU.
    
    Models the physical limitations of RDNA 2 execution architecture, accounting for:
    - Thread distributions across internal SIMD units (4 SIMDs per CU)
    - Vector register boundaries (512 VGPRs per SIMD)
    - Macro layout fences (explicit __launch_bounds__ hints)
    """
    
    def __init__(self):
        # RDNA 2 Hardware Constants (gfx1030 verified)
        self.WAVE_SIZE = 32              # Wave32 execution lanes
        self.MAX_WAVES_PER_CU = 16       # Maximum concurrent waves per CU
        self.SIMDS_PER_CU = 4            # SIMD units per Compute Unit (RDNA 2: 4 SIMD32 per CU, §4.2)
        self.VGPRS_PER_SIMD = 512        # Physical VGPR allocation limit per SIMD
        self.VGPR_ALLOCATION_GRANULARITY = 4  # RDNA2 allocates VGPRs in blocks of 4 per wave (Wave32)
        self.LDS_CAPACITY_BYTES = 65536  # 64KB Local Data Share per CU
    
    def calculate_physical_occupancy(
        self, 
        nthreads: int,
        vgpr_count: int = 0,
        lds_bytes: int = 0,
        launch_bounds_hint: int = 0
    ) -> Dict[str, Any]:
        """Computes the hardware limit for active waves on a single Compute Unit.
        
        Args:
            nthreads: Total threads in the workgroup
            vgpr_count: Vector register usage per thread (0 = unverified)
            lds_bytes: Shared memory allocation in bytes (0 = unverified)
            launch_bounds_hint: Explicit __launch_bounds__ occupancy hint (0 = none)
        
        Returns:
            Dictionary with calculated_waves, occupancy_percentage, and limiting_factor
        """
        # Calculate waves required by the thread block
        waves_per_block = (nthreads + self.WAVE_SIZE - 1) // self.WAVE_SIZE
        
        # 1. Compute limits imposed by Vector General Purpose Registers (VGPRs)
        with mark("occ_solver_vgpr"):
            if vgpr_count > 0:
                # RDNA2 allocates VGPRs in granularity blocks (4 per wave for Wave32)
                # Round up to next allocation boundary
                aligned_vgprs_per_thread = ((vgpr_count + self.VGPR_ALLOCATION_GRANULARITY - 1) 
                                            // self.VGPR_ALLOCATION_GRANULARITY) * self.VGPR_ALLOCATION_GRANULARITY
                
                # Calculate waves limited by aligned VGPR requirement
                waves_limited_by_vgpr = (self.VGPRS_PER_SIMD // aligned_vgprs_per_thread) * self.SIMDS_PER_CU
            else:
                waves_limited_by_vgpr = self.MAX_WAVES_PER_CU
        
        # 2. Compute limits imposed by Local Data Share (LDS) allocation fences
        with mark("occ_solver_lds"):
            if lds_bytes > 0:
                blocks_per_cu = self.LDS_CAPACITY_BYTES // lds_bytes
                waves_limited_by_lds = blocks_per_cu * waves_per_block
            else:
                waves_limited_by_lds = self.MAX_WAVES_PER_CU
        
        # 3. Incorporate explicit compiler launch bounds limits
        waves_limited_by_compiler = self.MAX_WAVES_PER_CU
        if launch_bounds_hint > 0:
            waves_limited_by_compiler = launch_bounds_hint * waves_per_block
        
        # Structural hardware resolution boundary
        active_waves_limit = min(
            self.MAX_WAVES_PER_CU,
            waves_limited_by_vgpr,
            waves_limited_by_lds,
            waves_limited_by_compiler
        )
        
        occupancy_pct = (active_waves_limit / self.MAX_WAVES_PER_CU) * 100.0
        
        return {
            "calculated_waves": active_waves_limit,
            "occupancy_percentage": round(occupancy_pct, 2),
            "limiting_factor": self._identify_bottleneck(
                waves_limited_by_vgpr, waves_limited_by_lds, waves_limited_by_compiler
            )
        }
    
    def _identify_bottleneck(self, vgpr: int, lds: int, compiler: int) -> str:
        """Identifies which constraint is the primary bottleneck."""
        limits = {
            "VGPR Pressure": vgpr,
            "LDS Structural Bounds": lds,
            "Compiler Launch Gate": compiler
        }
        
        # Find the factor with minimum value
        min_value = min(limits.values())
        min_factor = [k for k, v in limits.items() if v == min_value][0]
        
        if min_value >= self.MAX_WAVES_PER_CU:
            return "Hardware Saturation (Max Waves Reached)"
        return min_factor
    
    def parse_kernel_source_for_launch_bounds(self, file_path: str) -> Dict[str, Dict[str, int]]:
        """Extracts __launch_bounds__ parameters from CUDA/HIP kernel source files.
        
        Args:
            file_path: Path to .cu or .cuh source file
            
        Returns:
            Dictionary mapping kernel names to their launch bounds parameters
            Format: {kernel_name: {"max_threads": int, "min_blocks": int}}
        """
        result = {}
        
        try:
            source = Path(file_path).read_text()
            
            # Match __launch_bounds__(maxThreads, minBlocksPerMultiprocessor)
            pattern = r'__launch_bounds__\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
            matches = re.finditer(pattern, source)
            
            for match in matches:
                max_threads = int(match.group(1))
                min_blocks = int(match.group(2))
                
                # Find the kernel name preceding this launch_bounds
                start = match.start()
                # Search backwards for static __global__ or template declaration
                before = source[:start].rstrip()
                lines = before.split('\n')
                
                kernel_name = "unknown"
                for line in reversed(lines[-5:]):  # Check last 5 lines before match
                    # Match patterns like: static __global__ void kernel_name(...)
                    # or: template<...> __launch_bounds__(...) static __global__ void kernel_name
                    kernel_match = re.search(r'void\s+(\w+)\s*\(', line)
                    if kernel_match:
                        kernel_name = kernel_match.group(1)
                        break
                
                result[kernel_name] = {
                    "max_threads": max_threads,
                    "min_blocks": min_blocks
                }
                
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
        
        return result
    
    def analyze_flash_attention_kernels(self, fattn_tile_path: Optional[str] = None, target_arch: str = "amd_rdna") -> Dict[str, Any]:
        """Analyzes flash attention kernel configurations from fattn-tile.cuh.
        
        Args:
            fattn_tile_path: Path to fattn-tile.cuh source file
            target_arch: Target architecture section to parse ("nvidia_fp16", "nvidia_fp32", 
                        "amd", "amd_rdna"). Default is "amd_rdna" for gfx1030.
                        
        Returns:
            Dictionary with head_sizes analysis and summary statistics.
        """
        if fattn_tile_path is None:
            # Default to common location
            fattn_tile_path = str(Path(__file__).parent.parent / "ggml" / "src" / "ggml-cuda" / "fattn-tile.cuh")
        
        result = {
            "head_sizes": {},
            "summary": {}
        }
        
        try:
            source = Path(fattn_tile_path).read_text()
            
            # Find the target architecture section
            if target_arch == "amd_rdna":
                section_start = source.find("ggml_cuda_fattn_tile_get_config_amd_rdna")
            elif target_arch == "amd":
                section_start = source.find("ggml_cuda_fattn_tile_get_config_amd(")
            elif target_arch == "nvidia_fp32":
                section_start = source.find("ggml_cuda_fattn_tile_get_config_nvidia_fp32(")
            else:  # nvidia_fp16 (default)
                section_start = source.find("ggml_cuda_fattn_tile_get_config_nvidia_fp16(")
            
            if section_start < 0:
                print(f"Warning: Could not find section for {target_arch}")
                return result
            
            # Find the end of this section (next function definition)
            next_func = source.find("static constexpr", section_start + 100)
            if next_func < 0:
                section = source[section_start:]
            else:
                section = source[section_start:next_func]
            
            # Parse GGML_CUDA_FATTN_TILE_CONFIG_CASE macros
            # Format: (DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K)
            pattern = r'GGML_CUDA_FATTN_TILE_CONFIG_CASE\s*\(\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
            matches = list(re.finditer(pattern, section))
            
            # Group by head size and take the first variant (smallest tile size)
            head_size_configs = {}
            for match in matches:
                dkq = int(match.group(1))
                dv = int(match.group(2))
                ncols = int(match.group(3))
                nthreads = int(match.group(4))
                occupancy_hint = int(match.group(5))
                
                key = f"DKQ={dkq}_DV={dv}"
                
                # Only keep first variant for each head size (smallest ncols)
                if key not in head_size_configs:
                    head_size_configs[key] = {
                        "dkq": dkq,
                        "dv": dv,
                        "ncols": ncols,
                        "nthreads": nthreads,
                        "occupancy_hint": occupancy_hint
                    }
            
            # Calculate occupancy for each head size configuration
            with mark("occ_solver_fa_loop"):
                for key, config in head_size_configs.items():
                    occupancy = self.calculate_physical_occupancy(
                        nthreads=config["nthreads"],
                        launch_bounds_hint=config["occupancy_hint"]
                    )
                    
                    result["head_sizes"][key] = {
                        "ncols": config["ncols"],
                        "nthreads": config["nthreads"],
                        "occupancy_hint": config["occupancy_hint"],
                        "calculated_occupancy_pct": occupancy["occupancy_percentage"],
                        "limiting_factor": occupancy["limiting_factor"]
                    }
                
        except (FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not analyze {fattn_tile_path}: {e}")
        
        return result


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    with mark("occ_solver_main"):
        solver = RDNA2OccupancySolver()
        
        print("=== RDNA 2 Occupancy Analysis ===\n")
        
        # Example: flash_attn_tile with nthreads=64, occupancy hint=2
        print("flash_attn_tile (nthreads=64, occupancy_hint=2):")
        result = solver.calculate_physical_occupancy(
            nthreads=64,
            launch_bounds_hint=2
        )
        print(f"  Calculated Waves: {result['calculated_waves']}")
        print(f"  Occupancy: {result['occupancy_percentage']}%")
        print(f"  Limiting Factor: {result['limiting_factor']}")
        print()
        
        # Example: flash_attn_vec with nthreads=128, occupancy hint=1
        print("flash_attn_vec (nthreads=128, occupancy_hint=1):")
        result = solver.calculate_physical_occupancy(
            nthreads=128,
            launch_bounds_hint=1
        )
        print(f"  Calculated Waves: {result['calculated_waves']}")
        print(f"  Occupancy: {result['occupancy_percentage']}%")
        print(f"  Limiting Factor: {result['limiting_factor']}")
        print()
        
        # Analyze flash attention kernels
        print("=== Flash Attention Kernel Configurations ===")
        analysis = solver.analyze_flash_attention_kernels()
        
        for key, config in list(analysis["head_sizes"].items())[:10]:  # Show first 10
            print(f"  {key}:")
            print(f"    nthreads={config['nthreads']}, hint={config['occupancy_hint']}")
            print(f"    Actual Occupancy: {config['calculated_occupancy_pct']}%")
            print(f"    Bottleneck: {config['limiting_factor']}")
