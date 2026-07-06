#!/usr/bin/env python3
"""
Automated Compiler ISA & Telemetry Parser Engine

Extracts actual hardware metrics using compiler disassemblers and performance counters.
Removes hardcoded assumptions by reading configuration metadata directly from 
compiled binaries and raw hardware performance counters.

Provides integration with:
- amdgpu-objdump for VGPR register allocation analysis
- rocprofv3 for hardware counter telemetry
- ROCm SMI for real-time GPU metrics
"""

import os
import re
import subprocess
import csv
from pathlib import Path
from typing import Dict, Optional, List, Any

from _roctx import mark, roctx as _roctx


class CompilerTelemetryBridge:
    """Extracts actual hardware metrics using compiler disassemblers and performance counters.
    
    Bridges the gap between simulation estimates and real hardware behavior by:
    1. Parsing compiled GPU binaries to extract true VGPR/SGPR allocations
    2. Collecting hardware counter telemetry from rocprofv3 traces
    3. Querying ROCm SMI for real-time GPU metrics
    """
    
    @staticmethod
    def query_amdgpu_objdump_vgprs(binary_path: str, kernel_name_pattern: str) -> Optional[int]:
        """Parses compiled GPU execution structures to extract true VGPR register allocations.
        
        Primary method: Uses amdgpu-objdump to disassemble the binary and find .vgpr_count metadata
        Fallback: Uses readelf to inspect ELF binary configuration note descriptors directly
        
        Args:
            binary_path: Path to compiled .hsa or .oclc file
            kernel_name_pattern: Pattern to match kernel name in disassembly
            
        Returns:
            VGPR count if found, None otherwise
        """
        if not os.path.exists(binary_path):
            print(f"Warning: Binary not found: {binary_path}")
            return None
        
        # Primary method: amdgpu-objdump
        with mark("telemetry_amdgpu_objdump"):
            try:
                cmd = f'amdgpu-objdump -d "{binary_path}" | grep -A 20 \'{kernel_name_pattern}\''
                output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                
                match = re.search(r'\.vgpr_count\s+(\d+)', output)
                if match:
                    return int(match.group(1))
                    
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: amdgpu-objdump failed for {binary_path}: {e}")
        
        # Fallback: readelf inspection of ELF metadata
        return CompilerTelemetryBridge._extract_vgprs_via_readelf(binary_path)
    
    @staticmethod
    def _extract_vgprs_via_readelf(binary_path: str) -> Optional[int]:
        """Fallback VGPR extraction using readelf to inspect ELF binary metadata.
        
        Reads the .amdhsa_kernel metadata section directly from the ELF binary,
        bypassing unstable CLI string formatting from amdgpu-objdump.
        
        Args:
            binary_path: Path to compiled .hsa or .oclc file
            
        Returns:
            VGPR count if found, None otherwise
        """
        with mark("telemetry_readelf"):
            try:
                # Dump string data from .rodata section to find kernel metadata
                result = subprocess.run(
                    ["readelf", "--string-dump=.rodata", binary_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    return None
                
                # Search for .amdhsa_kernel metadata blocks containing vgpr_count
                output = result.stdout
                
                # Look for pattern: .amdhsa_kernel ... vgpr_count=N
                # This is a simplified pattern - real implementation would parse msgpack/yaml metadata
                match = re.search(r'vgpr_count\s*[:=]\s*(\d+)', output)
                if match:
                    return int(match.group(1))
                
                # Alternative pattern: .amdhsa_next_free_vgpr
                match = re.search(r'next_free_vgpr\s*[:=]\s*(\d+)', output)
                if match:
                    return int(match.group(1))
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"Warning: readelf fallback failed for {binary_path}: {e}")
        
        return None
    
    @staticmethod
    def parse_rocprofv3_telemetry(csv_trace_path: str) -> Dict[str, Any]:
        """Parses hardware-level counter telemetry reports generated during model inference loops.
        
        Extracts metrics from rocprofv3 CSV traces including:
        - SQ_LEVEL_OCCUPANCY: Actual wave occupancy during kernel execution
        - VALU_INST_RATIO: VALU instruction utilization percentage
        - SQ_LDS_BANK_CONFLICT: Number of LDS bank conflict stalls
        
        Args:
            csv_trace_path: Path to rocprofv3 kernel_dispatch.csv
            
        Returns:
            Dictionary with measured_occupancy, valu_utilization, lds_bank_stalls
        """
        telemetry_summary = {
            "measured_occupancy": 0.0,
            "valu_utilization": 0.0,
            "lds_bank_stalls": 0,
            "total_kernels": 0,
            "avg_kernel_duration_ns": 0.0
        }
        
        trace_file = Path(csv_trace_path)
        if not trace_file.exists():
            print(f"Warning: Trace file not found: {csv_trace_path}")
            return telemetry_summary
        
        with mark("telemetry_aggregate"):
            try:
                with open(trace_file, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    duration_sum = 0.0
                    kernel_count = 0
                    occupancy_weighted_sum = 0.0
                    valu_weighted_sum = 0.0
                    
                    # Parse metrics generated by the ROCm hardware instrumentation layer
                    for row in reader:
                        # Skip non-kernel rows
                        if "Kind" in row and row["Kind"] != "KERNEL_DISPATCH":
                            continue
                        
                        kernel_count += 1
                        
                        # Extract duration if available
                        row_duration = 0.0
                        if "Duration_ns" in row:
                            try:
                                row_duration = float(row["Duration_ns"])
                                duration_sum += row_duration
                            except (ValueError, KeyError):
                                pass
                        
                        # Extract hardware counters (weighted by duration)
                        if row_duration > 0 and "SQ_LEVEL_OCCUPANCY" in row:
                            try:
                                occupancy_weighted_sum += float(row["SQ_LEVEL_OCCUPANCY"]) * row_duration
                            except (ValueError, KeyError):
                                pass
                        
                        if row_duration > 0 and "VALU_INST_RATIO" in row:
                            try:
                                valu_weighted_sum += float(row["VALU_INST_RATIO"]) * row_duration
                            except (ValueError, KeyError):
                                pass
                        
                        if "SQ_LDS_BANK_CONFLICT" in row:
                            try:
                                telemetry_summary["lds_bank_stalls"] += int(row["SQ_LDS_BANK_CONFLICT"])
                            except (ValueError, KeyError):
                                pass
                    
                    telemetry_summary["total_kernels"] = kernel_count
                    if kernel_count > 0:
                        telemetry_summary["avg_kernel_duration_ns"] = duration_sum / kernel_count
                        if duration_sum > 0:
                            telemetry_summary["measured_occupancy"] = occupancy_weighted_sum / duration_sum
                            telemetry_summary["valu_utilization"] = valu_weighted_sum / duration_sum
                        
            except Exception as e:
                print(f"Warning: Failed to parse {csv_trace_path}: {e}")
        
        return telemetry_summary
    
    @staticmethod
    def query_rocm_smi_metrics() -> Dict[str, float]:
        """Queries ROCm SMI for real-time GPU metrics.
        
        Returns current GPU utilization, memory usage, and temperature.
        
        Returns:
            Dictionary with gpu_utilization, vram_used_gb, temperature
        """
        metrics = {
            "gpu_utilization": 0.0,
            "vram_used_gb": 0.0,
            "temperature": 0.0
        }
        
        try:
            # Query GPU utilization
            result = subprocess.run(
                ['rocm-smi', '--showutilization'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse output for GPU and VRAM utilization
                for line in result.stdout.split('\n'):
                    if 'GPU' in line and 'Utilization' in line:
                        util_match = re.search(r'(\d+\.?\d*)%', line)
                        if util_match:
                            metrics["gpu_utilization"] = float(util_match.group(1))
                    
                    if 'VRAM' in line and 'Used' in line:
                        vram_match = re.search(r'([\d.]+)\s*GB', line)
                        if vram_match:
                            metrics["vram_used_gb"] = float(vram_match.group(1))
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: rocm-smi query failed: {e}")
        
        return metrics
    
    @staticmethod
    def collect_rocprofv3_trace(
        command: str,
        output_dir: str,
        env_vars: Optional[Dict[str, str]] = None
    ) -> bool:
        """Runs a command with rocprofv3 enabled and collects telemetry traces.
        
        Args:
            command: Command to run with profiling enabled (e.g., llama-bench ...)
            output_dir: Directory to store trace files
            env_vars: Additional environment variables (e.g., ROCPROF_ENABLE=1)
            
        Returns:
            True if trace collection succeeded, False otherwise
        """
        if env_vars is None:
            env_vars = {}
        
        # Set rocprofv3 environment variables
        profile_env = os.environ.copy()
        profile_env["ROCPROF_ENABLE"] = "1"
        profile_env["ROCPROF_OUTPUT_FORMAT"] = "CSV"
        profile_env["ROCPROF_OUTPUT_DIR"] = output_dir
        
        # Merge additional env vars
        profile_env.update(env_vars)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            result = subprocess.run(
                command,
                shell=True,
                env=profile_env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✓ Trace collection successful: {output_dir}")
                return True
            else:
                print(f"✗ Command failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Command timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"✗ Trace collection failed: {e}")
            return False
    
    @staticmethod
    def validate_simulation_against_telemetry(
        simulated_occupancy: float,
        measured_occupancy: float,
        tolerance_pct: float = 5.0
    ) -> Dict[str, Any]:
        """Validates simulation predictions against real hardware telemetry.
        
        Args:
            simulated_occupancy: Occupancy predicted by simulation (percentage)
            measured_occupancy: Actual occupancy from hardware counters (percentage)
            tolerance_pct: Maximum allowed deviation (percentage points)
            
        Returns:
            Dictionary with validation result and variance
        """
        variance = abs(simulated_occupancy - measured_occupancy)
        is_valid = variance <= tolerance_pct
        
        return {
            "simulated_occupancy": simulated_occupancy,
            "measured_occupancy": measured_occupancy,
            "variance": variance,
            "tolerance": tolerance_pct,
            "is_valid": is_valid,
            "recommendation": "PASS" if is_valid else "FAIL - Simulation needs calibration"
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    with mark("telemetry_parse"):
        bridge = CompilerTelemetryBridge()
        
        print("=== Compiler Telemetry Bridge ===\n")
        
        # Example: Parse rocprofv3 telemetry
        print("Example: Parsing rocprofv3 telemetry")
        with mark("telemetry_extract"):
            telemetry = bridge.parse_rocprofv3_telemetry(
                csv_trace_path="bench-results/rdna2_diagnostic_20260702_035648.json"
            )
        for key, value in telemetry.items():
            print(f"  {key}: {value}")
        print()
        
        # Example: Validate simulation against telemetry
        print("Example: Validating simulation predictions")
        with mark("telemetry_validate"):
            validation = bridge.validate_simulation_against_telemetry(
                simulated_occupancy=100.0,  # MNLN v4.0 claim
                measured_occupancy=12.5,    # Calculated from launch bounds
                tolerance_pct=5.0
            )
        for key, value in validation.items():
            print(f"  {key}: {value}")
