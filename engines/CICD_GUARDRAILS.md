# MNLN v4.1 - CI/CD Operational Guardrails

## Overview

This document defines the operational constraints and best practices for running MNLN v4.1 simulation engines in automated CI/CD pipelines.

---

## Critical Path Constraints

### 1. Execution Directory Requirement

**Constraint:** MNLN scripts MUST be executed from the repository root directory where `engines/` resides.

**Reason:** The simulation engines use `Path(__file__).resolve().parent` to locate sibling modules. This relative path resolution breaks if the script is invoked via:
- Global symbolic links
- External execution paths (e.g., `python3 /abs/path/to/engines/mlnn.py`)
- Non-standard working directories

**Correct Usage:**
```bash
# FROM REPOSITORY ROOT
cd /path/to/llama.cpp
python3 engines/mlnn.py --mode kernels

# OR from engines directory directly
cd /path/to/llama.cpp/engines
python3 mlnn.py --mode kernels
```

**Incorrect Usage (will fail):**
```bash
# WRONG - relative path resolution breaks
cd /tmp
python3 /path/to/llama.cpp/engines/mlnn.py --mode kernels

# WRONG - symlink may resolve incorrectly
ln -s /path/to/llama.cpp/engines/mlnn.py /usr/local/bin/mlnn
mlnn --mode kernels
```

### 2. sys.path Mutation Warning

**Current Implementation:**
```python
_engines_path = Path(__file__).resolve().parent
if str(_engines_path) not in sys.path:
    sys.path.insert(0, str(_engines_path))
```

**Risk:** Dynamic path injection can cause module name collisions if the user's environment contains conflicting module names.

**Mitigation:** This is acceptable for internal CI/CD use but should be avoided in production packaging. Future migration to `pyproject.toml` with proper package structure is recommended.

---

## CI/CD Pipeline Integration

### GitHub Actions Example

```yaml
name: MNLN v4.1 Validation

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install numpy scipy
      
      - name: Run MNLN v4.1 validation
        working-directory: ${{ github.workspace }}
        run: |
          # CRITICAL: Must run from repository root
          python3 engines/mlnn.py --mode kernels --quick
```

### Local CI Testing

```bash
# Test from repository root (CORRECT)
cd /path/to/llama.cpp
python3 engines/mlnn.py --mode kernels --quick

# Test with hardware telemetry
rocprofv3 --hip-trace --stat -d ./telemetry_output/ \
  ./build/bin/llama-bench -m model.gguf -n 1000

python3 engines/mlnn_v40_runner.py --profile-dir telemetry_output
```

---

## Error Handling Behavior

### Import Failure (Graceful Degradation)

When RDNA2 simulation modules fail to load:

```
[WARN] RDNA2 accurate simulation modules not available: No module named 'rdna2_occupancy_solver'
[WARN] Falling back to legacy v4.0 metrics (physics-inaccurate)
[INFO] Set --allow-fallback explicitly if this is intentional
```

**Behavior:**
- `HAS_ACCURATE_SIM = False`
- Script continues with legacy v4.0 metrics
- No hard exit (backward compatible)

### Validation Gate (Hard Exit)

When `--mode kernels` is requested without accurate simulation:

```
[FATAL] RDNA2 accurate simulation modules failed to load.
[FATAL] Required for --mode kernels with physics-accurate occupancy/memory models.
[FATAL] Run with --allow-fallback to override with legacy v4.0 metrics (NOT RECOMMENDED).
```

**Behavior:**
- `sys.exit(1)` - Pipeline fails immediately
- Forces operator to acknowledge degraded state
- Prevents silent submission of inaccurate metrics

### Explicit Fallback Override

```bash
python3 engines/mlnn.py --mode kernels --allow-fallback
```

**Output:**
```
[WARN] Using legacy v4.0 metrics (--allow-fallback set)
```

**Behavior:**
- Continues execution with legacy metrics
- Operator explicitly acknowledges degraded state
- Use only when modules are intentionally unavailable

---

## Performance Characteristics

### Simulation Speed Benchmarks

| Operation | Context Size | Time |
|-----------|--------------|------|
| Occupancy calculation | N/A | 0.89 ms/call |
| Full kernel analysis | 12 kernels | 0.42 ms |
| Memory working set | 4K tokens | 0.04 ms |
| Memory working set | 112K tokens | 0.01 ms |

**Conclusion:** All operations complete in <1ms. Safe to run in every CI pipeline iteration without performance impact.

### Infinity Cache Threshold Analysis

| Context Size | KV Cache (MB) | Exceeds 128MB IC? | IC Hit Rate |
|--------------|---------------|-------------------|-------------|
| 4,096 | 0.66 | No ✓ | 100% |
| 16,384 | 2.62 | No ✓ | 100% |
| 32,768 | 5.25 | No ✓ | 100% |
| 65,536 | 10.50 | No ✓ | 100% |
| 112,000 | 17.94 | No ✓ | 100% |

**Note:** Current Qwen 3.6 configuration (d=128, turbo3_0) stays well within Infinity Cache limits. The cliff detection model is in place for future configurations that may exceed the threshold.

---

## Module Dependencies

### Required Modules (for --mode kernels)

- `rdna2_occupancy_solver.py` - Physics-accurate occupancy calculation
- `rdna2_memory_simulator.py` - Multi-tier memory hierarchy simulation
- `compiler_telemetry_bridge.py` - Hardware counter telemetry parser

### Optional Modules

- `master_debug_turbo.py` - Turbo3 quantization validation (standalone)
- `mlnn_v40_runner.py` - Standalone diagnostic runner

---

## Future Migration Path

### Recommended: pyproject.toml with Local Submodules

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mnnl-v41"
version = "4.1.0"
description = "Master Neural Learning Network v4.1 - RDNA 2 Simulation Engines"

[tool.setuptools.packages.find]
where = ["engines"]
include = ["engines*"]

[tool.setuptools.package-data]
"engines" = ["*.py"]
```

**Benefits:**
- Eliminates sys.path mutation
- Proper package discovery
- Standard Python packaging tools compatible
- Future-proof for distribution

**Timeline:** Deferred to future sprint (current sprint focuses on validation and correctness)

---

## Contact & Support

For issues with MNLN v4.1 simulation engines, refer to:
- `engines/mlnn.py` - Main entry point
- `engines/rdna2_occupancy_solver.py` - Occupancy calculations
- `engines/rdna2_memory_simulator.py` - Memory simulation
- `engines/compiler_telemetry_bridge.py` - Hardware telemetry
