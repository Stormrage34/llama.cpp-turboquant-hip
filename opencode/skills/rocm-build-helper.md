---
name: rocm-build-helper
description: ROCm 7.13 + CMake build troubleshooting for gfx1030
triggers: ["build", "cmake", "hipcc", "compile", "error", "ROCm"]
---
rocm-build-helper.md
# ROCm Build Helper (gfx1030 / ROCm 7.13-nightly)

## Required Environment
```bash
export ROCM_PATH=/opt/rocm-7.13
export PATH=$ROCM_PATH/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Force gfx1030 detection
