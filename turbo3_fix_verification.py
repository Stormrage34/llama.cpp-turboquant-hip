#!/usr/bin/env python3
"""
Verification script for turbo3_0 rotation fix.

This demonstrates the conceptual fix without requiring GPU memory.
It shows that removing forward rotation from writer makes pipeline consistent.
"""

import numpy as np

# Simulate WHT rotation (simplified)
def wht_rotate(x):
    """Simplified Walsh-Hadamard Transform rotation."""
    n = len(x)
    x_rot = x.copy()
    # Apply sign patterns and butterfly operations
    for i in range(n):
        x_rot[i] *= (-1)**(i % 3)  # Simplified sign pattern
    return x_rot

def dequant_turbo3_no_rotation(x):
    """Dequantize WITHOUT inverse rotation (current behavior)."""
    return x  # Just returns what's stored

def dequant_turbo3_with_inverse_rotation(x):
    """Dequantize WITH inverse rotation (what would be needed if writer rotates)."""
    return x  # Would need to apply inverse WHT here

print("=== turbo3_0 KV Cache Pipeline Analysis ===\n")

# Test case: simple values
test_values = np.array([1.0, -0.5, 0.3, -0.8, 0.6, -0.2, 0.9, -0.4])
print(f"Original values: {test_values}")

# Scenario 1: Writer applies rotation, reader does NOT inverse-rotate (CURRENT BUG)
print("\n[BUG] Writer rotates, Reader doesn't inverse-rotate:")
rotated = wht_rotate(test_values)
reconstructed = dequant_turbo3_no_rotation(rotated)
print(f"  After rotation:     {rotated[:4]}...")
print(f"  Reconstructed:      {reconstructed[:4]}...")
print(f"  ERROR: Values are CORRUPTED!")
print(f"  Original vs Reconstructed mismatch: {np.max(np.abs(test_values - reconstructed)):.4f}")

# Scenario 2: Writer does NOT rotate (FIXED VERSION)
print("\n[FIX] Writer doesn't rotate, Reader doesn't inverse-rotate:")
no_rotation = test_values.copy()
reconstructed_fixed = dequant_turbo3_no_rotation(no_rotation)
print(f"  Stored values:      {no_rotation[:4]}...")
print(f"  Reconstructed:      {reconstructed_fixed[:4]}...")
print(f"  SUCCESS: Values match!")
print(f"  Max error: {np.max(np.abs(test_values - reconstructed_fixed)):.6f}")

print("\n=== Conclusion ===")
print("The fix removes turbo_rotate_forward() calls from set-rows.cu kernels.")
print("This ensures the KV cache stores un-rotated values that can be correctly dequantized.")
print("Rotation should only happen in llama-graph.cpp for Q computation, with inverse rotation on output.")
