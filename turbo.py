import os
import re

def fix_ggml_h(filepath):
    """Fixes the GGML Type enum to include TQ and MTP types without overlap."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to find the conflict block in ggml_type
    # This replaces the messy conflict with a clean, combined list
    pattern = r"<<<<<<< HEAD.*?GGML_TYPE_COUNT\s+=\s+\d+,?\n=======\n.*?GGML_TYPE_COUNT\s+=\s+\d+,?\n>>>>>>>.*?"
    
    replacement = """        GGML_TYPE_TURBO3_0 = 41,
        GGML_TYPE_TURBO4_0 = 42,
        GGML_TYPE_TURBO2_0 = 43,
        GGML_TYPE_TQ3_1S   = 44,
        GGML_TYPE_TQ4_1S   = 45,
        GGML_TYPE_Q1_0     = 46,
        GGML_TYPE_COUNT    = 47,"""

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"✓ Fixed Enums in {filepath}")

def clean_conflict_markers(filepath):
    """Automatically keeps BOTH blocks of code and removes the Git markers."""
    if not os.path.exists(filepath):
        return
    
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # We delete the markers but keep everything in between
        if not any(marker in line for marker in ["<<<<<<<", "=======", ">>>>>>>"]):
            new_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"✓ Stripped markers from {filepath}")

# --- Execution ---
files_to_strip = [
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "include/llama.h",
    "src/llama-graph.cpp",
    "src/llama-mmap.cpp",
    "ggml/src/ggml.c"
]

print("Starting automatic conflict resolution...")

# 1. Specialized fix for the Header file
fix_ggml_h("ggml/include/ggml.h")

# 2. General strip for the logic files (keeping both sides)
for file in files_to_strip:
    clean_conflict_markers(file)

print("\nAll markers removed. Please verify your files before compiling.")