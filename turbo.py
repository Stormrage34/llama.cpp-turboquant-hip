import os

def surgical_fix():
    file_path = 'src/llama-graph.cpp'

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Are you in the root of the repo?")
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # The compiler says line 2547 is where the 'inner' function starts.
    # We need to close the previous block before that line.
    # Python indices are 0-based, so line 2547 is index 2546.
    target_index = 2546

    # Check if we already fixed it to avoid double-bracing
    if "}" not in lines[target_index - 1]:
        print(f"Inserting brace before line 2547...")
        # We insert a newline and a closing brace
        lines.insert(target_index, "\n}\n")

        # Now, let's clean any lingering conflict markers which usually cause this
        cleaned_lines = []
        for line in lines:
            if not any(m in line for m in ["<<<<<<<", "=======", ">>>>>>>"]):
                cleaned_lines.append(line)

        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        print("✓ Repair finished. Conflict markers (if any) removed.")
    else:
        print("! A brace already exists near that line. Let's try a different approach.")

if __name__ == "__main__":
    surgical_fix()
