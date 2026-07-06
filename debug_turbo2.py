#!/usr/bin/env python3
"""
Minimal turbo KV debug via llama-cli.
"""
import subprocess, sys, os, time, re

MODEL = "/home/stormrage/models/deepreinforce-ai_Ornith-1.0-35B-IQ4_XS.gguf"
CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build/bin/llama-cli")

BASE_FLAGS = [
    "-ngl", "99", "-c", "4096", "-fa", "on", "-n", "256",
    "--no-mmap", "--mlock", "-t", "8", "-tb", "12",
    "-ncmoe", "18", "--cpu-range", "0-7", "--cpu-strict", "1",
    "--numa", "isolate",
]

COMBOS = [
    ("q8_0",     "q8_0"),
    ("q8_0",     "turbo3_0"),
    ("turbo3_0", "turbo3_0"),
    ("q8_0",     "turbo2_0"),
    ("turbo2_0", "turbo2_0"),
    ("q8_0",     "turbo4_0"),
    ("turbo4_0", "turbo4_0"),
]

PROMPT = "What is 2+2? Answer in one sentence."


def is_garbled(text):
    if not text or not text.strip():
        return "EMPTY"
    s = text.strip()
    issues = []
    # Repetition check
    for pl in range(1, 6):
        for i in range(0, len(s) - pl * 5):
            c = s[i:i+pl]
            if c * 5 in s:
                issues.append(f"REP '{c}'")
                break
        if issues:
            break
    # Emoji flood
    emojis = len(re.findall(r'[\U0001F300-\U0001FAFF]', s))
    if emojis > 3 and emojis > len(s) * 0.05:
        issues.append(f"EMOJI({emojis}/{len(s)})")
    # Newline flood
    nl = s.count('\n')
    if nl > 20 and nl > len(s) * 0.3:
        issues.append(f"NLFLOOD({nl}/{len(s)})")
    return "; ".join(issues) if issues else "OK"


def run_one(kt, vt):
    cmd = [CLI] + BASE_FLAGS + [
        "-m", MODEL,
        "-ctk", kt, "-ctv", vt,
        "-p", PROMPT,
        "--no-display-prompt", "--special", "-e",
    ]
    label = f"{kt}/{vt}"
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=os.path.dirname(CLI))
        elapsed = time.time() - t0
        # Extract generation: lines after the last prompt echo
        out = r.stdout
        # Strip loading banner, keep only after first '> ' line
        lines = out.split('\n')
        gen_lines = []
        found_prompt = False
        for line in lines:
            if '> ' in line and not found_prompt:
                found_prompt = True
                # Text after '> ' on same line is first gen
                idx = line.rfind('> ')
                if idx >= 0:
                    rest = line[idx+2:].strip()
                    if rest:
                        gen_lines.append(rest)
                continue
            if found_prompt:
                gen_lines.append(line)
        gen = '\n'.join(gen_lines).strip()
        # Remove trailing prompt echo
        if gen.endswith('>'):
            gen = gen[:-1].strip()
        
        # Check stderr for key info
        err_lines = r.stderr.split('\n') if r.stderr else []
        tps_line = [l for l in err_lines if 't/s' in l or 'tok/s' in l]
        oom = any('out of memory' in l.lower() for l in err_lines)
        
        garble = is_garbled(gen)
        preview = gen[:150].replace('\n', ' | ') if gen else "(empty)"
        
        tps_info = ""
        if tps_line:
            tps_info = tps_line[-1].strip()[-60:]
        
        if oom:
            return label, "OOM", "(out of memory)", ""
        
        return label, garble, preview, f"{elapsed:.1f}s {tps_info}"
    except subprocess.TimeoutExpired:
        return label, "TIMEOUT", "", ""
    except Exception as e:
        return label, f"ERR: {e}", "", ""


def main():
    print(f"Turbo KV Debug (llama-cli) | {MODEL}")
    print(f"Prompt: {PROMPT}")
    print(f"{'='*90}")
    print(f"{'Combo':<25} {'Status':<10} {'Output (150 char preview)':<55}")
    print(f"{'─'*25} {'─'*10} {'─'*55}")

    for kt, vt in COMBOS:
        label, status, preview, info = run_one(kt, vt)
        status_icon = "OK  " if status == "OK" else "FAIL"
        print(f"{label:<25} {status_icon:<10} {preview[:55]}")
        if status not in ("OK", "EMPTY"):
            print(f"{'':25} {status}")
        if info:
            print(f"{'':25} {info}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
