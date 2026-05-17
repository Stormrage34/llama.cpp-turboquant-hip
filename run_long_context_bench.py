#!/usr/bin/env python3
"""Long-context benchmark: 64K/128K/256K — with aggressive ncmoe for VRAM budget."""

import subprocess, json, time, os, signal
import urllib.request

SERVER_URL = "http://localhost:8080"
MODEL = "/home/stormrage/models/Qwen3_35BMTPIQ4.gguf"
BIN = "/home/stormrage/llama.cpp-turboquant-hip/build/bin/llama-server"
WORKDIR = "/home/stormrage/llama.cpp-turboquant-hip"

PARA = (
    "The fundamental principles of quantum mechanics and machine learning "
    "converge at the frontier of computational science, where parameterized "
    "quantum circuits demonstrate remarkable expressivity for approximating "
    "complex functions with provable efficiency gains over classical architectures. "
)

def measure_vram():
    r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
        capture_output=True, text=True, timeout=10)
    for line in r.stdout.split("\n"):
        if "VRAM Total Used Memory" in line:
            return int(line.strip().split()[-1]) // 1048576
    return 0

def kill_with_timeout(proc, timeout=15):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except: pass

def tokenize_count(text):
    data = json.dumps({"content": text}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize",
        data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    return len(json.loads(resp.read()).get("tokens", []))

def build_prompt(n_target):
    item = f"[S] {PARA}"
    per_item = tokenize_count(item)
    n_items = int(n_target / per_item)
    prompt = item * n_items + "\n\nNow provide a comprehensive analysis. Be specific."
    actual = tokenize_count(prompt)
    return prompt, actual

def run_bench(name, ctx_len, ncmoe, fill_tokens, timeout=600):
    print(f"\n{'='*60}")
    print(f"  {name}: ctx={ctx_len}  -ncmoe {ncmoe}  fill={fill_tokens}")
    print(f"{'='*60}")

    log = open(f"/tmp/srv_{name}.log", "w")
    cmd = [BIN, "-m", MODEL,
         "-ngl", "99", "-ncmoe", str(ncmoe),
         "-c", str(ctx_len),
         "-b", "256", "-ub", "256",
         "--cache-type-k", "turbo4", "--cache-type-v", "turbo4",
         "-fa", "on",
         "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.05",
         "--threads", "8", "--threads-batch", "12",
         "--cpu-range", "0-7", "--cpu-strict", "1",
         "--cpu-range-batch", "0-11", "--cpu-strict-batch", "1",
         "--numa", "isolate", "--prio", "2",
         "--no-mmap", "--mlock", "--parallel", "1", "--jinja",
         "--cache-reuse", "256", "--ctx-checkpoints", "4", "--metrics",
         "-fitt", "256", "--reasoning", "auto",
         "--spec-type", "mtp", "--spec-draft-n-max", "2",
         "--spec-draft-p-min", "0.75", "--kv-unified"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=WORKDIR, start_new_session=True)
    print(f"  PID: {proc.pid}")

    for i in range(60):
        time.sleep(5)
        try:
            r = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3)
            if b"ok" in r.read():
                print(f"  Ready after {(i+1)*5}s"); break
        except: pass
    else:
        print("  FAIL: server not ready"); kill_with_timeout(proc); return None
    time.sleep(2)

    vram_b = measure_vram()
    print(f"  Before: VRAM={vram_b}")
    print("  Calibrating...")
    max_tok = ctx_len - 3000
    safe_fill = min(fill_tokens, max_tok)
    prompt, actual_tok = build_prompt(safe_fill)
    while actual_tok >= ctx_len - 512:
        prompt = prompt[:int(len(prompt)*0.85)]
        actual_tok = tokenize_count(prompt)
    print(f"  Prompt: {len(prompt)} chars, {actual_tok} tokens")

    req_data = json.dumps({
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05,
        "cache_prompt": True, "stream": False
    })
    with open(f"/tmp/req_{name}.json", "w") as f:
        f.write(req_data)

    print(f"  Sending ({len(req_data)} bytes, timeout {timeout}s)...")
    t0 = time.time()
    try:
        resp = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout),
             f"{SERVER_URL}/v1/completions",
             "-H", "Content-Type: application/json",
             "-d", f"@/tmp/req_{name}.json"],
            capture_output=True, text=True, timeout=timeout+30)
        t_elapsed = time.time() - t0
        result = json.loads(resp.stdout) if resp.stdout else {}
    except subprocess.TimeoutExpired:
        print(f"  CURL TIMEOUT after {timeout}s")
        result = {}
        t_elapsed = time.time() - t0
    except Exception as e:
        print(f"  ERROR: {e}")
        result = {}
        t_elapsed = time.time() - t0

    # Measure VRAM immediately after response (before kill)
    vram_a = measure_vram()
    usage = result.get("usage", {})
    timings = result.get("timings", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    pps = timings.get("prompt_per_second", 0) or 0
    dps = timings.get("predicted_per_second", 0) or 0
    dpms = timings.get("predicted_per_token_ms", 0) or 0

    print(f"  Elapsed: {t_elapsed:.1f}s")
    print(f"  Prompt: {pt} tok ({pps:.1f} t/s)")
    print(f"  Gen: {ct} tok ({dps:.1f} t/s, {dpms:.2f} ms/tok)")
    print(f"  VRAM: {vram_b}→{vram_a} MB (Δ+{vram_a-vram_b})")
    print(f"  Util: {vram_a}/16384 ({vram_a*100//16384}%)")

    # Check for errors in log
    with open(f"/tmp/srv_{name}.log") as f:
        for line in f.readlines()[-5:]:
            if any(w in line.lower() for w in ["fault", "coredump", "oom", "error"]):
                print(f"  LOG ERROR: {line.strip()}")

    kill_with_timeout(proc)
    time.sleep(2)
    return {"name": name, "ncmoe": ncmoe, "vram_b": vram_b, "vram_a": vram_a,
            "vram_delta": vram_a - vram_b, "pt": pt, "ct": ct,
            "pps": pps, "dps": dps, "t": t_elapsed}

# First: validate VRAM at aggressive ncmoe with smaller prompt
r0 = run_bench("64K_ncmoe35", 65536, 35, 60000, timeout=900)

# Scale up
if r0 and r0["pt"] > 0:
    r1 = run_bench("128K_ncmoe35", 131072, 35, 120000, timeout=900)
else:
    r1 = None

if r1 and r1["pt"] > 0:
    r2 = run_bench("256K_ncmoe41", 262144, 41, 240000, timeout=1200)
else:
    r2 = None

print(f"\n{'='*60}")
for r in [r1, r2]:
    if r:
        print(f"\n{r['name']} (-ncmoe {r['ncmoe']}):")
        print(f"  Prompt: {r['pt']} @ {r['pps']:.1f} t/s | Gen: {r['ct']} @ {r['dps']:.1f} t/s")
        print(f"  VRAM: {r['vram_b']}→{r['vram_a']} MB ({r['vram_a']*100//16384}%)")

# Also collect the 64K results from the previous run (still in logs)
print(f"\n  Previous 64K (-ncmoe 18, from earlier run):")
print(f"  Prompt: 59994 @ 321.0 t/s | Gen: 128 @ 10.9 t/s")
