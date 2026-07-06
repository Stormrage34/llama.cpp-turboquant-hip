#!/usr/bin/env python3
"""
debug_turbo.py - KV cache quantization garble diagnostic

Uses llama-server API with verbose per-token output.
Matches exact server config from the running instance.
Tests each -ctk/-ctv combo by restarting the server.

Key flags: -ncmoe 18 (MoE CPU offload), --no-mmap --mlock
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ── Config ─────────────────────────────────────────────────────────
SERVER_BIN = "./build/bin/llama-server"
MODEL = "/home/stormrage/models/deepreinforce-ai_Ornith-1.0-35B-IQ4_XS.gguf"
PORT = 8099
RESULTS_DIR = Path("/tmp/debug_turbo_results")

# Exact flags from user's working server, minus -ctk/-ctv
BASE_ARGS = [
    "-ngl", "99",
    "-c", "166000",
    "-b", "4096", "-ub", "4096",
    "-fa", "on",
    "-n", "16384",
    "-ctxcp", "64",
    "--no-mmap", "--mlock",
    "-t", "8", "-tb", "12",
    "-ncmoe", "18",
    "--cpu-range", "0-7", "--cpu-strict", "1",
    "--numa", "isolate",
    "-np", "1", "--parallel", "1",
    "--temp", "0.0",
]

# KV cache combos to test
COMBOS = [
    ("q8_q8",         "q8_0",     "q8_0"),
    ("q8_turbo3",     "q8_0",     "turbo3_0"),
    ("turbo3_turbo3", "turbo3_0", "turbo3_0"),
]

# 200-word prompt (~260 tokens) for stress-testing KV cache
PROMPT = """You are a helpful AI assistant with expertise in computer science, mathematics, physics, and engineering. You have been asked to provide a comprehensive explanation of how neural networks work, from the basic building blocks to advanced architectures.

A neural network is a computational model inspired by the biological neural networks in the human brain. It consists of layers of interconnected nodes (neurons) that process information using connectionist approaches to computation. Each connection between neurons has a weight that is adjusted during training.

The fundamental unit is the perceptron, which computes a weighted sum of its inputs, adds a bias term, and passes the result through an activation function. The most common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

When multiple layers of perceptrons are stacked, we get a deep neural network. The first layer processes raw input, hidden layers extract increasingly abstract features, and the output layer produces the final prediction.

Training uses backpropagation: compute the loss (difference between prediction and target), then propagate the error backward through the network, computing gradients with respect to each weight using the chain rule of calculus. An optimizer like Adam or SGD then updates the weights to minimize the loss.

Common architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) and Transformers for sequential data like text. Transformers use self-attention mechanisms to capture relationships between all positions in the input simultaneously.

Question: Given this context, explain how the attention mechanism in transformers works, including the roles of queries, keys, and values. Be specific about the mathematical operations involved.

Answer:"""

TEST_PROMPTS = [
    ("short", "Say hello in one sentence."),
    ("medium", PROMPT),
]

MAX_TOKENS = 200


# ── Helpers ────────────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_servers():
    """Kill all llama-server/llama-cli processes."""
    for pat in ["llama-server", "llama-cli"]:
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", pat], text=True, stderr=subprocess.DEVNULL
            )
            for pid in out.strip().split("\n"):
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        except subprocess.CalledProcessError:
            pass
    time.sleep(5)
    # Verify GPU is free
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.strip().split("\n"):
            if "Used" in line or "Total" in line:
                log(f"  GPU: {line.strip()}")
    except Exception:
        pass


def wait_server(port, max_wait=180):
    """Wait for server to be ready."""
    for i in range(max_wait):
        try:
            with urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def api_chat(prompt, max_tokens=300, port=PORT):
    """Send chat completion request, return raw response."""
    payload = json.dumps({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        return {"error": f"HTTP {e.code}: {body[:500]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Garble detection ───────────────────────────────────────────────
def detect_garble(text, label="content"):
    """Returns list of issue strings. Empty = OK."""
    issues = []
    if not text:
        issues.append(f"{label}: EMPTY")
        return issues

    if "\ufffd" in text:
        issues.append(f"{label}: REPLACEMENT_CHARS")

    ctrl = re.findall(r"[\x00-\x08\x0e-\x1f]", text)
    if ctrl:
        issues.append(f"{label}: CONTROL_CHARS ({len(ctrl)})")

    for patlen in (2, 3, 4, 5):
        for i in range(len(text) - patlen * 5):
            pat = text[i:i + patlen]
            if pat * 5 in text:
                issues.append(f"{label}: REPETITION '{pat}' x5+ at pos {i}")
                break

    nl = text.count("\n")
    if nl / max(len(text), 1) > 0.15 and len(text) > 100:
        issues.append(f"{label}: NEWLINE_FLOOD ({nl}/{len(text)})")

    alphanum = len(re.findall(r"[a-zA-Z0-9]", text))
    special = len(text) - alphanum
    if alphanum > 10 and special / alphanum > 5.0:
        issues.append(f"{label}: HIGH_SPECIAL_RATIO ({special}/{alphanum})")

    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))
    if cjk > 0 and alphanum > 10:
        issues.append(f"{label}: UNEXPECTED_CJK ({cjk})")

    if "\x00" in text:
        issues.append(f"{label}: NULL_BYTES")

    # Check for stuck token patterns
    words = text.split()
    if len(words) > 20:
        for w in (1, 2, 3):
            if len(words) > w * 10:
                seq = " ".join(words[-w:])
                count = text.count(seq)
                if count > 10:
                    issues.append(f"{label}: STUCK_TOKEN '{seq[:40]}' x{count}")
                    break

    return issues


def analyze_response(resp, run_label):
    """Analyze API response. Returns (issues, stats)."""
    issues = []
    stats = {}

    if "error" in resp:
        return [{"fatal": resp["error"]}], stats

    choices = resp.get("choices", [])
    if not choices:
        return [{"fatal": "NO_CHOICES"}], stats

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    finish = choices[0].get("finish_reason", "unknown")
    usage = resp.get("usage", {})

    stats = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "finish_reason": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "content": content,
        "reasoning_preview": reasoning[:200] if reasoning else "",
    }

    issues.extend(detect_garble(content, "content"))
    issues.extend(detect_garble(reasoning, "reasoning"))

    if not content.strip() and finish == "stop":
        issues.append("content: EMPTY_BUT_STOPPED")

    return issues, stats


# ── Server lifecycle ───────────────────────────────────────────────
def start_server(ctk, ctv, port=PORT):
    """Start llama-server with given cache types."""
    cmd = [SERVER_BIN, "-m", MODEL] + BASE_ARGS + [
        "-ctk", ctk, "-ctv", ctv,
        "--port", str(port),
    ]
    log(f"  Starting: ...{' '.join(cmd[-12:])}")
    log(f"  Full cmd: {' '.join(cmd)}")
    log_file = RESULTS_DIR / f"{ctk}_{ctv}_server.log"
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc, log_file


def dump_server_errors(log_file, n=30):
    """Print last N lines with errors/warnings from server log."""
    try:
        lines = Path(log_file).read_text(errors="replace").splitlines()
        errors = [l for l in lines[-n:]
                  if any(k in l.lower() for k in ("error", "warn", "fatal", "assert", "garbl"))]
        if errors:
            log(f"  Server log errors:")
            for e in errors[-10:]:
                log(f"    {e.strip()}")
        return lines
    except FileNotFoundError:
        return []


# ── Main ───────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("KV Cache Quantization Garble Debug (llama-server + -ncmoe 18)")
    log(f"Model: {MODEL}")
    log(f"Results: {RESULTS_DIR}")
    log("=" * 70)

    kill_servers()

    summary = []

    for label, ctk, ctv in COMBOS:
        log(f"\n{'─' * 70}")
        log(f"TEST: -ctk {ctk} -ctv {ctv} ({label})")
        log(f"{'─' * 70}")

        kill_servers()
        time.sleep(2)

        proc, log_file = start_server(ctk, ctv)

        if not wait_server(PORT):
            log(f"  FAILED: Server did not start in 180s")
            dump_server_errors(log_file)
            summary.append((label, "FAIL_START", []))
            proc.kill()
            proc.wait()
            continue

        log(f"  Server ready. Running tests...")

        combo_issues = []
        results = []

        for pname, prompt in TEST_PROMPTS:
            # Fresh run
            log(f"  [{pname}] fresh...")
            resp = api_chat(prompt, max_tokens=MAX_TOKENS)
            issues, stats = analyze_response(resp, f"{label}/{pname}/fresh")
            combo_issues.extend(issues)
            results.append(("fresh", pname, stats, issues))

            preview = stats.get("content", "")[:100]
            if issues:
                for iss in issues:
                    log(f"    !! {iss}")
            else:
                log(f"    OK ({stats['content_len']} chars, {stats['completion_tokens']} tok, "
                    f"cached={stats['cached_tokens']}) | {preview}...")

            # Cached run (same prompt should hit KV cache)
            log(f"  [{pname}] cached...")
            resp2 = api_chat(prompt, max_tokens=MAX_TOKENS)
            issues2, stats2 = analyze_response(resp2, f"{label}/{pname}/cached")
            combo_issues.extend(issues2)
            results.append(("cached", pname, stats2, issues2))

            if issues2:
                for iss in issues2:
                    log(f"    !! {iss}")
            else:
                log(f"    OK ({stats2['content_len']} chars, cached={stats2['cached_tokens']})")

            # Compare fresh vs cached content
            c1 = stats.get("content", "")
            c2 = stats2.get("content", "")
            if c1 and c2 and c1 == c2:
                log(f"    MATCH: deterministic output")
            elif c1 and c2:
                for i, (a, b) in enumerate(zip(c1, c2)):
                    if a != b:
                        log(f"    DIFF at char {i}")
                        break

            # Long generation
            log(f"  [{pname}] long (500 tokens)...")
            resp3 = api_chat(prompt, max_tokens=MAX_TOKENS)
            issues3, stats3 = analyze_response(resp3, f"{label}/{pname}/long")
            combo_issues.extend(issues3)
            results.append(("long", pname, stats3, issues3))

            if issues3:
                for iss in issues3:
                    log(f"    !! {iss}")
            else:
                log(f"    OK ({stats3['content_len']} chars, {stats3['completion_tokens']} tok)")

        # Save all outputs
        for run_type, pname, stats, issues in results:
            outfile = RESULTS_DIR / f"{label}_{pname}_{run_type}.txt"
            with open(outfile, "w") as f:
                f.write(f"# {label} {pname} {run_type}\n")
                f.write(f"# issues: {issues}\n")
                f.write(f"# stats: {json.dumps({k: v for k, v in stats.items() if k not in ('content',)}, indent=2)}\n\n")
                f.write(stats.get("content", "(empty)"))

        # Server log
        dump_server_errors(log_file)

        fatal = [i for i in combo_issues if isinstance(i, dict)]
        warnings = [i for i in combo_issues if isinstance(i, str)]
        verdict = "FAIL" if fatal else ("WARN" if warnings else "PASS")

        log(f"\n  >>> {verdict}: {len(warnings)} warnings, {len(fatal)} fatal")
        summary.append((label, verdict, warnings))

        proc.terminate()
        proc.wait(timeout=10)
        time.sleep(3)

    # ── Final summary ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")
    log(f"{'Combo':<20} {'Result':<8} {'Issues'}")
    log(f"{'─'*20} {'─'*8} {'─'*40}")
    for label, verdict, warnings in summary:
        iss = "; ".join(warnings[:3]) if warnings else "-"
        log(f"{label:<20} {verdict:<8} {iss}")
    log(f"\nFull outputs: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
