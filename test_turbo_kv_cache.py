#!/usr/bin/env python3
"""
test_turbo_kv_cache.py -- Integration test for turbo KV cache types

Tests llama-server with various --cache-type-k and --cache-type-v combinations.
Verifies that the server starts, processes a request, and produces output
without crashing (SIGSEGV, GPU page fault, etc.).

Usage:
    ./test_turbo_kv_cache.py [--model /path/to/model.gguf] [--build-dir build]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────
BASE_ARGS = [
    "-ngl", "99",
    "-c", "4096",
    "-b", "1024",
    "-ub", "1024",
    "-fa", "on",
    "--temp", "0.0",
    "--top-p", "1.0",
    "-t", "4",
    "-tb", "4",
    "-ncmoe", "18",
    "--no-mmap",
    "--numa", "isolate",
    "--jinja",
]

# Test matrix: (test_name, cache_type_k, cache_type_v, expect_crash)
TEST_CASES = [
    # Symmetric (same type K/V) — should always work
    ("sym-f16-f16",     "f16",     "f16",     False),
    ("sym-q8_0-q8_0",   "q8_0",    "q8_0",    False),
    ("sym-turbo3",      "turbo3_0","turbo3_0",False),
    ("sym-turbo4",      "turbo4_0","turbo4_0",False),
    ("sym-planar3",     "planar3_0","planar3_0",False),
    ("sym-iso3",        "iso3_0",  "iso3_0",  False),
    ("sym-rq_mse",      "rq_mse",  "rq_mse",  False),
    ("sym-rq_prod",     "rq_prod", "rq_prod", False),

    # Asymmetric (different K/V types) — may fall back to CPU FA
    ("asym-q8_0-turbo3","q8_0",    "turbo3_0",False),
    ("asym-q8_0-turbo4","q8_0",    "turbo4_0",False),
    ("asym-q8_0-planar3","q8_0",   "planar3_0",False),
    ("asym-q8_0-iso3",  "q8_0",    "iso3_0",  False),
    ("asym-q8_0-rq_mse","q8_0",    "rq_mse",  False),
    ("asym-q8_0-rq_prod","q8_0",   "rq_prod", False),
    ("asym-turbo3-q8_0","turbo3_0","q8_0",    False),
]


def check_server_ready(url, timeout=30):
    """Poll server health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    return False


def send_completion(url, prompt="The capital of France is"):
    """Send a completion request and return response text."""
    data = json.dumps({
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result.get("content", "")


def run_test(args, test_name, ctk, ctv, expect_crash):
    """Run a single test case and return (passed, output)."""
    server_args = [args.server_exe] + BASE_ARGS + [
        "-m", args.model,
        "-ctk", ctk,
        "-ctv", ctv,
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  ctk={ctk}  ctv={ctv}")
    print(f"  Command: {' '.join(server_args)}")
    print(f"{'='*60}")

    proc = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    url = f"http://127.0.0.1:{args.port}"
    ready = check_server_ready(url, timeout=60)

    if not ready:
        # Check if process crashed
        ret = proc.poll()
        if ret is not None:
            stderr = proc.stderr.read()
            print(f"  FAIL: Server exited with code {ret}")
            print(f"  STDERR (last 20 lines):")
            for line in stderr.strip().splitlines()[-20:]:
                print(f"    {line}")
            proc.stdout.close()
            proc.stderr.close()
            if expect_crash:
                print(f"  => EXPECTED crash (test passed)")
                return True, stderr
            print(f"  => UNEXPECTED crash (test FAILED)")
            return False, stderr
        print(f"  FAIL: Server not ready within timeout")
        proc.terminate()
        proc.wait()
        return False, ""

    print(f"  Server ready on {url}")

    try:
        output = send_completion(url)
        print(f"  Output: {output[:80]}...")

        # Shut down server gracefully
        try:
            urllib.request.urlopen(f"{url}/shutdown", timeout=3)
        except Exception:
            pass

    except Exception as e:
        print(f"  FAIL: Request error: {e}")
        output = ""

    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    proc.stdout.close()
    proc.stderr.close()

    if output and len(output) > 10:
        print(f"  PASS: got valid output ({len(output)} chars)")
        return True, output
    else:
        print(f"  FAIL: empty or too short output ({len(output)} chars)")
        return False, output


def main():
    parser = argparse.ArgumentParser(description="Turbo KV cache integration tester")
    parser.add_argument("--model", default=os.environ.get("TEST_MODEL"),
                        help="Path to GGUF model (or set TEST_MODEL env)")
    parser.add_argument("--build-dir", default="build",
                        help="Build directory (default: build)")
    parser.add_argument("--port", type=int, default=8888,
                        help="Server port (default: 8888)")
    parser.add_argument("--test", nargs="*",
                        help="Specific test names to run (default: all)")
    args = parser.parse_args()

    if not args.model:
        print("ERROR: No model specified. Use --model or set TEST_MODEL env var.")
        sys.exit(1)

    args.server_exe = os.path.join(args.build_dir, "bin", "llama-server")
    if not os.path.exists(args.server_exe):
        args.server_exe = os.path.join(args.build_dir, "bin", "llama-server")
    if not os.path.exists(args.server_exe):
        print(f"ERROR: llama-server not found in {args.build_dir}/bin/")
        sys.exit(1)

    results = {"pass": 0, "fail": 0, "skip": 0}
    failures = []

    for name, ctk, ctv, expect_crash in TEST_CASES:
        if args.test and name not in args.test:
            results["skip"] += 1
            continue

        ok, output = run_test(args, name, ctk, ctv, expect_crash)

        if ok:
            results["pass"] += 1
        else:
            results["fail"] += 1
            failures.append((name, output))

        # Small delay between tests
        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped")
    if failures:
        print(f"\nFAILURES:")
        for name, out in failures:
            print(f"  {name}")
    print(f"{'='*60}")

    return 1 if results["fail"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
