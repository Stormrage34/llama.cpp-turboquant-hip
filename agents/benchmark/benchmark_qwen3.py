#!/usr/bin/env python3
"""
llama-server benchmark script for Qwen3-35B MT PIQ4
Sends prompts via API, captures timings, saves to logs.md
"""

import subprocess
import time
import json
import requests
import os
import signal
import sys
from datetime import datetime

# Configuration
SERVER_URL = "http://localhost:8080"
MODEL_PATH = "/home/stormrage/models/Qwen3_35BMTPIQ4.gguf"
SERVER_BIN = "/home/stormrage/llama.cpp-turboquant-hip/build/bin/llama-server"
LOGS_DIR = "/home/stormrage/llama.cpp-turboquant-hip/agents/benchmark"
LOGS_FILE = os.path.join(LOGS_DIR, "logs.md")

# Server launch parameters
SERVER_ARGS = [
    SERVER_BIN,
    "-m", MODEL_PATH,
    "-ngl", "99",
    "-ncmoe", "39",
    "-c", "128000",
    "-b", "1024",
    "-ub", "6400",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "-fa", "on",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "20",
    "--min-p", "0.05",
    "--threads", "8",
    "--threads-batch", "12",
    "--cpu-range", "0-7",
    "--cpu-strict", "1",
    "--cpu-range-batch", "0-11",
    "--cpu-strict-batch", "1",
    "--numa", "isolate",
    "--prio", "2",
    "--no-mmap",
    "--mlock",
    "--parallel", "1",
    "--jinja",
    "--cache-reuse", "256",
    "--ctx-checkpoints", "8",
    "--metrics",
    "-fitt", "256",
    "--reasoning", "auto",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "2",
    "--spec-draft-p-min", "0.75",
    "--kv-unified",
    "--no-context-shift",
]

# Test prompts
PROMPTS = [
    {
        "name": "Algorithmic Logic & Pathfinding",
        "file": "/home/stormrage/llama.cpp-turboquant-hip/creativeMTP.txt",
        "n_predict": 512
    },
    {
        "name": "Cross-Disciplinary Knowledge",
        "file": "/home/stormrage/llama.cpp-turboquant-hip/researchmtp.txt",
        "n_predict": 512
    },
    {
        "name": "Needle-in-Haystack",
        "file": "/home/stormrage/llama.cpp-turboquant-hip/problemsolvingmtp.txt",
        "n_predict": 512
    },
    {
        "name": "Code Generation",
        "file": "/home/stormrage/llama.cpp-turboquant-hip/codingmtp.txt",
        "n_predict": 512
    }
]


def start_server():
    """Start llama-server with specified parameters."""
    print(f"[*] Starting llama-server...")
    proc = subprocess.Popen(
        SERVER_ARGS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server to be ready
    for i in range(60):
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=2)
            if resp.status_code == 200:
                print(f"[+] Server ready after {i+1}s")
                return proc
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    
    print("[!] Server failed to start in 60s")
    proc.terminate()
    return None


def stop_server(proc):
    """Stop llama-server gracefully."""
    if proc:
        print("[*] Stopping llama-server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[+] Server stopped")


def run_benchmark(prompt_config):
    """Run single benchmark and return timings."""
    name = prompt_config["name"]
    n_predict = prompt_config["n_predict"]
    
    # Load prompt
    with open(prompt_config["file"], "r") as f:
        prompt = f.read()
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Send request
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "cache_prompt": True,
        "stream": False
    }
    
    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/completions",
            json=payload,
            timeout=300
        )
        data = resp.json()
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        
        result = {
            "name": name,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_per_token_ms": timings.get("prompt_per_token_ms", 0),
            "prompt_per_second": timings.get("prompt_per_second", 0),
            "predicted_per_token_ms": timings.get("predicted_per_token_ms", 0),
            "predicted_per_second": timings.get("predicted_per_second", 0),
            "predicted_ms": timings.get("predicted_ms", 0),
            "total_ms": timings.get("total_ms", 0)
        }
        
        # Print results
        print(f"  Prompt tokens:     {result['prompt_tokens']}")
        print(f"  Generated tokens:  {result['completion_tokens']}")
        print(f"  Prompt:            {result['prompt_per_token_ms']:.2f} ms/tok → {result['prompt_per_second']:.2f} tok/s")
        print(f"  Generate:          {result['predicted_per_token_ms']:.2f} ms/tok → {result['predicted_per_second']:.2f} tok/s")
        print(f"  Total time:        {result['total_ms']/1000:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"[!] Error: {e}")
        return None


def save_results(all_results, timestamp):
    """Save benchmark results to logs.md."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    with open(LOGS_FILE, "a") as f:
        f.write(f"\n## Benchmark: {timestamp}\n\n")
        f.write(f"Model: `{MODEL_PATH}`\n\n")
        f.write(f"Server args: `{' '.join(SERVER_ARGS)}`\n\n")
        
        f.write("| Test | Prompt Toks | Gen Toks | Prompt t/s | Generate t/s | Total Time |\n")
        f.write("|------|-------------|----------|------------|--------------|------------|\n")
        
        for r in all_results:
            if r:
                f.write(
                    f"| {r['name']} "
                    f"| {r['prompt_tokens']} "
                    f"| {r['completion_tokens']} "
                    f"| {r['prompt_per_second']:.2f} "
                    f"| {r['predicted_per_second']:.2f} "
                    f"| {r['total_ms']/1000:.1f}s |\n"
                )
        
        f.write("\n### Detailed Timings\n\n")
        for r in all_results:
            if r:
                f.write(f"**{r['name']}**\n\n")
                f.write(f"- Prompt: {r['prompt_per_token_ms']:.2f} ms/tok\n")
                f.write(f"- Generate: {r['predicted_per_token_ms']:.2f} ms/tok\n")
                f.write(f"- Total: {r['total_ms']/1000:.1f}s\n\n")
    
    print(f"\n[+] Results saved to {LOGS_FILE}")


def main():
    print("="*60)
    print("llama-server Benchmark (Qwen3-35B MT PIQ4)")
    print("="*60)
    
    # Check if server already running
    try:
        requests.get(f"{SERVER_URL}/health", timeout=2)
        print("[!] Server already running on port 8080")
        should_start = input("Stop existing server and start new one? (y/n): ").strip().lower()
        if should_start == "y":
            subprocess.run(["pkill", "-f", "llama-server"])
            time.sleep(3)
        else:
            print("[*] Using existing server")
    except requests.exceptions.ConnectionError:
        pass
    
    # Start server
    server_proc = start_server()
    if not server_proc:
        print("[!] Failed to start server")
        sys.exit(1)
    
    try:
        # Run benchmarks
        all_results = []
        for prompt_config in PROMPTS:
            result = run_benchmark(prompt_config)
            all_results.append(result)
            time.sleep(2)  # Brief pause between runs
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(all_results, timestamp)
        
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
