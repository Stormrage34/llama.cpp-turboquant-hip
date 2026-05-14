#!/usr/bin/env python3
"""Parse llama-bench stdout to structured JSON for archival.

Usage:
    python3 scripts/parse_bench.py benchmarks/std_bench/20260514_moe-99/bench.log

Output:
    JSON with pp512/tg128 metrics, config, and variance data.
"""

import json
import re
import sys
from pathlib import Path


def parse_bench(log_path: str) -> dict:
    """Parse llama-bench output log into structured data."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    result = {
        "source": str(log_path),
        "metrics": {},
        "config": {},
        "raw_lines": [],
    }

    # Extract config from command line
    model_match = re.search(r"-m\s+(\S+)", text)
    ngl_match = re.search(r"-ngl\s+(\d+)", text)
    ctx_match = re.search(r"-c\s+(\d+)", text)
    prompt_match = re.search(r"-p\s+(\d+)", text)
    gen_match = re.search(r"-n\s+(\d+)", text)
    batch_match = re.search(r"-b\s+(\d+)", text)
    ubatch_match = re.search(r"-ub\s+(\d+)", text)
    ctk_match = re.search(r"-ctk\s+(\S+)", text)
    ctv_match = re.search(r"-ctv\s+(\S+)", text)
    runs_match = re.search(r"-r\s+(\d+)", text)

    result["config"] = {
        "model": model_match.group(1) if model_match else None,
        "ngl": int(ngl_match.group(1)) if ngl_match else None,
        "context": int(ctx_match.group(1)) if ctx_match else None,
        "prompt": int(prompt_match.group(1)) if prompt_match else None,
        "gen_len": int(gen_match.group(1)) if gen_match else None,
        "batch": int(batch_match.group(1)) if batch_match else None,
        "ubatch": int(ubatch_match.group(1)) if ubatch_match else None,
        "ctk": ctk_match.group(1) if ctk_match else None,
        "ctv": ctv_match.group(1) if ctv_match else None,
        "runs": int(runs_match.group(1)) if runs_match else None,
    }

    # Extract RDNA2 env vars
    for env_var in ["RDNA2_OPT_V1", "RDNA2_ASYNC_PIPELINE", "RDNA2_MATMUL_OPT_V1", "RDNA2_BFE_DISPATCHER"]:
        env_match = re.search(rf"{env_var}=(\d+)", text)
        if env_match:
            result["config"][env_var] = int(env_match.group(1))

    # Extract benchmark results
    # llama-bench outputs lines like: model_name  pp512  1234.56  ±  5.67  t/s
    pp512_values = []
    tg128_values = []

    for line in text.splitlines():
        # Match pp512 lines
        pp_match = re.search(r"pp\s*(\d+)\s+([\d.]+)\s*(?:±\s*([\d.]+))?\s*t/s", line)
        if pp_match:
            ctx_len = int(pp_match.group(1))
            median = float(pp_match.group(2))
            std = float(pp_match.group(3)) if pp_match.group(3) else None
            pp512_values.append({"context": ctx_len, "median": median, "std": std})

        # Match tg128 lines
        tg_match = re.search(r"tg\s*(\d+)\s+([\d.]+)\s*(?:±\s*([\d.]+))?\s*t/s", line)
        if tg_match:
            gen_len = int(tg_match.group(1))
            median = float(tg_match.group(2))
            std = float(tg_match.group(3)) if tg_match.group(3) else None
            tg128_values.append({"gen_len": gen_len, "median": median, "std": std})

    result["metrics"]["pp512"] = pp512_values if pp512_values else None
    result["metrics"]["tg128"] = tg128_values if tg128_values else None

    # Extract raw benchmark output lines
    result["raw_lines"] = [
        line for line in text.splitlines()
        if line.strip().startswith("llama") or "t/s" in line
    ]

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_bench.py <bench.log> [output.json]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = parse_bench(log_path)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"Written to {output_path}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()