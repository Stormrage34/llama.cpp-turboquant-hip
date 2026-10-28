#!/usr/bin/env python3
"""
ic_expose.py - expose what the Infinity Cache can and cannot tell us on this GPU.

WHY THIS EXISTS
    The gfx1030 (RX 6900 XT) carries 128 MB of "Infinity Cache" (AMD's name for
    the MALL - Memory Attached Last Level). The question "how do we use it" needs
    a measurement first, and the measurement problem is that AMD ships no
    dedicated Infinity Cache counter on this part. What does exist is the L2
    counter block (GL2C_*) and a derived metric (FETCH_SIZE) that reports bytes
    actually fetched from memory. This script inventories what the profiler can
    see, runs a workload, and turns the counters into cache metrics.

    It follows the tool layering in the ROCm profiling overview:
      ROCprofiler-SDK  -> infrastructure
      rocprofv3        -> CLI, per-kernel counters (used here)
      rocprofiler-compute -> deep per-kernel analysis incl. memory hierarchy
                             (AMD Instinct only; probed, not assumed)
      rocprofiler-systems -> system-wide timeline (not needed for a cache query)

WHAT IT CANNOT DO
    It cannot read Infinity Cache occupancy. There is no MALL counter in the
    rocprofv3 inventory on gfx1030. The script reports that absence as a result
    rather than inventing a number.

USAGE
    # 1. what does the toolchain expose on this machine?
    python3 scripts/ic_expose.py probe

    # 2. profile a command and analyse it in one step
    python3 scripts/ic_expose.py full \
        --out /tmp/opencode/ic-run \
        -- ./build-hip-gfx1030/bin/llama-bench \
           -m model.gguf -p 512 -n 0 -ngl 99 -fa 1 -ctk q8_0 -ctv q5_0 -dev ROCm0

    # 3. or analyse an existing rocprofv3 output directory
    python3 scripts/ic_expose.py analyze /tmp/opencode/ic-run --tokens 512

    # 4. machine-readable
    python3 scripts/ic_expose.py analyze DIR --tokens 512 --json out.json

EXIT CODES
    0  metric produced
    2  counters unavailable / not parseable (reported, not hidden)
    3  rocprofv3 missing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Counter inventory
# ---------------------------------------------------------------------------

# Prefixes that identify each level of the memory hierarchy in AMD's counter
# namespaces. Verification, not guesswork: every entry here was observed in
# `rocprofv3 --list-avail` on gfx1030, except the MALL/TCC/TCA families which
# exist on CDNA parts and are absent on RDNA2.
CACHE_FAMILIES = {
    "MALL / Infinity Cache": ("MALL", "TCC_", "TCA_"),
    "L2 (GL2C)": ("GL2C_", "L2CacheHit", "FETCH_SIZE", "WRITE_SIZE"),
    "L1 vector (TCP)": ("TCP_",),
    "Shader / SQ": ("SQ_", "SPI_", "TA_"),
    "Global memory controller": ("GRBM_", "MC_", "GCEA_"),
}

# The counters this script actually consumes, in preference order. `_sum`
# variants are used for the GL2C family because the raw counters are
# per-instance across the 32 L2 slices; summing them yields a card-wide total.
CACHE_PMCS = [
    "GL2C_HIT_sum",
    "GL2C_MISS_sum",
    "GL2C_MC_RDREQ_sum",
    "GL2C_EA_RDREQ_32B_sum",
    "GL2C_EA_RDREQ_64B_sum",
    "GL2C_MC_WRREQ_sum",
    "FETCH_SIZE",
]

# rocprofv3 writes different CSV names across versions; match on content, not
# on a fixed filename.
_CSV_COUNTER_HINT = re.compile(r"GL2C_|FETCH_SIZE|Counter_Name|Kernel_Name", re.I)


def which(name: str) -> str | None:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def _rocprofv3_list_avail(exe: str, timeout: int = 120) -> str:
    """Run `rocprofv3 --list-avail` and return stdout. Empty string on failure."""
    for argv in ([exe, "--list-avail"],
                 [exe, "avail"],
                 [exe, "--list-avail", "--pmc"]):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            continue
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if "gfx" in out or "Counter" in out or "GL2C" in out:
            return out
    return ""


_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")


def parse_counter_names(text: str) -> set[str]:
    """Extract plausible counter names from the --list-avail text.

    The format differs between rocprofiler-sdk releases (plain names, Name:
    fields, or an indented table), so accept any long upper-case token that
    appears on a line that also looks like a counter entry. False positives are
    harmless: they only inflate an inventory count and are filtered by family
    prefix when classified.
    """
    names: set[str] = set()
    for line in text.splitlines():
        low = line.lower()
        if not ("counter" in low or "name" in low or "expression" in low
                or "gpu" in low or line.strip().startswith("-")):
            # also accept bare indented token lines
            stripped = line.strip()
            if not stripped or " " in stripped:
                continue
        for m in _TOKEN.finditer(line):
            names.add(m.group(1))
    return names


def classify(names: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for family, prefixes in CACHE_FAMILIES.items():
        hits = sorted(n for n in names if n.startswith(prefixes))
        out[family] = hits
    return out


def cmd_probe(args: argparse.Namespace) -> int:
    # An offline inventory (a saved `rocprofv3 --list-avail` dump) can be parsed
    # without touching the GPU, which matters while a benchmark owns the card.
    if getattr(args, "avail_file", None):
        text = Path(args.avail_file).expanduser().read_text(errors="replace")
        names = parse_counter_names(text)
        fam = classify(names)
        print(f"offline inventory       {args.avail_file}")
        print("\nMemory-hierarchy counters exposed")
        print("-" * 62)
        for family in CACHE_FAMILIES:
            hits = fam[family]
            print(f"  {family:<28} "
                  f"{f'{len(hits):3d} available' if hits else 'NONE'}")
        ic = fam["MALL / Infinity Cache"]
        print("\nInfinity Cache verdict")
        print("-" * 62)
        if ic:
            print(f"  MALL counters present: {len(ic)}")
        else:
            print("  MALL counters present: 0")
            print("  -> Infinity Cache occupancy is NOT measurable on this part.")
            print(f"     GL2C (L2) counters available: {len(fam['L2 (GL2C)'])}")
        if args.json:
            Path(args.json).write_text(json.dumps({
                "source": args.avail_file,
                "cache_families": fam,
                "infinity_cache_counters": ic,
            }, indent=2), encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0 if fam["L2 (GL2C)"] or ic else 2

    tools = {
        "rocprofv3": which("rocprofv3"),
        "rocprofiler-compute": which("rocprofiler-compute") or which("rocprof-compute"),
        "rocprofiler-systems": which("rocprofiler-systems") or which("rocprof-sys-run"),
        "rocminfo": which("rocminfo"),
        "amd-smi": which("amd-smi"),
        "llvm-objdump": which("llvm-objdump"),
    }

    print("ROCm profiling toolchain")
    print("-" * 62)
    for name, path in tools.items():
        print(f"  {name:<22} {path or 'NOT INSTALLED'}")

    # rocprofiler-compute is the tool the ROCm docs point at for memory-hierarchy
    # analysis, but it formally targets AMD Instinct. Probe rather than assume.
    gfx = detect_gfx()
    print(f"\n  GPU arch                 {gfx or 'unknown'}")

    if not tools["rocprofv3"]:
        print("\nrocprofv3 is required for per-kernel counters and is missing.")
        print("Counters cannot be collected on this machine.")
        return 3

    text = _rocprofv3_list_avail(tools["rocprofv3"])
    if not text:
        print("\nrocprofv3 --list-avail returned nothing usable; counters unknown.")
        return 2

    names = parse_counter_names(text)
    fam = classify(names)
    print("\nMemory-hierarchy counters exposed by this toolchain")
    print("-" * 62)
    for family in CACHE_FAMILIES:
        hits = fam[family]
        state = f"{len(hits):3d} available" if hits else "NONE"
        print(f"  {family:<28} {state}")

    ic = fam["MALL / Infinity Cache"]
    print("\nInfinity Cache verdict")
    print("-" * 62)
    if ic:
        print(f"  MALL counters present: {len(ic)}")
        print("  Example:", ", ".join(ic[:5]))
        print("  -> Infinity Cache occupancy is directly measurable here.")
    else:
        print("  MALL counters present: 0")
        print("  -> Occupancy of the 128 MB Infinity Cache is NOT measurable.")
        print("     Closest observable proxy: L2 (GL2C) hit rate + FETCH_SIZE.")
        if fam["L2 (GL2C)"]:
            print(f"     GL2C counters available: {len(fam['L2 (GL2C)'])}")
        else:
            print("     No GL2C counters either: cache behaviour is unobservable.")

    if tools["rocprofiler-compute"]:
        print("\nrocprofiler-compute is installed. On non-Instinct parts it may")
        print("refuse the arch; run it directly to confirm before relying on it.")

    if args.json:
        payload = {
            "tools": tools,
            "gfx": gfx,
            "cache_families": {k: v for k, v in fam.items()},
            "infinity_cache_counters": ic,
            "pmcs_to_collect": [p for p in CACHE_PMCS if p.split("_sum")[0] in names
                                or p in names] or CACHE_PMCS,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if fam["L2 (GL2C)"] or ic else 2


def detect_gfx() -> str | None:
    exe = which("rocminfo")
    if not exe:
        return None
    try:
        p = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\bgfx[0-9a-f]{3,4}\b", p.stdout or "")
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    exe = which("rocprofv3")
    if not exe:
        print("rocprofv3 not found", file=sys.stderr)
        return 3

    out = Path(args.out).expanduser().resolve()

    pmcs = args.pmc or CACHE_PMCS
    argv = [exe, "--pmc", *pmcs]
    if args.kernel_filter:
        argv += ["--kernel-include-regex", args.kernel_filter]
    argv += ["-f", "csv", "-d", str(out), "--", *args.command]

    print("+ " + " ".join(argv))
    if getattr(args, "dry_run", False):
        print("dry run: not executing (no GPU touched)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(argv).returncode
    print(f"rocprofv3 exit {rc}; output in {out}")
    return 0 if rc == 0 else 1


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

_TOTAL_KEYS = ("sum", "total", "value", "avg", "mean")


def _pivot_long_rows(rows: list[dict]) -> list[dict]:
    """Pivot rocprofv3's long (one row per counter) CSV into wide rows.

    rocprofv3 emits either wide rows (one column per counter) or long rows
    (Counter_Name / Counter_Value columns). Pivot the latter so the rest of
    the analysis can treat both shapes uniformly.
    """
    if not rows:
        return rows
    keys = [k for k in rows[0].keys() if k]
    counter_name_col = None
    for k in keys:
        lk = (k or "").lower().replace(" ", "").replace("_", "")
        if lk in ("countername", "counter", "metricname", "metric"):
            counter_name_col = k
            break
        if "counter" in (k or "").lower() and "name" in (k or "").lower():
            counter_name_col = k
            break
    if counter_name_col is None:
        return rows

    value_col = None
    for k in keys:
        if "value" in (k or "").lower():
            value_col = k
            break
    if value_col is None:
        value_col = keys[-1]

    kernel_col = None
    for k in keys:
        if "kernel" in (k or "").lower():
            kernel_col = k
            break
    if kernel_col is None:
        kernel_col = keys[0]

    dispatch_col = None
    for k in keys:
        if "dispatch" in (k or "").lower():
            dispatch_col = k
            break

    dur_col = None
    for k in keys:
        lk = (k or "").lower()
        if "duration" in lk or "time" in lk:
            dur_col = k
            break

    groups: dict = {}
    order: list = []
    for row in rows:
        kernel = row.get(kernel_col) or "unknown"
        disp_id = row.get(dispatch_col) if dispatch_col else None
        key = (kernel, disp_id)
        if key not in groups:
            groups[key] = {"kernel": kernel, "kernel_col": kernel_col,
                           "dur": 0.0, "dur_col": dur_col, "has_dur": False,
                           "counters": {}}
            order.append(key)
        g = groups[key]
        cname = row.get(counter_name_col)
        if cname:
            g["counters"][cname] = g["counters"].get(cname, 0.0) + _numeric(row.get(value_col))
        if dur_col and not g["has_dur"]:
            v = row.get(dur_col)
            if v not in (None, ""):
                g["dur"] = _numeric(v)
                g["has_dur"] = True

    wide: list[dict] = []
    for key in order:
        g = groups[key]
        r: dict = {g["kernel_col"]: g["kernel"]}
        if g["dur_col"]:
            r[g["dur_col"]] = g["dur"]
        r.update(g["counters"])
        wide.append(r)
    return wide


def _load_counter_csv(directory: Path) -> tuple[list[dict], Path | None]:
    """Find and read the rocprofv3 CSV that holds counter values."""
    best: tuple[int, list[dict], Path] | None = None
    for path in sorted(directory.rglob("*.csv")):
        try:
            text = path.read_text(errors="replace")[:4096]
        except OSError:
            continue
        if not _CSV_COUNTER_HINT.search(text):
            continue
        try:
            with path.open(newline="", errors="replace") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, csv.Error):
            continue
        if not rows:
            continue
        # Score: prefer files that actually carry counter data. Wide files have
        # GL2C_/FETCH_SIZE columns; long files have a Counter_Name column.
        score = sum(1 for c in (rows[0] or {}) if c and re.search(r"GL2C_|FETCH_SIZE", c))
        keys_low = " ".join((rows[0] or {}).keys()).lower()
        if "counter_name" in keys_low or "counter name" in keys_low:
            score += 10
        if "kernel_name" in keys_low or "kernel name" in keys_low:
            score += 5
        if best is None or score > best[0]:
            best = (score, rows, path)
    if best is None or best[0] == 0:
        return [], None
    rows, path = best[1], best[2]
    rows = _pivot_long_rows(rows)
    return rows, path


def _numeric(value: str | None) -> float:
    if value is None:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        m = re.search(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", value.replace(",", ""))
        return float(m.group(0)) if m else 0.0


def _col(row: dict, needle: str) -> str | None:
    for key in row:
        if key and needle.lower() in key.lower():
            return key
    return None


def summarise(rows: list[dict], tokens: int, fetch_unit: str) -> dict:
    hit_c = _col(rows[0], "GL2C_HIT") or _col(rows[0], "L2CacheHit")
    miss_c = _col(rows[0], "GL2C_MISS")
    fetch_c = _col(rows[0], "FETCH_SIZE")
    dur_c = (_col(rows[0], "Duration") or _col(rows[0], "duration")
             or _col(rows[0], "time"))
    kern_c = _col(rows[0], "Kernel_Name") or _col(rows[0], "Kernel")

    per_kernel: dict[str, dict[str, float]] = defaultdict(
        lambda: {"hit": 0.0, "miss": 0.0, "fetch": 0.0, "dur": 0.0, "n": 0}
    )
    totals = {"hit": 0.0, "miss": 0.0, "fetch": 0.0, "dur": 0.0}

    for row in rows:
        name = (row.get(kern_c) or "unknown") if kern_c else "all"
        # rocprofv3 emits one row per (dispatch, counter) in some versions and
        # one wide row in others; sum defensively, it is monotonic either way.
        h = _numeric(row.get(hit_c)) if hit_c else 0.0
        m = _numeric(row.get(miss_c)) if miss_c else 0.0
        f = _numeric(row.get(fetch_c)) if fetch_c else 0.0
        d = _numeric(row.get(dur_c)) if dur_c else 0.0
        rec = per_kernel[name]
        rec["hit"] += h
        rec["miss"] += m
        rec["fetch"] += f
        rec["dur"] += d
        rec["n"] += 1
        totals["hit"] += h
        totals["miss"] += m
        totals["fetch"] += f
        totals["dur"] += d

    scale = 1024.0 if fetch_unit.lower() in ("kib", "kb", "k") else 1.0
    total_bytes = totals["fetch"] * scale
    lookups = totals["hit"] + totals["miss"]
    return {
        "rows": len(rows),
        "per_kernel": per_kernel,
        "hit": totals["hit"],
        "miss": totals["miss"],
        "hit_rate": (totals["hit"] / lookups) if lookups else None,
        "bytes_fetched": total_bytes,
        "bytes_per_token": (total_bytes / tokens) if tokens and total_bytes else None,
        "fetch_counter": fetch_c,
        "hit_counter": hit_c,
        "miss_counter": miss_c,
        "duration_counter": dur_c,
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    directory = Path(args.dir).expanduser().resolve()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    rows, path = _load_counter_csv(directory)
    if not rows:
        print(f"No counter CSV found under {directory}.")
        print("Either the run produced no counters, or the requested PMCs are")
        print("unsupported on this GPU. Re-run `probe` to see what is available.")
        return 2

    s = summarise(rows, args.tokens, args.fetch_unit)

    print(f"source        {path}")
    print(f"rows          {s['rows']}")
    print(f"counters      hit={s['hit_counter']}  miss={s['miss_counter']}  "
          f"fetch={s['fetch_counter']}")

    print("\nper kernel (top 12 by bytes fetched)")
    print("-" * 78)
    print(f"{'kernel':<44}{'hit rate':>10}{'bytes':>14}{'disp':>8}")
    ranked = sorted(s["per_kernel"].items(), key=lambda kv: kv[1]["fetch"], reverse=True)
    for name, rec in ranked[:12]:
        lookups = rec["hit"] + rec["miss"]
        rate = f"{100.0 * rec['hit'] / lookups:.1f}%" if lookups else "n/a"
        shown = name if len(name) <= 43 else name[:40] + "..."
        print(f"{shown:<44}{rate:>10}{rec['fetch']:>14.4g}{int(rec['n']):>8}")

    print("\ncard-wide")
    print("-" * 78)
    if s["hit_rate"] is not None:
        print(f"  L2 hit rate            {100.0 * s['hit_rate']:.2f}%")
        print(f"  L2 hits / misses       {s['hit']:.4g} / {s['miss']:.4g}")
    else:
        print("  L2 hit rate            n/a (hit/miss counters not present)")
    if s["bytes_fetched"]:
        print(f"  bytes fetched          {s['bytes_fetched']:.6g}")
        if s["bytes_per_token"]:
            print(f"  bytes per token        {s['bytes_per_token']:.6g}  "
                  f"(over {args.tokens} tokens)")
    else:
        print("  bytes fetched          n/a (FETCH_SIZE not present)")

    print("\nInfinity Cache")
    print("-" * 78)
    print("  A hit rate above is the L2 (GL2C) block, not the 128 MB MALL.")
    print("  No MALL counter exists on gfx1030, so IC occupancy stays unmeasurable.")
    print("  Use the L2 hit rate as an upper-bound proxy: if the working set is")
    print("  larger than the IC, low L2 hit rate means no reuse is happening.")

    if args.json:
        payload = {
            "source": str(path),
            "rows": s["rows"],
            "hit_rate": s["hit_rate"],
            "bytes_fetched": s["bytes_fetched"],
            "bytes_per_token": s["bytes_per_token"],
            "per_kernel": {
                k: {
                    "hit": v["hit"], "miss": v["miss"], "fetch": v["fetch"],
                    "dispatches": v["n"],
                    "hit_rate": (v["hit"] / (v["hit"] + v["miss"]))
                    if (v["hit"] + v["miss"]) else None,
                }
                for k, v in s["per_kernel"].items()
            },
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------

def cmd_full(args: argparse.Namespace) -> int:
    if not args.command:
        print("full requires a target command after `--`", file=sys.stderr)
        return 2
    rc = cmd_run(args)
    if rc not in (0, 1):
        return rc
    if getattr(args, "dry_run", False):
        return 0
    print()
    return cmd_analyze(args)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Expose what the Infinity Cache can and cannot tell us on this GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1] if "USAGE" in __doc__ else None,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", help="inventory profiling tools and cache counters")
    pp.add_argument("--json", help="write the inventory to this JSON file")
    pp.add_argument("--avail-file",
                    help="parse a saved `rocprofv3 --list-avail` dump instead of "
                         "running it (offline, does not touch the GPU)")
    pp.set_defaults(func=cmd_probe)

    pr = sub.add_parser("run", help="run a command under rocprofv3 with cache PMCs")
    pr.add_argument("--out", required=True, help="rocprofv3 output directory")
    pr.add_argument("--pmc", nargs="+", help="counter names (default: the cache set)")
    pr.add_argument("--kernel-filter", help="regex to restrict to one kernel")
    pr.add_argument("--tokens", type=int, default=0,
                    help="tokens generated in the run (for bytes/token)")
    pr.add_argument("--fetch-unit", default="kib", choices=["kib", "bytes"])
    pr.add_argument("--dry-run", action="store_true",
                    help="print the rocprofv3 command and exit without running it")
    pr.add_argument("command", nargs=argparse.REMAINDER,
                    help="target command, after `--`")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("analyze", help="analyse an existing rocprofv3 output dir")
    pa.add_argument("dir")
    pa.add_argument("--tokens", type=int, default=0,
                    help="tokens generated in the profiled run")
    pa.add_argument("--fetch-unit", default="kib", choices=["kib", "bytes"])
    pa.add_argument("--json", help="write the metrics to this JSON file")
    pa.set_defaults(func=cmd_analyze)

    pf = sub.add_parser("full", help="profile a command, then analyse the result")
    pf.add_argument("--out", required=True)
    pf.add_argument("--pmc", nargs="+")
    pf.add_argument("--kernel-filter")
    pf.add_argument("--tokens", type=int, default=0)
    pf.add_argument("--fetch-unit", default="kib", choices=["kib", "bytes"])
    pf.add_argument("--json")
    pf.add_argument("--dry-run", action="store_true",
                    help="print the rocprofv3 command and exit without running it")
    pf.add_argument("command", nargs=argparse.REMAINDER)
    pf.set_defaults(func=cmd_full)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
