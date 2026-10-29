#!/usr/bin/env python3
"""gfx1030 consolidated CLI."""
import argparse
import collections
import csv
import datetime
import glob
import json
import hashlib
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from common import ROOT, MODEL, FA, CTK, CTV, BENCH_EXCLUSIVE
# Servers (llama-server) are out of scope for this CLI: never start, stop, or query them.
PAYLOAD_GB = 11.353
HIP_BIN = ROOT / "build-hip-gfx1030" / "bin"
# The Vulkan binaries no longer live in `build/bin`: that path is now a symlink to the
# canonical HIP tree (build-hip-gfx1030/bin) so the server can never resolve to a build
# without the q8_0-q5_0 FA pair. The real Vulkan build is preserved here.
VK_BIN = ROOT / "build" / "bin.superseded-20261028"
HIP_BENCH = HIP_BIN / "llama-bench"
VK_BENCH = VK_BIN / "llama-bench"
HIP_SO = HIP_BIN / "libggml-hip.so.0"
_dev_idx = int(os.environ.get("GFX1030_DEVICE", "0"))
_dev_cards = sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"))
DEV = Path(_dev_cards[_dev_idx] if _dev_idx < len(_dev_cards) else _dev_cards[0]).parent
BENCH_ARGS = ["-ngl", "99", "-fa", FA, "-ctk", CTK, "-ctv", CTV, "-fitt", "128"]
PROFILERS = ["rocprofv3", "rocprofv2", "rocprof", "amd-smi", "rocm-smi", "rocminfo"]

TYPES = ["Q4_K", "IQ3_S", "IQ4_XS", "IQ3_XXS", "IQ2_S"]
GGML_TYPE = {t: "GGML_TYPE_" + t for t in TYPES}
REGEX_TYPE = {"Q4_K": "q4_K", "IQ3_S": "iq3_s", "IQ4_XS": "iq4_xs", "IQ3_XXS": "iq3_xxs", "IQ2_S": "iq2_s"}
PEAK_BW = 512e9

ISA_S = ROOT / "build-hip-gfx1030/ggml/src/ggml-hip/mmvq-hip-amdgcn-amd-amdhsa-gfx1030.s"
ISA_BASE = ROOT / "scripts" / ".isa_baseline.json"

SECTOR = 64
TYPE_LAYOUT = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
}
KV_BPV = {"f16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32}


def _which(name):
    return shutil.which(name) or shutil.which(name, path="/opt/rocm/bin")


def _importable(m):
    try:
        __import__(m)
        return True
    except Exception:
        return False


def _repo_package(name):
    try:
        out = subprocess.run(["pacman", "-Ss", name], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        if "/" in line and name in line:
            return line.strip()
    return ""


def _read_dpm(which):
    try:
        return (DEV / which).read_text()
    except OSError:
        return ""


def _active_clock(which):
    for line in _read_dpm(which).splitlines():
        if "*" in line:
            m = re.search(r"(\d+)\s*Mhz", line, re.I)
            if m:
                return int(m.group(1))
    return None


class ClockSampler(threading.Thread):
    def __init__(self, period=0.2):
        super().__init__(daemon=True)
        self.period = period
        self.stop = False
        self.sclk = collections.Counter()
        self.mclk = collections.Counter()

    def run(self):
        while not self.stop:
            self.sclk[_active_clock("pp_dpm_sclk")] += 1
            self.mclk[_active_clock("pp_dpm_mclk")] += 1
            time.sleep(self.period)

    @staticmethod
    def _fmt(c):
        tot = sum(c.values()) or 1
        return ", ".join(f"{k}:{100*v/tot:.0f}%" for k, v in c.most_common(3))


def _flag(txt, key):
    m = re.search(rf"^{re.escape(key)}:BOOL=(\w+)", txt, re.M)
    return None if m is None else m.group(1).upper()


def _var(txt, key):
    m = re.search(rf"^{re.escape(key)}:\w+=(.*)$", txt, re.M)
    return None if m is None else m.group(1).strip()


def _run(cmd, *, cwd=ROOT, timeout=None, quiet=False, check=False, env=None):
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True,
                       text=True, timeout=timeout, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    if not quiet and p.returncode != 0:
        print(f"  ! exit {p.returncode}: {' '.join(map(str, cmd))}")
    return p


def _bench_exclusive(binpath, args_list, timeout=7200, env=None):
    guard = BENCH_EXCLUSIVE
    cmd = [str(guard), str(binpath)] + [str(a) for a in args_list]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                         timeout=timeout, env=env)


def _parse_depths(s):
    out = []
    for tok in re.split(r"[,\s]+", s.strip()):
        if tok:
            out.append(int(tok))
    return out or [0]



def _percentile(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger(path):
    """Split a ledger into its meta rows and its dispatch rows."""
    meta, rows = {}, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "meta":
            meta = o
        elif o.get("type") == "dispatch" or "duration_us" in o:
            rows.append(o)
    return meta, rows


def _intervention_delta_ms(rec):
    """Numeric ground-truth delta for one intervention row, or None."""
    delta = rec.get("measured_delta_ms")
    if delta is None and isinstance(rec.get("measured"), dict):
        delta = rec["measured"].get("delta_ms")
    return delta


def _numeric_interventions(path):
    out = []
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if _intervention_delta_ms(rec) is not None:
            out.append(rec)
    return out


def _cell_key(d):
    return "|".join([
        d.get("kernel_short", "") or "",
        d.get("weight_type", "") or "",
        f"M{int(d.get('m') or 0)}",
        f"K{int(d.get('k') or 0)}",
    ])



def _load_census(path):
    meta = None
    nodes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if isinstance(o, dict) and o.get("type") == "meta":
            meta = o
        elif isinstance(o, dict) and (o.get("type") == "node" or ("i" in o and "op_name" in o and "role" in o)):
            nodes.append(o)
    return meta, nodes


def _op_to_kernel(op_name):
    mapping = {
        "MUL_MAT": "mul_mat_vec_q",
        "RMS_NORM": "rms_norm_f32",
        "UNARY": "unary_op_kernel",
        "ADD": "k_bin_bcast",
        "MUL": "k_bin_bcast",
        "CPY": "cpy_scalar",
        "GET_ROWS": "k_get_rows_float",
        "ROPE": "rope_multi",
        "FLASH_ATTN_EXT": "flash_attn_ext_gqa_gfx1030",
        "SSM_CONV": "ssm_conv_f32",
        "GATED_DELTA_NET": "gated_delta_net_cuda",
        "CONT": "concat_cont",
        "SCALE": "scale_f32",
        "CONCAT": "concat_cont",
        "GLU": "unary_gated_op_kernel",
    }
    return mapping.get(op_name)


def cmd_predict(args):
    if args.kernel_change and not (args.edit or args.candidate):
        print("ERROR: --kernel-change requires --candidate or --edit")
        return 2
    if not args.edit and not args.candidate:
        print("ERROR: need --candidate (or --edit) to describe the candidate graph")
        return 2
    base_meta, base_nodes = _load_census(args.baseline)
    skipped_ops = {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "NONE"}
    base_nodes = [n for n in base_nodes if n.get("op_name") not in skipped_ops]
    if args.edit:
        edits = json.loads(Path(args.edit).read_text())
        cand_nodes = [dict(n) for n in base_nodes]
        _added_nodes = []
        rem = edits.get("remove")
        # a change may suppress several node classes at once (e.g. a GET_ROWS
        # plus the CPY it made redundant), so accept a list of remove specs
        rems = rem if isinstance(rem, list) else ([rem] if rem else [])
        for spec in rems:
            match = spec.get("match", {})
            count = spec.get("count", 0)
            removed = 0
            new_nodes = []
            for n in cand_nodes:
                ok = True
                if removed < count:
                    for k, v in match.items():
                        val = n.get(k)
                        if isinstance(val, dict):
                            val = val.get("type_name", val.get("type"))
                        if k == "type_name":
                            w = n.get("weight")
                            wt = w.get("type_name") if isinstance(w, dict) else n.get("type_name")
                            if wt != v:
                                ok = False
                        elif val != v:
                            ok = False
                    if ok and removed < count:
                        removed += 1
                        continue
                new_nodes.append(n)
            cand_nodes = new_nodes
        add = edits.get("add")
        if add:
            count = add.get("count", 0)
            for _ in range(count):
                node = {
                    "i": len(cand_nodes),
                    "op_name": add.get("op_name", "CONT"),
                    "name": add.get("op_name", "CONT").lower(),
                    "type_name": add.get("type_name", "bf16"),
                    "ne": add.get("ne", [1, 1, 1, 1]),
                    "weight": {"name": "edited", "type_name": add.get("type_name", "bf16")},
                    "role": add.get("role", "other"),
                    "layer": -1,
                }
                cand_nodes.append(node)
                _added_nodes.append(node)
        cand_meta = dict(base_meta) if base_meta else {}
        cand_meta["n_nodes"] = len(cand_nodes) + (base_meta.get("n_skipped_views", 0) if base_meta else 0)
        cand_meta["n_real"] = len(cand_nodes)
        cand_meta["_added_nodes"] = _added_nodes
        cand_meta["model_sha256"] = base_meta.get("model_sha256") if base_meta else None
        cand_meta["binary_sha256"] = base_meta.get("binary_sha256") if base_meta else None
    else:
        _cm, _cn = _load_census(args.candidate)
        cand_meta = _cm
        cand_nodes = [n for n in _cn if n.get("op_name") not in skipped_ops]
    model = Path(args.model) if args.model else MODEL
    cost_model_path = ROOT / "scripts" / ".cost_model.json"
    if not cost_model_path.exists():
        print("no cost model; run calibrate")
        return 4
    cm = json.loads(cost_model_path.read_text())

    # census mode/ubatch must match tg decode
    for m in (base_meta, cand_meta):
        if m and (m.get("mode") not in (None, "tg") or (m.get("n_ubatch", 1) or 1) != 1):
            print("INSUFFICIENT")
            return 4
    # staleness checks
    stale = False
    if not base_meta or not base_meta.get("model_sha256"):
        stale = True
    elif sha256_file(str(model)) != base_meta.get("model_sha256"):
        stale = True
    if not cand_meta or not cand_meta.get("model_sha256"):
        stale = True
    elif sha256_file(str(model)) != cand_meta.get("model_sha256"):
        stale = True
    bsha = sha256_file(str(HIP_SO)) if HIP_SO.exists() else ""
    if not base_meta or not base_meta.get("binary_sha256"):
        stale = True
    elif bsha and bsha != base_meta.get("binary_sha256"):
        stale = True
    if not cand_meta or not cand_meta.get("binary_sha256"):
        stale = True
    elif bsha and bsha != cand_meta.get("binary_sha256"):
        stale = True
    if cm.get("binary_sha256") and HIP_BENCH.exists():
        if sha256_file(str(HIP_BENCH)) != cm.get("binary_sha256"):
            stale = True
    if stale:
        print("STALE")
        return 5

    # validity: calibrate marks the artifact invalid when a hard check failed
    if cm.get("valid") is False:
        print("INVALID")
        for reason in cm.get("invalid_reason", []):
            print(f"  {reason}")
        return 4

    cells = cm.get("cells", {})
    if not cells:
        print("INSUFFICIENT")
        return 4

    # support check
    support = cm.get("support", {})
    # count only rows with numeric ground truth, and do not trust the stored count
    n_int = len(_numeric_interventions(ROOT / "scripts" / ".interventions.jsonl"))
    n_cells = len(cells)
    refusal = cm.get("refusal", {})
    if n_int < refusal.get("min_interventions", 8) or n_cells < refusal.get("min_cells", 20):
        print("INSUFFICIENT")
        return 4
    # min shapes per class
    shapes = collections.defaultdict(set)
    for key in cells.keys():
        parts = key.split("|")
        if len(parts) >= 4 and parts[3].startswith("K"):
            shapes["|".join(parts[:2])].add(parts[3])
    min_shapes = min((len(v) for v in shapes.values()), default=0)
    if min_shapes < refusal.get("min_shapes_per_class", 5):
        print("INSUFFICIENT")
        return 4

    overhead = cm.get("overhead_us_per_dispatch")
    if overhead is None:
        print("INSUFFICIENT")
        return 4

    def _node_shape(node):
        """(M, K) of a matvec node, read from its weight tensor."""
        w = node.get("weight")
        wname = w.get("name") if isinstance(w, dict) else None
        for src in (node.get("src") or []):
            if wname and src.get("name") == wname:
                ne = src.get("ne") or []
                if ne:
                    return (int(ne[1]) if len(ne) > 1 and ne[1] else 0, int(ne[0]))
        ne = node.get("ne") or []
        return (0, int(ne[0]) if ne else 0)

    def node_cost(node):
        # returns (kernel_ms, p10, p90) predicted for one node
        op = node.get("op_name", "")
        wt = ""
        w = node.get("weight")
        if isinstance(w, dict):
            wt = w.get("type_name", "") or str(w.get("type", ""))
        if not wt:
            wt = node.get("type_name", "")
        if op in ("DUP", "SET_ROWS"):
            return (0.0, 0.0, 0.0)
        if op == "MUL_MAT":
            wt_low = (wt or "").lower()
            if "bf16" in wt_low or "f16" in wt_low:
                kernel = "mul_mat_vec_f"
            else:
                kernel = "mul_mat_vec_q"
        else:
            kernel = _op_to_kernel(op)
        if kernel is None:
            return None
        nm, nk = _node_shape(node)
        # exact cell: (kernel, weight type, M, K)
        key = f"{kernel}|{wt}|M{nm}|K{nk}"
        c = cells.get(key)
        if c is None and op != "MUL_MAT":
            key = f"{kernel}|{wt}|M0|K{nk}"
            c = cells.get(key)
        if c is not None:
            mean = c.get("mean_us", c.get("median_us", 0.0))
            p10 = c.get("p10", mean)
            p90 = c.get("p90", mean)
        else:
            # widen the interval across the same kernel/type (no silent substitution
            # of a point estimate: the interval is explicitly the observed range)
            matching = [k for k in cells.keys() if k.startswith(f"{kernel}|{wt}|")] \
                or [k for k in cells.keys() if k.startswith(f"{kernel}|")]
            if not matching:
                return None
            means = [cells[k].get("mean_us", cells[k].get("median_us", 0.0)) for k in matching]
            p10s = [cells[k].get("p10", cells[k].get("mean_us", 0.0)) for k in matching]
            p90s = [cells[k].get("p90", cells[k].get("mean_us", 0.0)) for k in matching]
            mean = statistics.median(means)
            p10 = min(p10s)
            p90 = max(p90s)
        # expansion: a quantized matvec also pays for its q8_1 activation quantize
        if op == "MUL_MAT" and kernel == "mul_mat_vec_q":
            qkey = [k for k in cells.keys() if k.startswith(f"quantize_q8_1|{wt}|")]
            if qkey:
                qc = cells[qkey[0]]
                mean = mean + qc.get("mean_us", 0.0)
                p10 = p10 + qc.get("p10", 0.0)
                p90 = p90 + qc.get("p90", 0.0)
        return (mean / 1000.0, p10 / 1000.0, p90 / 1000.0)

    if getattr(args, "kernel_change", False):
        out = {
            "Δdispatches": 0,
            "Δkernel_ms": None,
            "Δgap_ms": 0.0,
            "Δwall_ms": None,
            "p10": None,
            "p90": None,
            "break_even_kernel_us": round(overhead, 2),
            "unsupported": ["kernel-level change"],
            "confidence": "low",
            "decision": "UNSUPPORTED",
            "reason": "kernel-level change",
        }
        if args.json:
            print(json.dumps(out))
        else:
            for k, v in out.items():
                print(f"{k}: {v}")
        return 3

    if not args.edit and args.baseline == args.candidate:
        out = {
            "Δdispatches": 0,
            "Δkernel_ms": 0.0,
            "Δgap_ms": 0.0,
            "Δwall_ms": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "break_even_kernel_us": 0.0,
            "unsupported": [],
            "confidence": "high",
            "decision": "NO-CHANGE",
            "reason": "identity input",
        }
        if args.json:
            print(json.dumps(out))
        else:
            for k, v in out.items():
                print(f"{k}: {v}")
        return 0

    # unsupported detection
    unsupported = []
    skip_ops = {"DUP", "SET_ROWS", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
    for node in cand_nodes:
        op = node.get("op_name", "")
        if op in skip_ops:
            continue
        wt = ""
        w = node.get("weight")
        if isinstance(w, dict):
            wt = w.get("type_name", "") or str(w.get("type", ""))
        if not wt:
            wt = node.get("type_name", "")
        if op == "MUL_MAT":
            wt_low = (wt or "").lower()
            if "bf16" in wt_low or "f16" in wt_low:
                kernel = "mul_mat_vec_f"
            else:
                kernel = "mul_mat_vec_q"
        else:
            kernel = _op_to_kernel(op)
        if kernel is None:
            unsupported.append(f"new op {op}")
            continue
        matching = [k for k in cells.keys() if k.startswith(f"{kernel}|{wt}|")] or [k for k in cells.keys() if k.startswith(f"{kernel}|")]
        if not matching:
            unsupported.append(f"new kernel class {kernel}")
    # dedup
    unsupported = sorted(set(unsupported))

    if unsupported:
        # break-even only
        delta_dispatches = cand_meta["n_real"] - base_meta["n_real"]
        be = abs(delta_dispatches) * overhead
        out = {
            "Δdispatches": delta_dispatches,
            "Δkernel_ms": None,
            "Δgap_ms": round(delta_dispatches * overhead / 1000.0, 3),
            "Δwall_ms": None,
            "p10": None,
            "p90": None,
            "break_even_kernel_us": round(be, 1),
            "unsupported": unsupported,
            "confidence": "low",
            "decision": "UNSUPPORTED",
            "reason": "new op/kernel class",
        }
        if args.json:
            print(json.dumps(out))
        else:
            for k, v in out.items():
                print(f"{k}: {v}")
        return 3

    # compute costs
    def total_cost(nodes):
        med_sum = 0.0
        p10_sum = 0.0
        p90_sum = 0.0
        for node in nodes:
            res = node_cost(node)
            if res is None:
                return None
            med_sum += res[0]
            p10_sum += res[1]
            p90_sum += res[2]
        return med_sum, p10_sum, p90_sum

    base_cost = total_cost(base_nodes)
    cand_cost = total_cost(cand_nodes)
    if base_cost is None or cand_cost is None:
        unsupported.append("missing cell")
        delta_dispatches = cand_meta["n_real"] - base_meta["n_real"]
        be = abs(delta_dispatches) * overhead
        out = {
            "Δdispatches": delta_dispatches,
            "Δkernel_ms": None,
            "Δgap_ms": round(delta_dispatches * overhead / 1000.0, 3),
            "Δwall_ms": None,
            "p10": None,
            "p90": None,
            "break_even_kernel_us": round(be, 1),
            "unsupported": unsupported,
            "confidence": "low",
            "decision": "UNSUPPORTED",
            "reason": "missing cell",
        }
        if args.json:
            print(json.dumps(out))
        else:
            for k, v in out.items():
                print(f"{k}: {v}")
        return 3

    delta_dispatches = cand_meta["n_real"] - base_meta["n_real"]
    added_nodes = cand_meta.get("_added_nodes", [])
    added_cost = sum(node_cost(n)[0] for n in added_nodes) if added_nodes else 0.0
    delta_kernel_realistic = cand_cost[0] - base_cost[0]
    delta_kernel_best = delta_kernel_realistic - added_cost
    delta_gap_ms = delta_dispatches * overhead / 1000.0
    delta_wall_realistic = delta_kernel_realistic + delta_gap_ms
    delta_wall_best = delta_kernel_best + delta_gap_ms
    delta_kernel_ms = delta_kernel_realistic
    delta_wall_ms = delta_wall_realistic
    delta_p10 = (cand_cost[1] - base_cost[1]) + delta_gap_ms
    delta_p90 = (cand_cost[2] - base_cost[2]) + delta_gap_ms
    threshold = max(0.30, 0.01 * 32.26)

    if delta_wall_best <= -threshold:
        decision = "GO"
        rc = 0
    elif delta_wall_realistic >= threshold:
        decision = "NO-GO"
        rc = 1
    elif delta_p10 <= -threshold <= delta_p90:
        decision = "MEASURE"
        rc = 2
    else:
        decision = "NO-GO"
        rc = 1

    be = abs(delta_dispatches) * overhead
    out = {
        "Δdispatches": delta_dispatches,
        "Δkernel_ms": round(delta_kernel_ms, 3),
        "Δgap_ms": round(delta_gap_ms, 3),
        "Δwall_ms": round(delta_wall_ms, 3),
        "Δwall_best_case": round(delta_wall_best, 3),
        "Δwall_realistic": round(delta_wall_realistic, 3),
        "p10": round(delta_p10, 3),
        "p90": round(delta_p90, 3),
        "break_even_kernel_us": round(be, 1),
        "unsupported": [],
        "confidence": "medium",
        "decision": decision,
        "reason": decision,
    }
    if args.json:
        print(json.dumps(out))
    else:
        for k, v in out.items():
            print(f"{k}: {v}")
    if args.record:
        import datetime
        rec = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "decision": out.get("decision"),
            "prediction": out,
            "base_model_sha256": base_meta.get("model_sha256") if base_meta else None,
            "candidate_model_sha256": cand_meta.get("model_sha256") if cand_meta else None,
            "base_binary_sha256": base_meta.get("binary_sha256") if base_meta else None,
            "candidate_binary_sha256": cand_meta.get("binary_sha256") if cand_meta else None,
            "measured_delta_ms": None,
        }
        ipath = ROOT / "scripts" / ".interventions.jsonl"
        with open(ipath, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"recorded intervention -> {ipath}")
    return rc



def cmd_probe(args):
    print("=" * 78)
    print("gfx1030 profiling capability probe")
    print("=" * 78)
    avail = {}
    for p in PROFILERS:
        path = _which(p)
        avail[p] = path
        print(f"  {p:12s} {path or 'MISSING'}")

    for pat in ("librocprofiler*.so*", "libroctracer64.so*", "libhsa-runtime64.so*"):
        found = sorted(glob.glob(f"/opt/rocm/lib/{pat}"))
        print(f"  {pat:28s} {len(found)} file(s)" + (f"  e.g. {Path(found[0]).name}" if found else ""))

    pymods = [m for m in ("rocprofiler", "roctx", "torch", "triton") if _importable(m)]
    print(f"  python modules: {pymods or 'none'}")

    clocks = (DEV / "pp_dpm_sclk").exists(), (DEV / "pp_dpm_mclk").exists()
    print(f"  sysfs DPM: sclk={clocks[0]} mclk={clocks[1]}"
          + (f"  active sclk line: {_active_clock('pp_dpm_sclk')} MHz" if clocks[0] else ""))

    print("-" * 78)
    if avail.get("rocprofv3"):
        print("  VERDICT: kernel trace available (rocprofv3). PMC is plumbed via")
        print("           --pmc but untested on this host; probe permissions before")
        print("           relying on any counter.")
        return 0
    print("  VERDICT: no ROCm profiler installed; only interposer hooks are present")
    print("           (rocprofiler-register, libroctracer64). PMC counters and")
    print("           per-kernel traces are NOT obtainable until one is installed.")
    hint = _repo_package("rocprofiler")
    if hint:
        print(f"  ACTION:  sudo pacman -S rocprofiler    # {hint}")
    else:
        print("  ACTION:  install rocprofiler for your ROCm version (needs sudo)")
    print("           Until then: use `ab` for the clock/throughput separation and")
    print("           HIP_NODE_TIMING=1 + `gaps` for op-class attribution.")
    return 1



def _thermal_check():
    temp = ""
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showtemp"], quiet=True).stdout
        m = re.search(r"(\d+\.\d+)", out)
        if m:
            temp = m.group(1)
    elif shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], quiet=True).stdout.strip()
        if out:
            temp = out.splitlines()[0]
    if temp:
        print(f"  GPU temp before bench: {temp}°C")
        try:
            if float(temp) > 75:
                print(f"  WARNING: GPU hot ({temp}°C). Numbers may be thermally throttled.")
                time.sleep(5)
        except ValueError:
            pass


def _llama_bench_args(args, d):
    """llama-bench argv shared by bench/AB paths — pp/n/ubatch/ncmoe/extra are
    campaign-critical (-ub 128 is the T5 operating point) and must be settable."""
    al = ["-m", str(MODEL)] + BENCH_ARGS + [
        "-p", str(getattr(args, "pp", 0)),
        "-n", str(getattr(args, "n_gen", 128)),
        "-r", str(getattr(args, "reps", 8)), "-d", str(d), "-o", "jsonl"]
    if getattr(args, "ubatch", None):
        al += ["-ub", str(args.ubatch)]
    if getattr(args, "ncmoe", None):
        al += ["-ncmoe", str(args.ncmoe)]
    if getattr(args, "extra", None):
        al += shlex.split(args.extra)
    return al


def _git_head():
    p = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], quiet=True)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def _run_one_bench(binpath, depths, args, env=None):
    rows_out = []
    for d in depths:
        bargs = _llama_bench_args(args, d)
        sm = ClockSampler()
        sm.start()
        proc = _bench_exclusive(binpath, bargs, timeout=7200, env=env)
        sm.stop = True
        sm.join(timeout=1.0)
        parsed = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("avg_ts") is not None:
                parsed.append(j)
        for j in parsed:
            rows_out.append({
                "depth": j.get("n_depth", d),
                "ts": j.get("avg_ts", 0.0),
                "std": j.get("stddev_ts", 0.0),
                "clocks": dict(sm.sclk),
                "mclk": dict(sm.mclk),
                "rc": proc.returncode,
                "stderr": proc.stderr[-300:] if proc.returncode != 0 else "",
            })
        if not parsed:
            rows_out.append({"depth": d, "ts": 0.0, "std": 0.0, "clocks": dict(sm.sclk),
                             "mclk": dict(sm.mclk), "rc": proc.returncode,
                             "stderr": proc.stderr[-300:]})
    return rows_out


def cmd_bench(args):
    depths = _parse_depths(args.depths)
    reps = args.reps
    backend = args.backend
    ab = args.ab

    if getattr(args, "ab_stats", None):
        # offline resolution calculator: feed measured per-pair ms/token deltas
        vals = [float(x) for x in str(args.ab_stats).split(",") if x.strip()]
        _ab_resolution(vals, getattr(args, "min_detectable", None))
        return 0

    ab_mode = (bool(ab) or bool(getattr(args, "a_bindir", None))
               or bool(getattr(args, "b_bindir", None))
               or bool(getattr(args, "a_env", None))
               or bool(getattr(args, "b_env", None)))
    if ab_mode:
        return _bench_ab(args, ab, depths, reps)

    if getattr(args, "dry_run", False):
        for name, binpath in (("HIP", HIP_BENCH), ("VK", VK_BENCH)):
            if name == "VK" and args.backend not in ("vk", "both"):
                continue
            if name == "HIP" and args.backend not in ("hip", "both"):
                continue
            args_list = _llama_bench_args(args, depths[0] if depths else 0)
            print("DRY-RUN %s: %s" % (name, shlex.join([str(BENCH_EXCLUSIVE), str(binpath)] + args_list)))
        return 0

    _thermal_check()

    backends = []
    if backend in ("hip", "both"):
        backends.append(("HIP", HIP_BENCH))
    if backend in ("vk", "both"):
        backends.append(("VK", VK_BENCH))

    for name, binpath in backends:
        if not binpath.exists():
            print(f"missing binary: {binpath}")
            return 2

    print(f"depths={depths} reps={reps} backend={backend}")
    all_rows = {}
    for name, binpath in backends:
        print(f"\n== {name} ==")
        rows = _run_one_bench(binpath, depths, args)
        all_rows[name] = rows
        for r in rows:
            ts = r["ts"]
            stdpct = (100.0 * r["std"] / ts) if ts else 0.0
            flag = "  <- stddev >2%" if ts and r["std"] > 0.02 * ts else ""
            print(f"  d={r['depth']:>7d}: {ts:8.2f} ± {r['std']:.2f} t/s ({stdpct:4.1f}%){flag}")
            print(f"       sclk: {ClockSampler._fmt(collections.Counter(r['clocks']))}")
            if r["rc"] != 0:
                print(f"       rc={r['rc']} {r['stderr']}")
    if len(backends) == 2:
        print("\n== summary ==")
        print(f"  {'depth':>7s} {'HIP':>8s} {'VK':>8s} {'VK/HIP':>8s}")
        for d in depths:
            h = next((r["ts"] for r in all_rows["HIP"] if r["depth"] == d), 0.0)
            v = next((r["ts"] for r in all_rows["VK"] if r["depth"] == d), 0.0)
            ratio = (v / h) if h else float("nan")
            print(f"  {d:>7d} {h:8.2f} {v:8.2f} {ratio:8.3f}")
    if getattr(args, "record", False):
        rec = {"binary_sha256": sha256_file(str(HIP_BENCH)), "model_sha256": sha256_file(str(MODEL))}
        Path(ROOT / "scripts" / ".bench_record.json").write_text(json.dumps(rec, indent=2))
        print("recorded:", rec)
    return 0


def _t975(n):
    """Two-sided 95 % Student-t multiplier for n-1 degrees of freedom."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
             14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
             20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    df = max(1, n - 1)
    if df in table:
        return table[df]
    for k in sorted(table):
        if df < k:
            return table[k]
    return 1.96


def _n_required(sd_pair, delta_ms):
    """Interleaved pairs needed to resolve delta_ms at 95 % (n = (1.96*sd/delta)^2)."""
    if not sd_pair or sd_pair <= 0 or not delta_ms or delta_ms <= 0:
        return None
    return int(math.ceil((1.96 * sd_pair / delta_ms) ** 2))


def _ab_resolution(deltas_ms, target=None, base_ms=None, fix_ms=None):
    """Print and return the paired resolution of an A/B comparison.

    The decision quantity is the paired difference in ms/token.  Pairing
    cancels the run-to-run drift this host has, so the sd of the *differences*
    -- not the spread of the absolute runs -- sets the resolution.  A caller
    that wants to prove delta_ms needs about (1.96*sd/delta_ms)^2 pairs.
    Returns None below two complete pairs, where no sd exists.
    """
    good = [x for x in deltas_ms if x == x]          # drop nan
    print("\n== paired summary (ms/token, negative = FIX faster) ==")
    print(f"  pairs                 : {len(good)}")
    if base_ms is not None:
        print(f"  BASE                  : {base_ms:8.3f} ms/token")
    if fix_ms is not None:
        print(f"  FIX                   : {fix_ms:8.3f} ms/token")
    if len(good) < 2:
        print("  UNDERPOWERED: fewer than 2 complete pairs; no sd, no CI, no claim")
        return None
    mean_d = statistics.mean(good)
    sd_d = statistics.stdev(good)
    se = sd_d / math.sqrt(len(good))
    t975 = _t975(len(good))
    ci = t975 * se
    pct = (" (%+.2f%%)" % (100.0 * mean_d / base_ms)) if base_ms else ""
    print(f"  paired mean delta     : {mean_d:+8.3f} ms/token{pct}")
    print(f"  paired sd (of deltas) : {sd_d:8.3f} ms   <- spread of the per-pair differences")
    print(f"  paired SE             : {se:8.3f} ms")
    print(f"  95% CI half-width     : {ci:8.3f} ms  (t={t975:.3f}, df={len(good)-1})")
    if abs(mean_d) < ci:
        print(f"  no resolvable difference: |delta| {abs(mean_d):.3f} < CI {ci:.3f} ms")
    nreq = None
    if target:
        nreq = _n_required(sd_d, target)
        print(f"  target delta          : {target:8.3f} ms/token")
        if nreq:
            print(f"  required n            : {nreq} pairs  (n = (1.96*sd/delta)^2)")
        if ci > target:
            need = f"; need >= {nreq} pairs" if nreq else ""
            print(f"  UNDERPOWERED: this {len(good)}-pair A/B cannot resolve {target:.3f} ms "
                  f"(95% CI +/-{ci:.3f} ms){need}")
        else:
            print(f"  powered for {target:.3f} ms at {len(good)} pairs")
    print("  note: this CI is the decision threshold for a change. Per-model spreads in")
    print("        scripts/.baseline.json are pair-to-pair medians, not thresholds.")
    return {"pairs": len(good),
            "mean_delta_ms": round(mean_d, 4),
            "sd_pair_ms": round(sd_d, 4),
            "se_ms": round(se, 4),
            "ci95_ms": round(ci, 4),
            "t975": t975,
            "target_ms": target,
            "required_pairs": nreq,
            "underpowered": bool(target and ci > target)}


def _parse_env_list(items):
    env = {}
    for item in (items or []):
        for tok in re.split(r"[,\s]+", str(item).strip()):
            if not tok:
                continue
            if "=" in tok:
                k, v = tok.split("=", 1)
                env[k] = v
    return env


def _bindir_hashes(bindir):
    bd = Path(bindir)
    out = {"bindir": str(bd), "bench": None, "libs": []}
    bench = bd / "llama-bench"
    if bench.exists():
        out["bench"] = {"path": str(bench), "sha256": sha256_file(str(bench)),
                        "mtime": bench.stat().st_mtime}
    for p in sorted(bd.glob("libggml*.so*")):
        if p.is_file() or p.is_symlink():
            try:
                out["libs"].append({"name": p.name, "path": str(p), "sha256": sha256_file(str(p))})
            except OSError:
                pass
    return out


def _bench_ab(args, ab, depths, reps):
    parts = ab if isinstance(ab, list) else (ab.split() if ab else [])
    swap_libs = bool(parts)
    if swap_libs:
        if len(parts) != 2:
            print("usage: --ab BASE.so FIX.so")
            return 2
        base_so, fix_so = Path(parts[0]), Path(parts[1])
        for p in (base_so, fix_so):
            if not p.exists():
                print(f"missing: {p}")
                return 2
    else:
        base_so = fix_so = None

    a_dir = Path(getattr(args, "a_bindir", None) or HIP_BIN)
    b_dir = Path(getattr(args, "b_bindir", None) or HIP_BIN)
    for bd in (a_dir, b_dir):
        if not (bd / "llama-bench").exists():
            print(f"missing binary: {bd / 'llama-bench'}")
            return 2

    a_env_extra = _parse_env_list(getattr(args, "a_env", None))
    b_env_extra = _parse_env_list(getattr(args, "b_env", None))

    # when swapping a .so in place, both arms share one bindir/binary
    same_dir = (a_dir == b_dir)
    do_swap = swap_libs and same_dir

    if swap_libs and not same_dir:
        print("note: --ab library swap ignored because --a-bindir/--b-bindir differ; "
              "using each bindir's own library")

    d0 = depths[0] if depths else 0
    pairs = getattr(args, "pairs", 3)
    gate = getattr(args, "gate", None)
    # The project's documented A/B gate is five interleaved pairs at a +/-0.40 t/s decode floor
    # (docs/gfx1030-test-plan.md, "Standing corrections" 4). That is the threshold this script's
    # --gate is compared against, so warn when a run cannot even meet it. Twenty pairs tighten the
    # paired 95% CI to about +/-0.15-0.20 t/s; that is a precision note, not the gate.
    if pairs < 5:
        print(f"  WARNING: pairs={pairs} is below the documented five-pair A/B gate "
              f"(+/-0.40 t/s decode floor) — treat as preliminary")
    elif pairs < 20:
        print(f"  note: pairs={pairs} meets the five-pair gate (+/-0.40 t/s); "
              f"20 pairs resolve about +/-0.15-0.20 t/s")

    print(f"A/B: pairs={pairs} depth={d0} reps={reps}  "
          f"A={a_dir}  B={b_dir}  swap={do_swap}")
    if a_env_extra:
        print(f"  A env: {a_env_extra}")
    if b_env_extra:
        print(f"  B env: {b_env_extra}")

    # dry run: print exact commands, no launch
    if getattr(args, "dry_run", False):
        args_list = _llama_bench_args(args, d0)
        for label, bd, extra in (("A", a_dir, a_env_extra), ("B", b_dir, b_env_extra)):
            env = dict(os.environ)
            env.update(extra)
            env["LD_LIBRARY_PATH"] = str(bd.resolve()) + (":" + env.get("LD_LIBRARY_PATH", "")
                                                if env.get("LD_LIBRARY_PATH") else "")
            cmd = [str(BENCH_EXCLUSIVE), str(bd / "llama-bench")] + args_list
            envprefix = " ".join(f"{k}={v}" for k, v in sorted(env.items())
                                 if k in extra or k == "LD_LIBRARY_PATH")
            print(f"DRY-RUN {label}: {envprefix} {shlex.join(cmd)}")
        return 0

    # The swap target must be the arm's own bindir, not the hard-coded HIP_BIN: when
    # a_dir == b_dir but a_dir != HIP_BIN, writing to HIP_SO would clobber the real
    # tree and never affect the arm being tested (silent no-op A/B).
    swap_target = (a_dir / "libggml-hip.so.0") if do_swap else None
    backup = None
    if do_swap and swap_target is not None:
        backup = Path(tempfile.mkstemp(prefix="gfx1030_libbackup_")[1])
        shutil.copy2(swap_target, backup)

    try:
        records = []   # (pair, arm_label, test_label, ts, clocks, rc)
        for i in range(pairs):
            order = ("A", "B") if i % 2 == 0 else ("B", "A")
            for arm in order:
                if arm == "A":
                    bd = a_dir
                    extra = a_env_extra
                    libsrc = base_so
                else:
                    bd = b_dir
                    extra = b_env_extra
                    libsrc = fix_so
                if do_swap and libsrc is not None and swap_target is not None:
                    shutil.copy2(libsrc, swap_target)
                binpath = bd / "llama-bench"
                args_list = _llama_bench_args(args, d0)
                env = dict(os.environ)
                env.update(extra)
                env["LD_LIBRARY_PATH"] = str(bd.resolve()) + (":" + env.get("LD_LIBRARY_PATH", "")
                                                    if env.get("LD_LIBRARY_PATH") else "")
                sm = ClockSampler()
                sm.start()
                proc = _bench_exclusive(binpath, args_list, timeout=7200, env=env)
                sm.stop = True
                sm.join(timeout=1.0)
                # collect every row, keyed by its test label (pp* / tg*)
                row_ts = {}
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if j.get("avg_ts") is None:
                        continue
                    label_test = j.get("test") or ("tg%d" % j.get("n_gen", 0)
                                                   if j.get("n_gen") else "pp%d" % j.get("n_prompt", 0))
                    row_ts[label_test] = j["avg_ts"]
                if not row_ts:
                    print(f"  pair {i+1} {arm}: rc={proc.returncode} no jsonl rows — "
                          f"arm recorded NaN (pair will be dropped, not zeroed)")
                    row_ts["tg%d" % getattr(args, "n_gen", 128)] = float("nan")
                for tlabel, ts in row_ts.items():
                    records.append((i + 1, arm, tlabel, ts, dict(sm.sclk), proc.returncode))
                    print(f"  pair {i+1} {arm} {tlabel}: {ts:8.3f} t/s  sclk {ClockSampler._fmt(sm.sclk)}")
    finally:
        if do_swap and backup is not None and swap_target is not None:
            shutil.copy2(backup, swap_target)

    if do_swap and backup is not None:
        cmp_proc = subprocess.run(["cmp", str(backup), str(swap_target)], capture_output=True, text=True)
        if cmp_proc.returncode == 0:
            print("  restored libggml-hip.so.0 verified (cmp OK)")
        else:
            print("  WARNING: restored library differs from backup!")
            print(cmp_proc.stderr)

    # group by test label, report separately
    test_labels = sorted({r[2] for r in records})
    results = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pairs": pairs,
        "depth": d0,
        "reps": reps,
        "gate_tps": gate,
        "tree": {"git_head": _git_head()},
        "arms": {
            "A": {"bindir": str(a_dir), "env": a_env_extra, "provenance": _bindir_hashes(a_dir),
                  **({"lib": str(base_so), "lib_sha256": sha256_file(str(base_so))}
                     if do_swap else {})},
            "B": {"bindir": str(b_dir), "env": b_env_extra, "provenance": _bindir_hashes(b_dir),
                  **({"lib": str(fix_so), "lib_sha256": sha256_file(str(fix_so))}
                     if do_swap else {})},
        },
        "tests": {},
    }

    print("\n== per-test paired statistics ==")
    for tlabel in test_labels:
        by_pair = collections.defaultdict(dict)
        for pair, arm, tl, ts, clocks, rc in records:
            if tl != tlabel:
                continue
            by_pair[pair][arm] = ts
        base_vals, fix_vals, deltas_ms = [], [], []
        print(f"\n-- {tlabel} --")
        print(f"  {'pair':>4s} {'A':>9s} {'B':>9s} {'delta%':>8s} {'delta ms':>10s}")
        for pair in sorted(by_pair):
            a_ts = by_pair[pair].get("A", float("nan"))
            b_ts = by_pair[pair].get("B", float("nan"))
            delta = (100.0 * (b_ts - a_ts) / a_ts) if a_ts else float("nan")
            dms = (1000.0 / b_ts - 1000.0 / a_ts) if (a_ts and b_ts) else float("nan")
            base_vals.append(a_ts)
            fix_vals.append(b_ts)
            deltas_ms.append(dms)
            print(f"  {pair:4d} {a_ts:9.3f} {b_ts:9.3f} {delta:+8.2f}% {dms:+10.3f}")
        # a failed arm is NaN, not 0.0 — drop it from means/stats, never zero it
        # filter base, fix and deltas_ms in lockstep so the pair count agrees
        good_idx = [i for i, (a, b) in enumerate(zip(base_vals, fix_vals)) if a == a and b == b]
        base_vals = [base_vals[i] for i in good_idx]
        fix_vals = [fix_vals[i] for i in good_idx]
        deltas_ms = [deltas_ms[i] for i in good_idx]
        if base_vals and fix_vals:
            ma = sum(base_vals) / len(base_vals)
            mb = sum(fix_vals) / len(fix_vals)
            print(f"  mean  {ma:9.3f} {mb:9.3f} {100.0*(mb-ma)/ma:+8.2f}%")
            res = _ab_resolution(deltas_ms, getattr(args, "min_detectable", None),
                                 base_ms=1000.0 / ma, fix_ms=1000.0 / mb)
            results["tests"][tlabel] = {
                "pairs": len(base_vals),
                "mean_A_ts": round(ma, 4),
                "mean_B_ts": round(mb, 4),
                "paired": res,
            }
            if gate is not None:
                gain = mb - ma
                ok = gain >= gate
                print(f"  GATE {gate:.3f} t/s: {'PASS' if ok else 'FAIL'}  (gain {gain:+.3f} t/s)")
                results["tests"][tlabel]["gate"] = {"threshold_tps": gate, "gain_tps": round(gain, 4),
                                                   "pass": bool(ok)}

    # write results JSON
    out_path = getattr(args, "json_out", None) or getattr(args, "out", None)
    if not out_path:
        # outside the repo: A/B scratch output must never land in the tracked tree
        out_path = "/tmp/opencode/gfx1030/.ab_result.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote results JSON: {out_path}")

    if backup is not None:
        backup.unlink(missing_ok=True)
    return 0


# ------------------------------------------------------------------ profile --
#
# One traced llama-bench run -> a per-kernel-family cost table, a versioned JSON
# side-car, and a dense-vs-MoE diff.
#
# The trace is always taken with hipGraph disabled (GGML_CUDA_DISABLE_GRAPHS=1):
# hipGraph nodes are not HSA dispatch packets, so a graph-captured run traces
# only the model-load kernels. Every dispatch count below is the graph-free
# count, which is an upper bound on the graph-enabled one.
#
# Conventions (measured, do not re-derive):
#   * llama-bench runs one warmup generation token -> divide by (tokens + 1).
#   * bare-kernel launch floor on this host            1.67 us
#   * a class at or below 2.5 us/call is launch-bound
#   * in-situ matvec bandwidth ceiling                 458 GB/s = 0.89 of peak
#   * PEAK_BW 512 GB/s is the card's spec number; anything above it is a bug.

PROFILE_SCHEMA   = "gfx1030.profile/1"
DISPATCH_FLOOR_US = 1.67
DISPATCH_BOUND_US = 2.5
BW_BOUND_FRAC     = 0.55
PROFILE_MODELS = {
    "dense": MODEL,
    "moe": Path("/home/stormrage/Downloads/Qwen3.8-35B-A3B-Q4_K_M.gguf"),
}
PROFILE_BIN  = {"hip": HIP_BIN, "vk": VK_BIN}
PROFILE_DEV  = {"hip": "ROCm0", "vk": "Vulkan0"}
PROFILE_CLASSES = ["matvec", "moe-matvec", "attention", "ssm", "norm", "rope",
                   "elementwise", "gather", "quantize", "copy", "other"]
# Classes that are a geometry split, not a kernel family.  "moe-matvec" is
# grid.y > 1 (MUL_MAT_ID = expert ids); it is named after the expert path, not
# after a symbol, and at ubatch 1 the dispatch is served by the plain
# mul_mat_vec_q template while mul_mat_vec_q_moe never runs.  A class whose
# name implies a symbol has to say which symbols actually ran, or the label
# sends the next reader after a kernel that is not on the GPU.
GEOMETRY_CLASSES = {
    "moe-matvec": {
        "why": "grid.y = expert-id count (MUL_MAT_ID)",
        "implied": "mul_mat_vec_q_moe",
    },
}
# 2D weight tensors that are never touched by a plain decode graph.
PROFILE_SKIP_TENSOR = (".nextn.",)


def _profiler():
    for p in ("rocprofv3", "rocprofv2", "rocprof"):
        path = _which(p)
        if path:
            return p, path
    return None, None


def _profile_class(name):
    low = (name or "").lower()
    if "topk_moe" in low or "moe_weighted" in low:
        return "routing"
    if "mul_mat" in low:
        # geometry decides matvec vs expert matvec: grid.y is the id count
        return "matvec"
    if "flash" in low or "attn" in low or "fattn" in low:
        return "attention"
    if "ssm" in low or "gated_delta" in low or "fwht" in low:
        return "ssm"
    if "norm" in low:
        return "norm"
    if "rope" in low:
        return "rope"
    if "quantize" in low:
        return "quantize"
    if "get_rows" in low or "getrows" in low or "set_rows" in low:
        return "gather"
    if "rocclr" in low or "cpy" in low or "copy" in low:
        return "copy"
    for kw in ("bin_bcast", "scale", "concat", "unary", "silu", "sigmoid", "softplus", "add", "mul"):
        if kw in low:
            return "elementwise"
    return "other"


def _profile_type(name):
    """Weight dtype of a matvec instantiation, read from the kernel symbol."""
    m = re.search(r"\(ggml_type\)\s*(\d+)", name or "")
    if m:
        layout = TYPE_LAYOUT.get(int(m.group(1)))
        return layout[0] if layout else "?"
    head = (name or "").split("<", 1)[1][:60].lower() if "<" in (name or "") else ""
    if "__hip_bfloat16" in head:
        return "BF16"
    if "half" in head:
        return "F16"
    if "float" in head:
        return "F32"
    return "?"


def _profile_fused(name):
    """True when a matvec instantiation carries the gate/up fusion argument."""
    m = re.search(r"\(ggml_type\)\s*\d+,\s*\d+,\s*(true|false)", name or "")
    return bool(m and m.group(1) == "true")


def _profile_moe_rpb(name):
    """rows-per-block template argument of a MoE matvec (1 when not found)."""
    m = re.search(r"mul_mat_vec_q_moe<[^,]*,\s*(\d+)\s*,", name or "")
    return int(m.group(1)) if m else 1


def _profile_nmats(name):
    """Weight matrices one dispatch streams: 2 when a gate is fused, else 1.

    A fused swiglu pair reads the up matrix *and* the gate matrix: the fusion
    passes gate_n->src[0] as a second weight tensor and mmvq runs vec_dot
    against both.  Charging such a dispatch 1x understates the class by a
    third on this model.
    """
    return 2 if _profile_fused(name) else 1


def _kernel_label(name):
    """Compact label of one kernel instantiation: symbol<dtype,ncols[,fused]>.

    _kernel_short keeps only the symbol and therefore merges every dtype and
    every template variant into one row -- which is why the profiler could not
    tell mul_mat_vec_q from mul_mat_vec_q_moe.  Symbols without a template
    argument keep the plain short name.
    """
    short = _kernel_short(name)
    if "<" not in (name or "") or ">" not in name:
        return short
    inner = name.split("<", 1)[1].split(">", 1)[0]
    out = []
    for p in [x.strip() for x in inner.split(",")]:
        if p.startswith("(ggml_type)"):
            out.append(_profile_type(name) or p)
        elif p in ("true", "false"):
            out.append("fused" if p == "true" else "plain")
        elif re.fullmatch(r"\d+", p):
            out.append(p)
        else:
            return short
    # the first two slots identify the instantiation (dtype, ncols_dst / rpb);
    # the loose trailing flags are dropped unless they are set
    head, tail = out[:2], [x for x in out[2:] if x != "plain"]
    return "%s<%s>" % (short, ",".join(head + tail))


def _profile_gguf_index(path):
    """Weight inventory keyed by (dtype, n_rows), split by tensor rank.

    A plain MUL_MAT streams one 2D weight matrix; an expert matvec (MUL_MAT_ID,
    dispatched with grid.y = n_ids) streams one expert slice out of a 3D
    [n_cols, n_rows, n_expert] tensor.  Keying both by (dtype, n_rows) alone
    mixes a 15.6 GiB expert table with the 512-row dense matrices and inflates
    the dense byte estimate by 30x, so the two inventories stay separate.
    """
    idx2d, idx3d, info = {}, {}, {}
    if not path or not Path(path).exists():
        return idx2d, idx3d, info
    version, alignment, data_start, tensors, kv = _read_gguf(path)
    payload = 0
    for t in tensors:
        ne = t.get("ne") or []
        if len(ne) < 2:
            continue
        name, row_bytes, n_rows = type_geom(t["type"], t["ne"])
        nbytes = row_bytes * n_rows
        payload += nbytes
        if any(s in t["name"] for s in PROFILE_SKIP_TENSOR):
            continue
        out = idx2d if len(ne) == 2 else idx3d
        rows = 1
        for dim in ne[1:]:
            rows *= int(dim)
        e = out.setdefault((name, int(ne[1])), {"n": 0, "bytes": 0, "rows": 0})
        e["n"] += 1
        e["bytes"] += nbytes
        e["rows"] += rows
    info = {"payload_bytes": payload, "tensors": len(tensors), "kv": kv,
            "arch": kv.get("general.architecture", "?")}
    return idx2d, idx3d, info


def _profile_dispatches(csvs):
    """Flatten trace CSVs into one time-ordered dispatch list."""
    recs = []
    for path in csvs:
        fields, rows = _parse_csv(path)
        if not rows:
            continue
        lower = {f.lower(): f for f in fields}

        def fld(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        kf, sf, ef = fld("kernel_name"), fld("start_timestamp"), fld("end_timestamp")
        kindf = fld("kind")
        gf = [fld("grid_size_x"), fld("grid_size_y"), fld("grid_size_z")]
        wf = [fld("workgroup_size_x"), fld("workgroup_size_y"), fld("workgroup_size_z")]
        vgf, ldsf = fld("vgpr_count"), fld("lds_block_size")
        if not (kf and sf and ef):
            continue
        for row in rows:
            if kindf and row.get(kindf) and row[kindf] != "KERNEL_DISPATCH":
                continue
            try:
                start = float(row[sf])
                end = float(row[ef])
            except (TypeError, ValueError):
                continue
            recs.append({
                "kernel": row[kf],
                "kernel_short": _kernel_short(row[kf]),
                "kernel_label": _kernel_label(row[kf]),
                "start": start,
                "end": end,
                "duration_us": (end - start) / 1000.0,
                "grid": [int(row.get(f) or 0) for f in gf if f],
                "workgroup": [int(row.get(f) or 0) for f in wf if f],
                "vgpr": int(row.get(vgf) or 0) if vgf else 0,
                "lds": int(row.get(ldsf) or 0) if ldsf else 0,
            })
    recs.sort(key=lambda d: d["start"])
    return recs


def _profile_window(recs, tokens):
    """Cut the trace to the decode blocks and return how many there are.

    Token boundaries are the vocab-projection matvec (the largest M observed).
    Decode blocks all have the same dispatch count, so the trailing run of
    equal-length blocks is the measured generation: at -d 0 that run also
    swallows the single warmup generation (llama-bench does one before the rep
    loop), which is why the divisor is tokens + 1 there; at -d > 0 the depth
    prompt separates the warmup from the measured blocks and the divisor is
    just the number of blocks.  The first block also holds the model-load
    dispatches, so it is trimmed to the modal block length.
    """
    mvs = [(i, r["grid"][0] // r["workgroup"][0]) for i, r in enumerate(recs)
           if "mul_mat_vec" in r["kernel"].lower() and r["workgroup"] and r["workgroup"][0]]
    if not mvs:
        return recs, tokens + 1, "no matvec boundary; whole trace used and divided by tokens+1"
    vmax = max(m for _i, m in mvs)
    bnd = [i for i, m in mvs if m == vmax]
    if len(bnd) < 2:
        return recs, tokens + 1, "no usable boundary; whole trace used and divided by tokens+1"
    lens = [b - a for a, b in zip(bnd, bnd[1:])]
    last = lens[-1]
    k = 0
    for L in reversed(lens):
        if abs(L - last) > 0.10 * last:
            break
        k += 1
    if k == len(lens):
        # every block is a decode block: include block 0, trimmed of load noise
        start = max(0, bnd[0] - last + 1)
        blocks = k + 1
        note = "%d decode blocks of %d dispatches (first trimmed of model-load noise)" % (blocks, last)
    else:
        start = bnd[-k - 1] + 1
        blocks = k
        note = "%d decode blocks of %d dispatches after the last prompt block" % (blocks, last)
    blocks = max(1, min(blocks, tokens + 1))
    return recs[start: bnd[-1] + 1], blocks, note


def _profile_aggregate(dispatches, divisor, idx2d, idx3d, gguf_info, depth):
    """Build the class / family tables, deriving DRAM bytes per dispatch."""
    notes = []

    for d in dispatches:
        d["class"] = _profile_class(d["kernel"])
        d["bytes"] = 0
        d["bytes_basis"] = "none"
        d["type"] = ""
        d["m"] = 0
        d["ids"] = 1
        # a fused gate+up dispatch streams two weight matrices; charge both
        d["nmats"] = _profile_nmats(d["kernel"])
        if d["class"] not in ("matvec", "moe-matvec"):
            continue
        wgx = d["workgroup"][0] if d["workgroup"] else 0
        if not wgx:
            continue
        d["type"] = _profile_type(d["kernel"])
        d["m"] = (d["grid"][0] // wgx) * _profile_moe_rpb(d["kernel"])
        d["ids"] = d["grid"][1] if len(d["grid"]) > 1 and d["grid"][1] else 1
        if d["ids"] > 1 and (d["type"], d["m"]) in idx3d:
            # grid.y is the id count and the weight is a 3D expert table:
            # this is a MUL_MAT_ID dispatch.  A batched dense matvec also
            # raises grid.y (2 columns on this model) and must stay "matvec".
            d["class"] = "moe-matvec"
        elif d["ids"] > 1:
            notes.append("batched matvec %s grid.y=%d M=%d kept dense"
                         % (d["kernel_short"], d["ids"], d["m"]))
        d["bytes_basis"] = "dram-weights"

    # dense matvec: every dispatch streams one 2D weight matrix, except the
    # fused gate/up dispatches, which stream two.  The GGUF inventory is
    # apportioned by matrix reads, not by dispatch count, so the two are not
    # charged the same bytes.
    groups = {}
    for d in dispatches:
        if d["class"] != "matvec":
            continue
        key = (d["type"], d["m"])
        g = groups.setdefault(key, {"disp": 0, "disp_eff": 0})
        g["disp"] += 1
        g["disp_eff"] += d["nmats"]

    understated = 0
    for key, g in groups.items():
        ent = idx2d.get(key)
        if not ent or not ent["n"]:
            notes.append("matvec group %s/M=%d has no 2D GGUF tensor; bytes not derived" % (key[0], key[1]))
            continue
        scale = min(1.0, (g["disp_eff"] / divisor) / ent["n"])
        if scale < 0.95:
            understated += ent["bytes"] * (1.0 - scale)
        g["bytes_per_token"] = ent["bytes"] * scale

    for d in dispatches:
        if d["class"] != "matvec":
            continue
        g = groups.get((d["type"], d["m"]))
        if not g or not g.get("disp_eff") or "bytes_per_token" not in g:
            continue
        d["bytes"] = g["bytes_per_token"] / (g["disp_eff"] / divisor) * d["nmats"]

    # expert matvec: bytes are explicit -- active ids x rows x row bytes, from
    # the 3D [n_cols, n_rows, n_expert] inventory.
    moe_groups = {}
    for d in dispatches:
        if d["class"] != "moe-matvec":
            continue
        key = (d["type"], d["m"])
        if key not in moe_groups:
            ent = idx3d.get(key)
            moe_groups[key] = {
                "avg_row_bytes": (ent["bytes"] / max(ent["rows"], 1)) if ent and ent["n"] else 0.0,
                "gguf_n": ent["n"] if ent else 0,
                "full_bytes": ent["bytes"] if ent else 0,
            }
        mg = moe_groups[key]
        # active ids x rows x row bytes, once per weight matrix the dispatch
        # reads: the fused gate+up dispatch streams two expert tables, so it
        # is charged 2x.
        d["bytes"] = d["ids"] * d["m"] * mg["avg_row_bytes"] * d["nmats"]
        d["bytes_basis"] = ("dram-weights(active ids,fused)" if d["nmats"] > 1
                            else "dram-weights(active ids)")

    for key, mg in moe_groups.items():
        if not mg["avg_row_bytes"]:
            notes.append("expert group %s/M=%d has no 3D GGUF tensor; bytes not derived" % (key[0], key[1]))
    fused = sum(1 for d in dispatches
                if d["class"] in ("matvec", "moe-matvec") and d["nmats"] > 1)
    if fused:
        notes.append("fused gate+up dispatches charged 2 weight matrices "
                     "(%.1f dispatches/token); they stream both the up and the gate tensor"
                     % (fused / divisor))
    if understated:
        notes.append("matvec inventory short by %.1f MiB/token after per-matrix-read "
                     "apportionment: those groups read more matrices than the GGUF holds"
                     % (understated / divisor / 2 ** 20))
    if depth:
        notes.append("depth=%d: the depth prompt (batch matmuls) runs before the window and is excluded" % depth)

    # class + family accumulation over the window
    span_us = (dispatches[-1]["end"] - dispatches[0]["start"]) / 1000.0 if len(dispatches) > 1 else 0.0
    kernel_us = sum(d["duration_us"] for d in dispatches)
    cls = {}
    fam = {}
    for d in dispatches:
        for tbl, key in ((cls, d["class"]), (fam, d["kernel_short"])):
            e = tbl.setdefault(key, {"n": 0, "us": 0.0, "bytes": 0.0, "basis": set(),
                                     "depth": 0, "vgpr": 0, "grid1": 0})
            e["n"] += 1
            e["us"] += d["duration_us"]
            e["bytes"] += d["bytes"]
            e["basis"].add(d["bytes_basis"])
            if d["kernel_short"] in ("mul_mat_vec_q", "mul_mat_vec_f", "mul_mat_vec_q_moe"):
                e["depth"] += 1
            e["vgpr"] = max(e["vgpr"], d["vgpr"])

    # every class records which kernel instantiations produced it: a class
    # name that implies a symbol nobody ran is how the MoE class was
    # mis-attributed to mul_mat_vec_q_moe.
    cls_members = {}
    for d in dispatches:
        mem = cls_members.setdefault(d["class"], {}).setdefault(
            d["kernel_label"], {"n": 0, "us": 0.0, "bytes": 0.0})
        mem["n"] += 1
        mem["us"] += d["duration_us"]
        mem["bytes"] += d["bytes"]

    def row(key, e, members=None):
        dpt = e["n"] / divisor
        ms = e["us"] / divisor / 1000.0
        us_call = e["us"] / e["n"] if e["n"] else 0.0
        bt = e["bytes"] / divisor
        gbps = (bt / (e["us"] / divisor / 1e6)) / 1e9 if e["bytes"] and e["us"] else None
        bound = "latency"
        if us_call <= DISPATCH_BOUND_US:
            bound = "dispatch"
        elif gbps is not None:
            bound = "bandwidth" if gbps >= BW_BOUND_FRAC * PEAK_BW / 1e9 else "compute"
        flag = ""
        if gbps is not None and gbps > PEAK_BW / 1e9:
            flag = "BW>PEAK: measurement error, not a result"
        elif gbps is not None and gbps > 1.05 * 458.0:
            flag = "BW>458 in-situ ceiling"
        if us_call and us_call < DISPATCH_FLOOR_US:
            flag = (flag + "; " if flag else "") + "us/call<1.67 floor"
        out = {
            "name": key,
            "dispatches_per_token": round(dpt, 2),
            "ms_per_token": round(ms, 4),
            "us_per_call": round(us_call, 3),
            "pct_kernel_time": round(100.0 * e["us"] / kernel_us, 2) if kernel_us else 0.0,
            "floor_ms_per_token": round(dpt * DISPATCH_FLOOR_US / 1000.0, 4),
            "excess_ms_per_token": round(max(0.0, ms - dpt * DISPATCH_FLOOR_US / 1000.0), 4),
            "bytes_per_token": int(bt),
            "bytes_per_call": int(bt / dpt) if dpt else 0,
            "gbps": round(gbps, 1) if gbps is not None else None,
            "bytes_basis": ",".join(sorted(e["basis"])) if e["basis"] else "none",
            "bound": bound,
            "flag": flag,
        }
        if members:
            out["kernels"] = [
                {"kernel": k,
                 "dispatches": v["n"],
                 "dispatches_per_token": round(v["n"] / divisor, 2),
                 "us_per_call": round(v["us"] / v["n"], 3),
                 "bytes_per_call": int(v["bytes"] / v["n"]) if v["n"] else 0,
                 "gbps": (round(v["bytes"] / (v["us"] / 1e6) / 1e9, 1)
                          if v["us"] and v["bytes"] else None)}
                for k, v in sorted(members.items(), key=lambda kv: (-kv[1]["n"], kv[0]))]
            out["kernel_count"] = len(members)
        return out

    classes = [row(k, v, cls_members.get(k)) for k, v in cls.items()]
    families = [row(k, v) for k, v in fam.items()]
    classes.sort(key=lambda r: -r["ms_per_token"])
    families.sort(key=lambda r: -r["ms_per_token"])

    totals = {
        "dispatches": len(dispatches),
        "dispatches_per_token": round(len(dispatches) / divisor, 2),
        "kernel_ms_per_token": round(kernel_us / divisor / 1000.0, 4),
        "span_ms_per_token": round(span_us / divisor / 1000.0, 4),
        "gap_ms_per_token": round(max(0.0, span_us - kernel_us) / divisor / 1000.0, 4),
        "floor_ms_per_token": round(len(dispatches) / divisor * DISPATCH_FLOOR_US / 1000.0, 4),
        "matvec_bytes_per_token": int(sum(d["bytes"] for d in dispatches if d["class"] == "matvec") / divisor),
        "moe_bytes_per_token": int(sum(d["bytes"] for d in dispatches if d["class"] == "moe-matvec") / divisor),
        "derived_bytes_gbps": None,
    }
    derived = sum(d["bytes"] for d in dispatches)
    if derived:
        totals["derived_bytes_gbps"] = round(derived / (kernel_us / 1e6) / 1e9, 1)
    if totals["span_ms_per_token"] and totals["gap_ms_per_token"] > 0.25 * totals["span_ms_per_token"]:
        notes.append("graph-free launch gap is %.2f ms/token (%.2f us/launch); hipGraph collapses it"
                     % (totals["gap_ms_per_token"],
                        totals["gap_ms_per_token"] * 1000.0 / max(totals["dispatches_per_token"], 1e-9)))

    recon = {}
    if gguf_info.get("payload_bytes"):
        used = totals["matvec_bytes_per_token"] + totals["moe_bytes_per_token"]
        recon = {
            "payload_bytes": gguf_info["payload_bytes"],
            "weight_bytes_per_token": used,
            "ratio_of_payload": round(used / gguf_info["payload_bytes"], 4),
        }
    return {"classes": classes, "families": families, "totals": totals,
            "reconciliation": recon, "notes": notes}


def _profile_print(prof, limit_families=14):
    t = prof["totals"]
    m = prof["model"]
    print()
    print("model   : %s" % m["path"])
    print("          %s  payload %.2f GiB  %d layers  kv %.2f KiB/token"
          % (m["arch"], m["payload_bytes"] / 2 ** 30, m["block_count"], m["kv_bytes_per_token"] / 1024.0))
    c = prof["config"]
    print("backend : %s (%s)  tokens=%d (divisor %d) depth=%d  ngl=%d ncmoe=%d fa=%s ctk=%s ctv=%s"
          % (prof["backend"], prof["device"], c["tokens"], c["divisor"], c["depth"],
             c["ngl"], c["ncmoe"], c["fa"], c["ctk"], c["ctv"]))
    if prof["env"]:
        print("env     : %s" % " ".join("%s=%s" % kv for kv in sorted(prof["env"].items())))
    print("window  : %d dispatches (%.1f/token), %.3f ms/token of which %.3f ms kernels + %.3f ms gaps"
          % (t["dispatches"], t["dispatches_per_token"], t["span_ms_per_token"],
             t["kernel_ms_per_token"], t["gap_ms_per_token"]))
    if prof["wall"].get("graph_ms_per_token"):
        print("wall    : %.3f ms/token (%.2f t/s) with hipGraph on; graph-free trace above is the dispatch count"
              % (prof["wall"]["graph_ms_per_token"], 1000.0 / prof["wall"]["graph_ms_per_token"]))
    for cname, meta in GEOMETRY_CLASSES.items():
        crow = next((r for r in prof["classes"] if r["name"] == cname), None)
        if not crow:
            continue
        names = [m["kernel"] for m in crow.get("kernels") or []]
        print()
        print("NOTE: class %r is a geometry split (%s), not a kernel family;"
              % (cname, meta["why"]))
        print("      its members are %s." % (", ".join(names) if names else "no symbols"))
        if not any(meta["implied"] in n for n in names):
            print("      %s is NOT among them -- 0 dispatches in this window; read the class"
                  % meta["implied"])
            print("      as its member symbols, not as %s." % meta["implied"])
    print()
    print("%-12s %9s %8s %8s %7s %8s %8s %9s  %s"
          % ("class", "disp/tok", "ms/tok", "us/call", "%ker", "floor_ms", "GiB/tok", "GB/s", "bound"))
    print("-" * 108)
    for r in prof["classes"]:
        print("%-12s %9.1f %8.3f %8.2f %6.1f%% %8.3f %8.3f %9s  %s%s"
              % (r["name"], r["dispatches_per_token"], r["ms_per_token"], r["us_per_call"],
                 r["pct_kernel_time"], r["floor_ms_per_token"], r["bytes_per_token"] / 2 ** 30,
                 "%.0f" % r["gbps"] if r["gbps"] is not None else "-", r["bound"],
                 ("   [" + r["flag"] + "]") if r["flag"] else ""))
    print("-" * 108)
    print("%-12s %9.1f %8.3f %8s %6.1f%% %8.3f %8.3f %9s"
          % ("TOTAL", t["dispatches_per_token"], t["kernel_ms_per_token"], "",
             100.0, t["floor_ms_per_token"],
             (t["matvec_bytes_per_token"] + t["moe_bytes_per_token"]) / 2 ** 30,
             "%.0f" % t["derived_bytes_gbps"] if t["derived_bytes_gbps"] else "-"))
    if prof["reconciliation"]:
        r = prof["reconciliation"]
        if t["moe_bytes_per_token"]:
            print("weight bytes/token = %.3f GiB = %.1f%% of the %.2f GiB payload; the rest is the idle "
                  "expert table (only the routed experts are read) plus the embedding table"
                  % (r["weight_bytes_per_token"] / 2 ** 30, 100.0 * r["ratio_of_payload"],
                     r["payload_bytes"] / 2 ** 30))
        else:
            print("weight bytes/token = %.3f GiB = %.1f%% of the %.2f GiB payload "
                  "(the rest is the embedding table and tensors this graph never reads)"
                  % (r["weight_bytes_per_token"] / 2 ** 30, 100.0 * r["ratio_of_payload"],
                     r["payload_bytes"] / 2 ** 30))
    print()
    print("class membership (kernel symbols that actually ran)")
    print("%-12s %-30s %9s %7s %9s %7s"
          % ("class", "kernel", "disp/tok", "disp", "MiB/call", "GB/s"))
    print("-" * 80)
    for r in prof["classes"]:
        kers = r.get("kernels") or []
        if not kers:
            print("%-12s %-30s %9.1f %7s" % (r["name"], "(no symbol in window)", r["dispatches_per_token"], "-"))
            continue
        for m in kers[:4]:
            lbl = m["kernel"]
            if len(lbl) > 30:
                lbl = lbl[:27] + "..."
            print("%-12s %-30s %9.1f %7d %9.3f %7s"
                  % (r["name"], lbl, m["dispatches_per_token"], m["dispatches"],
                     m["bytes_per_call"] / 2 ** 20,
                     "%.0f" % m["gbps"] if m["gbps"] is not None else "-"))
        if len(kers) > 4:
            print("%-12s %-30s %9s %7s"
                  % ("", "(+%d more instantiations)" % (len(kers) - 4), "", ""))
    print()
    print("top families")
    print("%-40s %9s %8s %8s %7s  %s" % ("kernel", "disp/tok", "ms/tok", "us/call", "%ker", "bound"))
    print("-" * 96)
    for r in prof["families"][:limit_families]:
        print("%-40s %9.1f %8.3f %8.2f %6.1f%%  %s%s"
              % (r["name"][:40], r["dispatches_per_token"], r["ms_per_token"], r["us_per_call"],
                 r["pct_kernel_time"], r["bound"], ("  [" + r["flag"] + "]") if r["flag"] else ""))
    if prof["notes"]:
        print()
        for n in prof["notes"]:
            print("note: %s" % n)


def _profile_wall_ms(backend, cfg):
    """Graph-enabled wall time (ms/token) of the same config, through the guard."""
    out = _lb_command(backend, cfg, graphs=True)
    try:
        proc = subprocess.run(out, capture_output=True, text=True, timeout=3600, env=cfg["env_run"])
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"\|\s*tg(\d+)(?:\s*@\s*d\d+)?\s*\|\s*([\d.]+)", proc.stdout or "")
    if not m:
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        err = (proc.stderr or "").strip().splitlines()[-3:]
        print("wall run produced no tg row (rc=%d); stdout tail=%s stderr tail=%s"
              % (proc.returncode, tail, err))
        return None
    ts = float(m.group(2))
    return 1000.0 / ts if ts else None


def _lb_command(backend, cfg, graphs):
    binpath = PROFILE_BIN[backend] / "llama-bench"
    args = [str(binpath), "-m", str(cfg["model"]), "-p", "0", "-n", str(cfg["tokens"]),
            "-r", str(cfg["reps"]), "-d", str(cfg["depth"]), "-ngl", str(cfg["ngl"]),
            "-fa", str(cfg["fa"]), "-ctk", str(cfg["ctk"]), "-ctv", str(cfg["ctv"]),
            "-dev", cfg["dev"], "-o", "md"]
    if cfg["ncmoe"]:
        args += ["-ncmoe", str(cfg["ncmoe"])]
    if cfg["fitt"]:
        args += ["-fitt", str(cfg["fitt"])]
    if cfg["ot"]:
        args += ["-ot", cfg["ot"]]
    args += cfg["extra"]
    if graphs:
        return [str(ROOT / "scripts" / "bench-exclusive.sh")] + args
    return args


def _profile_config(args):
    model = PROFILE_MODELS.get(args.model, Path(args.model))
    extra = shlex.split(args.extra or "")
    env_run = dict(os.environ)
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env_run[k] = v
    cfg = {
        "model": Path(model),
        "tokens": args.tokens,
        "reps": args.reps,
        "depth": args.depth,
        "ngl": args.ngl,
        "ncmoe": args.ncmoe,
        "fa": args.flash_attn,
        "ctk": args.cache_type_k,
        "ctv": args.cache_type_v,
        "fitt": args.fitt,
        "ot": args.ot,
        "dev": args.dev or PROFILE_DEV[args.backend],
        "extra": extra,
        "env_run": env_run,
        "pmc": args.pmc,
        "no_fusion": args.no_fusion,
        "label": args.label,
    }
    return cfg


def _profile_run(backend, cfg, outdir, dry_run=False):
    """Trace one bench run through the exclusive guard. Returns (csvs, wall_ms)."""
    prof, path = _profiler()
    if not prof:
        print("no ROCm profiler installed; run `probe` for the install command.")
        return None, []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(cfg["env_run"])
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if cfg["no_fusion"]:
        env["GGML_CUDA_DISABLE_FUSION"] = "1"
    print("GGML_CUDA_DISABLE_GRAPHS=1 (hipGraph nodes are not HSA dispatch packets)")
    cmd = [str(ROOT / "scripts" / "bench-exclusive.sh"), path, "--kernel-trace"]
    if cfg["pmc"]:
        cmd += ["--pmc", cfg["pmc"]]
    cmd += ["--output-format", "csv", "-d", str(outdir), "--"] + _lb_command(backend, cfg, graphs=False)
    envprefix = " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("GGML_") or k.startswith("HIP_"))
    print(f"DRY-RUN: {envprefix} {shlex.join(cmd)}" if dry_run else " ".join(cmd))
    if dry_run:
        return None, []
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    csvs = sorted(outdir.rglob("*kernel*trace*.csv")) or sorted(outdir.rglob("*.csv"))
    if not csvs:
        print("no trace CSV produced (rc=%d) in %s" % (proc.returncode, outdir))
        if proc.stderr:
            print(proc.stderr[-800:])
        return None, []
    if proc.returncode != 0:
        print("warning: profiler exit %d" % proc.returncode)
        if proc.stderr:
            print(proc.stderr[-400:])
    return proc, csvs


def _parse_csv(path):
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        if not rows:
            return [], []
        return rdr.fieldnames or [], rows


def _row_duration_us(row, fields):
    lower = {f.lower(): f for f in fields}
    df = next((lower[k] for k in lower if "duration" in k), None)
    if df:
        try:
            dur = float(row[df])
        except (TypeError, ValueError):
            dur = 0.0
        # trace CSV durations are nanoseconds; convert to microseconds
        return dur / 1000.0
    sf = next((lower[k] for k in lower if "start" in k and "timestamp" in k), None)
    ef = next((lower[k] for k in lower if "end" in k and "timestamp" in k), None)
    if sf and ef:
        try:
            return (float(row[ef]) - float(row[sf])) / 1000.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _profile_defaults():
    """Flag defaults shared by `profile` and the legacy `trace` entry point."""
    return dict(model="dense", backend="hip", tokens=16, reps=1, depth=0, ngl=99, ncmoe=0,
                flash_attn="1", cache_type_k="q8_0", cache_type_v="q5_0", fitt=128, ot=None,
                dev=None, extra=None, env=[], label=None, out=None, json=None, pmc=None,
                no_fusion=False, no_wall=False, compare=None, ledger=None, census=None,
                reuse=None, wall_ms=None, dry_run=False)


def cmd_trace(args):
    """`trace` == `profile` with the census-ledger side-car and no wall run."""
    kt = getattr(args, "kernel_trace", None) or getattr(args, "kernel_trace_pos", None)
    if kt:
        args.kernel_trace = kt
        return _audit_trace(args)
    ns = argparse.Namespace(**_profile_defaults())
    ns.tokens = args.tokens
    ns.depth = args.depth
    ns.out = args.out
    ns.pmc = args.pmc
    ns.no_fusion = args.no_fusion
    ns.no_wall = True
    ns.ledger = args.ledger
    ns.census = args.census
    ns.dry_run = getattr(args, "dry_run", False)
    return cmd_profile(ns)


def cmd_profile(args):
    if args.compare:
        return _profile_compare(*args.compare)

    cfg = _profile_config(args)
    if not cfg["model"].exists():
        print("missing model: %s" % cfg["model"])
        return 2
    wall_ms = getattr(args, "wall_ms", None)
    reuse = getattr(args, "reuse", None)
    dry_run = getattr(args, "dry_run", False)
    if wall_ms:
        print("graph-enabled wall: %.3f ms/token (%.2f t/s) [from --wall-ms]" % (wall_ms, 1000.0 / wall_ms))
    elif not args.no_wall and not reuse and not dry_run:
        wall_ms = _profile_wall_ms(args.backend, cfg)
        if wall_ms:
            print("graph-enabled wall: %.3f ms/token (%.2f t/s)" % (wall_ms, 1000.0 / wall_ms))
        else:
            print("graph-enabled wall: not parsed from llama-bench output")

    if reuse:
        outdir = Path(reuse)
        csvs = sorted(outdir.rglob("*kernel*trace*.csv")) or sorted(outdir.rglob("*.csv"))
        if not csvs:
            print("no trace CSV under %s" % outdir)
            return 2
        print("reusing trace: %s" % outdir)
    else:
        outdir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="gfx1030_profile_"))
        proc, csvs = _profile_run(args.backend, cfg, outdir, dry_run=getattr(args, "dry_run", False))
        if getattr(args, "dry_run", False):
            return 0
        if csvs is None:
            return 1

    recs = _profile_dispatches(csvs)
    if not recs:
        print("trace CSVs had no kernel rows:", [p.name for p in csvs])
        return 1
    win, divisor, wnote = _profile_window(recs, cfg["tokens"])
    if wnote:
        print("window: %s" % wnote)
    if divisor != cfg["tokens"] + 1:
        print("divisor: %d decode blocks (not tokens+1; the depth prompt separates the warmup)" % divisor)

    idx2d, idx3d, gguf_info = _profile_gguf_index(cfg["model"])
    agg = _profile_aggregate(win, divisor, idx2d, idx3d, gguf_info, cfg["depth"])

    kv = gguf_info.get("kv") or {}
    kvinfo = audit_kv(kv, list(_read_gguf(cfg["model"])[3]), cfg["depth"] or 4096,
                      cfg["ctk"], cfg["ctv"]) if kv else {}
    prof = {
        "schema": PROFILE_SCHEMA,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": cfg["label"] or ("%s-%s-%s-d%d" % (cfg["model"].stem[:24], args.backend, cfg["fa"], cfg["depth"])),
        "model": {
            "path": str(cfg["model"]),
            "arch": gguf_info.get("arch", "?"),
            "payload_bytes": gguf_info.get("payload_bytes", 0),
            "tensors": gguf_info.get("tensors", 0),
            "block_count": kv.get("%s.block_count" % gguf_info.get("arch", ""), 0),
            "kv_bytes_per_token": kvinfo.get("bytes_per_token", 0),
        },
        "backend": args.backend,
        "device": cfg["dev"],
        "config": dict({k: cfg[k] for k in ("tokens", "reps", "depth", "ngl", "ncmoe", "fa",
                                            "ctk", "ctv", "fitt", "ot", "extra")},
                       divisor=divisor),
        "env": {k: v for k, v in cfg["env_run"].items() if k.startswith("GGML_") or k.startswith("HIP_")},
        "trace": {"dir": str(outdir),
                  "window_note": wnote, "dispatches_total": len(recs)},
        "wall": {"graph_ms_per_token": wall_ms,
                 "graph_ts": (1000.0 / wall_ms) if wall_ms else None},
        "totals": agg["totals"],
        "classes": agg["classes"],
        "families": agg["families"],
        "reconciliation": agg["reconciliation"],
        "notes": agg["notes"],
    }
    _profile_print(prof)
    out_json = Path(args.json) if args.json else outdir / "profile.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(prof, f, indent=1)
        f.write("\n")
    print("json: %s" % out_json)
    if getattr(args, "ledger", None):
        env = dict(cfg["env_run"])
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        _write_ledger(args.ledger, csvs, outdir, cfg["tokens"], cfg["depth"], env,
                      getattr(args, "census", None))
    return 0


def _profile_compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    for tag, p in (("A", a), ("B", b)):
        if p.get("schema") != PROFILE_SCHEMA:
            print("%s: unexpected schema %r (want %s)" % (tag, p.get("schema"), PROFILE_SCHEMA))
            return 2
    print("A = %s  %s  %.2f GiB payload  tokens=%d depth=%d"
          % (a.get("label"), Path(a["model"]["path"]).name, a["model"]["payload_bytes"] / 2 ** 30,
             a["config"]["tokens"], a["config"]["depth"]))
    print("B = %s  %s  %.2f GiB payload  tokens=%d depth=%d"
          % (b.get("label"), Path(b["model"]["path"]).name, b["model"]["payload_bytes"] / 2 ** 30,
             b["config"]["tokens"], b["config"]["depth"]))
    wa = a["wall"].get("graph_ms_per_token")
    wb = b["wall"].get("graph_ms_per_token")
    print()
    print("wall %s: A %s ms/token   B %s ms/token   delta %s"
          % ("(hipGraph on)", "%.3f" % wa if wa else "-", "%.3f" % wb if wb else "-",
             "%+.3f" % (wb - wa) if wa and wb else "-"))
    print("kernel time: A %.3f   B %.3f ms/token"
          % (a["totals"]["kernel_ms_per_token"], b["totals"]["kernel_ms_per_token"]))
    print("dispatches : A %.1f   B %.1f per token"
          % (a["totals"]["dispatches_per_token"], b["totals"]["dispatches_per_token"]))
    print()
    ca = {c["name"]: c for c in a["classes"]}
    cb = {c["name"]: c for c in b["classes"]}
    names = sorted(set(ca) | set(cb), key=lambda n: -(cb.get(n, {}).get("ms_per_token", 0.0)
                                                     + ca.get(n, {}).get("ms_per_token", 0.0)))
    print("%-12s %8s %8s %8s %9s %9s %9s %9s"
          % ("class", "A_ms", "B_ms", "d_ms", "A_disp", "B_disp", "A_GB/s", "B_GB/s"))
    print("-" * 86)
    for n in names:
        x, y = ca.get(n), cb.get(n)
        print("%-12s %8.3f %8.3f %8.3f %9.1f %9.1f %9s %9s"
              % (n,
                 x["ms_per_token"] if x else 0.0, y["ms_per_token"] if y else 0.0,
                 (y["ms_per_token"] if y else 0.0) - (x["ms_per_token"] if x else 0.0),
                 x["dispatches_per_token"] if x else 0.0, y["dispatches_per_token"] if y else 0.0,
                 "%.0f" % x["gbps"] if x and x["gbps"] else "-",
                 "%.0f" % y["gbps"] if y and y["gbps"] else "-"))
    print("-" * 86)
    ta, tb = a["totals"], b["totals"]
    print("%-12s %8.3f %8.3f %8.3f %9.1f %9.1f"
          % ("TOTAL", ta["kernel_ms_per_token"], tb["kernel_ms_per_token"],
             tb["kernel_ms_per_token"] - ta["kernel_ms_per_token"],
             ta["dispatches_per_token"], tb["dispatches_per_token"]))
    print()
    ba = ta["matvec_bytes_per_token"] + ta["moe_bytes_per_token"]
    bb = tb["matvec_bytes_per_token"] + tb["moe_bytes_per_token"]
    print("weight bytes/token: A %.2f GiB   B %.2f GiB" % (ba / 2 ** 30, bb / 2 ** 30))
    if tb["kernel_ms_per_token"]:
        print("B moves %.2fx the bytes/token of A at %.2fx the kernel time"
              % (bb / max(ba, 1), tb["kernel_ms_per_token"] / max(ta["kernel_ms_per_token"], 1e-9)))
    print("note: A and B are different models and different quantisations; ratios are not speedups.")
    return 0


def _load_census_nodes(path):
    nodes = []
    if not path:
        return nodes
    try:
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                o=json.loads(line)
                # duplicate "type" key: row type is lost; identify node rows by i+op_name+role
                if "i" in o and "op_name" in o and "role" in o:
                    nodes.append(o)
    except Exception as e:
        print("census load error:", e)
    return nodes


def _kernel_short(name):
    part = name.split("(")[0].strip()
    if part.startswith("void "):
        part = part[5:].strip()
    base = part.split("<")[0].strip()
    toks = base.split()
    base = toks[-1] if toks else base
    return base


MATVEC_FAMILIES = ("mul_mat_vec_q", "mul_mat_vec_f")
FAMILY_ALIAS = {
    "fwht_cuda": "gated_delta_net_cuda",
}


def _dispatch_m(d):
    """Rows per matvec dispatch: one workgroup-quantum per output row."""
    grid = d.get("grid") or [0, 0, 0]
    wg = d.get("workgroup") or [0, 0, 0]
    if grid and wg and grid[0] and wg[0] and grid[0] % wg[0] == 0:
        return grid[0] // wg[0]
    return 0


def _kernel_weight_type(name):
    """Weight dtype of a matvec instantiation, read from the kernel symbol."""
    m = re.search(r"\(ggml_type\)\s*(\d+)", name or "")
    if m:
        layout = TYPE_LAYOUT.get(int(m.group(1)))
        return layout[0].lower() if layout else ""
    low = (name or "").lower()
    for cpp, gg in (("__hip_bfloat16", "bf16"), ("half", "f16"), ("float", "f32")):
        if cpp in low:
            return gg
    return ""


def _node_weight_type(n):
    w = n.get("weight")
    wt = w.get("type_name") if isinstance(w, dict) else None
    return wt or n.get("type_name", "") or ""


def _node_m(n):
    """Number of weight rows (matvec M) of a census node."""
    w = n.get("weight")
    wname = w.get("name") if isinstance(w, dict) else None
    for src in (n.get("src") or []):
        if wname and src.get("name") == wname:
            ne = src.get("ne") or []
            return int(ne[1]) if len(ne) > 1 and ne[1] else 0
    return 0


def _write_ledger(path, csvs, outdir, tokens, depth, env, census_path):
    # collect dispatches in time order
    dispatches = []
    for cpath in csvs:
        fields, rows = _parse_csv(cpath)
        if not rows:
            continue
        lower = {f.lower(): f for f in fields}
        kf = next((lower[k] for k in lower if "kernel" in k and "name" in k), None)
        if not kf:
            continue
        sf = next((lower[k] for k in lower if "start" in k and "timestamp" in k), None)
        ef = next((lower[k] for k in lower if "end" in k and "timestamp" in k), None)
        gxf = next((lower[k] for k in lower if "grid_size_x" in k), None)
        gyf = next((lower[k] for k in lower if "grid_size_y" in k), None)
        gzf = next((lower[k] for k in lower if "grid_size_z" in k), None)
        wxf = next((lower[k] for k in lower if "workgroup_size_x" in k), None)
        wyf = next((lower[k] for k in lower if "workgroup_size_y" in k), None)
        wzf = next((lower[k] for k in lower if "workgroup_size_z" in k), None)
        vgf = next((lower[k] for k in lower if "vgpr_count" in k), None)
        ldsf = next((lower[k] for k in lower if "lds_block_size" in k), None)
        for row in rows:
            name = row[kf]
            dur = _row_duration_us(row, fields)
            start = float(row[sf]) if sf and row.get(sf) not in (None, "") else 0.0
            end = float(row[ef]) if ef and row.get(ef) not in (None, "") else 0.0
            dispatches.append({
                "kernel": name,
                "kernel_short": _kernel_short(name),
                "kernel_label": _kernel_label(name),
                "duration_us": dur,
                "start": start,
                "end": end,
                "grid": [int(row.get(gxf,0) or 0), int(row.get(gyf,0) or 0), int(row.get(gzf,0) or 0)],
                "workgroup": [int(row.get(wxf,0) or 0), int(row.get(wyf,0) or 0), int(row.get(wzf,0) or 0)],
                "vgpr": int(row.get(vgf,0) or 0) if vgf else 0,
                "lds": int(row.get(ldsf,0) or 0) if ldsf else 0,
            })

    # sort by start timestamp
    dispatches.sort(key=lambda d: d["start"])
    # Blocker 4: drop load-time kernels and warmup token
    dispatches = [d for d in dispatches if "__amd_rocclr" not in d["kernel"] and "fillBuffer" not in d["kernel"]]
    if tokens > 0 and len(dispatches) > tokens:
        warmup_n = len(dispatches) // (tokens + 1)
        dispatches = dispatches[warmup_n:]

    # gap
    prev_end = None
    for d in dispatches:
        if prev_end is None:
            d["gap_us"] = 0.0
        else:
            g = (d["start"] - prev_end) / 1000.0
            d["gap_us"] = g
        prev_end = d["end"]

    # census nodes
    nodes = _load_census_nodes(census_path)
    # real nodes (not views) in order; keep original i
    skipped_ops = {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "NONE"}
    real_nodes = []
    for n in nodes:
        if n.get("op_name") in skipped_ops:
            continue
        real_nodes.append(n)

    for d in dispatches:
        d["_node"] = None

    # Token boundaries: the vocab projection matvec runs once per token and ends
    # the graph. Used to (a) drop the partial warmup head the fixed-count warmup
    # trim leaves behind and (b) reset the join cursors per token, because a
    # global modulo would shift every group by a constant and mislabel shapes.
    mv_ms = [_dispatch_m(d) for d in dispatches if d["kernel_short"] in MATVEC_FAMILIES]
    token_m = max(mv_ms) if mv_ms else 0
    flags = [bool(token_m) and d["kernel_short"] in MATVEC_FAMILIES and _dispatch_m(d) == token_m
             for d in dispatches]
    if token_m and True in flags:
        first = flags.index(True)
        if first + 1 < len(dispatches):
            print(f"dropped {first + 1} leading dispatches (partial warmup token)")
            dispatches = dispatches[first + 1:]
            flags = flags[first + 1:]
    tok = 0
    for d, is_end in zip(dispatches, flags):
        d["_tok"] = tok
        if is_end:
            tok += 1
    print(f"tokens in ledger: {tok} (boundary = matvec with M={token_m})")
    # Global multiset matching: group dispatches and nodes by kernel family,
    # then assign round-robin. This yields ~100% join for launch kernels.
    reverse_map = {}
    ok_map = {
        "MUL_MAT": "mul_mat_vec_q", "RMS_NORM": "rms_norm_f32",
        "UNARY": "unary_op_kernel", "CPY": "cpy_scalar",
        "GET_ROWS": "k_get_rows_float", "ROPE": "rope_multi",
        "FLASH_ATTN_EXT": "flash_attn_ext_gqa_gfx1030",
        "SSM_CONV": "ssm_conv_f32", "GATED_DELTA_NET": "gated_delta_net_cuda",
        "CONT": "concat_cont", "SCALE": "scale_f32", "CONCAT": "concat_cont",
        "GLU": "unary_gated_op_kernel", "ADD": "k_bin_bcast", "MUL": "k_bin_bcast", "SET_ROWS": "set_rows_quant",
    }
    for op, ker in ok_map.items():
        reverse_map.setdefault(ker, op)
    # extra dispatch families that attribute to existing nodes
    for extra in ("quantize_q8_1", "mul_mat_vec_f", "fwht_cuda", "set_rows_quant"):
        reverse_map[extra] = extra
    reverse_map["mul_mat_vec_f"] = "mul_mat_vec_f"

    # group nodes by mapped kernel (split mul_mat by weight dtype)
    nodes_by_kernel = collections.defaultdict(list)
    for n in real_nodes:
        k = ok_map.get(n.get("op_name"))
        if k == "mul_mat_vec_q":
            w = n.get("weight")
            wt = (w.get("type_name") if isinstance(w, dict) else n.get("type_name")) or ""
            if "bf16" in wt.lower() or "f16" in wt.lower():
                k = "mul_mat_vec_f"
        if k is not None:
            nodes_by_kernel[k].append(n)

    # For each kernel group, assign dispatches to nodes.
    # matvec kernels are matched by observable shape (weight type from the kernel
    # template argument, M from grid/workgroup); everything else keeps a
    # per-family round-robin over nodes of the same family.
    node_groups = collections.defaultdict(list)
    node_groups_by_m = collections.defaultdict(list)
    node_groups_any = collections.defaultdict(list)
    for fam, ns in nodes_by_kernel.items():
        for n in ns:
            node_groups[(fam, _node_weight_type(n).lower(), _node_m(n))].append(n)
            node_groups_by_m[(fam, _node_m(n))].append(n)
            node_groups_any[fam].append(n)
    cursors = collections.Counter()

    def _pick(key, ns, tok=None):
        key = (key, tok)
        i = cursors[key]
        cursors[key] += 1
        return ns[i % len(ns)]

    def _family(d):
        low = (d.get("kernel") or "").lower()
        for k in reverse_map:
            if k in low:
                return FAMILY_ALIAS.get(k, k)
        return None

    unmatched = []
    pending_quantize = []
    for d in dispatches:
        fam = _family(d)
        if fam is None:
            d["_node"] = None
            d["_match"] = "no kernel family"
            unmatched.append((d, "no kernel family"))
            continue
        if fam in MATVEC_FAMILIES:
            wt = _kernel_weight_type(d["kernel"]).lower()
            m = _dispatch_m(d)
            key = (fam, wt, m)
            ns = node_groups.get(key)
            reason = ""
            if ns:
                d["_node"] = _pick(key, ns, d["_tok"])
            else:
                key = (fam, m)
                ns = node_groups_by_m.get(key)
                if ns:
                    d["_node"] = _pick(key, ns, d["_tok"])
                    reason = f"weight-type miss (wt={wt or '?'} M={m})"
                else:
                    ns = node_groups_any.get(fam)
                    if ns:
                        d["_node"] = _pick((fam, "any"), ns, d["_tok"])
                        reason = f"shape miss (wt={wt or '?'} M={m})"
                    else:
                        d["_node"] = None
                        reason = f"no nodes for family {fam}"
            d["_match"] = reason or "shape"
            # quantize_q8_1 is always issued immediately before its matvec
            for q in pending_quantize:
                q["_node"] = d["_node"]
                q["_match"] = "expansion of preceding matvec"
            pending_quantize = []
            if reason:
                unmatched.append((d, reason))
            continue
        if fam == "quantize_q8_1":
            pending_quantize.append(d)
            continue
        ns = nodes_by_kernel.get(fam)
        if not ns:
            d["_node"] = None
            d["_match"] = f"no nodes for family {fam}"
            unmatched.append((d, f"no nodes for family {fam}"))
            continue
        d["_node"] = _pick((fam, "any", 0), ns, d["_tok"])
        d["_match"] = "family"
    for q in pending_quantize:
        q["_node"] = None
        q["_match"] = "trailing quantize without matvec"
        unmatched.append((q, "trailing quantize without matvec"))

    # coverage
    matched_total = sum(1 for d in dispatches if d.get("_node") is not None)
    print(f"join coverage: {matched_total}/{len(dispatches)} = {100.0*matched_total/len(dispatches) if dispatches else 0:.1f}%")
    print(f"unmatched dispatches: {len(unmatched)}")
    if unmatched:
        reasons = collections.Counter(r for _d, r in unmatched)
        for r, n in reasons.most_common():
            print(f"  {n:6d}  {r}")
        for d, r in unmatched[:10]:
            print(f"    e.g. {d.get('kernel_short')} grid={d.get('grid')} :: {r}")
    by_match = collections.Counter(d.get("_match") for d in dispatches)
    for how, n in by_match.most_common():
        print(f"  join by {how}: {n}")

    # write ledger
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with open(out, "w") as f:
        f.write(json.dumps({
            "type": "meta",
            "tokens": tokens,
            "divisor": tokens + 1,
            "measured_tokens": tokens,
            "depth": depth,
            "census": str(census_path) if census_path else None,
            "join_coverage_pct": round(100.0 * matched_total / len(dispatches), 2) if dispatches else 0.0,
            "wall_ms_per_token": None,
            "note": "wall_ms_per_token is filled by a graph-enabled llama-bench run; "
                    "this trace runs with GGML_CUDA_DISABLE_GRAPHS=1",
        }) + "\n")
        for i, d in enumerate(dispatches):
            nd = d["_node"]
            if nd is not None:
                matched += 1
            node_idx = nd.get("i", -1) if nd else -1
            op_name = nd.get("op_name", "") if nd else ""
            weight = nd.get("weight") if nd else None
            weight_type = (weight.get("type_name") if isinstance(weight, dict) and "type_name" in weight
                           else (weight.get("type") if isinstance(weight, dict) else None))
            if weight_type is None and nd:
                weight_type = nd.get("type_name", "")
            role = nd.get("role", "") if nd else ""
            layer = nd.get("layer", None) if nd else None
            # K = reduction dim, M = number of weight rows (both from the weight tensor)
            kval = 0
            mval = 0
            wbytes = 0
            if nd and "MUL_MAT" in op_name:
                wname = weight.get("name") if isinstance(weight, dict) else None
                for src in (nd.get("src") or []):
                    if wname and src.get("name") == wname:
                        ne = src.get("ne") or [0, 0, 0, 0]
                        kval = int(ne[0]) if ne else 0
                        mval = int(ne[1]) if len(ne) > 1 and ne[1] else 0
                        wbytes = int(src.get("nbytes") or 0)
                        break
            elif nd:
                ne = nd.get("ne") or [0]
                kval = int(ne[0]) if ne else 0
            # per-dispatch traffic: a matvec streams its whole weight matrix from
            # DRAM once; quantize_q8_1 reads/writes one activation row; everything
            # else touches its own tensor once and stays L2-resident (a 0.9*DRAM
            # bandwidth check does not apply to those).
            nbytes = 0
            row_bytes = 0
            ks = d["kernel_short"]
            if nd and ks in MATVEC_FAMILIES:
                nbytes = wbytes
                basis = "dram"
                if mval:
                    row_bytes = nbytes // mval
            elif nd and ks == "quantize_q8_1":
                nbytes = int(kval * 4 + (kval // 32) * 36)
                basis = "l2"
            elif nd:
                nbytes = int(nd.get("nbytes") or 0)
                basis = "l2"
            else:
                basis = "none"
            row = {
                "type": "dispatch",
                "i": i,
                "kernel": d["kernel"],
                "kernel_short": d["kernel_short"],
                "duration_us": d["duration_us"],
                "gap_us": d["gap_us"],
                "grid": d["grid"],
                "workgroup": d["workgroup"],
                "vgpr": d["vgpr"],
                "lds": d["lds"],
                "node_idx": node_idx,
                "op_name": op_name,
                "weight_type": weight_type,
                "role": role,
                "layer": layer,
                "bytes": nbytes,
                "row_bytes": row_bytes,
                "bytes_basis": basis,
                "k": kval,
                "m": mval,
                "match": d.get("_match", ""),
            }
            f.write(json.dumps(row) + "\n")
    print(f"ledger written: {out} ({len(dispatches)} dispatches, {matched} matched = {100.0*matched/len(dispatches) if dispatches else 0:.1f}%)")


MEM_OPS = ("flat_load", "flat_store", "global_load", "global_store", "s_load", "s_store",
           "buffer_load", "buffer_store", "ds_read", "ds_write", "scratch_load", "scratch_store")
BIT_OPS = ("v_lshr", "v_lshl", "v_and", "v_or", "v_xor", "v_bfe", "v_bfi", "v_perm",
           "v_alignbit", "v_alignbyte", "s_lshr", "s_lshl", "s_and", "s_bfe", "s_mov", "v_mov")
MATH_OPS = ("v_dot", "v_mac", "v_fma", "v_mad", "v_mul", "v_sub", "v_add", "v_cndmask",
            "v_cmp", "v_max", "v_min", "v_cvt", "v_med3", "v_lerp")


def isa_scan(path=ISA_S):
    if not path.exists():
        return {}
    out, cur, counts = {}, None, None
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith(".type") and s.endswith("@function"):
            cur = s.split()[1].split(",")[0]
            counts = {"mem": 0, "bit": 0, "math": 0, "total": 0}
            out[cur] = counts
            continue
        if cur is None or not line.startswith("\t") or s.startswith(".") or not s:
            continue
        op = s.split()[0].split("//")[0]
        counts["total"] += 1
        if op.startswith("v_"):
            if op in ("v_lshrrev_b32", "v_lshlrev_b32"):
                counts["bit"] += 1
                continue
        if any(op.startswith(m) for m in MEM_OPS):
            counts["mem"] += 1
        elif any(op.startswith(m) for m in BIT_OPS):
            counts["bit"] += 1
        elif any(op.startswith(m) for m in MATH_OPS):
            counts["math"] += 1
    return out


def isa_decode(sym):
    m = re.search(r"mul_mat_vec_qIL9ggml_type(\d+)ELi(\d+)E", sym)
    if not m:
        return None
    tid, ncols = int(m.group(1)), int(m.group(2))
    tags = "".join(re.findall(r"Lb(\d)E", sym))
    return GGML_TYPE.get(tid, f"type{tid}"), ncols, tags


def cmd_isa(args):
    scan = isa_scan()
    if not scan:
        print(f"ERROR: no device assembly at {ISA_S}")
        print("       run a --save-temps build first")
        return 1

    name_by_id = {12: "Q4_K", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS"}

    rows = {}
    for sym, c in scan.items():
        d = isa_decode(sym)
        if d is None or d[1] != 1 or d[2] not in ("00", "000"):
            continue
        mt = re.match(r"type(\d+)$", d[0])
        name = name_by_id.get(int(mt.group(1))) if mt else d[0]
        if name in TYPES:
            rows[name] = c

    base = {}
    if ISA_BASE.exists():
        try:
            base = json.loads(ISA_BASE.read_text())
        except Exception:
            base = {}

    print(f"{'type':9s} {'instr':>7s} {'mem':>6s} {'bit':>6s} {'math':>6s} {'bit%':>6s}  verdict")
    warnings = []
    for t in TYPES:
        c = rows.get(t)
        if not c or not c["total"]:
            print(f"{t:9s} {'n/a':>7s}")
            continue
        bit_pct = 100.0 * c["bit"] / c["total"]
        b = base.get(t)
        if b:
            dbit = 100.0 * (c["bit"] - b["bit"]) / max(b["bit"], 1)
            dmem = 100.0 * (c["mem"] - b["mem"]) / max(b["mem"], 1)
            dtot = 100.0 * (c["total"] - b["total"]) / max(b["total"], 1)
            if dbit > 5.0 and dtot >= 0.0:
                v = f"ALU-REGRESSION bit +{dbit:.0f}% total {dtot:+.0f}%"
                warnings.append(t)
            else:
                v = f"bit {dbit:+.0f}% mem {dmem:+.0f}% tot {dtot:+.0f}%"
        else:
            v = "baseline"
        print(f"{t:9s} {c['total']:7d} {c['mem']:6d} {c['bit']:6d} {c['math']:6d} {bit_pct:5.1f}%  {v}")

    if args.save_baseline:
        ISA_BASE.write_text(json.dumps({t: rows[t] for t in rows if t in rows}, indent=1))
        print(f"\nsaved baseline -> {ISA_BASE}")

    for t in warnings:
        c = rows[t]
        print(f"\nWARNING: {t} is ALU-bound via unpack logic "
              f"({100.0*c['bit']/c['total']:.0f}% bit-manipulation, {c['mem']} memory ops) "
              f"- memory optimisations will not pay off. Reject before benchmarking.")
    return 1 if (warnings and args.strict) else 0


def audit_builds():
    out = []
    for cache in sorted(ROOT.glob("build*/CMakeCache.txt")):
        txt = cache.read_text(errors="replace")
        hip = _flag(txt, "GGML_HIP") in ("ON", "TRUE")
        no_vmm = _flag(txt, "GGML_HIP_NO_VMM")
        if not hip:
            vmm = None
        elif no_vmm in ("ON", "TRUE"):
            vmm = "hipMalloc - VMM disabled (GGML_HIP_NO_VMM=ON)"
        elif no_vmm in ("OFF", "FALSE"):
            vmm = "explicit VMM pools (GGML_HIP_NO_VMM=OFF)"
        else:
            vmm = "explicit VMM pools (flag unset -> VMM active)"
        out.append({
            "dir": cache.parent.name,
            "backend": "HIP" if hip else "VULKAN",
            "targets": _var(txt, "AMDGPU_TARGETS"),
            "build_type": _var(txt, "CMAKE_BUILD_TYPE"),
            "vmm": vmm,
            "graphs": _flag(txt, "GGML_HIP_GRAPHS") if hip else None,
            "flash_attn": _flag(txt, "GGML_CUDA_FA"),
            "native": _flag(txt, "GGML_NATIVE"),
        })
    return out


def device_alignment():
    src = ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu"
    if not src.exists():
        return None, "source not found"
    txt = src.read_text(errors="replace")
    m = re.search(
        r"ggml_backend_cuda_buffer_type_get_alignment\(ggml_backend_buffer_type_t buft\)\s*\{\s*return\s+(\d+)\s*;",
        txt)
    if not m:
        return None, "pattern not found (backend refactored?)"
    return int(m.group(1)), "ggml/src/ggml-cuda/ggml-cuda.cu"


import struct


def _read_gguf(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")

        def rd_val(t):
            if t == 0:
                return struct.unpack("<B", f.read(1))[0]
            if t == 1:
                return struct.unpack("<b", f.read(1))[0]
            if t == 2:
                return struct.unpack("<H", f.read(2))[0]
            if t == 3:
                return struct.unpack("<h", f.read(2))[0]
            if t == 4:
                return struct.unpack("<I", f.read(4))[0]
            if t == 5:
                return struct.unpack("<i", f.read(4))[0]
            if t == 6:
                return struct.unpack("<f", f.read(4))[0]
            if t == 7:
                return bool(struct.unpack("<B", f.read(1))[0])
            if t == 8:
                return rd_str()
            if t in (10, 11, 12):
                if t == 10:
                    fmt = "<q"
                elif t == 11:
                    fmt = "<Q"
                else:
                    fmt = "<d"
                return struct.unpack(fmt, f.read(8))[0]
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [rd_val(et) for _ in range(n)]
            raise ValueError(f"unknown kv type {t}")

        kv = {}
        for _ in range(n_kv):
            key = rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            val = rd_val(t)
            if not isinstance(val, list):
                kv[key] = val

        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            n_dims = struct.unpack("<I", f.read(4))[0]
            ne = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "type": ttype, "ne": ne, "off": off})

        alignment = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
    return version, alignment, data_start, tensors, kv


def type_geom(ttype, ne):
    if ttype not in TYPE_LAYOUT:
        raise ValueError(f"unknown ggml type id {ttype}")
    name, be, bb = TYPE_LAYOUT[ttype]
    row_elems = ne[0]
    row_bytes = max(1, (row_elems + be - 1) // be) * bb
    n_rows = 1
    for d in ne[1:]:
        n_rows *= d
    return name, row_bytes, n_rows


def audit_rows(tensors, big=1 << 20):
    per_type = {}
    agg = {"big_bytes": 0, "big_hi": 0, "small_bytes": 0, "small_hi": 0}
    for t in tensors:
        name, row_bytes, n_rows = type_geom(t["type"], t["ne"])
        nbytes = row_bytes * n_rows
        hi = 0 if row_bytes % SECTOR == 0 else n_rows * min(SECTOR, row_bytes)
        st = per_type.setdefault(name, {"tensors": 0, "bytes": 0, "hi": 0, "rows": 0,
                                        "row_bytes": set()})
        st["tensors"] += 1
        st["bytes"] += nbytes
        st["rows"] += n_rows
        st["hi"] += hi
        st["row_bytes"].add(row_bytes)
        if nbytes >= big:
            agg["big_bytes"] += nbytes
            agg["big_hi"] += hi
        else:
            agg["small_bytes"] += nbytes
            agg["small_hi"] += hi
    for v in per_type.values():
        v["row_bytes"] = sorted(v["row_bytes"])
        v["overhead_hi_pct"] = 100.0 * v["hi"] / max(v["bytes"], 1)
    agg["overhead_hi_pct"] = 100.0 * agg["big_hi"] / max(agg["big_bytes"], 1)
    return per_type, agg


def audit_kv(kv, tensors, n_ctx, ctk="q8_0", ctv="q5_0"):
    arch = kv.get("general.architecture", "?")
    get = lambda k: kv.get(f"{arch}.{k}")
    n_layer = get("block_count")
    kv_heads = get("attention.head_count_kv")
    k_len = get("attention.key_length")
    v_len = get("attention.value_length")
    if None in (n_layer, kv_heads, k_len, v_len):
        return {"error": "missing attention metadata"}

    attn_blocks = sorted({int(t["name"].split(".")[1]) for t in tensors
                          if t["name"].startswith("blk.") and t["name"].endswith("attn_k.weight")})
    mtp_blocks = sorted({int(t["name"].split(".")[1]) for t in tensors
                         if t["name"].startswith("blk.") and ".nextn." in t["name"]})
    used = [b for b in attn_blocks if b not in mtp_blocks]

    b_k, b_v = KV_BPV.get(ctk, 2.0), KV_BPV.get(ctv, 2.0)
    per_token = len(used) * kv_heads * (k_len * b_k + v_len * b_v)
    return {
        "arch": arch, "block_count": n_layer,
        "attn_blocks": len(attn_blocks), "mtp_blocks": mtp_blocks,
        "attn_blocks_used": len(used),
        "kv_heads": kv_heads, "k_len": k_len, "v_len": v_len,
        "ctk": ctk, "ctv": ctv,
        "bytes_per_token": per_token,
        "bytes_at_ctx": per_token * n_ctx,
        "n_ctx": n_ctx,
        "k_row_bytes": k_len * b_k,
        "k_row_sectors": -(-int(k_len * b_k) // SECTOR),
    }


def audit_fallback_hazards():
    fns = ("get_vec_dot_q_cuda", "get_vdr_mmvq", "get_mmq_q8_1")
    dispatch = []
    for ext in ("*.cu", "*.cuh"):
        for f in (ROOT / "ggml/src/ggml-cuda").rglob(ext):
            try:
                t = f.read_text(errors="replace")
            except OSError:
                continue
            for fn in fns:
                i = t.find(fn + "(")
                if i == -1:
                    continue
                chunk = t[i:i + 3500]
                m = re.search(r"default:\s*return ([^;]+);", chunk)
                if m:
                    val = m.group(1).strip()
                    kind = "NULL-DISPATCH" if "nullptr" in val or "NULL" in val else f"returns {val}"
                    dispatch.append((str(f.relative_to(ROOT)), fn, kind))
    aborts = 0
    for name in ("mmvq.cu", "mmq.cu"):
        f = ROOT / "ggml/src/ggml-cuda" / name
        if f.exists():
            aborts += f.read_text(errors="replace").count("GGML_ABORT(")
    return {"dispatch_defaults": dispatch, "dispatch_aborts": aborts}


def cmd_audit(args):
    builds = audit_builds()
    dev_align, dev_src = device_alignment()
    model = MODEL
    n_ctx = 112128

    report = {"builds": builds, "device_alignment": dev_align,
              "device_alignment_source": dev_src}

    if model.exists():
        version, file_align, data_start, tensors, kv = _read_gguf(model)
        rows, agg = audit_rows(tensors)
        total_bytes = sum(v["bytes"] for v in rows.values())
        total_extra = agg["big_hi"]
        report["container"] = {"version": version, "file_alignment": file_align,
                               "data_start": data_start, "tensors": len(tensors)}
        report["rows"] = rows
        report["weights_bytes"] = total_bytes
        report["stream_extra_bytes"] = total_extra
        report["stream_overhead_pct"] = agg["overhead_hi_pct"]
        report["stream_agg"] = agg
        report["kv"] = audit_kv(kv, tensors, n_ctx)
    else:
        report["container"] = {"error": f"model not found: {model}"}
        rows, kv = {}, {}
        agg = {"big_hi": 0, "overhead_hi_pct": 0.0}

    report["fallback_hazards"] = audit_fallback_hazards()

    scan_root = Path(getattr(args, "repo", ".")).resolve()
    scan_results = _scan_rules(scan_root)
    report["scan_rules"] = scan_results

    if args.json:
        print(json.dumps(report, indent=2, default=list))
        return 0

    print("=" * 78)
    print("  STRUCTURAL AUDIT - llama.cpp HIP fork (gfx1030) mechanical checks")
    print("=" * 78)

    print("\n[1] Build / runtime configuration")
    if not builds:
        print("    no build*/CMakeCache.txt found")
    for b in builds:
        print(f"    {b['dir']:20s} {b['backend']:7s} targets={str(b['targets']):9s} {b['build_type']}")
        if b["backend"] == "HIP":
            print(f"    {'':20s} {b['vmm']}  graphs={b['graphs']} fa={b['flash_attn']}")

    print("\n[2] Device-side tensor alignment")
    if dev_align is None:
        print(f"    unknown ({dev_src})")
    else:
        print(f"    GGML CUDA/HIP buffer base alignment = {dev_align} B  ({dev_src})")

    print("\n[3] Row-stride sector overhead")
    if not rows:
        print("    no tensor table")
    else:
        print(f"    stream = {total_bytes/2**30:.3f} GiB")
        print(f"    row-boundary waste upper bound: {agg['big_hi']/2**20:.1f} MiB ({agg['overhead_hi_pct']:.3f}%)")
        for t, v in sorted(rows.items(), key=lambda kv_: -kv_[1]["bytes"]):
            rb = ",".join(str(x) for x in v["row_bytes"][:4])
            print(f"    {t:10s} {v['bytes']/2**30:8.3f} GiB rows={v['rows']:8d} row_bytes={rb} waste={v['overhead_hi_pct']:.3f}%")

    print("\n[4] KV cache geometry")
    kvp = report.get("kv", {})
    if kvp and "error" not in kvp:
        print(f"    {kvp['arch']}: {kvp['attn_blocks_used']} attn blocks, {kvp['kv_heads']} kv heads")
        print(f"    per token: {kvp['bytes_per_token']/1024:.2f} KiB  at {kvp['n_ctx']} tokens: {kvp['bytes_at_ctx']/2**30:.2f} GiB")
        print(f"    K row = {kvp['k_row_bytes']:.0f} B = {kvp['k_row_sectors']} sectors")
    else:
        print(f"    {(kvp or {}).get('error', 'not computed')}")

    print("\n[5] Fallback hazards")
    hz = report["fallback_hazards"]
    print(f"    dispatch default arms: {len(hz['dispatch_defaults'])}  GGML_ABORT in mmvq/mmq: {hz['dispatch_aborts']}")
    for f, fn, kind in hz["dispatch_defaults"][:6]:
        print(f"      {fn}: {kind}   ({f})")

    print("\n[6] Static scan rules (13)")
    for rid, sev, hits, samp in scan_results:
        print(f"    [{sev:4}] {rid:12} hits={hits:4}  {samp[0] if samp else ''}")
    return 0



def cmd_census(args):
    binary = ROOT / "build-hip-gfx1030" / "bin" / "test-graph-census"
    if not binary.exists():
        print(f"missing binary: {binary}")
        return 2
    model = Path(args.model) if args.model else MODEL
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-m", str(model), "-c", str(args.ctx), "-ub", str(args.ubatch),
           "-ngl", "0", "--mode", args.mode, "--jsonl", "--out", str(out)]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stderr[-500:] if proc.stderr else "failed")
        return proc.returncode
    # verify
    metas = [json.loads(l) for l in out.read_text().splitlines() if l.strip().startswith("{")]
    meta = [m for m in metas if m.get("type") == "meta"][0]
    print(f"census: nodes={meta['n_nodes']} skipped={meta['n_skipped_views']} real={meta['n_real']}")
    return 0


def cmd_rotcheck(args):
    """Double-rotation census: detect the same tensor rotated twice (graph-level #897 hazard).

    Runs the detector's own negative control first, then censuses the tg and pp graphs, each
    built twice (determinism) and checks per-(layer, role) rotation counts for duplicates.
    """
    binary = ROOT / "build-hip-gfx1030" / "bin" / "test-graph-census"
    if not binary.exists():
        print(f"missing binary: {binary}")
        return 2
    st = subprocess.run([str(binary), "--selftest"], capture_output=True, text=True, timeout=120)
    if st.returncode != 0:
        print("rotation detector selftest FAILED:", (st.stdout or st.stderr).strip()[-500:])
        return st.returncode
    print("detector selftest:", st.stdout.strip())
    model = Path(args.model) if args.model else MODEL
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for mode in ("tg", "pp"):
        out = outdir / f"rotation-{mode}.jsonl"
        cmd = [str(binary), "-m", str(model), "-c", str(args.ctx), "-ub", str(args.ubatch),
               "-ngl", "0", "--mode", mode, "--jsonl", "--verify", "--out", str(out)]
        print(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rows = ([json.loads(l) for l in out.read_text().splitlines() if l.strip().startswith("{")]
                if out.exists() else [])
        summary = next((r for r in rows if r.get("type") == "rotation_summary"), None)
        verify = next((r for r in rows if r.get("type") == "rotation_verify"), None)
        viol = [r for r in rows if r.get("type") == "rotation_violation"]
        if summary is None:
            print(f"  {mode}: no rotation summary (rc={proc.returncode})")
            rc = proc.returncode or 1
            continue
        print(f"  {mode}: rotation nodes={summary['n_rotation_nodes']} "
              f"violations={summary['violations']} "
              f"builds-symmetric={verify['ok'] if verify else 'n/a'}")
        for v in viol[:10]:
            print(f"    {v['kind']}: {v.get('op_name')} on {v.get('src_name', v.get('name'))}"
                  f" -> {v.get('names', '')}")
        if not summary.get("ok"):
            rc = proc.returncode or 3
    return rc



# ---------------------------------------------------------------- absorbed from gfx1030_audit.py
SCAN_RULES = [
    ("WPB-GATE", "ggml/src/ggml-cuda/mmq.cu", r"should_use_mmq", "INFO",
     "locate ggml_cuda_should_use_mmq — check expert-width arm covers RDNA2"),
    ("DEAD-WRAPPER", "ggml/src/ggml-cuda", r"__(ldg|ldcg|ldcs|ldlu)\s*\(", "WARN",
     "read-only load wrappers are no-ops on gfx1030 (closed lane) — don't add"),
    ("CACHE-BITS", "ggml/src/ggml-cuda", r"(slc|dlc|glc)\b|nontemporal", "INFO",
     "ISA-level cache-policy usage — expected ~0 today; Probes C/D test reach"),
    ("SYNC", "ggml/src/ggml-cuda", r"(hip|cuda)(Stream|Device)Synchronize", "WARN",
     "sync point — syncs are real in code but off the timed path: cpy_tensor is "
     "never called in single-sequence prefill, and skipping all eight blocking "
     "syncs measured +0.65 t/s (95% CI +/-1.76), i.e. zero"),
    ("MEMCPY", "ggml/src/ggml-cuda", r"hipMemcpyAsync", "INFO",
     "async copy call site — check it isn't inside the timed pass"),
    ("PERM-DEQ", "ggml/src/ggml-cuda/vecdotq.cuh", r"__byte_perm|__builtin_amdgcn_perm", "OK",
     "V_PERM_B32 table dequant present (verified in-tree)"),
    ("SDOT4", "ggml/src/ggml-cuda", r"__builtin_amdgcn_sdot4|v_dot4c_i32_i8", "OK",
     "dot4 dot-product path present"),
    ("BPERMUTE", "ggml/src/ggml-cuda", r"ds_bpermute|ds_swizzle|bpermute", "GAP",
     "LDS-pipe cross-lane permute NOT used — reductions still on VALU shfl"),
    ("SHFL-RED", "ggml/src/ggml-cuda", r"__shfl_(xor|down|up)_sync|__shfl_sync", "INFO",
     "VALU-pipe shuffle reductions — candidates to move to ds_bpermute"),
    ("RDNA2-GUARD", "ggml/src/ggml-cuda", r"__gfx103|GGML_HIP_RDNA|RDNA2|__HIP_ARCH__\s*==\s*1030", "INFO",
     "gfx1030-specific guard sites — check coverage where arms differ"),
    ("DEAD-CTR", "scripts|engines|tools", r"FETCH_SIZE|SQ_WAIT_ANY|SQC_LDS_IDX_ACTIVE|SQ_INSTS_VALU", "WARN",
     "script references a counter verified dead on gfx1030 — bytes=GL2C_MC_RDREQ*64"),
    ("BENCH-EXCL", "scripts", r"bench-exclusive", "OK",
     "exclusive-bench harness present — mandatory for measured claims"),
]

def _iter_files(root, globset):
    for pat in globset.split("|"):
        p = root / pat
        if p.is_file():
            yield p
        elif p.is_dir():
            for ext in ("*.cu", "*.cuh", "*.cpp", "*.h", "*.hpp", "*.py", "*.sh", "*.md", "*.hip"):
                yield from p.rglob(ext)

def _scan_rules(root):
    root = Path(root).resolve()
    out = []
    for rid, globset, regex, sev, msg in SCAN_RULES:
        hits = 0; samples = []
        try:
            rx = re.compile(regex)
        except re.error as e:
            out.append((rid, "ERR", 0, [f"bad regex: {e}"])); continue
        for f in _iter_files(root, globset):
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits += 1
                        if len(samples) < 4:
                            samples.append(f"{f.relative_to(root)}:{i}: {line.strip()[:100]}")
            except (OSError, UnicodeDecodeError):
                continue
        out.append((rid, sev, hits, samples))
    return out


def _is_copy(name):
    return ("__amd_rocclr_copyBuffer" in name) or ("fillBufferAligned" in name)

def _steady_start(rows):
    last_idx = None
    for i, r in enumerate(rows):
        if not _is_copy(r["Kernel_Name"]):
            continue
        if i + 1 < len(rows):
            g = int(rows[i + 1]["Start_Timestamp"]) - int(r["End_Timestamp"])
            if g > 1_000_000:
                last_idx = i
    if last_idx is not None:
        return int(rows[last_idx]["End_Timestamp"])
    for r in rows:
        if not _is_copy(r["Kernel_Name"]):
            return int(r["End_Timestamp"])
    return int(rows[0]["Start_Timestamp"])

def _analyze_trace_rows(rows, args):
    if not rows:
        sys.exit("no rows in window")
    rows = sorted(rows, key=lambda r: int(r["Start_Timestamp"]))
    span = int(rows[-1]["End_Timestamp"]) - int(rows[0]["Start_Timestamp"])
    busy = 0; cur_s = cur_e = None
    for r in rows:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if cur_s is None: cur_s, cur_e = s, e
        elif s <= cur_e: cur_e = max(cur_e, e)
        else: busy += cur_e - cur_s; cur_s, cur_e = s, e
    busy += cur_e - cur_s
    gaps_ns = span - busy
    per_kernel = {}
    pair_gaps = {}
    prev = None
    for r in rows:
        n = r["Kernel_Name"]
        st, en = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        k = per_kernel.setdefault(n, [0, 0]); k[0] += 1; k[1] += en - st
        if prev is not None:
            g = st - int(prev["End_Timestamp"])
            if g > 0:
                pk = pair_gaps.setdefault((prev["Kernel_Name"], n), [0, 0])
                pk[0] += 1; pk[1] += g
        prev = r
    tok = max(int(getattr(args, "tokens", 1) or 1), 1)
    res = {
        "span_ms": span/1e6, "busy_ms": busy/1e6, "gap_ms": gaps_ns/1e6,
        "idle_pct": 100*gaps_ns/span, "dispatches": len(rows),
        "per_token": {"span_ms": span/1e6/tok, "busy_ms": busy/1e6/tok,
                       "gap_ms": gaps_ns/1e6/tok, "dispatches": len(rows)/tok},
        "top_kernels": sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:12],
        "top_gap_pairs": sorted(pair_gaps.items(), key=lambda kv: -kv[1][1])[:12],
    }
    wg = getattr(args, "weights_gb", 0) or 0
    if wg:
        mmv_ns = sum(v[1] for k, v in per_kernel.items() if "mul_mat_vec_q" in k)
        mmv_cnt = sum(v[0] for k, v in per_kernel.items() if "mul_mat_vec_q" in k)
        if mmv_ns:
            wb = wg * 1e9 * tok / mmv_ns
            res["matvec_dispatches"] = mmv_cnt
            res["matvec_gbs_in_window"] = round(wb, 1)
            res["matvec_pct_of_peak"] = round(100*wb/500.6, 1)
        res["end_to_end_gbs"] = round(wg*1e9/(span/tok), 1)
    return res


def _audit_trace(args):
    path = getattr(args, "kernel_trace", None) or getattr(args, "kernel_trace_pos", None)
    if not path:
        return None
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("Kind") == "KERNEL_DISPATCH":
                rows.append(r)
    if not rows:
        sys.exit("no KERNEL_DISPATCH rows")
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    full_res = _analyze_trace_rows(rows, args)
    full_res["window"] = "full trace"
    _boundary = _steady_start(rows)
    steady_rows = [r for r in rows if int(r["Start_Timestamp"]) >= _boundary]
    steady_res = _analyze_trace_rows(steady_rows, args)
    steady_res["window"] = "steady state"
    skipped_res = None
    if getattr(args, "skip_ms", None) is not None:
        boundary = int(rows[0]["Start_Timestamp"]) + args.skip_ms * 1e6
        skipped_rows = [r for r in rows if int(r["Start_Timestamp"]) >= boundary]
        skipped_res = _analyze_trace_rows(skipped_rows, args)
        skipped_res["window"] = f"skip {args.skip_ms} ms"
    if getattr(args, "steady", False):
        primary = steady_res
    elif skipped_res is not None:
        primary = skipped_res
    else:
        primary = steady_res
    primary = dict(primary)
    primary["full_trace"] = full_res
    primary["steady_state"] = steady_res
    if skipped_res is not None:
        primary["skipped_window"] = skipped_res
    if getattr(args, "json", False):
        print(json.dumps(primary, indent=1)); return 0
    # text
    print("# gfx1030 audit trace")
    print(f"full trace: span {full_res['span_ms']:.0f} ms | idle {full_res['idle_pct']:.1f}%")
    print(f"steady state: span {steady_res['span_ms']:.0f} ms | idle {steady_res['idle_pct']:.1f}%")
    print(f"primary window ({primary['window']}): idle {primary['idle_pct']:.1f}%")
    print("top gap pair:", primary["top_gap_pairs"][0][0][0], "->", primary["top_gap_pairs"][0][0][1])
    return 0


DEFAULT_WPS = {
    "WP-E": ["dispatch/launch bubbles — ceiling 14.7% of the token (closing all "
             "idle gives 31.1 -> ~36.5 t/s); floor is uniform ~3.6 us/launch so "
             "no boundary is individually fixable; ~278 launches ~ 1 t/s. "
             "Realistic gain ~0.2 t/s.", 0.2, None, 3, 4, 2],
    "WP-B": ["expert-width gate (one line, external +15-42%)", 4.0, None, 1, 4, 1],
    "WP-D": ["MTP bandwidth gap decomposition (0.4x unclaimed)", 3.0, None, 2, 3, 3],
    "H13": ["split-K GEMV for decode matvec", 4.0, None, 3, 2, 3],
    "IC-SLC": ["SLC stream-marking on weight stream — slc bit confirmed in SASS; "
               "Probe D measures the effect", 2.0, None, 2, 4, 2],
    "WP-A": ["MMQ scale pooling (+20% prefill, shared layout)", None, 20.0, 5, 2, 3],
    "DOT2": ["pre-scaled f16 -> V_DOT2_F32_F16 inner loop", 5.0, None, 4, 2, 3],
}

def _normalize_wp(wid, v):
    v = list(v)
    if len(v) == 5:
        desc, gain, eff, al, be = v
        if "prefill" in str(desc).lower():
            gd, gp = None, gain
        else:
            gd, gp = gain, None
        return [desc, gd, gp, eff, al, be]
    if len(v) >= 6:
        return v[:6]
    return v

def _prioritize(args):
    state_f = Path(getattr(args, "state", None) or (Path(__file__).resolve().parent / ".audit_state.json"))
    if state_f.exists():
        raw = json.loads(state_f.read_text())
        reg = {wid: _normalize_wp(wid, v) for wid, v in raw.items()}
    else:
        reg = {wid: list(v) for wid, v in DEFAULT_WPS.items()}
    for u in getattr(args, "update", None) or []:
        wid, res = u
        if wid not in reg: sys.exit(f"unknown WP id {wid}")
        if res == "success": reg[wid][4] += getattr(args, "strength", 1)
        else: reg[wid][5] += getattr(args, "strength", 1)
    if getattr(args, "update", None) or not state_f.exists():
        state_f.parent.mkdir(parents=True, exist_ok=True)
        state_f.write_text(json.dumps(reg, indent=1))
    target = getattr(args, "target", None) or "decode"
    ranked = []
    for wid, entry in reg.items():
        entry = _normalize_wp(wid, entry)
        desc, gd, gp, eff, al, be = entry[:6]
        gain = gd if target == "decode" else gp
        if gain is None:
            continue
        pm = al/(al+be)
        score = gain*pm/eff
        lo95 = al/(al+be) - 1.96*math.sqrt(al*be/((al+be)**2*(al+be+1)))
        ranked.append((score, wid, desc, gain, eff, pm, max(lo95,0), al, be))
    ranked.sort(reverse=True)
    return ranked

def _fmt_md(scan, trc, prio, a):
    L = ["# gfx1030 audit report\n"]
    if trc:
        summaries = []
        if "full_trace" in trc:
            summaries.append(("full trace", trc["full_trace"]))
        if "steady_state" in trc and trc.get("steady_state") is not trc.get("full_trace"):
            summaries.append(("steady state", trc["steady_state"]))
        if "skipped_window" in trc:
            summaries.append((trc["skipped_window"].get("window", "skipped"), trc["skipped_window"]))
        if not summaries:
            summaries.append((trc.get("window", "trace"), trc))
        L += ["## Trace windows (idle % of span)", "",
              "| window | span ms | busy ms | idle ms | idle % | dispatches | per-token span ms |",
              "|---|---|---|---|---|---|---|"]
        for label, w in summaries:
            p = w["per_token"]
            L.append(f"| {label} | {w['span_ms']:.0f} | {w['busy_ms']:.0f} | {w['gap_ms']:.0f} | "
                     f"{w['idle_pct']:.1f}% | {w['dispatches']} | {p['span_ms']:.2f} |")
        p = trc["per_token"]
        L += ["", f"## Trace detail — primary window: {trc.get('window','trace')}",
              f"- span {trc['span_ms']:.0f} ms | busy {trc['busy_ms']:.0f} ms "
              f"| idle {trc['gap_ms']:.0f} ms (**{trc['idle_pct']:.1f}%**) | {trc['dispatches']} dispatches",
              f"- per token: {p['span_ms']:.2f} ms span, {p['busy_ms']:.2f} busy, "
              f"{p['gap_ms']:.2f} gaps, {p['dispatches']:.0f} dispatches"]
        if "matvec_gbs_in_window" in trc:
            L += [f"- matvec in-window {trc['matvec_gbs_in_window']} GB/s "
                  f"({trc['matvec_pct_of_peak']}% of 500.6 peak); "
                  f"end-to-end {trc['end_to_end_gbs']} GB/s"]
        if "matvec_warning" in trc:
            L += [f"- **WARNING:** {trc['matvec_warning']}"]
        L += ["\n### Top gap boundaries (WP-E attack surface)\n",
              "| boundary | count | gap ms |", "|---|---|---|"]
        for (a_, b_), (c, g) in trc["top_gap_pairs"]:
            L.append(f"| `{a_}` -> `{b_}` | {c} | {g/1e6:.1f} |")
        L += ["\n### Top kernels\n", "| kernel | dispatches | total ms |", "|---|---|---|"]
        for n, (c, t) in trc["top_kernels"]:
            L.append(f"| `{n}` | {c} | {t/1e6:.1f} |")
    if scan:
        L += ["\n## Static scan\n", "| check | sev | hits | sample |", "|---|---|---|---|"]
        for rid, sev, hits, samp in scan:
            L.append(f"| {rid} | {sev} | {hits} | {samp[0] if samp else ''} |")
    L += ["\n## Priority ranking (score = gain x P(win) / effort)\n",
          "| rank | WP | desc | gain | eff | P(win) | score |", "|---|---|---|---|---|---|---|"]
    for i, (s, wid, d, g, e, pm, lo, al, be) in enumerate(prio, 1):
        L.append(f"| {i} | {wid} | {d} | {g} | {e} | {pm:.2f} [{lo:.2f}] | {s:.2f} |")
    L += ["\n_Noise floor 0.20 t/s — gains below it are unmeasurable._"]
    return "\n".join(L)


def cmd_score(args):
    ranked = _prioritize(args)
    md = _fmt_md(None, None, ranked, args)
    print(md)
    return 0


def cmd_report(args):
    scan = _scan_rules(Path(args.repo)) if getattr(args, "repo", None) else None
    # compute primary by running analysis (no printing)
    if getattr(args, "kernel_trace", None):
        rows = []
        with open(args.kernel_trace, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("Kind") == "KERNEL_DISPATCH":
                    rows.append(r)
        rows.sort(key=lambda r: int(r["Start_Timestamp"]))
        full_res = _analyze_trace_rows(rows, args)
        full_res["window"] = "full trace"
        _b = _steady_start(rows)
        steady_rows = [r for r in rows if int(r["Start_Timestamp"]) >= _b]
        steady_res = _analyze_trace_rows(steady_rows, args)
        steady_res["window"] = "steady state"
        primary = steady_res if getattr(args, "steady", False) else steady_res
        primary = dict(primary)
        primary["full_trace"] = full_res
        primary["steady_state"] = steady_res
        trc = primary
    else:
        trc = None
    prio = _prioritize(args)
    md = _fmt_md(scan, trc, prio, args)
    out = getattr(args, "out", None)
    if out:
        Path(out).write_text(md); print(f"wrote {out}")
    else:
        print(md)
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="capability probe (read-only)").set_defaults(func=cmd_probe)

    p = sub.add_parser("bench", help="llama-bench depth sweep or --ab dual-artifact swap")
    p.add_argument("--depths", default="0")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--backend", choices=["hip", "vk", "both"], default="both")
    p.add_argument("--ab", nargs=2, metavar=("BASE.so", "FIX.so"), help="dual-artifact swap")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--pp", type=int, default=0,
                   help="prompt tokens per test (0 = decode-only; use 512 for prefill A/B)")
    p.add_argument("--n", type=int, default=128, dest="n_gen",
                   help="generated tokens per test")
    p.add_argument("--ubatch", type=int, default=None,
                   help="llama-bench -ub (logical ubatch; -ub 128 is the T5 operating point)")
    p.add_argument("--ncmoe", type=int, default=None,
                   help="llama-bench --n-cpu-moe (MoE host-offload operating point)")
    p.add_argument("--extra", default=None,
                   help="extra llama-bench args, shell-quoted")
    p.add_argument("--ab-stats", default=None, metavar="D1,D2,...", dest="ab_stats",
                   help="offline: per-pair ms/token deltas -> paired sd, 95%% CI and required n "
                        "(no bench run, no GPU)")
    p.add_argument("--min-detectable", type=float, default=None, metavar="MS", dest="min_detectable",
                   help="delta in ms/token the A/B must resolve; prints required pair count and "
                        "flags UNDERPOWERED when the measured 95%% CI is wider")
    p.add_argument("--a-bindir", default=None, dest="a_bindir",
                   help="bin/ dir for arm A (two-build-dir A/B)")
    p.add_argument("--b-bindir", default=None, dest="b_bindir",
                   help="bin/ dir for arm B (two-build-dir A/B)")
    p.add_argument("--a-env", action="append", default=[], dest="a_env",
                   metavar="K=V", help="environment for arm A (env-only A/B); repeatable")
    p.add_argument("--b-env", action="append", default=[], dest="b_env",
                   metavar="K=V", help="environment for arm B (env-only A/B); repeatable")
    p.add_argument("--gate", type=float, default=None, metavar="T/S",
                   help="PASS/FAIL throughput gate in t/s for the A/B comparison")
    p.add_argument("--out", default=None, help="write A/B results JSON to this path")
    p.add_argument("--json", default=None, dest="json_out", help="alias of --out for results JSON")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the exact command that would run, then exit")
    p.add_argument("--record", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("profile", help="graph-free kernel trace -> scored class table + versioned JSON")
    p.add_argument("--model", default="dense", help="preset key (dense|moe) or a .gguf path")
    p.add_argument("--backend", choices=["hip", "vk"], default="hip")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ncmoe", type=int, default=0, help="llama-bench --n-cpu-moe")
    p.add_argument("-fa", "--flash-attn", default="1", dest="flash_attn")
    p.add_argument("-ctk", "--cache-type-k", default="q8_0", dest="cache_type_k")
    p.add_argument("-ctv", "--cache-type-v", default="q5_0", dest="cache_type_v")
    p.add_argument("--fitt", type=int, default=128, help="llama-bench --fit-target MiB (0 disables)")
    p.add_argument("-ot", "--override-tensor", default=None, dest="ot")
    p.add_argument("--dev", default=None, help="explicit -dev value (default: ROCm0 / Vulkan0)")
    p.add_argument("--extra", default=None, help="extra llama-bench args, shell-quoted")
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None, help="side-car path (default <out>/profile.json)")
    p.add_argument("--pmc", default=None)
    p.add_argument("--no-fusion", action="store_true", dest="no_fusion")
    p.add_argument("--no-wall", action="store_true", dest="no_wall",
                   help="skip the graph-enabled reference run")
    p.add_argument("--reuse", default=None, help="re-aggregate an existing trace dir (no bench)")
    p.add_argument("--wall-ms", type=float, default=None, dest="wall_ms",
                   help="graph-enabled ms/token to record (skips the reference run)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the exact command that would run, then exit")
    p.add_argument("--ledger", default=None, help="also write a census-join ledger")
    p.add_argument("--census", default=None)
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="diff two profile JSONs instead of measuring")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("trace", help="rocprofv3 kernel trace (alias of profile, ledger-friendly)")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--out", default=None)
    p.add_argument("--pmc", default=None)
    p.add_argument("--ledger", default=None)
    p.add_argument("--no-fusion", action="store_true", dest="no_fusion")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--census", default=None)
    p.add_argument("--kernel-trace", default=None, dest="kernel_trace",
                   help="standalone kernel-trace CSV to analyze")
    p.add_argument("kernel_trace_pos", nargs="?", default=None,
                   help="positional kernel-trace CSV (alias of --kernel-trace)")
    p.add_argument("--steady", action="store_true", help="auto-detect steady-state window")
    p.add_argument("--skip-ms", type=float, default=None, dest="skip_ms",
                   help="discard rows within N ms of the trace's first row")
    p.add_argument("--weights-gb", type=float, default=0, dest="weights_gb")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("isa", help="static instruction mix + 5%% ALU gate")
    p.add_argument("--save-baseline", action="store_true", dest="save_baseline")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_isa)

    p = sub.add_parser("audit", help="mechanical structural checks")
    p.add_argument("--repo", default=".")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("census", help="graph census (CPU-only)")
    p.add_argument("--model", default=None)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ubatch", type=int, default=1)
    p.add_argument("--mode", choices=["tg", "pp"], default="tg")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_census)

    p = sub.add_parser("rotcheck", help="double-rotation census over tg and pp graphs")
    p.add_argument("--model", default=None)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ubatch", type=int, default=2048)
    p.add_argument("--outdir", default=str(ROOT / "scripts" / ".rotcheck"))
    p.set_defaults(func=cmd_rotcheck)

    p = sub.add_parser("predict", help="predict candidate performance")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--backend", default="hip")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--edit", default=None)
    p.add_argument("--kernel-change", action="store_true", dest="kernel_change")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("score", help="Beta-Binomial work-package prioritisation")
    p.add_argument("--state", default=str(Path(__file__).resolve().parent / ".audit_state.json"))
    p.add_argument("--target", choices=["decode", "prefill", "moe"], default="decode")
    p.add_argument("--update", nargs=2, action="append", metavar=("WP", "success|failure"))
    p.add_argument("--strength", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="combined markdown audit report")
    p.add_argument("--repo", default=".")
    p.add_argument("--kernel-trace", default=None, dest="kernel_trace")
    p.add_argument("--tokens", type=int, default=131)
    p.add_argument("--weights-gb", type=float, default=0, dest="weights_gb")
    p.add_argument("--state", default=str(Path(__file__).resolve().parent / ".audit_state.json"))
    p.add_argument("--target", choices=["decode", "prefill", "moe"], default="decode")
    p.add_argument("--steady", action="store_true")
    p.add_argument("--skip-ms", type=float, default=None, dest="skip_ms")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)


    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
