"""Shared measurement settings for the gfx1030 campaign scripts.

One place for the values every measurement script agrees on.  Import with
``from common import ...`` (scripts/ is on sys.path when a script in this
directory is run).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL = Path("/home/stormrage/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
FA = "1"
CTK = "q8_0"
CTV = "q5_0"

BENCH_EXCLUSIVE = ROOT / "scripts" / "bench-exclusive.sh"
