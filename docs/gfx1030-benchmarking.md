# gfx1030 local benchmarking rules

These override any other instruction about servers or measurement on this host.

- **`llama-server` must never be active.** Do not launch it, stop it, query it, or
  health-check it. All performance work uses `llama-bench` and `test-backend-ops` only.
- **Exactly one `llama-bench` process at a time.** Concurrent instances exhaust VRAM
  (`unable to allocate ROCm0 buffer`) and silently corrupt each other's numbers.
  Runs are strictly sequential, never parallel, never backgrounded while another runs.
- Preflight before every measurement:
  `pgrep -a -f "llama-server|llama-bench"` must print nothing, and
  `rocm-smi --showmemuse` must show ~0 VRAM allocated.
- `scripts/bench-exclusive.sh` enforces both rules; prefer it over calling
  `llama-bench` directly.

## A/B resolving power (how small a delta this host can actually prove)

Run-to-run decode speed on this host has a standard deviation of roughly
1.56 t/s (~2.3%, ~0.33 ms/token at 68 t/s). That is far larger than the effects
being chased, so an unpaired single run proves nothing and an interleaved pair
list must be judged by its **paired** statistics.

- Report the paired mean delta and the **sd of the per-pair differences**, in
  ms/token, plus the 95% CI half-width. `gfx1030.py bench --ab` does this and
  accepts `--min-detectable <ms>` to print the required pair count;
  `bench --ab-stats "d1,d2,..."` does the same arithmetic offline.
- Required pairs: `n = (1.96 * sd_pair / delta)^2`. At the measured
  `sd_pair = 0.64 ms`, resolving 0.2 ms needs ~40 pairs, 0.141 ms needs ~80,
  and 0.08 ms needs ~250. Five pairs resolve only ~0.8 ms (95% CI).
- The per-model spreads in `scripts/.baseline.json` are pair-to-pair **medians
  of absolute runs**, not decision thresholds. Comparing a delta against them
  under-states the noise and has been used to accept changes that the paired
  statistics cannot distinguish from zero.
- Any claim below ~0.3 ms therefore needs either hundreds of interleaved pairs
  or a different instrument: the per-class bandwidth table from
  `gfx1030.py profile` (which resolves a geometry change on the class it
  touches) rather than whole-token `llama-bench` wall time.