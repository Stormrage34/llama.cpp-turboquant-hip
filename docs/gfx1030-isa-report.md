# gfx1030 bottleneck report, grounded in the RDNA 2 ISA manual and the RDNA2 fork wiki

Sources:
- AMD, *"RDNA 2" Instruction Set Architecture: Reference Guide*, 30 November 2020, 283 pages.
  Read locally via `pdftotext` from `~/Downloads/rdna2-shader-instruction-set-architecture.pdf`.
  Section numbers below are the manual's own.
- `https://blivioniag.github.io/gfx1030-wiki/llama-cpp/rdna2-overview.html` — the GFX1030 wiki
  page describing `edwinbrowwn/llama.cpp-rdna2`.

Everything measured was measured on this host, this fork, IQ3_S weights.
Nothing here is copied from another card's numbers.

---

## 1. The finding that changes the picture

The manual states, §2.3.1 *Local Data Share*:

> "Each workgroup processor (WGP) has a 128 kB memory space ... configured with 64 banks,
> each with 512 entries of 4 bytes. ... **A single workgroup may allocate up to 64kB of LDS
> space.**"

Our prefill kernel — `mul_mat_q` for IQ3_S at J=64 — allocates **45.9 KiB (46,976 B) of
dynamic LDS per block**. Two blocks would need **91.8 KiB**.

That is above the 64 kB a single workgroup may allocate, and above the 64 kB each compute unit
owns of the WGP's 128 kB. So the `occupancy = 2` field that the RDNA2 config table requests on
**every one of its 22 types** (`ggml/src/ggml-cuda/mmq-config-rdna2.cuh`) is not "hard to
achieve" or "in tension with register pressure" — it is **architecturally impossible at J=64**.

## 2. The register half of the same story

The manual states, §3.6.4 *VPGPR Allocation and Alignment*:

> "VGPRs are allocated in groups of 8 Dwords for wave64, and **16 Dwords for wave32**."

Our kernel uses **234 VGPRs**, which rounds up to **240** under 16-dword granularity.
Per wave: 240 × 32 lanes = 7,680 registers. A SIMD has 65,536, so **8 waves fit** — exactly one
block of 256 threads, which is what the hardware counters measured (**7.57 of 8 waves**).

To hold two blocks (16 waves) a thread would need **≤128 VGPRs**. That is a 45 % cut from 234,
on a kernel that currently spills **nothing** (`private_segment_fixed_size = 0`).

So both halves of the occupancy request fail independently, and either one alone is sufficient
to prevent a second block.

## 3. What this closes, and what it leaves

**Closed:** the long-running "reclaim the withheld registers" hypothesis. The table does not
withhold registers — the compiler emitted 234 despite `__launch_bounds__(256, 2)`, i.e. it could
not honour the request and did not try to fake it. There is no phantom second block to pay for.

**Closed:** "shrink the footprint so two blocks fit" as a *tuning* change. J=32 would roughly
halve the tile to ~23 KiB and two blocks would fit in 64 kB on the LDS side — but it also halves
the columns each block amortises the weight load over, doubling the number of tiles and the
number of times weights are re-read. That trades a latency win we have already measured as
unavailable for a bandwidth loss on a memory-bound path.

**Left open, and now the only prefill lever:** the instruction mix. The counters put the kernel at
~95 % issue-slot utilisation, so the gap to the 957.6 t/s dp4a ceiling is the price of unpacking
3-bit weights before every dot product — not latency, not occupancy, not the scheduler.

## 4. Where the wiki agrees, and where it points somewhere else

The wiki's fork (`edwinbrowwn/llama.cpp-rdna2`, ROCm 7.14, validated on four Radeon PRO V620)
lists graph fusions that this fork built independently and measured: ADD/RMSNorm, **Q8_1 reuse**,
**SwiGLU→Q8_1**, GDN sibling projection, MTP/DFlash. Two teams, same silicon, same four fusions.

Its headline gains (+10 % generation, +20 % prefill) come from **RCCL tensor parallelism across
four GPUs** — irrelevant to one card, and not transferable.

Its one quantified claim about *our* situation is the quant guidance:

> "Fastest fork quants ... are **Q4_0** and **Q8_0** (native MMVQ paths). **Q6 / UD-Q4_K can be
> slower even when they 'fit'.** **MXFP4** has been reported a bit faster than Q4_0 on some 27B
> runs — A/B on your cards."

This fork's entire campaign — 31.1 t/s decode, 598.9 t/s prefill, every counter run in this
report — was measured on **IQ3_S**, which is not on that list. IQ3_S is a 3-bit format, and 3-bit
unpacking is precisely the instruction-heavy path §3 above identifies as the binding constraint.

The two sources point at the same conclusion from opposite directions: the ISA says the remaining
prefill gap is unpack cost, and the wiki says the formats with cheap unpack are Q4_0/Q8_0.

## 5. Findings

| ID | Area | Finding | Impact | Risk | Recommendation |
|---|---|---|---|---|---|
| S1 | Occupancy | `occupancy = 2` in the RDNA2 MMQ table is impossible at J=64: 2 × 45.9 KiB = 91.8 KiB exceeds both the 64 kB per-workgroup cap (ISA §2.3.1) and the 64 kB per CU. | Closes the occupancy lane with a source, not an inference | None | Stop treating `occupancy` as a tunable for these tiles |
| S2 | Registers | 234 VGPRs rounds to 240 at 16-dword wave32 granularity (ISA §3.6.4) → 8 waves/SIMD = one block. Two blocks need ≤128 VGPRs, a 45 % cut on a zero-spill kernel. | Same lane, independent confirmation | None | Do not attempt 2-block occupancy by register squeeze |
| S3 | Wave size | Hardware is natively wave32; wave64 VALU issues twice (ISA §2.1). | The fork's wave32-only compile is correct | None | No change |
| S4 | LDS banks | 64 banks × 512 × 4 B per WGP. A wave32 reading 32 consecutive floats hits 32 distinct banks — conflict-free by construction. | Explains why no bank-conflict win appeared | None | Stop chasing LDS conflicts |
| S5 | Instruction mix | Counters: ~95 % issue-slot utilisation, 7.57/8 waves. The remaining gap is unpack work, not latency. | Prefill's 37.5 % headroom is unpack-bound | Irreducible within IQ3_S | The only lever is a format with cheaper unpack |
| S6 | Format | Wiki: Q4_0/Q8_0 are the native fast paths; this campaign is measured entirely on IQ3_S. | Potentially the largest remaining gain on this card | Requires a different model file; quality must be re-measured | A/B Q4_0 against IQ3_S on the same source, same placement, before any further kernel work |
| S7 | Corroboration | The reference fork's fusion list matches this fork's landed work item-for-item. | Independent validation of direction | None | Treat as confirmation; no work left in that list |
| S8 | Topology | The wiki's +10 %/+20 % is four-GPU RCCL tensor parallel. | None for one card | None | Not applicable |

## 6. What this means for you

Two independent sources, one manual and one fork built for this exact GPU, converge on the same
answer: **the remaining performance is in the number format, not in the kernels.**

The manual proves the two things we kept trying — more blocks per compute unit, and less LDS —
cannot both happen at this tile size, so that entire family of tuning is finished. What is left
is the cost of unpacking 3-bit weights, and the fork that did this work for a living says the
formats with cheap unpacking are Q4_0 and Q8_0.

That points at a test I have not run: quantise the same 27B source to Q4_0, measure it on this
host under identical placement, and compare both speed and quality against the IQ3_S numbers.
It is a model-file question, which you have ruled out as a lead answer — but it is now supported
by the ISA and by the only other team that has measured this card, rather than by convenience.
