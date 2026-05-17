*(~40 tokens vs 500+ for raw data)*

## 3. Knowledge Reuse Patterns
✅ Create `RESEARCH_LOG.md` entries → reference by ID, not re-explain  
   `"Per RL-042: SLC=1 bypass validated for Q4_K_M"`  

✅ Use skill triggers → auto-load relevant context  
   Mention `"ISA"` → auto-attach `rdna2-isa-reference.md`  

✅ Chunk large tasks → one skill per subtask  
   Don't ask for "full kernel rewrite"; ask for "dequant inline pattern"  

## 4. Token Budget Guidelines
| Task Type | Target Token Budget | Strategy |
|-----------|-------------------|----------|
| Code review | ≤100 tokens | Reference lines, not full file |
| Telemetry analysis | ≤80 tokens | Report deltas, not raw values |
| ISA question | ≤60 tokens | Use mnemonics, not encodings |
| Build debug | ≤120 tokens | Paste error + 3 lines context |

## 5. Auto-Compression Checklist
Before sending prompt:
- [ ] Removed boilerplate (imports, license headers)
- [ ] Replaced full paths with relative (`ggml-hip/mmvq.cu`)
- [ ] Summarized telemetry to key metrics + deltas
- [ ] Used skill triggers instead of re-explaining concepts
- [ ] Referenced prior decisions by ID (`Per CTX256-DIR-006`)

## Emergency Token Reduction
If context window nears limit:
1. Replace code blocks with `// [pattern: fused_dequant_stream_k]`
2. Replace telemetry tables with `// [telemetry: tg128=61.8±0.9, WAVE_ISSUE_WAIT↓18%]`
3. Use abbreviations: `VGPR` not "vector general-purpose register"

Use these patterns to maximize token efficiency without losing technical precision.