## Librarian Report: Server-Aware Benchmarking Documentation

### Docs Updated
- `opencode/agents/oracle.md` — Added server-aware benchmarking metrics and pre-bench detection gate
- `opencode/reports/explorer_bench_analysis.md` — Created comprehensive analysis of benchmark script issues

### Data Archived
- Explorer analysis saved to `opencode/reports/explorer_bench_analysis.md`
- Current llama-server state: PID 17813, Qwen3.6-35B-A3B, running since session start

### Git Status
- Branch: current working branch
- Dirty: Yes (new files created, scripts to be modified)

### Upstream Viability: No
- Server-aware benchmarking is project-specific (local dev workflow)
- Not suitable for upstream llama.cpp (which has no llama-server concept in same form)
- Should remain in fork-only `scripts/` directory

### Documentation Gaps to Fill

1. **README.md**: Add "Benchmarking" section with server-aware workflow
2. **scripts/README.md**: Create script documentation explaining server detection
3. **RESEARCH_LOG.md**: Log the server-aware benchmarking design decision

### Next Actions
1. Create `scripts/server_check.sh` shared utility (Fixer task)
2. Update all 4 benchmark scripts with server detection (Fixer task)
3. Add cloud benchmarking documentation (Librarian task)
4. Update README.md with benchmarking workflow (Librarian task)
