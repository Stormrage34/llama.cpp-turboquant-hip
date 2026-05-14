# Git Workflow for RDNA2 Contributors

> **Status**: Phase 0 complete (backups created). Phase 1 (bisect branch) deferred pending conflict resolution.

## Current Branch Structure

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Production code + all infrastructure | ✅ Active, validated |
| `feat/upstream-bfe-fence` | Upstream PR draft + model download | ✅ Merged to main |
| `feat/rdna2-bisect-safe` | Clean bisectable branch (from upstream/master) | ⏸️ Needs conflict resolution |
| `backup/v0.3.2-pre-restructure` | Immutable backup before any git surgery | ✅ Safety net |
| `backup-pre-restructure` | Pre-restructure backup (alias) | ✅ Safety net |

## Tags

| Tag | Commit | Purpose |
|-----|--------|---------|
| `v0.3.1-stable` | `60dedded` | Production baseline (MoE prefill +110%, ±0.17% variance) |
| `v0.3.1-stable-final` | `ae8085b73` | Final state before git hygiene work |
| `v0.3.2-p1` | Phase 1 telemetry | Baseline counters |
| `v0.3.2-p2` | Phase 2 DPP | DPP scale broadcast (reverted) |
| `v0.3.2-p3` | Phase 3 BFE | BFE dispatcher framework |

## Development Workflow (Enforced)

### Rule 1: Never push directly to `main`
```bash
# Create feature branch
git checkout -b feat/my-feature main

# Develop, commit, test
git commit -m "feat(rdna2): description"

# Merge via --no-ff (preserves history)
git checkout main
git merge --no-ff feat/my-feature -m "chore: merge my-feature"
```

### Rule 2: Sync upstream weekly
```bash
git fetch upstream --tags
git merge upstream/master --no-ff  # or rebase if clean
```

### Rule 3: Atomic commits only
- One fix/feature per commit
- Use `feat:`, `fix:`, `chore:`, `docs:` prefixes
- Amend before push if needed (`git commit --amend`)

### Rule 4: Tag every release
```bash
git tag -a v0.x.x -m "Description of what changed"
```

## Bisect Branch Status

The `feat/rdna2-bisect-safe` branch was created from `upstream/master` but cherry-picking the 10+ critical RDNA2 commits produced significant conflicts (upstream `convert.cu` and `fattn-common.cuh` have diverged).

**Resolution plan** (deferred):
1. Resolve conflicts manually for each cherry-pick
2. Verify build after each commit
3. Run smoke test: `llama-bench -m model.gguf -ngl 99 -c 4096 -p 512 -n 128 -r 5`
4. Document in this file when bisect branch is ready

**Current workaround for debugging**:
- Use `git log --oneline -- ggml/src/ggml-cuda/mmq.cuh` to trace changes to specific files
- Use `git diff <commit>^..<commit> -- <file>` to see what changed
- Tag-based debugging: `v0.3.1-stable` is known-good, `v0.3.0-experimental` is known-bad

## Patch Export (For Upstream PR)

```bash
# Export RDNA2-specific commits as patches
git format-patch 36a694c9..main --output-directory rdna2-patches --subject-prefix="RDNA2"

# Apply patches to fresh branch (for upstream PR)
git checkout -b feat/upstream-bfe-clean upstream/master
git am --signoff --3way rdna2-patches/*.patch
```

## Rollback Procedure

If anything goes wrong:
```bash
# Return to production baseline
git checkout main
git reset --hard backup/v0.3.2-pre-restructure

# Or restore specific files
git checkout backup/v0.3.2-pre-restructure -- <file>
```