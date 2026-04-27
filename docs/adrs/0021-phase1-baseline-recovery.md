# ADR 0021: Phase 1 - Baseline Recovery and Reproducibility

**Status**: Accepted

## Context
The original LouvainNE repository was found to be non-self-contained. The pre-compiled `recpart` binary was stale relative to the C source, and Python scripts were writing unweighted edge lists while the binary expected weighted ones, leading to non-reproducible and silent failures.

## Decision
Rewrite the build pipeline to compile LouvainNE from source on every run and patch the Python writer to emit `u v w` triples.

## Rationale
- **Integrity**: Ensures that the executed code matches the source.
- **Correctness**: Fixes the silent ignoring of edge weight information.

## Consequences
- **Build Time**: Adds a small overhead to the setup phase to compile the C binaries.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
