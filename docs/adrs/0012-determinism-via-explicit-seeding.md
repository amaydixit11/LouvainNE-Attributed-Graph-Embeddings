# ADR 0012: Determinism via Explicit Seeding

**Status**: Accepted

## Context
The C-pipeline originally used `time(NULL)` for seeding random number generators, making results non-reproducible across runs.

## Decision
Replace all instances of `time(NULL)` with explicit CLI-controlled seeds in `recpart.c` and `hi2vec.c`.

## Rationale
- **Scientific Rigor**: Reproducibility is mandatory for benchmarking and ablation studies.
- **Debugging**: Allows for isolating bugs by recreating the exact state of a failed run.

## Consequences
- **CLI Complexity**: Adds `--seed` arguments to the C binaries.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
