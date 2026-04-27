# ADR 0012: Determinism via Explicit Seeding

**Status**: Accepted

## Context
The C-pipeline originally used `srand(time(NULL))` for initializing the random number generator used in `recpart` (for node movement order) and `hi2vec` (for community vector generation). This meant that two runs on the same graph would produce different embeddings, making it impossible to perform controlled ablation studies.

## Decision
Remove all `time(NULL)` calls and implement **explicit CLI-controlled seeding**. The binaries now accept a `--seed` (or positional) argument that is passed directly to `srand()`.

## Rationale
- **Scientific Reproducibility**: In research, the ability to reproduce the exact same result is mandatory. Explicit seeds allow us to report "mean $\pm$ std" across a fixed set of seeds.
- **Bug Isolation**: When a crash occurs (e.g., the community collapse bug), an explicit seed allows the developer to recreate the exact state and sequence of events that led to the failure.
- **Comparison Fairness**: Ensures that improvements in accuracy are due to algorithmic changes, not "lucky" random initialization.

## Rejected Alternatives
- **Fixed Global Seed**: Rejected because we need to run ensembles (multiple different seeds) to average out the variance of the Louvain algorithm.
- **Deterministic Partitioning**: Rejected because Louvain's performance actually improves when nodes are processed in a random order (preventing local minima).

## Consequences
- **CLI Change**: The binaries now have a different argument signature, requiring updates to the Python `LouvainNERunner` wrapper.
- **Seed Management**: The orchestration scripts must now track and report which seeds were used for each run.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
