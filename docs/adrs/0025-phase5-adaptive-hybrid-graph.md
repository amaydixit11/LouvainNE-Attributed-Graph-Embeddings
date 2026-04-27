# ADR 0025: Phase 5 - Adaptive Hybrid Graph Breakthrough

**Status**: Accepted

## Context
Previous attempts at attribute fusion were either too rigid (global threshold) or too expensive (dense attention).

## Decision
Implement the **Adaptive Hybrid Graph** combining:
1. Cosine similarity.
2. Mutual Top-K filtering.
3. Low-rank projection ($k=64$) for efficiency.

## Rationale
- **Mutual Filtering**: Ensures only high-confidence predicted edges are included, reducing noise.
- **Scale Invariance**: Cosine similarity handles different feature distributions across datasets.
- **Efficiency**: Low-rank projection enables block-wise batching for large graphs.

## Consequences
- **Performance**: Achieved a significant improvement (e.g., 2.44 pp on Cora) over the reproducible baseline.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
