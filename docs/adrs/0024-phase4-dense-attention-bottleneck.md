# ADR 0024: Phase 4 - Dense Self-Attention (Abandoned)

**Status**: Abandoned

## Context
To refine LouvainNE embeddings, we attempted to apply a dense attention layer using the raw feature matrix as key/query.

## Decision
Abandoned due to $O(n^2)$ memory and time complexity.

## Rationale
- **Scalability**: While marginally better than unweighted early fusion, it became infeasible for datasets like PubMed ($n \approx 20k$) and BlogCatalog.

## Consequences
- **Pivot**: Led to the implementation of Sparse Neighbourhood Attention (ADR 0004).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
