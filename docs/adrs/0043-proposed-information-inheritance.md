# ADR 0043: Proposed Information Inheritance (Future)

**Status**: Proposed

## Context
`hi2vec` currently uses a bottom-up sum of random vectors.

## Decision
Replace static generation with **downward propagation**, where embeddings are learned at the coarse level and propagated down to the leaves.

## Rationale
- **Consistency**: Ensures that children inherit a smoothed version of their parent's signal, mimicking the "information inheritance" seen in HireGC.

## Consequences
- **Iterative Process**: Changes the embedding generation from a single pass to an iterative refinement process.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
