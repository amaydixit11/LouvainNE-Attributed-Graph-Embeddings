# ADR 0010: Hierarchical Vector Aggregation (hi2vec)

**Status**: Accepted

## Context
We need to transform the community hierarchy (a set of IDs) into a fixed-dimensional embedding vector.

## Decision
Implement **weighted aggregation of random community vectors**:
1. Assign a random $D$-dimensional vector to every community at every level.
2. For node $v$, its embedding $z_v$ is the normalized sum: $\sum_{l=0}^L w_l r_{c_l(v)}$, where $w_l$ is an exponentially decaying weight.

## Rationale
- **Consistency**: Nodes in the same community at level $l$ share the same component of the embedding.
- **Tunability**: The decay parameter $\lambda$ allows us to control whether the embedding emphasizes local or global structure.

## Consequences
- **Randomness**: Requires fixed seeds for the random vector generation to ensure reproducibility.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
