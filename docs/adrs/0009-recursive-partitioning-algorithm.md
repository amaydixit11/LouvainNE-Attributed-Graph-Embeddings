# ADR 0009: Recursive Partitioning via Louvain (recpart)

**Status**: Accepted

## Context
To capture multi-scale structural information, we need a hierarchy of communities rather than a single flat partition.

## Decision
Use the Louvain algorithm's natural hierarchical property: after a level of optimization, nodes in the same community are collapsed into a single "super-node," and the process repeats. We record the community ID for every node at every level of this collapse.

## Rationale
- **Multi-Resolution**: Captures both local micro-structures and global macro-structures.
- **Efficiency**: Louvain is $O(n \log n)$, making it the only viable option for large-scale hierarchical clustering.

## Consequences
- **Storage**: The resulting hierarchy file grows with the number of levels $L$, though $L$ is typically small ($\log n$).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
