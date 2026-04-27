# ADR 0035: Super-Node Representation in Hierarchy

**Status**: Accepted

## Context
The Louvain algorithm recursively collapses communities into "super-nodes" to build the hierarchy.

## Decision
Treat every community in the hierarchy tree as a **single entity (super-node)** with its own associated random vector, regardless of how many original nodes it contains.

## Rationale
- **Abstraction**: Simplifies the embedding process by treating the hierarchy as a tree of entities rather than a set of overlapping node groups.
- **Efficiency**: Keeps the `hi2vec` complexity linear in the number of communities, not the number of nodes.

## Consequences
- **Granularity**: The representation is inherently discrete at each level; node-level nuances are only captured at the leaf level (Level 0).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
