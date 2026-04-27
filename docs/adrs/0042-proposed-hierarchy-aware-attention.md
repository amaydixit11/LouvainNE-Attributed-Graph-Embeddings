# ADR 0042: Proposed Hierarchy-Aware Attention (Future)

**Status**: Proposed

## Context
The current Sparse Attention only looks at the immediate neighbourhood in $\mathcal{G}_H$.

## Decision
Implement **Attention over Hierarchy**, where a node attends to:
- Local neighbours.
- Community-level peers.
- Super-community representatives.

## Rationale
- **Multi-Scale Signal**: Allows the model to dynamically weigh local vs global context depending on the node's position in the graph.

## Consequences
- **Complexity**: Increases the attention computation to $O(|\mathcal{E}_H| + L \times N)$.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
