# ADR 0004: Sparse Neighbourhood Attention

**Status**: Accepted

## Context
Raw LouvainNE embeddings can be coarse. Dense self-attention (Phase 4) could refine these but scales at $O(n^2)$, making it infeasible for graphs larger than a few thousand nodes.

## Decision
Implement **Sparse Neighbourhood Attention**, where the attention mechanism is restricted to the immediate neighbourhood in the Hybrid Graph $\mathcal{G}_H$.

## Rationale
- **Efficiency**: Reduces complexity from $O(n^2)$ to $O(|\mathcal{E}_H|)$, enabling scaling to millions of nodes.
- **Inductive Bias**: Graph-based attention assumes that a node's representation should be most influenced by its local structural and attribute-driven peers.

## Consequences
- **Refinement**: Provides a "smoothing" effect that improves node classification and link prediction metrics without adding training parameters.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
