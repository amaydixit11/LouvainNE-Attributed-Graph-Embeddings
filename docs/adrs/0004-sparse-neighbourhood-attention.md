# ADR 0004: Sparse Neighbourhood Attention

**Status**: Accepted

## Context
The raw embeddings produced by the LouvainNE pipeline can be coarse because they are based on discrete community assignments. To refine these representations and "smooth" the boundaries between communities, an attention mechanism is desired.

## Decision
Implement **Sparse Neighbourhood Attention**, where the attention weight $a_{uv}$ for node $v$ is computed only over its immediate neighbours in the Hybrid Graph $\mathcal{G}_H \cup \{v\}$.

The refined embedding $\tilde{z}_v$ is a residual combination:
$$\tilde{z}_v = (1 - \gamma) z_v + \gamma \sum_{u \in \mathcal{N}(v) \cup \{v\}} a_{uv} z_u$$

## Rationale
- **Complexity**: Dense self-attention (calculating $N \times N$ attention weights) scales at $O(n^2)$, which is infeasible for graphs larger than a few thousand nodes (Phase 4 failure). Sparse attention scales at $O(|\mathcal{E}_H|)$, maintaining efficiency.
- **Inductive Bias**: By restricting attention to the hybrid neighborhood, we assume that a node's representation should be most influenced by peers that are already structurally or attribute-similar.
- **Boundary Refinement**: The attention mechanism acts as a low-pass filter, reducing the "steppiness" of the hierarchical embeddings and improving downstream classification.

## Rejected Alternatives
- **Dense Self-Attention**: Rejected due to the $O(n^2)$ memory wall.
- **GNN-style Message Passing**: Rejected because it would require training weights and multiple iterations, violating the "training-free" paradigm.

## Consequences
- **Hyperparameter Introduction**: Introduces $\gamma$ (smoothing coefficient) and $\tau$ (temperature), which must be tuned.
- **Refinement Gain**: Empirically improves Micro-F1 by smoothing embeddings without introducing significant runtime overhead.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
