# ADR 0005: SVD Attribute-Residual Branch

**Status**: Accepted

## Context
The Louvain community-detection pathway is inherently lossy. It compresses high-dimensional feature space into discrete IDs and then maps them back to vectors. This process effectively removes fine-grained linear signals that might be critical for classification but aren't captured by the community structure.

## Decision
Implement a parallel **SVD Attribute-Residual branch**. We compute a rank-$r$ Truncated SVD of the centered feature matrix $\mathbf{X}$ to obtain a compact linear projection $\mathbf{F} \in \mathbb{R}^{N \times r}$. The final embedding is the concatenation:
$$\mathbf{e}_v = [\tilde{z}_v \parallel \mathbf{f}_v]$$

## Rationale
- **Linear Signal Preservation**: The SVD branch provides the downstream classifier with direct access to the most significant linear components of the feature space, bypassing the lossy community-detection path.
- **Complementarity**: The structural branch (Louvain) captures non-linear, topological clusters, while the residual branch captures global linear variance.
- **Efficiency**: For a fixed rank (e.g., $r=128$), SVD is computationally efficient and doesn't require iterative training.

## Rejected Alternatives
- **Raw Feature Concatenation**: Rejected because the original feature dimension (e.g., 8192 in some datasets) is too large, leading to the "curse of dimensionality" and overfitting in the linear probe.
- **Autoencoder**: Rejected because it requires training and GPU resources.

## Consequences
- **Dimension Increase**: The embedding dimensionality increases from $D$ to $D+r$.
- **Information Balance**: The linear probe must now balance the signal from the structural and residual branches.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
