# ADR 0014: L2 Normalization of Input Features

**Status**: Accepted

## Context
The attribute fusion stage relies on calculating the similarity between node feature vectors $\mathbf{x}_u$ and $\mathbf{x}_v$. In datasets like Cora (TF-IDF), some nodes have significantly more words than others, leading to vectors with very different magnitudes.

A raw dot product $\mathbf{x}_u^\top \mathbf{x}_v$ is influenced by both the angle (similarity) and the magnitude (length) of the vectors. This means a node with a very long vector would appear "similar" to almost everything, creating spurious attribute edges.

## Decision
Apply **$\ell_2$-normalization** to all node feature vectors before any similarity computations:
$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

## Rationale
- **Cosine Similarity**: L2 normalization transforms the dot product into the cosine similarity: $\hat{\mathbf{x}}_u^\top \hat{\mathbf{x}}_v = \frac{\mathbf{x}_u^\top \mathbf{x}_v}{\|\mathbf{x}_u\|_2 \|\mathbf{x}_v\|_2}$. This measures the orientation (content) rather than the magnitude.
- **Numerical Stability**: Prevents "exploding" dot product values that could occur with unnormalized high-dimensional features.
- **Fairness**: Ensures that nodes are compared based on their relative feature distribution, not their absolute frequency.

## Rejected Alternatives
- **Min-Max Scaling**: Rejected because it doesn't account for the vector's direction and is sensitive to outliers.
- **Z-Score Normalization**: Rejected because it is usually applied per-feature across nodes, rather than per-node across features.

## Consequences
- **Preprocessing Step**: Adds a $O(N \cdot D)$ normalization pass at the start of the pipeline.
- **Consistency**: Ensures that the similarity scores are always bounded in $[-1, 1]$.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
