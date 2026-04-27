# ADR 0015: Low-Rank Projection for Top-K Search

**Status**: Accepted

## Context
Calculating all-pairs cosine similarity for $N$ nodes with $D$ features is an $O(N^2 D)$ operation. For a dataset with $100k$ nodes and $128$ features, this involves $\sim 1.2$ billion multiplications. While possible, it is a significant bottleneck when performing hyperparameter sweeps over different $K$ and $\epsilon$ values.

## Decision
Project the $\ell_2$-normalized features into a lower-dimensional space (e.g., $d' = 64$) using a **Random Projection** or **Truncated SVD** before calculating the similarity matrix.

## Rationale
- **Johnson-Lindenstrauss Lemma**: High-dimensional points can be projected into a lower-dimensional space while preserving pairwise distances with a small error $\epsilon$.
- **Performance**: Reduces the constant factor of the $O(N^2)$ search by a factor of $D/d'$.
- **Memory Reduction**: Halves the memory required for temporary similarity buffers.

## Rejected Alternatives
- **Symmetric Hashing (LSH)**: Rejected because LSH is approximate and can miss the "Mutual Top-K" requirement, which needs precise rankings for the top few neighbors.
- **Full-rank computation**: Rejected because it is too slow for iterative tuning and scalability benchmarks.

## Consequences
- **Approximation Error**: The similarity scores used for Top-K selection are now approximations of the original space. However, empirical results show that the rank-order of the top 15 neighbors remains highly stable.
- **Preprocessing Step**: Adds an $O(N \cdot D \cdot d')$ projection step.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
