# ADR 0015: Low-Rank Projection for Top-K Search

**Status**: Accepted

## Context
Calculating all-pairs cosine similarity for $N$ nodes with $D$ features is $O(N^2 D)$. For $N=100k$ and $D=128$, this is computationally expensive.

## Decision
Project features into a lower-dimensional space ($k=64$) using a random projection or SVD before performing the Top-K search.

## Rationale
- **Memory Reduction**: Halves the memory required for similarity matrices.
- **Speed**: Reduces the constant factor of the $O(N^2)$ search without significantly impacting the rank-order of the Top-K neighbors.

## Consequences
- **Approximation**: The similarity is now an approximation of the original space, though empirical results show this is negligible for Top-K selection.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
