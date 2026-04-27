# ADR 0020: Synthetic Graph Community-Based Sampling

**Status**: Accepted

## Context
The original synthetic graph generator used nested loops over all node pairs to ensure community structure, leading to $O(N^2)$ complexity. This made generating 1M node graphs impossible.

## Decision
Implement **community-based sampling**: 
1. Pre-group nodes by community.
2. Sample pairs *within* the same community for intra-edges.
3. Sample pairs *across* different communities for inter-edges.

## Rationale
- **Complexity**: Reduces generation time from $O(N^2)$ to $O(E)$.
- **Preservation**: Maintains the intended community structure (cluster coefficient and modularity) while enabling massive scale.

## Consequences
- **Approximate Edges**: The exact number of edges might vary slightly due to duplicate sampling, but is corrected by using a set.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
