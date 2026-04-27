# ADR 0027: Phase 7 - OGB Scalability Adaptations

**Status**: Accepted

## Context
The $O(n^2)$ complexity of global similarity search makes the standard Mutual Top-K approach infeasible for OGB-scale graphs (e.g., ogbn-arxiv).

## Decision
Restrict the attribute-similarity search to **observed structural edges**. We only reweight existing edges based on cosine similarity.

## Rationale
- **Computational Feasibility**: Reduces complexity from $O(n^2)$ to $O(|\mathcal{E}|)$.
- **Heuristic Accuracy**: In large scientific networks, attribute similarity is often strongly correlated with existing structural links.

## Consequences
- **Linear Scaling**: Enables the pipeline to run on graphs with hundreds of thousands of nodes in seconds.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
