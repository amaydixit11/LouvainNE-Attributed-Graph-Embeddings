# ADR 0007: OGB Scalability Adaptation (Edge-Restricted Search)

**Status**: Accepted

## Context
For massive graphs like `ogbn-arxiv` ($n \approx 170k$), calculating all-pairs similarity for Mutual Top-K is $O(n^2)$, which is computationally prohibitive.

## Decision
Adapt the attribute fusion stage to operate **only on observed structural edges**. Instead of a global search, we compute cosine similarity and reweight only for existing $(u, v) \in \mathcal{E}_{struct}$.

## Rationale
- **Complexity Reduction**: Reduces the computation from $O(n^2)$ to $O(|\mathcal{E}|)$, making the pipeline linear in the number of edges.
- **Heuristic Efficiency**: In many real-world graphs, attribute similarity is highly correlated with structural proximity.

## Consequences
- **Coverage Loss**: We may miss attribute-similar nodes that are not structurally connected. However, the trade-off is necessary for scalability to the million-node regime.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
