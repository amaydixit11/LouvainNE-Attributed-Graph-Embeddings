# ADR 0007: OGB Scalability Adaptation (Edge-Restricted Search)

**Status**: Accepted

## Context
The standard Mutual Top-K fusion requires an all-pairs similarity search, which is $O(n^2)$. For OGB datasets like `ogbn-arxiv` ($n \approx 169,343$), this would require calculating $\sim 28$ billion similarities, which is computationally prohibitive even with block-wise processing.

## Decision
Adapt the attribute fusion stage to operate **only on observed structural edges**. Instead of a global search, we compute the cosine similarity $s_{uv}$ only for $(u, v) \in \mathcal{E}_{struct}$ and use this to reweight the existing edges.

## Rationale
- **Linear Complexity**: Reduces the attribute-graph computation from $O(n^2)$ to $O(|\mathcal{E}|)$, making the pipeline truly linear in the number of edges.
- **Heuristic Efficiency**: In large-scale scientific networks, attribute similarity is highly concentrated around existing structural neighbors.
- **Practicality**: This is the only way to process graphs with $10^5$ to $10^6$ nodes on commodity CPU hardware.

## Rejected Alternatives
- **Approximate Nearest Neighbors (ANN)**: Using libraries like Faiss could reduce $O(n^2)$, but it introduces external dependencies and may still be too slow for million-node graphs given the need for mutual filtering.
- **Random Sampling**: Sampling a subset of nodes for similarity search would miss critical connections in sparse graphs.

## Consequences
- **Coverage Loss**: The pipeline can no longer "predict" new edges that aren't already structural edges. It becomes a "reweighting" pipeline rather than an "augmenting" pipeline for OGB-scale graphs.
- **Accuracy Shift**: May result in a slight drop in link prediction AUC compared to full Mutual Top-K, but is the only feasible path for scalability.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
