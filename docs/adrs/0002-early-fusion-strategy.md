# ADR 0002: Early Fusion for Attribute Integration

**Status**: Accepted

## Context
The core LouvainNE algorithm is structural. To handle attributed graphs, we must integrate node features $\mathbf{X}$ into the embedding process. There are two primary ways to do this:
1. **Late Fusion**: Generate structural embeddings $\mathbf{Z}_{struct}$ and attribute embeddings $\mathbf{Z}_{attr}$ independently, then merge them (e.g., $\mathbf{Z} = [\mathbf{Z}_{struct} \parallel \mathbf{Z}_{attr}]$).
2. **Early Fusion**: Modify the graph topology itself by adding edges based on attribute similarity *before* running the embedding algorithm.

## Decision
We implement **Early Fusion**, specifically augmenting the structural graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with attribute-derived edges $\mathcal{E}_{attr}$ to create a Hybrid Graph $\mathcal{G}_H = (\mathcal{V}, \mathcal{E} \cup \mathcal{E}_{attr}, \mathbf{w})$.

## Rationale
- **Empirical Superiority**: Testing in Phase 3 demonstrated that late fusion often performs worse than pure structural baselines. This is because attribute-only embeddings generated from unstructured similarity graphs are noisy and lack the topological constraints that make LouvainNE effective.
- **Topological Synergy**: Early fusion allows the Louvain algorithm to discover communities that are *simultaneously* structurally connected and attribute-similar, creating a more coherent partitioning of the feature space.
- **Information Flow**: Attributes influence the community hierarchy at every level of the recursive collapse, rather than being a post-hoc additive.

## Rejected Alternatives
- **Late Fusion (Concatenation/Summation)**: Rejected because it doubles the embedding dimension without providing complementary structural information, often introducing noise that degrades linear probe accuracy.
- **Feature-based Louvain (Modularity Modification)**: We considered modifying the Louvain modularity function $Q$ to include an attribute term $\beta \times Q_{attr}$. However, structural reweighting (Early Fusion) proved more stable and easier to tune.

## Consequences
- **Graph Expansion**: The number of edges $|\mathcal{E}_H|$ is larger than $|\mathcal{E}|$, slightly increasing the runtime of the Louvain phase.
- **Noise Sensitivity**: The quality of the embeddings becomes highly dependent on the attribute-edge generation logic (filtered by Mutual Top-K, see ADR 0003).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
