# ADR 0003: Mutual Top-K Attribute Filtering

**Status**: Accepted

## Context
Early fusion requires adding edges between nodes with similar attributes. A naive approach uses a global similarity threshold (e.g., $\text{sim}(u, v) > 0.5$). However, feature distributions vary wildly across datasets (e.g., binary features in BlogCatalog vs. TF-IDF in Cora), making a single scalar threshold non-generalizable.

## Decision
Implement **Mutual Top-K filtering**: An edge $(u, v)$ is added to the Hybrid Graph if and only if:
1. Node $u$ is among the top-$K$ most similar nodes to $v$.
2. Node $v$ is among the top-$K$ most similar nodes to $u$.
3. The similarity $\text{sim}(u, v)$ exceeds a minimum threshold $\epsilon$ (to filter out completely dissimilar nodes in sparse feature spaces).

## Rationale
- **Scale Invariance**: The Top-K criterion is relative to the local neighborhood of each node, making the filter robust to different feature scales and distributions.
- **Symmetry and Confidence**: Mutual filtering ensures that the resulting graph is undirected and excludes "hubs"—nodes that are similar to many others but are not reciprocally similar—which significantly reduces noise injection.
- **Complexity Control**: Fixing $K$ (e.g., $K=15$) provides a hard upper bound on the number of attribute edges added per node ($|\mathcal{E}_{attr}| \le N \cdot K$), ensuring the $O(n \log n)$ scaling of Louvain is preserved.

## Rejected Alternatives
- **Global Similarity Thresholding**: Rejected because it is dataset-sensitive and fails to generalize (Phase 2 failure).
- **Unilateral Top-K**: Rejected because it creates directed-like asymmetries and allows high-degree "attribute hubs" to distort the community structure.

## Consequences
- **Computational Overhead**: Requires calculating all-pairs similarity (or block-wise) and sorting for each node, introducing an $O(N^2)$ or $O(N \cdot \text{block})$ bottleneck.
- **Hyperparameter Dependency**: The choice of $K$ and $\epsilon$ becomes a critical tuning point for accuracy.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
