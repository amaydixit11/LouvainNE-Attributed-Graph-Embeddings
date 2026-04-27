# ADR 0003: Mutual Top-K Attribute Filtering

**Status**: Accepted

## Context
Early fusion requires adding edges between nodes with similar attributes. Simple global thresholding (Phase 2) failed to generalize across datasets because feature distributions vary.

## Decision
Implement **Mutual Top-K filtering**: An edge $(u, v)$ is added to the Hybrid Graph if and only if $u \in \text{TopK}(v)$ AND $v \in \text{TopK}(u)$ based on cosine similarity.

## Rationale
- **Scale Invariance**: Top-K is relative to the local distribution of each node, making it more robust than a global scalar threshold.
- **Symmetry**: Mutual filtering ensures the resulting graph is undirected and suppresses "hub" nodes that might be similar to many nodes but are not reciprocally similar.
- **Noise Reduction**: Significantly reduces the number of spurious edges compared to simple Top-K.

## Consequences
- **Computation**: Requires computing and sorting similarity lists for every node.
- **Memory**: Temporary storage of Top-K lists is required before final edge fusion.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
