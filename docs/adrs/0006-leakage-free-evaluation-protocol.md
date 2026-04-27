# ADR 0006: Leakage-Free Evaluation Protocol

**Status**: Accepted

## Context
In link prediction, a common pitfall is "data leakage": if edges intended for the test set are used to build the embedding graph, the model "sees" the answer, leading to artificially inflated AUC/AP scores. In an attributed setting, this is even more dangerous because attribute-derived edges might inadvertently include test edges.

## Decision
Implement a strict **Leakage-Free Protocol**:
1. **Edge Canonicalization**: All edges are stored as $(u, v)$ where $u < v$ to prevent duplicate directed pairs.
2. **Strict Partitioning**: Randomly split positive edges into Train (93%), Validation (2%), and Test (5%) sets.
3. **Restricted Graph Construction**: The Hybrid Graph $\mathcal{G}_H$ is built using **only** training edges.
4. **Predicted Edge Filtering**: Any attribute-derived predicted edge that overlaps with the validation or test sets is explicitly discarded.

## Rationale
- **Scientific Integrity**: Ensures that the link prediction task is a genuine prediction of unseen connectivity.
- **Standardization**: Aligns the evaluation with the rigorous protocols used in VGAE and other SOTA generative models.
- **Reproducibility**: By using a fixed seed for the split, the results can be audited and reproduced.

## Rejected Alternatives
- **Post-hoc Edge Removal**: Removing test edges *after* embedding generation is insufficient because the Louvain hierarchy is already influenced by the test edges.
- **Simple Random Split**: Ignoring the "mutual" nature of undirected edges can lead to leakage where $(u, v)$ is in train but $(v, u)$ is in test.

## Consequences
- **Strict Pipeline**: Requires the data loading and fusion layers to be aware of the edge masks throughout the entire process.
- **Conservative Results**: Reported AUC/AP are lower but scientifically valid.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
