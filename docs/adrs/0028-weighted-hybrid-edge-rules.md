# ADR 0028: Weighted Hybrid Edge Logic

**Status**: Accepted

## Context
When fusing structural and attribute edges, a simple binary (exist/not-exist) approach loses the confidence signal of the similarity metric.

## Decision
Implement a weighted fusion rule:
- **Overlap Edges** (exist in both): Weight = $1 + \text{similarity}$
- **Pure Structural Edges**: Weight = $1$
- **Pure Attribute Edges**: Weight = $0.75 \times \text{similarity}$

## Rationale
- **Prioritization**: Structural edges are treated as ground truth (base weight 1).
- **Reinforcement**: Edges supported by both structure and attributes are given the highest priority.
- **Confidence**: Attribute-only edges are discounted by a factor (0.75) to prevent them from dominating the modularity optimization.

## Consequences
- **Modularity Influence**: The Louvain algorithm's community movement is now guided by the confidence of the connection.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
