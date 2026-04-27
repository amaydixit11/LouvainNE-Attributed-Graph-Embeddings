# ADR 0018: Negative Sampling for Link Prediction

**Status**: Accepted

## Context
Link prediction is a binary classification task. While positive edges are given, negative examples (non-edges) must be sampled.

## Decision
Implement **random uniform negative sampling**: draw node pairs $(u, v)$ that are not present in the full graph.

## Rationale
- **Standard Practice**: Aligns with the VGAE and Node2Vec evaluation protocols.
- **Simplicity**: Avoids the bias introduced by "hard" negative mining which can overfit to specific structural gaps.

## Consequences
- **Sample Ratio**: Typically uses a 1:1 ratio of positive to negative edges for the test set.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
