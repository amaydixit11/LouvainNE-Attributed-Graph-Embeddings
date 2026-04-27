# ADR 0039: Link Prediction Split Ratios (93/2/5)

**Status**: Accepted

## Context
Standard evaluation for link prediction requires a separation of training, validation, and test edges.

## Decision
Use a **93% Train / 2% Validation / 5% Test** split for positive edges.

## Rationale
- **Stability**: A larger training set (93%) ensures the structural embedding is as complete as possible.
- **Statistical Significance**: 5% of edges in large graphs (like Cora/CiteSeer) provide enough test samples for stable AUC/AP calculations.

## Consequences
- **Consistency**: Aligns with the VGAE evaluation protocol.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
