# ADR 0029: Sparse Attention Hyperparameters ($\gamma, \tau$)

**Status**: Accepted

## Context
The sparse attention mechanism uses a temperature parameter $\tau$ and a residual interpolation coefficient $\gamma$ to refine embeddings.

## Decision
Set $\tau = 1.0$ and $\gamma = 0.5$ as defaults based on validation.

## Rationale
- **Smoothing ($\gamma$)**: $\gamma=0.5$ provides an even balance between the raw Louvain embedding and the neighbourhood-pooled representation.
- **Temperature ($\tau$)**: $\tau=1.0$ maintains the relative distribution of similarity scores without over-sharpening or over-smoothing the attention weights.

## Consequences
- **Stability**: These parameters proved robust across the Cora and CiteSeer datasets.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
