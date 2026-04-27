# ADR 0034: Exponential Decay for Hierarchical Aggregation

**Status**: Accepted

## Context
A node $v$ belongs to a sequence of communities $c_0, c_1, \dots, c_L$ across $L$ levels of the hierarchy. We must aggregate these into a single vector.

## Decision
Use **exponentially decaying weights** $w_\ell = e^{-\lambda \ell}$ for the aggregation: $z_v = \sum_{\ell=0}^L w_\ell r_{c_\ell(v)}$.

## Rationale
- **Scale Control**: $\lambda$ allows us to tune whether the embedding focuses on local micro-communities (high $\lambda$) or global macro-structures (low $\lambda$).
- **Hierarchical Prior**: Reflects the intuition that the most specific community (level 0) usually contains the most precise identity signal.

## Consequences
- **Hyperparameter Tuning**: $\lambda$ becomes a critical parameter that must be tuned per dataset.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
