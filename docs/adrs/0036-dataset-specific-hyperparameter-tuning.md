# ADR 0036: Dataset-Specific Hyperparameter Tuning

**Status**: Accepted

## Context
The pipeline involves several critical hyperparameters: $\lambda, \beta, K, \gamma, r$.

## Decision
Conduct **grid searches and weighted scoring** (e.g., $0.7 \times \text{NodeAcc} + 0.3 \times \text{LinkAUC}$) to find optimal configurations for each dataset.

## Rationale
- **Distribution Shift**: Cora (citation) and BlogCatalog (binary membership) have fundamentally different attribute and structural distributions.
- **Trade-offs**: Parameters that maximize node classification may not necessarily maximize link prediction.

## Consequences
- **Non-Generalization**: A "one size fits all" configuration is likely to be sub-optimal, requiring a tuning phase before final benchmarking.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
