# ADR 0017: Linear Probe (Logistic Regression) Classifier

**Status**: Accepted

## Context
Using a complex non-linear classifier (e.g., a deep MLP) for downstream node classification could hide poor embedding quality, as the classifier itself "learns" the mapping.

## Decision
Use a **linear probe**: a simple $\ell_2$-regularized Logistic Regression (L-BFGS) fit exclusively on training nodes.

## Rationale
- **Isolation**: Ensures that the Micro-F1 score is a direct proxy for the quality of the embeddings $\mathbf{Z}$.
- **Efficiency**: Training is near-instantaneous.
- **Baseline Comparison**: Aligns with standard evaluation protocols for unsupervised embeddings.

## Consequences
- **Lower Bound**: The reported accuracy is a lower bound on what a powerful supervised model could achieve.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
