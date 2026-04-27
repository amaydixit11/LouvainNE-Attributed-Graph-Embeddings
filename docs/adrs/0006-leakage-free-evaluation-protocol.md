# ADR 0006: Leakage-Free Evaluation Protocol

**Status**: Accepted

## Context
In link prediction, "data leakage" occurs if edges used for testing are visible during the embedding process, leading to artificially inflated AUC/AP scores.

## Decision
Implement a strict **Leakage-Free Protocol**:
1. Canonicalise edges $(u, v)$ with $u < v$.
2. Partition edges into Train/Val/Test *before* any processing.
3. Build the Hybrid Graph using **only** training edges.
4. Discard any attribute-derived predicted edges that overlap with validation or test sets.

## Rationale
- **Scientific Validity**: Ensures that the model is genuinely predicting unseen links rather than remembering them.
- **Reproducibility**: Aligns the evaluation with standard GAE/VGAE protocols.

## Consequences
- **Implementation Rigor**: Requires careful tracking of edge indices across the fusion and embedding stages.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
