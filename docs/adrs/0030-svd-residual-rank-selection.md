# ADR 0030: SVD Residual Rank Selection ($r=128$)

**Status**: Accepted

## Context
The attribute residual branch uses a truncated SVD to provide a compact linear representation of the features.

## Decision
Fix the SVD rank at $r = 128$.

## Rationale
- **Dimensionality Match**: Matches the common dimensionality of the structural embeddings, preventing one branch from dominating the linear probe.
- **Variance Capture**: For most citation datasets, the top 128 singular vectors capture the vast majority of the feature variance.

## Consequences
- **Efficiency**: Keeps the final embedding size manageable while preserving high-signal linear information.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
