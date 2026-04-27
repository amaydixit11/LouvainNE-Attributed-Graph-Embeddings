# ADR 0005: SVD Attribute-Residual Branch

**Status**: Accepted

## Context
The Louvain community-detection pathway is inherently lossy—it collapses high-dimensional feature space into discrete community IDs and then back into vectors. Some fine-grained linear feature information is lost in this process.

## Decision
Add a parallel **SVD Attribute-Residual branch**: Compute a rank-$r$ SVD of the centered feature matrix and concatenate the resulting low-rank projection to the structural embedding.

## Rationale
- **Information Preservation**: Provides the downstream classifier with direct access to the most significant linear components of the feature space.
- **Complementarity**: The Louvain branch captures non-linear community structure, while the SVD branch captures linear global variance.

## Consequences
- **Embedding Dimension**: Increases the final embedding size from $D$ to $D+r$.
- **Computational Cost**: Adds an SVD step, though this is efficient for the chosen rank ($r=128$).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
