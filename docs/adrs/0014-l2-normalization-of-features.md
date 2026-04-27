# ADR 0014: L2 Normalization of Input Features

**Status**: Accepted

## Context
Node features (e.g., TF-IDF) can have vastly different magnitudes, which distorts the dot-product similarity used for attribute fusion.

## Decision
Apply $\ell_2$-normalization to all node feature vectors before calculating similarities.

## Rationale
- **Cosine Similarity**: L2 normalization turns the dot product into cosine similarity, which measures the angle (orientation) rather than the magnitude.
- **Numerical Stability**: Prevents high-magnitude features from dominating the Top-K selection.

## Consequences
- **Preprocessing Step**: Adds a constant-time normalization pass over the feature matrix.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
