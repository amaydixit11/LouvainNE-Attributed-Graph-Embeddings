# ADR 0022: Phase 2 - Global Thresholding (Abandoned)

**Status**: Abandoned

## Context
An early attempt to integrate attributes involved computing raw feature dot-product similarity for all pairs and applying a global threshold to decide which pairs receive an attribute edge.

## Decision
This approach was abandoned in favor of local, scale-invariant filtering (Mutual Top-K).

## Rationale
- **Generalization Failure**: The optimal threshold for Cora performed poorly on PubMed's TF-IDF and BlogCatalog's binary features.
- **Information Loss**: Flattening all accepted edges to weight 1 ignored the magnitude of similarity.

## Consequences
- **Pivot**: Led to the development of the Adaptive Hybrid Graph (ADR 0003).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
