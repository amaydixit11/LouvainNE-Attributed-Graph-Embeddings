# ADR 0023: Phase 3 - Late Fusion (Abandoned)

**Status**: Abandoned

## Context
We explored constructing independent structure-only and attribute-only LouvainNE embeddings and combining them via concatenation or summation.

## Decision
This approach was abandoned in favor of Early Fusion.

## Rationale
- **Poor Performance**: Late-fusion concatenation performed worse than the pure structure baseline.
- **Noise**: Attribute embeddings from unstructured similarity graphs are inherently noisy; merging them post-hoc does not leverage the structural context.

## Consequences
- **Conclusion**: Confirmed that integrating attributes *into the graph* before running Louvain is strictly superior.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
