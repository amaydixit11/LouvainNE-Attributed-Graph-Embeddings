# ADR 0002: Early Fusion for Attribute Integration

**Status**: Accepted

## Context
Integrating node attributes into a structural embedding pipeline can be done at different stages:
1. **Late Fusion**: Generate structural and attribute embeddings separately, then concatenate.
2. **Early Fusion**: Integrate attributes into the graph topology *before* embedding.

## Decision
We use **Early Fusion**, specifically augmenting the structural graph with attribute-derived edges to create a "Hybrid Graph."

## Rationale
- **Empirical Superiority**: Phase 3 testing showed that late fusion performed worse than pure structural baselines, as attribute-only embeddings from unstructured similarity graphs are noisy.
- **Synergy**: Integrating attributes into the topology allows the Louvain algorithm to discover communities that are both structurally and attribute-consistent.

## Consequences
- **Complexity**: Increases the initial graph construction time (calculating similarities).
- **Noise Risk**: Poorly chosen attribute edges can distort the structural signal, necessitating a rigorous filtering mechanism (see ADR 0005).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
