# ADR 0040: Proposed SAGES-Style Sampling (Future)

**Status**: Proposed

## Context
The current Louvain community detection is static and deterministic.

## Decision
Explore replacing static communities with **guided sampling** where Louvain communities define the sampling distribution for training embeddings.

## Rationale
- **Hybridization**: Combines the global structural awareness of Louvain with the local refinement of sampling-based methods (like GraphSAGE).

## Consequences
- **Shift to Training**: This would move the system from "training-free" to "guided training," potentially increasing accuracy at the cost of speed.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
