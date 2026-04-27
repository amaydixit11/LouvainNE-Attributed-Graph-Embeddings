# ADR 0041: Proposed Multi-Loss Objective (Future)

**Status**: Proposed

## Context
The current pipeline is a feed-forward transformation with no internal loss function.

## Decision
Research a **multi-loss objective** that simultaneously optimizes for:
1. Structural reconstruction.
2. Attribute reconstruction.
3. Community consistency.

## Rationale
- **Joint Optimization**: Forcing the embeddings to reconstruct both attributes and structure would prevent the "lossy" nature of the Louvain path.

## Consequences
- **Architecture Change**: Requires moving from a purely algorithmic pipeline to a differentiable one.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
