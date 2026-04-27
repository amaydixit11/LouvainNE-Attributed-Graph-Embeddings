# ADR 0033: Random Vector Initialization for Communities

**Status**: Accepted

## Context
The `hi2vec` stage needs to map discrete community IDs at each level of the hierarchy to vectors in $\mathbb{R}^D$.

## Decision
Assign **independent random vectors** (sampled from a uniform distribution $[-1, 1]$) to every community at every level of the hierarchy.

## Rationale
- **Curse of Dimensionality**: In high-dimensional space, random vectors are naturally nearly orthogonal, ensuring that different communities are represented by distinct signals.
- **Training-Free**: Eliminates the need for a learned codebook or an expensive training phase.

## Consequences
- **Requirement for Seeds**: Necessitates a global seed for the random number generator to ensure that the same community ID always maps to the same vector across different runs.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
