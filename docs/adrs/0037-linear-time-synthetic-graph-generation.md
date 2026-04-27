# ADR 0037: Linear-Time Synthetic Graph Generation

**Status**: Accepted

## Context
Generating synthetic graphs with $10^6$ nodes using nested loops to ensure community structure was $O(n^2)$, which hung the system.

## Decision
Implement **community-based sampling** where edges are sampled directly from community lists.

## Rationale
- **Complexity**: Reduces generation time to $O(E)$, allowing for the creation of million-node benchmarks in seconds.
- **Structural Integrity**: Maintains the desired modularity and cluster coefficient by controlling the ratio of intra- vs inter-community samples.

## Consequences
- **Scalability Proof**: Enables the empirical verification of LouvainNE's $O(n \log n)$ scaling claim.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
