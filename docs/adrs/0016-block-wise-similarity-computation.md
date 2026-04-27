# ADR 0016: Block-Wise Similarity Computation

**Status**: Accepted

## Context
A full $N \times N$ similarity matrix cannot fit in RAM for large graphs (e.g., $100k \times 100k$ floats $\approx 40$ GB).

## Decision
Implement **block-wise computation**: divide the node set into blocks of size $B$ (e.g., 2048) and compute similarities for one block against all others, updating Top-K lists incrementally.

## Rationale
- **Constant Memory**: Memory usage becomes $O(N \times K + B \times N)$ instead of $O(N^2)$.
- **Cache Locality**: Better utilizes CPU cache by processing contiguous chunks of memory.

## Consequences
- **Implementation Complexity**: Requires managing indices and partial Top-K heaps.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
