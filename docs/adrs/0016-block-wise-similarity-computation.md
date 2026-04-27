# ADR 0016: Block-Wise Similarity Computation

**Status**: Accepted

## Context
Even with low-rank projection, a full $N \times N$ similarity matrix is too large to fit in RAM. For $N=100,000$, a float32 matrix would require $\sim 40$ GB of contiguous memory, which exceeds the capacity of most commodity machines.

## Decision
Implement **Block-Wise Computation**. Instead of computing the full matrix, the system:
1. Divides the node set into blocks of size $B$ (e.g., $B=2048$).
2. Computes the similarity of one block against all $N$ nodes.
3. Updates the Top-K heaps for the nodes in that block.
4. Discards the block's similarity results and moves to the next.

## Rationale
- **Constant Memory**: Memory usage is reduced from $O(N^2)$ to $O(N \cdot K + B \cdot N)$, allowing the pipeline to scale to millions of nodes.
- **Cache Efficiency**: Processing data in blocks of size 2048 fits better within the L3 cache of modern CPUs, improving the throughput of the matrix multiplication.
- **Parallelism**: This structure naturally allows for multi-threading (each block can be processed by a different core).

## Rejected Alternatives
- **Disk-backed Matrices (Memory Mapping)**: Rejected because random access to a 40GB file on disk is orders of magnitude slower than RAM access.
- **Sparse Similarity**: Rejected because the feature space is often dense (especially after SVD), so we cannot assume the similarity matrix is sparse before computing it.

## Consequences
- **Implementation Complexity**: Requires managing a set of partial Top-K lists (heaps) and coordinate transformations between block-local and global indices.
- **I/O-CPU Balance**: The bottleneck shifts from memory capacity to CPU compute power.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
