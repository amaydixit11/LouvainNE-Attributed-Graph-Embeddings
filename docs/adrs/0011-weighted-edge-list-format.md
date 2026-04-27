# ADR 0011: Weighted Edge List Format for C-Pipeline

**Status**: Accepted

## Context
The original LouvainNE implementation was designed for unweighted graphs. However, our "Adaptive Hybrid Graph" relies on confidence weights $w_{uv}$ to distinguish between ground-truth structural edges and predicted attribute edges.

## Decision
Standardize the communication between the Python fusion layer and the C embedding pipeline using a **weighted edge list format**: a plain-text file where each line is `u v w` (source\_id target\_id weight).

## Rationale
- **Algorithmic Necessity**: The Louvain modularity $Q$ is defined over edge weights. To make attributes influence the community moves, the weights must be explicitly passed to the `recpart` binary.
- **Interoperability**: Plain text is the "universal interface" between Python and C, avoiding the complexities of binary serialization (like Protobuf) for a simple list of triples.
- **Debuggability**: Developers can manually inspect the fused graph by reading the text file to verify that weights are being assigned correctly (e.g., checking if overlap edges have $w > 1$).

## Rejected Alternatives
- **Binary Edge Lists**: Rejected because the performance bottleneck is the Louvain algorithm itself, not the I/O of the edge list. The gain from binary I/O would be negligible.
- **In-Memory IPC**: Rejected because it would require rewriting the C-pipeline as a library (linked via `ctypes` or `Cython`), which would significantly increase implementation complexity and risk of memory leaks.

## Consequences
- **Disk I/O Overhead**: For graphs with millions of edges, writing and reading text files can take several seconds. This is mitigated by using buffered I/O and `fscanf` in C.
- **Precision Loss**: Using `float` in text files can lead to precision loss. We use `.12f` formatting in Python to minimize this.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
