# ADR 0001: Training-Free CPU-Bound Paradigm

**Status**: Accepted

## Context
Modern Graph Neural Networks (GNNs) such as GCN, GAT, and GraphSAGE provide high accuracy for node classification and link prediction. However, they rely on iterative message-passing and gradient-based optimization, which imposes a heavy "GPU Tax":
- **Memory Bottleneck**: Storing gradients and activations for large graphs often exceeds typical VRAM limits (e.g., GAT OOMs on ogbn-arxiv on standard GPUs).
- **Time Complexity**: Training requires multiple epochs over the entire graph, leading to runtimes in the order of hours for massive networks.
- **Dependency**: Tight coupling with CUDA/GPU drivers limits deployment on commodity CPU hardware.

## Decision
We adopt a **training-free, CPU-bound embedding pipeline** that replaces iterative learning with hierarchical community detection (Louvain algorithm) and random vector aggregation.

## Rationale
- **Hardware Independence**: The system is entirely CPU-bound, invoking no GPU operations. This allows it to run on any commodity machine without CUDA dependencies.
- **Execution Speed**: By removing backpropagation and epoch-based training, the time complexity is reduced from $O(E \cdot d \cdot \text{epochs})$ to approximately $O(n \log n)$ for the Louvain phase, reducing runtime from hours to seconds.
- **Determinism**: Unlike SGD, which is stochastic, a fixed-seed Louvain pipeline is fully deterministic, making the embeddings highly reproducible across different environments.

## Rejected Alternatives
- **Supervised GNNs**: Rejected due to the GPU requirement and slow training cycles.
- **Random Walk-based Methods (DeepWalk, node2vec)**: While also training-free in their initial phase, they often require an expensive skip-gram training phase (Word2Vec) to produce final embeddings, which can still be slow and memory-intensive for massive graphs.

## Consequences
- **Accuracy Gap**: We accept a modest accuracy trade-off (typically within 2-10 percentage points of supervised SOTA) in exchange for orders-of-magnitude faster execution.
- **Architecture Shift**: The optimization objective shifts from "minimizing a loss function" to "maximizing modularity" and "optimizing fusion heuristics."

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
