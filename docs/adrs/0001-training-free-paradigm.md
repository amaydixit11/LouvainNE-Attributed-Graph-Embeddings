# ADR 0001: Training-Free CPU-Bound Paradigm

**Status**: Accepted

## Context
Modern Graph Neural Networks (GNNs) like GCN and GAT provide high accuracy but require GPU acceleration, significant memory (VRAM), and long training times. This creates a "GPU Tax" that limits accessibility for users with commodity hardware and slows down the research iteration cycle.

## Decision
We adopt a **training-free, CPU-bound embedding pipeline** based on hierarchical community detection (Louvain) rather than iterative message passing (GNNs).

## Rationale
- **Hardware Accessibility**: Enables processing of massive graphs on standard CPUs.
- **Speed**: Removes the need for backpropagation and epoch-based training, reducing runtime from hours to seconds/minutes.
- **Determinism**: By using fixed seeds and structural algorithms, results are more easily reproducible than stochastic gradient descent.

## Consequences
- **Accuracy Trade-off**: We accept a small accuracy gap (usually <10 pp) compared to supervised SOTA GNNs in exchange for massive efficiency gains.
- **Architecture Shift**: The focus shifts from optimizing neural network weights to optimizing the graph construction (fusion) and embedding aggregation logic.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
