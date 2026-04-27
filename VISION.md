# Vision: TurboGraph & LouvainNE-Attributed

## The Mission
TurboGraph is a research initiative aimed at breaking the "GPU Tax" of modern graph representation learning. While Graph Neural Networks (GNNs) have set the state-of-the-art for node classification and link prediction, they impose a heavy requirement for high-memory GPUs and long training cycles. 

The goal is to create a **training-free, CPU-bound embedding pipeline** that achieves competitive accuracy with supervised GNNs while maintaining orders-of-magnitude faster execution and near-linear scalability.

## Core Thesis
The inherent community structure of graphs, when combined with high-dimensional node attributes through a principled "Early Fusion" process, can replace the need for iterative message-passing. By augmenting the structural graph with attribute-driven edges *before* performing hierarchical clustering (Louvain), we can bake feature information directly into the structural topology.

## The Evolution (The 7 Phases)
The system evolved through a series of rigorous empirical phases:
1. **Baseline Recovery**: Fixing the original LouvainNE C-pipeline to ensure reproducibility.
2. **Global Thresholding**: Initial attempt at attribute fusion using a single global similarity cutoff (failed to generalize).
3. **Late Fusion**: Concatenating separate structural and attribute embeddings (found inferior to early fusion).
4. **Dense Self-Attention**: Applying global attention to embeddings (hit $O(n^2)$ memory wall).
5. **Mutual Top-K**: Implementing a scale-invariant, mutual-similarity filter for attribute edge generation (the breakthrough for noise reduction).
6. **The Full Pipeline**: Integrating Mutual Top-K with Sparse Neighbourhood Attention and an SVD Attribute-Residual branch.
7. **OGB Adaptation**: Scaling the pipeline to massive graphs (e.g., ogbn-arxiv) by restricting attribute search to observed structural edges.

## Success Metrics
- **Accuracy**: Closing the gap to GNN SOTA (e.g., within 2-10 pp on citation benchmarks).
- **Efficiency**: $\sim 10-40\times$ speedup over GCN/GAT.
- **Accessibility**: 100% CPU-bound; zero CUDA dependency.
- **Scalability**: Empirical $O(n^{1.5})$ or better, enabling processing of millions of nodes on commodity hardware.
