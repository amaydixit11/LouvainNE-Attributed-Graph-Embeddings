# OGB Benchmark Results: LouvainNE vs SOTA GNNs

**Protocol Disclaimer:** Node classification uses official OGB splits. Link prediction uses a custom protocol
(10% test, 5% val edge split with negative sampling) on ogbn-* graphs, NOT the official ogbl-* protocol.

## Node Classification Results

| Dataset | Method | Micro-F1 | Link AUC | Setup Time (s) | Per-Seed Time (s) | Training-Free? |
|---|---|---:|---:|---:|---:|---|
| ogbn-arxiv | LouvainNE (structure) | 0.5557 ± 0.0092 | 0.8522 ± 0.0028 | 0.00 | 125.03 ± 2.96 | ✓ |
| ogbn-arxiv | LouvainNE (improved) | 0.6116 ± 0.0015 | 0.9097 ± 0.0023 | 12.44 | 125.39 ± 6.58 | ✓ |
| ogbn-arxiv | GCN (GNN) | 0.7169 | N/A | N/A | 0.5 (per epoch) | ✗ |
| ogbn-arxiv | GAT (GNN) | 0.7281 | N/A | N/A | 1.2 (per epoch) | ✗ |
| ogbn-arxiv | GraphSAGE (GNN) | 0.7230 | N/A | N/A | 0.7 (per epoch) | ✗ |
| ogbn-arxiv | APPNP (GNN) | 0.7360 | N/A | N/A | 0.8 (per epoch) | ✗ |
| ogbn-arxiv | SGC (GNN) | 0.7150 | N/A | N/A | 0.3 (per epoch) | ✗ |

## Scalability Comparison

| Dataset | Nodes | Edges | LouvainNE Time (s) | GNN Time/Epoch (s) | Speedup Factor |
|---|---:|---:|---:|---:|---:|
| ogbn-arxiv | 169,343 | 1,166,243 | 125.39 | 0.3 | 0.0x |
