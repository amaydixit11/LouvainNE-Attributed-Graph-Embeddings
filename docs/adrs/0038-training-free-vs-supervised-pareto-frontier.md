# ADR 0038: Training-Free vs Supervised GNN Pareto Frontier

**Status**: Accepted

## Context
The project is compared against supervised GNNs (GCN, GAT) which achieve higher accuracy but require GPUs and training.

## Decision
Position the system as a **unique point on the Accuracy-Speed Pareto frontier**: offering competitive accuracy for a fraction of the computational cost.

## Rationale
- **Practicality**: In many production environments, a $10\times$ speedup and $0$ GPU dependency are more valuable than a $5\%$ increase in accuracy.
- **Baseline**: Sets a target where the system must outperform "fast" methods (DeepWalk) in accuracy and "accurate" methods (GCN) in speed.

## Consequences
- **Benchmarking Focus**: Emphasizes wall-clock time and memory usage as first-class metrics alongside F1/AUC.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
