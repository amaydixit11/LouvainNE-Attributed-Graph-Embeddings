# Hybrid Louvain-GNN Implementation Log

This document tracks the implementation progress of the hybrid node classification system.

## 🕒 Implementation Date: 2026-04-24

### ✅ Phase 0: Environment & Data Preparation
- Identified available dependencies in `venv`.
- Created `hybrid_src/data.py` for dataset loading using PyTorch Geometric and NetworkX.
- Supported datasets: `Cora`, `CiteSeer`, `PubMed`.

### ✅ Phase 1: Baseline Establishment
- Created `hybrid_src/baseline.py`.
- Implemented Louvain community detection with fallback to `networkx.algorithms.community`.
- Implemented Majority Vote labeling per community.
- Implemented Confidence Scoring:
  - Formula: `confidence(node) = (agreeing_neighbors) / (total_neighbors)`

### ✅ Phase 2 & 3: Hard Node Selection & Subgraph Extraction
- Created `hybrid_src/subgraph.py`.
- Implemented `identify_hard_nodes(tau)` to filter nodes below confidence threshold.
- Implemented `get_khop_subgraph_nodes(k)` for local neighborhood extraction.
- Optimized extraction using `data.subgraph()` for PyG compatibility.

### ✅ Phase 4: GNN Implementation
- Created `hybrid_src/gnn.py`.
- Implemented GNN architectures:
  - **GCN**: 2-layer Graph Convolutional Network.
  - **GAT**: Graph Attention Network with multi-head attention.
  - **GraphSAGE**: SAGEConv-based architecture.
- Implemented training loop scoped to the extracted subgraph nodes.

### ✅ Phase 5 & 6: Fusion & Evaluation
- Created `hybrid_src/fusion.py` for prediction merging:
  - **Hard Switch**: GNN labels for hard nodes, Louvain for others.
  - **Soft Fusion**: Weighted average based on node-wise confidence.
- Created `hybrid_src/eval.py` for Accuracy and F1 metrics.
- Created `run_hybrid.py` as the main execution engine.

### ✅ Infrastructure & Automation
- Created `configs/default.yaml` for hyperparameter management.
- Created `scripts/run_experiments.py` for automated $\tau$ sweeps and comparative analysis.
- Generated `hybrid_requirements.txt` for environmental consistency.

---

## 📊 Experimental Snapshot (Cora)

| Model | Tau ($\tau$) | k-hop | Accuracy | Notes |
|-------|-------|-------|----------|-------|
| Louvain (Baseline) | N/A | N/A | 76.60% | Majority vote per community |
| **Hybrid (Hard Switch)** | 0.75 | 1 | **77.30%** | **+0.7% Improvement** |
| Hybrid (Soft Fusion) | 0.75 | 1 | 76.60% | Weighted merge |

### Observations:
- **Hard nodes identified**: 267 (approx 10% of Cora).
- **Subgraph scaling**: Successfully reduced graph processing from 2,708 nodes to 841 nodes for GNN phase.
- **Speed**: Louvain phase takes ~0.05s, GNN training on subgraph takes ~1.2s.

---

## 🛠️ Current File Manifest

- `hybrid_src/`: Core logic modules.
- `configs/`: YAML experiment configs.
- `scripts/`: Python utility scripts.
- `run_hybrid.py`: CLI Entry point.
- `results/`: Output JSONs and plots.
