# Getting Started: LouvainNE-Attributed

This guide provides instructions for reproducing the benchmarks and running the attributed embedding pipeline.

## System Requirements
- **OS**: Linux (Ubuntu recommended)
- **Language**: Python 3.10+
- **Compiler**: GCC (for the C-pipeline)
- **Hardware**: CPU-bound (no GPU required)

## Installation

### 1. Environment Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate
pip install -r hybrid_requirements.txt
```

### 2. Compile C-Binaries
The core embedding engine is written in C for maximum performance.
```bash
cd LouvainNE
make
cd ..
```
This builds the following binaries:
- `renum`: Node ID normalization.
- `recpart`: Recursive Louvain partitioning.
- `hi2vec`: Hierarchy-to-vector embedding generation.

## Running the Pipeline

### Full Reproducible Benchmark
To run the full suite of experiments (Cora, CiteSeer, etc.) and generate results:
```bash
python src/run_louvainne_experiments.py --tune-runs 2 --eval-runs 5 --output-json results/louvainne_results.json
```
This script handles everything:
- Dataset preparation.
- Hyperparameter tuning.
- Binary execution.
- Linear probe evaluation.

## Evaluation
The pipeline outputs a `results/louvainne_results.json` file.
- **Node Classification**: Measured via Micro-F1 on a linear probe.
- **Link Prediction**: Measured via AUC and AP using a leakage-free protocol.
- **Runtime**: Wall-clock time from data loading to final embedding.

## Project Structure
- `LouvainNE/`: C source code for the core embedding engine.
- `src/`: Python orchestration, fusion, and evaluation scripts.
- `results/`: JSON results, PNG plots, and the final PDF report.
- `data/`: Processed graph datasets.
