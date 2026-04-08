# LouvainNE-Attributed-Graph-Embeddings

This repository contains a cleaned, reproducible version of the LouvainNE-with-attributes experiments for node classification on multiple attributed graph benchmarks.

It is based on LouvainNE:
- upstream reference: https://github.com/maxdan94/LouvainNE
- paper: "LouvainNE: Hierarchical Louvain Method for High Quality and Scalable Network Embedding" (WSDM 2020)

## What is maintained in this repo

- `prepare_datasets.py`
  The supported dataset bootstrap script. It creates `data/` and fetches only `Cora`, `CiteSeer`, `PubMed`, and `BlogCatalog`.

- `run_louvainne_experiments.py`
  Main Cora experiment runner. This reproduces the tuned baseline vs improved pipeline on the original Cora setup, using data prepared by `prepare_datasets.py`.

- `benchmark_datasets.py`
  Multi-dataset benchmark runner for `Cora`, `CiteSeer`, `PubMed`, and `BlogCatalog`. It writes one subfolder per dataset under `results/`.

- `summay.md`
  Analysis of what was implemented in the original repo, what was changed, and the measured results.

- `results/louvainne_results.json`
  Saved output from the reproducible Cora experiment run.

- `results/louvainne_comparison.png`
  Accuracy plus runtime-split comparison plot for the best reproducible Cora baseline vs the improved pipeline.

- `results/benchmark_summary.json`
- `results/benchmark_summary.md`
- `results/benchmark_summary.png`
  Aggregated multi-dataset results across `Cora`, `CiteSeer`, `PubMed`, and `BlogCatalog`, including one-time setup time and repeated per-seed evaluation time.

- `results/<Dataset>/comparison_results.json`
- `results/<Dataset>/comparison_plot.png`
  Per-dataset result artifacts written by `benchmark_datasets.py`.

- `data/`
  Generated dataset cache created by `prepare_datasets.py`. This folder is not the source of truth for the repo; it is regenerated state.

- `LouvainNE/`
  Patched LouvainNE source code used by the runner.

## What was removed

Legacy notebooks, stale generated text files, and old compiled binaries were removed from the main workflow because they were either redundant, not reproducible, or superseded by the runner above.

## Setup

Create the conda environment:

```bash
conda env create --name <envname> --file=environment.yml
conda activate <envname>
```

Both runners rebuild the needed LouvainNE binaries automatically under `build/louvainne/`.

## Run experiments

### Prepare datasets

Run this first:

```bash
python prepare_datasets.py
```

This creates `data/` and prepares exactly these datasets:

- `Cora`
- `CiteSeer`
- `PubMed`
- `BlogCatalog`

### Master experiment runner (recommended)

Run all benchmarks end-to-end:

```bash
python run_all_experiments.py
```

This executes:
1. Dataset preparation
2. Standard benchmarks (Node Classification + Link Prediction)
3. OGB large-scale benchmarks (if `ogb` package installed)
4. Comprehensive SOTA comparison report generation

Options:

```bash
# Skip specific stages
python run_all_experiments.py --skip-ogb  # Skip OGB if ogb not installed
python run_all_experiments.py --skip-report  # Skip report generation

# Custom datasets
python run_all_experiments.py --datasets Cora CiteSeer --ogb-datasets ogbn-arxiv

# Check dependencies only
python run_all_experiments.py --check-only
```

### Individual benchmark scripts

#### Cora reproduction

Quick smoke test:

```bash
python run_louvainne_experiments.py --tune-runs 1 --eval-runs 1 --output-json /tmp/louvainne_smoke.json
```

Full reproducible run:

```bash
python run_louvainne_experiments.py \
  --tune-runs 2 \
  --eval-runs 5 \
  --output-json results/louvainne_results.json \
  --plot-path results/louvainne_comparison.png
```

#### Multi-dataset benchmark (with link prediction)

Run all maintained datasets with **both node classification and link prediction**:

```bash
python benchmark_datasets_lp.py --datasets Cora CiteSeer PubMed BlogCatalog
```

Run only one dataset:

```bash
python benchmark_datasets_lp.py --datasets CiteSeer
```

This writes:

- `results/Cora/`
- `results/CiteSeer/`
- `results/PubMed/`
- `results/BlogCatalog/`
- `results/benchmark_summary.json`
- `results/benchmark_summary.md`
- `results/benchmark_summary.png`

#### OGB large-scale benchmarks

Run on Open Graph Benchmark datasets (requires `ogb` package):

```bash
pip install ogb
python benchmark_ogb.py --datasets ogbn-arxiv
python benchmark_ogb.py --datasets ogbn-arxiv ogbn-products --embedding-dim 256
```

This writes:

- `results/ogbn_arxiv/ogb_results.json`
- `results/ogb_benchmark_summary.json`
- `results/ogb_sota_comparison.md`
- `results/ogb_comparison.png`

#### SOTA comparison report

Generate comprehensive markdown report comparing LouvainNE with published GNN results:

```bash
python generate_sota_report.py
```

This writes:

- `results/comprehensive_benchmark_report.md`

## Testing guidelines

### Minimal smoke test

1. Prepare the datasets:

```bash
python prepare_datasets.py
```

2. Run the Cora smoke test:

```bash
python run_louvainne_experiments.py --tune-runs 1 --eval-runs 1 --output-json /tmp/louvainne_smoke.json --plot-path /tmp/louvainne_smoke.png
```

### Full benchmark suite

3. Run the multi-dataset benchmark with link prediction:

```bash
python benchmark_datasets_lp.py --datasets Cora CiteSeer PubMed BlogCatalog
```

4. Run OGB benchmarks (optional, requires `ogb`):

```bash
pip install ogb
python benchmark_ogb.py --datasets ogbn-arxiv
```

5. Generate comprehensive report:

```bash
python generate_sota_report.py
```

6. Verify the key outputs exist:

```bash
ls \
  data/manifest.json \
  results/benchmark_summary.json \
  results/benchmark_summary.md \
  results/benchmark_summary.png \
  results/comprehensive_benchmark_report.md \
  results/Cora/comparison_results.json \
  results/CiteSeer/comparison_results.json \
  results/PubMed/comparison_results.json \
  results/BlogCatalog/comparison_results.json
```

7. Inspect the multi-dataset summary:

```bash
cat results/benchmark_summary.md
```

8. View the comprehensive report:

```bash
cat results/comprehensive_benchmark_report.md
```

## Current best result

From the latest saved Cora run:

- best reproducible repo-style baseline: micro-F1 `0.7356 ± 0.0038`, macro-F1 `0.7291 ± 0.0039`
- improved pipeline: micro-F1 `0.7722 ± 0.0026`, macro-F1 `0.7623 ± 0.0021`
- baseline setup time: `0.15` seconds
- improved setup time: `1.13` seconds
- baseline per-seed evaluation time: `2.22 ± 0.10` seconds
- improved per-seed evaluation time: `0.88 ± 0.12` seconds

## Current multi-dataset benchmark

From `results/benchmark_summary.md`:

- `Cora`: baseline `0.5916 ± 0.0031`, improved `0.7226 ± 0.0053`, setup `0.53s` vs `0.32s`, per-seed `3.30s` vs `1.06s`
- `CiteSeer`: baseline `0.4958 ± 0.0125`, improved `0.6638 ± 0.0047`, setup `0.53s` vs `0.36s`, per-seed `3.43s` vs `1.23s`
- `PubMed`: baseline `0.5830 ± 0.0167`, improved `0.7246 ± 0.0031`, setup `2.52s` vs `1.52s`, per-seed `8.53s` vs `6.87s`
- `BlogCatalog`: baseline `0.7517 ± 0.0080`, improved `0.9143 ± 0.0066`, setup `1.30s` vs `0.90s`, per-seed `6.72s` vs `4.21s`

See `summay.md` for the repo analysis, `results/louvainne_comparison.png` for the Cora comparison, and `results/benchmark_summary.png` for the cross-dataset comparison.
