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

### Cora reproduction

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

### Multi-dataset benchmark

Run all maintained datasets:

```bash
python benchmark_datasets.py --datasets Cora CiteSeer PubMed BlogCatalog
```

Run only one dataset:

```bash
python benchmark_datasets.py --datasets CiteSeer
```

This writes:

- `results/Cora/`
- `results/CiteSeer/`
- `results/PubMed/`
- `results/BlogCatalog/`
- `results/benchmark_summary.json`
- `results/benchmark_summary.md`
- `results/benchmark_summary.png`

## Testing guidelines

Use this as the minimal check after any code change:

1. Prepare the datasets:

```bash
python prepare_datasets.py
```

2. Run the Cora smoke test:

```bash
python run_louvainne_experiments.py --tune-runs 1 --eval-runs 1 --output-json /tmp/louvainne_smoke.json --plot-path /tmp/louvainne_smoke.png
```

3. Run the full Cora benchmark:

```bash
python run_louvainne_experiments.py --tune-runs 2 --eval-runs 5 --output-json results/louvainne_results.json --plot-path results/louvainne_comparison.png
```

4. Run the multi-dataset benchmark:

```bash
python benchmark_datasets.py --datasets Cora CiteSeer PubMed BlogCatalog
```

5. Verify the key outputs exist:

```bash
ls \
  data/manifest.json \
  results/louvainne_results.json \
  results/louvainne_comparison.png \
  results/benchmark_summary.json \
  results/benchmark_summary.md \
  results/benchmark_summary.png \
  results/Cora/comparison_results.json \
  results/CiteSeer/comparison_results.json \
  results/PubMed/comparison_results.json \
  results/BlogCatalog/comparison_results.json \
  summay.md
```

6. Inspect the saved Cora metrics quickly:

```bash
python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('results/louvainne_results.json').read_text())
for key in ['baseline', 'improved']:
    item = payload['final_results'][key]
    print(
        key,
        'micro', round(item['test_micro_f1_mean'], 4),
        'macro', round(item['test_macro_f1_mean'], 4),
        'setup_s', round(item['setup_time_seconds'], 2),
        'per_seed_s', round(item['per_seed_eval_time_seconds_mean'], 2),
    )
PY
```

7. Inspect the multi-dataset summary quickly:

```bash
sed -n '1,120p' results/benchmark_summary.md
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
