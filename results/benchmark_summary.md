# Multi-Dataset Benchmark Summary (Node Classification + Link Prediction)

## Node Classification Results

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| BlogCatalog | prepared_split | 0.7440 ± 0.0095 | 0.9105 ± 0.0057 | 1.98 | 1.69 | 24.38 ± 3.49 | 20.54 ± 1.69 |

## Link Prediction Results

| Dataset | Baseline Link AUC | Improved Link AUC | Baseline Link AP | Improved Link AP |
|---|---:|---:|---:|---:|
| BlogCatalog | 0.6857 ± 0.0046 | 0.6972 ± 0.0062 | 0.8249 ± 0.0024 | 0.8287 ± 0.0021 |
