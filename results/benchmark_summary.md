# Multi-Dataset Benchmark Summary (Node Classification + Link Prediction)

## Node Classification Results

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| BlogCatalog | prepared_split | 0.7165 ± 0.0059 | 0.9061 ± 0.0071 | 2.39 | 1.99 | 31.85 ± 2.45 | 22.26 ± 1.81 |

## Link Prediction Results

| Dataset | Baseline Link AUC | Improved Link AUC | Baseline Link AP | Improved Link AP |
|---|---:|---:|---:|---:|
| BlogCatalog | 0.6627 ± 0.0028 | 0.7006 ± 0.0023 | 0.7909 ± 0.0009 | 0.8177 ± 0.0009 |
