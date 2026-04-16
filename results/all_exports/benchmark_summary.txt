# Multi-Dataset Benchmark Summary (Node Classification + Link Prediction)

## Node Classification Results

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | classifier_seed | 0.5504 ± 0.0055 | 0.7190 ± 0.0000 | 0.61 | 0.38 | 9.33 ± 1.90 | 8.47 ± 1.15 |
| CiteSeer | classifier_seed | 0.5704 ± 0.0078 | 0.6660 ± 0.0000 | 0.93 | 0.72 | 7.92 ± 2.18 | 9.53 ± 0.99 |
| PubMed | classifier_seed | 0.5800 ± 0.0294 | 0.7380 ± 0.0000 | 4.69 | 2.61 | 24.55 ± 1.93 | 40.38 ± 4.01 |
| BlogCatalog | prepared_split | 0.7449 ± 0.0145 | 0.9136 ± 0.0040 | 2.11 | 1.44 | 21.15 ± 1.40 | 23.38 ± 1.00 |

## Link Prediction Results

| Dataset | Baseline Link AUC | Improved Link AUC | Baseline Link AP | Improved Link AP |
|---|---:|---:|---:|---:|
| Cora | 0.7037 ± 0.0057 | 0.8726 ± 0.0018 | 0.8381 ± 0.0026 | 0.9273 ± 0.0007 |
| CiteSeer | 0.7354 ± 0.0078 | 0.9072 ± 0.0018 | 0.8619 ± 0.0034 | 0.9497 ± 0.0006 |
| PubMed | 0.7069 ± 0.0060 | 0.9114 ± 0.0008 | 0.8527 ± 0.0025 | 0.9521 ± 0.0002 |
| BlogCatalog | 0.6508 ± 0.0099 | 0.7014 ± 0.0006 | 0.7980 ± 0.0035 | 0.8160 ± 0.0004 |
