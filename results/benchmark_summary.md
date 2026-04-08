# Multi-Dataset Benchmark Summary (Node Classification + Link Prediction)

## Node Classification Results

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | classifier_seed | 0.5614 ± 0.0095 | 0.7094 ± 0.0026 | 0.71 | 0.41 | 6.92 ± 1.30 | 2.66 ± 0.50 |
| CiteSeer | classifier_seed | 0.4958 ± 0.0125 | 0.6638 ± 0.0047 | 0.94 | 0.63 | 4.85 ± 0.37 | 2.72 ± 0.54 |
| PubMed | classifier_seed | 0.5850 ± 0.0139 | 0.7246 ± 0.0031 | 5.34 | 3.14 | 13.82 ± 0.77 | 11.92 ± 1.34 |
| BlogCatalog | prepared_split | 0.7498 ± 0.0096 | 0.9138 ± 0.0058 | 2.32 | 1.73 | 27.04 ± 5.64 | 22.98 ± 1.18 |

## Link Prediction Results

| Dataset | Baseline Link AUC | Improved Link AUC | Baseline Link AP | Improved Link AP |
|---|---:|---:|---:|---:|
| Cora | 0.7001 ± 0.0067 | 0.9106 ± 0.0011 | 0.6937 ± 0.0130 | 0.9123 ± 0.0005 |
| CiteSeer | 0.7340 ± 0.0132 | 0.9522 ± 0.0018 | 0.7410 ± 0.0181 | 0.9509 ± 0.0008 |
| PubMed | 0.7045 ± 0.0110 | 0.9254 ± 0.0010 | 0.7432 ± 0.0107 | 0.9266 ± 0.0005 |
| BlogCatalog | 0.6633 ± 0.0149 | 0.6982 ± 0.0025 | 0.6672 ± 0.0101 | 0.6941 ± 0.0009 |
