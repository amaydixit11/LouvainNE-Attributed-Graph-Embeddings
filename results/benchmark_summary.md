# Multi-Dataset Benchmark Summary

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | classifier_seed | 0.5916 ± 0.0031 | 0.7226 ± 0.0053 | 0.53 | 0.32 | 3.30 ± 1.53 | 1.06 ± 0.02 |
| CiteSeer | classifier_seed | 0.4958 ± 0.0125 | 0.6638 ± 0.0047 | 0.53 | 0.36 | 3.43 ± 1.64 | 1.23 ± 0.01 |
| PubMed | classifier_seed | 0.5830 ± 0.0167 | 0.7246 ± 0.0031 | 2.52 | 1.52 | 8.53 ± 1.69 | 6.87 ± 1.98 |
| BlogCatalog | prepared_split | 0.7517 ± 0.0080 | 0.9143 ± 0.0066 | 1.30 | 0.90 | 6.72 ± 2.64 | 4.21 ± 1.93 |
