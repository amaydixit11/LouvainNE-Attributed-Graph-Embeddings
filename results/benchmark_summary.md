# Multi-Dataset Benchmark Summary (Node Classification + Link Prediction)

## Node Classification Results

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | classifier_seed | 0.5756 ± 0.0032 | 0.7400 ± 0.0000 | 1.94 | 1.44 | 21.94 ± 2.83 | 15.07 ± 1.73 |
| CiteSeer | classifier_seed | 0.5704 ± 0.0087 | 0.6660 ± 0.0000 | 1.38 | 1.06 | 14.56 ± 1.71 | 20.91 ± 0.49 |
| PubMed | classifier_seed | 0.5800 ± 0.0328 | 0.7380 ± 0.0000 | 10.19 | 5.14 | 53.26 ± 2.41 | 72.72 ± 2.32 |
| BlogCatalog | prepared_split | 0.7449 ± 0.0162 | 0.9136 ± 0.0045 | 4.08 | 3.00 | 46.63 ± 2.81 | 57.08 ± 3.88 |

## Link Prediction Results

| Dataset | Baseline Link AUC | Improved Link AUC | Baseline Link AP | Improved Link AP |
|---|---:|---:|---:|---:|
| Cora | 0.7055 ± 0.0062 | 0.8697 ± 0.0019 | 0.8361 ± 0.0031 | 0.9272 ± 0.0005 |
| CiteSeer | 0.7354 ± 0.0087 | 0.9072 ± 0.0020 | 0.8619 ± 0.0038 | 0.9497 ± 0.0007 |
| PubMed | 0.7069 ± 0.0067 | 0.9114 ± 0.0008 | 0.8527 ± 0.0028 | 0.9521 ± 0.0002 |
| BlogCatalog | 0.6508 ± 0.0111 | 0.7014 ± 0.0007 | 0.7980 ± 0.0039 | 0.8160 ± 0.0004 |
