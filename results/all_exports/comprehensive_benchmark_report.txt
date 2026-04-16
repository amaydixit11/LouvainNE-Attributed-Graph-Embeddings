# Comprehensive Benchmark Report: LouvainNE vs SOTA

## Scope

This report builds large comparison tables for node classification, link prediction, and runtime.
External accuracy metrics are sourced from benchmark pages and paper-reported baselines.
Our metrics and timings come from saved repo artifacts under `results/`.

## Protocol Notes

- External benchmark pages do not provide fully standardized runtime numbers, so external runtime cells are left as `N/A` unless directly available.
- Our link prediction numbers come from the repo's train/val/test edge split protocol.
- OpenCodePapers leaderboard protocols may differ from our preprocessing or split details.
- `BlogCatalog` has strong local results in this repo, but external sourced node/link leaderboards are sparse.

## Best-Per-Dataset Summary

| Dataset | Best External Node Model | Node Acc | Our Node Acc | Best External Link Model | Link AUC / AP | Our Link AUC / AP | Our Time (s) |
|---|---|---:|---:|---|---:|---:|---:|
| Cora | OGC | 86.90% | 71.02% | NESS | 0.9846 / 0.9871 | 0.8694 / 0.9253 | 7.18 |
| CiteSeer | APPNP | 74.20% | 66.64% | NESS | 0.9943 / 0.9950 | 0.9072 / 0.9497 | 9.34 |
| PubMed | GraphSAGE+DropEdge | 91.70% | 72.82% | NESS | 0.9810 / 0.9810 | 0.9114 / 0.9521 | 26.74 |
| BlogCatalog | node2vec | 33.60% | 90.61% | N/A | N/A | 0.7006 / 0.8177 | 22.26 |

## Node Classification: Large Comparison Table

| Dataset | Model | Type | Accuracy | Our Improved | Gap vs Ours | Per-Seed Time (s) | External Runtime | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| Cora | OGC | External | 86.90% | 71.02% | 15.88% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCN-TV | External | 86.30% | 71.02% | 15.28% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCNII | External | 85.50% | 71.02% | 14.48% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GRAND | External | 85.40% | 71.02% | 14.38% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | CPF-ind-APPNP | External | 85.30% | 71.02% | 14.28% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCN | External | 85.10% | 71.02% | 14.08% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | AIR-GCN | External | 84.70% | 71.02% | 13.68% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | H-GCN | External | 84.50% | 71.02% | 13.48% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | DAGNN | External | 84.40% | 71.02% | 13.38% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GAT | External | 83.00% | 71.02% | 11.98% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GraphSAGE | External | 74.50% | 71.02% | 3.48% | 7.18 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | LouvainNE (structure) | Ours | 58.58% | 71.02% | -12.44% | 11.85 | repo measured | local results |
| Cora | LouvainNE (improved) | Ours | 71.02% | 71.02% | 0.00% | 7.18 | repo measured | local results |
| CiteSeer | APPNP | External | 74.20% | 66.64% | 7.56% | 9.34 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GAT | External | 72.50% | 66.64% | 5.86% | 9.34 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GraphSAGE | External | 70.80% | 66.64% | 4.16% | 9.34 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GCN | External | 70.30% | 66.64% | 3.66% | 9.34 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | LouvainNE (structure) | Ours | 57.04% | 66.64% | -9.60% | 13.12 | repo measured | local results |
| CiteSeer | LouvainNE (improved) | Ours | 66.64% | 66.64% | 0.00% | 9.34 | repo measured | local results |
| PubMed | GraphSAGE+DropEdge | External | 91.70% | 72.82% | 18.88% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | ASGCN | External | 90.60% | 72.82% | 17.78% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | FDGATII | External | 90.35% | 72.82% | 17.53% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | GCNII | External | 90.30% | 72.82% | 17.48% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | FastGCN | External | 88.00% | 72.82% | 15.18% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | GraphSAGE | External | 87.10% | 72.82% | 14.28% | 26.74 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | LouvainNE (structure) | Ours | 58.00% | 72.82% | -14.82% | 45.19 | repo measured | local results |
| PubMed | LouvainNE (improved) | Ours | 72.82% | 72.82% | 0.00% | 26.74 | repo measured | local results |
| BlogCatalog | node2vec | External | 33.60% | 90.61% | -57.01% | 22.26 | N/A | N/A |
| BlogCatalog | DeepWalk | External | 32.90% | 90.61% | -57.71% | 22.26 | N/A | N/A |
| BlogCatalog | LINE | External | 32.10% | 90.61% | -58.51% | 22.26 | N/A | N/A |
| BlogCatalog | LouvainNE (structure) | Ours | 71.65% | 90.61% | -18.96% | 31.85 | repo measured | local results |
| BlogCatalog | LouvainNE (improved) | Ours | 90.61% | 90.61% | 0.00% | 22.26 | repo measured | local results |

## Link Prediction: Large Comparison Table

| Dataset | Model | Type | AUC | AP | Our Improved AUC | Gap vs Ours | Per-Seed Time (s) | Source |
|---|---|---|---:|---:|---:|---:|---:|---|
| Cora | NESS | External | 0.9846 | 0.9871 | 0.8694 | 0.1152 | 7.18 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | WalkPooling | External | 0.9590 | 0.9600 | 0.8694 | 0.0896 | 7.18 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | NBFNet | External | 0.9560 | 0.9620 | 0.8694 | 0.0866 | 7.18 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | VGNAE | External | 0.9560 | 0.9570 | 0.8694 | 0.0866 | 7.18 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | VGAE | External | 0.9140 | 0.9230 | 0.8694 | 0.0446 | 7.18 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | LouvainNE (structure) | Ours | 0.8084 | 0.8887 | 0.8694 | -0.0610 | 11.85 | local results |
| Cora | LouvainNE (improved) | Ours | 0.8694 | 0.9253 | 0.8694 | 0.0000 | 7.18 | local results |
| CiteSeer | NESS | External | 0.9943 | 0.9950 | 0.9072 | 0.0871 | 9.34 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | VGNAE | External | 0.9700 | 0.9710 | 0.9072 | 0.0628 | 9.34 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | Graph InfoClust | External | 0.9700 | 0.9680 | 0.9072 | 0.0628 | 9.34 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | GNAE | External | 0.9650 | 0.9700 | 0.9072 | 0.0578 | 9.34 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | VGAE | External | 0.8630 | 0.8810 | 0.9072 | -0.0442 | 9.34 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | LouvainNE (structure) | Ours | 0.8811 | 0.9336 | 0.9072 | -0.0261 | 13.12 | local results |
| CiteSeer | LouvainNE (improved) | Ours | 0.9072 | 0.9497 | 0.9072 | 0.0000 | 9.34 | local results |
| PubMed | NESS | External | 0.9810 | 0.9810 | 0.9114 | 0.0696 | 26.74 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | WalkPooling | External | 0.9640 | 0.9650 | 0.9114 | 0.0526 | 26.74 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | SEAL | External | 0.9680 | 0.9690 | 0.9114 | 0.0566 | 26.74 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | NBFNet | External | 0.9580 | 0.9610 | 0.9114 | 0.0466 | 26.74 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | LouvainNE (structure) | Ours | 0.8848 | 0.9354 | 0.9114 | -0.0266 | 45.19 | local results |
| PubMed | LouvainNE (improved) | Ours | 0.9114 | 0.9521 | 0.9114 | 0.0000 | 26.74 | local results |
| BlogCatalog | LouvainNE (structure) | Ours | 0.6627 | 0.7909 | 0.7006 | -0.0379 | 31.85 | local results |
| BlogCatalog | LouvainNE (improved) | Ours | 0.7006 | 0.8177 | 0.7006 | 0.0000 | 22.26 | local results |

## Runtime Notes

| Dataset | Our Structure Time (s) | Our Improved Time (s) | External Runtime Availability |
|---|---:|---:|---|
| Cora | 11.85 | 7.18 | Not consistently reported on sourced leaderboard pages |
| CiteSeer | 13.12 | 9.34 | Not consistently reported on sourced leaderboard pages |
| PubMed | 45.19 | 26.74 | Not consistently reported on sourced leaderboard pages |
| BlogCatalog | 31.85 | 22.26 | Not consistently reported on sourced leaderboard pages |

