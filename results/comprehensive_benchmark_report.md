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
| Cora | OGC | 86.90% | 74.00% | NESS | 0.9846 / 0.9871 | 0.8697 / 0.9272 | 15.07 |
| CiteSeer | APPNP | 74.20% | 66.60% | NESS | 0.9943 / 0.9950 | 0.9072 / 0.9497 | 20.91 |
| PubMed | GraphSAGE+DropEdge | 91.70% | 73.80% | NESS | 0.9810 / 0.9810 | 0.9114 / 0.9521 | 72.72 |
| BlogCatalog | node2vec | 33.60% | 91.36% | N/A | N/A | 0.7014 / 0.8160 | 57.08 |

## Node Classification: Large Comparison Table

| Dataset | Model | Type | Accuracy | Our Improved | Gap vs Ours | Per-Seed Time (s) | External Runtime | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| Cora | OGC | External | 86.90% | 74.00% | 12.90% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCN-TV | External | 86.30% | 74.00% | 12.30% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCNII | External | 85.50% | 74.00% | 11.50% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GRAND | External | 85.40% | 74.00% | 11.40% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | CPF-ind-APPNP | External | 85.30% | 74.00% | 11.30% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GCN | External | 85.10% | 74.00% | 11.10% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | AIR-GCN | External | 84.70% | 74.00% | 10.70% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | H-GCN | External | 84.50% | 74.00% | 10.50% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | DAGNN | External | 84.40% | 74.00% | 10.40% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GAT | External | 83.00% | 74.00% | 9.00% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | GraphSAGE | External | 74.50% | 74.00% | 0.50% | 15.07 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html |
| Cora | LouvainNE (structure) | Ours | 57.56% | 74.00% | -16.44% | 21.94 | repo measured | local results |
| Cora | LouvainNE (improved) | Ours | 74.00% | 74.00% | 0.00% | 15.07 | repo measured | local results |
| CiteSeer | APPNP | External | 74.20% | 66.60% | 7.60% | 20.91 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GAT | External | 72.50% | 66.60% | 5.90% | 20.91 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GraphSAGE | External | 70.80% | 66.60% | 4.20% | 20.91 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | GCN | External | 70.30% | 66.60% | 3.70% | 20.91 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html |
| CiteSeer | LouvainNE (structure) | Ours | 57.04% | 66.60% | -9.56% | 14.56 | repo measured | local results |
| CiteSeer | LouvainNE (improved) | Ours | 66.60% | 66.60% | 0.00% | 20.91 | repo measured | local results |
| PubMed | GraphSAGE+DropEdge | External | 91.70% | 73.80% | 17.90% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | ASGCN | External | 90.60% | 73.80% | 16.80% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | FDGATII | External | 90.35% | 73.80% | 16.55% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | GCNII | External | 90.30% | 73.80% | 16.50% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | FastGCN | External | 88.00% | 73.80% | 14.20% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | GraphSAGE | External | 87.10% | 73.80% | 13.30% | 72.72 | N/A | https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html |
| PubMed | LouvainNE (structure) | Ours | 58.00% | 73.80% | -15.80% | 53.26 | repo measured | local results |
| PubMed | LouvainNE (improved) | Ours | 73.80% | 73.80% | 0.00% | 72.72 | repo measured | local results |
| BlogCatalog | node2vec | External | 33.60% | 91.36% | -57.76% | 57.08 | N/A | N/A |
| BlogCatalog | DeepWalk | External | 32.90% | 91.36% | -58.46% | 57.08 | N/A | N/A |
| BlogCatalog | LINE | External | 32.10% | 91.36% | -59.26% | 57.08 | N/A | N/A |
| BlogCatalog | LouvainNE (structure) | Ours | 74.49% | 91.36% | -16.86% | 46.63 | repo measured | local results |
| BlogCatalog | LouvainNE (improved) | Ours | 91.36% | 91.36% | 0.00% | 57.08 | repo measured | local results |

## Link Prediction: Large Comparison Table

| Dataset | Model | Type | AUC | AP | Our Improved AUC | Gap vs Ours | Per-Seed Time (s) | Source |
|---|---|---|---:|---:|---:|---:|---:|---|
| Cora | NESS | External | 0.9846 | 0.9871 | 0.8697 | 0.1149 | 15.07 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | WalkPooling | External | 0.9590 | 0.9600 | 0.8697 | 0.0893 | 15.07 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | NBFNet | External | 0.9560 | 0.9620 | 0.8697 | 0.0863 | 15.07 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | VGNAE | External | 0.9560 | 0.9570 | 0.8697 | 0.0863 | 15.07 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | VGAE | External | 0.9140 | 0.9230 | 0.8697 | 0.0443 | 15.07 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html |
| Cora | LouvainNE (structure) | Ours | 0.7055 | 0.8361 | 0.8697 | -0.1642 | 21.94 | local results |
| Cora | LouvainNE (improved) | Ours | 0.8697 | 0.9272 | 0.8697 | 0.0000 | 15.07 | local results |
| CiteSeer | NESS | External | 0.9943 | 0.9950 | 0.9072 | 0.0871 | 20.91 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | VGNAE | External | 0.9700 | 0.9710 | 0.9072 | 0.0628 | 20.91 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | Graph InfoClust | External | 0.9700 | 0.9680 | 0.9072 | 0.0628 | 20.91 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | GNAE | External | 0.9650 | 0.9700 | 0.9072 | 0.0578 | 20.91 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | VGAE | External | 0.8630 | 0.8810 | 0.9072 | -0.0442 | 20.91 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html |
| CiteSeer | LouvainNE (structure) | Ours | 0.7354 | 0.8619 | 0.9072 | -0.1718 | 14.56 | local results |
| CiteSeer | LouvainNE (improved) | Ours | 0.9072 | 0.9497 | 0.9072 | 0.0000 | 20.91 | local results |
| PubMed | NESS | External | 0.9810 | 0.9810 | 0.9114 | 0.0696 | 72.72 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | WalkPooling | External | 0.9640 | 0.9650 | 0.9114 | 0.0526 | 72.72 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | SEAL | External | 0.9680 | 0.9690 | 0.9114 | 0.0566 | 72.72 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | NBFNet | External | 0.9580 | 0.9610 | 0.9114 | 0.0466 | 72.72 | https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html |
| PubMed | LouvainNE (structure) | Ours | 0.7069 | 0.8527 | 0.9114 | -0.2044 | 53.26 | local results |
| PubMed | LouvainNE (improved) | Ours | 0.9114 | 0.9521 | 0.9114 | 0.0000 | 72.72 | local results |
| BlogCatalog | LouvainNE (structure) | Ours | 0.6508 | 0.7980 | 0.7014 | -0.0506 | 46.63 | local results |
| BlogCatalog | LouvainNE (improved) | Ours | 0.7014 | 0.8160 | 0.7014 | 0.0000 | 57.08 | local results |

## Runtime Notes

| Dataset | Our Structure Time (s) | Our Improved Time (s) | External Runtime Availability |
|---|---:|---:|---|
| Cora | 21.94 | 15.07 | Not consistently reported on sourced leaderboard pages |
| CiteSeer | 14.56 | 20.91 | Not consistently reported on sourced leaderboard pages |
| PubMed | 53.26 | 72.72 | Not consistently reported on sourced leaderboard pages |
| BlogCatalog | 46.63 | 57.08 | Not consistently reported on sourced leaderboard pages |

