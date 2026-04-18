# LouvainNE-Attributed-Graph-Embeddings: Repo Analysis and Optimization Summary

## What was implemented in the original repo before cleanup

- `node_classification_early_fusion.py`
  Uses Cora, raw feature dot-product similarity, global thresholding, and unweighted early fusion. This is your "method 1" path: add attribute-derived edges, give every final edge weight `1`, then run LouvainNE.

- `node_classification_late_fusion.py`
  Builds a separate attribute graph and a structure graph, gets independent LouvainNE embeddings, and fuses them later. The checked-in `.py` file currently executes the `sum` variant. The notebooks also try `concat`, but late fusion is still weaker than early fusion.

- `node_classification_earlyfusion_SA_weight.ipynb`
  This is the notebook with the self-attention refinement. The important code detail is that the attention weights are computed from the raw feature matrix `data.x`, not from learned functional embeddings. So the code and the project note are slightly different here.

- Weighted early-fusion variants
  The project note describes methods 2/3/4. They were explored in notebooks, but there was no clean reproducible script to compare them end to end.

These legacy files were analyzed and then removed from the cleaned repo because they were redundant or non-reproducible.

## Issues found during analysis

1. The checked-in `LouvainNE/recpart` binary was stale relative to the source, so the repo was not reliably reproducible from the current tree.
2. The Louvain partition code had a root-level null `g->map` dereference when the full graph collapsed to one community.
3. `LouvainNE/recpart.c` expects weighted `u v w` edge lists in the current source, while the old Python scripts still write unweighted `u v` files.
4. `LouvainNE/recpart.c` and `LouvainNE/hi2vec.c` seeded randomness from `time(NULL)`, so repeated experiments were not properly reproducible.
5. The original checkout had an empty `data/Planetoid/Cora/raw` directory, so the old scripts were not self-contained.

## Changes introduced

1. Fixed the LouvainNE crash
   Patched `LouvainNE/recpart.c` so leaf output falls back to node ids when `g->map == NULL`.

2. Added deterministic seeds to the C binaries
   Patched `LouvainNE/recpart.c` and `LouvainNE/hi2vec.c` so experiments can pass explicit seeds.

3. Added a reproducible experiment runner
   `run_louvainne_experiments.py` now:
   - reads Cora from `data/Planetoid/Cora`
   - expects datasets to be prepared by `prepare_datasets.py`
   - rebuilds fresh LouvainNE binaries under `build/louvainne`
   - reproduces the repo baselines
   - searches stronger attributed-graph variants
   - writes full results to `results/louvainne_results.json`

4. Added a supported dataset preparation script
   `prepare_datasets.py` now creates `data/` and prepares exactly:
   - `Cora`
   - `CiteSeer`
   - `PubMed`
   - `BlogCatalog`

5. Removed obsolete files
   Deleted legacy notebooks, stale generated text files, prebuilt LouvainNE binaries, object files, and the unused duplicate `LouvainNE/recpart_weighted.c`.

6. Added a better attributed graph construction strategy
   Replaced the repo's global threshold over max-normalized raw dot products with:
   - cosine similarity
   - mutual top-k neighbors
   - minimum similarity filtering
   - weighted overlap/new-edge rules

7. Added sparse neighborhood attention
   Replaced dense all-node attention with sparse attention over the hybrid graph neighborhood. The best setup used residual mixing with `gamma = 0.5`.

8. Added an attribute residual branch
   Concatenated a `128`-dimensional SVD projection of the raw feature matrix with the refined LouvainNE embedding before classification.

9. Added a scalable prepared-dataset benchmark
   `benchmark_datasets.py` now benchmarks only the prepared dataset set above and uses:
   - low-rank feature projection
   - projected top-k graph construction
   - sparse neighborhood attention
   This keeps `PubMed` and `BlogCatalog` tractable without dense all-pairs similarity matrices.

## Reproducible command

```bash
python run_louvainne_experiments.py --tune-runs 2 --eval-runs 5 --output-json results/louvainne_results.json
```

## Main results

`Structure only`, `Best reproducible repo-style baseline`, `Adaptive hybrid graph only`, and `Final improved pipeline`
are test-set mean ± std over 5 evaluation seeds.

The three intermediate repo baselines (`Repo early fusion`, `Repo early fusion + dense attention`, and
`Late fusion concat`) come from the 2-seed coarse search pass and are listed only to show the relative ranking of
the original ideas.

| Method | Micro-F1 | Macro-F1 | Notes |
|---|---:|---:|---|
| Structure only | `0.6760 ± 0.0052` | `0.6717 ± 0.0058` | Pure LouvainNE on the original graph |
| Repo early fusion (best unweighted threshold) | `0.7230 ± 0.0000` | `0.7125 ± 0.0000` | Best coarse-search unweighted early-fusion baseline at `alpha = 0.25` |
| Repo early fusion + dense attention | `0.7250 ± 0.0000` | `0.7149 ± 0.0000` | Best coarse-search repo-style self-attention baseline at `alpha = 0.25` |
| Late fusion concat | `0.6250 ± 0.0000` | `0.6214 ± 0.0000` | Much weaker than early fusion |
| Best reproducible repo-style baseline | `0.7356 ± 0.0038` | `0.7291 ± 0.0039` | Weighted method 2 with `alpha = 0.2` |
| Adaptive hybrid graph only | `0.7600 ± 0.0070` | `0.7507 ± 0.0045` | Cosine + mutual top-k + weighted hybrid graph |
| Final improved pipeline | `0.7722 ± 0.0026` | `0.7623 ± 0.0021` | Adaptive graph + sparse attention + 128-d attribute residual |

## Runtime comparison

- Best reproducible repo-style baseline:
  - setup time: `0.15` seconds
  - per-seed evaluation time: `2.22 ± 0.10` seconds
- Final improved pipeline:
  - setup time: `1.13` seconds
  - per-seed evaluation time: `0.88 ± 0.12` seconds
- Visual comparison: `results/louvainne_comparison.png`

## Multi-dataset benchmark

The cleaned repo now benchmarks the maintained baseline and improved pipeline on four prepared attributed graph datasets:
`Cora`, `CiteSeer`, `PubMed`, and `BlogCatalog`.

Artifacts are written in an organized layout:

- `results/<Dataset>/comparison_results.json`
- `results/<Dataset>/comparison_plot.png`
- `results/benchmark_summary.json`
- `results/benchmark_summary.md`
- `results/benchmark_summary.png`

Latest saved benchmark:

| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | `classifier_seed` | `0.5916 ± 0.0031` | `0.7226 ± 0.0053` | `0.53` | `0.32` | `3.30 ± 1.53` | `1.06 ± 0.02` |
| CiteSeer | `classifier_seed` | `0.4958 ± 0.0125` | `0.6638 ± 0.0047` | `0.53` | `0.36` | `3.43 ± 1.64` | `1.23 ± 0.01` |
| PubMed | `classifier_seed` | `0.5830 ± 0.0167` | `0.7246 ± 0.0031` | `2.52` | `1.52` | `8.53 ± 1.69` | `6.87 ± 1.98` |
| BlogCatalog | `prepared_split` | `0.7517 ± 0.0080` | `0.9143 ± 0.0066` | `1.30` | `0.90` | `6.72 ± 2.64` | `4.21 ± 1.93` |

## Best configurations found

### Best reproducible repo-style baseline

- Fusion mode: method 2
- Threshold: `alpha = 0.2`
- Rule:
  - existing structure edges start at weight `1`
  - new predicted edges get weight `similarity`
  - predicted edges that already existed get weight `1 + similarity`

### Best adaptive hybrid graph

- Similarity: cosine
- Top-k: `15`
- Mutual neighbors only: `True`
- Minimum similarity: `0.2`
- Overlap edge weight: `1 + 1.0 * similarity`
- New edge weight: `0.75 * similarity`

### Best final improved pipeline

- Base graph: the adaptive hybrid graph above
- Sparse attention residual: `gamma = 0.5`
- Attention temperature: `1.0`
- Attribute residual branch: `128`-dimensional SVD feature embedding
- Ensemble size: `1` was already best in the searched region

## Accuracy gain

- Against structure only:
  - micro-F1 improved from `0.6760` to `0.7722`
  - absolute gain: `+0.0962`

- Against the best reproducible repo-style baseline:
  - micro-F1 improved from `0.7356` to `0.7722`
  - absolute gain: `+0.0366`

## Practical takeaway

The strongest direction was not late fusion and not the repo's dense all-node attention. The biggest improvements came from:

1. building a cleaner attribute graph with mutual top-k cosine neighbors,
2. using weighted hybrid edges instead of unweighted thresholded additions,
3. refining embeddings with sparse neighborhood attention, and
4. keeping a compact attribute residual branch alongside the LouvainNE embedding.
