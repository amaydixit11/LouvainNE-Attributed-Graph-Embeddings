# Link Prediction Benchmark Design

**Date:** 2026-04-09

**Goal**

Add a reproducible link-prediction benchmark framework to this repository that:
- evaluates existing LouvainNE-based pipelines on link prediction,
- supports both small benchmark datasets and large OGB datasets,
- records accuracy and timing in paper-ready artifacts,
- leaves clean extension points for external baselines and scalability comparisons.

## Scope

This design covers:
- link prediction evaluation code,
- dataset and split handling for current prepared datasets and OGB datasets,
- embedding backends for current LouvainNE methods,
- reporting artifacts for tables and plots.

This design does not yet cover:
- report manuscript edits,
- final literature-sourced SOTA numbers from papers,
- large-scale cluster or multi-machine execution.

## Current Repo State

The repo already contains:
- reproducible node-classification experiments for `Cora`, `CiteSeer`, `PubMed`, and `BlogCatalog`,
- a maintained LouvainNE runner,
- timing-aware result summaries written under `results/`.

The repo does not currently contain:
- link-prediction edge splitting,
- negative sampling,
- link decoders,
- OGB dataset loading,
- OGB evaluator integration,
- link-prediction summary tables.

## Requirements

1. Reuse the current LouvainNE embedding pipelines where possible.
2. Support small-dataset metrics:
   - `AUC`
   - `Average Precision (AP)`
3. Support OGB-style metrics where required:
   - `Hits@K`
   - `MRR`
4. Track runtime in a way that supports the project thesis:
   - one-time setup/preprocessing time,
   - embedding time,
   - decoder training time if any,
   - full evaluation time.
5. Produce machine-readable and human-readable outputs:
   - per-dataset JSON
   - benchmark summary JSON
   - benchmark summary markdown
6. Keep the framework extensible for later external baselines.

## Proposed Architecture

### 1. New Benchmark Entry Point

Add a new top-level script:
- `benchmark_link_prediction.py`

Responsibilities:
- parse CLI arguments,
- load requested datasets,
- dispatch dataset-specific evaluation protocol,
- run selected methods,
- aggregate metrics and timing,
- write result artifacts.

This mirrors the role currently played by `benchmark_datasets.py` for node classification.

### 2. Dataset Abstraction

Introduce a dataset bundle abstraction for link prediction with:
- node features,
- graph edges,
- optional labels for metadata only,
- optional predefined split payloads,
- dataset family tag:
  - `prepared_small`
  - `ogb_linkprop`

Prepared datasets:
- `Cora`
- `CiteSeer`
- `PubMed`
- `BlogCatalog`

Planned OGB datasets:
- at least one smaller link benchmark for correctness and comparison,
- at least one large benchmark for scalability claims.

Initial OGB targets should be selected from datasets with practical CPU evaluation and standard link prediction protocol. The implementation should not hard-code assumptions that only fit citation graphs.

### 3. Split Protocol Layer

Add two protocol adapters.

#### A. Small-Graph Edge Classification Protocol

For prepared datasets:
- remove a validation edge set and a test edge set from the graph,
- build embeddings on the remaining train graph only,
- sample negative edges separately for validation and test,
- score positive and negative edges,
- compute `AUC` and `AP`.

Constraints:
- maintain undirected-edge uniqueness,
- exclude self-loops,
- avoid negative edges that already exist in the full graph,
- use deterministic seeds.

#### B. OGB Protocol

For OGB datasets:
- use the dataset-provided split where available,
- use official evaluator metrics and candidate sets,
- compute `Hits@K` and/or `MRR` exactly as required by the chosen dataset.

This avoids invalid comparisons caused by custom splits on OGB benchmarks.

### 4. Embedding Backend Interface

Define a method interface shaped like:
- build train graph representation,
- generate node embeddings,
- expose timing breakdown.

Initial in-repo methods:
- `louvainne_structural`
- `louvainne_repo_baseline`
- `louvainne_improved`

Planned external baseline hooks:
- `deepwalk`
- `node2vec`
- `gae` or `vgae`
- optional message-passing baselines for later scalability comparison

External methods should be optional so the repo remains usable even when some dependencies are unavailable.

### 5. Edge Decoder Layer

Add decoder/scorer options:
- `dot`
- `cosine`
- `hadamard_logreg`

Design choice:
- use a non-parametric scorer like `dot` or `cosine` as the default for pure embedding benchmarks,
- allow `hadamard_logreg` as a stronger learned decoder for paper tables.

Timing must separate:
- embedding construction,
- decoder fitting,
- full evaluation.

### 6. Reporting Schema

Write outputs in a structure parallel to the existing benchmark layout:
- `results/link_prediction/<Dataset>/comparison_results.json`
- `results/link_prediction/<Dataset>/comparison_plot.png` if plotting is implemented
- `results/link_prediction/benchmark_summary.json`
- `results/link_prediction/benchmark_summary.md`

Each method result should include:
- metric means/std if repeated,
- raw per-run values,
- split protocol,
- decoder type,
- setup time,
- embedding time,
- decoder time,
- total evaluation time.

### 7. Paper-Ready Table Support

Summary markdown should support three table families:
- in-repo LouvainNE method comparison,
- external baseline comparison when enabled,
- scalability table with time and graph size columns.

The summary format should make it easy to lift values into the UGQ301 report without manual recomputation.

## File Layout

Planned additions:
- `benchmark_link_prediction.py`
- `results/link_prediction/` generated artifacts

Likely helper module split:
- `link_prediction_data.py`
- `link_prediction_protocols.py`
- `link_prediction_methods.py`
- `link_prediction_decoders.py`
- `link_prediction_reporting.py`

The final split can be adjusted to match repo scale, but responsibilities should remain separated.

## Evaluation Policy

### Small Datasets

Primary metrics:
- `AUC`
- `AP`

Recommended repeated-evaluation policy:
- fixed edge split seed list,
- deterministic negative sampling per split seed,
- report mean and std.

### OGB Datasets

Primary metrics:
- official evaluator metrics only

Recommended repeated-evaluation policy:
- use official splits,
- repeat only where the method itself has randomness,
- do not invent alternate leaderboard metrics for the headline table.

## Baseline Strategy

Phase 1 required:
- current LouvainNE structural baseline,
- current repo-style attributed baseline,
- current improved pipeline.

Phase 2 planned:
- `DeepWalk`
- `node2vec`
- at least one GNN/autoencoder-style link predictor where computationally feasible.

The benchmark must be able to run without phase-2 methods installed.

## Scalability Positioning

The project claim for large graphs is expected to be:
- our method may not beat supervised GNNs in raw link-prediction accuracy,
- but it should be competitive enough to be credible,
- and should win on preprocessing/embedding time and graph-scale practicality.

Therefore the benchmark must preserve:
- exact node and edge counts,
- per-method runtime,
- optional memory notes if later added.

## Risks and Mitigations

### Risk 1: Leakage through split construction

If embeddings are built on graphs that still include validation/test positives, link-prediction numbers are invalid.

Mitigation:
- always embed on train graph only,
- centralize split handling in one module.

### Risk 2: Invalid negative sampling on dense graphs

Naive rejection sampling can become slow or biased on dense datasets.

Mitigation:
- implement a deterministic sampler with existence checks and bounded retries,
- support dataset-provided negatives where available.

### Risk 3: OGB-specific protocol mismatch

Using custom metrics or custom negatives on OGB would make the results non-comparable.

Mitigation:
- use official OGB evaluator paths for OGB datasets.

### Risk 4: Decoder choice obscures embedding comparison

A strong learned decoder can hide differences between embeddings.

Mitigation:
- report decoder type explicitly,
- keep a simple default decoder for core embedding comparisons.

## Recommended Implementation Order

1. Build small-dataset split and scoring pipeline.
2. Plug in the three existing LouvainNE methods.
3. Write JSON and markdown reporting.
4. Add OGB dataset loading and official evaluator support.
5. Add optional external baselines.
6. Add paper-facing comparison tables.

## Success Criteria

The design is successful if the repo can produce:
- reproducible link-prediction results on the current four datasets,
- at least one working OGB link-prediction benchmark,
- timing-aware summaries for all methods,
- result files that can directly support the UGQ301 report’s link-prediction section.
