#!/usr/bin/env python3
"""
Hyperparameter optimization for LouvainNE.
Searches for the best top_k, ensemble_size, and min_similarity to maximize node classification accuracy.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import torch
import numpy as np
from run_louvainne_experiments import (
    GraphData,
    LouvainNERunner,
    load_processed_graph,
    unique_undirected_edges,
    edges_to_dict,
    build_topk_predictions,
    fuse_adaptive_edges,
    low_rank_projection,
    normalize_rows,
    concat_features,
    fit_linear_probe,
    aligned_ensemble,
    set_global_seed,
)

REPO_ROOT = Path(__file__).resolve().parent

def run_optimization(dataset_name: str,
                    top_k_range: List[int],
                    ensemble_range: List[int],
                    sim_range: List[float],
                    dataset_path: Path):

    print(f"Starting optimization for {dataset_name}...")
    data = load_processed_graph(dataset_path)
    num_nodes = data.num_nodes

    # Fixed settings for the "improved" pipeline
    overlap_scale = 1.0
    new_scale = 0.75
    feature_dim = 128 # PCA dim

    runner = LouvainNERunner(REPO_ROOT, num_nodes, 256)
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))

    # Precompute feature projection to save time
    projected = low_rank_projection(data.x, feature_dim)
    graph_features = normalize_rows(projected[:, :min(64, projected.shape[1])])
    feature_embedding = normalize_rows(projected)

    best_score = -1.0
    best_params = {}
    all_results = []

    # Grid Search
    for k in top_k_range:
        for sim in sim_range:
            for ens in ensemble_range:
                print(f"Testing: k={k}, sim={sim}, ens={ens}...", end=" ", flush=True)

                # 1. Build attribute graph
                # Note: we use a simplified cosine similarity for the top-k search
                # as build_topk_predictions expects a similarity matrix.
                # To keep it efficient, we compute similarity in blocks inside the function.
                # However, build_topk_predictions takes a torch.Tensor.
                # We'll use a small trick: compute similarity on the fly or use the repo's logic.

                # Since build_topk_predictions needs the full matrix, for Cora it's fine.
                # For larger graphs, we should use build_blockwise_topk_predictions.
                from run_louvainne_experiments import build_blockwise_topk_predictions

                predicted = build_blockwise_topk_predictions(
                    graph_features,
                    top_k=k,
                    mutual=True,
                    min_similarity=sim,
                    block_size=512
                )

                fused_edges = fuse_adaptive_edges(
                    structure_edges,
                    predicted,
                    overlap_scale=overlap_scale,
                    new_scale=new_scale
                )

                # 2. Generate embeddings (Ensembled)
                seeds = list(range(ens))
                graph_embedding = aligned_ensemble(runner, fused_edges, seeds)

                # 3. Final embedding concatenation
                final_embeddings = concat_features([graph_embedding, feature_embedding])

                # 4. Evaluate using linear probe
                # We use a single seed for the probe to keep it fast during search
                metrics = fit_linear_probe(final_embeddings, data, [0.0, 1e-3, 1e-2], 42)
                score = metrics["selection_score"]

                print(f"Score: {score:.4f}")

                all_results.append({
                    "k": k,
                    "sim": sim,
                    "ens": ens,
                    "score": score
                })

                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "sim": sim, "ens": ens}

    print(f"\nOptimization Complete!")
    print(f"Best Params: {best_params} with Score: {best_score:.4f}")

    return best_params, all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    args = parser.parse_args()

    # Define search space
    top_k_range = [10, 15, 20, 30]
    ensemble_range = [1, 3, 5]
    sim_range = [0.1, 0.2, 0.3]

    # Path to Cora processed data
    path = REPO_ROOT / "data/Planetoid/Cora/processed/data.pt"

    best_params, all_res = run_optimization(
        args.dataset,
        top_k_range,
        ensemble_range,
        sim_range,
        path
    )

    # Save results
    output_dir = REPO_ROOT / "results" / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump({"best_params": best_params, "all_results": all_res}, f, indent=2)

    print(f"Results saved to {output_dir / 'optimization_results.json'}")

if __name__ == "__main__":
    main()
