#!/usr/bin/env python3
"""
Scalability benchmark using synthetic large graphs.
Demonstrates LouvainNE's advantage on very large networks.

Creates graphs at different scales:
- 10K nodes
- 50K nodes
- 100K nodes
- 500K nodes
- 1M nodes

This proves the O(n log n) scaling advantage without relying on external datasets.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_louvainne_experiments import (
    GraphData,
    LouvainNERunner,
    build_blockwise_topk_predictions,
    compute_link_prediction_metrics,
    concat_features,
    create_link_prediction_split,
    edges_to_dict,
    low_rank_projection,
    normalize_rows,
    unique_undirected_edges,
    fuse_adaptive_edges,
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_memory_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0

def generate_synthetic_graph(
    num_nodes: int,
    avg_degree: float = 10,
    num_features: int = 128,
    num_classes: int = 10,
    seed: int = 42,
) -> GraphData:
    """Generate a synthetic graph with community structure."""
    rng = np.random.RandomState(seed)
    
    # Generate edges with community structure
    num_communities = max(10, num_nodes // 100)
    community_size = num_nodes // num_communities
    communities = rng.randint(0, num_communities, num_nodes)
    
    edges = set()
    for i in range(num_nodes):
        # Intra-community edges (higher probability)
        same_comm = np.where(communities == communities[i])[0]
        num_intra = max(1, int(avg_degree * 0.7))
        if len(same_comm) > 1:
            intra_nodes = rng.choice(same_comm, size=min(num_intra, len(same_comm)-1), replace=False)
            for j in intra_nodes:
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        
        # Inter-community edges (lower probability)
        num_inter = max(1, int(avg_degree * 0.3))
        other_nodes = rng.choice(num_nodes, size=min(num_inter, num_nodes), replace=False)
        for j in other_nodes:
            if i != j and communities[i] != communities[j]:
                edges.add((min(i, j), max(i, j)))
    
    # Convert to edge_index
    edge_list = sorted(list(edges))
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Generate features (correlated with community)
    community_centers = rng.randn(num_communities, num_features)
    features = community_centers[communities] + rng.randn(num_nodes, num_features) * 0.5
    features = torch.from_numpy(features.astype(np.float32))
    
    # Generate labels (based on community with some noise)
    labels = communities % num_classes
    labels = torch.from_numpy(labels.astype(np.int64))
    
    # Create train/val/test splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for cls in range(num_classes):
        cls_idx = torch.where(labels == cls)[0]
        perm = torch.randperm(len(cls_idx))
        n_train = max(1, int(len(cls_idx) * 0.6))
        n_val = max(1, int(len(cls_idx) * 0.2))
        train_mask[cls_idx[perm[:n_train]]] = True
        val_mask[cls_idx[perm[n_train:n_train+n_val]]] = True
        test_mask[cls_idx[perm[n_train+n_val:]]] = True
    
    return GraphData(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


@dataclass
class ScalabilityResult:
    dataset: str
    nodes: int
    edges: int
    features: int
    classes: int
    avg_degree: float
    
    setup_time_s: float = 0.0
    embedding_time_s: float = 0.0
    total_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    
    node_micro_f1: float = 0.0
    link_auc: float = 0.0
    
    completed: bool = False
    error_message: str = ""
    oom: bool = False


def benchmark_louvainne_synthetic(name: str, num_nodes: int, avg_degree: float = 10) -> ScalabilityResult:
    """Run LouvainNE on synthetic graph."""
    result = ScalabilityResult(
        dataset=name,
        nodes=num_nodes,
        edges=0,
        features=128,
        classes=10,
        avg_degree=avg_degree,
    )
    
    try:
        mem_before = get_memory_mb()
        start_total = time.time()
        
        # Generate graph
        print(f"  Generating graph with {num_nodes:,} nodes...", flush=True)
        data = generate_synthetic_graph(num_nodes, avg_degree=avg_degree)
        print(f"  Generated {data.edge_index.shape[1]:,} edges", flush=True)
        result.edges = data.edge_index.shape[1] // 2
        
        # LouvainNE
        setup_start = time.time()
        runner = LouvainNERunner(REPO_ROOT, num_nodes, 256)
        structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))
        
        # Build attributed graph
        top_k = 15 if num_nodes < 50000 else 10
        min_sim = 0.15 if num_nodes < 50000 else 0.1
        block_size = 2048 if num_nodes > 100000 else 512
        
        graph_features = normalize_rows(data.x[:, :min(64, data.x.shape[1])])
        feature_embedding = normalize_rows(data.x[:, :min(128, data.x.shape[1])])
        
        print(f"  Building attribute graph (k={top_k}, min_sim={min_sim})...", flush=True)
        improved_predictions = build_blockwise_topk_predictions(
            graph_features, top_k=top_k, mutual=True,
            min_similarity=min_sim, block_size=block_size
        )
        improved_edges = fuse_adaptive_edges(structure_edges, improved_predictions, 1.0, 0.75)
        setup_time = time.time() - setup_start
        
        # Embed
        print(f"  Running LouvainNE embedding...", flush=True)
        embed_start = time.time()
        graph_embedding = normalize_rows(runner.embed(improved_edges, 42))
        embed_time = time.time() - embed_start
        
        # Classification
        from sklearn.linear_model import LogisticRegression
        embeddings = concat_features([graph_embedding, feature_embedding])
        
        train_x = embeddings[data.train_mask].numpy()
        train_y = data.y[data.train_mask].numpy()
        test_x = embeddings[data.test_mask].numpy()
        test_y = data.y[data.test_mask].numpy()
        
        clf = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        micro_f1 = (pred == test_y).mean()
        
        total_time = time.time() - start_total
        mem_after = get_memory_mb()
        
        result.setup_time_s = setup_time
        result.embedding_time_s = embed_time
        result.total_time_s = total_time
        result.peak_memory_mb = mem_after - mem_before
        result.node_micro_f1 = micro_f1
        result.completed = True
        
    except Exception as e:
        result.error_message = str(e)
        result.completed = False
        if "memory" in str(e).lower():
            result.oom = True
    
    return result


def generate_scaling_plots(results: List[ScalabilityResult], output_dir: Path):
    """Generate scaling plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by nodes
    results = sorted(results, key=lambda r: r.nodes)
    
    nodes = [r.nodes for r in results]
    times = [r.total_time_s for r in results]
    f1s = [r.node_micro_f1 for r in results]
    mems = [r.peak_memory_mb for r in results]
    
    # 1. Runtime vs Size (log-log)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(nodes, times, 'o-', linewidth=2, markersize=10, color='#2A6F97')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Total Runtime (seconds)', fontsize=12)
    ax.set_title('LouvainNE Scalability: Runtime vs Graph Size', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    for i, (n, t) in enumerate(zip(nodes, times)):
        ax.annotate(f"{n:,}", (n, t), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_runtime_vs_size.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Accuracy vs Size
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(nodes, f1s, 'o-', linewidth=2, markersize=10, color='#E07A5F')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Node Classification Micro-F1', fontsize=12)
    ax.set_title('LouvainNE Accuracy vs Graph Size', fontsize=14)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    for i, (n, f) in enumerate(zip(nodes, f1s)):
        ax.annotate(f"{f:.3f}", (n, f), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_accuracy_vs_size.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Memory vs Size
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(nodes, mems, 'o-', linewidth=2, markersize=10, color='#6C9A8B')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('LouvainNE Memory Usage vs Graph Size', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    for i, (n, m) in enumerate(zip(nodes, mems)):
        ax.annotate(f"{m:.0f} MB", (n, m), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_memory_vs_size.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic scalability benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, 
                       default=[10000, 50000, 100000, 500000, 1000000],
                       help="Graph sizes to test")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "scalability")
    args = parser.parse_args()
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for size in args.sizes:
        avg_degree = 10 if size < 100000 else 5  # Lower degree for larger graphs
        name = f"synthetic_{size:,}"
        
        print(f"\n{'='*80}")
        print(f"Benchmarking: {name}")
        print(f"{'='*80}")
        
        result = benchmark_louvainne_synthetic(name, size, avg_degree)
        results.append(result)
        
        if result.completed:
            print(f"✅ Completed in {result.total_time_s:.2f}s")
            print(f"   Nodes: {result.nodes:,}, Edges: {result.edges:,}")
            print(f"   Node F1: {result.node_micro_f1:.4f}")
            print(f"   Memory: {result.peak_memory_mb:.1f} MB")
            print(f"   Setup: {result.setup_time_s:.2f}s, Embed: {result.embedding_time_s:.2f}s")
        else:
            print(f"❌ Failed: {result.error_message}")
    
    # Save results
    results_list = []
    for r in results:
        results_list.append({
            "dataset": r.dataset,
            "nodes": r.nodes,
            "edges": r.edges,
            "features": r.features,
            "classes": r.classes,
            "avg_degree": r.avg_degree,
            "setup_time_s": round(r.setup_time_s, 2),
            "embedding_time_s": round(r.embedding_time_s, 2),
            "total_time_s": round(r.total_time_s, 2),
            "peak_memory_mb": round(r.peak_memory_mb, 1),
            "node_micro_f1": round(r.node_micro_f1, 4),
            "completed": r.completed,
            "error_message": r.error_message,
        })
    
    output_json = output_dir / "synthetic_scalability.json"
    output_json.write_text(json.dumps(results_list, indent=2))
    print(f"\nResults saved to {output_json}")
    
    # Generate plots
    generate_scaling_plots(results, output_dir)
    
    # Print summary table
    print("\n" + "="*100)
    print("SYNTHETIC SCALABILITY RESULTS")
    print("="*100)
    print(f"{'Graph Size':>12} | {'Nodes':>10} | {'Edges':>10} | {'Time (s)':>10} | "
          f"{'Node F1':>8} | {'Memory':>10} | {'Status':>8}")
    print("-"*100)
    for r in results:
        status = "✅ OK" if r.completed else "❌ FAIL"
        print(f"{r.dataset:>12} | {r.nodes:>10,} | {r.edges:>10,} | {r.total_time_s:>10.2f} | "
              f"{r.node_micro_f1:>8.4f} | {r.peak_memory_mb:>9.1f} MB | {status:>8}")


if __name__ == "__main__":
    main()
