#!/usr/bin/env python3
"""
Ablation study: isolate the contribution of each component in the improved pipeline.

Components tested:
1. Structure-only LouvainNE embeddings (baseline)
2. Structure + attribute-derived edges (graph construction ablation)
3. Structure-only + SVD features (feature concatenation ablation)
4. Structure + attribute edges + SVD features (combined, no attention)
5. Full improved pipeline (structure + attribute edges + SVD features + attention)

This isolates how much of the improvement comes from:
- The improved graph construction (attribute-derived edges)
- The SVD feature concatenation (which is a well-known technique, similar to TADW/ANRL)
- The sparse attention mechanism
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_louvainne_experiments import (
    GraphData,
    LouvainNERunner,
    concat_features,
    edges_to_dict,
    fit_linear_probe,
    fuse_adaptive_edges,
    load_processed_graph,
    normalize_rows,
    unique_undirected_edges,
)

DEFAULT_DATASETS = ["Cora", "CiteSeer", "PubMed"]
DEFAULT_PENALTIES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
EVAL_SEEDS = [1548, 1549, 1550, 1551, 1552]
GRAPH_PROJECTION_DIM = 64
RESIDUAL_DIM = 128
IMPROVED_TOP_K = 15
IMPROVED_MIN_SIMILARITY = 0.2
BLOCK_SIZE = 512


@dataclass
class DatasetBundle:
    name: str
    source: str
    data: GraphData


def load_dataset(name: str) -> DatasetBundle:
    path = REPO_ROOT / "data" / "Planetoid" / name / "processed" / "data.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {name} at {path}. Run `python prepare_datasets.py` first."
        )
    data = load_processed_graph(path)
    return DatasetBundle(
        name=name,
        source=str(path.relative_to(REPO_ROOT)),
        data=data,
    )


def low_rank_projection(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    centered = x.to(torch.float32) - x.to(torch.float32).mean(dim=0, keepdim=True)
    q = min(max(out_dim + 8, out_dim), min(centered.shape))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    return centered @ v[:, :out_dim]


def normalize_rows_safe(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def build_blockwise_topk_predictions(
    features: torch.Tensor,
    top_k: int,
    mutual: bool,
    min_similarity: float,
    block_size: int = BLOCK_SIZE,
) -> Dict[Tuple[int, int], float]:
    normalized = normalize_rows_safe(features)
    num_nodes = normalized.shape[0]
    top_k = max(1, min(top_k, num_nodes - 1))
    neighbor_maps: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]

    for start in range(0, num_nodes, block_size):
        end = min(start + block_size, num_nodes)
        scores = normalized[start:end] @ normalized.t()
        row_ids = torch.arange(start, end)
        scores[torch.arange(end - start), row_ids] = -1.0
        values, indices = torch.topk(scores, k=top_k, dim=1)
        for row_offset, node_id in enumerate(range(start, end)):
            row_values = values[row_offset]
            row_indices = indices[row_offset]
            for score, neighbor in zip(row_values.tolist(), row_indices.tolist()):
                if score < min_similarity:
                    continue
                neighbor_maps[node_id][int(neighbor)] = float(score)

    predictions: Dict[Tuple[int, int], float] = {}
    if mutual:
        for node_id, neighbors in enumerate(neighbor_maps):
            for neighbor, score in neighbors.items():
                if node_id >= neighbor:
                    continue
                reverse = neighbor_maps[neighbor].get(node_id)
                if reverse is None:
                    continue
                predictions[(node_id, neighbor)] = float((score + reverse) * 0.5)
        return predictions

    for node_id, neighbors in enumerate(neighbor_maps):
        for neighbor, score in neighbors.items():
            a, b = (node_id, neighbor) if node_id < neighbor else (neighbor, node_id)
            if a == b:
                continue
            previous = predictions.get((a, b))
            predictions[(a, b)] = score if previous is None else max(previous, score)
    return predictions


def apply_sparse_attention(
    embeddings: torch.Tensor,
    edge_scores: Dict[Tuple[int, int], float],
    gamma: float,
    temperature: float,
) -> torch.Tensor:
    if gamma <= 0.0:
        return normalize_rows_safe(embeddings)

    num_nodes = embeddings.shape[0]
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for (u, v), score in edge_scores.items():
        logit = float(score) / temperature
        adjacency[u].append((v, logit))
        adjacency[v].append((u, logit))

    refined = torch.empty_like(embeddings)
    for node_id, neighbors in enumerate(adjacency):
        indices = [node_id] + [neighbor for neighbor, _ in neighbors]
        logits = torch.tensor(
            [1.0 / temperature] + [score for _, score in neighbors],
            dtype=embeddings.dtype,
        )
        weights = torch.softmax(logits, dim=0).unsqueeze(1)
        refined[node_id] = (weights * embeddings[indices]).sum(dim=0)

    return normalize_rows_safe((1.0 - gamma) * embeddings + gamma * refined)


def summarize_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    result = {}
    for key in ["val_micro_f1", "val_macro_f1", "test_micro_f1", "test_macro_f1"]:
        values = [run[key] for run in runs]
        result[f"{key}_mean"] = float(sum(values) / len(values))
        # Bessel's correction (ddof=1)
        std = (sum((v - result[f"{key}_mean"]) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
        result[f"{key}_std"] = float(std)
    return result


def run_ablation(dataset_name: str, embedding_dim: int) -> Dict[str, object]:
    """Run ablation study on a single dataset."""
    bundle = load_dataset(dataset_name)
    data = bundle.data
    runner = LouvainNERunner(REPO_ROOT, data.num_nodes, embedding_dim)
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))

    # Prepare features
    projection_dim = min(RESIDUAL_DIM, data.x.shape[0] - 1, data.x.shape[1])
    projected = low_rank_projection(data.x, projection_dim)
    graph_dim = min(GRAPH_PROJECTION_DIM, projected.shape[1])
    graph_features = normalize_rows_safe(projected[:, :graph_dim])
    feature_embedding = normalize_rows_safe(projected[:, :projection_dim])

    # Build attribute-derived edges (same as improved pipeline)
    improved_predictions = build_blockwise_topk_predictions(
        graph_features,
        top_k=IMPROVED_TOP_K,
        mutual=True,
        min_similarity=IMPROVED_MIN_SIMILARITY,
    )
    improved_edges = fuse_adaptive_edges(structure_edges, improved_predictions, 1.0, 0.75)

    print(f"\n{'='*60}")
    print(f"Ablation Study: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Nodes: {data.num_nodes}, Edges: {len(structure_edges)} (structure), "
          f"{len(improved_predictions)} (attribute-derived)")
    print(f"  Feature dim: {data.x.shape[1]}, Projected: {projection_dim}")

    # Define ablation configurations
    ablation_configs = [
        {
            "name": "1_structure_only",
            "description": "Structure-only LouvainNE (baseline)",
            "edges": structure_edges,
            "use_svd": False,
            "attention_gamma": 0.0,
        },
        {
            "name": "2_structure_plus_attr_edges",
            "description": "Structure + attribute-derived edges (no SVD, no attention)",
            "edges": improved_edges,
            "use_svd": False,
            "attention_gamma": 0.0,
        },
        {
            "name": "3_structure_only_plus_svd",
            "description": "Structure-only + SVD features (no attribute edges, no attention)",
            "edges": structure_edges,
            "use_svd": True,
            "attention_gamma": 0.0,
        },
        {
            "name": "4_structure_plus_attr_edges_plus_svd",
            "description": "Structure + attribute edges + SVD features (no attention)",
            "edges": improved_edges,
            "use_svd": True,
            "attention_gamma": 0.0,
        },
        {
            "name": "5_full_improved",
            "description": "Full pipeline: structure + attribute edges + SVD + attention",
            "edges": improved_edges,
            "use_svd": True,
            "attention_gamma": 0.5,
        },
    ]

    ablation_results = {}
    for config in ablation_configs:
        print(f"\n  Running: {config['description']}...", flush=True)
        start = time.perf_counter()

        runs = []
        for seed in EVAL_SEEDS:
            # Build embeddings
            graph_embedding = normalize_rows_safe(runner.embed(config["edges"], seed))

            if config["attention_gamma"] > 0.0:
                graph_embedding = apply_sparse_attention(
                    graph_embedding,
                    improved_predictions,
                    gamma=config["attention_gamma"],
                    temperature=1.0,
                )

            parts = [graph_embedding]
            if config["use_svd"]:
                parts.append(feature_embedding)
            embeddings = concat_features(parts)

            metrics = fit_linear_probe(embeddings, data, DEFAULT_PENALTIES, seed)
            metrics["seed"] = float(seed)
            runs.append(metrics)

        elapsed = time.perf_counter() - start
        summary = summarize_runs(runs)
        summary["time_seconds"] = float(elapsed)
        summary["embedding_dim"] = int(embeddings.shape[1])
        ablation_results[config["name"]] = {
            "description": config["description"],
            "config": config,
            "runs": runs,
            "summary": summary,
        }

        print(f"    test_micro_f1: {summary['test_micro_f1_mean']:.4f} ± {summary['test_micro_f1_std']:.4f}  "
              f"({elapsed:.1f}s, emb_dim={embeddings.shape[1]})", flush=True)

    return {
        "dataset": dataset_name,
        "num_nodes": data.num_nodes,
        "num_edges_structure": len(structure_edges),
        "num_edges_attribute_derived": len(improved_predictions),
        "feature_dim": int(data.x.shape[1]),
        "projected_dim": projection_dim,
        "embedding_dim": embedding_dim,
        "ablation": ablation_results,
    }


def write_ablation_report(
    all_results: List[Dict[str, object]],
    output_path: Path,
    plot_path: Path,
) -> None:
    """Write markdown report and generate plots."""
    lines = [
        "# Ablation Study: Improved Pipeline Component Analysis",
        "",
        "This study isolates the contribution of each component in the improved pipeline:",
        "",
        "1. **Structure-only**: Baseline LouvainNE on graph structure",
        "2. **Structure + attribute edges**: Adds cosine-similarity-derived edges",
        "3. **Structure + SVD**: Adds SVD-compressed raw features (no attribute edges)",
        "4. **Structure + attribute edges + SVD**: Combined, no attention",
        "5. **Full improved**: All components + sparse attention (gamma=0.5)",
        "",
        "---",
        "",
    ]

    # Summary table per dataset
    for result in all_results:
        dataset = result["dataset"]
        ablation = result["ablation"]

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"**Dataset stats:** {result['num_nodes']} nodes, "
                     f"{result['num_edges_structure']} structure edges, "
                     f"{result['num_edges_attribute_derived']} attribute-derived edges")
        lines.append("")
        lines.append("| Component | Test Micro-F1 | Std | Time (s) | Emb Dim |")
        lines.append("|---|---:|---:|---:|---:|")

        for key in ["1_structure_only", "2_structure_plus_attr_edges",
                     "3_structure_only_plus_svd", "4_structure_plus_attr_edges_plus_svd",
                     "5_full_improved"]:
            if key not in ablation:
                continue
            item = ablation[key]
            s = item["summary"]
            lines.append(
                f"| {item['description']} | "
                f"{s['test_micro_f1_mean']:.4f} | ± {s['test_micro_f1_std']:.4f} | "
                f"{s['time_seconds']:.1f} | {s['embedding_dim']} |"
            )

        # Compute deltas
        base = ablation["1_structure_only"]["summary"]["test_micro_f1_mean"]
        attr_delta = ablation["2_structure_plus_attr_edges"]["summary"]["test_micro_f1_mean"] - base
        svd_delta = ablation["3_structure_only_plus_svd"]["summary"]["test_micro_f1_mean"] - base
        combined_delta = ablation["4_structure_plus_attr_edges_plus_svd"]["summary"]["test_micro_f1_mean"] - base
        full_delta = ablation["5_full_improved"]["summary"]["test_micro_f1_mean"] - base

        lines.append("")
        lines.append("### Component Contribution Analysis")
        lines.append("")
        lines.append(f"- **Baseline (structure-only):** {base:.4f}")
        lines.append(f"- **Attribute edges only:** +{attr_delta:.4f}")
        lines.append(f"- **SVD features only:** +{svd_delta:.4f}")
        lines.append(f"- **Combined (no attention):** +{combined_delta:.4f}")
        lines.append(f"- **Full pipeline (with attention):** +{full_delta:.4f}")
        lines.append("")

        # Key insight
        if abs(svd_delta) > abs(attr_delta):
            lines.append(f"**Key insight:** SVD feature concatenation contributes more ({svd_delta:+.4f}) "
                         f"than attribute-derived edges ({attr_delta:+.4f}). This suggests the improved "
                         f"pipeline's win may be largely driven by the SVD features, a well-known technique "
                         f"similar to TADW/ANRL.")
        else:
            lines.append(f"**Key insight:** Attribute-derived edges contribute more ({attr_delta:+.4f}) "
                         f"than SVD features alone ({svd_delta:+.4f}).")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Cross-dataset comparison
    lines.append("## Cross-Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Structure | +Attr Edges | +SVD | Combined | Full |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for result in all_results:
        ablation = result["ablation"]
        vals = []
        for key in ["1_structure_only", "2_structure_plus_attr_edges",
                     "3_structure_only_plus_svd", "4_structure_plus_attr_edges_plus_svd",
                     "5_full_improved"]:
            if key in ablation:
                vals.append(f"{ablation[key]['summary']['test_micro_f1_mean']:.4f}")
            else:
                vals.append("N/A")
        lines.append(f"| {result['dataset']} | {' | '.join(vals)} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Generate plots
    num_datasets = len(all_results)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5))
    if num_datasets == 1:
        axes = [axes]

    component_names = ["Structure", "+Attr Edges", "+SVD", "Combined", "Full"]
    component_keys = ["1_structure_only", "2_structure_plus_attr_edges",
                      "3_structure_only_plus_svd", "4_structure_plus_attr_edges_plus_svd",
                      "5_full_improved"]
    colors = ["#2A6F97", "#6C9A8B", "#8F5EA2", "#C8553D", "#D1495B"]

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        ablation = result["ablation"]
        means = []
        stds = []
        for key in component_keys:
            if key in ablation:
                means.append(ablation[key]["summary"]["test_micro_f1_mean"])
                stds.append(ablation[key]["test_micro_f1_std"])
            else:
                means.append(0.0)
                stds.append(0.0)

        x = range(len(component_names))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(component_names)])
        ax.set_xticks(list(x), component_names, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Test Micro-F1")
        ax.set_title(result["dataset"])
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for i, v in enumerate(means):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Ablation Study: Component Contributions to Improved Pipeline", fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for improved pipeline components.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args()

    all_results = []
    for dataset_name in args.datasets:
        result = run_ablation(dataset_name, args.embedding_dim)
        all_results.append(result)

    # Write results
    output_dir = REPO_ROOT / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "ablation_report.md"
    plot_path = output_dir / "ablation_plot.png"
    json_path = output_dir / "ablation_results.json"

    write_ablation_report(all_results, report_path, plot_path)
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nAblation report: {report_path}")
    print(f"Ablation plot: {plot_path}")
    print(f"Ablation JSON: {json_path}")


if __name__ == "__main__":
    main()
