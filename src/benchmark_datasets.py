#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
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
    fuse_repo_edges,
    load_processed_graph,
    normalize_rows,
    unique_undirected_edges,
    write_comparison_plot,
)

DEFAULT_DATASETS = ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]
DEFAULT_PENALTIES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
EVAL_SEEDS = [1548, 1549, 1550, 1551, 1552]
GRAPH_PROJECTION_DIM = 64
RESIDUAL_DIM = 128
BASELINE_TOP_K = 50
BASELINE_ALPHA = 0.2
IMPROVED_TOP_K = 15
IMPROVED_MIN_SIMILARITY = 0.2
BLOCK_SIZE = 512


@dataclass
class DatasetBundle:
    name: str
    source: str
    data: GraphData
    evaluation_axis: str
    unit_count: int


def canonical_name(name: str) -> str:
    lookup = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "blogcatalog": "BlogCatalog",
    }
    try:
        return lookup[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


def load_dataset(name: str) -> DatasetBundle:
    name = canonical_name(name)
    if name in {"Cora", "CiteSeer", "PubMed"}:
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
            evaluation_axis="classifier_seed",
            unit_count=len(EVAL_SEEDS),
        )

    if name == "BlogCatalog":
        path = REPO_ROOT / "data" / "BlogCatalog" / "processed" / "data.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing BlogCatalog at {path}. Run `python prepare_datasets.py` first."
            )
        data = load_processed_graph(path)
        if data.train_mask.ndim != 2:
            raise ValueError("Expected BlogCatalog to contain precomputed multi-split masks.")
        return DatasetBundle(
            name=name,
            source=str(path.relative_to(REPO_ROOT)),
            data=data,
            evaluation_axis="prepared_split",
            unit_count=int(data.train_mask.shape[1]),
        )

    raise ValueError(f"Unsupported dataset: {name}")


def select_split(data: GraphData, split_idx: int) -> GraphData:
    return GraphData(
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask[:, split_idx].to(torch.bool),
        val_mask=data.val_mask[:, split_idx].to(torch.bool),
        test_mask=data.test_mask[:, split_idx].to(torch.bool),
    )


def build_eval_units(bundle: DatasetBundle) -> List[Tuple[str, GraphData, int]]:
    if bundle.data.train_mask.ndim == 1:
        return [
            (f"seed_{seed}", bundle.data, seed)
            for seed in EVAL_SEEDS[: bundle.unit_count]
        ]
    return [
        (f"split_{split_idx}", select_split(bundle.data, split_idx), 3000 + split_idx)
        for split_idx in range(bundle.unit_count)
    ]


def summarize_runs(name: str, config: Dict[str, object], runs: List[Dict[str, float]]) -> Dict[str, object]:
    result: Dict[str, object] = {"name": name, "config": config, "runs": runs}
    for key in [
        "val_micro_f1",
        "val_macro_f1",
        "test_micro_f1",
        "test_macro_f1",
        "selection_score",
        "embedding_time_seconds",
        "pipeline_time_seconds",
    ]:
        values = [run[key] for run in runs]
        result[f"{key}_mean"] = float(sum(values) / len(values))
        # Use Bessel's correction (ddof=1) for unbiased sample std estimate
        std = (sum((value - result[f"{key}_mean"]) ** 2 for value in values) / max(len(values) - 1, 1)) ** 0.5
        result[f"{key}_std"] = float(std)
    return result


def attach_timing_summary(
    result: Dict[str, object],
    runs: Sequence[Dict[str, float]],
    setup_time_seconds: float,
) -> Dict[str, object]:
    result["setup_time_seconds"] = float(setup_time_seconds)
    for key in ["embedding_time_seconds", "classifier_time_seconds", "per_seed_eval_time_seconds"]:
        values = [run[key] for run in runs]
        result[f"{key}_mean"] = float(sum(values) / len(values))
        # Use Bessel's correction (ddof=1) for unbiased sample std estimate
        std = (sum((value - result[f"{key}_mean"]) ** 2 for value in values) / max(len(values) - 1, 1)) ** 0.5
        result[f"{key}_std"] = float(std)
    result["pipeline_time_seconds_mean"] = result["per_seed_eval_time_seconds_mean"]
    result["pipeline_time_seconds_std"] = result["per_seed_eval_time_seconds_std"]
    return result


def evaluate_units(
    name: str,
    config: Dict[str, object],
    units: Sequence[Tuple[str, GraphData, int]],
    penalties: Sequence[float],
    embedding_fn,
    setup_time_seconds: float = 0.0,
) -> Dict[str, object]:
    runs = []
    for unit_name, unit_data, seed in units:
        start = time.perf_counter()
        embeddings = embedding_fn(seed).to(torch.float32)
        embedding_elapsed = time.perf_counter() - start
        metrics = fit_linear_probe(embeddings, unit_data, penalties, seed)
        per_seed_elapsed = time.perf_counter() - start
        metrics["unit"] = unit_name
        metrics["seed"] = float(seed)
        metrics["embedding_time_seconds"] = float(embedding_elapsed)
        metrics["classifier_time_seconds"] = float(per_seed_elapsed - embedding_elapsed)
        metrics["per_seed_eval_time_seconds"] = float(per_seed_elapsed)
        metrics["pipeline_time_seconds"] = float(per_seed_elapsed)
        runs.append(metrics)
    result = summarize_runs(name, config, runs)
    return attach_timing_summary(result, runs, setup_time_seconds)


def low_rank_projection(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    centered = x.to(torch.float32) - x.to(torch.float32).mean(dim=0, keepdim=True)
    q = min(max(out_dim + 8, out_dim), min(centered.shape))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    return centered @ v[:, :out_dim]


def build_blockwise_topk_predictions(
    features: torch.Tensor,
    top_k: int,
    mutual: bool,
    min_similarity: float,
    block_size: int = BLOCK_SIZE,
) -> Dict[Tuple[int, int], float]:
    normalized = normalize_rows(features)
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
        return normalize_rows(embeddings)

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

    return normalize_rows((1.0 - gamma) * embeddings + gamma * refined)


def build_paths(dataset_name: str) -> Dict[str, Path]:
    base = REPO_ROOT / "results" / dataset_name
    return {
        "base": base,
        "json": base / "comparison_results.json",
        "plot": base / "comparison_plot.png",
    }


def benchmark_dataset(dataset_name: str, embedding_dim: int) -> Dict[str, object]:
    bundle = load_dataset(dataset_name)
    data = bundle.data
    units = build_eval_units(bundle)
    penalties = DEFAULT_PENALTIES

    runner = LouvainNERunner(REPO_ROOT, data.num_nodes, embedding_dim)
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))

    projection_dim = min(RESIDUAL_DIM, data.x.shape[0] - 1, data.x.shape[1])
    projection_start = time.perf_counter()
    projected = low_rank_projection(data.x, projection_dim)
    projection_elapsed = time.perf_counter() - projection_start

    graph_dim = min(GRAPH_PROJECTION_DIM, projected.shape[1])
    graph_features = normalize_rows(projected[:, :graph_dim])
    feature_embedding = normalize_rows(projected[:, :projection_dim])

    baseline_setup_start = time.perf_counter()
    baseline_predictions = build_blockwise_topk_predictions(
        graph_features,
        top_k=BASELINE_TOP_K,
        mutual=False,
        min_similarity=BASELINE_ALPHA,
    )
    baseline_edges = fuse_repo_edges(structure_edges, baseline_predictions, "method2")
    baseline_setup_time_seconds = projection_elapsed + (time.perf_counter() - baseline_setup_start)

    improved_setup_start = time.perf_counter()
    improved_predictions = build_blockwise_topk_predictions(
        graph_features,
        top_k=IMPROVED_TOP_K,
        mutual=True,
        min_similarity=IMPROVED_MIN_SIMILARITY,
    )
    improved_edges = fuse_adaptive_edges(structure_edges, improved_predictions, 1.0, 0.75)
    improved_setup_time_seconds = projection_elapsed + (time.perf_counter() - improved_setup_start)

    baseline = evaluate_units(
        "projected_threshold_method2_baseline",
        {
            "dataset": bundle.name,
            "graph_projection_dim": graph_dim,
            "baseline_top_k": BASELINE_TOP_K,
            "baseline_alpha": BASELINE_ALPHA,
            "evaluation_axis": bundle.evaluation_axis,
        },
        units,
        penalties,
        lambda seed: runner.embed(baseline_edges, seed),
        setup_time_seconds=baseline_setup_time_seconds,
    )

    def improved_embedding(seed: int) -> torch.Tensor:
        graph_embedding = normalize_rows(runner.embed(improved_edges, seed))
        graph_embedding = apply_sparse_attention(
            graph_embedding,
            improved_predictions,
            gamma=0.5,
            temperature=1.0,
        )
        return concat_features([graph_embedding, feature_embedding])

    improved = evaluate_units(
        "projected_mutual_topk_improved",
        {
            "dataset": bundle.name,
            "graph_projection_dim": graph_dim,
            "graph_top_k": IMPROVED_TOP_K,
            "graph_mutual": True,
            "graph_min_similarity": IMPROVED_MIN_SIMILARITY,
            "graph_overlap_scale": 1.0,
            "graph_new_scale": 0.75,
            "attention_gamma": 0.5,
            "attention_temperature": 1.0,
            "feature_dim": projection_dim,
            "evaluation_axis": bundle.evaluation_axis,
        },
        units,
        penalties,
        improved_embedding,
        setup_time_seconds=improved_setup_time_seconds,
    )

    paths = build_paths(bundle.name)
    paths["base"].mkdir(parents=True, exist_ok=True)
    write_comparison_plot(paths["plot"], baseline, improved)

    payload = {
        "dataset": bundle.name,
        "source": bundle.source,
        "evaluation_axis": bundle.evaluation_axis,
        "unit_count": bundle.unit_count,
        "num_nodes": data.num_nodes,
        "num_edges_directed": int(data.edge_index.shape[1]),
        "num_classes": data.num_classes,
        "num_features": int(data.x.shape[1]),
        "graph_builder": {
            "projection_dim": projection_dim,
            "graph_projection_dim": graph_dim,
            "baseline_top_k": BASELINE_TOP_K,
            "baseline_alpha": BASELINE_ALPHA,
            "improved_top_k": IMPROVED_TOP_K,
            "improved_min_similarity": IMPROVED_MIN_SIMILARITY,
            "block_size": BLOCK_SIZE,
        },
        "graph_stats": {
            "structure_edges_undirected": len(structure_edges),
            "baseline_predicted_edges": len(baseline_predictions),
            "baseline_final_edges": len(baseline_edges),
            "improved_predicted_edges": len(improved_predictions),
            "improved_final_edges": len(improved_edges),
        },
        "results": {
            "baseline": baseline,
            "improved": improved,
        },
        "artifacts": {
            "plot": str(paths["plot"].relative_to(REPO_ROOT)),
        },
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_summary(summary_path: Path, markdown_path: Path, plot_path: Path, dataset_payloads: Sequence[Dict[str, object]]) -> None:
    summary_path.write_text(json.dumps(dataset_payloads, indent=2), encoding="utf-8")

    lines = [
        "# Multi-Dataset Benchmark Summary",
        "",
        "| Dataset | Eval Axis | Baseline Micro-F1 | Improved Micro-F1 | Baseline Setup (s) | Improved Setup (s) | Baseline Per-Seed (s) | Improved Per-Seed (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in dataset_payloads:
        base = payload["results"]["baseline"]
        improved = payload["results"]["improved"]
        lines.append(
            f"| {payload['dataset']} | {payload['evaluation_axis']} | "
            f"{base['test_micro_f1_mean']:.4f} ± {base['test_micro_f1_std']:.4f} | "
            f"{improved['test_micro_f1_mean']:.4f} ± {improved['test_micro_f1_std']:.4f} | "
            f"{base['setup_time_seconds']:.2f} | "
            f"{improved['setup_time_seconds']:.2f} | "
            f"{base['per_seed_eval_time_seconds_mean']:.2f} ± {base['per_seed_eval_time_seconds_std']:.2f} | "
            f"{improved['per_seed_eval_time_seconds_mean']:.2f} ± {improved['per_seed_eval_time_seconds_std']:.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    datasets = [payload["dataset"] for payload in dataset_payloads]
    baseline_micro = [payload["results"]["baseline"]["test_micro_f1_mean"] for payload in dataset_payloads]
    improved_micro = [payload["results"]["improved"]["test_micro_f1_mean"] for payload in dataset_payloads]
    baseline_setup = [payload["results"]["baseline"]["setup_time_seconds"] for payload in dataset_payloads]
    improved_setup = [payload["results"]["improved"]["setup_time_seconds"] for payload in dataset_payloads]
    baseline_time = [payload["results"]["baseline"]["per_seed_eval_time_seconds_mean"] for payload in dataset_payloads]
    improved_time = [payload["results"]["improved"]["per_seed_eval_time_seconds_mean"] for payload in dataset_payloads]

    x = range(len(datasets))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar([i - width / 2 for i in x], baseline_micro, width, label="Baseline", color="#2A6F97")
    axes[0].bar([i + width / 2 for i in x], improved_micro, width, label="Improved", color="#E07A5F")
    axes[0].set_xticks(list(x), datasets)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Micro-F1")
    axes[0].set_title("Accuracy Across Datasets")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar([i - width / 2 for i in x], baseline_setup, width, label="Baseline", color="#6C9A8B")
    axes[1].bar([i + width / 2 for i in x], improved_setup, width, label="Improved", color="#C8553D")
    axes[1].set_xticks(list(x), datasets)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Setup Time Across Datasets")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].bar([i - width / 2 for i in x], baseline_time, width, label="Baseline", color="#7D4E9E")
    axes[2].bar([i + width / 2 for i in x], improved_time, width, label="Improved", color="#D1495B")
    axes[2].set_xticks(list(x), datasets)
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Per-Seed Evaluation Time")
    axes[2].legend()
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("LouvainNE Benchmark Across Prepared Attributed Datasets", fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the maintained baseline and improved pipeline on prepared datasets.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args()

    dataset_payloads = []
    for dataset_name in args.datasets:
        print(f"Running {dataset_name}...", flush=True)
        payload = benchmark_dataset(dataset_name, args.embedding_dim)
        dataset_payloads.append(payload)
        base = payload["results"]["baseline"]
        improved = payload["results"]["improved"]
        print(
            payload["dataset"],
            f"baseline_micro={base['test_micro_f1_mean']:.4f}",
            f"improved_micro={improved['test_micro_f1_mean']:.4f}",
            f"baseline_setup={base['setup_time_seconds']:.2f}s",
            f"improved_setup={improved['setup_time_seconds']:.2f}s",
            f"baseline_per_seed={base['per_seed_eval_time_seconds_mean']:.2f}s",
            f"improved_per_seed={improved['per_seed_eval_time_seconds_mean']:.2f}s",
            flush=True,
        )

    summary_json = REPO_ROOT / "results" / "benchmark_summary.json"
    summary_md = REPO_ROOT / "results" / "benchmark_summary.md"
    summary_plot = REPO_ROOT / "results" / "benchmark_summary.png"
    write_summary(summary_json, summary_md, summary_plot, dataset_payloads)
    print(f"Summary written to {summary_json}")
    print(f"Summary markdown written to {summary_md}")
    print(f"Summary plot written to {summary_plot}")


if __name__ == "__main__":
    main()
