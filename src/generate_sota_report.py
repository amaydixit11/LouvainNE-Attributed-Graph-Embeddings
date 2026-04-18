#!/usr/bin/env python3
"""
Generate sourced comparison tables for node classification, link prediction,
and runtime using saved repo artifacts plus published leaderboard values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_SOURCES = {
    ("Cora", "node"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-cora-with-public-split.html",
    ("Cora", "link"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-cora.html",
    ("CiteSeer", "node"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-citeseer-full-supervised.html",
    ("CiteSeer", "link"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-citeseer.html",
    ("PubMed", "node"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/node-classification-on-pubmed-full-supervised.html",
    ("PubMed", "link"): "https://opencodepapers-b7572d.gitlab.io/benchmarks/link-prediction-on-pubmed.html",
}

SOTA_NODE_CLASSIFICATION: Dict[str, List[Dict[str, object]]] = {
    "Cora": [
        {"model": "OGC", "accuracy": 0.869, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GCN-TV", "accuracy": 0.863, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GCNII", "accuracy": 0.855, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GRAND", "accuracy": 0.854, "reference": "OpenCodePapers", "training_free": False},
        {"model": "CPF-ind-APPNP", "accuracy": 0.853, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GCN", "accuracy": 0.851, "reference": "OpenCodePapers", "training_free": False},
        {"model": "AIR-GCN", "accuracy": 0.847, "reference": "OpenCodePapers", "training_free": False},
        {"model": "H-GCN", "accuracy": 0.845, "reference": "OpenCodePapers", "training_free": False},
        {"model": "DAGNN", "accuracy": 0.844, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GAT", "accuracy": 0.830, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GraphSAGE", "accuracy": 0.745, "reference": "OpenCodePapers", "training_free": False},
    ],
    "CiteSeer": [
        {"model": "APPNP", "accuracy": 0.742, "reference": "paper baseline", "training_free": False},
        {"model": "GAT", "accuracy": 0.725, "reference": "paper baseline", "training_free": False},
        {"model": "GraphSAGE", "accuracy": 0.708, "reference": "paper baseline", "training_free": False},
        {"model": "GCN", "accuracy": 0.703, "reference": "paper baseline", "training_free": False},
    ],
    "PubMed": [
        {"model": "GraphSAGE+DropEdge", "accuracy": 0.917, "reference": "OpenCodePapers", "training_free": False},
        {"model": "ASGCN", "accuracy": 0.906, "reference": "OpenCodePapers", "training_free": False},
        {"model": "FDGATII", "accuracy": 0.903524, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GCNII", "accuracy": 0.903, "reference": "OpenCodePapers", "training_free": False},
        {"model": "FastGCN", "accuracy": 0.880, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GraphSAGE", "accuracy": 0.871, "reference": "OpenCodePapers", "training_free": False},
    ],
    "BlogCatalog": [
        {"model": "node2vec", "accuracy": 0.336, "reference": "classic baseline", "training_free": True},
        {"model": "DeepWalk", "accuracy": 0.329, "reference": "classic baseline", "training_free": True},
        {"model": "LINE", "accuracy": 0.321, "reference": "classic baseline", "training_free": True},
    ],
}

SOTA_LINK_PREDICTION: Dict[str, List[Dict[str, object]]] = {
    "Cora": [
        {"model": "NESS", "auc": 0.9846, "ap": 0.9871, "reference": "OpenCodePapers", "training_free": False},
        {"model": "WalkPooling", "auc": 0.9590, "ap": 0.9600, "reference": "OpenCodePapers", "training_free": False},
        {"model": "NBFNet", "auc": 0.9560, "ap": 0.9620, "reference": "OpenCodePapers", "training_free": False},
        {"model": "VGNAE", "auc": 0.9560, "ap": 0.9570, "reference": "OpenCodePapers", "training_free": False},
        {"model": "VGAE", "auc": 0.9140, "ap": 0.9230, "reference": "paper baseline", "training_free": False},
    ],
    "CiteSeer": [
        {"model": "NESS", "auc": 0.9943, "ap": 0.9950, "reference": "OpenCodePapers", "training_free": False},
        {"model": "VGNAE", "auc": 0.9700, "ap": 0.9710, "reference": "OpenCodePapers", "training_free": False},
        {"model": "Graph InfoClust", "auc": 0.9700, "ap": 0.9680, "reference": "OpenCodePapers", "training_free": False},
        {"model": "GNAE", "auc": 0.9650, "ap": 0.9700, "reference": "OpenCodePapers", "training_free": False},
        {"model": "VGAE", "auc": 0.8630, "ap": 0.8810, "reference": "paper baseline", "training_free": False},
    ],
    "PubMed": [
        {"model": "NESS", "auc": 0.9810, "ap": 0.9810, "reference": "OpenCodePapers", "training_free": False},
        {"model": "WalkPooling", "auc": 0.9640, "ap": 0.9650, "reference": "OpenCodePapers", "training_free": False},
        {"model": "SEAL", "auc": 0.9680, "ap": 0.9690, "reference": "OpenCodePapers", "training_free": False},
        {"model": "NBFNet", "auc": 0.9580, "ap": 0.9610, "reference": "OpenCodePapers", "training_free": False},
    ],
}


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_results() -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    summary_json = REPO_ROOT / "results" / "benchmark_summary.json"
    payloads = load_json(summary_json) or []
    for payload in payloads:
        dataset_name = payload["dataset"]
        results[dataset_name] = {
            "baseline": payload["results"]["baseline"],
            "improved": payload["results"]["improved"],
            "num_nodes": payload.get("num_nodes", 0),
            "num_edges": payload.get("num_edges_directed", 0),
            "num_features": payload.get("num_features", 0),
            "num_classes": payload.get("num_classes", 0),
            "source": payload.get("source", ""),
        }

    for dataset_name in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        if dataset_name in results:
            continue
        payload = load_json(REPO_ROOT / "results" / dataset_name / "comparison_results.json")
        if payload is None:
            continue
        results[dataset_name] = {
            "baseline": payload["results"]["baseline"],
            "improved": payload["results"]["improved"],
            "num_nodes": payload.get("num_nodes", 0),
            "num_edges": payload.get("num_edges_directed", 0),
            "num_features": payload.get("num_features", 0),
            "num_classes": payload.get("num_classes", 0),
            "source": payload.get("source", ""),
        }

    ogb_summary = load_json(REPO_ROOT / "results" / "ogb_benchmark_summary.json") or []
    for payload in ogb_summary:
        dataset_name = payload["dataset"]
        results[dataset_name] = {
            "baseline": payload["results"]["baseline_structure"],
            "improved": payload["results"]["improved"],
            "num_nodes": payload.get("num_nodes", 0),
            "num_edges": payload.get("num_edges", 0),
            "num_features": payload.get("num_features", 0),
            "num_classes": payload.get("num_classes", 0),
            "source": "",
        }

    return results


def fmt_metric(value: Optional[float], as_percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%" if as_percent else f"{value:.4f}"


def fmt_time(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def best_node_model(dataset_name: str) -> Optional[Dict[str, object]]:
    models = SOTA_NODE_CLASSIFICATION.get(dataset_name, [])
    if not models:
        return None
    return max(models, key=lambda item: float(item["accuracy"]))


def best_link_model(dataset_name: str) -> Optional[Dict[str, object]]:
    models = SOTA_LINK_PREDICTION.get(dataset_name, [])
    if not models:
        return None
    return max(models, key=lambda item: float(item["auc"]))


def build_node_table(results: Dict[str, dict]) -> List[str]:
    lines = [
        "## Node Classification: Large Comparison Table",
        "",
        "| Dataset | Model | Type | Accuracy | Our Improved | Gap vs Ours | Per-Seed Time (s) | External Runtime | Source |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]

    body_lines: List[str] = []
    for dataset_name in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        our = results.get(dataset_name)
        our_acc = our["improved"]["test_micro_f1_mean"] if our else None
        our_time = our["improved"]["per_seed_eval_time_seconds_mean"] if our else None
        source_url = TASK_SOURCES.get((dataset_name, "node"), "N/A")
        for item in SOTA_NODE_CLASSIFICATION.get(dataset_name, []):
            gap = (float(item["accuracy"]) - our_acc) if our_acc is not None else None
            body_lines.append(
                f"| {dataset_name} | {item['model']} | External | "
                f"{fmt_metric(float(item['accuracy']), as_percent=True)} | "
                f"{fmt_metric(our_acc, as_percent=True)} | "
                f"{fmt_metric(gap, as_percent=True) if gap is not None else 'N/A'} | "
                f"{fmt_time(our_time)} | "
                f"N/A | {source_url} |"
            )
        if our:
            baseline = our["baseline"]
            improved = our["improved"]
            body_lines.append(
                f"| {dataset_name} | LouvainNE (structure) | Ours | "
                f"{fmt_metric(baseline['test_micro_f1_mean'], as_percent=True)} | "
                f"{fmt_metric(improved['test_micro_f1_mean'], as_percent=True)} | "
                f"{fmt_metric(baseline['test_micro_f1_mean'] - improved['test_micro_f1_mean'], as_percent=True)} | "
                f"{baseline['per_seed_eval_time_seconds_mean']:.2f} | repo measured | local results |"
            )
            body_lines.append(
                f"| {dataset_name} | LouvainNE (improved) | Ours | "
                f"{fmt_metric(improved['test_micro_f1_mean'], as_percent=True)} | "
                f"{fmt_metric(improved['test_micro_f1_mean'], as_percent=True)} | "
                f"{fmt_metric(0.0, as_percent=True)} | "
                f"{improved['per_seed_eval_time_seconds_mean']:.2f} | repo measured | local results |"
            )
    return lines + body_lines + [""]


def build_link_table(results: Dict[str, dict]) -> List[str]:
    lines = [
        "## Link Prediction: Large Comparison Table",
        "",
        "| Dataset | Model | Type | AUC | AP | Our Improved AUC | Gap vs Ours | Per-Seed Time (s) | Source |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for dataset_name in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        our = results.get(dataset_name)
        our_auc = our["improved"].get("link_auc_mean") if our else None
        our_time = our["improved"]["per_seed_eval_time_seconds_mean"] if our else None
        source_url = TASK_SOURCES.get((dataset_name, "link"), "N/A")
        for item in SOTA_LINK_PREDICTION.get(dataset_name, []):
            gap = (float(item["auc"]) - our_auc) if our_auc is not None else None
            lines.append(
                f"| {dataset_name} | {item['model']} | External | "
                f"{fmt_metric(float(item['auc']))} | "
                f"{fmt_metric(float(item['ap']))} | "
                f"{fmt_metric(our_auc)} | "
                f"{fmt_metric(gap) if gap is not None else 'N/A'} | "
                f"{fmt_time(our_time)} | "
                f"{source_url} |"
            )
        if our:
            baseline = our["baseline"]
            improved = our["improved"]
            lines.append(
                f"| {dataset_name} | LouvainNE (structure) | Ours | "
                f"{fmt_metric(baseline.get('link_auc_mean'))} | "
                f"{fmt_metric(baseline.get('link_ap_mean'))} | "
                f"{fmt_metric(improved.get('link_auc_mean'))} | "
                f"{fmt_metric((baseline.get('link_auc_mean') or 0.0) - (improved.get('link_auc_mean') or 0.0))} | "
                f"{baseline['per_seed_eval_time_seconds_mean']:.2f} | local results |"
            )
            lines.append(
                f"| {dataset_name} | LouvainNE (improved) | Ours | "
                f"{fmt_metric(improved.get('link_auc_mean'))} | "
                f"{fmt_metric(improved.get('link_ap_mean'))} | "
                f"{fmt_metric(improved.get('link_auc_mean'))} | "
                f"{fmt_metric(0.0)} | "
                f"{improved['per_seed_eval_time_seconds_mean']:.2f} | local results |"
            )
    return lines + [""]


def build_best_summary_table(results: Dict[str, dict]) -> List[str]:
    lines = [
        "## Best-Per-Dataset Summary",
        "",
        "| Dataset | Best External Node Model | Node Acc | Our Node Acc | Best External Link Model | Link AUC / AP | Our Link AUC / AP | Our Time (s) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for dataset_name in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        our = results.get(dataset_name)
        node_best = best_node_model(dataset_name)
        link_best = best_link_model(dataset_name)
        our_node = our["improved"]["test_micro_f1_mean"] if our else None
        our_link_auc = our["improved"].get("link_auc_mean") if our else None
        our_link_ap = our["improved"].get("link_ap_mean") if our else None
        our_time = our["improved"]["per_seed_eval_time_seconds_mean"] if our else None
        node_model = node_best["model"] if node_best else "N/A"
        node_score = fmt_metric(float(node_best["accuracy"]), as_percent=True) if node_best else "N/A"
        link_model = link_best["model"] if link_best else "N/A"
        link_score = (
            f"{fmt_metric(float(link_best['auc']))} / {fmt_metric(float(link_best['ap']))}"
            if link_best else "N/A"
        )
        our_link_score = (
            f"{fmt_metric(our_link_auc)} / {fmt_metric(our_link_ap)}"
            if our_link_auc is not None and our_link_ap is not None else "N/A"
        )
        lines.append(
            f"| {dataset_name} | {node_model} | {node_score} | {fmt_metric(our_node, as_percent=True)} | "
            f"{link_model} | {link_score} | {our_link_score} | "
            f"{fmt_time(our_time)} |"
        )
    return lines + [""]


def build_runtime_notes(results: Dict[str, dict]) -> List[str]:
    lines = [
        "## Runtime Notes",
        "",
        "| Dataset | Our Structure Time (s) | Our Improved Time (s) | External Runtime Availability |",
        "|---|---:|---:|---|",
    ]
    for dataset_name in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        our = results.get(dataset_name)
        if not our:
            continue
        lines.append(
            f"| {dataset_name} | {our['baseline']['per_seed_eval_time_seconds_mean']:.2f} | "
            f"{our['improved']['per_seed_eval_time_seconds_mean']:.2f} | "
            "Not consistently reported on sourced leaderboard pages |"
        )
    return lines + [""]


def generate_comprehensive_report(output_path: Path) -> None:
    results = load_benchmark_results()
    lines = [
        "# Comprehensive Benchmark Report: LouvainNE vs SOTA",
        "",
        "## Scope",
        "",
        "This report builds large comparison tables for node classification, link prediction, and runtime.",
        "External accuracy metrics are sourced from benchmark pages and paper-reported baselines.",
        "Our metrics and timings come from saved repo artifacts under `results/`.",
        "",
        "## Protocol Notes",
        "",
        "- External benchmark pages do not provide fully standardized runtime numbers, so external runtime cells are left as `N/A` unless directly available.",
        "- Our link prediction numbers come from the repo's train/val/test edge split protocol.",
        "- OpenCodePapers leaderboard protocols may differ from our preprocessing or split details.",
        "- `BlogCatalog` has strong local results in this repo, but external sourced node/link leaderboards are sparse.",
        "",
    ]
    lines.extend(build_best_summary_table(results))
    lines.extend(build_node_table(results))
    lines.extend(build_link_table(results))
    lines.extend(build_runtime_notes(results))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comprehensive report written to {output_path}")


def main() -> None:
    report_path = REPO_ROOT / "results" / "comprehensive_benchmark_report.md"
    generate_comprehensive_report(report_path)


if __name__ == "__main__":
    main()
