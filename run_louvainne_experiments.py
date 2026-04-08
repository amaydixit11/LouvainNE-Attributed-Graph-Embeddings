#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_processed_graph(path: Path) -> GraphData:
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict):
        data_dict = payload
    elif isinstance(payload, tuple) and isinstance(payload[0], dict):
        data_dict = payload[0]
    elif isinstance(payload, tuple) and hasattr(payload[0], "x"):
        data_dict = {
            "x": payload[0].x,
            "edge_index": payload[0].edge_index,
            "y": payload[0].y,
            "train_mask": payload[0].train_mask,
            "val_mask": payload[0].val_mask,
            "test_mask": payload[0].test_mask,
        }
    else:
        raise ValueError(f"Unsupported processed Cora payload at {path}")
    return GraphData(
        x=data_dict["x"].to(torch.float32).cpu(),
        edge_index=data_dict["edge_index"].to(torch.long).cpu(),
        y=data_dict["y"].to(torch.long).cpu(),
        train_mask=data_dict["train_mask"].to(torch.bool).cpu(),
        val_mask=data_dict["val_mask"].to(torch.bool).cpu(),
        test_mask=data_dict["test_mask"].to(torch.bool).cpu(),
    )


def load_raw_graph(raw_dir: Path) -> GraphData:
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph"]
    objects = []
    for name in names:
        with (raw_dir / f"ind.cora.{name}").open("rb") as handle:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                objects.append(pickle.load(handle, encoding="latin1"))
    x, tx, allx, y, ty, ally, graph = objects

    test_idx_reorder = np.loadtxt(raw_dir / "ind.cora.test.index", dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    edges: List[Tuple[int, int]] = []
    for src, dsts in graph.items():
        for dst in dsts:
            edges.append((int(src), int(dst)))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = features.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[: y.shape[0]] = True
    val_mask[y.shape[0] : y.shape[0] + 500] = True
    test_mask[torch.from_numpy(test_idx_range)] = True

    return GraphData(
        x=torch.from_numpy(features.toarray()).to(torch.float32),
        edge_index=edge_index,
        y=torch.from_numpy(labels).to(torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_cora_graph() -> Tuple[GraphData, str]:
    repo_processed = REPO_ROOT / "data/Planetoid/Cora/processed/data.pt"
    if repo_processed.is_file():
        return load_processed_graph(repo_processed), str(repo_processed)

    raw_candidates = [REPO_ROOT / "data/Planetoid/Cora/raw"]
    required = {
        "ind.cora.x",
        "ind.cora.tx",
        "ind.cora.allx",
        "ind.cora.y",
        "ind.cora.ty",
        "ind.cora.ally",
        "ind.cora.graph",
        "ind.cora.test.index",
    }
    for candidate in raw_candidates:
        if candidate.exists() and required.issubset({path.name for path in candidate.iterdir()}):
            return load_raw_graph(candidate), str(candidate)

    raise FileNotFoundError(
        "Could not locate Cora under data/Planetoid/Cora. Run `python prepare_datasets.py` first."
    )


def unique_undirected_edges(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    seen = set()
    edges: List[Tuple[int, int]] = []
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b))
    edges.sort()
    return edges


def edges_to_dict(edges: Iterable[Tuple[int, int]], weight: float = 1.0) -> Dict[Tuple[int, int], float]:
    return {edge: float(weight) for edge in edges}


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def repo_similarity(x: torch.Tensor) -> torch.Tensor:
    scores = x @ x.t()
    scores.fill_diagonal_(0.0)
    max_score = float(scores.max().item())
    if max_score > 0.0:
        scores = scores / max_score
    return scores


def cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    normed = normalize_rows(x)
    scores = normed @ normed.t()
    scores.fill_diagonal_(0.0)
    return scores


def build_threshold_predictions(similarity: torch.Tensor, alpha: float) -> Dict[Tuple[int, int], float]:
    mask = torch.triu(similarity > alpha, diagonal=1)
    indices = mask.nonzero(as_tuple=False)
    return {
        (int(i.item()), int(j.item())): float(similarity[i, j].item())
        for i, j in indices
    }


def build_topk_predictions(
    similarity: torch.Tensor,
    top_k: int,
    mutual: bool,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    n = similarity.size(0)
    top_k = max(1, min(top_k, n - 1))
    masked = similarity.clone()
    masked.fill_diagonal_(-1.0)
    _, top_idx = torch.topk(masked, k=top_k, dim=1)
    neighbor_mask = torch.zeros_like(masked, dtype=torch.bool)
    neighbor_mask.scatter_(1, top_idx, True)
    if mutual:
        neighbor_mask = neighbor_mask & neighbor_mask.t()
    else:
        neighbor_mask = neighbor_mask | neighbor_mask.t()
    neighbor_mask &= similarity >= min_similarity
    indices = torch.triu(neighbor_mask, diagonal=1).nonzero(as_tuple=False)
    return {
        (int(i.item()), int(j.item())): float(similarity[i, j].item())
        for i, j in indices
    }


def fuse_repo_edges(
    structure_edges: Dict[Tuple[int, int], float],
    predicted_edges: Dict[Tuple[int, int], float],
    mode: str,
) -> Dict[Tuple[int, int], float]:
    fused = dict(structure_edges)
    for edge, sim in predicted_edges.items():
        if mode == "repo_unweighted":
            fused[edge] = 1.0
        elif mode == "method2":
            fused[edge] = 1.0 + sim if edge in fused else sim
        elif mode == "method3":
            fused[edge] = 1.0 + sim if edge in fused else 1.0
        elif mode == "method4":
            fused[edge] = 2.0 if edge in fused else 1.0
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
    return fused


def fuse_adaptive_edges(
    structure_edges: Dict[Tuple[int, int], float],
    predicted_edges: Dict[Tuple[int, int], float],
    overlap_scale: float,
    new_scale: float,
) -> Dict[Tuple[int, int], float]:
    fused = dict(structure_edges)
    for edge, sim in predicted_edges.items():
        if edge in fused:
            fused[edge] = 1.0 + overlap_scale * sim
        else:
            fused[edge] = new_scale * sim
    return fused


def concat_features(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=1)


def micro_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    micro = float((y_true == y_pred).float().mean().item())
    f1s = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum().item())
        fp = int(((y_true != cls) & (y_pred == cls)).sum().item())
        fn = int(((y_true == cls) & (y_pred != cls)).sum().item())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    macro = float(sum(f1s) / len(f1s))
    return micro, macro


def create_link_prediction_split(
    edge_index: torch.Tensor,
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edges into train/val/test for link prediction on undirected graphs.
    
    IMPORTANT: This function handles undirected graphs where each edge appears
    as both (u,v) and (v,u). It canonicalizes edges to avoid leakage.
    
    Returns:
        train_pos_edges, val_pos_edges, test_pos_edges,
        train_edge_index, val_neg_edges, test_neg_edges
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Canonicalize: keep only (min(u,v), max(u,v)) pairs
    edge_list = edge_index.t().tolist()
    canonical_edges = set()
    for u, v in edge_list:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        canonical_edges.add((a, b))
    
    canonical_list = sorted(list(canonical_edges))
    num_canonical = len(canonical_list)
    
    # Shuffle canonical edges
    perm = torch.randperm(num_canonical, generator=generator)
    canonical_list = [canonical_list[i] for i in perm.tolist()]
    
    num_test = max(1, int(num_canonical * test_ratio))
    num_val = max(1, int(num_canonical * val_ratio))
    
    # Split canonical edges
    test_canonical = canonical_list[:num_test]
    val_canonical = canonical_list[num_test:num_test + num_val]
    train_canonical = canonical_list[num_test + num_val:]
    
    # Helper to expand canonical to directed
    def to_directed(canonical_pairs):
        directed = []
        for u, v in canonical_pairs:
            directed.append((u, v))
            directed.append((v, u))
        return torch.tensor(directed, dtype=torch.long).t() if directed else torch.empty((2, 0), dtype=torch.long)
    
    # Expand back to directed edges for train
    train_edges_directed = []
    for u, v in train_canonical:
        train_edges_directed.append((u, v))
        train_edges_directed.append((v, u))
    train_edge_index = torch.tensor(train_edges_directed, dtype=torch.long).t().contiguous() if train_edges_directed else edge_index
    train_pos_edges = to_directed(train_canonical)
    
    # Positive edges for evaluation (directed form)
    val_pos_edges = to_directed(val_canonical)
    test_pos_edges = to_directed(test_canonical)
    
    # Build adjacency set from ALL original edges (for negative sampling)
    adj_set = set()
    for u, v in edge_list:
        adj_set.add((u, v))
        adj_set.add((v, u))
    
    num_nodes = int(edge_index.max()) + 1
    
    def sample_negative_edges(num_neg: int, generator: torch.Generator) -> torch.Tensor:
        neg_edges = []
        attempts = 0
        max_attempts = num_neg * 100
        while len(neg_edges) < num_neg and attempts < max_attempts:
            u = torch.randint(0, num_nodes, (1,), generator=generator).item()
            v = torch.randint(0, num_nodes, (1,), generator=generator).item()
            attempts += 1
            if u != v and (u, v) not in adj_set:
                neg_edges.append((u, v))
        if len(neg_edges) < num_neg:
            while len(neg_edges) < num_neg:
                u = torch.randint(0, num_nodes, (1,), generator=generator).item()
                v = torch.randint(0, num_nodes, (1,), generator=generator).item()
                if u != v:
                    neg_edges.append((u, v))
        return torch.tensor(neg_edges, dtype=torch.long).t()
    
    val_neg_edges = sample_negative_edges(len(val_canonical), generator)
    test_neg_edges = sample_negative_edges(len(test_canonical), generator)
    
    return train_pos_edges, val_pos_edges, test_pos_edges, train_edge_index, val_neg_edges, test_neg_edges


def compute_link_prediction_metrics(
    embeddings: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
) -> Dict[str, float]:
    """Compute AUC and AP for link prediction using dot product scoring."""
    pos_scores = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1)
    neg_scores = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=1)
    
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    
    sorted_indices = torch.argsort(all_scores, descending=True)
    sorted_labels = all_labels[sorted_indices]
    
    num_pos = int(pos_scores.shape[0])
    num_neg = int(neg_scores.shape[0])
    
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (num_pos + 1e-10)
    
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    
    ap = float(torch.sum((recall[1:] - recall[:-1]) * precision[1:]).item())
    
    tpr = tp / (num_pos + 1e-10)
    fpr = fp / (num_neg + 1e-10)
    
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    
    sorted_fpr, sort_idx = torch.sort(fpr)
    sorted_tpr = tpr[sort_idx]
    
    auc = float(torch.trapz(sorted_tpr, sorted_fpr).item())
    auc = max(0.0, min(1.0, auc))
    
    return {"link_auc": auc, "link_ap": ap}


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_linear_probe(
    embeddings: torch.Tensor,
    data: GraphData,
    penalties: Sequence[float],
    seed: int,
) -> Dict[str, float]:
    train_x = embeddings[data.train_mask]
    train_y = data.y[data.train_mask]
    val_x = embeddings[data.val_mask]
    val_y = data.y[data.val_mask]
    test_x = embeddings[data.test_mask]
    test_y = data.y[data.test_mask]

    best = None
    for penalty in penalties:
        set_global_seed(seed)
        model = LinearProbe(embeddings.size(1), data.num_classes)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            max_iter=200,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = model(train_x)
            loss = F.cross_entropy(logits, train_y)
            if penalty > 0.0:
                loss = loss + 0.5 * penalty * (
                    model.linear.weight.square().sum() + model.linear.bias.square().sum()
                )
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            val_pred = model(val_x).argmax(dim=1)
            test_pred = model(test_x).argmax(dim=1)
        val_micro, val_macro = micro_macro_f1(val_y, val_pred, data.num_classes)
        test_micro, test_macro = micro_macro_f1(test_y, test_pred, data.num_classes)
        candidate = {
            "penalty": float(penalty),
            "val_micro_f1": val_micro,
            "val_macro_f1": val_macro,
            "test_micro_f1": test_micro,
            "test_macro_f1": test_macro,
            "selection_score": 0.5 * (val_micro + val_macro),
        }
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    assert best is not None
    return best


class LouvainNERunner:
    def __init__(self, repo_root: Path, num_nodes: int, embedding_dim: int, damping: float = 0.01) -> None:
        self.repo_root = repo_root
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.damping = damping
        self.build_dir = repo_root / "build" / "louvainne"
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.recpart_bin = self.build_dir / "recpart"
        self.hi2vec_bin = self.build_dir / "hi2vec"
        self._compile_if_needed()

    def _compile_if_needed(self) -> None:
        sources = [
            self.repo_root / "LouvainNE" / "partition.c",
            self.repo_root / "LouvainNE" / "recpart.c",
            self.repo_root / "LouvainNE" / "hi2vec.c",
        ]
        targets = [self.recpart_bin, self.hi2vec_bin]
        if all(target.exists() for target in targets):
            newest_source = max(source.stat().st_mtime for source in sources)
            oldest_target = min(target.stat().st_mtime for target in targets)
            if oldest_target >= newest_source:
                return
        partition_obj = self.build_dir / "partition.o"
        recpart_obj = self.build_dir / "recpart.o"
        commands = [
            ["gcc", "-O3", "-c", str(self.repo_root / "LouvainNE" / "partition.c"), "-o", str(partition_obj)],
            ["gcc", "-O3", "-c", str(self.repo_root / "LouvainNE" / "recpart.c"), "-o", str(recpart_obj)],
            ["gcc", "-O3", str(partition_obj), str(recpart_obj), "-o", str(self.recpart_bin)],
            ["gcc", "-O3", str(self.repo_root / "LouvainNE" / "hi2vec.c"), "-o", str(self.hi2vec_bin), "-lm"],
        ]
        for command in commands:
            subprocess.run(command, cwd=self.repo_root, check=True)

    def embed(self, edge_weights: Dict[Tuple[int, int], float], seed: int) -> torch.Tensor:
        with tempfile.TemporaryDirectory(prefix="louvainne_", dir=self.repo_root / "build") as tmp_dir:
            tmp_path = Path(tmp_dir)
            edge_path = tmp_path / "edges.txt"
            hierarchy_path = tmp_path / "hierarchy.txt"
            vectors_path = tmp_path / "vectors.txt"
            with edge_path.open("w", encoding="utf-8") as handle:
                for (u, v), weight in sorted(edge_weights.items()):
                    handle.write(f"{u} {v} {weight:.12f}\n")
            subprocess.run(
                [str(self.recpart_bin), str(edge_path), str(hierarchy_path), "1", str(seed)],
                cwd=self.repo_root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    str(self.hi2vec_bin),
                    str(self.embedding_dim),
                    str(self.damping),
                    str(hierarchy_path),
                    str(vectors_path),
                    str(seed),
                ],
                cwd=self.repo_root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raw = np.loadtxt(vectors_path)
            if raw.ndim == 1:
                raw = raw[None, :]
            node_ids = raw[:, 0].astype(np.int64)
            embeddings = np.zeros((self.num_nodes, raw.shape[1] - 1), dtype=np.float32)
            embeddings[node_ids] = raw[:, 1:].astype(np.float32)
            return torch.from_numpy(embeddings)


def procrustes_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(source_centered.t() @ target_centered, full_matrices=False)
    rotation = u @ vh
    return source_centered @ rotation + target.mean(dim=0, keepdim=True)


def aligned_ensemble(
    runner: LouvainNERunner,
    edge_weights: Dict[Tuple[int, int], float],
    seeds: Sequence[int],
) -> torch.Tensor:
    embeddings = [normalize_rows(runner.embed(edge_weights, seed)) for seed in seeds]
    reference = embeddings[0]
    aligned = [reference]
    for embedding in embeddings[1:]:
        aligned.append(procrustes_align(embedding, reference))
    return torch.stack(aligned, dim=0).mean(dim=0)


def repo_dense_attention(features: torch.Tensor) -> torch.Tensor:
    scores = (features @ features.t()) / math.sqrt(features.size(1))
    mask = torch.eye(scores.size(0), dtype=torch.bool)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(mask, 0.0)
    attn.fill_diagonal_(1.0)
    return attn


def sparse_attention_from_edges(
    num_nodes: int,
    edge_weights: Dict[Tuple[int, int], float],
    similarity: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scores = torch.full((num_nodes, num_nodes), float("-inf"), dtype=torch.float32)
    for (u, v) in edge_weights:
        sim = float(similarity[u, v].item())
        scores[u, v] = sim / temperature
        scores[v, u] = sim / temperature
    scores.fill_diagonal_(1.0 / temperature)
    return torch.softmax(scores, dim=-1)


def summarize_runs(name: str, config: Dict[str, object], runs: List[Dict[str, float]]) -> Dict[str, object]:
    result: Dict[str, object] = {"name": name, "config": config, "runs": runs}
    for key in [
        "val_micro_f1",
        "val_macro_f1",
        "test_micro_f1",
        "test_macro_f1",
        "selection_score",
    ]:
        values = [run[key] for run in runs]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def attach_timing_summary(
    result: Dict[str, object],
    runs: Sequence[Dict[str, float]],
    setup_time_seconds: float,
) -> Dict[str, object]:
    result["setup_time_seconds"] = float(setup_time_seconds)
    for key in ["embedding_time_seconds", "classifier_time_seconds", "per_seed_eval_time_seconds"]:
        values = [run[key] for run in runs]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    # Keep the previous field name as a compatibility alias for existing consumers.
    result["pipeline_time_seconds_mean"] = result["per_seed_eval_time_seconds_mean"]
    result["pipeline_time_seconds_std"] = result["per_seed_eval_time_seconds_std"]
    return result


def evaluate_embeddings(
    name: str,
    config: Dict[str, object],
    embeddings_per_seed: Sequence[Tuple[int, torch.Tensor]],
    data: GraphData,
    penalties: Sequence[float],
) -> Dict[str, object]:
    runs = []
    for seed, embeddings in embeddings_per_seed:
        metrics = fit_linear_probe(embeddings.to(torch.float32), data, penalties, seed)
        metrics["seed"] = float(seed)
        runs.append(metrics)
    return summarize_runs(name, config, runs)


def evaluate_method_with_timing(
    name: str,
    config: Dict[str, object],
    seeds: Sequence[int],
    data: GraphData,
    penalties: Sequence[float],
    embedding_fn: Callable[[int], torch.Tensor],
    setup_time_seconds: float = 0.0,
) -> Dict[str, object]:
    runs = []
    for seed in seeds:
        pipeline_start = time.perf_counter()
        embeddings = embedding_fn(seed).to(torch.float32)
        embedding_elapsed = time.perf_counter() - pipeline_start
        metrics = fit_linear_probe(embeddings, data, penalties, seed)
        per_seed_elapsed = time.perf_counter() - pipeline_start
        metrics["seed"] = float(seed)
        metrics["embedding_time_seconds"] = float(embedding_elapsed)
        metrics["classifier_time_seconds"] = float(per_seed_elapsed - embedding_elapsed)
        metrics["per_seed_eval_time_seconds"] = float(per_seed_elapsed)
        metrics["pipeline_time_seconds"] = float(per_seed_elapsed)
        runs.append(metrics)

    result = summarize_runs(name, config, runs)
    return attach_timing_summary(result, runs, setup_time_seconds)


def feature_svd(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    projected = u[:, :out_dim] * s[:out_dim]
    return normalize_rows(projected.to(torch.float32))


def low_rank_projection(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Project features to lower dimension using PCA."""
    centered = x.to(torch.float32) - x.to(torch.float32).mean(dim=0, keepdim=True)
    q = min(max(out_dim + 8, out_dim), min(centered.shape))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    return centered @ v[:, :out_dim]


def build_blockwise_topk_predictions(
    features: torch.Tensor,
    top_k: int,
    mutual: bool,
    min_similarity: float,
    block_size: int = 512,
) -> Dict[Tuple[int, int], float]:
    """Build top-k predictions using blockwise cosine similarity computation."""
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


def build_baseline_results(
    runner: LouvainNERunner,
    data: GraphData,
    repo_sim: torch.Tensor,
    seeds: Sequence[int],
    penalties: Sequence[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))
    results: List[Dict[str, object]] = []
    dense_attention = repo_dense_attention(data.x)

    structure_embeddings = [(seed, runner.embed(structure_edges, seed)) for seed in seeds]
    structure_result = evaluate_embeddings(
        "structure_only",
        {"embedding_dim": runner.embedding_dim},
        structure_embeddings,
        data,
        penalties,
    )
    results.append(structure_result)

    late_fusion_results = []
    for alpha in [0.03125, 0.25]:
        attr_predictions = build_threshold_predictions(repo_sim, alpha)
        attr_edges = edges_to_dict(attr_predictions.keys())
        embeddings = []
        for seed in seeds:
            struct_embedding = runner.embed(structure_edges, seed)
            attr_embedding = runner.embed(attr_edges, seed + 1000)
            embeddings.append((seed, concat_features([struct_embedding, attr_embedding])))
        late_fusion_result = evaluate_embeddings(
            f"late_fusion_concat_alpha_{alpha}",
            {"alpha": alpha, "aggregation": "concat"},
            embeddings,
            data,
            penalties,
        )
        late_fusion_results.append(late_fusion_result)
        results.append(late_fusion_result)

    repo_graph_results = []
    for alpha in [0.2, 0.25, 0.3, 0.35]:
        predicted = build_threshold_predictions(repo_sim, alpha)
        fused_edges = fuse_repo_edges(structure_edges, predicted, "repo_unweighted")
        embeddings = [(seed, runner.embed(fused_edges, seed)) for seed in seeds]
        repo_graph_results.append(
            evaluate_embeddings(
                f"repo_early_fusion_alpha_{alpha}",
                {"alpha": alpha, "mode": "repo_unweighted"},
                embeddings,
                data,
                penalties,
            )
        )
        refined = [
            (seed, dense_attention @ embedding)
            for seed, embedding in embeddings
        ]
        repo_graph_results.append(
            evaluate_embeddings(
                f"repo_early_fusion_attention_alpha_{alpha}",
                {"alpha": alpha, "mode": "repo_unweighted", "attention": "dense_repo"},
                refined,
                data,
                penalties,
            )
        )
    results.extend(repo_graph_results)

    weighted_results = []
    for alpha in [0.2, 0.25, 0.3]:
        predicted = build_threshold_predictions(repo_sim, alpha)
        for mode in ["method2", "method3", "method4"]:
            fused_edges = fuse_repo_edges(structure_edges, predicted, mode)
            embeddings = [(seed, runner.embed(fused_edges, seed)) for seed in seeds]
            weighted_results.append(
                evaluate_embeddings(
                    f"weighted_{mode}_alpha_{alpha}",
                    {"alpha": alpha, "mode": mode},
                    embeddings,
                    data,
                    penalties,
                )
            )
    results.extend(weighted_results)

    best_baseline = max(results, key=lambda item: item["selection_score_mean"])
    return results, best_baseline


def search_adaptive_graph(
    runner: LouvainNERunner,
    data: GraphData,
    cosine_sim: torch.Tensor,
    seeds: Sequence[int],
    penalties: Sequence[float],
) -> Dict[str, object]:
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))
    candidates = []
    for top_k in [10, 15, 20]:
        for mutual in [True]:
            for min_similarity in [0.15, 0.2]:
                predicted = build_topk_predictions(cosine_sim, top_k, mutual, min_similarity)
                if not predicted:
                    continue
                for overlap_scale in [1.0, 1.5]:
                    for new_scale in [0.5, 0.75]:
                        fused_edges = fuse_adaptive_edges(
                            structure_edges,
                            predicted,
                            overlap_scale=overlap_scale,
                            new_scale=new_scale,
                        )
                        embeddings = [
                            (seed, normalize_rows(runner.embed(fused_edges, seed)))
                            for seed in seeds
                        ]
                        candidates.append(
                            evaluate_embeddings(
                                (
                                    "adaptive_graph"
                                    f"_k_{top_k}_mutual_{int(mutual)}"
                                    f"_min_{min_similarity}_ov_{overlap_scale}_new_{new_scale}"
                                ),
                                {
                                    "top_k": top_k,
                                    "mutual": mutual,
                                    "min_similarity": min_similarity,
                                    "overlap_scale": overlap_scale,
                                    "new_scale": new_scale,
                                },
                                embeddings,
                                data,
                                penalties,
                            )
                        )
    return max(candidates, key=lambda item: item["selection_score_mean"])


def search_improved_pipeline(
    runner: LouvainNERunner,
    data: GraphData,
    cosine_sim: torch.Tensor,
    graph_config: Dict[str, object],
    seeds: Sequence[int],
    penalties: Sequence[float],
) -> Dict[str, object]:
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))
    predicted = build_topk_predictions(
        cosine_sim,
        int(graph_config["top_k"]),
        bool(graph_config["mutual"]),
        float(graph_config["min_similarity"]),
    )
    fused_edges = fuse_adaptive_edges(
        structure_edges,
        predicted,
        overlap_scale=float(graph_config["overlap_scale"]),
        new_scale=float(graph_config["new_scale"]),
    )
    feature_cache = {dim: feature_svd(data.x, dim) for dim in [32, 64, 128]}

    candidates = []
    for ensemble_size in [1, 3]:
        for gamma in [0.0, 0.25, 0.5]:
            for temperature in [1.0]:
                for feature_dim in [0, 64, 128]:
                    attention_matrix = None
                    if gamma > 0.0:
                        attention_matrix = sparse_attention_from_edges(
                            data.num_nodes,
                            fused_edges,
                            cosine_sim,
                            temperature=temperature,
                        )
                    embeddings = []
                    for seed in seeds:
                        if ensemble_size == 1:
                            graph_embedding = normalize_rows(runner.embed(fused_edges, seed))
                        else:
                            graph_embedding = aligned_ensemble(
                                runner,
                                fused_edges,
                                [seed + (1000 * idx) for idx in range(ensemble_size)],
                            )
                        if attention_matrix is not None:
                            graph_embedding = (1.0 - gamma) * graph_embedding + gamma * (
                                attention_matrix @ graph_embedding
                            )
                        graph_embedding = normalize_rows(graph_embedding)
                        parts = [graph_embedding]
                        if feature_dim > 0:
                            parts.append(feature_cache[feature_dim])
                        embeddings.append((seed, concat_features(parts)))
                    candidates.append(
                        evaluate_embeddings(
                            (
                                "improved_pipeline"
                                f"_ens_{ensemble_size}_gamma_{gamma}"
                                f"_temp_{temperature}_feat_{feature_dim}"
                            ),
                            {
                                "ensemble_size": ensemble_size,
                                "gamma": gamma,
                                "temperature": temperature,
                                "feature_dim": feature_dim,
                                "graph_config": graph_config,
                            },
                            embeddings,
                            data,
                            penalties,
                        )
                    )
    return max(candidates, key=lambda item: item["selection_score_mean"])


def run_final_evaluation(
    runner: LouvainNERunner,
    data: GraphData,
    repo_sim: torch.Tensor,
    cosine_sim: torch.Tensor,
    best_baseline: Dict[str, object],
    best_adaptive_graph: Dict[str, object],
    best_improved: Dict[str, object],
    seeds: Sequence[int],
    penalties: Sequence[float],
) -> Dict[str, object]:
    structure_edges = edges_to_dict(unique_undirected_edges(data.edge_index))

    baseline_name = str(best_baseline["name"])
    baseline_config = dict(best_baseline["config"])
    baseline_setup_start = time.perf_counter()
    if baseline_name == "structure_only":
        def baseline_embedding_fn(seed: int) -> torch.Tensor:
            return runner.embed(structure_edges, seed)
    elif baseline_name.startswith("late_fusion_concat"):
        alpha = float(baseline_config["alpha"])
        attr_predictions = build_threshold_predictions(repo_sim, alpha)
        attr_edges = edges_to_dict(attr_predictions.keys())

        def baseline_embedding_fn(seed: int) -> torch.Tensor:
            struct_embedding = runner.embed(structure_edges, seed)
            attr_embedding = runner.embed(attr_edges, seed + 1000)
            return concat_features([struct_embedding, attr_embedding])
    elif baseline_name.startswith("repo_early_fusion_attention"):
        dense_attention = repo_dense_attention(data.x)
        alpha = float(baseline_config["alpha"])
        predicted = build_threshold_predictions(repo_sim, alpha)
        fused_edges = fuse_repo_edges(structure_edges, predicted, "repo_unweighted")

        def baseline_embedding_fn(seed: int) -> torch.Tensor:
            graph_embedding = runner.embed(fused_edges, seed)
            return dense_attention @ graph_embedding
    elif baseline_name.startswith("repo_early_fusion"):
        alpha = float(baseline_config["alpha"])
        predicted = build_threshold_predictions(repo_sim, alpha)
        fused_edges = fuse_repo_edges(structure_edges, predicted, str(baseline_config["mode"]))

        def baseline_embedding_fn(seed: int) -> torch.Tensor:
            return runner.embed(fused_edges, seed)
    else:
        alpha = float(baseline_config["alpha"])
        mode = str(baseline_config["mode"])
        predicted = build_threshold_predictions(repo_sim, alpha)
        fused_edges = fuse_repo_edges(structure_edges, predicted, mode)

        def baseline_embedding_fn(seed: int) -> torch.Tensor:
            return runner.embed(fused_edges, seed)
    baseline_setup_time_seconds = time.perf_counter() - baseline_setup_start

    final_baseline = evaluate_method_with_timing(
        f"{baseline_name}_final",
        baseline_config,
        seeds,
        data,
        penalties,
        baseline_embedding_fn,
        setup_time_seconds=baseline_setup_time_seconds,
    )

    adaptive_config = dict(best_adaptive_graph["config"])
    adaptive_setup_start = time.perf_counter()
    predicted = build_topk_predictions(
        cosine_sim,
        int(adaptive_config["top_k"]),
        bool(adaptive_config["mutual"]),
        float(adaptive_config["min_similarity"]),
    )
    fused_edges = fuse_adaptive_edges(
        structure_edges,
        predicted,
        overlap_scale=float(adaptive_config["overlap_scale"]),
        new_scale=float(adaptive_config["new_scale"]),
    )
    adaptive_setup_time_seconds = time.perf_counter() - adaptive_setup_start

    improved_config = dict(best_improved["config"])
    improved_setup_start = time.perf_counter()
    improved_predicted = build_topk_predictions(
        cosine_sim,
        int(adaptive_config["top_k"]),
        bool(adaptive_config["mutual"]),
        float(adaptive_config["min_similarity"]),
    )
    improved_fused_edges = fuse_adaptive_edges(
        structure_edges,
        improved_predicted,
        overlap_scale=float(adaptive_config["overlap_scale"]),
        new_scale=float(adaptive_config["new_scale"]),
    )
    feature_dim = int(improved_config["feature_dim"])
    feature_embedding = feature_svd(data.x, feature_dim) if feature_dim > 0 else None
    gamma = float(improved_config["gamma"])
    temperature = float(improved_config["temperature"])
    ensemble_size = int(improved_config["ensemble_size"])
    attention_matrix = None
    if gamma > 0.0:
        attention_matrix = sparse_attention_from_edges(
            data.num_nodes,
            improved_fused_edges,
            cosine_sim,
            temperature=temperature,
        )
    improved_setup_time_seconds = time.perf_counter() - improved_setup_start

    def adaptive_embedding_fn(seed: int) -> torch.Tensor:
        return normalize_rows(runner.embed(fused_edges, seed))

    final_adaptive = evaluate_method_with_timing(
        "adaptive_graph_final",
        adaptive_config,
        seeds,
        data,
        penalties,
        adaptive_embedding_fn,
        setup_time_seconds=adaptive_setup_time_seconds,
    )

    def improved_embedding_fn(seed: int) -> torch.Tensor:
        if ensemble_size == 1:
            graph_embedding = normalize_rows(runner.embed(improved_fused_edges, seed))
        else:
            graph_embedding = aligned_ensemble(
                runner,
                improved_fused_edges,
                [seed + (1000 * idx) for idx in range(ensemble_size)],
            )
        if attention_matrix is not None:
            graph_embedding = (1.0 - gamma) * graph_embedding + gamma * (attention_matrix @ graph_embedding)
        graph_embedding = normalize_rows(graph_embedding)
        parts = [graph_embedding]
        if feature_embedding is not None:
            parts.append(feature_embedding)
        return concat_features(parts)

    final_improved = evaluate_method_with_timing(
        "improved_final",
        improved_config,
        seeds,
        data,
        penalties,
        improved_embedding_fn,
        setup_time_seconds=improved_setup_time_seconds,
    )

    return {
        "baseline": final_baseline,
        "adaptive_graph": final_adaptive,
        "improved": final_improved,
    }


def write_comparison_plot(
    output_path: Path,
    baseline: Dict[str, object],
    improved: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = ["Baseline", "Improved"]
    micro_means = [baseline["test_micro_f1_mean"], improved["test_micro_f1_mean"]]
    micro_stds = [baseline["test_micro_f1_std"], improved["test_micro_f1_std"]]
    macro_means = [baseline["test_macro_f1_mean"], improved["test_macro_f1_mean"]]
    macro_stds = [baseline["test_macro_f1_std"], improved["test_macro_f1_std"]]
    setup_means = [baseline["setup_time_seconds"], improved["setup_time_seconds"]]
    per_seed_means = [baseline["per_seed_eval_time_seconds_mean"], improved["per_seed_eval_time_seconds_mean"]]
    per_seed_stds = [baseline["per_seed_eval_time_seconds_std"], improved["per_seed_eval_time_seconds_std"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x = np.arange(len(methods))
    width = 0.34

    axes[0].bar(x - width / 2, micro_means, width, yerr=micro_stds, capsize=4, label="Micro-F1", color="#2A6F97")
    axes[0].bar(x + width / 2, macro_means, width, yerr=macro_stds, capsize=4, label="Macro-F1", color="#E07A5F")
    axes[0].set_xticks(x, methods)
    axes[0].set_ylim(0.0, 0.9)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Accuracy Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x, setup_means, width=0.5, color=["#6C9A8B", "#C8553D"])
    axes[1].set_xticks(x, methods)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("One-Time Setup Time")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].bar(x, per_seed_means, width=0.5, yerr=per_seed_stds, capsize=4, color=["#8F5EA2", "#D1495B"])
    axes[2].set_xticks(x, methods)
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Per-Seed Evaluation Time")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    for idx, value in enumerate(micro_means):
        axes[0].text(idx - width / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(macro_means):
        axes[0].text(idx + width / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(setup_means):
        axes[1].text(idx, value + 0.02, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(per_seed_means):
        axes[2].text(idx, value + max(per_seed_stds[idx], 0.02) + 0.02, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("LouvainNE Baseline vs Improved Pipeline", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and improve LouvainNE node classification experiments.")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--tune-runs", type=int, default=2)
    parser.add_argument("--eval-runs", type=int, default=6)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "louvainne_results.json",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=REPO_ROOT / "results" / "louvainne_comparison.png",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 8))))

    data, data_source = load_cora_graph()
    runner = LouvainNERunner(REPO_ROOT, data.num_nodes, args.embedding_dim)
    repo_sim = repo_similarity(data.x)
    cosine_sim = cosine_similarity(data.x)
    penalties = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    tune_seeds = [548 + idx for idx in range(args.tune_runs)]
    eval_seeds = [1548 + idx for idx in range(args.eval_runs)]

    print("Running baseline search...", flush=True)
    baseline_results, best_baseline = build_baseline_results(
        runner,
        data,
        repo_sim,
        tune_seeds,
        penalties,
    )
    print(f"Best baseline: {best_baseline['name']}", flush=True)
    print("Running adaptive graph search...", flush=True)
    best_adaptive_graph = search_adaptive_graph(
        runner,
        data,
        cosine_sim,
        tune_seeds,
        penalties,
    )
    print(f"Best adaptive graph: {best_adaptive_graph['name']}", flush=True)
    print("Running improved pipeline search...", flush=True)
    best_improved = search_improved_pipeline(
        runner,
        data,
        cosine_sim,
        best_adaptive_graph["config"],
        tune_seeds,
        penalties,
    )
    print(f"Best improved search result: {best_improved['name']}", flush=True)
    print("Running final evaluation...", flush=True)
    final_results = run_final_evaluation(
        runner,
        data,
        repo_sim,
        cosine_sim,
        best_baseline,
        best_adaptive_graph,
        best_improved,
        eval_seeds,
        penalties,
    )

    payload = {
        "data_source": data_source,
        "embedding_dim": args.embedding_dim,
        "tune_runs": args.tune_runs,
        "eval_runs": args.eval_runs,
        "baseline_search": baseline_results,
        "best_baseline": best_baseline,
        "best_adaptive_graph": best_adaptive_graph,
        "best_improved_search": best_improved,
        "final_results": final_results,
        "artifacts": {
            "comparison_plot": str(args.plot_path),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_comparison_plot(args.plot_path, final_results["baseline"], final_results["improved"])

    baseline = final_results["baseline"]
    improved = final_results["improved"]
    print(f"Data source: {data_source}")
    print(
        "Baseline:",
        baseline["name"],
        f"micro={baseline['test_micro_f1_mean']:.4f}+-{baseline['test_micro_f1_std']:.4f}",
        f"macro={baseline['test_macro_f1_mean']:.4f}+-{baseline['test_macro_f1_std']:.4f}",
        f"setup={baseline['setup_time_seconds']:.2f}s",
        f"per_seed={baseline['per_seed_eval_time_seconds_mean']:.2f}s+-{baseline['per_seed_eval_time_seconds_std']:.2f}s",
    )
    print(
        "Improved:",
        improved["name"],
        f"micro={improved['test_micro_f1_mean']:.4f}+-{improved['test_micro_f1_std']:.4f}",
        f"macro={improved['test_macro_f1_mean']:.4f}+-{improved['test_macro_f1_std']:.4f}",
        f"setup={improved['setup_time_seconds']:.2f}s",
        f"per_seed={improved['per_seed_eval_time_seconds_mean']:.2f}s+-{improved['per_seed_eval_time_seconds_std']:.2f}s",
    )
    print(f"Results written to {args.output_json}")
    print(f"Comparison plot written to {args.plot_path}")


if __name__ == "__main__":
    main()
