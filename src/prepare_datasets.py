#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import download_google_url, extract_zip
from torch_geometric.datasets import Planetoid

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
PLANETOID_ROOT = DATA_ROOT / "Planetoid"
BLOGCATALOG_ROOT = DATA_ROOT / "BlogCatalog"
BLOGCATALOG_GOOGLE_ID = "178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS"
REQUIRED_BLOGCATALOG_FILES = ["attrs.npz", "edgelist.txt", "labels.txt"]
SPLIT_SEEDS = [1548, 1549, 1550, 1551, 1552]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_stratified_masks(
    y: torch.Tensor,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, torch.Tensor]:
    num_nodes = int(y.numel())
    seeds = list(seeds)
    train_mask = torch.zeros((num_nodes, len(seeds)), dtype=torch.bool)
    val_mask = torch.zeros((num_nodes, len(seeds)), dtype=torch.bool)
    test_mask = torch.zeros((num_nodes, len(seeds)), dtype=torch.bool)

    classes = torch.unique(y).tolist()
    for split_idx, seed in enumerate(seeds):
        generator = torch.Generator()
        generator.manual_seed(seed)
        for cls in classes:
            cls_idx = torch.where(y == cls)[0]
            perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
            train_count = max(1, int(round(cls_idx.numel() * train_ratio)))
            val_count = max(1, int(round(cls_idx.numel() * val_ratio)))
            if train_count + val_count >= cls_idx.numel():
                val_count = max(1, cls_idx.numel() - train_count - 1)
            train_end = train_count
            val_end = min(cls_idx.numel() - 1, train_end + val_count)

            train_nodes = perm[:train_end]
            val_nodes = perm[train_end:val_end]
            test_nodes = perm[val_end:]
            if test_nodes.numel() == 0:
                test_nodes = val_nodes[-1:].clone()
                val_nodes = val_nodes[:-1]
                if val_nodes.numel() == 0:
                    val_nodes = train_nodes[-1:].clone()
                    train_nodes = train_nodes[:-1]

            train_mask[train_nodes, split_idx] = True
            val_mask[val_nodes, split_idx] = True
            test_mask[test_nodes, split_idx] = True

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def prepare_planetoid_dataset(name: str) -> Dict[str, object]:
    dataset = Planetoid(root=str(PLANETOID_ROOT), name=name)
    data = dataset[0]
    return {
        "name": name,
        "source": f"data/Planetoid/{name}",
        "num_nodes": int(data.num_nodes),
        "num_edges_directed": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(data.y.max().item()) + 1,
    }


def find_extracted_blogcatalog_dir(raw_dir: Path) -> Path:
    direct = raw_dir / "blogcatalog"
    if direct.is_dir():
        return direct
    for candidate in raw_dir.iterdir():
        if candidate.is_dir() and (candidate / "attrs.npz").is_file():
            return candidate
    raise FileNotFoundError("Could not locate extracted BlogCatalog files after download.")


def download_blogcatalog_raw(force: bool = False) -> None:
    raw_dir = BLOGCATALOG_ROOT / "raw"
    ensure_directory(raw_dir)
    if not force and all((raw_dir / name).is_file() for name in REQUIRED_BLOGCATALOG_FILES):
        return

    for name in REQUIRED_BLOGCATALOG_FILES:
        path = raw_dir / name
        if path.exists():
            path.unlink()

    zip_path = Path(download_google_url(BLOGCATALOG_GOOGLE_ID, str(raw_dir), "data.zip"))
    extract_zip(str(zip_path), str(raw_dir))
    zip_path.unlink(missing_ok=True)

    extracted_dir = find_extracted_blogcatalog_dir(raw_dir)
    for name in REQUIRED_BLOGCATALOG_FILES:
        shutil.move(str(extracted_dir / name), str(raw_dir / name))

    for artifact in [extracted_dir, raw_dir / "__MACOSX"]:
        if artifact.exists():
            shutil.rmtree(artifact, ignore_errors=True)


def load_blogcatalog_labels(path: Path) -> torch.Tensor:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append([int(part) for part in line.split()])

    if not rows:
        raise ValueError("BlogCatalog labels.txt is empty.")

    labels = torch.empty(len(rows), dtype=torch.long)
    for row in rows:
        node_id, *node_labels = row
        if len(node_labels) != 1:
            raise ValueError("BlogCatalog multi-label rows are not supported by this benchmark.")
        labels[node_id] = int(node_labels[0]) - 1
    return labels


def process_blogcatalog(force: bool = False) -> Dict[str, object]:
    raw_dir = BLOGCATALOG_ROOT / "raw"
    processed_dir = BLOGCATALOG_ROOT / "processed"
    processed_path = processed_dir / "data.pt"
    ensure_directory(processed_dir)

    if force and processed_path.exists():
        processed_path.unlink()
    if processed_path.is_file():
        payload = torch.load(processed_path, weights_only=False)
        return {
            "name": "BlogCatalog",
            "source": "data/BlogCatalog",
            "num_nodes": int(payload["x"].shape[0]),
            "num_edges_directed": int(payload["edge_index"].shape[1]),
            "num_features": int(payload["x"].shape[1]),
            "num_classes": int(payload["y"].max().item()) + 1,
        }

    download_blogcatalog_raw(force=force)

    x_sparse = sp.load_npz(raw_dir / "attrs.npz").tocsr()
    x = torch.from_numpy(x_sparse.toarray()).to(torch.float32)
    edge_index = torch.from_numpy(np.loadtxt(raw_dir / "edgelist.txt", dtype=np.int64).T).to(torch.long)
    y = load_blogcatalog_labels(raw_dir / "labels.txt")
    masks = build_stratified_masks(y, SPLIT_SEEDS)

    payload = {
        "x": x,
        "edge_index": edge_index.contiguous(),
        "y": y,
        "train_mask": masks["train_mask"],
        "val_mask": masks["val_mask"],
        "test_mask": masks["test_mask"],
    }
    torch.save(payload, processed_path)
    return {
        "name": "BlogCatalog",
        "source": "data/BlogCatalog",
        "num_nodes": int(x.shape[0]),
        "num_edges_directed": int(edge_index.shape[1]),
        "num_features": int(x.shape[1]),
        "num_classes": int(y.max().item()) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the datasets used by this repo.")
    parser.add_argument("--force", action="store_true", help="Re-download and reprocess the supported datasets.")
    args = parser.parse_args()

    ensure_directory(DATA_ROOT)

    manifest: Dict[str, object] = {"datasets": []}
    for name in ["Cora", "CiteSeer", "PubMed"]:
        info = prepare_planetoid_dataset(name)
        manifest["datasets"].append(info)
        print(
            name,
            f"nodes={info['num_nodes']}",
            f"edges={info['num_edges_directed']}",
            f"features={info['num_features']}",
            flush=True,
        )

    blogcatalog = process_blogcatalog(force=args.force)
    manifest["datasets"].append(blogcatalog)
    print(
        "BlogCatalog",
        f"nodes={blogcatalog['num_nodes']}",
        f"edges={blogcatalog['num_edges_directed']}",
        f"features={blogcatalog['num_features']}",
        flush=True,
    )

    manifest_path = DATA_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
