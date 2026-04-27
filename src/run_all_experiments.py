#!/usr/bin/env python3
"""
Master experiment runner.
Runs all benchmarks: Node Classification, Link Prediction, and Runtime comparison
across all datasets (standard + OGB large-scale).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            capture_output=False,
        )
        print(f"✓ {description} completed successfully", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}", flush=True)
        return False


def check_dependencies() -> List[str]:
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import torch_geometric
    except ImportError:
        missing.append("torch-geometric")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import ogb
    except ImportError:
        missing.append("ogb (optional, needed for OGB benchmarks)")
    
    return missing


def prepare_datasets() -> bool:
    """Run dataset preparation."""
    return run_command(
        [sys.executable, str(REPO_ROOT / "src" / "prepare_datasets.py")],
        "Prepare datasets"
    )


def run_standard_benchmarks(datasets: List[str], embedding_dim: int) -> bool:
    """Run standard benchmarks with link prediction."""
    return run_command(
        [sys.executable, str(REPO_ROOT / "src" / "benchmark_datasets_lp.py"),
         "--datasets"] + datasets + ["--embedding-dim", str(embedding_dim)],
        f"Standard benchmarks (link prediction enabled) for {datasets}"
    )


def run_ogb_benchmarks(datasets: List[str], embedding_dim: int, no_attributes: bool = False) -> bool:
    """Run OGB benchmarks."""
    cmd = [sys.executable, str(REPO_ROOT / "src" / "benchmark_ogb.py"),
           "--datasets"] + datasets + ["--embedding-dim", str(embedding_dim)]
    if no_attributes:
        cmd.append("--no-attributes")
    
    return run_command(
        cmd,
        f"OGB benchmarks for {datasets}"
    )


def generate_report() -> bool:
    """Generate comprehensive SOTA comparison report."""
    return run_command(
        [sys.executable, str(REPO_ROOT / "src" / "generate_sota_report.py")],
        "Generate comprehensive SOTA report"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Master experiment runner for all benchmarks.")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation (use if already prepared)",
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip standard benchmarks",
    )
    parser.add_argument(
        "--skip-ogb",
        action="store_true",
        help="Skip OGB benchmarks",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora", "CiteSeer", "PubMed", "BlogCatalog"],
        help="Standard datasets to benchmark",
    )
    parser.add_argument(
        "--ogb-datasets",
        nargs="+",
        default=["ogbn-arxiv"],
        help="OGB datasets to benchmark",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for all methods",
    )
    parser.add_argument(
        "--ogb-no-attributes",
        action="store_true",
        help="Run OGB benchmarks without attributes",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and exit",
    )
    args = parser.parse_args()
    
    print("="*70)
    print("LouvainNE Attributed Graph Embeddings - Master Experiment Runner")
    print("="*70)
    print()
    print("This runner will execute the following experiments:")
    if not args.skip_prepare:
        print("  1. Dataset preparation")
    if not args.skip_standard:
        print("  2. Standard benchmarks (Node Classification + Link Prediction)")
    if not args.skip_ogb:
        print("  3. OGB large-scale benchmarks (Node Classification + Link Prediction)")
    if not args.skip_report:
        print("  4. Comprehensive SOTA comparison report generation")
    print()
    
    missing = check_dependencies()
    if missing:
        print(f"⚠ Missing dependencies: {', '.join(missing)}")
        print("Install missing packages before running experiments.")
        if "ogb (optional, needed for OGB benchmarks)" in missing and not args.skip_ogb:
            print("  To install OGB: pip install ogb")
            print("  Or skip OGB benchmarks with: --skip-ogb")
        sys.exit(1)
    
    if args.check_only:
        print("✓ All dependencies are installed.")
        return
    
    success_count = 0
    total_count = 0
    
    if not args.skip_prepare:
        total_count += 1
        if prepare_datasets():
            success_count += 1
        else:
            print("⚠ Dataset preparation failed. Continuing with existing data...")
    
    if not args.skip_standard:
        total_count += 1
        if run_standard_benchmarks(args.datasets, args.embedding_dim):
            success_count += 1
        else:
            print("⚠ Standard benchmarks failed.")
    
    if not args.skip_ogb:
        total_count += 1
        if run_ogb_benchmarks(args.ogb_datasets, args.embedding_dim, args.ogb_no_attributes):
            success_count += 1
        else:
            print("⚠ OGB benchmarks failed (this is expected if ogb is not installed or datasets are unavailable).")
    
    if not args.skip_report:
        total_count += 1
        if generate_report():
            success_count += 1
        else:
            print("⚠ Report generation failed.")
    
    print()
    print("="*70)
    print(f"Experiment Summary: {success_count}/{total_count} completed successfully")
    print("="*70)
    
    if success_count == total_count:
        print()
        print("All experiments completed successfully!")
        print("Results are available in the results/ directory:")
        print("  - results/benchmark_summary.json: Standard benchmark results")
        print("  - results/benchmark_summary.md: Markdown summary")
        print("  - results/benchmark_summary.png: Comparison plots")
        print("  - results/ogb_benchmark_summary.json: OGB benchmark results")
        print("  - results/ogb_sota_comparison.md: OGB SOTA comparison")
        print("  - results/comprehensive_benchmark_report.md: Full report")
        print()
        print("To view the comprehensive report:")
        print(f"  cat {REPO_ROOT / 'results' / 'comprehensive_benchmark_report.md'}")


if __name__ == "__main__":
    main()
