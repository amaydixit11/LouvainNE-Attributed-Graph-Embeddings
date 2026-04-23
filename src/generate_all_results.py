#!/usr/bin/env python3
"""
Comprehensive results generator.
Clears old results and runs every experiment type in the repository.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_command(cmd: List[str], description: str) -> bool:
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        print(f"✓ {description} completed successfully", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}", flush=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate all possible results for the project.")
    parser.add_argument("--skip-clean", action="store_true", help="Do not clear the results directory")
    args = parser.parse_args()

    # 1. Clear old results
    if not args.skip_clean:
        print("Cleaning old results...")
        results_dir = REPO_ROOT / "results"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ {results_dir} cleared.")

    # List of scripts to run in order
    # We use sys.executable to ensure we use the same python environment
    experiments = [
        # a. Data Preparation
        ([ "src/prepare_datasets.py" ], "Dataset Preparation"),

        # b. Master Runner (covers standard benchmarks, OGB, and SOTA report)
        ([ "src/run_all_experiments.py" ], "Master Experiment Runner (Standard + OGB + SOTA Report)"),

        # c. Detailed Cora Experiment (with tuning)
        ([ "src/run_louvainne_experiments.py", "--tune-runs", "2", "--eval-runs", "5" ], "Detailed Cora Experiment (Baseline vs Improved)"),

        # d. Hyperparameter Optimization
        ([ "src/optimize_louvainne.py" ], "Hyperparameter Optimization"),

        # e. Scalability Benchmarks
        ([ "src/benchmark_scalability.py" ], "Scalability Benchmarks (Real Graphs)"),
        ([ "src/benchmark_scalability_synthetic.py" ], "Scalability Benchmarks (Synthetic Graphs)"),

        # f. Ablation Study
        ([ "src/ablation_study.py" ], "Ablation Study"),

        # g. PDF Report Generation
        ([ "src/generate_pdf_report.py" ], "PDF Report Generation"),
    ]

    success_count = 0
    for script_args, description in experiments:
        cmd = [sys.executable] + script_args
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"⚠ Warning: {description} failed. Continuing with others...")

    print("\n" + "="*70)
    print(f"All-in-one execution complete. {success_count}/{len(experiments)} tasks succeeded.")
    print(f"Results are located in: {REPO_ROOT / 'results'}")
    print("="*70)

if __name__ == "__main__":
    main()
