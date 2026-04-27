import subprocess
import json
import numpy as np
import os
import argparse

def run_experiment(dataset, tau, seed):
    output_file = f"results/cv_{dataset}_s{seed}.json"
    cmd = [
        "venv/bin/python", "run_hybrid.py",
        "--dataset", dataset,
        "--tau", str(tau),
        "--seed", str(seed),
        "--adaptive",
        "--learned_conf",
        "--filter",
        "--fusion", "hard",
        "--output", output_file
    ]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    print(f"--- Cross-Validation for {args.dataset} (n={args.seeds}) ---")
    
    results_list = []
    for i in range(args.seeds):
        print(f"Seed {i}...", end="", flush=True)
        res = run_experiment(args.dataset, args.tau, i * 42)
        if res:
            results_list.append(res)
            print("Done")
        else:
            print("Failed")

    if not results_list:
        print("No results collected.")
        return

    # Aggregate
    base_accs = [r["metrics"]["baseline"]["accuracy"] for r in results_list]
    hybrid_accs = [r["metrics"]["hybrid"]["accuracy"] for r in results_list]
    hard_gain_accs = []
    for r in results_list:
        h = r["metrics"]["hard_split"]["hybrid"]["accuracy"]
        b = r["metrics"]["hard_split"]["baseline"]["accuracy"]
        if h > 0: # Avoid empty hard nodes
            hard_gain_accs.append(h - b)

    print("\n--- Final Statistics ---")
    print(f"Baseline Acc: {np.mean(base_accs):.4f} +/- {np.std(base_accs):.4f}")
    print(f"Hybrid Acc: {np.mean(hybrid_accs):.4f} +/- {np.std(hybrid_accs):.4f}")
    if hard_gain_accs:
        print(f"Hard Node Lift: {np.mean(hard_gain_accs):.4f} +/- {np.std(hard_gain_accs):.4f}")

if __name__ == "__main__":
    main()
