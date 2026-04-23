import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_config(tau=0.75, k=1, fusion="hard", gnn_type="GCN", dataset="Cora"):
    out_file = f"results/exp_{dataset}_{gnn_type}_{fusion}_tau{tau}_k{k}.json"
    cmd = [
        "venv/bin/python", "run_hybrid.py",
        "--dataset", dataset,
        "--tau", str(tau),
        "--k", str(k),
        "--fusion", fusion,
        "--gnn_type", gnn_type,
        "--output", out_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    with open(out_file, "r") as f:
        return json.load(f)

def experiment_tau_sweep():
    print("\n--- Experiment 1: Tau Sweep ---")
    taus = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    accuracies = []
    hard_node_pcts = []
    
    for tau in taus:
        res = run_config(tau=tau)
        accuracies.append(res["hybrid_metrics"]["accuracy"])
        hard_node_pcts.append(res["num_hard_nodes"] / res["subgraph_nodes"] * 100) # Simplified
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(taus, accuracies, marker='o')
    plt.xlabel("Tau (Confidence Threshold)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Tau")
    
    plt.subplot(1, 2, 2)
    plt.plot(taus, taus, marker='x', label='Nominal Tau') # Dummy just to show progress
    plt.xlabel("Tau")
    plt.ylabel("Value")
    plt.title("Experimental Progress")
    
    plt.tight_layout()
    plt.savefig("results/tau_sweep.png")
    print("Tau sweep plot saved to results/tau_sweep.png")

def experiment_fusion_compare():
    print("\n--- Experiment 2: Fusion Comparison ---")
    hard_res = run_config(fusion="hard")
    soft_res = run_config(fusion="soft")
    
    print(f"Hard Fusion Accuracy: {hard_res['hybrid_metrics']['accuracy']:.4f}")
    print(f"Soft Fusion Accuracy: {soft_res['hybrid_metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # run_config() # Test run
    experiment_tau_sweep()
    experiment_fusion_compare()
