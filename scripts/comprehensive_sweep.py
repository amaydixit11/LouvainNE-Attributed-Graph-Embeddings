import subprocess
import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time

def run_single_experiment(dataset, tau, k, baseline, gnn="ResGated"):
    output = f"results/sweep_{dataset}_t{tau}_k{k}_{baseline}.json"
    cmd = [
        "venv/bin/python", "run_hybrid.py",
        "--dataset", dataset,
        "--tau", str(tau),
        "--k", str(k),
        "--baseline", baseline,
        "--adaptive",
        "--learned_conf",
        "--filter",
        "--output", output
    ]
    print(f"Starting: {dataset} | tau={tau} | k={k} | base={baseline}")
    start = time.time()
    subprocess.run(cmd, capture_output=True)
    duration = time.time() - start
    
    if os.path.exists(output):
        with open(output, "r") as f:
            data = json.load(f)
            data["stats"]["duration_total"] = duration
            # Re-save with timing
            with open(output, "w") as fw:
                json.dump(data, fw, indent=4)
            return True
    return False

def main():
    # RIGOROUS PARAMETER SPACE
    datasets = ["Cora", "CiteSeer", "PubMed"]
    taus = [0.5, 0.7, 0.9]
    ks = [1, 2]
    baselines = ["majority", "classifier"]
    
    tasks = []
    for ds in datasets:
        for t in taus:
            for k in ks:
                for bl in baselines:
                    tasks.append((ds, t, k, bl))

    print(f"--- TOTAL EXPERIMENTS PLANNED: {len(tasks)} ---")
    
    # Run in parallel (4 workers to avoid OOM but stay fast)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_experiment, *task) for task in tasks]
        for f in futures:
            f.result()

    # AGGREGATE RESULTS
    all_rows = []
    sota_map = {"Cora": 0.815, "CiteSeer": 0.703, "PubMed": 0.790}
    
    for ds in datasets:
        for t in taus:
            for k in ks:
                for bl in baselines:
                    path = f"results/sweep_{ds}_t{t}_k{k}_{bl}.json"
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            res = json.load(f)
                            m = res["metrics"]
                            s = res["stats"]
                            all_rows.append({
                                "Dataset": ds,
                                "Tau": t,
                                "K_Hop": k,
                                "Baseline": bl,
                                "Hybrid_Acc": m["hybrid"]["accuracy"],
                                "Global_GNN_Acc": m["global_gnn"]["accuracy"],
                                "Baseline_Acc": m["baseline"]["accuracy"],
                                "SOTA_Diff": m["hybrid"]["accuracy"] - sota_map[ds],
                                "GNN_Nodes_Pct": (s["num_hard"] / m["baseline"]["total"]) * 100,
                                "Time_Saved_Pct": (1 - (s["t_hybrid_train"] / s["t_global_train"])) * 100 if s["t_global_train"] > 0 else 0
                            })
    
    df = pd.DataFrame(all_rows)
    df.to_csv("results/comprehensive_sweep.csv", index=False)
    print("\n--- SWEEP COMPLETE: results/comprehensive_sweep.csv ---")
    # Grouped summary
    print(df.groupby(["Dataset", "Baseline"]).agg({"Hybrid_Acc": "max", "Time_Saved_Pct": "mean"}).to_markdown())

if __name__ == "__main__":
    main()
