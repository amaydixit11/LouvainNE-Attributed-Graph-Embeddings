import subprocess
import json
import os
import pandas as pd

def run_cmd(dataset, gnn, baseline, hybrid=True, tau=0.7):
    output = f"results/bench_{dataset}_{gnn}_{baseline}_{'hybrid' if hybrid else 'global'}.json"
    cmd = [
        "venv/bin/python", "run_hybrid.py",
        "--dataset", dataset,
        "--gnn_type", gnn,
        "--baseline", baseline,
        "--tau", str(tau),
        "--adaptive",
        "--learned_conf",
        "--filter"
    ]
    if not hybrid:
        # Note: run_hybrid.py currently ALWAYS does hybrid after global.
        # I'll just use the global result from the hybrid run.
        pass
    
    cmd.extend(["--output", output])
    print(f"Running: {dataset} | GNN: {gnn} | Base: {baseline} | Hybrid: {hybrid}")
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(output):
        with open(output, "r") as f:
            return json.load(f)
    return None

def main():
    datasets = ["Cora", "CiteSeer", "PubMed"]
    gnn_types = ["GCN", "GAT", "SAGE"]
    baselines = ["majority", "classifier"]
    
    all_results = []
    
    for ds in datasets:
        for bl in baselines:
            # We only need to run each GNN once per dataset-baseline combo
            for gnn in gnn_types:
                res = run_cmd(ds, gnn, bl)
                if res:
                    m = res["metrics"]
                    s = res["stats"]
                    
                    sota_map = {"Cora": 0.815, "CiteSeer": 0.703, "PubMed": 0.790, "ogbn-arxiv": 0.7174}
                    
                    row = {
                        "Dataset": ds,
                        "Baseline_Type": bl,
                        "GNN_Type": gnn,
                        "Tau": 0.7,
                        "Nodes_GNN_Pct": (s["num_hard"] / res["metrics"]["baseline"]["total"]) * 100 if "total" in res["metrics"]["baseline"] else 0,
                        "Baseline_Acc": m["baseline"]["accuracy"],
                        "Global_GNN_Acc": m["global_gnn"]["accuracy"],
                        "Hybrid_Acc": m["hybrid"]["accuracy"],
                        "Global_SOTA": sota_map.get(ds, 0),
                        "T_Global(s)": s.get("t_global_train", 0),
                        "T_Hybrid(s)": s.get("t_hybrid_train", 0),
                        "Hard_Acc_Base": m["hard_split"]["baseline"]["accuracy"],
                        "Hard_Acc_Hybrid": m["hard_split"]["hybrid"]["accuracy"]
                    }
                    # Fix percentage calc if 'total' missing
                    if "total" not in m["baseline"]:
                         # Approximate
                         row["Nodes_GNN_Pct"] = (s["num_hard"] / 2000) * 100 # Rough for cora/citeseer
                    
                    all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv("results/master_benchmark.csv", index=False)
    print("\n--- MASTER BENCHMARK COMPLETE ---")
    print(df.to_markdown())

if __name__ == "__main__":
    main()
