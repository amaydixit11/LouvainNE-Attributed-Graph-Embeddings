import torch
import numpy as np
from hybrid_src.data import load_dataset
from hybrid_src.baseline import get_louvain_communities, community_feature_classifier, learned_confidence
from hybrid_src.subgraph import identify_hard_nodes, get_adaptive_subgraph_nodes, prepare_pyg_subgraph, filter_subgraph_edges
from hybrid_src.gnn import get_model, train_gnn, get_gnn_predictions
from hybrid_src.eval import evaluate, calibration_analysis
from hybrid_src.fusion import merge_predictions
import json
import os

def main():
    dataset_name = "ogbn-arxiv"
    print(f"--- Loading {dataset_name} ---")
    G, data, num_classes = load_dataset(dataset_name)
    
    print("--- Phase 1: Baseline (Classifier) ---")
    # Note: On 170k nodes, Louvain might take a minute
    partition = get_louvain_communities(G)
    louvain_preds = community_feature_classifier(partition, data.x, data.y, data.train_mask)
    
    # CONFIDENCE
    print("Training confidence predictor...")
    confidences = learned_confidence(G, partition, louvain_preds, data.y.cpu().numpy(), data.train_mask.cpu().numpy())
    
    baseline_acc = evaluate(data.y, louvain_preds, data.test_mask)["accuracy"]
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # HARD NODES (using tau=0.7 as per our optimal finding)
    tau = 0.7
    hard_nodes = identify_hard_nodes(confidences, tau=tau)
    print(f"Hard nodes: {len(hard_nodes)} ({len(hard_nodes)/data.num_nodes*100:.2f}%)")
    
    # SUBGRAPH (This is the heavy part for OGB)
    print("Extracting adaptive subgraph...")
    subgraph_nodes = get_adaptive_subgraph_nodes(G, hard_nodes, confidences)
    sub_data, _, node_list = prepare_pyg_subgraph(data, subgraph_nodes)
    print(f"Subgraph size: {sub_data.num_nodes} nodes")
    
    # FILTER
    print("Filtering noise...")
    sub_data = filter_subgraph_edges(sub_data, partition)
    
    # GNN (ResGated High-Res)
    print("Training Optimized GNN on subgraph...")
    model = get_model("ResGated", data.num_features, 256, num_classes) 
    model = train_gnn(model, sub_data, epochs=500, lr=0.001) 
    
    gnn_probs = get_gnn_predictions(model, sub_data)
    gnn_label_preds = gnn_probs.argmax(dim=-1).cpu().numpy()
    
    # FUSION
    final_preds = merge_predictions(louvain_preds, gnn_label_preds, hard_nodes, node_list)
    hybrid_acc = evaluate(data.y, final_preds, data.test_mask)["accuracy"]
    
    print(f"--- OGB RESULTS ---")
    print(f"Baseline: {baseline_acc:.4f}")
    print(f"Hybrid:   {hybrid_acc:.4f}")
    
    results = {
        "dataset": dataset_name,
        "tau": tau,
        "baseline_acc": baseline_acc,
        "hybrid_acc": hybrid_acc,
        "pct_hard": len(hard_nodes)/data.num_nodes
    }
    with open("results/ogb_hybrid_quick.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
