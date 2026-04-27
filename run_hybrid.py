import argparse
import torch
import numpy as np
import time
import json
import os
import random

from hybrid_src.data import load_dataset
from hybrid_src.baseline import (
    get_louvain_communities, 
    majority_vote_prediction, 
    community_feature_classifier,
    compute_confidence
)
from hybrid_src.subgraph import identify_hard_nodes, get_adaptive_subgraph_nodes, prepare_pyg_subgraph
from hybrid_src.gnn import train_gnn, get_gnn_predictions, get_model
from hybrid_src.fusion import merge_predictions, soft_fusion
from hybrid_src.eval import evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Hybrid Graph Inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--baseline", type=str, default="majority", choices=["majority", "classifier"])
    parser.add_argument("--tau", type=float, default=0.75, help="Hard node threshold")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive k-hop expansion")
    parser.add_argument("--k", type=int, default=1, help="Fixed k-hop if not adaptive")
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--targeted", action="store_true", help="Supervise GNN ONLY on hard nodes")
    parser.add_argument("--filter", action="store_true", help="Filter noisy edges in subgraph")
    parser.add_argument("--learned_conf", action="store_true", help="Use ML model for confidence")
    parser.add_argument("--fusion", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden dimension for GNN")
    parser.add_argument("--output", type=str, default="results/hybrid_result.json")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print(f"--- Phase 0: Loading {args.dataset} ---")
    G, data, num_classes = load_dataset(args.dataset)
    
    print(f"--- Phase 1: Baseline Logic ---")
    partition = get_louvain_communities(G)
    if args.baseline == "majority":
        louvain_preds = majority_vote_prediction(partition, data.y, data.train_mask)
    else:
        louvain_preds = community_feature_classifier(partition, data.x, data.y, data.train_mask)
    
    # CONFIDENCE LOGIC
    if args.learned_conf:
        from hybrid_src.baseline import learned_confidence
        print("Training learned confidence predictor...")
        confidences = learned_confidence(G, partition, louvain_preds, data.y.cpu().numpy(), data.train_mask.cpu().numpy())
    else:
        confidences = compute_confidence(G, partition, louvain_preds)
    
    baseline_test_metrics = evaluate(data.y, louvain_preds, data.test_mask)
    print(f"Baseline Accuracy (Test): {baseline_test_metrics['accuracy']:.4f}")
    
    print(f"--- Phase 2: Identify Hard Nodes ---")
    hard_nodes = identify_hard_nodes(confidences, tau=args.tau)
    hard_node_mask = np.zeros(data.num_nodes, dtype=bool)
    hard_node_mask[hard_nodes] = True
    print(f"Hard nodes: {len(hard_nodes)} ({len(hard_nodes)/data.num_nodes*100:.2f}%)")
    
    print(f"--- Phase 3: Subgraph Extraction ---")
    if args.adaptive:
        print("Using adaptive expansion...")
        subgraph_nodes = get_adaptive_subgraph_nodes(G, hard_nodes, confidences)
    else:
        # Re-implement fixed k logic easily
        from hybrid_src.subgraph import nx
        subgraph_nodes = set(hard_nodes)
        for node in hard_nodes:
            subgraph_nodes.update(nx.single_source_shortest_path_length(G, node, cutoff=args.k).keys())
            
    import time
    
    print(f"--- Phase 4: Full-Graph GNN Baseline ---")
    num_communities = len(set(partition.values()))
    global_model = get_model(args.gnn_type, data.num_features, args.hidden_channels, num_classes, 
                           num_communities=num_communities)
    # Pass full data comm_ids
    all_comm_ids = torch.tensor([partition[i] for i in range(data.num_nodes)], dtype=torch.long)
    data.comm_ids = all_comm_ids
    
    t_start_global = time.time()
    global_model = train_gnn(global_model, data, epochs=200)
    t_total_global = time.time() - t_start_global
    
    global_probs = get_gnn_predictions(global_model, data)
    global_preds = global_probs.argmax(dim=-1).cpu().numpy()
    global_test_metrics = evaluate(data.y, global_preds, data.test_mask)
    print(f"Global GNN Accuracy (Test): {global_test_metrics['accuracy']:.4f} | Time: {t_total_global:.2f}s")

    print(f"--- Phase 5: Targeted Hybrid GNN Training ---")
    sub_data, sub_node_mask, node_list = prepare_pyg_subgraph(data, subgraph_nodes)
    
    if args.filter:
        from hybrid_src.subgraph import filter_subgraph_edges
        print("Filtering subgraph noise...")
        sub_data = filter_subgraph_edges(sub_data, partition)
        
    print(f"Subgraph size: {sub_data.num_nodes} nodes, {sub_data.edge_index.size(1)} edges")
    sub_data.comm_ids = all_comm_ids[node_list]
    
    sub_confidences = torch.tensor(confidences[node_list], dtype=torch.float)
    node_weights = (1.0 - sub_confidences) + 0.1 
    
    hybrid_gnn_model = get_model(args.gnn_type, data.num_features, args.hidden_channels, num_classes, 
                               num_communities=num_communities)
    
    supervision_mask = None
    if args.targeted:
        print("Targeting supervision to hard nodes only...")
        supervision_mask = torch.tensor(hard_node_mask[node_list])
        
    t_start_hybrid = time.time()
    hybrid_gnn_model = train_gnn(hybrid_gnn_model, sub_data, 
                               supervision_mask=supervision_mask, 
                               node_weights=node_weights)
    t_total_hybrid = time.time() - t_start_hybrid
    
    gnn_probs = get_gnn_predictions(hybrid_gnn_model, sub_data)
    print(f"Hybrid GNN Training | Time: {t_total_hybrid:.2f}s")
    
    print(f"--- Phase 6: Predicted Fusion ---")
    if args.fusion == "hard":
        from hybrid_src.fusion import merge_predictions, boundary_smoothing
        gnn_label_preds = gnn_probs.argmax(dim=-1).cpu().numpy()
        final_preds = merge_predictions(louvain_preds, gnn_label_preds, hard_nodes, node_list)
        
        print("Applying Confidence-Boundary Smoothing...")
        final_preds = boundary_smoothing(G, final_preds, confidences, iterations=1)
    else:
        from hybrid_src.baseline import get_community_label_distribution
        baseline_probs = get_community_label_distribution(partition, data.y, data.train_mask, num_classes)
        final_preds = soft_fusion(louvain_preds, gnn_probs, confidences, node_list, num_classes, data.num_nodes, 
                                 baseline_probs=baseline_probs)
    
    print(f"--- Phase 7: Diagnostic Evaluation ---")
    total_test_nodes = data.test_mask.sum().item()
    test_hard_mask = (data.test_mask.cpu().numpy() & hard_node_mask)
    test_easy_mask = (data.test_mask.cpu().numpy() & ~hard_node_mask)
    
    # ADVANCED FIX: Calibration Analysis
    from hybrid_src.eval import calibration_analysis
    cal_res = calibration_analysis(louvain_preds, data.y.cpu().numpy(), confidences, mask=data.test_mask.cpu().numpy())
    print(f"Confidence/Error Correlation: {cal_res['correlation']:.4f}")
    
    hybrid_test_metrics = evaluate(data.y, final_preds, data.test_mask)
    
    # Split accuracies (Hybrid)
    res_easy = evaluate(data.y, final_preds, test_easy_mask)
    res_hard = evaluate(data.y, final_preds, test_hard_mask)
    
    # Split accuracies (Baseline)
    base_res_easy = evaluate(data.y, louvain_preds, test_easy_mask)
    base_res_hard = evaluate(data.y, louvain_preds, test_hard_mask)
    
    print(f"Hybrid Accuracy (Total Test): {hybrid_test_metrics['accuracy']:.4f}")
    print(f"Global GNN Accuracy: {global_test_metrics['accuracy']:.4f}")
    if test_easy_mask.any():
        print(f"Easy nodes (Baseline: {base_res_easy['accuracy']:.4f} -> Hybrid: {res_easy['accuracy']:.4f})")
    if test_hard_mask.any():
        print(f"Hard nodes (Baseline: {base_res_hard['accuracy']:.4f} -> Hybrid: {res_hard['accuracy']:.4f})")
        
    results = {
        "metrics": {
            "baseline": baseline_test_metrics,
            "global_gnn": global_test_metrics,
            "hybrid": hybrid_test_metrics,
            "easy_split": {"baseline": base_res_easy, "hybrid": res_easy},
            "hard_split": {"baseline": base_res_hard, "hybrid": res_hard},
            "calibration": cal_res
        },
        "stats": {
            "num_hard": len(hard_nodes),
            "subgraph_size": sub_data.num_nodes,
            "subgraph_edges": sub_data.edge_index.size(1),
            "targeted": args.targeted,
            "adaptive": args.adaptive,
            "filtered": args.filter,
            "t_global_train": t_total_global,
            "t_hybrid_train": t_total_hybrid
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
