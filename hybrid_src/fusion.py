import numpy as np
import torch

def merge_predictions(louvain_preds, gnn_label_preds, hard_nodes, node_list):
    """
    Hard switch: Final[node] = GNN[node] if node is hard else Louvain[node]
    
    gnn_label_preds: array of labels for nodes in node_list
    node_list: mapping from subgraph index to original index
    """
    final_preds = louvain_preds.copy()
    
    # Map subgraph predictions back to original node IDs
    gnn_full_map = {}
    for i, orig_idx in enumerate(node_list):
        gnn_full_map[orig_idx] = gnn_label_preds[i]
        
    for node in hard_nodes:
        if node in gnn_full_map:
            final_preds[node] = gnn_full_map[node]
            
    return final_preds

def soft_fusion(louvain_preds, gnn_probs, confidences, node_list, num_classes, num_nodes, baseline_probs=None):
    """
    Adaptive Fusion: Final = w * GNN + (1-w) * Louvain
    where w = (1 - confidence)^2  (non-linear weighting)
    """
    if baseline_probs is None:
        louvain_probs = np.zeros((num_nodes, num_classes))
        for i, p in enumerate(louvain_preds):
            louvain_probs[i, int(p)] = 1.0
    else:
        louvain_probs = baseline_probs
        
    full_gnn_probs = np.zeros((num_nodes, num_classes))
    for i, orig_idx in enumerate(node_list):
        full_gnn_probs[orig_idx] = gnn_probs[i].cpu().numpy()
        
    # NON-LINEAR TRANSFORMATION for confidence
    # Small uncertainty -> keep baseline. Large uncertainty -> aggressively move to GNN.
    uncertainty = 1.0 - confidences
    w = uncertainty ** 2 
    
    subgraph_mask = np.zeros(num_nodes)
    subgraph_mask[node_list] = 1.0
    w = w * subgraph_mask 
    
    w = w.reshape(-1, 1)
    combined_probs = w * full_gnn_probs + (1.0 - w) * louvain_probs
    return combined_probs.argmax(axis=-1)
def boundary_smoothing(G, final_preds, confidences, iterations=1):
    """
    Smoothing at the decision boundary.
    Only nodes in the 'Transition Zone' (medium confidence) update their labels
    based on their neighbors' consensus.
    """
    import networkx as nx
    smoothed = final_preds.copy()
    
    # Identify nodes in the "Uncertainty Boundary" (0.4 - 0.7 confidence)
    boundary_nodes = np.where((confidences > 0.4) & (confidences < 0.7))[0]
    
    for _ in range(iterations):
        for node in boundary_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors: continue
            
            # Count neighbor labels
            neighbor_labels = [smoothed[nbr] for nbr in neighbors]
            counts = np.bincount(neighbor_labels)
            majority_neighbor = np.argmax(counts)
            
            # Simple soft update: if neighbor consensus is strong, adopt it
            if counts[majority_neighbor] / len(neighbors) > 0.6:
                smoothed[node] = majority_neighbor
                
    return smoothed
