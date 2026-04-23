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
    Soft fusion: Final = w * GNN + (1-w) * Louvain
    where w = 1 - confidence
    """
    # If baseline_probs not provided, convert Louvain preds to one-hot
    if baseline_probs is None:
        louvain_probs = np.zeros((num_nodes, num_classes))
        for i, p in enumerate(louvain_preds):
            louvain_probs[i, int(p)] = 1.0
    else:
        louvain_probs = baseline_probs
        
    # Map GNN probs back to original indices
    full_gnn_probs = np.zeros((num_nodes, num_classes))
    for i, orig_idx in enumerate(node_list):
        full_gnn_probs[orig_idx] = gnn_probs[i].cpu().numpy()
        
    # Weights based on confidence
    w = 1.0 - confidences
    # For nodes NOT in the subgraph, we must use Louvain (w=0)
    subgraph_mask = np.zeros(num_nodes)
    subgraph_mask[node_list] = 1.0
    # Also, we might only want to blend nodes where GNN was supervised/targeted
    # But for now we trust the whole subgraph output
    w = w * subgraph_mask 
    
    w = w.reshape(-1, 1)
    
    combined_probs = w * full_gnn_probs + (1.0 - w) * louvain_probs
    return combined_probs.argmax(axis=-1)
