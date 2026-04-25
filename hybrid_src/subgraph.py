import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def identify_hard_nodes(confidences, tau=0.75):
    """Returns indices of nodes with confidence below tau."""
    hard_nodes = [i for i, conf in enumerate(confidences) if conf < tau]
    return hard_nodes

def filter_subgraph_edges(sub_data, partition, similarity_threshold=0.5):
    """
    Noise reduction by edge filtering:
    Keep edge if:
    1. Both nodes in same community
    2. OR feature cosine similarity > threshold
    """
    import torch.nn.functional as F
    
    edge_index = sub_data.edge_index
    num_edges = edge_index.size(1)
    
    # We need access to original node IDs to check communities
    # Assuming sub_data.node_id exists (it doesn't by default, let's add it in preparation)
    if not hasattr(sub_data, 'orig_node_id'):
        print("WARNING: filter_subgraph_edges requires sub_data.orig_node_id. Skipping.")
        return sub_data
        
    orig_ids = sub_data.orig_node_id
    
    keep_mask = []
    
    # Batch similarity calculation for performance
    src, dst = edge_index
    src_feat = sub_data.x[src]
    dst_feat = sub_data.x[dst]
    # Cosine similarity
    sim = F.cosine_similarity(src_feat, dst_feat, dim=-1)
    
    for i in range(num_edges):
        u, v = src[i].item(), dst[i].item()
        u_orig = orig_ids[u].item()
        v_orig = orig_ids[v].item()
        
        # Check community
        same_comm = (partition[u_orig] == partition[v_orig])
        
        if same_comm or sim[i] > similarity_threshold:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    sub_data.edge_index = edge_index[:, keep_mask]
    return sub_data

def get_adaptive_subgraph_nodes(G, hard_nodes, confidences, threshold=0.6):
    """
    Adaptive expansion: 
    - if confidence < threshold, use k=2
    - else use k=1
    """
    S = set(hard_nodes)
    for node in hard_nodes:
        k = 2 if confidences[node] < threshold else 1
        S.update(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())
    return S

def prepare_pyg_subgraph(data, subgraph_nodes):
    """
    Creates a PyG subgraph data object from selected nodes.
    Uses data.subgraph() which handles node re-indexing and edge filtering.
    """
    # Convert set to list for indexing
    node_list = sorted(list(subgraph_nodes))
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_mask[node_list] = True
    
    subgraph_data = data.subgraph(node_mask)
    # Store original IDs for mapping (needed for filtering)
    subgraph_data.orig_node_id = torch.tensor(node_list, dtype=torch.long)
    return subgraph_data, node_mask, node_list
