import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def identify_hard_nodes(confidences, tau=0.75):
    """Returns indices of nodes with confidence below tau."""
    hard_nodes = [i for i, conf in enumerate(confidences) if conf < tau]
    return hard_nodes

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
    return subgraph_data, node_mask, node_list
