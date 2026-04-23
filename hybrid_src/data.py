import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import os

def load_dataset(name="Cora", root="dataset"):
    """Loads a Planetoid dataset and returns G (NetworkX) and data (PyG)."""
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    
    # Convert to NetworkX for Louvain and other graph ops
    # Note: data.edge_index is the source of truth
    G = to_networkx(data, to_undirected=True)
    
    # Add labels to nodes in NetworkX for easier access in baseline
    for i in range(data.num_nodes):
        G.nodes[i]['label'] = data.y[i].item()
        
    return G, data, dataset.num_classes
