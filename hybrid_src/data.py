import os
import torch
import functools

# Monkeypatch torch.load for OGB compatibility in PyTorch 2.6+
_original_load = torch.load
@functools.wraps(_original_load)
def _permissive_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _permissive_load

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import to_networkx

def load_dataset(name="Cora", root="dataset"):
    """Loads Planetoid, Amazon, Coauthor or OGB datasets."""
    if name.lower().startswith("ogbn"):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=root)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[split_idx['valid']] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[split_idx['test']] = True
        data.y = data.y.squeeze()
        num_classes = dataset.num_classes
    elif name.lower() in ["computers", "photo"]:
        dataset = Amazon(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
        # Amazon datasets dont have splits, create random ones
        indices = torch.randperm(data.num_nodes)
        val_size = int(data.num_nodes * 0.1)
        test_size = int(data.num_nodes * 0.2)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[indices[:val_size]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[val_size:val_size+test_size]] = True
        data.train_mask = ~(data.val_mask | data.test_mask)
    elif name.lower() in ["cs", "physics"]:
        dataset = Coauthor(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
        # Coauthor datasets dont have splits
        indices = torch.randperm(data.num_nodes)
        val_size = int(data.num_nodes * 0.1)
        test_size = int(data.num_nodes * 0.2)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[indices[:val_size]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[val_size:val_size+test_size]] = True
        data.train_mask = ~(data.val_mask | data.test_mask)
    else:
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
    
    G = to_networkx(data, to_undirected=True)
    return G, data, num_classes
