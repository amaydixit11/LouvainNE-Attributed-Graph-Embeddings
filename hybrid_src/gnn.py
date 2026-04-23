import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_communities=None, comm_dim=8):
        super().__init__()
        self.use_comm = num_communities is not None
        if self.use_comm:
            self.comm_emb = torch.nn.Embedding(num_communities, comm_dim)
            in_channels += comm_dim
            
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, comm_ids=None):
        if self.use_comm and comm_ids is not None:
            c = self.comm_emb(comm_ids)
            x = torch.cat([x, c], dim=-1)
            
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_communities=None, comm_dim=8):
        super().__init__()
        self.use_comm = num_communities is not None
        if self.use_comm:
            self.comm_emb = torch.nn.Embedding(num_communities, comm_dim)
            in_channels += comm_dim
            
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index, comm_ids=None):
        if self.use_comm and comm_ids is not None:
            c = self.comm_emb(comm_ids)
            x = torch.cat([x, c], dim=-1)
            
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_communities=None, comm_dim=8):
        super().__init__()
        self.use_comm = num_communities is not None
        if self.use_comm:
            self.comm_emb = torch.nn.Embedding(num_communities, comm_dim)
            in_channels += comm_dim
            
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, comm_ids=None):
        if self.use_comm and comm_ids is not None:
            c = self.comm_emb(comm_ids)
            x = torch.cat([x, c], dim=-1)
            
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def get_model(model_type, in_channels, hidden_channels, out_channels, num_communities=None, comm_dim=8):
    if model_type.lower() == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, num_communities, comm_dim)
    elif model_type.lower() == "gat":
        return GAT(in_channels, hidden_channels, out_channels, 8, num_communities, comm_dim)
    elif model_type.lower() == "sage":
        return GraphSAGE(in_channels, hidden_channels, out_channels, num_communities, comm_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_gnn(model, data, supervision_mask=None, epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Trains the GNN model on the given (subgraph) data.
    Ensures NO LEAKAGE by only using labels from the original train_mask.
    If supervision_mask is provided (e.g. hard nodes), it only trains on those nodes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Base training mask: must be in original train_mask
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        mask = data.train_mask
    else:
        print("WARNING: No train_mask found. Cannot train GNN safely.")
        return model

    # Targeted supervision: intersect with supervision_mask if provided
    if supervision_mask is not None:
        mask = mask & supervision_mask
        
    if mask.sum() == 0:
        print("WARNING: No supervised nodes found in subgraph. Model will remain untrained.")
        return model

    model.train()
    comm_ids = getattr(data, 'comm_ids', None)
    for epoch in range(epochs):
        optimizer.zero_grad()
        if comm_ids is not None:
            out = model(data.x, data.edge_index, comm_ids=comm_ids)
        else:
            out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        
    return model

@torch.no_grad()
def get_gnn_predictions(model, data):
    """Returns soft probabilities for all nodes in the given data object."""
    model.eval()
    comm_ids = getattr(data, 'comm_ids', None)
    if comm_ids is not None:
        out = model(data.x, data.edge_index, comm_ids=comm_ids)
    else:
        out = model(data.x, data.edge_index)
    return F.softmax(out, dim=-1)
