import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge

class StableHybridGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, num_communities=None, comm_dim=16):
        super().__init__()
        self.use_comm = num_communities is not None
        if self.use_comm:
            self.comm_emb = torch.nn.Embedding(num_communities + 1, comm_dim)
            in_channels += comm_dim
            
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.jk = JumpingKnowledge("max")
        self.post_lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, comm_ids=None):
        if self.use_comm and comm_ids is not None:
            c = self.comm_emb(comm_ids)
            x = torch.cat([x, c], dim=-1)
            
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            xs.append(x)
            
        x = self.jk(xs)
        return self.post_lin(x)

def get_model(model_type, in_channels, hidden_channels, out_channels, num_communities=None, comm_dim=16):
    return StableHybridGNN(in_channels, hidden_channels, out_channels, num_communities=num_communities)

def train_gnn(model, data, supervision_mask=None, node_weights=None, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        return model

    mask = data.train_mask
    if supervision_mask is not None:
        mask = mask & supervision_mask
        
    if mask.sum() == 0:
        return model

    comm_ids = getattr(data, 'comm_ids', None)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, comm_ids=comm_ids)
        loss = F.cross_entropy(out[mask], data.y[mask]) # Simplified for stability
        loss.backward()
        optimizer.step()
        
    return model

@torch.no_grad()
def get_gnn_predictions(model, data):
    model.eval()
    comm_ids = getattr(data, 'comm_ids', None)
    out = model(data.x, data.edge_index, comm_ids=comm_ids)
    return F.softmax(out, dim=-1)
