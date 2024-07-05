import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, Linear, GATConv, GATv2Conv

class GraphSAGE2(torch.nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 sage_aggr: str):
        super(GraphSAGE2, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.sage_aggr = sage_aggr
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.sage1 = SAGEConv((-1, -1), embedding_dim, aggr=sage_aggr)
        else:
            self.sage1 = SAGEConv((-1, -1), hidden_dim, aggr=sage_aggr)
            for _ in range(num_layers - 2):
                self.sage_layers.append(SAGEConv((-1, -1), hidden_dim, aggr=sage_aggr))
            self.sage2 = SAGEConv((-1, -1), embedding_dim, aggr=sage_aggr)

        self.out = Linear(embedding_dim, output_dim)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.num_layers > 1:
            for layer in self.sage_layers:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.sage2(h, edge_index)
        out = self.out(h)
        
        return out
    
class GAT2(torch.nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 heads: int):
        super(GAT2, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.heads = heads
        self.gat_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.gat1 = GATv2Conv((-1, -1), embedding_dim, heads=heads, add_self_loops=False)
            self.skip1 = Linear(-1, embedding_dim)
        else:
            self.gat1 = GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False)
            self.skip1 = Linear(-1, hidden_dim)
            for _ in range(num_layers - 2):
                self.gat_layers.append(GATv2Conv((-1, -1), hidden_dim, heads=heads, add_self_loops=False))
                self.skip_layers.append(Linear(-1, hidden_dim))
            self.gat2 = GATv2Conv((-1, -1), embedding_dim, heads=heads, add_self_loops=False)
            self.skip2 = Linear(-1, embedding_dim)

        self.out = Linear(embedding_dim, output_dim)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index) + self.skip1(x)
        h = F.relu(h)
        h = self.dropout(h)
        if self.num_layers > 1:
            for gat_layer, skip_layer in zip(self.gat_layers, self.skip_layers):
                h_new = gat_layer(h, edge_index) + skip_layer(h)
                h = F.relu(h_new)
                h = self.dropout(h)
            h = self.gat2(h, edge_index) + self.skip2(h)
        out = self.out(h)
        
        return out

class GAT3(torch.nn.Module):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 heads: int):
        super(GAT3, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.heads = heads
        self.gat_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.gat_layers.append(GATv2Conv(in_dim, embedding_dim // heads, heads=heads, concat=True, add_self_loops=False))
        else:
            self.gat_layers.append(GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True, add_self_loops=False))
            for _ in range(num_layers - 2):
                self.gat_layers.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, add_self_loops=False))
            self.gat_layers.append(GATv2Conv(hidden_dim, embedding_dim // heads, heads=heads, concat=True, add_self_loops=False))

        self.out = Linear(embedding_dim, output_dim)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        x = F.elu(x)
        out = self.out(x)
        
        return out

# GraphSAGE for node classification, OK
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Graph Attention Network for node classification, OK
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x