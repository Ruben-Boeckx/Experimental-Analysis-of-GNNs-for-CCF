from torch_geometric.nn import SAGEConv, GCNConv, GATConv, Linear
import torch
import torch.nn.functional as F
from torch.nn import Linear

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

# Try GAT with more layers
class GAT_more_layers(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_channels, add_self_loops=False))
        self.lins.append(Linear(input_dim, hidden_channels))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.lins.append(Linear(hidden_channels, hidden_channels))
        
        # Last layer
        self.convs.append(GATConv(hidden_channels, out_channels, add_self_loops=False))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, lin in zip(self.convs[:-1], self.lins[:-1]):
            x = conv(x, edge_index) + lin(x)
            x = F.relu(x)
        
        # Last layer (no activation)
        x = self.convs[-1](x, edge_index) + self.lins[-1](x)
        
        return x