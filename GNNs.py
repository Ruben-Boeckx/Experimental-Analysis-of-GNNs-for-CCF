from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import torch

# GraphSAGE for node classification
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
    
# Graph Convolutional Network for node classification
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob):
        super(GCN, self).__init__()
        self.conv1 = GCNConv((-1, -1), hidden_channels)
        self.conv2 = GCNConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Graph Attention Network for node classification
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads)
        self.conv2 = GATConv((-1, -1), out_channels, heads=1)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x