from re import search
import torch.nn as nn 
import torch 
from torch_geometric.nn import GCNConv, GATv2Conv,GINEConv,GraphSAGE, SAGEConv, GATConv
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F

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

# copied from Tiukhova paper, will probably need to be modified
class GATs(torch.nn.Module): 
    def __init__(self,input_dim, embedding_dim,output_dim,edge_dim,heads, n_layers, dropout_rate, **kwargs) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gat1 = GATv2Conv(input_dim, embedding_dim, heads=heads,edge_dim = edge_dim) # dim_h * num heads
        self.layer_norm1 = LayerNorm(embedding_dim*heads, elementwise_affine=True)
        self.GAT_list = torch.nn.ModuleList([GATv2Conv(embedding_dim*heads, embedding_dim, heads=heads,edge_dim = edge_dim)  for _ in range(n_layers-2)])
        self.gat2 = GATv2Conv(embedding_dim*heads, output_dim, heads=1, edge_dim = edge_dim)
        self.layer_norm2 = LayerNorm(output_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inp, edge_index,edge_feats):

        h = self.gat1(inp, edge_index,edge_attr = edge_feats)
        h = F.elu(h)
        h = self.dropout(h)
        for l in self.GAT_list:
            h = l(h, edge_index,edge_attr = edge_feats)
            h = self.layer_norm1(h)
            h = F.elu(h)
            h = self.dropout(h)
        h = self.gat2(h, edge_index, edge_feats)
        h = self.layer_norm2(h)
        h = F.elu(h)
        h = self.dropout(h)
        return h

def get_GNN(gnn ):
    
    if gnn == 'GAT':
        return GATs
    elif gnn == 'GIN':
        return GINs
    elif gnn == 'SAGE':
        return SAGEs
    else: 
        return GCNs