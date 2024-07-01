import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, Linear
from typing import Union, Dict


class GraphSAGE1(torch.nn.Module):
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 sage_aggr: str):
        super(GraphSAGE1, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.sage_aggr = sage_aggr
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        
        self.sage_layers.append(SAGEConv(in_channels, hidden_dim, aggr=sage_aggr))
        for _ in range(num_layers - 2):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr))
        self.sage_layers.append(SAGEConv(hidden_dim, embedding_dim, aggr=sage_aggr))

        self.out = Linear(embedding_dim, output_dim)

    def forward(self, x_dict, edge_index_dict):
        out = x_dict
        for i, layer in enumerate(self.sage_layers):
            out = {key: layer(out[key], edge_index_dict[key]) for key in out.keys()}
            if i != len(self.sage_layers) - 1:
                out = {key: F.relu(value) for key, value in out.items()}
                out = {key: self.dropout(value) for key, value in out.items()}
        
        out_transaction = self.out(out['transaction'])
        return out_transaction

class GraphSAGE2(torch.nn.Module):
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 sage_aggr: str):
        super(GraphSAGE2, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.sage_aggr = sage_aggr
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.sage1 = SAGEConv((-1,-1), embedding_dim, aggr=sage_aggr)
        else:
            self.sage1 = SAGEConv((-1,-1), hidden_dim, aggr=sage_aggr)
            for _ in range(num_layers-2):
                self.sage_layers.append(SAGEConv((-1,-1), hidden_dim, aggr=sage_aggr))
            self.sage2 = SAGEConv((-1,-1), embedding_dim, aggr=sage_aggr)

        self.out = Linear(embedding_dim, output_dim)

        """
        self.sage_layers.append(SAGEConv((in_channels, in_channels), hidden_dim, aggr=sage_aggr))
        for _ in range(num_layers - 2):
            self.sage_layers.append(SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr=sage_aggr))
        self.sage_layers.append(SAGEConv(hidden_dim, embedding_dim, aggr=sage_aggr))
        """

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.sage_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.sage2(h, edge_index)
        out = self.out(h)
        
        return out, h
    
class GraphSAGE3(torch.nn.Module):
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 sage_aggr: str):
        super(GraphSAGE3, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.sage_aggr = sage_aggr
        self.num_layers = num_layers
        self.sage_layers = torch.nn.ModuleList()
        
        if isinstance(in_channels, int):
            self.sage_layers.append(SAGEConv(in_channels, hidden_dim, aggr=sage_aggr))
        elif isinstance(in_channels, dict):
            self.sage_layers.append(SAGEConv((-1, -1), hidden_dim, aggr=sage_aggr))
        else:
            raise ValueError("in_channels must be either int or dict")

        for _ in range(num_layers - 2):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr))
        self.sage_layers.append(SAGEConv(hidden_dim, embedding_dim, aggr=sage_aggr))

        self.out = Linear(embedding_dim, output_dim)
    
    def forward(self, x, edge_index):
        for conv in self.sage_layers:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
        x = self.out(x)

        return x

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