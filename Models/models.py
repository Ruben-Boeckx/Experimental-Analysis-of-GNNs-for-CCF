from pprint import pprint
from typing import Tuple
import torch 
import torch.nn as nn
from models.GNNs import get_GNN
from decoder.decoder import get_decoder

# Copied from Tiukhova paper, will probably need to be modified
class GNN_only(Model): 
    def __init__(self, GNN,DECODER, gnn_input_dim, gnn_embedding_dim , gnn_output_dim,gnn_layers,heads, dropout_rate, edge_dim ,eps ,train_eps,search_depth,**kwargs  ) -> None:
        super().__init__()
        gnn_kw = {
            'embedding_dim': gnn_embedding_dim, 
            'input_dim' : gnn_input_dim,
            'n_layers' : gnn_layers,
            'heads' : heads,
            'dropout_rate': dropout_rate,
            'edge_dim' : edge_dim,
            'output_dim' : gnn_output_dim,
            'train_eps' : train_eps,
            'eps':  eps ,
            'search_depth':search_depth
        }
        self.GNN = get_GNN(GNN)(**gnn_kw)
        self.decoder = get_decoder(DECODER)(gnn_output_dim)

    def forward_call(self, data, device): 
        labs = data.y
        emb = self.GNN(torch.Tensor(data.x).float().to(device), torch.Tensor(data.edge_index).type(torch.int64).to(device),torch.tensor(data.edge_attr).float().to(device))
        scores = self.decoder(emb.to(device))
        h0 = None
        synth_index = []
        return scores,torch.Tensor(labs).to(device),h0,synth_index

    def forward(self,month, data_dict, device,h0=None, train = False):
        assert type(month) == int, 'CANNOT USE WINDOWS WITH ONLY GNN'
        return self.forward_call(data_dict[month], device)