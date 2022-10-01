from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm

class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args: 
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)    


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)

class GCN(torch.nn.Module):
    def __init__(self,config, hidden_channels, num_features, num_classes):
        super().__init__()
        self.num_layers = config['num_layers']
        self.net_layers = nn.ModuleList([
            GCNConv(num_features, hidden_channels)]
        )
        self.net_layers.append(GCNConv(hidden_channels, num_classes))
        self.dropout = config['dropout']
        self.activation = getattr(F, config['activation'])

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = self.net_layers[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.net_layers[-1](x, edge_index)
        return x




class APPNP_Net(nn.Module):
    def __init__(self, in_channels,hid_channels, out_channels, act = nn.ReLU(), alpha = 0.1, dropout = False, num_layers = 3, K = 3):
        super(APPNP_Net, self).__init__()
        self.dropout =  dropout
        self.mlp1 = nn.Linear(in_channels, hid_channels)
        self.mlp2 = nn.Linear(hid_channels,out_channels)
        self.conv1 = APPNP(K=K, alpha=alpha)
        self.act = act

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
    
    def forward(self, x_, edge_index, edge_weight = None):
        x = F.dropout(x_, p=self.dropout, training=self.training)
        x = self.act(self.mlp1(x))
        x = F.dropout(x_, p=self.dropout, training=self.training)
        x = self.mlp2(x)
        self.conv1(x, edge_index)
        return x

# GIN