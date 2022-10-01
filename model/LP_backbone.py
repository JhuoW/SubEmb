from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.nn.norm import GraphNorm,GraphSizeNorm
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool


############## Link Prediction for Pretrain node feature ####################

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


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))  # 系数邻接矩阵
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()  # 图的度向量
    deg[deg < 0.5] += 1.0   # 对于没有邻居的节点，度设为1，即为没有邻居的节点添加self-loop
    if aggr == "mean":
        deg = 1.0 / deg  # D^-1 A
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":  # A
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)  # D^-1 A D^-1
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError


class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.  # 最后一层是否激活
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,  # 2
                 dropout=0,
                 tail_activation=False,   
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)

class MyGCNConv(nn.Module):    
    '''
    单层message passing 
    '''
    def __init__(self, in_channels, out_channels, act = nn.ReLU(inplace=True), aggr = "mean"):
        super(MyGCNConv, self).__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = act
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
    
    def reset_parameters(self) :
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()
    
    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_nodes = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_nodes, self.aggr)  # edge_index -> sparse normalization adj
        x = self.trans_fn(x_)  # feature transformation
        x = self.activation(x) 
        x = self.adj @ x   # Propagation
        x = self.gn(x)     # GraphNorm
        x = torch.cat((x, x_), dim=-1)  # residual GCN
        x = self.comb_fn(x) 
        return x

class MyGCN(nn.Module):
    # K layer MyGCNConv + GraphNorm
    def __init__(self,
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 K, 
                 num_attr, 
                 dropout = 0, 
                 act = nn.ReLU(inplace=True),
                 conv = MyGCNConv,
                 gn = True,
                 jk = False,
                 aggr = "mean"):
        super(MyGCN, self).__init__()
        # encode each attribute
        self.input_emb = nn.Embedding(num_attr, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        if K >1:
            self.convs.append(
                conv(in_channels, hidden_channels, aggr=aggr)
            )
            for _ in range(K-2):
                self.convs.append(conv(hidden_channels, hidden_channels, aggr = aggr))
            
            self.convs.append(conv(hidden_channels, out_channels, aggr = aggr))
        else:
            self.convs.append(conv(in_channels, out_channels, aggr = aggr))
        
        self.act = act
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(K-1):  # 层间norm
                self.gns.append(GraphNorm(hidden_channels))
            
        else:
            self.gns = None
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()


    def forward(self, x, edge_index, edge_weight):
        xs = []
        input_feature = self.input_emb(x.reshape(-1))  # x 为node id/ degree， 有相同node id/degree的节点被encode为相同的特征向量

        x = F.dropout(input_feature,p = self.dropout, training=self.training)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[i](x)
            
            xs.append(x)
            x = self.act(x)
            x =  F.dropout(x, p=self.dropout, training=self.training)
        
        xs.append(self.convs[-1](x, edge_index, edge_weight))  # 最后一层后不用bn和act
        if self.jk:
            return torch.cat(xs, dim = -1)
        else:
            return xs[-1]


class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv  # EmbGConv
        self.preds = preds
        self.pools = pools

    def nodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]): # [21521, 1, 1] 节点特征， 每个节点可以有multi features，第一位控制特征数量，对每个特征分别做卷积
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]), # x的shape从 [N, 1, D] -> [N,D]
                            edge_index, edge_weight)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))  # node embedding after num_layer MyGCNs
        emb = torch.cat(embs, dim=1) # multi feature cat [21521, 2, D]  # 每个节点2个 feature vectors
        emb = torch.mean(emb, dim=1)  # [21521, D] 
        return emb

    def pool(self, emb, pos_neg_edge, pool):
        emb = emb[pos_neg_edge]  # subG_node为一个batch的边
        emb = torch.mean(emb, dim=1)  # 边两端节点取平均得到边的emb
        return emb

    def forward(self, x, edge_index, edge_weight, pos_neg_edge, z=None, id=0):
        emb = self.nodeEmb(x, edge_index, edge_weight)  # Node embedding by num_layer MyGCNs
        emb = self.pool(emb, pos_neg_edge, self.pools[id])
        return self.preds[id](emb)
