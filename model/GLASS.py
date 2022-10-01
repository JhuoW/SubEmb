import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from utils.utils import pad2batch
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool
import os.path as osp

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
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError


class GLASSConv(nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    使用不同的参数来对 节点特征最变换
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,  # 0.95
                 dropout=0.2):  # 0.5
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask):
        '''
        x_: 整个图的node features
        edge_index: 整个图的edge_index
        mask: True 节点在batch子图中   False 节点不在batch子图中
        '''
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)  # D^-1 A
        # transform node features with different parameters individually.
        # 对node feature使用两个不同的Linear层分别做线性变换
        x1 = self.activation(self.trans_fns[1](x_))  # 节点充当batch子图内节点时的embedding
        x0 = self.activation(self.trans_fns[0](x_))  # 节点充当batch子图外节点时的embedding

        # mix transformed feature.  在message passing时区分子图内外的节点
        # 如果节点在batch 子图内，那么它的feature为self.z_ratio * x1 + (1 - self.z_ratio) * x0 
        # 如果节点在batch 子图外，那么它的feature为self.z_ratio * x0 + (1 - self.z_ratio) * x1
        # 因为z_ratio = 0.95， 所以如果节点是batch子图内，那么保留更多x1， 如果节点在batch子图外，保留更多x0, 这样就区分了labeled graph中batch子图内外的节点
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        # pass messages.
        x = self.adj @ x  
        x = self.gn(x)  # Graph Norm
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)   # jump knowledge 和初始特征拼接
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)  # residual后 新特征再做两个线性变换
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,   # 如果当前节点是batch子图内节点，那么保留更多x1, 如果当前节点是batch子图外节点，保留更多x0
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 hidden_channels,  # 64
                 output_channels,  # 64
                 num_layers,       # 2 
                 max_deg,          # 一共存在多少种特征
                 dropout=0,         # 0.5
                 activation=nn.ReLU(),   # nn.ELU(inplace=True)
                 conv=GLASSConv,
                 gn=True,
                 jk=False,
                 aggr = "mean",
                 z_ratio = 0.2):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=False)   # 每种feature编码 从预训练好的node embedding 获得
        self.emb_gn = GraphNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     aggr=aggr,
                     z_ratio=z_ratio,
                     dropout=dropout))
        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 aggr=aggr,
                     z_ratio=z_ratio,
                     dropout=dropout))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()
 
    def forward(self, x, edge_index, edge_weight, z=None):  # * 传入的是一个batch subgraphs 的 labeled graph,  
        # z is the node label.
        # x: node features  (num_nodes, 1)
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)  # 如果没有指示那些节点是子图总节点，那么所有节点都是True
        else:
            mask = (z > 0.5).reshape(-1, 1)       # batch子图中的节点为true, 不在batch子图中的节点为False
        # convert integer input to vector node features.  x.shape = [21521, 64]
        x = self.input_emb(x).reshape(x.shape[0], -1)  # node feature 为integer, 一共有max_deg+1中node feature， 将每个节点的node feature转为64维向量
        x = self.emb_gn(x)   # Graph Norm
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):   # 计算每个GLASSConv层的输出
            x = conv(x, edge_index, edge_weight, mask)  # 一层GLASSConv, 后会得到
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                 pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        '''
        x: 整个图所有节点的node features  每个节点的特征是一个 1xd的矩阵, 这里d=1
        z: 当前batch所对应的labeled graph, 存在于batch subgraphs中的节点labels为1, 不在的为0
        '''
        embs = []
        for _ in range(x.shape[1]):  # 遍历
            # x[:,_,:]: 节点特征 size: (num_nodes, 1)
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)  # [21521, 128]
        return emb

    def Pool(self, emb, subG_node, pool):   # pool = sum
        batch, pos = pad2batch(subG_node)  # pos为所有子图节点， batch为每个节点所在的子图
        emb = emb[pos]
        emb = pool(emb, batch)  # 每个子图内的节点做pooling
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        '''
        x: 所有节点node feature [21521, 1, 1]  每个节点的特征shape为(1,1)
        edge_index: 整个图的edge_index
        edge_weight: 每条边的edge_weight=edge_attr: 每条边的权重都是1
        subG_node = tpos: [[],[],[],...] batch内每个子图的节点 
        z: 当前batch subgraphs 所对应的labeled graph, [0,0,1,0,1,1,0,0,1,...]  在batch subgraphs中的节点label为1 不在的为0 len = num_nodes
        id=0
        '''
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)  # MLP 输出维度为有多少种label

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


class GLASSModel(object):
    def __init__(self, config, output_channel, num_attr):
        super(GLASSModel, self).__init__()
        # encode nodes , input features init from lookup nn.Embedding and then use Link pred pretrained node feature 
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout    = config['dropout']
        jk         = config['jk']
        z_ratio    = config['z_ratio']
        gn         = config['gn']
        act        = getattr(nn, config['act'])(inplace = True)
        aggr       = config['aggr']
        multilayer_GLASSConv = EmbZGConv(hidden_channels=hidden_dim,
                                    output_channels=hidden_dim,
                                    num_layers=num_layers, # num of GLASSConv layers
                                    max_deg=num_attr,
                                    activation=act,
                                    jk=jk,
                                    dropout=dropout,
                                    conv=GLASSConv,   # 每层都区分batch内外节点
                                    aggr=aggr,  # mean
                                    z_ratio=z_ratio,
                                    gn = gn)
        
        if config['node_feat_type'] == "node_id":
            print("load ", f"/pretrain_node_features/{config['dataset']}_{hidden_dim}.pt")
            pretrain_node_feat = torch.load(osp.join("GLASS_pretrain",f"{config['dataset']}_{hidden_dim}.pt"),
                                            map_location=torch.device('cpu')).detach()
            multilayer_GLASSConv.input_emb = nn.Embedding.from_pretrained(pretrain_node_feat, freeze=False)
        

        mlp = nn.Linear(hidden_dim * (num_layers) if jk else hidden_dim, output_channel)

        pool_fn_fn = {
            "mean": MeanPool,
            "max": MaxPool,
            "sum": AddPool,
            "size": SizePool
        }
        pool = config['pool']

        if pool in pool_fn_fn:
            pool_fn1 = pool_fn_fn[pool]()
        else:
            raise NotImplementedError
        
        self.model = GLASS(multilayer_GLASSConv, nn.ModuleList([mlp]), nn.ModuleList([pool_fn1])).cuda()

    