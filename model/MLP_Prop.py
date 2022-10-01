from distutils.command.config import config
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from utils.utils import pad2batch
import os.path as osp
from model.Backbone import MaxPool, MeanPool, AddPool, SizePool
from utils.utils import buildAdj
import functools
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

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
                 tail_activation=False,     # 最后一层Norm
                 activation=nn.ReLU(inplace=True),
                 gn=False):  # 每层Norm
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


class MLP_Model(nn.Module):
    def __init__(self, config, 
                       num_attr, 
                       output_dim) :
        super(MLP_Model, self).__init__()
        self.config          = config
        feat_dim         = config['feat_dim']
        hidden_dim       = config['hidden_dim'] 
        num_layers       = config['num_layers'] 
        input_gn         = config['input_gn'] 
        input_dropout    = config['input_dropout']
        dropout          = config['dropout'] 
        pool_fn          = config['pool'] 
        num_pred_layers  = config['num_pred_layers']
        tail_act         = config['tail_act']
        act              = nn.ReLU(inplace=True)
        gn               = config['gn']

        self.input_feat      = nn.Embedding(num_attr, feat_dim, scale_grad_by_freq=False)  # 
        self.input_gn        = input_gn
        if input_gn:
            self.input_gn  = GraphNorm(feat_dim)
        self.input_dropout = input_dropout
        self.mlp  = MLP(feat_dim, hidden_dim, hidden_dim, num_layers, dropout, tail_act, act, gn)

        self.node_reps = None
        self.reset_parameters() 
        pool_fn_fn = {
            "mean": MeanPool,
            "max": MaxPool,
            "sum": AddPool,
            "size": SizePool
        }
        if pool_fn in pool_fn_fn:
            self.pool_fn = pool_fn_fn[pool_fn]()
        else:
            raise NotImplementedError

        self.pred_head = MLP(hidden_dim, hidden_dim, output_dim, num_pred_layers, dropout)



    def reset_parameters(self):
        self.input_feat.reset_parameters()
        if self.input_gn:
            self.input_gn.reset_parameters()
        # self.mlp.reset_parameters()

    def Pool(self, emb, batch, nodes_in_subG, pool_func):
        emb = emb[nodes_in_subG]
        sub_emb = pool_func(emb, batch)
        return sub_emb

    def load_pretrain(self):
        if self.config['node_feat_type'] == 'node_id':
            pretrained_emb = torch.load(osp.join("GLASS_pretrain",f"{self.config['dataset']}_{self.config['hidden_dim']}.pt"),
                                            map_location=torch.device('cpu')).detach()
            self.input_feat = nn.Embedding.from_pretrained(pretrained_emb, freeze=False).cuda()
        return

    def init_node_feat(self, x, num_nodes):
        x                = self.input_feat(x).reshape(num_nodes, -1)
        if self.input_gn:
            x = self.input_gn(x)
        
        return x

    def get_node_embeddings(self, x):
        x = F.dropout(x, p = self.input_dropout, training=self.training)
        node_emb = self.mlp(x)
        return node_emb


    def forward(self, x, edge_index, batch, nodes_in_subG, graph):
        """
        subG_nodes: subG_node = tpos: [[],[],[],...] batch内每个子图的节点  被padding -1
        nodes_in_subG: 每个子图中的节点[1,2,5,1,3,5]
        batch: 每个节点所在的子图      [0,0,1,1,1,2]
        """

        # initialize node feature
        x = self.init_node_feat(x, graph.num_nodes)

        out = self.get_node_embeddings(x)

        sub_embedding = self.Pool(out, batch, nodes_in_subG, self.pool_fn)

        pred_res = self.pred_head(sub_embedding)

        
        return pred_res




class MLP_Prop(nn.Module):
    def __init__(self, config, num_layers, input_dim, output_dim, use_input, residual = None, dropout = 0, num_pred_layers = 2):
        super(MLP_Prop, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.use_input = use_input
        
        pool_fn          = config['pool'] 
        if residual == 'sum' or residual =='mean' or residual is None:
            self.input_dim = input_dim
        elif residual == 'cat' and use_input:
            self.input_dim = input_dim * num_layers + input_dim
        elif residual == 'cat' and (not use_input):
            self.input_dim = input_dim * num_layers


        self.residual_func = {'sum': functools.partial(torch.sum, dim=0),
                              'mean': functools.partial(torch.mean, dim=-1),
                              'cat': functools.partial(torch.cat, dim=-1)}
        pool_fn_fn = {
                    "mean": MeanPool,
                    "max": MaxPool,
                    "sum": AddPool,
                    "size": SizePool
                }
        if pool_fn in pool_fn_fn:
            self.pool_fn = pool_fn_fn[pool_fn]()
        else:
            raise NotImplementedError

        self.out_head = MLP(self.input_dim, config['hidden_dim'], config['hidden_dim'], num_pred_layers, dropout=dropout)
        self.pred_head = MLP(config['hidden_dim'], output_dim, output_dim, 1, dropout=dropout)

    def Pool(self, emb, batch, nodes_in_subG):
        emb = emb[nodes_in_subG]
        sub_emb = self.pool_fn(emb, batch)
        return sub_emb


    def forward(self, xs, adj, batch, nodes_in_subG):

        if self.residual is None:
            out = self.pred_head(xs[-1])
        elif self.residual == 'cat':
            xss = self.residual_func[self.residual](xs)
            out = self.out_head(xss)
        elif self.residual == 'sum' or self.residual == 'mean':
            xss = self.residual_func[self.residual](torch.stack(xs, dim=0))  # 21521x64
            out = self.out_head(xss)
        else:
            raise NotImplementedError
 
        
        sub_emb = self.Pool(out, batch, nodes_in_subG)
        pred_res = self.pred_head(sub_emb)
        return pred_res



def train_prop(model_prop, graph, train_loader, optimizer, propagated_emb, adj, loss_func):
    model_prop.train(True)
    graph = graph.cuda()
    total_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_labels = batch[-3]
        subG_nodes = batch[3]
        batch_idx, nodes_in_subG  =  pad2batch(subG_nodes)
        out = model_prop(propagated_emb, adj, batch_idx, nodes_in_subG)
        loss = loss_func(out, batch_labels)
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    loss = sum(total_loss) / len(total_loss)
    return model_prop, float(loss)

@torch.no_grad()
def mlp_prop_evaluation(model_prop, graph, loader, loss_func, propagated_emb, adj):
    model_prop.eval()
    preds = []
    ys    = []
    graph = graph.cuda()
    for batch in loader:
        labels = batch[-3]
        subG_nodes = batch[3]
        batch_idx, nodes_in_subG  =  pad2batch(subG_nodes)
        out = model_prop(propagated_emb, adj, batch_idx, nodes_in_subG)
        preds.append(out)
        ys.append(labels)
    
    preds_ = torch.cat(preds, dim=0)
    y_     = torch.cat(ys, dim=0)
    
    prediction = np.argmax(preds_.detach().cpu().numpy(), axis=1)
    micro = f1_score(y_.detach().cpu().numpy(), prediction, average = "micro")
    macro = f1_score(y_.detach().cpu().numpy(), prediction, average = "macro")
    loss = loss_func(preds_, y_)
    return {"micro": round(micro * 100 , 2) , "macro": round(macro * 100 , 2)}, float(loss)


def propagate(base_emb, adj, num_layers, use_input):
    xs = []
    xs.append(base_emb)
    for _ in range(num_layers):
        xs.append(torch.spmm(adj, xs[-1]))

    return xs if use_input else xs[1:]


def train_MLP_Prop(mlp_model, graph, config, output_dim, loss_func, train_loader, val_loader, test_loader):
    
    with torch.no_grad():
        input_node_feat = mlp_model.init_node_feat(graph.x, graph.num_nodes)
        base_emb        = mlp_model.get_node_embeddings(input_node_feat)
        adj             = buildAdj(graph.edge_index, graph.edge_attr, graph.num_nodes, aggr=config['adj_type'])

    propagated_emb = propagate(base_emb, adj, num_layers = config['prop_layers'], use_input=config['use_input'])

    model_prop = MLP_Prop(config,
                            num_layers=config['prop_layers'],
                            input_dim= base_emb.shape[1], 
                            output_dim = output_dim, 
                            use_input=config['use_input'], 
                            residual=config['residual'], 
                            dropout=config['prop_dropout'], 
                            num_pred_layers=config['pred_head_layers']).cuda()
                            
    optimizer = torch.optim.Adam(model_prop.parameters(), lr=0.0005, weight_decay=5e-4)
    pbar_mlp_prop       = tqdm(range(config['prop_epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    dev_result          = {"micro": 0 ,"macro": 0}
    maj_metric          = "micro"
    best_dev            = 0
    best_metric_epoch   = -1
    report_dev_res      = {"micro": 0 ,"macro": 0}
    test_result         = {"micro": 0 ,"macro": 0}
    report_tes_res      = {"micro": 0 ,"macro": 0}
    dev_loss            = 0
    patience_cnt        = 0

    for epoch in pbar_mlp_prop:
        model_prop, train_loss = train_prop(model_prop, graph, train_loader, optimizer, propagated_emb, adj, loss_func)

        # val/ test
        dev_result, dev_loss   = mlp_prop_evaluation(model_prop, graph, val_loader, loss_func, propagated_emb, adj)
        test_result, test_loss = mlp_prop_evaluation(model_prop, graph, test_loader,loss_func, propagated_emb, adj)

        now_dev = dev_result[maj_metric]

        if now_dev >= best_dev:
            best_dev          = now_dev
            best_metric_epoch = epoch
            report_dev_res    = dev_result
            report_tes_res    = test_result
            patience_cnt      = 0
        else:
            patience_cnt      += 1

        if config['patience_prop'] > 0 and patience_cnt >= config['patience_prop']:
            break

        postfix_str = "<Epoch %d> [Train Loss] %.4f [Curr Dev Acc] %.2f <Best Epoch %d> [Best Dev Acc] %.2f [Test] %.2f ([Report Test] %.2f) " % ( 
                        epoch ,      train_loss,    dev_result[maj_metric], best_metric_epoch ,report_dev_res[maj_metric], test_result[maj_metric], report_tes_res[maj_metric])
        
        pbar_mlp_prop.set_postfix_str(postfix_str)
    
    return model_prop, report_dev_res, report_tes_res, train_loss






            
        

        

