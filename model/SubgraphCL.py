import imp
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm





class SubGraphCL(nn.Module):
    def __init__(self, config, num_attr, feat_dim, hidden_dim, output_dim, input_gn):
        super(SubGraphCL, self).__init__()
        self.config     = config
        self.input_feat = nn.Embedding(num_attr, feat_dim, scale_grad_by_freq=False)  # 
        self.input_gn   = GraphNorm(feat_dim) 



    def forward(self, graph, subG_nodes, batch_nodes, batch_nodes_mask):
        '''
        subG_node: each node in batch
        '''
        # Base 
        x                = graph.x  # node id
        edge_index       = graph.edge_index
        edge_weight      = graph.edge_weight
        num_nodes        = graph.num_nodes

        # initialize node feature
        x                = self.input_feat(x).reshape(num_nodes, -1)
        x                = self.input_gn(x)

        # input gn
        if self.input_gn:
            x = self.input_gn(x)
        





        return 





class SubCLModel(object):
    def __init__(self, config, batch_subgraphs):


        pass

    

    def get_model(self, graph, subG_nodes, batch_nodes, batch_nodes_mask):
        return 