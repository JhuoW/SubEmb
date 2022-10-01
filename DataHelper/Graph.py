from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, to_networkx, degree
import torch
import torch.nn.functional as F

class Graph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_labels, mask, pos):
        # pos: 每个子图所包含的节点  [num_subgraphs, num_nodes_in_each_subgraph]
        super(Graph, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_weight, y=subG_labels, pos=pos)
        self.mask = mask
        self.subG_labels = subG_labels
        self.to_undirected()
        

    def feat_Is_Degree(self, mod = 1, one_hot = False):
        '''use node degree as node feature'''
        adj    = torch.sparse_coo_tensor(self.edge_index, self.edge_attr, (self.num_nodes, self.num_nodes))
        degree = torch.sparse.sum(adj, dim = 1).to_dense().to(torch.int64)   # undirected
        degree.div_(mod, rounding_mode='floor')  
        degree = torch.unique(degree, return_inverse = True)[1]  # 相当于吧degree 从0开始re-index，返回每个节点的index
        if not one_hot:
            self.x = degree.reshape(self.num_nodes, 1, -1)
        else:
            self.x = F.one_hot(degree, num_class = -1).reshape(self.num_nodes,1,-1)
    
    def feat_Is_One(self):
        self.x = torch.ones((self.num_nodes,1,1),dtype=torch.int64)   #

    def feat_Is_NodeID(self, one_hot = False):
        self.x = torch.arange(self.num_nodes,dtype=torch.int64).reshape(self.num_nodes, 1, -1) if not one_hot else F.one_hot(torch.arange(self.num_nodes,dtype=torch.int64)).reshape(self.num_nodes, 1, -1) 
                 
    def to_undirected(self):
        if not is_undirected(self.edge_index): # 有向图
            self.edge_index, self.edge_attr = to_undirected(self.edge_index, self.edge_attr)

    def set_num_attr(self, num_attr):
        self.num_attr = num_attr
    
    def set_num_nodes(self):
        self.num_nodes = self.x.shape[0]

    def get_LPdataset(self, use_loop=False):
        neg_edge_index = negative_sampling(self.edge_index)  # negative edges 为每个positive边采样neg边 即positive edges和negative edges 数量相同
        pos_edge_index = self.edge_index
        pos_neg_edges = torch.cat((pos_edge_index, neg_edge_index), dim = 1).t()
        y = torch.cat((torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1]))).cuda()
        return self.x, pos_edge_index, self.edge_attr, pos_neg_edges, y


    # def get_split(self):
       

