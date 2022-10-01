# Code Adapted from https://github.com/Xi-yuanWang/GLASS
import imp
from .dataset_helper import dataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
import os.path as op
import torch
from torch.nn.utils.rnn import pad_sequence
import networkx as nx
from .Graph import Graph
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DatasetLocal(dataset):
    '''
    only hpo_neuro is multilable dataset
    '''
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    mask = None
    feat_transform = None
    recache = False

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def get_graph(self):
        return self.data


    def load(self,config):
        if config['feat_norm']:
            self.feat_transform = T.NormalizeFeatures()

        if self.dataset_name in ["coreness", "cut_ratio", "density", "component"]:
            # TODO
            obj = np.load(op.join(self.dataset_source_folder_path,'tmp.npy'), allow_pickle=True).item()
            pass
        elif self.dataset_name in ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user"]:
            self.multilabel = config['multilabel']
            if op.exists(op.join(self.dataset_source_folder_path, 'train_sub_G.pt')) and self.dataset_name != 'hpo_neuro':
                train_subGs       = torch.load(op.join(self.dataset_source_folder_path, 'train_subGs.pt'))
                train_subG_labels = torch.load(op.join(self.dataset_source_folder_path, 'train_subG_labels.pt'))
                val_subGs         = torch.load(op.join(self.dataset_source_folder_path, 'val_subGs.pt'))
                val_subG_labels   = torch.load(op.join(self.dataset_source_folder_path, 'val_subG_labels.pt'))
                test_subGs        = torch.load(op.join(self.dataset_source_folder_path, 'test_subGs.pt'))
                test_subG_labels  = torch.load(op.join(self.dataset_source_folder_path, 'test_subG_labels.pt'))
            else:
                train_subGs, train_subG_labels, val_subGs, val_subG_labels, test_subGs, test_subG_labels = self.read_subgraphs(
                    op.join(self.dataset_source_folder_path,'subgraphs.pth')
                )
                torch.save(train_subGs,       op.join(self.dataset_source_folder_path, 'train_subGs.pt'))
                torch.save(train_subG_labels, op.join(self.dataset_source_folder_path, 'train_subG_labels.pt'))
                torch.save(val_subGs,         op.join(self.dataset_source_folder_path, 'val_subGs.pt'))
                torch.save(val_subG_labels,   op.join(self.dataset_source_folder_path, 'val_subG_labels.pt'))
                torch.save(test_subGs,        op.join(self.dataset_source_folder_path, 'test_subGs.pt'))
                torch.save(test_subG_labels,  op.join(self.dataset_source_folder_path, 'test_subG_labels.pt'))

            mask = torch.cat((torch.zeros(len(train_subG_labels),dtype=torch.int64),
                              torch.ones (len(val_subG_labels)  ,dtype=torch.int64),
                              2*torch.ones(len(test_subG_labels),dtype=torch.int64)), dim=0)
            if self.multilabel:
                tlist     = train_subG_labels + val_subG_labels + test_subG_labels
                max_label = max([max(i) for i in tlist])  # 存在的label数
                label     = torch.zeros(len(tlist), max_label + 1)   # 每个子图的multilabel 用{0,1}向量表示
                for idx, ll in enumerate(tlist):
                    label[idx][torch.LongTensor(ll)] = 1  # one-hot like label vector [0,0,1,0,1,0,0] multi label
            else:
                label     = torch.cat((train_subG_labels, val_subG_labels, test_subG_labels))
            
            # 所有子图补到相同节点数
            pos           = pad_sequence([torch.tensor(i) for i in train_subGs + val_subGs + test_subGs], 
                                          batch_first=True, 
                                          padding_value=-1) # [num_subgraphs, num_dimensions] 子图的position matrix 用其所包含的节点来描述
            rawedge       = nx.read_edgelist(op.join(self.dataset_source_folder_path,'edge_list.txt')).edges
            edge_index    = torch.tensor([[int(i[0]), int(i[1])] for i in rawedge]).t()
            self.num_nodes     = max([torch.max(pos), torch.max(edge_index)]) + 1
            x             = torch.empty((self.num_nodes, 1, 0))
            edge_weight   = torch.ones(edge_index.shape[1])
            data =  Graph(x = x, edge_index= edge_index, edge_weight=edge_weight,pos = pos,subG_labels=label.to(torch.float), mask=mask)
            data.set_num_nodes()
            assert data.num_nodes == self.num_nodes, 'Number of nodes error'
        else:
            raise NotImplementedError()

        
        self.config_data(data)

    def read_subgraphs(self, subG_path, split = True):
        labels = {}
        label_idx = 0
        train_subGs, val_subGs, test_subGs = [],[],[]  # store nodes of each subgraphs
        train_subG_labels, val_subG_labels, test_subG_labels = [],[],[]  # store labels of each subgraphs
        train_mask, val_mask, test_mask = [],[],[]
        with open(subG_path) as f:  
            subgraph_idx = 0
            for line in f: # 每行是一个subgraph
                nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
                if len(nodes) !=0:
                    if len(nodes) == 1:
                        print(nodes)
                    sub_labels = line.split("\t")[1].split("-")
                    if len(sub_labels) >1:
                        self.multilabel = True
                    # assign idx foe labels
                    for label_name in sub_labels:  
                        if label_name not in labels.keys():
                            labels[label_name] = label_idx
                            label_idx += 1
                    # subG in training set
                    if line.split("\t")[2].strip() == "train": 
                        train_subGs.append(nodes)   # nodes of this subgraph, appended into train_subGs
                        train_subG_labels.append([labels[l] for l in sub_labels])  # labels of this subgraph
                        train_mask.append(subgraph_idx) 
                    elif line.split("\t")[2].strip() == "val":
                        val_subGs.append(nodes)
                        val_subG_labels.append([labels[l] for l in sub_labels])
                        val_mask.append(subgraph_idx)
                    elif  line.split("\t")[2].strip() == "test":
                        test_subGs.append(nodes)
                        test_subG_labels.append([labels[l] for l in sub_labels])
                        test_mask.append(subgraph_idx)
                    subgraph_idx += 1
        if not self.multilabel:
            train_subG_labels = torch.tensor(train_subG_labels).squeeze()
            val_subG_labels   = torch.tensor(val_subG_labels).squeeze()
            test_subG_labels  = torch.tensor(test_subG_labels).squeeze()
        if len(val_mask) < len(test_mask):  # val 和test 中数量多的为val set
            return train_subGs, train_subG_labels, test_subGs, test_subG_labels, val_subGs, val_subG_labels
        return train_subGs, train_subG_labels, val_subGs, val_subG_labels, test_subGs, test_subG_labels


    def split(self, config):
        if config['node_feat_type'] == 'deg':
            self.data.feat_Is_Degree(one_hot=config['one_hot'])  # num_attr = max(x) +1
        elif config['node_feat_type'] == 'one':
            self.data.feat_Is_One()   # num_attr = max(x)
        elif config['node_feat_type'] == 'node_id':
            self.data.feat_Is_NodeID(one_hot=config['one_hot'])  # num_attr = max(x)+1
        elif config['node_feat_type'] == 'RWPE':
            # TODO
            pass
        elif config['node_feat_type'] == 'Kernel':     
            # TODO
            pass
        elif config['node_feat_type'] == 'Learnable':
            # TODO
            pass
        else:
            raise NotImplementedError

        if config['node_feat_type'] == 'deg' or config['node_feat_type'] == 'node_id':
            num_attr = torch.max(self.data.x)+1 if not config['one_hot'] else self.data.x.shape[-1]
        elif config['node_feat_type'] == 'one':
            num_attr = torch.max(self.data.x)+1
        else:
            # TODO
            pass
        self.data.cuda()
        self.data.set_num_attr(num_attr)
        train_dataset = SubgraphDataset(*self.get_split('train'))
        val_dataset   = SubgraphDataset(*self.get_split('val'))
        test_dataset  = SubgraphDataset(*self.get_split('test'))

        return train_dataset, val_dataset, test_dataset

    def get_split(self, mask):
        mask_idx = {'train':0, 'val':1, 'test':2}[mask]
        return self.data.x, self.data.edge_index, self.data.edge_attr, self.data.pos[self.data.mask == mask_idx], self.data.subG_labels[self.data.mask == mask_idx]
    
    def config_data(self, data):
        self.data: Graph = data
        self.features    = data.x
        self.nfeat       = data.x.shape[1]
        self.edge_index  = data.edge_index
        self.subG_labels = data.subG_labels
        self.num_classes = data.subG_labels.shape[1] if self.multilabel else int(max(data.subG_labels)) + 1


class SubgraphDataset(Dataset):
    def __init__(self, x, edge_index, edge_attr, pos, y):
        '''
        dataset每个item为一个子图内节点，即pos的一个元素
        dataset的edge_index, x, edge_attr, y, pos属性为原图属性
        '''
        super(SubgraphDataset,self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.y = y  # 子图的label
        self.num_nodes = x.shape[0]
    
    def __len__(self):
        return self.pos.shape[0]  # 子图数  pos每个元素为每个子图的padding 后节点

    def __getitem__(self, index):
        return self.pos[index], self.y[index]
    
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        return self


class SubgraphDataLoader(DataLoader):
    def __init__(self, Gdataset: SubgraphDataset, batch_size = 64, shuffle = True, drop_last = False, multilabel = False, binaryclass=False):
        super(SubgraphDataLoader, self).__init__(dataset=torch.arange(len(Gdataset)).to(Gdataset.x.device),
                                                 batch_size= batch_size,
                                                 shuffle= shuffle,
                                                 drop_last=drop_last)
        self.Gdataset = Gdataset
        self.multilabel = multilabel
        self.binaryclass = binaryclass
        
    
    def get_x(self):
        return self.Gdataset.x
    
    def get_ei(self):
        return self.Gdataset.edge_index
    
    def get_ea(self):
        return self.Gdataset.edge_attr
    
    def get_pos(self):
        return self.Gdataset.pos
    
    def get_y(self):
        return self.Gdataset.y

    def batch_nodes_mask(self, pos):
        z    = torch.zeros(self.Gdataset.num_nodes, dtype = torch.int64).cuda()
        pos_ = torch.clone(pos)   # subgraphs in batch
        pos_ = pos_.flatten()     # all nodes in batch subgraphs
        tpos = pos_[pos_ >= 0].cuda()  # 所有batch子图中的节点
        z[tpos] = 1
        return tpos, z  

    def __iter__(self) :
        self.iter  = super(SubgraphDataLoader, self).__iter__()  # 创建一个迭代器  每次next(iter)都会返回下一个batch的idx
        return self
    
    def __next__(self):
        perm       = next(self.iter)  # 取下一个迭代的 item indices 即下一个batch的子图id
        x          = self.get_x()
        edge_index = self.get_ei()
        edge_attr  = self.get_ea()
        pos        = self.get_pos()[perm] # 下一次batch的子图 
        y          = self.get_y()[perm]   # 下一个batch的子图label
        batch_nodes, batch_nodes_mask = self.batch_nodes_mask(pos)  # batch 子图在全图中的mask
        if not self.binaryclass and not self.multilabel:
            y = y.to(torch.int64)
        return x, edge_index, edge_attr, pos, y, batch_nodes, batch_nodes_mask