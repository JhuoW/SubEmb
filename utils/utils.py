import yaml
import numpy as np
from sklearn import metrics
from torch_geometric.data import Data
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def load_data(args):
    path = osp.join(args.data_dir, args.dataset)
    dataset = DatasetLocal(args.dataset, "")
    dataset.dataset_source_folder_path = path
    dataset.recache = args.recache
    return dataset
    


def homophily(edge_index, y, method: str = 'edge'):
    assert method in ['edge', 'node']
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        return int((y[row] == y[col]).sum()) / row.size(0)  # out neigh 的同质率
    else:
        out = torch.zeros_like(row, dtype=float)
        out[y[row] == y[col]] = 1.
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        return float(out.mean())


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def getneighborslst(data: Data):
    ei = data.edge_index.numpy()
    lst = {i: set(ei[1][ei[0] == i]) for i in range(data.num_nodes)} 
    return lst

def get_device(cuda_id: int):
    device = torch.device('cuda' if cuda_id < 0 else 'cuda:%d' % cuda_id)
    return device


def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    # print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_index_to_torch_sparse_tensor(edge_index, edge_weight = None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),)).cuda()
    
    n_node = edge_index.max().item() + 1

    return torch.cuda.sparse.FloatTensor(edge_index, edge_weight, torch.Size((n_node, n_node)))


## SubGCL utils
def between_compontents(subG_nodes, batch_nodes, batch_nodes_mask):



    return 

def in_compontent():


    return





### GLASS utils
from torch.nn.utils.rnn import pad_sequence

def batch2pad(batch):
    '''
    The j-th element in batch vector is i if node j is in the i-th subgraph.
    The i-th row of pad matrix contains the nodes in the i-th subgraph.
    batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    '''
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni >= 0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    '''
    pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    pad: subG_nodes, 一个batch中的每个子图 [num_subgraphs, nodes_in_subGs_padding_-1]
    '''
    batch = torch.arange(pad.shape[0])  # batch = 子图数量   [0,1,2,3,...]
    batch = batch.reshape(-1, 1)  # 1列   [0,1,2,3,...]^T
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]  # 假设最多节点数为3  batch[:, [0,0,0]]  batch = [[0,0,0], [1,1,1], [2,2,2], ...]
    batch = batch.to(pad.device).flatten()  # [0,0,0, 1,1,1, 2,2,2,...]
    pos = pad.flatten()                     # [0,2,3, 1,4,5, 6,7,-1,...]
    idx = pos >= 0  # 把 -1的部分给去掉
    return batch[idx], pos[idx]   # pos为所有子图节点，batch为每个节点所在的子图


@torch.jit.script
def MaxZOZ(x, pos):
    '''
    produce max-zero-one label
    x is node feature  所有node features
    pos is a pad matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph. batch中每个子图的节点
    -1 is padding value.
    '''
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)  # len = num_nodes
    pos = pos.flatten()  # 得到 batch 子图中的 所有节点
    # pos[pos >= 0] removes -1 from pos
    tpos = pos[pos >= 0].to(z.device)  # 一个batch中所有存在于子图中的节点
    z[tpos] = 1   # Graph中 存在于子图中的节点为1 不在子图中的节点为0
    return z


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
