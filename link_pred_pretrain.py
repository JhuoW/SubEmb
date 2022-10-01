from torch.optim import Adam, lr_scheduler
import optuna
from torch.nn import BCEWithLogitsLoss
import argparse
import torch.nn as nn
import numpy as np
from utils.utils import get_device, load_data, get_config
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
from model.LP_backbone import MyGCN, MyGCNConv, MLP, EdgeGNN, MeanPool
from sklearn.metrics import f1_score, roc_auc_score

## Pretrain on link Prediction Task to produce input node embeddings for GLASS  预训练node id 生成初始input feature.
# 原先节点特征为node id， 先将node id 编码 nn.Embedding， 再基于link pred来训练节点特征

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
# Node feature settings. 
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true', default=True)  
# Train settings
parser.add_argument('--repeat', type=int, default=1)
# Optuna Settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--abl', action='store_true')

parser.add_argument('--optruns', type=int, default=100)  # 100次训练
parser.add_argument('--path', type=str, default="pretrain_node_features/")
parser.add_argument('--name', type=str, default="opt")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--num_workers', default=8, type=int, choices=[0,8])
parser.add_argument('--data_dir', type= str, default="datasets/") 
parser.add_argument('--synthetic', type = bool, default=False)
parser.add_argument('--hyper_file', type=str, default= 'config/')
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)   
parser.add_argument('--no_dev', action = "store_true" , default = False)
parser.add_argument('--gpu_id', type = int  , default = 0)
parser.add_argument('--model', type = str, default='GLASS')  


args = parser.parse_args()
device = get_device(args.device)
config_path = osp.join(args.hyper_file, args.dataset + '.yml')
config = get_config(config_path)
model_name = args.model
multilabel = config['multilabel']
config = config[model_name] 
config['multilabel'] = multilabel


datahelper = load_data(args)
datahelper.load(config)


baseG= datahelper.get_graph()
num_attr, output_channels = 0, 1

class LinkSet(Dataset):
    def __init__(self, pos_neg_edges, pos_neg_y):
        super(LinkSet, self).__init__()
        self.pos_neg_edges = pos_neg_edges
        self.pos_neg_y = pos_neg_y
    
    def __len__(self):
        return self.pos_neg_edges.shape[0]
    
    def __getitem__(self, index):
        return self.pos_neg_edges[index], self.pos_neg_y[index]


class LinkLoader(DataLoader):
    def __init__(self, linkset: LinkSet, batch_size = 64, shuffle = True, drop_last = False):
        super(LinkLoader, self).__init__(dataset=torch.arange(len(linkset)).cuda(),
                                         batch_size=batch_size,
                                         shuffle=shuffle, 
                                         drop_last=drop_last)  # 用编号作为数据集

        self.linkset = linkset
    
    def get_pos_neg_edges(self):
        return self.linkset.pos_neg_edges

    def get_pos_neg_y(self):
        return self.linkset.pos_neg_y
    
    def __iter__(self):
        self.iter = super(LinkLoader, self).__iter__()  # 返回一个迭代器 ，调用迭代器的next方法可以得到下一个batch的item indices
        return self


    def __next__(self):
        perm = next(self.iter)  # 当前迭代的index
        return self.get_pos_neg_edges()[perm], self.get_pos_neg_y()[perm]

def split():
    '''
    load and split dataset.
    '''
    global trn_dataset, val_dataset
    global num_attr, output_channels, loader_fn, tloader_fn
    # 为图中每个节点赋予原始特征
    if args.use_deg:
        baseG.feat_Is_Degree()   # 有max(x) +1 种
    elif args.use_one:
        baseG.feat_Is_One()    # 有1中
    elif args.use_nodeid:
        baseG.feat_Is_NodeID()  # 最多有 max(x) +1 种 
    else:
        raise NotImplementedError
    num_attr = torch.max(baseG.x) + 1  # 最大度/ 1 /  最大node_id
    baseG.to(device)
    
    link_pred_datasets = baseG.get_LPdataset()   # pos: positive edges + negative edges
    _, _ , _, pos_neg_edges, y = link_pred_datasets
    
    # print(x.shape) # [21521, 1, 1]
    idx = torch.randperm(pos_neg_edges.shape[0]).cuda()
    trn_len = int(0.95 * idx.shape[0])  # 95%的边用来训练
    trn_idx = idx[:trn_len]  
    val_idx = idx[trn_len:]
    trn_dataset = LinkSet(pos_neg_edges=pos_neg_edges[trn_idx], pos_neg_y=y[trn_idx])  # 训练集边
    val_dataset = LinkSet(pos_neg_edges=pos_neg_edges[val_idx], pos_neg_y=y[val_idx])  # 验证集边

    def loader_fn(ds, bs):
        return LinkLoader(ds, bs)

    def tloader_fn(ds, bs):
        return LinkLoader(ds, bs, shuffle=False)


split()


def buildModel(hidden_dim, K, dropout, jk):
    '''
    Build a EdgeGNN model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        K: number of GLASSConv.
    '''
    tmp2 = hidden_dim * (K) if jk else hidden_dim
    
    # K 层MyGCNConv来得到node embedding
    # EmbGConv来学习node embeddings， 是conv_layer层的MyGCNConv
    conv = MyGCN(in_channels=hidden_dim,      # 输入的1位特征在模型内被 embedding lookup操作转化为 64为 输入特征，所以GNN的实际输入为64维
                 hidden_channels=hidden_dim,
                 out_channels= hidden_dim,
                 K=K,
                 num_attr=num_attr,
                 act=nn.ReLU(inplace=True),
                 jk = jk,
                 dropout=dropout,
                 conv=MyGCNConv,
                 gn=True,
                 aggr=args.aggr)   # 3种节点特征赋值方式 最多可能存在多少种特征


    # 
    edge_ssl = MLP(input_channels = tmp2,   # 输入feature dim = tmp2 为边表示向量的维度
                   hidden_channels = hidden_dim,
                   output_channels = 1,    # output dim  为每组待预测节点对输出一个一维scalar作为score
                   num_layers = 2,
                   dropout=dropout,
                   activation=nn.ReLU(inplace=True))

    gnn = EdgeGNN(conv, nn.ModuleList([edge_ssl]), nn.ModuleList([MeanPool()])).cuda()
    return gnn


def work(hidden_dim, conv_layer, dropout, jk, lr, batch_size):
    '''
    try a set of hyperparameters for pretrained GNN.
    '''
    trn_loader = loader_fn(trn_dataset, batch_size)  # 训练边 
    val_loader = tloader_fn(val_dataset, val_dataset.pos_neg_y.shape[0])  # 验证边   直接找出最佳GNN参数就可以了，无需测试集
    outs = []
    loss_fn = lambda x, y: BCEWithLogitsLoss()(x.flatten(), y.flatten())  # 二分类损失
    for _ in range(args.repeat):
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk)
        with torch.no_grad():
            emb = gnn.nodeEmb(baseG.x, baseG.edge_index,
                              baseG.edge_attr).detach().cpu()   # node embedding by num_layer MyGCNConv
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=0.7,
                                             min_lr=5e-5,
                                             patience=50)
        best_score = 0.0
        early_stop = 0
        for i in range(100):  # 400
            gnn.train()
            losss = []
            for ib, batch in enumerate(trn_loader):
                optimizer.zero_grad()
                emb = gnn.nodeEmb(baseG.x, baseG.edge_index,
                                  baseG.edge_attr)   # 用原图中的所有边来训练GNN
                edge_emb = gnn.pool(emb, batch[0], None)  # batch[0] 为一个batch的训练边
                edge_pred = gnn.preds[0](edge_emb)   # 用MLP预测一个batch中每条边的值，为一个实数 为是否存在边的预测值
                loss = loss_fn(edge_pred, batch[1])   # BClogicLoss: sigmoid归一化到0-1之间 和 边的label(0或1) 计算损失
                loss.backward()
                scd.step(loss)
                losss.append(loss.item())
                optimizer.step()
                if ib >= 9:
                    break
            if i % 5 == 0:  # 每5个epoch验证一次
                score, _ = test(gnn, val_loader, binaryf1, loss_fn)
                print(f"iter {i} loss {np.average(losss)} score {score}",
                      flush=True)
                early_stop += 1
                if score > best_score:  # 如果当前模型最好，取出当前的node embedding 赋值给emb
                    with torch.no_grad():
                        emb = gnn.nodeEmb(
                            baseG.x, baseG.edge_index,
                            baseG.edge_attr).detach().cpu()
                    best_score = score
                    early_stop = 0
                if early_stop >= 3:
                    break
            else:
                print(f"iter {i} loss {np.average(losss)}", flush=True)

        outs.append(best_score)

    return np.average(outs) - np.std(outs), emb  # 当前超参数配置下的准确率


best_score = 0


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:  # 64个训练子图  每个item包括  pos:[[],[],[],...] 每个元素为一个子图中的节点 , y: [...]  
        optimizer.zero_grad()
        pred = model(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])   # cross-entropy   batch[-1]为ground truth label
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)



@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:  # val_loader 
        pred = model(baseG.x, baseG.edge_index, baseG.edge_weight, batch[0])  # 验证
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

def binaryf1(pred, label):
    '''
    pred, label are numpy array
    can process multi-label target
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")

def obj(trial):
    '''
    a trial of hyperparameter optimization.
    '''
    global trn_dataset, val_dataset, tst_dataset, args
    global input_channels, output_channels, loader_fn, tloader_fn
    global loss_fn, best_score
    hidden_dim = 64
    conv_layer = trial.suggest_int('conv_layer', 2, 5, step=1)  # 层数2-5
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    args.aggr = trial.suggest_categorical("aggr", ["sum", "mean", "gcn"])
    jk = 0
    lr = 1e-3
    batch_size = 131072
    jk = (jk == 1)
    score, emb = work(hidden_dim, conv_layer, dropout, jk, lr, batch_size)  # score为当前超参数配置下的准确率
    # save best embeddings
    if score > best_score:
        torch.save(emb, f"{args.path}{args.dataset}_{hidden_dim}.pt")
        best_score = score
    return score


print(args)
# tuning hyperparameters of pretrained GNNs.
study = optuna.create_study(direction="maximize",
                            storage="sqlite:///" + args.path + args.name +
                            ".db",
                            study_name=args.name,
                            load_if_exists=True)
study.optimize(obj, n_trials=args.optruns)
print("best params ", study.best_params)
print("best valf1 ", study.best_value)
