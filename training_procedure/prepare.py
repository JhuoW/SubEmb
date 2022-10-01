import imp
import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal
from DataHelper.Graph import Graph
from model.SubgraphCL import SubGraphCL
from model.GLASS import *
from model.MLP_Prop import MLP_Model

def prepare_loss(self, graph:Graph):
    multilabel = False  # 每个节点是否有多个label
    binaryclass = False  # 是否只存在2种label
    if graph.subG_labels.unique().shape[0] == 2:  
        #shape[0] == 2 有两种情况，1）每个节点1个label，数据集中有2种label 2).每个节点multi label 编码为0,1向量    BCE用于label为0,1二值的情况
        loss_func = nn.BCEWithLogitsLoss()
        # 如果每个节点由多label 被编码为0-1向量  ， 即 subG_labels shape为 [N, C]
        # print(graph.subG_labels.shape)
        if graph.subG_labels.ndim >1 :  # ndim表示维数 [N,C]的ndim = 2
            multilabel = True
            binaryclass = False
            output_dim = graph.subG_labels.shape[1]
            
        else: # 如果每个节点只有一个label，即sub_labels shape为 (N, ) 共2种label
            multilabel = False
            binaryclass = True
            output_dim = 1

    else:  # 每个节点一个label， 共有多种label
        multilabel = False
        binaryclass = False
        graph.y = graph.y.to(torch.int64)
        loss_func = nn.CrossEntropyLoss()
        output_dim = graph.subG_labels.unique().shape[0]
    
    return loss_func, output_dim, graph, multilabel, binaryclass

def prepare_train(self, model):
    config = self.config
    optimizer = getattr(torch.optim, config['optimizer'])(  params          = model.parameters(), 
                                                            lr              = config['lr'] ,
                                                            weight_decay    = config.get('weight_decay', 0) )
    if config.get('lr_scheduler', False):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'],gamma=config['gamma'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['resi'], min_lr = 5e-5)
    
    return optimizer, scheduler

def prepare_model(self, graph, output_dim):
    config = self.config
    model_name = config['model_name']
    if model_name in ["SubgraphCL"]:
        model = SubGraphCL(config).cuda()
    elif model_name == "GLASS":
        model = GLASSModel(config, output_channel=output_dim,num_attr=graph.num_attr).model
    elif model_name in ["MLP", "MLP_Prop"]:
        model = MLP_Model(config, graph.num_attr, output_dim).cuda()
        model.load_pretrain()
    return model

def load_data(self, idx, synthetic = False):
    args = self.args
    path = osp.join("datasets/" if not synthetic else "synthetic/")
    dataset = DatasetLocal(args.dataset, "")
    dataset.dataset_source_folder_path = path
    dataset.mask = idx
    data = dataset.load()
    return data, dataset

def init(self, graph):
    config = self.config
    loss_func, output_dim, graph, multilabel, binaryclass = self.prepare_loss(graph)
    model = self.prepare_model(graph, output_dim)
    optimizer, scheduler = self.prepare_train(model)
    
    return graph, model, optimizer, loss_func, scheduler, multilabel, binaryclass, output_dim