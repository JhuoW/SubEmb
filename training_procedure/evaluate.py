import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.utils import Evaluation
import numpy as np
from utils.utils import pad2batch


@torch.no_grad()
def get_eval_result(self, labels, pred_l, loss = None):

    # if self.config['multilabel']:
    #     micro , macro = Evaluation(pred_l , labels)
    # else:
    pred_l = np.argmax(pred_l, axis=1)
    micro = f1_score(labels, pred_l, average = "micro")
    macro = f1_score(labels, pred_l, average = "macro")

    return {
        "micro": round(micro * 100 , 2) , # to percentage
        "macro": round(macro * 100 , 2)
    }

@torch.no_grad()
def evaluate(self, graph, model, loss_func, vt_loader):
    model.eval()
    preds = []
    ys    = []
    # total_loss  = []
    x = graph.x.cuda()
    edge_index = graph.edge_index.cuda()
    graph = graph.cuda()
    for batch in vt_loader:
        labels = batch[-3]
        if self.config['model_name'] in ['GLASS']:
            out = model(graph.x, graph.edge_index, graph.edge_attr, batch[3], batch[6])
            preds.append(out)
            ys.append(labels)
            # loss = loss_func(out, labels)
            # total_loss.append(loss)
        if self.config['model_name'] in ['MLP', 'MLP_Prop']:
            subG_nodes = batch[3]
            batch_idx, nodes_in_subG  =  pad2batch(subG_nodes)
            out = model(x, edge_index, batch_idx, nodes_in_subG, graph)
            preds.append(out)
            ys.append(labels)
    
    preds_ = torch.cat(preds, dim = 0)
    y_ = torch.cat(ys, dim = 0)
    result = get_eval_result(self, y_.detach().cpu().numpy(), preds_.detach().cpu().numpy())
    loss = loss_func(preds_, y_)
    
    return result, float(loss)