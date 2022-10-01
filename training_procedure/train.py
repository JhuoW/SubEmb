
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils.utils import pad2batch

def train(self, graph, model, loss_func, optimizer, train_loader):
    model.train(True)
    config = self.config 
    graph = graph.cuda()
    x = graph.x.cuda()
    edge_index = graph.cuda()
    total_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_labels = batch[-3]
        if config['model_name'] in ['GLASS']:
            out = model(graph.x, graph.edge_index, graph.edge_attr, batch[3], batch[6])
            loss = loss_func(out, batch_labels)
            loss.backward()
            total_loss.append(loss.detach().item())
            optimizer.step()
        elif config['model_name'] in ['SubgraphCL']:
            subG_nodes = batch[3]  # [num_subgraph_in_batch, node_idx_with_padding_-1]
            batch_nodes = batch[5] # all nodes indexes in batch
            batch_nodes_mask = batch[6]
            out = model(graph, subG_nodes, batch_nodes, batch_nodes_mask)
        elif config['model_name'] in ['MLP', 'MLP_Prop']:
            subG_nodes = batch[3]
            batch_idx, nodes_in_subG  =  pad2batch(subG_nodes)
            out = model(x, edge_index, batch_idx.cuda(), nodes_in_subG, graph)
            loss = loss_func(out, batch_labels)
            loss.backward()
            total_loss.append(loss.detach().item())
            optimizer.step()
    loss = sum(total_loss) / len(total_loss)
    return model, float(loss)