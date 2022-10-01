import os.path as osp
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import torch
import random
from model.MLP_Prop import train_MLP_Prop
from utils.logger import Logger
from utils.utils import *
from utils.random_seeder import set_random_seed, set_cudnn_backends
from training_procedure import Trainer
from DataHelper.DatasetLocal import SubgraphDataset, SubgraphDataLoader
from utils.utils import buildAdj


def main(args, config, logger: Logger, run_id: int, ori_graph, 
        train_dataset: SubgraphDataset, val_dataset: SubgraphDataset, test_dataset:SubgraphDataset):
    T = Trainer(config=config, args= args, logger= logger)
    graph, model, optimizer, loss_func, scheduler, multilabel, binaryclass, output_dim = T.init(ori_graph)   # model of current split
    batch_size   = config['batch_size']
    train_loader = SubgraphDataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = SubgraphDataLoader(val_dataset,  batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = SubgraphDataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


    pbar                = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    patience_cnt 		= 0
    maj_metric 			= "micro"   # or macro
    best_metric 	  	= 0
    best_metric_epoch 	= -1 # best number on dev set
    report_dev_res 		= 0
    report_tes_res 		= 0
    best_dev_loss       = 100000.0
    val_loss_history    = []
    num_div             = test_dataset.y.shape[0] / batch_size
    dev_result          = {"micro": 0 ,"macro": 0}
    report_dev_res      = {"micro": 0 ,"macro": 0}
    test_result         = {"micro": 0 ,"macro": 0}
    report_tes_res      = {"micro": 0 ,"macro": 0}
    dev_loss            = 0
    test_loss           = 0
    for epoch in pbar:
        model, loss = T.train(graph, model, loss_func, optimizer, train_loader)
        if config['lr_scheduler']:
            scheduler.step(loss)

        ########## GLASS ##########
        if config['model_name'] == 'GLASS':

            if epoch >= 100 / num_div: 
                dev_result, dev_loss   = T.evaluation(graph, model, loss_func, val_loader)  # return 2 list, 
                test_result, test_loss = T.evaluation(graph, model, loss_func, test_loader)
                now_metric = dev_result[maj_metric]
                if config['monitor'] == 'val_acc':
                    if now_metric >= best_metric:
                        best_metric       = now_metric
                        best_metric_epoch = epoch
                        report_dev_res    = dev_result
                        report_tes_res    = test_result
                        patience_cnt      = 0
                    else:
                        patience_cnt      += 1
                elif config['monitor'] == 'val_loss':
                    if dev_loss <= best_dev_loss:
                        best_dev_loss           = dev_loss
                        best_metric_epoch       = epoch
                        report_dev_res          = dev_result  
                        report_tes_res          = test_result
                        patience_cnt            = 0
                        
                    else:
                        patience_cnt            += 1
            if patience_cnt > 100 / num_div :
                break

        elif config['model_name'] in ['MLP', 'MLP_Prop']:
            dev_result, dev_loss   = T.evaluation(graph, model, loss_func, val_loader)
            test_result, test_loss = T.evaluation(graph, model, loss_func, test_loader)
            now_metric = dev_result[maj_metric]
            if config['monitor'] == 'val_acc':
                if now_metric >= best_metric:
                    best_metric       = now_metric
                    best_metric_epoch = epoch
                    report_dev_res    = dev_result
                    report_tes_res    = test_result
                    patience_cnt      = 0
                else:
                    patience_cnt      += 1
            elif config['monitor'] == 'val_loss':
                if dev_loss <= best_dev_loss:
                    best_dev_loss           = dev_loss
                    best_metric_epoch       = epoch
                    report_dev_res          = dev_result  
                    report_tes_res          = test_result
                    patience_cnt            = 0
                    
                else:
                    patience_cnt            += 1 

            if config['patience'] > 0 and patience_cnt >= config['patience'] and config['monitor'] in ['val_acc', 'val_loss']:
                break

        postfix_str = "<Epoch %d> [Train Loss] %.4f [Curr Dev Acc] %.2f <Best Epoch %d> [Best Dev Acc] %.2f [Test] %.2f ([Report Test] %.2f) " % ( 
                        epoch ,      loss,         dev_result[maj_metric], best_metric_epoch ,report_dev_res[maj_metric], test_result[maj_metric], report_tes_res[maj_metric])

        pbar.set_postfix_str(postfix_str)
    
    if config['model_name'] == "MLP_Prop":
       model,  report_dev_res, report_tes_res, loss =  train_MLP_Prop(model, graph, config, output_dim, loss_func, train_loader, val_loader, test_loader)

        
    logger.log("best epoch is %d" % best_metric_epoch)
    logger.log("Best Epoch Valid Acc is %.2f" % (report_dev_res[maj_metric]))
    logger.log("Best Epoch Test  Acc is %.2f" % (report_tes_res[maj_metric]))
    return model,  report_dev_res, report_tes_res, loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'ppi_bp') 
    parser.add_argument('--num_workers', default=8, type=int, choices=[0,8])
    parser.add_argument('--seed', default=0, type=int, choices=[0, 1, 1234])
    # parser.add_argument('--data_dir', type= str, default="datasets/") 
    parser.add_argument('--synthetic', type = bool, default=False)
    parser.add_argument('--hyper_file', type=str, default= 'config/')
    parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)   
    parser.add_argument('--no_dev', action = "store_true" , default = False)
    parser.add_argument('--patience', type = int  , default = -1)
    parser.add_argument('--gpu_id', type = int  , default = 0)
    parser.add_argument('--model', type = str, default='MLP_Prop')  
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    logger = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()

    config_path = osp.join(args.hyper_file, args.dataset + '.yml')
    config = get_config(config_path)
    multilabel = config['multilabel']
    model_name = args.model
    config = config[model_name] 
    config['multilabel'] = multilabel
    config['model_name'] = model_name
    config['dataset']    = args.dataset
    dev_ress = []
    tes_ress = []
    tra_ress = []
    if config.get('seed',-1) >= 0:
        set_random_seed(config['seed'])
        set_cudnn_backends()
        logger.log ("Seed set. %d" % (config['seed']))
    # seeds = [random.randint(0,233333333) for _ in range(config['multirun'])]

    args.data_dir = "datasets/" if not args.synthetic else "synthetic/"
    dataset_helper = load_data(args)
    dataset_helper.load(config)  # config dataset
    
    print_config(config)
    
    train_dataset, val_dataset, test_dataset = dataset_helper.split(config)
    data = dataset_helper.get_graph()  # The whole graph
    logger.log ("Num Nodes. %d" % data.num_nodes)
    for run_id in range(config['multirun']):   # one mask
        set_random_seed((1<<run_id) - 1)  # 2^run_id -1
        logger.add_line()
        logger.log ("\t\t%d th Run" % run_id)
        logger.add_line()
        # set_random_seed(seeds[run_id])
        # logger.log ("Seed set to %d." % seeds[run_id])

        model, report_dev_res, report_tes_res, loss = main(args, config, logger, run_id, data, train_dataset, val_dataset, test_dataset)
        logger.log("Current Seed: %d" % ((1<<run_id) - 1))
        logger.log("%d th Run ended. Final Train Loss is %s" % (run_id , str(loss)))
        logger.log("%d th Run ended. Best Epoch Valid Result is %s" % (run_id , str(report_dev_res)))
        logger.log("%d th Run ended. Best Epoch Test  Result is %s" % (run_id , str(report_tes_res)))
        
        dev_ress.append(report_dev_res) 
        tes_ress.append(report_tes_res)  
    logger.add_line()
    for metric in ["micro", "macro"]: # 2个metric
        for result, name in zip( [dev_ress, tes_ress],
                                 ['Dev', 'Test']):

            now_res = [x[metric] for x in result]   

            logger.log ("%s of %s : %s" % (metric , name , str([round(x,2) for x in now_res])))

            avg = sum(now_res) / config['multirun']
            std = (sum([(x - avg) ** 2 for x in now_res]) / config['multirun']) ** 0.5

            logger.log("%s of %s : avg / std = %.2f / %.2f" % (metric , name , avg , std))
        logger.log("")           


# GLASS: 61.32 / 1.57  