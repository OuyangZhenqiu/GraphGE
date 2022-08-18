import os
import os.path as osp
from urllib import response
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import fitlog
import pickle
from utils import set_random_seed, ScheduledOptim, EarlyStopping
from sklearn.metrics import roc_auc_score, average_precision_score

# model
from model.GraphGE import GNN
from TCGA_GNN_dataset import load_skfold_dataloader_list

def args_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # basis
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--model', type=str, default='TCGA_GraphGE')
    parser.add_argument('--freeze_cell_encoder', type=bool, default=False)

    # cell line 256 2 256
    parser.add_argument('--gene_num', type=int, default=663)
    parser.add_argument('--gene_feature_dim', type=int, default=1)
    parser.add_argument('--gene_hidden_dim', type=int, default=128)
    parser.add_argument('--cell_graph_num_layer', type=int, default=2)
    parser.add_argument('--cell_graph_hidden_dim', type=int, default=256)
    parser.add_argument('--cell_hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=75)
    parser.add_argument('--cell_edge_index_thresh', type=float, default=0.95)
    
    return parser.parse_args()

def get_metrics(pred, target):
    rocauc_list, ap_list = [], []
    for i in range(target.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if torch.sum(target[:,i] == 1) > 0 and torch.sum(target[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = target[:,i] == target[:,i]
            rocauc_list.append(roc_auc_score(target[is_labeled,i], pred[is_labeled,i]))
            ap_list.append(average_precision_score(target[is_labeled,i], pred[is_labeled,i]))
    AP = sum(ap_list)/len(ap_list)
    ROC_AUC = sum(rocauc_list)/len(rocauc_list)
    return ROC_AUC, AP

def train(args, model, loader, opt):
    model.train()
    device = next(model.parameters()).device
    for batch in tqdm(loader, desc='Iteration'):
        cell_graph, response = batch.cell_graph, batch.response
        cell_graph, response = cell_graph.to(device), response.to(device)
        output = model(cell_graph)
        mask = response==response
        loss = F.binary_cross_entropy_with_logits(output[mask], response.to(torch.float32)[mask])
        opt.zero_grad()
        loss.backward()
        opt.step_and_update_lr()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(args, model, loader, average=True):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Iteration'):
            cell_graph, response = batch.cell_graph, batch.response
            cell_graph, response = cell_graph.to(device), response.to(device)
            output = model(cell_graph)
            y_true.append(response)
            y_pred.append(torch.sigmoid(output))

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    roc_auc, ap =  get_metrics(y_pred.cpu(), y_true.cpu())
    return roc_auc, ap
    
def main(args):
    set_random_seed(args.seed)
    train_loader_list, val_loader_list, test_loader_list = load_skfold_dataloader_list(args)
    val_roc_auc_list, test_roc_auc_list = [], []
    val_ap_list, test_ap_list = [], []
    print(args)
    dir = os.path.join("./logs", "multi_task")
    if not os.path.exists(dir):
        os.makedirs(dir)
    fitlog.set_log_dir(dir, new_log=True)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    
    for idx in range(5):
        print("=======Fold {}======".format(idx))
        train_loader, val_loader, test_loader = train_loader_list[idx], val_loader_list[idx], test_loader_list[idx]
        model = GNN(args).to(args.device)

        if args.freeze_cell_encoder:
            for name, param in model.cell_encoder.named_parameters():
                param.requires_grad = False

        optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=0), factor=0.15, d_model=args.cell_hidden_dim, n_warmup_steps=30)

        stopper = EarlyStopping(mode='higher', patience=20)
        for epoch in range(args.num_epoch):
            print("Epoch {}:".format(epoch))
            print("--------train--------:")
            loss = train(args, model, train_loader, optimizer)
            roc_auc, ap = validate(args, model, val_loader)
            print("Val roc_auc: {}, ap:{}".format(roc_auc, ap))
            
            fitlog.add_loss({"Val":{"epoch":epoch, "ap":roc_auc}}, step=idx)
            if stopper.step(ap, model):
                break
            
        print("---------Training finish-------")
        stopper.load_checkpoint(model)
        train_roc_auc, train_ap = validate(args, model, train_loader)
        print("Train_roc_auc:{}, train_ap:{}".format(train_roc_auc, train_ap))
        val_roc_auc, val_ap = validate(args, model, val_loader)
        print("Val_roc_auc:{}, val_ap:{}".format(val_roc_auc, val_ap))
        test_roc_auc, test_ap = validate(args, model, test_loader)
        print("Test_roc_auc:{}, test_ap:{}".format(test_roc_auc, test_ap))
        val_roc_auc_list.append(val_roc_auc)
        val_ap_list.append(val_ap)
        test_roc_auc_list.append(test_roc_auc)
        test_ap_list.append(test_ap)
        save_path = "./save_path"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), "{}/MLP_{}.pth".format(save_path, idx))
        fitlog.add_metric({"train":{"roc_auc":train_roc_auc,"ap":train_ap}, \
            "val":{"micro_ap":val_roc_auc,"ap":val_ap}, \
            "test":{"micro_ap":test_roc_auc,"ap":test_ap}}, step=idx)
    
    print("---------Finish--------")
    print("Model:", args.model)
    val_roc_auc_mean, val_roc_auc_std = np.mean(val_roc_auc_list), np.std(val_roc_auc_list)
    print("Val roc_auc, mean: {}, std: {}".format(val_roc_auc_mean, val_roc_auc_std))
    val_ap_mean, val_ap_std = np.mean(val_ap_list), np.std(val_ap_list)
    print("Val ap, mean: {}, std: {}".format(val_ap_mean, val_ap_std))
    test_roc_auc_mean, test_roc_auc_std = np.mean(test_roc_auc_list), np.std(test_roc_auc_list)
    print("Test roc_auc, mean: {}, std: {}".format(test_roc_auc_mean, test_roc_auc_std))
    test_ap_mean, test_ap_std = np.mean(test_ap_list), np.std(test_ap_list)
    print("Test ap, mean: {}, std: {}".format(test_ap_mean, test_ap_std))

    fitlog.add_best_metric({
        "val":{"roc_auc_mean":val_roc_auc_mean, "roc_auc_std":val_roc_auc_std, "ap_mean":val_ap_mean, "ap_std":val_ap_std}, 
        "test":{"roc_auc_mean":test_roc_auc_mean, "roc_auc_std":test_roc_auc_std, "ap_mean":test_ap_mean, "ap_std":test_ap_std}
    })


if __name__ == '__main__':
    args = args_parse()
    main(args)

    
 