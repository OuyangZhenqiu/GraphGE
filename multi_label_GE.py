import os
import os.path as osp
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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from utils import set_random_seed, ScheduledOptim, EarlyStopping

# model
from model.GraphGE import GNN
from GNN_dataset import load_skfold_dataloader_list

def args_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # basis
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--model', type=str, default='GraphGE')
    parser.add_argument('--freeze_cell_encoder', type=bool, default=False)

    # cell line 256 2 256
    parser.add_argument('--gene_num', type=int, default=663)
    parser.add_argument('--gene_feature_dim', type=int, default=1)
    parser.add_argument('--gene_hidden_dim', type=int, default=128)
    parser.add_argument('--cell_graph_num_layer', type=int, default=2)
    parser.add_argument('--cell_graph_hidden_dim', type=int, default=64)
    parser.add_argument('--cell_hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=173)
    parser.add_argument('--cell_edge_index_thresh', type=float, default=0.95)
    
    return parser.parse_args()

def get_metrics(pred, target, average=True):
    rmse_list = []
    mae_list = []
    pearson_r_list = []
    num_label = 0
    for i in range(target.shape[1]):
        # ignore nan values
        is_labeled = target[:,i] == target[:,i]
        num_label += is_labeled.sum()
        pearson_r_list.append(pearsonr(target[is_labeled, i], pred[is_labeled, i])[0])
        rmse_list.append(np.sqrt(((target[is_labeled, i]-pred[is_labeled, i])**2).mean()))
        mae_list.append(np.abs(target[is_labeled, i]-pred[is_labeled, i]).mean())
    
    pearson_r = sum(pearson_r_list)/len(pearson_r_list)
    if average:
        RMSE = sum(rmse_list)/len(rmse_list)
    else:
        RMSE = rmse_list
    MAE = sum(mae_list)/len(mae_list)

    return  pearson_r, RMSE, MAE

def train(args, model, loader, opt):
    model.train()
    device = next(model.parameters()).device
    for batch in tqdm(loader, desc='Iteration'):
        cell_graph, ic50 = batch.cell_graph, batch.ic50
        cell_graph, ic50 = cell_graph.to(device), ic50.to(device)
        output = model(cell_graph)
        mask = ic50==ic50
        loss = F.mse_loss(output[mask], ic50.to(torch.float32)[mask], reduction="mean")
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
            cell_graph, ic50 = batch.cell_graph, batch.ic50
            cell_graph, ic50 = cell_graph.to(device), ic50.to(device)
            output = model(cell_graph)
            y_true.append(ic50)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    pearson_r, rmse, mae =  get_metrics(y_pred.cpu(), y_true.cpu(), average)
    return pearson_r, rmse, mae
    
def main(args):
    set_random_seed(args.seed)
    train_loader_list, val_loader_list, test_loader_list = load_skfold_dataloader_list(args)
    val_pearson_r_list, test_pearson_r_list = [], []
    val_rmse_list, test_rmse_list = [], []
    val_mae_list, test_mae_list = [], []
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

        stopper = EarlyStopping(patience=20)
        for epoch in range(args.num_epoch):
            print("Epoch {}:".format(epoch))
            print("--------train--------:")
            loss = train(args, model, train_loader, optimizer)
            val_pearson_r, val_rmse, val_mae = validate(args, model, val_loader)
            print("Val pearson_r: {}, rmse:{}".format(val_pearson_r, val_rmse))
            
            fitlog.add_loss({"Val":{"epoch":epoch, "rmse":val_rmse}}, step=idx)
            if stopper.step(val_rmse, model):
                break
            
        print("---------Training finish-------")
        stopper.load_checkpoint(model)
        train_pearson_r, train_rmse, train_mae = validate(args, model, train_loader)
        print("Train_pearson_r:{}, train_rmse:{}, train_mae:{}".format(train_pearson_r, train_rmse, train_mae))
        val_pearson_r, val_rmse, val_mae = validate(args, model, val_loader)
        print("Val_pearson_r:{}, val_rmse:{}, val_mae:{}".format(val_pearson_r, val_rmse, val_mae))
        test_pearson_r, test_rmse, test_mae = validate(args, model, test_loader)
        print("Test_pearson_r:{}, test_rmse:{}, test_mae:{}".format(test_pearson_r, test_rmse, test_mae))
        val_pearson_r_list.append(val_pearson_r)
        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)
        test_pearson_r_list.append(test_pearson_r)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        save_path = "./save_path"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), "{}/GraphGE_{}.pth".format(save_path, idx))
        fitlog.add_metric({"train":{"pearson_r":train_pearson_r,"rmse":train_rmse, "mae":train_mae}, \
            "val":{"micro_rmse":val_pearson_r,"rmse":val_rmse, "mae":val_mae}, \
            "test":{"micro_rmse":test_pearson_r,"rmse":test_rmse, "mae":test_mae}}, step=idx)
    
    print("---------Finish--------")
    print("Model:", args.model)
    val_pearson_r_mean, val_pearson_r_std = np.mean(val_pearson_r_list), np.std(val_pearson_r_list)
    print("Val pearson_r, mean: {}, std: {}".format(val_pearson_r_mean, val_pearson_r_std))
    val_rmse_mean, val_rmse_std = np.mean(val_rmse_list), np.std(val_rmse_list)
    print("Val rmse, mean: {}, std: {}".format(val_rmse_mean, val_rmse_std))
    val_mae_mean, val_mae_std = np.mean(val_mae_list), np.std(val_mae_list)
    print("Val mae, mean: {}, std: {}".format(val_mae_mean, val_mae_std))
    test_pearson_r_mean, test_pearson_r_std = np.mean(test_pearson_r_list), np.std(test_pearson_r_list)
    print("Test pearson_r, mean: {}, std: {}".format(test_pearson_r_mean, test_pearson_r_std))
    test_rmse_mean, test_rmse_std = np.mean(test_rmse_list), np.std(test_rmse_list)
    print("Test rmse, mean: {}, std: {}".format(test_rmse_mean, test_rmse_std))
    test_mae_mean, test_mae_std = np.mean(test_mae_list), np.std(test_mae_list)
    print("Test mae, mean: {}, std: {}".format(test_mae_mean, test_mae_std))
    
    fitlog.add_best_metric({
        "val":{"pearson_r_mean":val_pearson_r_mean, "pearson_r_std":val_pearson_r_std, "rmse_mean":val_rmse_mean, "rmse_std":val_rmse_std, "mae_mean":val_mae_mean, "mae_std":val_mae_std}, 
        "test":{"pearson_r_mean":test_pearson_r_mean, "pearson_r_std":test_pearson_r_std, "rmse_mean":test_rmse_mean, "rmse_std":test_rmse_std, "mae_mean":test_mae_mean, "mae_std":test_mae_std}
    })


if __name__ == '__main__':
    args = args_parse()
    main(args)

    
 