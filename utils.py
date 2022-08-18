import numpy as np
import pandas as pd
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from dgllife.utils import *
from torch_geometric.data import Data
from tqdm import tqdm
from itertools import compress
import numpy as np

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ScheduledOptim:
    """ https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py """

    def __init__(self, optimizer, factor, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.factor = factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def view_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self.factor * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class EarlyStopping():

    def __init__(self, mode='lower', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            if filename is None:
                filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format( \
                    dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['acc', 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'acc' or 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['acc', 'r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))
        
        
class ModelSaving():

    def __init__(self, mode='lower', filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['acc', 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'acc' or 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['acc', 'r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.filename = filename
        self.best_score = None
        self.last_score = None

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
        self.last_score = score

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))

def mask_drug_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    for idx in idx_mask:
        data.x[idx, 0] = 119
    return data

def mask_cellline_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    for idx in idx_mask:
        data.x[idx] = 0
    return data

def mask_align_gene_features(exp, mu, cn, aug_ratio):
    node_num = exp.shape[0]
    mask_num = int(node_num * aug_ratio)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    exp[idx_mask], cn[idx_mask], mu[idx_mask] = 0, 0, 0
    return (exp, cn, mu)

def mask_unalign_gene_features(exp, mu, cn, aug_ratio):
    node_num = exp.shape[0]
    mask_num = int(node_num * aug_ratio)
    exp_idx_mask = np.random.choice(node_num, mask_num, replace=False)
    mu_idx_mask = np.random.choice(node_num, mask_num, replace=False)
    cn_idx_mask = np.random.choice(node_num, mask_num, replace=False)
    exp[exp_idx_mask], cn[mu_idx_mask], mu[cn_idx_mask] = 0, 0, 0
    return (exp, cn, mu)
