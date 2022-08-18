import os
import os.path as osp
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import numpy as np
import pickle
import json
import copy
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from collections import namedtuple
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import scale
from genes2graph import get_ppi_edge_index
from itertools import repeat, product
from collections.abc import Sequence
Batch_GNN = namedtuple("Batch_GNN", ['cell_graph', 'ic50'])

with open("./data/GDSC_dataset.dict", "rb") as f:
        gdsc_data_dict = pickle.load(f)
multi_task_df, cell_exp_df, cell_mu_df = gdsc_data_dict['multi_task_df'], gdsc_data_dict['cell_exp'], gdsc_data_dict['cell_mu']

def get_cell_graph(args, cell_exp, cell_mu):
    string_edge_index, humannet_edge_index = get_ppi_edge_index(args.cell_edge_index_thresh)
    string_edge_index = torch.tensor(string_edge_index).to(torch.long)
    cell_exp = cell_exp[...,np.newaxis]
    cell_mu = cell_mu[...,np.newaxis]

    cell = np.concatenate((cell_exp, cell_mu), axis=-1)
    cell_graph_list = []
    for i in range(cell.shape[0]):  
        cell_graph_list.append(Data(x=torch.tensor(cell[i].astype(float), dtype=torch.float), pos=torch.arange(len(cell[i])), edge_index=string_edge_index))
    return cell_graph_list

class GNN_dataset(Dataset):
    def __init__(self, args, index):
        super(GNN_dataset, self).__init__()
        
        self._indices = index
        self.ic50 = multi_task_df.iloc[:, :].values
        self.cell_graphs = get_cell_graph(args, cell_exp_df.iloc[:, 4:].values, cell_mu_df.iloc[:, 4:].values)
    
    def __getitem__(self, idx):
        cell_idx = self.indices()[idx]
        return self.cell_graphs[cell_idx], self.ic50[cell_idx]
    
    def indices(self):
        return range(len(self.ic50)) if self._indices is None else self._indices
    
    def shuffle(self, return_perm: bool = False):
        indices = self.indices()
        perm = torch.randperm(len(self))
        dataset = copy.copy(self)
        dataset._indices = [indices[i] for i in perm]
        return (dataset, perm) if return_perm is True else dataset
    
    def __len__(self):
        return len(self.indices())

    def collate(samples):
        cell_graphs,  ic50 = map(list, zip(*samples)) 
        batch_cell_graphs = Batch.from_data_list(cell_graphs)
        batch_ic50 = torch.from_numpy(np.vstack(ic50).astype(np.float32))
        return Batch_GNN(cell_graph=batch_cell_graphs, ic50=batch_ic50)

def load_skfold_dataloader_list(args):

    total_index = list(range(len(multi_task_df)))
    my_dataset = GNN_dataset
    my_collate_fn = GNN_dataset.collate
        
    sfolder = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    train_loader_list, val_loader_list, test_loader_list = [], [], []
    for train_val_index, test_index in sfolder.split(total_index, cell_exp_df['cancer_type'].tolist()):
        train_index, val_index = train_test_split(train_val_index, test_size=len(test_index), random_state=args.seed, stratify=cell_exp_df['cancer_type'].iloc[train_val_index])
        train_set = my_dataset(args, train_index)
        val_set = my_dataset(args, val_index)
        test_set = my_dataset(args, test_index)
        train_loader_list.append(DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn))
        val_loader_list.append(DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn))
        test_loader_list.append(DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn))
    return train_loader_list, val_loader_list, test_loader_list

def load_kfold_dataloader_list(args):
    total_index = list(range(len(multi_task_df)))
    my_dataset = GNN_dataset
    my_collate_fn = GNN_dataset.collate
        
    folder = KFold(n_splits=5, random_state=args.seed, shuffle=True)
    train_loader_list, val_loader_list, test_loader_list = [], [], []
    for train_val_index, test_index in folder.split(total_index):
        train_index, val_index = train_test_split(train_val_index, test_size=len(test_index), random_state=args.seed)
        train_set = my_dataset(args, train_index)
        val_set = my_dataset(args, val_index)
        test_set = my_dataset(args, test_index)
        train_loader_list.append(DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn))
        val_loader_list.append(DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn))
        test_loader_list.append(DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn))
    return train_loader_list, val_loader_list, test_loader_list
