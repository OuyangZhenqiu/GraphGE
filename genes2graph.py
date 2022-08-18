import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

def ensp_to_hugo_map():
    with open('./data/9606.protein.info.v11.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map


def hugo_to_ncbi_map():
    with open('./data/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

    return hugo_map

import pickle
def get_ppi_edge_index(thresh=0.95):
    save_path = './data/edge_index_PPI_{}.npy'.format(thresh)
    if not os.path.exists(save_path):
        # gene_list
        with open("./data/gene_id.list", 'rb') as f:
            gene_list = pickle.load(f)
        gene_list = [int(gene) for gene in gene_list]

        # load STRING
        ensp_map = ensp_to_hugo_map()
        hugo_map = hugo_to_ncbi_map()
        string = pd.read_csv('./data/9606.protein.links.detailed.v11.0.txt', sep=' ')
        humannet = pd.read_csv('./data/HumanNet-PI.tsv', sep='\t')

        # edge_index
        selected_edges = string['combined_score'] > (thresh * 1000)
        string_edge_list = string[selected_edges][["protein1", "protein2"]].values.tolist()
        string_edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in string_edge_list if
                     edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]

        string_edge_list = [[hugo_map[edge[0]], hugo_map[edge[1]]] for edge in string_edge_list if
                     edge[0] in hugo_map.keys() and edge[1] in hugo_map.keys()]
        
        string_edge_index = []
        for i in string_edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):
                string_edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
                string_edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        string_edge_index = list(set(string_edge_index))
        string_edge_index = np.array(string_edge_index, dtype=np.int64).T
        
        
        humannet_edge_list = humannet[['EntrezGeneID1', 'EntrezGeneID2']].values.tolist()
        
        humannet_edge_index = []
        for i in humannet_edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):
                humannet_edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
                humannet_edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        humannet_edge_index = list(set(humannet_edge_index))
        humannet_edge_index = np.array(humannet_edge_index, dtype=np.int64).T

        pickle.dump((string_edge_index, humannet_edge_index), open(save_path, 'wb'))
    else:
        string_edge_index, humannet_edge_index = pickle.load(open(save_path, 'rb'))

    return string_edge_index, humannet_edge_index

if __name__ == '__main__':
    edge_index = get_ppi_edge_index(thresh=0.95)
