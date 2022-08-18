import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.data import Batch, Data

class GeneEncoder(torch.nn.Module):
    
    def __init__(self, args):
        super(GeneEncoder, self).__init__()
        self.emb_dim = args.gene_hidden_dim // 2
        self.gene_num = args.gene_num
        self.weight_exp = torch.nn.Parameter(torch.Tensor(self.gene_num, self.emb_dim))
        self.bias_exp = torch.nn.Parameter(torch.Tensor(self.gene_num, self.emb_dim))
        self.weight_mu = torch.nn.Parameter(torch.Tensor(self.gene_num, 2, self.emb_dim))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(self.gene_num, self.emb_dim))
        torch.nn.init.xavier_normal_(self.weight_exp.data)
        torch.nn.init.xavier_normal_(self.bias_exp.data)
        torch.nn.init.xavier_normal_(self.weight_mu.data)
        torch.nn.init.xavier_normal_(self.bias_mu.data)
        
    def forward(self, cell_graph):
        x = cell_graph.x
        pos = cell_graph.pos
        mu = F.one_hot(x[...,1].long(), num_classes=2).float().unsqueeze(1)
        exp = x[...,0].unsqueeze(-1)

        exp_weight_embedding = self.weight_exp[pos]
        exp_bias_embedding = self.bias_exp[pos]
        mu_weight_embedding = torch.matmul(mu, self.weight_mu[pos]).squeeze(1)
        mu_bias_embedding = self.bias_mu[pos]

        exp_embedding = exp_weight_embedding*exp+exp_bias_embedding
        mu_embedding = (mu_weight_embedding+mu_bias_embedding)

        embedding = torch.cat([exp_embedding, mu_embedding], dim=-1)
        return embedding
    
    def cos_sin(self, x):
        x = torch.cat([torch.cos(2*torch.pi*x), torch.sin(2*torch.pi*x)], dim=-1)
        return x

class Cell_Grapher(nn.Module):
    def __init__(self, args):
        super(Cell_Grapher, self).__init__()
        self.device = args.device
        self.gene_encoder = GeneEncoder(args)
        self.gene_num = args.gene_num
        self.gene_hidden_dim = args.gene_hidden_dim
        self.graph_num_layer = args.cell_graph_num_layer
        self.cell_graph_hidden_dim = args.cell_graph_hidden_dim
        self.cell_hidden_dim = args.cell_hidden_dim
        self.dropout_ratio = args.dropout
        self.convs = nn.ModuleList([
            GATConv(self.cell_graph_hidden_dim, self.cell_graph_hidden_dim) if layer else GATConv(self.gene_hidden_dim, self.cell_graph_hidden_dim) for layer in range(self.graph_num_layer)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.cell_graph_hidden_dim)  for _ in range(self.graph_num_layer)])
        self.ffn = nn.Sequential(
            nn.Linear(self.cell_graph_hidden_dim*2, self.cell_hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, cell_graph):
        x = self.gene_encoder(cell_graph)
        for i in range(self.graph_num_layer):
            x = F.relu(self.convs[i](x, cell_graph.edge_index))
            x = self.bns[i](x)
        cell_representation = torch.cat([global_mean_pool(x, cell_graph.batch), global_max_pool(x, cell_graph.batch)], -1)
        cell_representation = self.ffn(cell_representation)
        return cell_representation

class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.cell_encoder = Cell_Grapher(args)
        self.regression = nn.Sequential(
            nn.Linear(args.cell_hidden_dim, args.cell_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.cell_hidden_dim, args.output_dim) 
        )
             
    
    def forward(self, data):
        cell = self.cell_encoder(data)
        x = self.regression(cell)
        return x