"""
models.py — Model Definitions Using PyTorch Geometric
=============================================================
External Code Citations:
1. PyTorch Geometric (pyg-team/pytorch_geometric, release tag v2.6.0):
   - Uses `GCNConv` and `SGConv` layers and `examples/gcn.py` starter structure.
2. Official GCN Repo (tkipf/gcn, commit 39a4089, gcn/models.py & gcn/train.py; tkipf/pygcn master):
   - Kipf & Welling (ICLR 2017) apply L2 weight decay (5e-4) strictly to the
     first layer weights W_1, with dropout 0.5 and hidden dimension 16.
3. Official SGC Repo (Tiiiger/SGC, commit 2c7a272, citation.py & tuning.py):
   - Wu et al. (ICML 2019) precompute S^K X once (`SGConv(..., cached=True)`)
     followed by a single linear layer with dataset-tuned L2 weight decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv


class GCN(nn.Module):
    """Multi-layer GCN (default L=2) using PyG GCNConv."""
    def __init__(self, in_channels, out_channels, hidden_channels=16, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, cached=True))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class SGC(nn.Module):
    """Simple Graph Convolution using PyG SGConv (cached=True precomputes S^K X)."""
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class MLP(nn.Module):
    """2-layer MLP baseline on node features X (ignores graph edges)."""
    def __init__(self, in_channels, out_channels, hidden_channels=16, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, edge_index=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)
