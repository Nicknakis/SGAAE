from torch_geometric.nn import GINConv,GCNConv,GraphConv,SignedGCN
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch.nn.functional as F

import torch.nn as nn
import torch



class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
       
        self.fc0=nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
       
        self.convs.append(GCNConv(hidden_dim, hidden_dim))                        
        for layer in range(n_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim)) 
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2=nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu=nn.ReLU()
       

    def forward(self, x,edge_index,batch,pool=False,training=True):
        x=self.fc0(x)
        x=self.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.relu(x)
            x = F.dropout(x, self.dropout, training=training)
        if pool:
            x = global_add_pool(x,batch)
        if x.shape[0]>1:
            x = self.bn1(x)
        else:
            x = self.bn2(x)
        # x = self.fc1(self.relu(x))
        # x = self.fc2(x)

        return x
    
