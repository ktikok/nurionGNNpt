import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

from model.PointConv import PointConvNet
from model.PoolingNet import PoolingNet
def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class GNN1layer_re_lhe(nn.Module):
    def __init__(self,**kwargs):
        super(GNN1layer_re_lhe, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
      
        self.conv1 = PointConvNet(MLP([self.fea+3, 64, 128]))
#         self.pool = PoolingNet(MLP([128+3, 256]))
        self.pool = PoolingNet(MLP([128+3, 128]))

        self.fc = nn.Sequential(
            nn.Linear( 128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear( 64,  self.cla), nn.Softplus(),
        )
        
    def forward(self, data):
       
        x, batch, edge_index = self.conv1(data)
        
        x, batch = self.pool(x, batch)
        
        out = self.fc(x)
#         x = self.fc(x)
#         out = F.log_softmax(x)
        return out
