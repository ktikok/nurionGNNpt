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

class EdgeConv(nn.Module):
    def __init__(self,**kwargs):
        super(EdgeConv, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
      
        self.conv1 = DynamicEdgeConv(MLP([self.fea, 64, 64, 64]), 16, max)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128, 128]), 16, max)
        self.conv3 = DynamicEdgeConv(MLP([2 *128, 256, 256]), 16, max)
        
        self.lin1 = MLP([128 + 64, 1024])

   
        
    def forward(self, data):
        pos, 
        x, pos, batch, edge_index = self.conv1(data)
        
        x, pos, batch = self.pool(x, pos, batch)
        
        out = self.fc(x)
#         x = self.fc(x)
#         out = F.log_softmax(x)
        return out
