import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

class PointConvNet(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet, self).__init__()
        self.conv = PyG.PointConv(net)
        
    def forward(self, data, batch=None):
        
        x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index
        
        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
    
class PointConvNet2(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet2, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, x, data, batch=None):
        pos, batch, edge_index = data.pos, data.batch, data.edge_index
        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
    
    
