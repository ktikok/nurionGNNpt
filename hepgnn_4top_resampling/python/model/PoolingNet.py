import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

class PoolingNet(nn.Module):
    def __init__(self, net):
        super(PoolingNet, self).__init__()
        self.net = net

    def forward(self, x, pos, batch):
#         print(x.shape)
#         print(pos.shape)
        x = self.net(torch.cat([x, pos], dim=1))
        x = PyG.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
