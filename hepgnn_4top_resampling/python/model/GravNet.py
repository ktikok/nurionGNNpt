import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GravNetConv
import numpy as np
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData


class GravNet(nn.Module):

    def __init__(self,**kwargs):
        super(GravNet, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        
        self.GravNetconv1 = GravNetConv(self.fea, 36, 4, 22, 40)
        self.GravNetconv2 = GravNetConv(36, 36, 4, 22, 40)
        self.GravNetconv3 = GravNetConv(36, 48, 4, 22, 40)
        self.GravNetconv4 = GravNetConv(48, 48, 4, 22, 40)
   
        
        self.fc = nn.Sequential(
            nn.Linear( self.GravNetconv1.out_channels + self.GravNetconv2.out_channels
                              + self.GravNetconv3.out_channels + self.GravNetconv4.out_channels, 64 ), nn.ReLU(),
            nn.Linear( 64,  self.cla),
        )
        
 
    def forward(self, data):
#         print(data,'dddddd')
        
#         dat = PyGData(x = data.x, batch = data.batch)
#         print(dat,'after')
        print(data.batch,'dddd')
        x1 = self.GravNetconv1(data.x, data.batch)
        x2 = self.GravNetconv2(x1)
        x3 = self.GravNetconv3(x2)
        x4 = self.GravNetconv4(x3)
        out = self.fc(torch.cat([x1,x2,x3,x4], dim=1))
        print(torch.cat([x1,x2,x3,x4],dim=1).shape,'cat')
        print(out.shape,'out')
        
        return out


