#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import DataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Data
import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/delphes'
graph_path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/graph/'
output_path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/h5/'

# path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex1/220324_055935/delphes'
# graph_path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex1/220324_055935/graph/'
# output_path = '/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex1/220324_055935/h5/'

res = [] 
for root, dirs, files in os.walk(graph_path):
    rootpath = os.path.join(os.path.abspath(graph_path), root)
    for file in files:
        filepath = os.path.join(rootpath, file)
        
        filetype = filepath.split('.')[-1]
        if filetype != 'h5': continue
  
        res.append(filepath)
    
    
    
res = res[-81:-71]
for files in tqdm(range(len(res))):
    file = h5py.File(res[files],'r')

    datalist = []
    for i in range(len(file['events']['m'])):
    # for i in range(1000):
        feature = torch.Tensor((np.ones(len(file['events']['m'][i]))*np.array(file['events']['id'][i])).reshape(-1,1))

        if len(feature[(torch.abs(feature) > 10) & (torch.abs(feature)<20)]) >0:
            continue
        else:
            status = torch.Tensor(np.array(file['events']['status'][i]).reshape(-1,1))

            weight = torch.Tensor((np.ones(len(file['events']['m'][i]))*np.array(file['events']['weight'][i])).reshape(-1,1))


            px = torch.Tensor(np.array(file['events']['px'][i]).reshape(-1,1))
            py = torch.Tensor(np.array(file['events']['py'][i]).reshape(-1,1))
            pz = torch.Tensor(np.array(file['events']['pz'][i]).reshape(-1,1))

            edge = torch.Tensor(np.concatenate((np.array(file['graphs']['edge2'])[i].reshape(-1,1).T,np.array(file['graphs']['edge1'])[i].reshape(-1,1).T),axis=0)).type(dtype = torch.long)
            data = PyGData(x = feature,edge_index = edge, weight = weight)
            data.st = status
            data.px = px
            data.py = py
            data.pz = pz
 
 

            datalist.append(data)
        file_name = 'hadr' + res[files].split('/')[-1][:-3] + '.pt'
        torch.save(datalist, file_name)