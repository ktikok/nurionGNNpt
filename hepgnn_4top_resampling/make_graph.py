import argparse
import numpy as np
import uproot
import h5py
import os
import torch
import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from torch_geometric.data import Data
import math
import pandas as pd
import csv
parser = argparse.ArgumentParser()
# parser.add_argument('input', nargs='+', action='store', type=str, help='input file name')
parser.add_argument('-i', '--input', action='store', type=str, help='input directory name', required=True)
parser.add_argument('-o', '--output', action='store', type=str, help='output directory name', required=True)

parser.add_argument('--deltaR', action='store', type=float, default=0, help='maximum deltaR to build graphs')
args = parser.parse_args()



##################################################################
# @numba.njit(nogil=True, fastmath=True, parallel=True)
# @numba.njit(nogil=True, fastmath=True, parallel=True)
def buildGraph(jetss_pt, jetss_eta, jetss_phi):
#     maxDR2 = 100
#     maxDR2 = 0 ## maximum deltaR value to connect two jets
    graphs = []
    g = []
    
    selJets = (jetss_pt > 35) & (np.fabs(jetss_eta) < 2.4)

    jets_eta = jetss_eta[selJets]
    jets_phi = jetss_phi[selJets]
#     jets_eta = jetss_eta
#     jets_phi = jetss_phi
    
    nJet = len(jets_eta)
#     prange = numba.prange
#     for i in prange(nJet):
#         for j in prange(i):
    for i in range(nJet):
        for j in range(i):
            dEta = jets_eta[i]-jets_eta[j]
            dPhi = jets_phi[i]-jets_phi[j]
            ## Move dPhi to [-pi,pi] range
            if   dPhi >= math.pi: dPhi -= 2*math.pi
            elif dPhi < -math.pi: dPhi += 2*math.pi
            ## Compute deltaR^2 and ask it is inside of our ball
            dR2 = dEta*dEta + dPhi*dPhi
            if dR2 > args.deltaR: continue
            g.append([i,j])
            
    graphs.append(g)
    return graphs
##################################################################


res = []
for root, dirs, files in os.walk(args.input):
    rootpath = os.path.join(os.path.abspath(args.input), root)
        
    for file in files:
        filepath = os.path.join(rootpath, file)
            
        filetype = filepath.split('.')[-1]
        if filetype != 'pt': continue
  
        res.append(filepath)
##################################################################    
graph_gen = {'file_csv':[], 'pre_event':[], 'after_event':[]}



for i in range(len(res)):
    
    file_name = res[i]
     
   # cla = file_name.split('/')[-1]
    cla = file_name.split('/')[-1].split('.')[-2]
    #save_f_name = cla + '_' + num
    
    #claa = cla
    #if cla == 'QCDBkg': cla = cla + '/' + file_name.split('/')[-3]
    #else: cla = cla
    
    save_path = args.output
    
    
    
    
    if not os.path.exists(save_path): os.makedirs(save_path)
        
  
    save_file_name = save_path + cla + '.pt'
    print(save_file_name)

    print('-----loading---' + cla + '--data-----')
    
### load data
    try:
        r_data = torch.load(file_name)
#     r_data = uproot.open(file_name)["Delphes"]
    except KeyError:
        print("delphes 없음")
        pass
    
    print('-----root event num =' + str(len(r_data)))
    print('-----making-----')
## make data
    datalist=[]
    for j in range(len(r_data)):

        JetPt = r_data[j].x[:,0]
        JetEta = r_data[j].x[:,1]
        JetPhi = r_data[j].x[:,2]
        edge = buildGraph(JetPt, JetEta, JetPhi)
        
        if len(edge[0]) == 0:
            edge_index = [[],[]]
        else:
            edge_index = np.concatenate((np.array(edge[0]).T, np.array(edge[0]).T[[1,0]]), axis = 1)
      
        
        edge_t = edge_index
        
        edges = torch.Tensor(edge_t).type(dtype = torch.long)
    

        label = torch.Tensor([0])
        

	
        data = PyGData(x = r_data[j].x, pos = r_data[j].pos, edge_index = edges, y = r_data[j].y)
        datalist.append(data)
    print('----remain event num =' + str(len(datalist)))
    graph_gen['after_event'].append(str(len(datalist)))
    print('---- saving--' + cla)
    torch.save(datalist, save_file_name)

