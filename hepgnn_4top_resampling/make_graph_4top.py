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

def countNBjets(JetBTag):
    nEvent = int(len(JetBTag))
    NBjets = []
    for i in range(nEvent):
        selBjets = (JetBTag[i] > 0.5)
        numBjets = np.sum(selBjets)
        NBjets.append(numBjets)
    return np.array(NBjets, dtype=np.dtype('int64'))
##################################################################

res = []
for root, dirs, files in os.walk(args.input):
    rootpath = os.path.join(os.path.abspath(args.input), root)
        
    for file in files:
        filepath = os.path.join(rootpath, file)
            
        filetype = filepath.split('.')[-1]
        if filetype != 'root': continue
  
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
        
    graph_gen['file_csv'].append(cla)
    save_file_name = save_path + cla + '.pt'
    
    print('-----loading---' + cla + '--data-----')
    
### load data
    try:
        r_data = uproot.open(file_name)["Delphes"] 
#     r_data = uproot.open(file_name)["Delphes"]
    except KeyError:
        print("delphes 없음")
        pass
    
    weights = np.asarray(r_data["Event"]["Event.Weight"])
    Muon_size = np.asarray(r_data["Muon_size"])
    Elec_size = np.asarray(r_data["Electron_size"])
    
    JetPt = np.asarray(r_data["Jet.PT"])
    JetEta = np.asarray(r_data["Jet.Eta"])
    JetPhi = np.asarray(r_data["Jet.Phi"])
    JetMass = np.asarray(r_data["Jet.Mass"])
    JetBTag = np.asarray(r_data["Jet.BTag"])
    
    FMass = np.asarray(r_data["FatJet.Mass"])
    FPt = np.asarray(r_data["FatJet.PT"])
    FEta = np.asarray(r_data["FatJet.Eta"])

    
    pos = []
    x_fea = []
    edge_t = []
    datalist = []
    graph_gen['pre_event'].append(str(len(FPt)))
    print('-----root event num =' + str(len(FPt)))
    print('-----making-----')
## make data    
    for j in range(len(FPt)):

        numLepton = (Muon_size[j] + Elec_size[j])
        if numLepton > 0 : continue
            
        selJets = (JetPt[j] > 35) & (np.fabs(JetEta[j]) < 2.4)
        if np.sum(selJets) < 9: continue 
        ht = np.sum(JetPt[j][selJets])
        if  ht < 700: continue
        NBjets = countNBjets(JetBTag[j])
        x_fea = torch.Tensor([np.array(JetPt[j]), np.array(JetEta[j]), np.array(JetPhi[j]), np.array(JetMass[j]), np.array(JetBTag[j]), NBjets, np.ones((len(JetPt[j]),))*np.array(weights[j])]).transpose(0,1)
#         x_fea = torch.Tensor([np.array(JetPt[j]), np.array(JetEta[j]), np.array(JetPhi[j]), np.array(JetMass[j]), np.array(JetBTag[j]), NBjets]).transpose(0,1)
        pos = torch.Tensor([xx for xx in np.dstack([np.cos(JetPhi[j]), np.sin(JetPhi[j]), JetEta[j]])][0])
        
        edge = buildGraph(JetPt[j], JetEta[j], JetPhi[j])
        
        if len(edge[0]) == 0:
            edge_index = [[],[]]
        else:
            edge_index = np.concatenate((np.array(edge[0]).T, np.array(edge[0]).T[[1,0]]), axis = 1)
      
        
        edge_t = edge_index
        
        edges = torch.Tensor(edge_t).type(dtype = torch.long)
    
       
        #if claa == 'QCDBkg': 
        #    label = torch.Tensor([0])
        #else: 
        label = torch.Tensor([2])
        

	
        data = PyGData(x = x_fea, pos = pos, edge_index = edges, y = label)
        datalist.append(data)
    print('----remain event num =' + str(len(datalist)))
    graph_gen['after_event'].append(str(len(datalist)))
    print('---- saving--' + cla)
    torch.save(datalist, save_file_name)
    with open(os.path.join(args.output,'graph_gen.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = graph_gen.keys()
        writer.writerow(keys)
        for row in zip(*[graph_gen[key] for key in keys]):
            writer.writerow(row)
