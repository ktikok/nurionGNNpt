#!/usr/bin/env python
# coding: utf-8
# %%
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math



   
        
class HEPGNNDataset_h5_LHE_resampling(PyGDataset):
    def __init__(self, **kwargs):
        super(HEPGNNDataset_h5_LHE_resampling, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight",  "fileIdx"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1

        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)
     
        
    
        weight = self.weightList[fileIdx][idx]
        procIdxs = self.procList[fileIdx][idx]
               
        feats = torch.Tensor(self.feaList[fileIdx][idx])
        edges = torch.Tensor(self.edgeList[fileIdx][idx])
        edges = edges.type(dtype = torch.long)
        
    
        data = PyGData(x = feats, edge_index = edges)
        data.ww = weight.item()


        return data
    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvent':0,
               
                'fileName':fName, 'fileIdx':fileIdx,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)




    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
    def initialize(self,edge):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())



        self.weightList = []
      
        self.procList = []
        self.feaList = []
        self.edgeList = []
        
        
        
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):

            f = h5py.File(fName, 'r', libver='latest', swmr=True)
            
            nEvent = len(f['events']['m'])
     
            self.sampleInfo.loc[i, 'nEvent'] = nEvent

           
            weight = self.sampleInfo['weight'][i]

            graphlist = []
            weightlist = []
            weightslist = 0

        
            f_m = f['events']['m']
            f_px = f['events']['px']
            f_py = f['events']['py']
            f_pz = f['events']['pz']
            f_id = f['events']['id']
            f_st = f['events']['status']
      
       
                        
            
            

            f_weight = f['events']['weight']
   
            if edge == 0:               
                f_edge1 = f['graphs']['edge1']
                f_edge2 = f['graphs']['edge2']
            elif edge == 1:
                f_edge1 = f['graphs']['edgeColor1']
                f_edge2 = f['graphs']['edgeColor2']
    
            f_fea_list = []
            f_edge_list = []
            
            for j in range(nEvent):
     
                f_fea_reshape = torch.cat((torch.from_numpy(f_m[j][f_st[j]!=-1]).reshape(-1,1),torch.from_numpy(f_px[j][f_st[j]!=-1]).reshape(-1,1),torch.from_numpy(f_py[j][f_st[j]!=-1]).reshape(-1,1),torch.from_numpy(f_pz[j][f_st[j]!=-1]).reshape(-1,1)),1).float()   
                f_edge_reshape = torch.tensor([[],[]])
                if edge ==2:
                    f_edge_reshape = torch.tensor([[],[]])
                else:
                    f_edge_reshape = torch.cat((torch.from_numpy(f_edge1[j]).reshape(1,-1),torch.from_numpy(f_edge2[j]).reshape(1,-1)),0).float()
                    
                
                    f_edge_reshape = np.array(f_edge_reshape)
                    f_edge_reshape = np.delete(f_edge_reshape,np.where(f_edge_reshape[0] == 1),axis=1)
                    f_edge_reshape = np.delete(f_edge_reshape,np.where(f_edge_reshape[0] == 0),axis=1)
                    
                    f_edge_reshape = np.delete(f_edge_reshape,np.where(f_edge_reshape[1] == 1),axis=1)
                    f_edge_reshape = np.delete(f_edge_reshape,np.where(f_edge_reshape[1] == 0),axis=1)
                    f_edge_reshape = torch.Tensor(f_edge_reshape).float()
                    f_edge_reshape = f_edge_reshape - 2
               
                weights = f_weight[j]
                weights = weights/np.abs(weights)
                weightlist.append(weights)  
                
                
                
                
                f_fea_list.append(f_fea_reshape)
              
                f_edge_list.append(f_edge_reshape)
       
            self.weightList.append(weightlist)
 
            self.feaList.append(f_fea_list)
            self.edgeList.append(f_edge_list)
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*procIdx)
        print("")
        
        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

          

       
        print('-'*80)
        self.isLoaded = True
