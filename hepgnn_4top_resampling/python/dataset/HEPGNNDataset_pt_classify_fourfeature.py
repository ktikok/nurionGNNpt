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



   
        
class HEPGNNDataset_pt_classify_fourfeature(PyGDataset):
    def __init__(self, **kwargs):
        super(HEPGNNDataset_pt_classify_fourfeature, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "label", "fileIdx","sumweight"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1

        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)

        
        data = self.graphList[fileIdx][idx]

        
        label = self.labelList[fileIdx][idx]
        weight = self.weightList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]
        procIdxs = self.procList[fileIdx][idx]
    
     
        data.ww = weight.item()

        data.y = label
        
        data.ss = rescale.item()

        return data
    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".pt"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvent':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx, 'sumweight':0,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)




    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
    def initialize(self):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())


        self.graphList = []
        self.labelList = []
        self.weightList = []
        self.rescaleList = []
        self.procList = []
        
    
        
        
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
#             print("Loading files... (%d/%d) %s" % (i+1,nFiles,fName), end='\r')
#             f = h5py.File(fName, 'r', libver='latest', swmr=True)['group']
            f = torch.load(fName)
            
#             nEvent = len(f['pos'].get('pos'))
            nEvent = len(f)
             
            self.sampleInfo.loc[i, 'nEvent'] = nEvent

            
                
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*label
            self.labelList.append(labels)
            
            weight = self.sampleInfo['weight'][i]

            graphlist = []
            weightlist = []
            weightslist = 0
            
          
            for j in range(nEvent):
                weights = f[j].x[:,6][0]

                ### weights = w_ik
                ### weight = sigma_k / M_k
                weightlist.append(weights*weight)  
                weightslist = weightslist + weights

                f[j].x = f[j].x[:,:4]
           
                graphlist.append(f[j])
            
            
            
            sumw = weightslist
#             print(sumw, 'sumw')
            self.sampleInfo.loc[i, 'sumweight'] = sumw
            
            self.weightList.append(weightlist)
            self.graphList.append(graphlist)
            self.rescaleList.append(torch.ones(nEvent, dtype=torch.float32, requires_grad=False))
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*procIdx)
        print("")
        
        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

        print('---------')
        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for label in self.sampleInfo['label']:
            label = int(label)
            ### w = sigma_k / M_k
            w = self.sampleInfo[self.sampleInfo.label==label]['weight']
            ### e = n_k
            e = self.sampleInfo[self.sampleInfo.label==label]['nEvent']
            ### sw = m_k
            sw = self.sampleInfo[self.sampleInfo.label==label]['sumweight']
            ### sumWByLabel = sum_k[(sigma_k/M_k)*m_k]
            sumWByLabel[label] = (w*sw).sum()
            ### sumEByLabel = sum(n_k)
            sumEByLabel[label] = e.sum()


        
        ### maxSumELabel -> signal sum(n_k)
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
  
        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
   
            for l in label: ## this loop runs only once, by construction.
  
                ### rescale ---> 
                ### [(1/sumWByLabel)*sumEByLabel]*[maxSumELabel/sumEByLabel]
                self.rescaleList[fileIdx] *= ((1/sumWByLabel[l])*sumEByLabel[l])*(sumEByLabel[maxSumELabel]/sumEByLabel[l])

                break ## this loop runs only once, by construction. this break is just for a confirmation
                print('-'*80)
        for label in sumWByLabel.keys():
            print("Label=%d sumE=%d, sumW=%g" % (label, sumEByLabel[label], sumWByLabel[label]))
            

       
        print('-'*80)
        self.isLoaded = True
