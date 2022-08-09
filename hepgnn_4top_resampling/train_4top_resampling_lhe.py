#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
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


sys.path.append("./python")
from model.allModel import *



parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--epoch', action='store', type=int, default=400,help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--lr', action='store', type=float, default=1e-4,help='Learning rate')
parser.add_argument('--seed', action='store', type=int, default=12345,help='random seed')
parser.add_argument('--fea', action='store', type=int, default=6, help='# fea')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')
parser.add_argument('--edge', action='store', type=int, default=0, help='# edge type')
parser.add_argument('--activation', action='store', type=int, default=0, help='# activation type')
models = ['GNN1layer_re', 'GNN2layer_re', 'GNN3layer_re', 'GNN1layer_re_lhe','GCN2','GCN_noedge','GCN2_no','GCN2_tanh','GCN3','GNN1layer']
parser.add_argument('--model', choices=models, default=models[0], help='model name')



args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
config['training']['learningRate'] = float(config['training']['learningRate'])
if args.seed: config['training']['randomSeed1'] = args.seed
if args.epoch: config['training']['epoch'] = args.epoch
if args.lr: config['training']['learningRate'] = args.lr


torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)


 
##### Define dataset instance #####
# from dataset.HEPGNNDataset_h5_LHE_resampling import *
# dset = HEPGNNDataset_h5_LHE_resampling()
from dataset.HEPGNNDataset_pt_classify_fourfeature_v5 import *
dset = HEPGNNDataset_pt_classify_fourfeature_v5()

for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize()

lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()), 'pin_memory':False}
# trnLoader = DataLoader(dset, batch_size=args.batch, shuffle=False, **kwargs)
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
exec('model = '+args.model+'(fea=args.fea, cla=args.cla, activation=args.activation)')
torch.save(model, os.path.join('result/' + args.output, 'model.pth'))

device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

##### Define optimizer instance #####
optm = optim.Adam(model.parameters(), lr=config['training']['learningRate'])



##### Start training #####
with open('result/' + args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()



from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    trn_loss = 0.
    nProcessed = 0
    optm.zero_grad()

    for i, data in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)
        

        weight = data.ww.float().to(device)
        
        pred = model(data)
      
        crit = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        loss = crit(pred.view(-1), weight)
        loss.backward()

        optm.step()
        optm.zero_grad()


        ibatch = len(weight)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch

        
        
        
        
    trn_loss /= nProcessed 

    print(trn_loss,'trn_loss')

    model.eval()
    val_loss = 0.
    nProcessed = 0
    for i, data in enumerate(tqdm(valLoader)):
        
        data = data.to(device)

        weight = data.ww.float().to(device)

        pred = model(data)
        
        crit = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        loss = crit(pred.view(-1), weight)

        ibatch = len(weight)
        nProcessed += ibatch
        val_loss += loss.item()*ibatch

            
            
    val_loss /= nProcessed

    print(val_loss,'val_loss')
    

        
    if epoch % 100 == 0:
        processState = model.to('cpu').state_dict()
        torch.save(processState, os.path.join('result/'+args.output,'weight' + str(epoch) + '.pth'))
        model.to(device)
    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)

    train['loss'].append(trn_loss)

    train['val_loss'].append(val_loss)


    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))


