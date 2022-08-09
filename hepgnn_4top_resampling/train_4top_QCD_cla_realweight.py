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

models = ['GNN1layer', 'GNN2layer', 'GNN3layer',
         'GNN1layer_mul', 'GNN2layer_mul', 'GNN3layer_mul']
parser.add_argument('--model', choices=models, default=models[2], help='model name')



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
from dataset.HEPGNNDataset_pt_classify_fourfeature_v2 import *
dset = HEPGNNDataset_pt_classify_fourfeature_v2()
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
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
exec('model = '+args.model+'(fea=args.fea, cla=args.cla)')
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
train = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    optm.zero_grad()

    for i, data in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)
        if args.cla == 3:
            label = data.y.long().to(device=device)
        else:
            label = data.y.float().to(device=device)
        
            
        scale = data.ss.float().to(device)
        weight = data.rw.float().to(device)
        scaledweight = weight*scale
#         scaledweight = torch.abs(scaledweight)
        pred = model(data)
 
        if args.cla ==3:
            crit = torch.nn.CrossEntropyLoss(reduction='none')
            
            loss = crit(pred, label)
            loss = loss * scaledweight
            loss.mean().backward()

            optm.step()
            optm.zero_grad()


            ibatch = len(label)
            nProcessed += ibatch

            pred = torch.argmax(pred, 1)
            trn_loss += loss.mean().item()*ibatch
            trn_acc += accuracy_score(label.to('cpu'), pred.to('cpu'), 
                                      sample_weight=scaledweight.to('cpu'))*ibatch
        else:
            crit = torch.nn.BCEWithLogitsLoss(weight=scaledweight)
      
            loss = crit(pred.view(-1), label)
            loss.backward()

            optm.step()
            optm.zero_grad()

            label = label.reshape(-1)
            ibatch = len(label)
            nProcessed += ibatch
            trn_loss += loss.item()*ibatch
        
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), 
                                      sample_weight=scaledweight.to('cpu'))*ibatch
        
        
        
        
        
    trn_loss /= nProcessed 
    trn_acc  /= nProcessed
    print(trn_loss,'trn_loss')
    print(trn_acc,'trn_acc')
    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    for i, data in enumerate(tqdm(valLoader)):
        
        data = data.to(device)
        if args.cla == 3:
            label = data.y.long().to(device=device)
        else:
            label = data.y.float().to(device=device)
        
        scale = data.ss.float().to(device)
        weight = data.rw.float().to(device)
        scaledweight = weight*scale
#         scaledweight = torch.abs(scaledweight)
        pred = model(data)
        if args.cla == 3:
            crit = nn.CrossEntropyLoss(reduction='none')
            loss = crit(pred, label)
            loss = loss * scaledweight




            ibatch = len(label)
            nProcessed += ibatch

            pred=torch.argmax(pred,1)
            val_loss += loss.mean().item()*ibatch
            val_acc += accuracy_score(label.to('cpu'), pred.to('cpu'), 
                                      sample_weight=scaledweight.to('cpu'))*ibatch
        else:
            crit = torch.nn.BCEWithLogitsLoss(weight=scaledweight)
            loss = crit(pred.view(-1), label)

            label = label.reshape(-1)
            ibatch = len(label)
            nProcessed += ibatch
            val_loss += loss.item()*ibatch
       
            val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0), 
                                      sample_weight=scaledweight.to('cpu'))*ibatch
            
            
            
            
    val_loss /= nProcessed
    val_acc  /= nProcessed
    print(val_loss,'val_loss')
    print(val_acc,'val_acc')
    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)

    train['loss'].append(trn_loss)
    train['acc'].append(trn_acc)
    train['val_loss'].append(val_loss)
    train['val_acc'].append(val_acc)

    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))


