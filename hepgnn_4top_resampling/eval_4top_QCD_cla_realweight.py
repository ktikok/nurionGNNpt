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
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.append("./python")

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')

parser.add_argument('-a', '--all', action='store_true', help='use all events for the evaluation, no split')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')
parser.add_argument('--weight', action='store', type=int, default=0, help='resample weight')

parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--seed', action='store', type=int, default=12345, help='random seed')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
if args.seed: config['training']['randomSeed1'] = args.seed

sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

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
kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
          'batch_size':args.batch, 'pin_memory':False}

if args.all:
    testLoader = DataLoader(dset, **kwargs)
else:
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
    #testLoader = DataLoader(trnDset, **kwargs)
    #testLoader = DataLoader(valDset, **kwargs)
    testLoader = DataLoader(testDset, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.allModel import *

model = torch.load('result/' + args.output+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location='cpu'))
if args.cla == 1:
    model.fc.add_module('output', torch.nn.Sigmoid())


device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

dd = 'result/' + args.output + '/train.csv'

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['figure.figsize'] = (4*2, 3.5*3)

ax1 = plt.subplot(3, 2, 1, yscale='log', ylabel='Loss(train)', xlabel='epoch')
ax2 = plt.subplot(3, 2, 2, yscale='log', ylabel='Loss(val)', xlabel='epoch')
ax3 = plt.subplot(3, 2, 3, ylabel='Accuracy(train)', xlabel='epoch')
ax4 = plt.subplot(3, 2, 4, ylabel='Accuracy(val)', xlabel='epoch')
#ax1.set_ylim([3e-2,1e-0])
#ax2.set_ylim([3e-2,1e-0])
ax3.set_ylim([0.50,1])
ax4.set_ylim([0.50,1])
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(which='major', axis='both', linestyle='-.')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlim([0,400])
lines, labels = [], []

dff = pd.read_csv(dd)

label = dd.split('/')[-1].replace('__', ' ').replace('_', '=')

l = ax1.plot(dff['loss'], '.-', label=label)
ax2.plot(dff['val_loss'], '.-', label=label)

ax3.plot(dff['acc'], '.-', label=label)
ax4.plot(dff['val_acc'], '.-', label=label)

lines.append(l[0])
labels.append(label)

ax5 = plt.subplot(3,1,3)
ax5.legend(lines, labels)
ax5.axis('off')

plt.tight_layout()
plt.savefig('result/' + args.output + '/' + args.output + '_acc_loss.png', dpi=300)

#plt.show()
#plt.close()
plt.clf()



##### Start evaluation #####
from tqdm import tqdm
labels, preds = [], []
weights = []
scaledWeights = []
procIdxs = []
fileIdxs = []
idxs = []
features = []
batch_size = []
real_weights = []
scales = []

eval_resampling = []
eval_real = []
model.eval()
val_loss, val_acc = 0., 0.
# for i, (data, label0, weight, rescale, procIdx, fileIdx, idx, dT, dVertex, vertexX, vertexY, vertexZ) in enumerate(tqdm(testLoader)):
for i, data in enumerate(tqdm(testLoader)):
    
    data = data.to(device)
    label = data.y.float().to(device=device)
    scale = data.ss.float().to(device)
    weight = data.rw.float().to(device)
    real_weight = data.rw.float().to(device)
    
    eval_resampling_weight = data.es.float().to(device)
    eval_real_weight = data.er.float().to(device)
#     scaledweight = weight*scale
#     scaledweight = torch.abs(scaledweight)
    
    
    pred = model(data)
    
    
    
#     pred = F.log_softmax(pred, dim=1)   
    
    scales.extend([x.item() for x in scale])
    labels.extend([x.item() for x in label])
#     weights.extend([x.item() for x in weight])
#     preds.extend([x.item() for x in pred.view(-1)])
#     scaledWeights.extend([x.item() for x in (scaledweight).view(-1)])
    
    
#     real_weights.extend([x.item() for x in real_weight])
    real_weights.append(real_weight.to("cpu").numpy()[0])
    
    eval_resampling.append(eval_resampling_weight.to("cpu").numpy()[0])
    eval_real.append(eval_real_weight.to("cpu").numpy()[0])

    weights.extend([x.item() for x in weight])
    preds.extend([x.item() for x in pred.view(-1)])
   
    features.extend([x.item() for x in data.x.view(-1)])
    batch_size.append(data.x.shape[0])
    

df = pd.DataFrame({'prediction':preds, 'weight':weights, 'label':labels,'real_weight':real_weights,'scale':scales,'eval_resampling':eval_resampling,'eval_real':eval_real})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)


df3 = pd.DataFrame({'feature':features})
fPred3 = 'result/' + args.output + '/' + args.output + '_feature.csv'
df3.to_csv(fPred3, index=False)

df4 = pd.DataFrame({'batch':batch_size})
fPred4 = 'result/' + args.output + '/' + args.output + '_batch.csv'
df4.to_csv(fPred4, index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
# if args.cla ==3:
#     df = pd.DataFrame({'label':labels, 'weight':weights, 'scaledWeight':scaledWeights})
#     fPred = 'result/' + args.output + '/' + args.output + '.csv'
#     df.to_csv(fPred, index=False)

#     df2 = pd.DataFrame({'prediction':preds})
#     predonlyFile = 'result/' + args.output + '/' + args.output + '_pred.csv'
#     df2.to_csv(predonlyFile, index=False)
# else:
#     df = pd.DataFrame({'label':labels, 'prediction':preds,
#                      'weight':weights, 'scaledWeight':scaledWeights})
#     fPred = 'result/' + args.output + '/' + args.output + '.csv'
#     df.to_csv(fPred, index=False)



