#!/bin/bash


python train_4top_resampling_lhe.py --config config_resamp_jet2.yaml --epoch 300 --batch 1024 -o date20220629_resampling_fea4_9s_test1 --device 0 --cla 1 --model GNN1layer_re --fea 4 --lr 1e-3

python eval_4top_resampling_lhe.py --config config_resamp_jet2.yaml --batch 1 -o date20220629_resampling_fea4_9s_test1 --device 0 --cla 1



python train_4top_resampling_lhe.py --config config_resamp_jet.yaml --epoch 300 --batch 1024 -o date20220629_resampling_fea4_all_test1 --device 0 --cla 1 --model GNN1layer_re --fea 4 --lr 1e-3

python eval_4top_resampling_lhe.py --config config_resamp_jet.yaml --batch 1 -o date20220629_resampling_fea4_all_test1 --device 0 --cla 1
