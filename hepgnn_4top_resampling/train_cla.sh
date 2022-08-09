#!/bin/bash

python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o date20220705_4top_cla_alledge_w1_L1 --device 0 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 4

python eval_4top_QCD_cla.py --config config_4top_QCD_w1.yaml --batch 1 -o date20220705_4top_cla_alledge_w1_L1 --device 0 --cla 1 --weight 4



python train_4top_QCD_cla_resam.py --config config_4top_QCD_w2.yaml --epoch 200 --batch 1024 -o date20220705_4top_cla_alledge_w2_L1 --device 0 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 4

python eval_4top_QCD_cla.py --config config_4top_QCD_w2.yaml --batch 1 -o date20220705_4top_cla_alledge_w2_L1 --device 0 --cla 1 --weight 4




python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o date20220705_4top_cla_alledge_w1_L2 --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-3 --weight 4

python eval_4top_QCD_cla.py --config config_4top_QCD_w1.yaml --batch 1 -o date20220705_4top_cla_alledge_w1_L2 --device 0 --cla 1 --weight 4



python train_4top_QCD_cla_resam.py --config config_4top_QCD_w2.yaml --epoch 200 --batch 1024 -o date20220705_4top_cla_alledge_w2_L2 --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-3 --weight 4

python eval_4top_QCD_cla.py --config config_4top_QCD_w2.yaml --batch 1 -o date20220705_4top_cla_alledge_w2_L2 --device 0 --cla 1 --weight 4
