# py39 h5
cd /scratch/hpc22a06/hepgnn_4top_resampling; python train_4top_QCD_cla_resam_h5.py --config ~/0me/nurionGNN/config_4top_QCD_w1_no.yaml --epoch 200 --batch 1024 -o ../output/train20220712_4top_cla_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 3

# py37 pt
cd /scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling; python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o date20220705_4top_cla_alledge_w1_L1 --device 0 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 4

cd /scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling; python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o ../output/train20220712_4top_cla_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 3
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch -y; conda install pyg -c pyg -y
# /scratch/hpc22a06/h5/