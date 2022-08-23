# Useful commands
## In the ui20 server, send files from ui20 to nurion.
```
nohup scp -rp -P 22 ui20server_file_path your_nurion_id@nurion.ksc.re.kr:nurion_file_path > nohup.out 2>&1
press "ctrl + z"
bg 
```

## in the ui20 server, bring files from nurion to ui20
```
scp -rp -P 22 your_nurion_id@nurion.ksc.re.kr:nurion_file_path ui20server_file_path
press "ctrl + z"
bg 
```

```
source /apps/applications/miniconda3/etc/profile.d/conda.sh
```

## If you want to know currently loaded modules name,
```
module list
```

# env for h5 data files
```
conda create --prefix -n your_nurion_scratch_path_/py39 python=3.9 -y

conda activate py39

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch -y ; conda install pyg -c pyg -y
```

After finish the env setup, do a test run something like the below line. You sould change data path and output file names.
```
cd /scratch/hpc22a06/hepgnn_4top_resampling; python train_4top_QCD_cla_resam_h5.py --config config_4top_QCD_w1_no_tikim.yaml --epoch 200 --batch 1024 -o ../output/train20220712_4top_cla_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 3
```

# env for pt data files
```
module load gcc/8.3.0
conda create --prefix your_nurion_scratch_path/gcc83 python=3.7 -y
conda activate your_nurion_scratch_path/gcc83
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Jun 24, 2020
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Nov 2, 2020
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Oct 31, 2020
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Mar 1, 2020
pip install torch-geometric==1.6.3 # Dec 2, 2020 
```

After finish the env setup, do a test run something like the below line. You sould change data path and output file names.
```
cd /scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling; python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o 20220716train_4top_QCD_cla_resam_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 4
```

# run the below lines on login shell just for test. 
```
source /apps/applications/miniconda3/etc/profile.d/conda.sh
conda activate your_nurion_scratch_path/gcc83
```

## training
```
python train_4top_QCD_cla_resam.py \
                                   --config 0config_4top_QCD_w2.yaml \
                                   --epoch 1 \
                                   --batch 1024 \
                                   -o test_train20220815_4top_cla_alledge_w2_L1 \
                                   --cla 1 \
                                   --model GNN1layer \
                                   --fea 4 \
                                   --lr 1e-3 \
                                   --weight 4
```
## evaluation
```
python eval_4top_QCD_cla.py \
                                   --config 0ui20_config_4top_QCD_w2.yaml \
                                   -o train20220816_4top_cla_alledge_w2_L1_1ncpu_48h \
                                   --cla 1 \
                                   --weight 4
                                   
```
### When evaluation takes several hours, nohup is useful.
```
nohup python eval_4top_QCD_cla.py \
                                   --config 0ui20_config_4top_QCD_w2.yaml \
                                   -o train20220816_4top_cla_alledge_w2_L1_1ncpu_48h \
                                   --cla 1 \
                                   --weight 4
```

# make your own sh file for job submision and then,
```
qsub 0run_ti.sh
```
## see the status
```
qstat -w -T -u hpc22a06
```
## If you want to delete a job
```
qdel -x 12345678.pbs
```
 You can see more qstat options by typing "man qstat"

## After traing is done,
### To see cpu usage
```
python drawPerfHistory.py "/scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling/result/test_train20220815_4top_cla_alledge_w2_L1/"
```
### copy your result from nurion to ui20 and then run evaluation code on ui20 server

# In the run.sh file, you should load craype-mic-knl first before source anything.
