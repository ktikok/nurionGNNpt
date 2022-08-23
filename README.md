# Useful commands
## In the ui20 server, send files from ui20 to nurion.

```
nohup scp -rp -P 22 ui20server_file_path your_nurion_id@nurion.ksc.re.kr:nurion_file_path > nohup.out 2>&1
press "ctrl + z"
bg 
```

## in the ui20 server, bring files from nurion to ui20

scp -rp -P 22 your_nurion_id@nurion.ksc.re.kr:nurion_file_path ui20server_file_path

// and then press "ctrl + z"

// type "bg" and press enter 

source /apps/applications/miniconda3/etc/profile.d/conda.sh

# // it works for h5 files--------------------------------------------

conda create --prefix -n your_nurion_scratch_path_/py39 python=3.9 -y

conda activate py39

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch -y ; conda install pyg -c pyg -y

// pyg=2.0.4

// it works for h5 files--------------------------------------------

// After finish the env setup, do a test run something like the below line.

cd /scratch/hpc22a06/hepgnn_4top_resampling; python train_4top_QCD_cla_resam_h5.py --config config_4top_QCD_w1_no_tikim.yaml --epoch 200 --batch 1024 -o ../output/train20220712_4top_cla_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 3

// h5 files

# // it works for pt files --------------------------------------------

module load gcc/8.3.0

conda create --prefix your_nurion_scratch_path/gcc83 python=3.7 -y

conda activate your_nurion_scratch_path/gcc83
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Jun 24, 2020

pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Nov 2, 2020

pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Oct 31, 2020

pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html # Mar 1, 2020

pip install torch-geometric==1.6.3 # Dec 2, 2020 

// it works for pt files --------------------------------------------

// After finish the env setup, do a test run something like the below line. You sould change data path and output file names.

cd /scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling; python train_4top_QCD_cla_resam.py --config config_4top_QCD_w1.yaml --epoch 200 --batch 1024 -o 20220716train_4top_QCD_cla_resam_alledge_w1_L1 --cla 1 --model GNN1layer --fea 4 --lr 1e-3 --weight 4

// pt files


// Once you finish every env setups, you will run the below lines every day. 

source /apps/applications/miniconda3/etc/profile.d/conda.sh

conda activate your_nurion_scratch_path/gcc83

// training

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

// evaluation

python eval_4top_QCD_cla.py \
                                   --config 0ui20_config_4top_QCD_w2.yaml \
                                   -o train20220816_4top_cla_alledge_w2_L1_1ncpu_48h \
                                   --cla 1 \
                                   --weight 4
                                   
// When evaluation takes several hours, nohup is useful.

nohup python eval_4top_QCD_cla.py \
                                   --config 0ui20_config_4top_QCD_w2.yaml \
                                   -o train20220816_4top_cla_alledge_w2_L1_1ncpu_48h \
                                   --cla 1 \
                                   --weight 4

// To see cpu usage

python drawPerfHistory.py "/scratch/hpc22a06/nurionGNNpt/hepgnn_4top_resampling/result/test_train20220815_4top_cla_alledge_w2_L1/"

// submit job. You don't have to use nohup here.

nohup qsub 0run_ti.sh > 0nohup_0run_ti.out & tail -f 0nohup_0run_ti.out

// You can see more qstat options by typing "man qstat"

// Your submssion state

qstat -w -T -u hpc22a06

// If you want to delete your job,

qdel -x 12345678.pbs

// If you want to know currently loaded modules name,

module list

// Currently Loaded Modulefiles:

//  1) craype-network-opa

// this is a default module
