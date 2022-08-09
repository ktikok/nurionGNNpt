#### 4top resamping & 4top,QCD GNN classification


### python folder

- python/model : GNN models
- python/dataset/HEPGNNDataset_h5_LHE_resampling.py : used to weight resampling
- python/dataset/HEPGNNDataset_h5_LHE_resampling4.py : used to weight resampling 4fea
- python/dataset/HEPGNNDataset_pt_classify_fourfeature.py : used to classification old version
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_abs.py : training with abs weight 1
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_negative.py : training with real weight 1 or -1
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_v2.py : classification 6fea
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_v3.py : resampling 6fea
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_v4.py : classification 4fea
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_v5.py : resampling 4fea
- python/dataset/HEPGNNDataset_pt_classify_fourfeature_v6.py : classification 4fea with resampling

### config file

- config_4top_QCD_old.yaml : for classification training
- config_4top_QCD_v2.yaml : for classification training (weight resave path)
- config_4top_QCD_w1.yaml : 4top & QCD classification with 9s resampling 
- config_4top_QCD_w2.yaml : 4top & QCD classification with all resampling
- config_resamp_jet.yaml : all resampling
- config_resamp_jet2.yaml : 9s resampling
- config_resampling.yaml : for 4top weight resampling
- config_resampling_eval.yaml : for 4top weight resampling evaluation

### Make file

About make file processing : [root file to h5(graph) file (KNU Tier-3 (resampling))](https://www.notion.so/root-file-to-h5-graph-file-KNU-Tier-3-resampling-a9ec7d9bbaec459aa254dbece96f4a94) 

- lhe2graph.py : lhe file convert to h5 graph file
- lhe2graph_loop.sh : loop script of lhe2graph.py
- LHE_word_add.sh : input missing word to each lhe file
- LHEReader.py : jhgoh’s new LHE reader file - need to using

### Training & Validation & Evaluation

- 4top & QCD classification
    - train_4top_QCD_cla_resam.py : 4top & QCD classify trainscript (validation resampling weight)
    - train_4top_QCD_cla_realweight.py : 4top & QCD classify train script (validation real weight)
    - eval_4top_QCD_cla.py : 4top & QCD classify eval script
    - train_cla.sh : 4top & QCD classification bash file
- 4top resampling
    - train_4top_resampling_lhe.py : 4top resampling train script
    - eval_4top_resampling_lhe.py : 4top resampling evaluation script
    - train_resampling.sh : 4top resampling bash file

### ipynb script

- acc_loss.ipynb : draw several results acc & loss
- result_plot : draw various result data
- acc_loss_resample_lhe.ipynb : resampling model result visualization
- graph_draw.ipynb : 4top flow decay graph plot represent
- make_graph.ipynb : make “pt” file - involving resampling weight, after selection cut
- check_resampling_weight.ipynb : almost same “make_graph.ipynb”, just confirm weight distribution
