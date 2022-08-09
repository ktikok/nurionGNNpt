#!/bin/bash


for i in {1..332}
do

    python lhe2graph.py -i /users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/0000/4top_ex2_${i}.lhe -o /users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/graph/4top_ex2_${i}.h5 -v
done

# for i in {000..001}
# do

#     python LHEReader.py "/store/hep/users/yewzzang/test_lhe/run_01${i}.lhe.gz" 
# done
# # "/users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/jhpowheg/run_01${i}.h5"
