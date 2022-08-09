#!/bin/bash

for i in {1..332}
do
	sed -i '1i <LesHouchesEvents>' /users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex2/220323_084833/0000/4top_ex2_${i}.lhe
done	
