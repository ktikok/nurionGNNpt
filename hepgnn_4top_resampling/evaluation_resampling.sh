#!/bin/bash


python eval_4top_resampling_lhe_20220412.py --config config_eval_one.yaml --batch 1 -o 20220411_4top_run1_one_eval --device 0 --color 0

python eval_4top_resampling_lhe_20220412.py --config config_eval.yaml --batch 1 -o 20220411_4top_run2_many_eval --device 0 --color 0
