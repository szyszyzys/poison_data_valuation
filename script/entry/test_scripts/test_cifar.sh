#!/bin/bash

agg_name="fedavg"
dataset_name="CIFAR"
bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name "$dataset_name" --aggregation_method "$agg_name"
