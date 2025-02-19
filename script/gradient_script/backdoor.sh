#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

# Define arrays for the parameters you want to vary.
n_sellers_list=(10)
n_adversaries_list=(1 0)
poison_strength_list=(1.0 0.5)

# Set fixed parameters.
dataset_name="FMINIST"
global_rounds=100
backdoor_target_label=0
trigger_type="blended_patch"
exp_name="experiment_$(date +%Y%m%d_%H%M%S)"  # Unique experiment name with timestamp.
model_arch="resnet18"
seed=42
gpu_ids="6"
poison_test_sample=10000
# Loop over each combination.
for n_sellers in "${n_sellers_list[@]}"; do
    for n_adversaries in "${n_adversaries_list[@]}"; do
        for poison_strength in "${poison_strength_list[@]}"; do
            echo "Running experiment with n_sellers=$n_sellers, n_adversaries=$n_adversaries, poison_strength=$poison_strength"
            python entry/gradient_market/backdoor_attack.py \
                --dataset_name "$dataset_name" \
                --n_sellers "$n_sellers" \
                --n_adversaries "$n_adversaries" \
                --global_rounds "$global_rounds" \
                --backdoor_target_label "$backdoor_target_label" \
                --trigger_type "$trigger_type" \
                --exp_name "$exp_name" \
                --poison_strength "$poison_strength" \
                --model_arch "$model_arch" \
                --seed "$seed" \
                --gpu_ids "$gpu_ids" \
                --poison_test_sample "$poison_test_sample"
        done
    done
done
