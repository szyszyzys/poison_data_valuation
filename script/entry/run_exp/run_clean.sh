#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --adv_rate "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

n_sellers_list=(30)
poison_strength_list=(1.0)
gradient_manipulation_mode="single"  # Default; can be overridden via command-line

# Fixed parameters
dataset_name="FMNIST"
global_rounds=1000
backdoor_target_label=0
trigger_type="blended_patch"
exp_name="experiment_$(date +%Y%m%d_%H%M%S)"  # Unique experiment name with timestamp.
model_arch="resnet18"
seed=42
gpu_ids="7"
poison_test_sample=10000
local_lr="1e-2"
adv_rate_arg="1"
aggregation_method='martfl'
# Initialize these to empty strings to check later
local_epoch_arg="2"
poison_strength_arg="0.1"
trigger_rate="0.5"
# gradient_manipulation_mode is already set to "single" by default
is_sybil_flag=""
sybil_mode=""
bkd_loc="bottom_right"
data_split_mode="IID"
change_base="True"
trigger_attack_mode=""
# Parse command-line arguments

# Loop over combinations and run experiments
for local_epoch in "${local_epoch_list[@]}"; do
  for n_sellers in "${n_sellers_list[@]}"; do
      for adv_rate in "${adv_rate_list[@]}"; do
          for poison_strength in "${poison_strength_list[@]}"; do
              echo "Running experiment with n_sellers=$n_sellers, adv_rate=$adv_rate, poison_strength=$poison_strength"
              python entry/gradient_market/backdoor_attack.py \
                  --dataset_name "$dataset_name" \
                  --n_sellers "$n_sellers" \
                  --adv_rate "$adv_rate" \
                  --global_rounds "$global_rounds" \
                  --backdoor_target_label "$backdoor_target_label" \
                  --trigger_type "$trigger_type" \
                  --exp_name "$exp_name" \
                  --poison_strength "$poison_strength" \
                  --model_arch "$model_arch" \
                  --seed "$seed" \
                  --gpu_ids "$gpu_ids" \
                  --poison_test_sample "$poison_test_sample"\
                  --local_epoch "$local_epoch" \
                  --local_lr "$local_lr" \
                  --gradient_manipulation_mode "$gradient_manipulation_mode" \
                  --aggregation_method "$aggregation_method" \
                  --trigger_rate "$trigger_rate" \
                  --bkd_loc "$bkd_loc" \
                  --data_split_mode "$data_split_mode" \
                  --change_base "$change_base"\
                  --trigger_attack_mode "$trigger_attack_mode" \
                  $is_sybil_flag
          done
        done
    done
done
