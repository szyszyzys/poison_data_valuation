#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --adv_rate "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

n_sellers=30
gradient_manipulation_mode="None"  # Default; can be overridden via command-line

# Fixed parameters
dataset_name="FMNIST"
global_rounds=1000
exp_name="experiment_$(date +%Y%m%d_%H%M%S)"  # Unique experiment name with timestamp.
model_arch="resnet18"
seed=42
gpu_ids="7"
local_lr="1e-2"
aggregation_method='martfl'
# Initialize these to empty strings to check later
local_epoch="2"
# gradient_manipulation_mode is already set to "single" by default
data_split_modes=("dirichlet" "adversaryfirst")
change_base="True"

for data_split_mode in "${data_split_modes[@]}"; do
  python entry/gradient_market/backdoor_attack.py \
      --dataset_name "$dataset_name" \
      --n_sellers "$n_sellers" \
      --global_rounds "$global_rounds" \
      --exp_name "$exp_name" \
      --model_arch "$model_arch" \
      --seed "$seed" \
      --gpu_ids "$gpu_ids" \
      --local_epoch "$local_epoch" \
      --local_lr "$local_lr" \
      --aggregation_method "$aggregation_method" \
      --data_split_mode "$data_split_mode" \
      --change_base "$change_base"\
      --gradient_manipulation_mode "$gradient_manipulation_mode"\
      --adv_rate "0"
done