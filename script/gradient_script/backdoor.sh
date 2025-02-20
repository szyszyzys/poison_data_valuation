#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

# Define arrays for the parameters you want to vary.
n_sellers_list=(10)
#n_adversaries_list=(1)
poison_strength_list=(1.0)
#local_epoch_list=(3 2 1)
gradient_manipulation_mode="single"

# Set fixed parameters.
dataset_name="FMINIST"
global_rounds=100
backdoor_target_label=0
trigger_type="blended_patch"
exp_name="experiment_$(date +%Y%m%d_%H%M%S)"  # Unique experiment name with timestamp.
model_arch="resnet18"
seed=42
gpu_ids="7"
poison_test_sample=10000
# Loop over each combination.

n_adversaries_arg="1"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_adversaries)
            n_adversaries_arg="$2"
            shift 2
            ;;
        --local_epoch)
            local_epoch_arg="$2"
            shift 2
            ;;
        --local_lr)
            local_lr="$2"
            shift 2
            ;;
        --gpu_ids)
            gpu_ids="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;

    esac
done

# Ensure variables are set before using them
if [[ -n "$n_adversaries_arg" ]]; then
    IFS=',' read -r -a n_adversaries_list <<< "$n_adversaries_arg"
else
    echo "Error: --n_adversaries argument is missing or empty."
    exit 1
fi

if [[ -n "$local_epoch_arg" ]]; then
    IFS=',' read -r -a local_epoch_list <<< "$local_epoch_arg"
else
    echo "Error: --local_epoch argument is missing or empty."
    exit 1
fi


for local_epoch in "${local_epoch_list[@]}"; do
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
                  --poison_test_sample "$poison_test_sample"\
                  --local_epoch "$local_epoch" \
                  --local_lr "$local_lr" \
                  --gradient_manipulation_mode "$gradient_manipulation_mode"
          done
        done
    done
done
