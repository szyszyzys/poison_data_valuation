#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

n_sellers_list=(10)
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
n_adversaries_arg="1"
aggregation_method='martfl'
# Initialize these to empty strings to check later
local_epoch_arg="2"
poison_strength_arg="0.1"
trigger_rate="0.5"
# gradient_manipulation_mode is already set to "single" by default
is_sybil_flag=""

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
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --aggregation_method)
            aggregation_method="$2"
            shift 2
            ;;
        --poison_strength)
            poison_strength_arg="$2"
            shift 2
            ;;
        --gradient_manipulation_mode)
            gradient_manipulation_mode="$2"
            shift 2
            ;;
        --trigger_rate)
            trigger_rate="$2"
            shift 2
            ;;
        --is_sybil)
            # If the flag is provided, set sybil mode to true.
            is_sybil_flag="--is_sybil"
            shift 1
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Process command-line inputs that may be comma-separated lists.
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

if [[ -n "$poison_strength_arg" ]]; then
    IFS=',' read -r -a poison_strength_list <<< "$poison_strength_arg"
fi

# Debug: Print final configuration
echo "Final configuration:"
echo "  n_sellers: ${n_sellers_list[@]}"
echo "  n_adversaries: ${n_adversaries_list[@]}"
echo "  local_epoch: ${local_epoch_list[@]}"
echo "  poison_strength: ${poison_strength_list[@]}"
echo "  gradient_manipulation_mode: $gradient_manipulation_mode"
echo "  dataset_name: $dataset_name"
echo "  global_rounds: $global_rounds"
echo "  aggregation_method: $aggregation_method"
echo "  gpu_ids: $gpu_ids"
echo "  trigger_rate: $trigger_rate"
echo "  local_lr: $local_lr"
echo "  is_sybil_flag: $is_sybil_flag"

# Loop over combinations and run experiments
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
                  --gradient_manipulation_mode "$gradient_manipulation_mode" \
                  --aggregation_method "$aggregation_method" \
                  --trigger_rate "$trigger_rate" \
                  $is_sybil_flag
          done
        done
    done
done
