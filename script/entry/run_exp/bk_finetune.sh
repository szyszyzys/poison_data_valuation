#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --adv_rate "0.2" --gpu_ids 6 --dataset_name FMNIST --aggregation_method martfl --gradient_manipulation_mode single --trigger_rate 0.25 --is_sybil
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
adv_rate_arg="0.2"
aggregation_method='martfl'
# Initialize these to empty strings to check later
local_epoch_arg="2"
poison_strength_arg="0.1"
trigger_rate="0.25"
# gradient_manipulation_mode is already set to "single" by default
is_sybil_flag=""
sybil_mode="mimic"
bkd_loc="bottom_right"
data_split_mode="discovery"
change_base="True"
trigger_attack_mode="static"
buyer_data_mode="random"
discovery_quality="0.2"
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --adv_rate)
            adv_rate_arg="$2"
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


        --buyer_data_mode)
            buyer_data_mode="$2"
            shift 2
            ;;
        --discovery_quality)
            discovery_quality="$2"
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
        --sybil_mode)
            sybil_mode="$2"
            shift 2
            ;;
        --data_split_mode)
            data_split_mode="$2"
            shift 2
            ;;
        --bkd_loc)
          bkd_loc="$2"
            shift 2
            ;;
        --trigger_attack_mode)
            trigger_attack_mode="$2"
            shift 2
            ;;
        --change_base)
          change_base="$2"
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
if [[ -n "$adv_rate_arg" ]]; then
    IFS=',' read -r -a adv_rate_list <<< "$adv_rate_arg"
else
    echo "Error: --adv_rate argument is missing or empty."
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
echo "  adv_rate: ${adv_rate_list[@]}"
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
                  --buyer_data_mode "$buyer_data_mode" \
                  --discovery_quality "$discovery_quality"\
                  --clip\
                  --remove_baseline\
                  --exp_name "test_clip" \
                  $is_sybil_flag
          done
        done
    done
done
