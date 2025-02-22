#!/bin/bash
# run_experiments.sh
# Script for running backdoor attack experiments with configurable parameters

WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH

# Default experiment parameters based on common backdoor attack settings
declare -A DEFAULTS=(
    # System parameters
    ["gpu_ids"]="0"
    ["seed"]=42
    ["exp_name"]="experiment_$(date +%Y%m%d_%H%M%S)"

    # Dataset and model parameters
    ["dataset_name"]="CIFAR10"        # CIFAR10, FMNIST
    ["model_arch"]="resnet18"         # resnet18, cnn
    ["poison_test_sample"]=1000       # Number of test samples to poison

    # Training parameters
    ["global_rounds"]=200             # Total federation rounds
    ["local_epoch"]=2                 # Local training epochs
    ["local_lr"]="1e-2"              # Learning rate
    ["n_sellers"]=10                  # Total number of clients

    # Attack parameters
    ["n_adversaries"]=1               # Number of malicious clients
    ["poison_strength"]=0.5           # Attack strength (0-1)
    ["trigger_rate"]=0.1              # Proportion of samples to poison
    ["trigger_type"]="checkerboard"   # blended_patch, checkerboard, noise, gradient
    ["backdoor_target_label"]=0       # Target label for backdoor
    ["gradient_manipulation_mode"]="single"  # single, cmd
    ["aggregation_method"]="fedavg"   # fedavg, martfl
)

# Function to print all parameters
print_parameters() {
    echo "Current Parameter Settings:"
    echo "-------------------------"
    for key in "${!DEFAULTS[@]}"; do
        local value="${!key:-${DEFAULTS[$key]}}"
        printf "%-25s = %s\n" "$key" "$value"
    done
    echo "-------------------------"
}

# Function to validate numeric input
validate_numeric() {
    local value=$1
    local param_name=$2
    if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error: $param_name must be a number"
        exit 1
    fi
}

# Function to validate dataset name
validate_dataset() {
    local dataset=$1
    case $dataset in
        CIFAR10|FMNIST|MNIST) ;;
        *)
            echo "Error: Invalid dataset name. Must be CIFAR10, FMNIST, or MNIST"
            exit 1
            ;;
    esac
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_adversaries)
            n_adversaries_arg="$2"
            validate_numeric "$2" "n_adversaries"
            shift 2
            ;;
        --local_epoch)
            local_epoch_arg="$2"
            validate_numeric "$2" "local_epoch"
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
            validate_dataset "$2"
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
            validate_numeric "$2" "trigger_rate"
            shift 2
            ;;
        --help)
            print_parameters
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Use --help to see all parameters"
            exit 1
            ;;
    esac
done

# Process comma-separated lists
if [[ -n "$n_adversaries_arg" ]]; then
    IFS=',' read -r -a n_adversaries_list <<< "$n_adversaries_arg"
else
    n_adversaries_list=(${DEFAULTS["n_adversaries"]})
fi

if [[ -n "$local_epoch_arg" ]]; then
    IFS=',' read -r -a local_epoch_list <<< "$local_epoch_arg"
else
    local_epoch_list=(${DEFAULTS["local_epoch"]})
fi

if [[ -n "$poison_strength_arg" ]]; then
    IFS=',' read -r -a poison_strength_list <<< "$poison_strength_arg"
else
    poison_strength_list=(${DEFAULTS["poison_strength"]})
fi

# Print configuration for debugging
echo "Configuration:"
echo "  n_sellers: ${n_sellers_list[@]}"
echo "  n_adversaries: ${n_adversaries_list[@]}"
echo "  local_epoch: ${local_epoch_list[@]}"
echo "  poison_strength: ${poison_strength_list[@]}"
echo "  gradient_manipulation_mode: $gradient_manipulation_mode"
echo "  dataset_name: $dataset_name"
echo "  global_rounds: $global_rounds"
echo "  gpu_ids: $gpu_ids"

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
                  --trigger_rate "$trigger_rate"
          done
        done
    done
done
