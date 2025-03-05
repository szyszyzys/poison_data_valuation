#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH



agg_name=${1:-"martfl"}
dataset_name=${2:-"FMNIST"}
gpu_ids=${3:-"2"}
change_base=${4:-"True"}

# Fixed local attack param sets
GRAD_MODES=("None")
# Sybil param sets
data_split_modes=("discovery")
buyer_data_modes=("random" "biased")
discovery_qualitys=(0.2 0.5)
for GMODE in "${GRAD_MODES[@]}"; do
    for data_split_mode in "${data_split_modes[@]}"; do
        for buyer_data_mode in "${buyer_data_modes[@]}"; do
          for discovery_quality in "${discovery_qualitys[@]}"; do
            # If is_sybil=False, we skip sybil-mode loops entirely:
                bash script/gradient_script/backdoor.sh \
                 --gradient_manipulation_mode=$GMODE \
                 --dataset_name "$dataset_name" \
                 --aggregation_method "$agg_name" \
                 --gpu_ids "$gpu_ids" \
                 --data_split_mode "$data_split_mode"\
                 --change_base "$change_base" \
                 --buyer_data_mode "$buyer_data_mode"\
                 --discovery_quality "$discovery_quality"
        done
    done
  done
done

echo "All experiments completed."
