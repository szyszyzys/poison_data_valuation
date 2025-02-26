#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH



agg_name=${1,-"martfl"}
dataset_name=${2,-"FMNIST"}
gpu_ids=${3,-"6"}
change_base=${4,-"True"}

# Fixed local attack param sets
TRIGGER_TYPES=("blended_patch")
TRIGGER_RATES=(0.25 0.5)
GRAD_MODES=("single")
POISON_STRENGTHS=(1)
# Sybil param sets
SYBIL_MODES=("mimic")
N_ADVS=(0.1 0.2 0.3 0.4)
ALPHAS=(0.5)
data_split_modes=("IID")
# We'll run local-only (is_sybil=False) for some combos as well.
IS_SYBIL_VALUES=("False" "True")
#IS_SYBIL_VALUES=("False")

# For each combination, we call run_experiment.py with the corresponding args.
# In practice, you might want to limit how big this grid is, or do partial loops.

for IS_SYBIL in "${IS_SYBIL_VALUES[@]}"; do
  for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    for PS in "${POISON_STRENGTHS[@]}"; do
      for RATE in "${TRIGGER_RATES[@]}"; do
        for GMODE in "${GRAD_MODES[@]}"; do
          for NADV in "${N_ADVS[@]}"; do
            for data_split_mode in "${data_split_modes[@]}"; do
            # If is_sybil=False, we skip sybil-mode loops entirely:
            if [ "$IS_SYBIL" = "False" ]; then
              echo "Running local-only: $TRIGGER_TYPE, $PS, $RATE, $GMODE"
                    bash script/gradient_script/backdoor.sh \
                     --poison_strength $PS \
                     --gradient_manipulation_mode $GMODE \
                     --dataset_name "$dataset_name" \
                     --sybil_mode "$SYBIL_MODE" \
                     --aggregation_method "$agg_name" \
                     --gpu_ids "$gpu_ids" \
                     --adv_rate "$NADV"\
                     --trigger_rate "$RATE" \
                     --data_split_mode "$data_split_mode"\
                     --change_base "$change_base"
            else
              # If is_sybil=True, we loop over sybil modes and param combos.
                for SYBIL_MODE in "${SYBIL_MODES[@]}"; do
                    for AALPHA in "${ALPHAS[@]}"; do
                          echo "Running sybil: mode=$SYBIL_MODE, n_adv=$NADV, alpha=$AALPHA "
                          bash script/gradient_script/backdoor.sh \
                           --is_sybil \
                           --poison_strength $PS \
                           --gradient_manipulation_mode $GMODE \
                           --dataset_name "$dataset_name" \
                           --sybil_mode "$SYBIL_MODE" \
                           --aggregation_method "$agg_name" \
                           --gpu_ids "$gpu_ids" \
                           --adv_rate "$NADV" \
                           --trigger_rate "$RATE" \
                           --data_split_mode "$data_split_mode"\
                          --change_base "$change_base"
                  done
                done
            fi
          done
         done
        done
      done
    done
  done
done

echo "All experiments completed."
