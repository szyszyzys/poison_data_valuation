#!/bin/bash
# run_experiments.sh
# This script runs the Python script with varying parameters.
# bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name fmnist --aggregation_method fedavg --poison_strength "0.5,1" --gradient_manipulation_mode cmd --trigger_rate 0.5
WORK_DIR="/scratch/zzs5287/poison_data_valuation"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo $PYTHONPATH



agg_name="fedavg"
dataset_name="FMNIST"

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --dataset_name CIFAR



#!/usr/bin/env bash

# Example: run multiple experiments by iterating over different param combos
# Usage: ./run_experiments.sh

# Fixed local attack param sets
TRIGGER_TYPES=("blended_patch")
POISON_STRENGTHS=(0.1 0.5 0.7)
TRIGGER_RATES=(0.1 0.5)
GRAD_MODES=("cmd" "single" "None")

# Sybil param sets
SYBIL_MODES=("mimic")
N_ADVS=(1 3 5)
ALPHAS=(0.5 0.8)

# We'll run local-only (is_sybil=False) for some combos as well.
IS_SYBIL_VALUES=("False" "True")

# For each combination, we call run_experiment.py with the corresponding args.
# In practice, you might want to limit how big this grid is, or do partial loops.

for IS_SYBIL in "${IS_SYBIL_VALUES[@]}"; do
  for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    for PS in "${POISON_STRENGTHS[@]}"; do
      for RATE in "${TRIGGER_RATES[@]}"; do
        for GMODE in "${GRAD_MODES[@]}"; do

          # If is_sybil=False, we skip sybil-mode loops entirely:
          if [ "$IS_SYBIL" = "False" ]; then
            echo "Running local-only: $TRIGGER_TYPE, $PS, $RATE, $GMODE"
            python run_experiment.py \
              --sybil_mode="none" \
              --backdoor_target_label=0 \
              --trigger_type=$TRIGGER_TYPE \
              --poison_strength=$PS \
              --trigger_rate=$RATE \
              --gradient_manipulation_mode=$GMODE \
              --n_adversaries=0 \
              bash script/gradient_script/backdoor.sh --poison_strength $PS --gradient_manipulation_mode $GMODE --n_adversaries "0" --dataset_name FMINIST --sybil_mode "none"
              # ... add other needed args like dataset, global_rounds, seeds etc.

          else
            # If is_sybil=True, we loop over sybil modes and param combos.
            for SYBIL_MODE in "${SYBIL_MODES[@]}"; do
              for NADV in "${N_ADVS[@]}"; do
                for AALPHA in "${ALPHAS[@]}"; do
                      echo "Running sybil: mode=$SYBIL_MODE, n_adv=$NADV, alpha=$AALPHA "
                      python run_experiment.py \
                        --is_sybil \
                        --sybil_mode=$SYBIL_MODE \
                        --alpha=$AALPHA \
                        --n_adversaries=$NADV \
                        --backdoor_target_label=0 \
                        --trigger_type=$TRIGGER_TYPE \
                        --poison_strength=$PS \
                        --trigger_rate=$RATE \
                        --gradient_manipulation_mode=$GMODE

                done
              done
            done
          fi
        done
      done
    done
  done
done

echo "All experiments completed."
