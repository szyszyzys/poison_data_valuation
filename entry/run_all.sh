#!/bin/bash

# Activate your conda environment if you're using one
# conda activate ddm

# Define the base directory where your configs are located
CONFIGS_BASE_DIR="configs_generated"

# Loop through all config files in the 'poison_vary_adv_rate_celeba' directory
echo "--- Running configs for 'poison_vary_adv_rate_celeba' ---"
find "${CONFIGS_BASE_DIR}/poison_vary_adv_rate_celeba" -name "config.yaml" | while read config_path; do
    echo "Running: python test.py ${config_path}"
    python entry/gradient_market/run_all_exp.py "${config_path}"
    echo "--------------------------------------------------------"
done

# Loop through all config files in the 'poison_vary_poison_rate_celeba' directory
echo "--- Running configs for 'poison_vary_poison_rate_celeba' ---"
find "${CONFIGS_BASE_DIR}/poison_vary_poison_rate_celeba" -name "config.yaml" | while read config_path; do
    echo "Running: python test.py ${config_path}"
    python entry/gradient_market/run_all_exp.py "${config_path}"
    echo "--------------------------------------------------------"
done

echo "All config runs completed."