# FILE: generate_step4_training_sensitivity.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES, get_tuned_defense_params,
    # Not used directly
    # Import valuation helper
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config  # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_image_backdoor_attack,use_cifar100_config # Example

# Import needed attack modifiers
# ## USER ACTION ##: Ensure this import path is correct
try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose
# Quantify the "Initialization Cost" or "Easiness of Tuning" for each defense.
# This experiment tests how sensitive each *tuned* defense mechanism's performance
# (utility/robustness) is to variations in the *training* parameters (Optimizer, LR, Epochs)
# when facing both benign ('no_attack') and attacked ('with_attack') conditions
# on Non-IID data. A defense that performs well across many training HPs has a
# low initialization cost and is considered more robust and practical. 🛠️

# --- Training Parameters to Sweep ---
# ## USER ACTION ##: Verify these sweep ranges are appropriate
ADAM_LRS_TO_SWEEP = [0.001, 0.0005, 0.0001]
SGD_LRS_TO_SWEEP = [0.1, 0.05, 0.01]
OPTIMIZERS_TO_SWEEP = ["Adam", "SGD"]
LOCAL_EPOCHS_TO_SWEEP = [1, 2, 5]

# --- Focus Setup for Sensitivity Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset) for this analysis
# Using a single, well-understood case makes the analysis clearer.
# SENSITIVITY_SETUP = {
#     "modality_name": "image",
#     "base_config_factory": get_base_image_config,
#     "dataset_name": "CIFAR10",  # lowercase
#     "model_config_param_key": "experiment.image_model_config_name",
#     "model_config_name": "cifar10_cnn",  # lowercase, use your best model
#     "dataset_modifier": use_cifar10_config,
#     "attack_modifier": use_image_backdoor_attack  # Standard attack for 'with_attack' state
# }
SENSITIVITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",  #
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",  # <-- CHANGED
    "dataset_modifier": use_cifar100_config,  # <-- CHANGED
    "attack_modifier": use_image_backdoor_attack  # Standard attack for 'with_attack' state
}


# --- Attack States to Test ---
ATTACK_STATES = ["no_attack", "with_attack"]


def generate_training_sensitivity_scenarios() -> List[Scenario]:
    """Generates base scenarios for sweeping training HPs against tuned defenses."""
    print("\n--- Generating Step 4: Training Sensitivity Scenarios ---")
    scenarios = []
    modality = SENSITIVITY_SETUP["modality_name"]
    model_cfg_name = SENSITIVITY_SETUP["model_config_name"]
    # Determine relevant defenses based on modality
    current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES

    for defense_name in current_defenses:
        print(f"-- Processing Defense: {defense_name}")

        for attack_state in ATTACK_STATES:
            print(f"  -- Attack State: {attack_state}")

            # --- 2. THIS IS THE FIX ---
            # Call the new helper function INSIDE the attack_state loop
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state=attack_state,
                default_attack_type_for_tuning="backdoor"
            )
            if not tuned_defense_params:
                print(f"  Skipping {defense_name} for {attack_state}: No tuned params found.")
                continue

            # --- END FIX ---
            def create_setup_modifier_sens(
                    current_attack_state=attack_state,
                    current_defense_name=defense_name,
                    current_defense_params=tuned_defense_params
            ):
                # --- END FIX ---

                # Closure to capture state
                def modifier(config: AppConfig) -> AppConfig:
                    # Apply Tuned Defense HPs from common utils

                    # Use the bound variables
                    for key, value in current_defense_params.items():
                        set_nested_attr(config, key, value)
                    print(f"    Applied Tuned Defense HPs for {current_defense_name}")
                    # Apply Attack State
                    if current_attack_state == "with_attack":
                        config.experiment.adv_rate = DEFAULT_ADV_RATE
                        # Apply the attack type modifier
                        config = SENSITIVITY_SETUP["attack_modifier"](config)
                        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                        print(
                            f"    Applied Attack: {DEFAULT_ADV_RATE * 100}% rate, {DEFAULT_POISON_RATE * 100}% poison")
                    else:  # no_attack
                        config.experiment.adv_rate = 0.0
                        config.adversary_seller_config.poisoning.type = PoisonType.NONE
                        # Ensure Sybil/Buyer attacks off if needed
                        config.adversary_seller_config.sybil.is_sybil = False
                        config.buyer_attack_config.is_active = False
                        print("    Applied No Attack.")

                    # Ensure Non-IID Data
                    set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

                    # Set SkyMask type if needed (redundant? also set below)
                    if current_defense_name == "skymask":  # <-- Use the bound variable
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # Turn off valuation for this step
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier_sens()

            # Base grid contains FIXED elements + LISTS for HP sweep
            grid = {
                SENSITIVITY_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [SENSITIVITY_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.dataset_type": [modality],  # Needed for HP loop logic
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                # Training HPs to be swept by the main loop:
                "training.optimizer": OPTIMIZERS_TO_SWEEP,
                "training.local_epochs": LOCAL_EPOCHS_TO_SWEEP,
                # Base SGD params needed for structure (values overridden in loop)
                "training.momentum": [0.0],  # Placeholder, set correctly in loop
                "training.weight_decay": [0.0],  # Placeholder, set correctly in loop
            }
            # Add SkyMask model type to grid (needed by generator expansion)
            if defense_name == "skymask":
                model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                grid["aggregation.skymask.sm_model_type"] = [model_struct]

            scenarios.append(Scenario(
                # Name identifies the fixed parts; HP combo added later
                name=f"step4_train_sens_{defense_name}_{attack_state}_{modality}_{SENSITIVITY_SETUP['dataset_name']}",
                base_config_factory=SENSITIVITY_SETUP["base_config_factory"],
                modifiers=[setup_modifier_func, SENSITIVITY_SETUP["dataset_modifier"]],
                parameter_grid=grid  # Grid contains lists for HPs
            ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step4_training_sensitivity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_training_sensitivity_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 4 ---")
    # --- Manual Loop for Training HP Sweep (Optimizer/LR/Epochs) ---
    # We use this loop because we need conditional logic for LR lists (Adam vs SGD)
    # and conditional inclusion of momentum/weight_decay for SGD.
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Extract sweep parameters and static grid from the base scenario
        optimizers = scenario.parameter_grid.get("training.optimizer", ["Adam"])
        epochs_list = scenario.parameter_grid.get("training.local_epochs", [1])
        # Collect all parameters EXCEPT the ones we manually loop over
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k not in [
            "training.optimizer", "training.learning_rate", "training.local_epochs",
            "training.momentum", "training.weight_decay"
        ]}
        modality = scenario.parameter_grid["experiment.dataset_type"][0]

        # Loop through each combination of training HPs
        for opt in optimizers:
            lrs = SGD_LRS_TO_SWEEP if opt == "SGD" else ADAM_LRS_TO_SWEEP
            for lr in lrs:
                for epochs in epochs_list:
                    # Create the specific grid for this HP combination
                    current_grid = static_grid.copy()
                    current_grid["training.optimizer"] = [opt]
                    current_grid["training.learning_rate"] = [lr]
                    current_grid["training.local_epochs"] = [epochs]

                    # Add/Remove SGD params based on optimizer
                    if opt == "SGD" and modality == "image":  # Adjust condition if needed
                        current_grid["training.momentum"] = [0.9]
                        current_grid["training.weight_decay"] = [5e-4]
                    # Ensure momentum/wd are NOT included for Adam if they were in base grid
                    elif "training.momentum" in current_grid:
                        del current_grid["training.momentum"]
                        del current_grid["training.weight_decay"]

                    # Define unique output path for results and config file name
                    hp_suffix = f"opt_{opt}_lr_{lr}_epochs_{epochs}"
                    unique_save_path = f"./results/{scenario.name}/{hp_suffix}"  # Subfolder for results
                    current_grid["experiment.save_path"] = [unique_save_path]
                    temp_scenario_name = f"{scenario.name}/{hp_suffix}"  # Config file path includes HP

                    # Create a temporary Scenario object for this specific HP combo
                    temp_scenario = Scenario(
                        name=temp_scenario_name,
                        base_config_factory=scenario.base_config_factory,
                        modifiers=scenario.modifiers,  # Modifiers apply fixed defense/attack
                        parameter_grid=current_grid  # Grid now has specific HPs
                    )

                    # Generate the config file
                    base_config = temp_scenario.base_config_factory()
                    modified_base_config = copy.deepcopy(base_config)
                    # Apply modifiers (sets tuned defense, attack state, non-iid)
                    for modifier in temp_scenario.modifiers:
                        modified_base_config = modifier(modified_base_config)
                    # Generate uses the modified base and applies the specific HP combo from current_grid
                    num_gen = generator.generate(modified_base_config, temp_scenario)
                    task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 4 (Training Sensitivity) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure TUNED_DEFENSE_PARAMS in config_common_utils.py is correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results using analyze_sensitivity.py pointing to './results/'")
    print("   -> This calculates the 'Initialization Cost' metric.")
    print("   -> Remember to fill IID_BASELINES in analyze_sensitivity.py using Step 1 results.")
