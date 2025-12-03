# FILE: entry/gradient_market/automate_exp/evaluate_defense_strictness.py (New Name)

import copy
import sys
from typing import Callable

# --- (Imports remain the same) ---
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_image_backdoor_attack, )

# from entry.gradient_market.automate_exp.scenarios import use_trec_config # Add if needed
try:
    from marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Use Golden Training Parameters (Same as before) ---
GOLDEN_PARAMS_PER_MODALITY = {
    "image": {"learning_rate": 0.01, "local_epochs": 2},  # Replace with actual values
    # Add other modalities if testing them
}

# --- Fixed Attack Settings (Same as before) ---
FIXED_ATTACK_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5
NUM_SEEDS_PER_CONFIG = 3

# --- Defense Strictness Grids ---
# Focus sweeps ONLY on parameters directly controlling filtering strength
STRICTNESS_GRIDS = {
    "fltrust": {
        "aggregation.method": ["fltrust"],
        # Sweep clip_norm - this is the main strictness control
        "aggregation.clip_norm": [None, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0],  # Wider range
    },
    "martfl": {
        "aggregation.method": ["martfl"],
        # --- FIX other parameters based on tuning results or reasonable defaults ---
        "aggregation.martfl.change_base": [True],  # Fix this as requested
        "aggregation.martfl.clip": [True],  # Fix clipping enabled
        "aggregation.clip_norm": [5.0],  # Fix clip_norm (e.g., to a reasonable value)
        "aggregation.martfl.initial_baseline": ["buyer"],  # Fix initial baseline
        # --- SWEEP max_k - controls clustering strictness ---
        "aggregation.martfl.max_k": [2, 3, 5, 7, 10],  # Sweep k
    },
    "skymask": {
        "aggregation.method": ["skymask"],
        # --- FIX other parameters ---
        "aggregation.skymask.clip": [True],
        "aggregation.clip_norm": [10.0],
        "aggregation.skymask.mask_epochs": [20],  # Fix based on tuning or default
        "aggregation.skymask.mask_lr": [0.01],  # Fix based on tuning or default
        "aggregation.skymask.mask_clip": [1.0],  # Fix based on tuning or default
        # --- SWEEP mask_threshold - directly controls filtering ---
        "aggregation.skymask.mask_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],  # Sweep threshold
    }
}


# --- (apply_tuning_setup function remains the same) ---
def apply_tuning_setup(config: AppConfig, modality: str, attack_modifier: Callable) -> AppConfig:
    # ... (Implementation is identical to tune_defenses.py) ...
    # 1. Apply Golden Training Parameters
    if modality in GOLDEN_PARAMS_PER_MODALITY:
        params = GOLDEN_PARAMS_PER_MODALITY[modality]
        config.training.learning_rate = params["learning_rate"]
        config.training.local_epochs = params["local_epochs"]
        print(f"  Applied Golden Params for {modality}: lr={params['learning_rate']}, E={params['local_epochs']}")
    else:
        print(f"  Warning: No Golden Params defined for modality '{modality}', using defaults.")

    # 2. Apply Attack Settings (Fixed for Tuning)
    config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
    config = attack_modifier(config)  # Apply backdoor pattern/target
    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)
    # config.adversary_seller_config.sybil.is_sybil = True # Uncomment if tuning against Sybil attack
    # config.adversary_seller_config.sybil.gradient_default_mode = "mimic"

    print(f"  Applied Fixed Attack: adv_rate={FIXED_ATTACK_ADV_RATE}, poison_rate={FIXED_ATTACK_POISON_RATE}")
    return config


# --- Define ONE Model/Dataset combination to focus on ---
# (Reduces computational cost for this analysis)
MODELS_DATASETS_TO_EVALUATE = [
    {  # Example: Image Cifar10/ResNet18
        "modality_name": "image",
        "base_config_factory": get_base_image_config,
        "dataset_name": "cifar10",
        "model_structure": "resnet18",
        "model_config_param_key": "experiment.image_model_config_name",
        "model_config_name": "cifar10_resnet18",
        "attack_modifier": use_image_backdoor_attack,
        "dataset_modifier": use_cifar10_config,
    },
]

# --- Main Execution Block ---
if __name__ == "__main__":

    output_dir = "./configs_generated/step3b_strictness_evaluation"  # New directory
    generator = ExperimentGenerator(output_dir)
    all_strictness_scenarios = []

    print("\n--- Generating Defense Strictness Evaluation Scenarios ---")

    for combo_config in MODELS_DATASETS_TO_EVALUATE:
        modality = combo_config["modality_name"]
        print(
            f"\n-- Processing Modality: {modality}, Dataset: {combo_config['dataset_name']}, Model: {combo_config['model_structure']}")

        # Iterate through each defense method
        # --- Use STRICTNESS_GRIDS now ---
        for defense_name, defense_grid in STRICTNESS_GRIDS.items():

            # Skip incompatible defenses
            if modality != "image" and defense_name == "skymask":
                print(f"   Skipping {defense_name} tuning for non-image modality.")
                continue

            scenario_name = f"strictness_{defense_name}_{modality}_{combo_config['dataset_name']}_{combo_config['model_structure']}"
            print(f"  - Defining scenario for {defense_name}: {scenario_name}")


            # Define the setup modifier (Same logic as before)
            def create_setup_modifier(current_combo_config):
                def setup_modifier_inner(config: AppConfig) -> AppConfig:
                    modified_config = current_combo_config["dataset_modifier"](config)
                    return apply_tuning_setup(modified_config, current_combo_config["modality_name"],
                                              current_combo_config["attack_modifier"])

                return setup_modifier_inner


            current_setup_modifier = create_setup_modifier(combo_config)

            # Determine sm_model_type for SkyMask (Same logic as before)
            sm_model_type_list = [None]
            if defense_name == "skymask":
                model_structure = combo_config['model_structure']
                if 'cnn' in model_structure:
                    sm_model_type_list = ['flexiblecnn']
                elif 'resnet' in model_structure:
                    sm_model_type_list = [model_structure]
                else:
                    sm_model_type_list = [model_structure]

            # Combine fixed params with the defense strictness grid
            full_parameter_grid = {
                # Fixed parameters for this model/dataset
                "experiment.dataset_name": [combo_config["dataset_name"]],
                "experiment.model_structure": [combo_config["model_structure"]],
                combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],  # Keep ON to save time
                "experiment.patience": [10],
                "aggregation.skymask.sm_model_type": sm_model_type_list,

                # The defense method and its specific tuning grid
                **defense_grid  # Use the STRICTNESS_GRIDS
            }

            # Clean up grid (Same logic as before)
            if defense_name != "skymask":
                if "aggregation.skymask.sm_model_type" in full_parameter_grid:
                    del full_parameter_grid["aggregation.skymask.sm_model_type"]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=combo_config["base_config_factory"],
                modifiers=[current_setup_modifier],
                parameter_grid=full_parameter_grid
            )
            all_strictness_scenarios.append(scenario)

    # --- Generate Config Files (Same logic as before) ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    # ... (rest of the generation loop is identical) ...
    for scenario in all_strictness_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All defense strictness configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments using configs in '{output_dir}'.")
    print("2. Aggregate results.")
    print("3. Analyze results:")
    print("   - For FLTrust: Plot test_acc, test_asr, FPR vs. aggregation.clip_norm.")
    print("   - For MartFL: Plot test_acc, test_asr, FPR vs. aggregation.martfl.max_k.")
    print("   - For SkyMask: Plot test_acc, test_asr, FPR vs. aggregation.skymask.mask_threshold.")
    print("   - This shows the trade-off between robustness and impact on benign sellers.")
