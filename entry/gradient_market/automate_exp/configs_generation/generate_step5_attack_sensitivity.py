# FILE: generate_step5_attack_sensitivity.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,  # Use defaults for fixed params
    IMAGE_DEFENSES, create_fixed_params_modifier,  # Use the standard helper
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config  # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_image_backdoor_attack, use_label_flipping_attack  # Example

# Import needed attack modifiers
# ## USER ACTION ##: Ensure these import paths are correct

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose
# Measure the robustness boundaries of each TUNED defense mechanism.
# This experiment tests how well defenses perform (utility vs. robustness)
# when faced with varying ATTACK STRENGTHS (adversary rate and poison rate)
# for standard attack types (backdoor and label flipping). It uses the
# GOLDEN training parameters and TUNED defense parameters identified previously.
# Helps determine the "breaking point" of each defense. 🛡️🔥

# --- Attack Parameters to Sweep ---
# ## USER ACTION ##: Define the ranges for attack strength variation
# Sweep Adversary Rate (fix poison rate)
ATTACK_PARAMETER_GRID_ADV = {
    "experiment.adv_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Include 0.0 for baseline
    "adversary_seller_config.poisoning.poison_rate": [DEFAULT_POISON_RATE]  # Fixed poison intensity
}
# Sweep Poison Rate (fix adversary rate)
ATTACK_PARAMETER_GRID_POISON = {
    "experiment.adv_rate": [DEFAULT_ADV_RATE],  # Fixed proportion of attackers
    "adversary_seller_config.poisoning.poison_rate": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]  # Include 0.0 for baseline
}

# --- Attack Types to Test ---
ATTACK_TYPES = ["backdoor", "labelflip"]

# --- Focus Setup for Attack Sensitivity Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
SENSITIVITY_SETUP_STEP5 = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",  # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_resnet18",  # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    "backdoor_attack_modifier": use_image_backdoor_attack,
    "labelflip_attack_modifier": use_label_flipping_attack
}


def generate_attack_sensitivity_scenarios() -> List[Scenario]:
    """Generates scenarios sweeping attack strength against tuned defenses."""
    print("\n--- Generating Step 5: Attack Sensitivity Scenarios ---")
    scenarios = []
    modality = SENSITIVITY_SETUP_STEP5["modality_name"]
    model_cfg_name = SENSITIVITY_SETUP_STEP5["model_config_name"]
    current_defenses = IMAGE_DEFENSES  # Adjust if testing other modalities

    for defense_name in current_defenses:
        if defense_name not in TUNED_DEFENSE_PARAMS:
            print(f"  Skipping {defense_name}: No tuned parameters found.")
            continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        # Create the modifier that applies Golden Training + Tuned Defense HPs
        fixed_params_modifier = create_fixed_params_modifier(
            modality, tuned_defense_params, model_cfg_name, apply_noniid=True
        )

        for attack_type in ATTACK_TYPES:
            attack_modifier_key = f"{attack_type}_attack_modifier"
            if attack_modifier_key not in SENSITIVITY_SETUP_STEP5: continue
            attack_modifier = SENSITIVITY_SETUP_STEP5[attack_modifier_key]
            print(f"  -- Attack Type: {attack_type}")

            # --- Scenario for varying Adv Rate ---
            grid_adv = {
                SENSITIVITY_SETUP_STEP5["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [SENSITIVITY_SETUP_STEP5["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                **ATTACK_PARAMETER_GRID_ADV  # Sweep adv_rate
            }
            scenarios.append(Scenario(
                name=f"step5_atk_sens_adv_{defense_name}_{attack_type}_{modality}_{SENSITIVITY_SETUP_STEP5['dataset_name']}",
                base_config_factory=SENSITIVITY_SETUP_STEP5["base_config_factory"],
                # Modifiers apply fixed HPs first, then dataset specifics, then attack type
                modifiers=[
                    fixed_params_modifier,
                    SENSITIVITY_SETUP_STEP5["dataset_modifier"],
                    attack_modifier,
                    # Turn off valuation for this step unless specifically desired
                    # enable_valuation(influence=False, loo=False, kernelshap=False)
                ],
                parameter_grid=grid_adv
            ))

            # --- Scenario for varying Poison Rate ---
            grid_poison = {
                SENSITIVITY_SETUP_STEP5["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [SENSITIVITY_SETUP_STEP5["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                **ATTACK_PARAMETER_GRID_POISON  # Sweep poison_rate
            }
            scenarios.append(Scenario(
                name=f"step5_atk_sens_poison_{defense_name}_{attack_type}_{modality}_{SENSITIVITY_SETUP_STEP5['dataset_name']}",
                base_config_factory=SENSITIVITY_SETUP_STEP5["base_config_factory"],
                modifiers=[
                    fixed_params_modifier,
                    SENSITIVITY_SETUP_STEP5["dataset_modifier"],
                    attack_modifier,
                    # enable_valuation(influence=False, loo=False, kernelshap=False)
                ],
                parameter_grid=grid_poison
            ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step5_attack_sensitivity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_attack_sensitivity_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 5 ---")
    # --- Standard Generator Loop (Sweeps Grid Internally) ---
    # The generator applies modifiers first, then expands the grid (adv_rate or poison_rate).
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train HPs, tuned defense HPs, attack type)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (the attack strength sweep)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 5 (Attack Sensitivity) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results using analyze_attack_sensitivity.py pointing to './results/'")
    print("   -> This generates plots of Acc/ASR vs. Attack Strength for each defense/attack type.")
