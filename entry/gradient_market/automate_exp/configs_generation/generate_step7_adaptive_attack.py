# FILE: generate_step7_adaptive_attack.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE,  # Use default adversary rate
    IMAGE_DEFENSES, create_fixed_params_modifier,  # Use the standard helper
    enable_valuation
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config  # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, use_adaptive_attack  # Example

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
# Evaluate defense robustness against ADAPTIVE attackers who learn from feedback.
# This experiment tests how well TUNED defenses perform against an adversary
# (specifically the 'black_box' threat model) that tries different strategies
# (gradient manipulation or data manipulation) and adapts based on selection success.
# Uses GOLDEN training parameters. Tests resistance to learning/exploitation. 🧠

# --- Adaptive Attack Settings ---
# ## USER ACTION ##: Confirm these settings are appropriate
ADAPTIVE_MODES_TO_TEST = ["gradient_manipulation", "data_manipulation"]
ADAPTIVE_THREAT_MODEL = "black_box"  # Focus on learning from binary feedback
EXPLORATION_ROUNDS = 30  # Rounds for the attacker to explore strategies
ADAPTIVE_THREAT_MODELS_TO_TEST = ["black_box", "gradient_inversion", "oracle"]
# --- Focus Setup for Adaptive Attack Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
ADAPTIVE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR10",  # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn",  # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    # No base attack modifier needed, adaptive attacker IS the attack
}


def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against adaptive attackers."""
    print("\n--- Generating Step 7: Adaptive Attack Scenarios ---")
    scenarios = []
    modality = ADAPTIVE_SETUP["modality_name"]
    model_cfg_name = ADAPTIVE_SETUP["model_config_name"]
    current_defenses = IMAGE_DEFENSES

    for defense_name in current_defenses:
        if defense_name not in TUNED_DEFENSE_PARAMS:
            print(f"  Skipping {defense_name}: No tuned parameters found.")
            continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        fixed_params_modifier = create_fixed_params_modifier(
            modality, tuned_defense_params, model_cfg_name, apply_noniid=True
        )

        # --- UPDATE: Loop over threat models FIRST ---
        for threat_model in ADAPTIVE_THREAT_MODELS_TO_TEST:
            print(f"  -- Threat Model: {threat_model}")

            # --- Loop through adaptive attack modes (as before) ---
            for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
                print(f"    -- Adaptive Mode: {adaptive_mode}")

                # --- LOGIC OPTIMIZATION ---
                # Oracle and GradInv are gradient manipulations, so skip data_manipulation
                if threat_model != "black_box" and adaptive_mode == "data_manipulation":
                    print(f"       Skipping data_manipulation for {threat_model} (N/A)")
                    continue

                # --- Create the adaptive attack modifier ---
                adaptive_modifier = use_adaptive_attack(
                    mode=adaptive_mode,
                    threat_model=threat_model,  # <-- Use the loop variable
                    exploration_rounds=EXPLORATION_ROUNDS
                )

                # --- Define the grid (fixed parameters for this run) ---
                grid = {
                    ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
                    "experiment.adv_rate": [DEFAULT_ADV_RATE],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                }

                # --- Create the Scenario ---
                scenario_name = f"step7_adaptive_{threat_model}_{adaptive_mode}_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
                    modifiers=[
                        fixed_params_modifier,
                        ADAPTIVE_SETUP["dataset_modifier"],
                        adaptive_modifier,
                        lambda config: enable_valuation(
                            config,
                            influence=True,
                            loo=False,
                            loo_freq=10,
                            kernelshap=False
                        )

                    ],
                    parameter_grid=grid
                )
                scenarios.append(scenario)

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step7_adaptive_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_adaptive_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 7 ---")
    # --- Standard Generator Loop ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, adaptive attack, valuation)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator just applies the fixed grid parameters
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 7 (Adaptive Attack Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Implement/Verify the AdaptiveAttackerSeller logic, especially '{ADAPTIVE_THREAT_MODEL}'.")
    print(f"3. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"4. Analyze results: Compare performance (Acc, ASR if applicable, Selection Rate, Influence Score)")
    print(f"   across defenses for each adaptive mode ({ADAPTIVE_MODES_TO_TEST}).")
    print(f"   Plot metrics over rounds to see learning/adaptation.")
