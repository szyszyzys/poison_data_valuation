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

# --- Focus Setup for Adaptive Attack Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
ADAPTIVE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",  # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_resnet18",  # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    # No base attack modifier needed, adaptive attacker IS the attack
}


def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against adaptive attackers."""
    print("\n--- Generating Step 7: Adaptive Attack Scenarios ---")
    scenarios = []
    modality = ADAPTIVE_SETUP["modality_name"]
    model_cfg_name = ADAPTIVE_SETUP["model_config_name"]
    # Determine relevant defenses based on modality
    current_defenses = IMAGE_DEFENSES  # Adjust if testing other modalities

    for defense_name in current_defenses:
        if defense_name not in TUNED_DEFENSE_PARAMS:
            print(f"  Skipping {defense_name}: No tuned parameters found.")
            continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        # --- Create the modifier that applies Golden Training + Tuned Defense HPs ---
        # Note: apply_noniid=True ensures Non-IID seller data
        fixed_params_modifier = create_fixed_params_modifier(
            modality, tuned_defense_params, model_cfg_name, apply_noniid=True
        )

        # --- Loop through adaptive attack modes ---
        for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
            print(f"  -- Adaptive Mode: {adaptive_mode}")

            # --- Create the adaptive attack modifier ---
            adaptive_modifier = use_adaptive_attack(
                mode=adaptive_mode,
                threat_model=ADAPTIVE_THREAT_MODEL,
                exploration_rounds=EXPLORATION_ROUNDS
            )

            # --- Define the grid (fixed parameters for this run) ---
            # Attack rate is fixed, defense/training params are fixed by modifiers
            grid = {
                ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
                "experiment.adv_rate": [DEFAULT_ADV_RATE],  # Fixed % of adaptive attackers
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],  # Keep early stopping
                "experiment.patience": [10],
            }

            # --- Create the Scenario ---
            scenario_name = f"step7_adaptive_{ADAPTIVE_THREAT_MODEL}_{adaptive_mode}_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
                modifiers=[
                    fixed_params_modifier,  # Sets Golden Train + Tuned Def HPs + Non-IID
                    ADAPTIVE_SETUP["dataset_modifier"],
                    adaptive_modifier,  # Enables the specific adaptive attack
                    # Valuation: Enable Influence to track attacker value score over time
                    enable_valuation(influence=True, loo=False, kernelshap=False)
                ],
                parameter_grid=grid  # Grid only contains fixed exp params here
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
