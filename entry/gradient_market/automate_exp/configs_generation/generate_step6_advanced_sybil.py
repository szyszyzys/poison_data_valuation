# FILE: generate_step6_advanced_sybil.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, create_fixed_params_modifier,  # Use the standard helper
    enable_valuation, use_sybil_attack_strategy
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config  # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_image_backdoor_attack  # Example

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
# Evaluate defense robustness against Sybil attackers aiming to manipulate gradient selection.
# Compares different levels of attacker sophistication:
#   1. Baseline: Standard poisoning attack (e.g., backdoor) with NO Sybil coordination.
#   2. Historical Mimicry: Sybils coordinate to mimic the centroid of previously selected gradients (practical heuristic).
#   3. Oracle Blend: Sybils coordinate using perfect knowledge of the current round's ideal benign centroid (theoretical upper bound). Includes sweep of blend factor (alpha).
#   4. Systematic Probe (Placeholder): Represents Sybils actively exploring the selection boundary.
# Uses GOLDEN training parameters and TUNED defense parameters. 👾

# --- Sybil Strategies & Parameters to Test ---
# ## USER ACTION ##: Adjust oracle alpha sweep range if needed
SYBIL_TEST_CONFIG = {
    "baseline_no_sybil": None,  # Special case for baseline comparison
    "mimic": {},  # Standard historical mimicry (uses history window)
    "oracle_blend": {"blend_alpha": [0.05, 0.1, 0.2, 0.5, 0.8]},  # Sweep alpha: 5% up to 80% malicious
    "systematic_probe": {},  # Placeholder for probing strategy logic
}

# --- Focus Setup for Advanced Sybil Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
SYBIL_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",  # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_resnet18",  # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    "attack_modifier": use_image_backdoor_attack  # Base malicious intent
}

# --- Fixed Attack Parameters (Strength) ---
# Use the same strength as defense tuning for consistency
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE


def generate_advanced_sybil_scenarios() -> List[Scenario]:
    """Generates scenarios comparing different Sybil strategies."""
    print("\n--- Generating Step 6: Advanced Sybil Comparison Scenarios ---")
    scenarios = []
    modality = SYBIL_SETUP["modality_name"]
    model_cfg_name = SYBIL_SETUP["model_config_name"]
    # Determine relevant defenses based on modality
    current_defenses = IMAGE_DEFENSES  # Adjust if testing other modalities

    for defense_name in current_defenses:
        if defense_name not in TUNED_DEFENSE_PARAMS:
            print(f"  Skipping {defense_name}: No tuned parameters found.")
            continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        # --- Create the modifier that applies Golden Training + Tuned Defense HPs ---
        # The base attack type/strength will be added by separate modifiers/grid settings
        fixed_params_modifier = create_fixed_params_modifier(
            modality, tuned_defense_params, model_cfg_name, apply_noniid=True
        )

        # --- Base Grid (fixed parts for this defense/model combo) ---
        base_grid = {
            SYBIL_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [SYBIL_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],  # Fixed num attackers
            "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],  # Fixed base attack strength
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
        }

        # --- Loop through Sybil strategies ---
        for strategy_name, strategy_params_sweep in SYBIL_TEST_CONFIG.items():
            print(f"  - Strategy: {strategy_name}")
            scenario_name = f"step6_adv_sybil_{strategy_name}_{defense_name}_{SYBIL_SETUP['dataset_name']}"
            current_modifiers = [fixed_params_modifier, SYBIL_SETUP["dataset_modifier"], SYBIL_SETUP["attack_modifier"]]
            current_grid = base_grid.copy()  # Start with base fixed params

            # --- Configure Sybil status and parameters ---
            if strategy_name == "baseline_no_sybil":
                # Explicitly set is_sybil to False for baseline in the grid
                current_grid["adversary_seller_config.sybil.is_sybil"] = [False]
                # Valuation: Influence + Periodic LOO (to see baseline value)
                current_modifiers.append(enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False))

            # --- Handle Strategies with Parameters to Sweep (like Oracle Alpha) ---
            elif strategy_params_sweep:
                # Find the parameter being swept (e.g., "blend_alpha")
                sweep_key, sweep_values = next(iter(strategy_params_sweep.items()))
                config_key_path = f"adversary_seller_config.sybil.{sweep_key}"  # Path to set in config

                # Base Sybil modifier just enables Sybil and sets strategy name
                sybil_base_modifier = use_sybil_attack_strategy(strategy=strategy_name)
                current_modifiers.append(sybil_base_modifier)

                # Add the swept parameter to the grid
                current_grid[config_key_path] = sweep_values
                scenario_name += f"_sweep_{sweep_key}"  # Indicate sweep in name
                # Valuation: Influence + Periodic LOO (see how value changes with alpha)
                current_modifiers.append(enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False))


            # --- Handle Strategies with Fixed/No Extra Parameters ---
            else:
                # Pass fixed params if any needed via kwargs here (e.g., probe params)
                sybil_modifier = use_sybil_attack_strategy(strategy=strategy_name)  # Pass fixed params if any
                current_modifiers.append(sybil_modifier)
                # Valuation: Influence + Periodic LOO
                current_modifiers.append(enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False))

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=SYBIL_SETUP["base_config_factory"],
                modifiers=current_modifiers,
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step6_advanced_sybil"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_advanced_sybil_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 6 ---")
    # --- Standard Generator Loop (Sweeps Grid Internally) ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, attack type, sybil type, valuation)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (e.g., the oracle alpha sweep)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 6 (Advanced Sybil Comparison) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print("2. Implement the 'oracle_blend' and 'systematic_probe' logic in SybilCoordinator.")
    print("3. Implement `hypothetical_select` in aggregators if using Oracle.")
    print(f"4. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"5. Analyze results: Compare Acc/ASR across strategies. Plot Acc/ASR vs. oracle_blend_alpha.")
    print(f"6. Analyze valuation results (Influence/LOO scores) to check for value inflation.")
