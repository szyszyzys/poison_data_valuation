# FILE: generate_step6_advanced_sybil.py

import copy
import sys
from pathlib import Path
from typing import List, Dict, Any

# --- Imports ---
from config_common_utils import (
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, create_fixed_params_modifier,
    enable_valuation, use_sybil_attack_strategy
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_image_backdoor_attack

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose (Your purpose is excellent)
# ...

# --- Sybil Strategies & Parameters to Test ---
SYBIL_TEST_CONFIG = {
    "baseline_no_sybil": None,
    "mimic": {},
    "oracle_blend": {"blend_alpha": [0.05, 0.1, 0.2, 0.5, 0.8]},
    "systematic_probe": {},
}

# --- Focus Setup for Advanced Sybil Analysis ---
SYBIL_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_resnet18",
    "dataset_modifier": use_cifar10_config,
    "attack_modifier": use_image_backdoor_attack
}

# --- Fixed Attack Parameters (Strength) ---
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE


def generate_advanced_sybil_scenarios() -> List[Scenario]:
    """Generates base scenarios for comparing different Sybil strategies."""
    print("\n--- Generating Step 6: Advanced Sybil Comparison Scenarios ---")
    scenarios = []
    modality = SYBIL_SETUP["modality_name"]
    model_cfg_name = SYBIL_SETUP["model_config_name"]
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

        # Base Grid (fixed parts)
        base_grid = {
            SYBIL_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [SYBIL_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
        }

        # Loop through Sybil strategies
        for strategy_name, strategy_params_sweep in SYBIL_TEST_CONFIG.items():
            print(f"  - Strategy: {strategy_name}")
            scenario_name = f"step6_adv_sybil_{strategy_name}_{defense_name}"

            # --- We add a custom key to store sweep info for the main loop ---
            current_grid = base_grid.copy()
            current_grid["_strategy_name"] = [strategy_name] # Store strategy name
            if strategy_params_sweep:
                current_grid["_sweep_params"] = [strategy_params_sweep]
            else:
                current_grid["_sweep_params"] = [None]

            # Valuation is enabled for ALL runs in this step
            current_modifiers = [
                fixed_params_modifier,
                SYBIL_SETUP["dataset_modifier"],
                SYBIL_SETUP["attack_modifier"],
                enable_valuation(influence=True, loo=True, loo_freq=10, kernelshap=False)
            ]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=SYBIL_SETUP["base_config_factory"],
                modifiers=current_modifiers,
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (FIXED) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step6_advanced_sybil"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_advanced_sybil_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 6 ---")

    # --- Manual Loop REQUIRED for unique save paths & strategy logic ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid params
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k.startswith("_") == False}

        # Get our custom strategy keys
        strategy_name = scenario.parameter_grid["_strategy_name"][0]
        sweep_params = scenario.parameter_grid["_sweep_params"][0]

        # Base suffix for save path (constant for this scenario)
        base_hp_suffix = f"adv_{FIXED_ATTACK_ADV_RATE}_poison_{FIXED_ATTACK_POISON_RATE}"

        # --- Sub-loop: Iterate over any swept parameters (like blend_alpha) ---

        # Case 1: Strategy with parameters to sweep (e.g., oracle_blend)
        if sweep_params:
            sweep_key, sweep_values = next(iter(sweep_params.items()))
            config_key_path = f"adversary_seller_config.sybil.{sweep_key}" # e.g., ...sybil.blend_alpha

            for sweep_value in sweep_values:
                current_grid = static_grid.copy()

                # Set the swept parameter
                current_grid[config_key_path] = [sweep_value]

                # Create unique path: .../adv_0.3_poison_0.5_blend_alpha_0.1
                hp_suffix = f"{base_hp_suffix}_{sweep_key}_{sweep_value}"

                # --- Create temporary scenario for this specific config ---
                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    # Apply all modifiers, including the one to enable Sybil
                    modifiers=scenario.modifiers + [use_sybil_attack_strategy(strategy=strategy_name)],
                    parameter_grid=current_grid
                )
                temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                # Generate
                base_config = temp_scenario.base_config_factory()
                modified_base_config = copy.deepcopy(base_config)
                for modifier in temp_scenario.modifiers:
                    modified_base_config = modifier(modified_base_config)

                num_gen = generator.generate(modified_base_config, temp_scenario)
                task_configs += num_gen

        # Case 2: Strategy with no params (e.g., mimic) OR baseline
        else:
            current_grid = static_grid.copy()
            hp_suffix = base_hp_suffix # Path is just .../adv_0.3_poison_0.5

            # --- Create temporary scenario ---
            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers, # Base modifiers
                parameter_grid=current_grid
            )
            temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            # --- Apply strategy-specific logic ---
            if strategy_name == "baseline_no_sybil":
                # Do NOT add the sybil modifier. Set is_sybil to False.
                temp_scenario.parameter_grid["adversary_seller_config.sybil.is_sybil"] = [False]
            else:
                # Add the modifier for this strategy (e.g., "mimic")
                temp_scenario.modifiers.append(use_sybil_attack_strategy(strategy=strategy_name))

            # Generate
            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)
            for modifier in temp_scenario.modifiers:
                modified_base_config = modifier(modified_base_config)

            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 6 (Advanced Sybil Comparison) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")