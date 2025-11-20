# FILE: generate_step6_advanced_sybil.py

import copy
import sys
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports (Assuming these are available in your environment) ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, enable_valuation, use_sybil_attack_strategy,
    get_tuned_defense_params, GOLDEN_TRAINING_PARAMS
)
# NOTE: AppConfig and set_nested_attr must be available from these imports
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# --- Sybil Strategies & Parameters to Test ---
SYBIL_TEST_CONFIG = {
    # "baseline_no_sybil": None,
    # "mimic": {},
    # "pivot": {},
    # "knock_out": {}

    # NOTE: 'blend_alpha' here is the key expected by use_sybil_attack_strategy kwargs
    "oracle_blend": {"blend_alpha": [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]},
    # "systematic_probe": {},
}

# --- Focus Setup for Advanced Sybil Analysis ---
SYBIL_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack
}

# --- Fixed Attack Parameters (Strength) ---
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE


# === FUNCTION TO GENERATE BASE SCENARIOS ===
def generate_advanced_sybil_scenarios() -> List[Scenario]:
    """Generates base scenarios for comparing different Sybil strategies."""
    print("\n--- Generating Step 6: Advanced Sybil Comparison Scenarios ---")
    scenarios = []
    modality = SYBIL_SETUP["modality_name"]
    model_cfg_name = SYBIL_SETUP["model_config_name"]
    current_defenses = ['martfl']

    for defense_name in current_defenses:
        # 2. Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state='with_attack',
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")

        # 3. Create the setup modifier INSIDE the loop
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # --- Apply Golden Training HPs (from Step 2.5) ---
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # --- Apply Tuned Defense HPs (from Step 3) ---
                if current_tuned_params:
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Tuned HPs found for {current_defense_name}!")

                if current_defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # --- Apply other fixed settings ---
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                return config

            return modifier

        # 4. Create the modifier function
        setup_modifier_func = create_setup_modifier()

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

            current_grid = base_grid.copy()
            current_grid["_strategy_name"] = [strategy_name]  # Store strategy name
            if strategy_params_sweep:
                current_grid["_sweep_params"] = [strategy_params_sweep]
            else:
                current_grid["_sweep_params"] = [None]

            # 5. Build the modifier list (Sybil strategy applied in the main block)
            current_modifiers = [
                setup_modifier_func,
                SYBIL_SETUP["dataset_modifier"],
                SYBIL_SETUP["attack_modifier"],
                lambda config: enable_valuation(
                    config,
                    influence=True,
                    loo=True,
                    loo_freq=10,
                    kernelshap=False
                )
            ]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=SYBIL_SETUP["base_config_factory"],
                modifiers=current_modifiers,
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


# ----------------------------------------------------------------------
## MAIN EXECUTION BLOCK (The Fix is Applied Here)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step6_advanced_sybil"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_advanced_sybil_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 6 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Filter out metadata keys starting with "_"
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if not k.startswith("_")}
        strategy_name = scenario.parameter_grid["_strategy_name"][0]
        sweep_params = scenario.parameter_grid["_sweep_params"][0]
        base_hp_suffix = f"adv_{FIXED_ATTACK_ADV_RATE}_poison_{FIXED_ATTACK_POISON_RATE}"

        if sweep_params:
            sweep_key, sweep_values = next(iter(sweep_params.items()))  # e.g., 'blend_alpha', list_of_values

            for sweep_value in sweep_values:
                current_grid = static_grid.copy()
                hp_suffix = f"{base_hp_suffix}_{sweep_key}_{sweep_value}"

                # **CRITICAL FIX:** Pass the sweep parameter as kwargs to the sybil modifier
                strategy_kwargs = {sweep_key: sweep_value}

                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    # Combine existing modifiers AND the Sybil modifier with the specific parameter
                    modifiers=scenario.modifiers + [
                        use_sybil_attack_strategy(strategy=strategy_name, **strategy_kwargs)],
                    parameter_grid=current_grid
                )
                temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                base_config = temp_scenario.base_config_factory()
                modified_base_config = copy.deepcopy(base_config)

                # Run all modifiers, which now include the Sybil modifier that sets the blend_alpha
                for modifier in temp_scenario.modifiers:
                    modified_base_config = modifier(modified_base_config)

                print(f"    - Setting Sybil strategy '{strategy_name}' with {sweep_key}: {sweep_value}")

                # Use the modified_base_config for generation
                num_gen = generator.generate(modified_base_config, temp_scenario)
                task_configs += num_gen

        else:
            # Logic for non-sweeping strategies (e.g., mimic, baseline)
            current_grid = static_grid.copy()
            hp_suffix = base_hp_suffix

            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers,
                parameter_grid=current_grid
            )
            temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            if strategy_name == "baseline_no_sybil":
                temp_scenario.parameter_grid["adversary_seller_config.sybil.is_sybil"] = [False]
            else:
                # For non-sweeping sybil strategies (e.g., mimic, pivot)
                temp_scenario.modifiers.append(use_sybil_attack_strategy(strategy=strategy_name))

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