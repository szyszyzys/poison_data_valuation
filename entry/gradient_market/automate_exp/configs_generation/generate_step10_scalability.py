# FILE: generate_step10_scalability.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,  # <-- ADDED
    NUM_SEEDS_PER_CONFIG,
    # create_fixed_params_modifier,  <-- REMOVED
    get_tuned_defense_params
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# ... (Constants are all correct) ...
MARKETPLACE_SIZES = [10, 30, 50, 100]
FIXED_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5
SCALABILITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack
}
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"]


# === THIS IS THE CORRECTED FUNCTION ===
def generate_scalability_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying n_sellers."""
    print("\n--- Generating Step 10: Scalability Scenarios (Fixed Rate) ---")
    scenarios = []
    modality = SCALABILITY_SETUP["modality_name"]
    model_cfg_name = SCALABILITY_SETUP["model_config_name"]
    print(f"Setup: {SCALABILITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Adv Rate: {FIXED_ADV_RATE * 100}%")

    for defense_name in DEFENSES_TO_TEST:
        # === FIX 1: Removed the buggy `if defense_name not in ...` check ===

        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",  # Use default
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")
        # This is the correct check:
        if not tuned_defense_params:
            print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
            continue

        # === FIX 2: Create the setup modifier INSIDE the loop ===
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_attack_modifier=SCALABILITY_SETUP["attack_modifier"]
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Apply Golden Training HPs (from Step 2.5)
                golden_hp_key = f"{current_defense_name}_{current_model_cfg_name}_local_clip"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # 2. Apply Tuned Defense HPs (from Step 3)
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)

                # 3. Apply the fixed attack type and strength
                config = current_attack_modifier(config)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                # 4. Apply other fixed settings
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- Define the parameter grid (FIXED params, no sweeps) ---
        parameter_grid = {
            SCALABILITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [SCALABILITY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.adv_rate": [FIXED_ADV_RATE],
            # n_sellers will be set by the main loop
        }

        scenario_name = f"step10_scalability_{defense_name}_{SCALABILITY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=SCALABILITY_SETUP["base_config_factory"],
            modifiers=[setup_modifier_func, SCALABILITY_SETUP["dataset_modifier"]],
            parameter_grid=parameter_grid  # Does NOT sweep n_sellers
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (FIXED) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step10_scalability"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_scalability_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 10 ---")

    # === FIX 3: Manual loop to set unique save path for each n_sellers ===
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid
        static_grid = scenario.parameter_grid.copy()

        # Loop through each marketplace size
        for n_sellers in MARKETPLACE_SIZES:

            # 1. Create the specific grid for this combination
            current_grid = static_grid.copy()
            current_grid["experiment.n_sellers"] = [n_sellers]  # Set the size

            # 2. Define unique output path
            hp_suffix = f"n_sellers_{n_sellers}"
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            # 3. Create a temporary Scenario
            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers,
                parameter_grid=current_grid
            )

            # 4. Generate the config
            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)
            for modifier in temp_scenario.modifiers:
                modified_base_config = modifier(modified_base_config)

            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 10 (Scalability Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results by plotting 'n_sellers' vs. 'test_acc'/'backdoor_asr' for each defense.")
