# FILE: generate_step11_heterogeneity.py

import copy
import sys
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,  # <-- ADDED
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, get_tuned_defense_params,
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_image_backdoor_attack, use_cifar100_config

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# ... (Constants are all correct) ...
DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.5, 0.1]
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE
HETEROGENEITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack
}
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"]


# === THIS IS THE CORRECTED FUNCTION ===
def generate_heterogeneity_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying Dirichlet alpha."""
    print("\n--- Generating Step 11: Heterogeneity Impact Scenarios ---")
    scenarios = []
    modality = HETEROGENEITY_SETUP["modality_name"]
    model_cfg_name = HETEROGENEITY_SETUP["model_config_name"]
    print(f"Setup: {HETEROGENEITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Attack")

    for defense_name in DEFENSES_TO_TEST:
        # === FIX 3: Removed the buggy `if defense_name not in ...` check ===

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
                current_attack_modifier=HETEROGENEITY_SETUP["attack_modifier"]
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Apply Golden Training HPs (from Step 2.5)
                #    This is the CORRECT defense-specific key
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # 2. Apply Tuned Defense HPs (from Step 3)
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)
                if current_defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # 3. Apply the fixed attack type and strength
                config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
                config = current_attack_modifier(config)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                # 4. Data distribution is set in the grid, ensure strategy is compatible
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")

                # 5. Turn off valuation
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- Define the parameter grid (FIXED params, no sweeps) ---
        parameter_grid = {
            HETEROGENEITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [HETEROGENEITY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            # dirichlet_alpha will be set by the main loop
        }

        scenario_name = f"step11_heterogeneity_{defense_name}_{HETEROGENEITY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=HETEROGENEITY_SETUP["base_config_factory"],
            modifiers=[setup_modifier_func, HETEROGENEITY_SETUP["dataset_modifier"]],
            parameter_grid=parameter_grid  # Does NOT sweep alpha
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (FIXED) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step11_heterogeneity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_heterogeneity_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 11 ---")

    # === FIX 1: Manual loop to set unique save path for each alpha ===
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid
        static_grid = scenario.parameter_grid.copy()
        modality = HETEROGENEITY_SETUP["modality_name"]  # Get modality for data key

        # Loop through each alpha value
        for alpha in DIRICHLET_ALPHAS_TO_SWEEP:

            # 1. Create the specific grid for this combination
            current_grid = static_grid.copy()
            current_grid[f"data.{modality}.strategy"] = ["dirichlet"]
            current_grid[f"data.{modality}.dirichlet_alpha"] = [alpha]  # Set the alpha

            # 2. Define unique output path
            hp_suffix = f"alpha_{alpha}"
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

    print(f"\n✅ Step 11 (Heterogeneity Impact Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results by plotting 'dirichlet_alpha' vs. 'test_acc'/'backdoor_asr' for each defense.")