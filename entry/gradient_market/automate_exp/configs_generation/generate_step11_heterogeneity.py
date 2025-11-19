# FILE: generate_step11_heterogeneity.py

import copy
import sys
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
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

# --- Constants ---
DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.5, 0.1]
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE
UNIFORM_ALPHA = 100.0  # Alpha value used for uniform distribution

# Define the bias types to test
BIAS_TYPES = ["market_wide", "buyer_only", "seller_only"]

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


# === FUNCTION TO GENERATE BASE SCENARIOS (UNCHANGED LOGIC) ===
def generate_heterogeneity_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying Dirichlet alpha and bias source."""
    print("\n--- Generating Step 11: Heterogeneity Impact Scenarios ---")
    scenarios = []
    modality = HETEROGENEITY_SETUP["modality_name"]
    model_cfg_name = HETEROGENEITY_SETUP["model_config_name"]
    print(f"Setup: {HETEROGENEITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Attack")

    # Loop over Bias Types
    for bias_type in BIAS_TYPES:
        for defense_name in DEFENSES_TO_TEST:

            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",
                default_attack_type_for_tuning="backdoor"
            )
            print(f"-- Processing Defense: {defense_name} for Bias Type: {bias_type}")

            if not tuned_defense_params and defense_name != "fedavg":
                print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
                continue

            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_attack_modifier=HETEROGENEITY_SETUP["attack_modifier"]
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # 1. Apply Golden Training HPs
                    golden_hp_key = f"{current_model_cfg_name}"
                    training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)

                    # 2. Apply Tuned Defense HPs
                    if current_tuned_params:
                        for key, value in current_tuned_params.items():
                            set_nested_attr(config, key, value)
                    if current_defense_name == "skymask":
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # 3. Apply the fixed attack type and strength
                    config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
                    config = current_attack_modifier(config)
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                    # 4. Valuation
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # --- Define the parameter grid (FIXED params, strategy is in the grid) ---
            parameter_grid = {
                HETEROGENEITY_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [HETEROGENEITY_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],

                # CRITICAL: This field will be set by the generator after initialization.
                "adversary_seller_config.poisoning.data_distribution.strategy": ["dirichlet"],
            }

            # NEW: Include bias_type in scenario name and grid
            scenario_name = f"step11_{bias_type}_{defense_name}_{HETEROGENEITY_SETUP['dataset_name']}"
            parameter_grid["_bias_type"] = [bias_type]  # Store bias type for main loop

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=HETEROGENEITY_SETUP["base_config_factory"],
                modifiers=[setup_modifier_func, HETEROGENEITY_SETUP["dataset_modifier"]],
                parameter_grid=parameter_grid
            )
            scenarios.append(scenario)

    return scenarios


# === MINIMUM CHANGE FIX: Initialization Utility (Uses Dictionary) ===

def initialize_adversary_data_distribution(config: AppConfig) -> AppConfig:
    """
    Ensures the adversary poisoning config has a data_distribution field,
    resolving the AttributeError with minimum change (assigning a dict).
    """
    poisoning_cfg = config.adversary_seller_config.poisoning

    # Check if the attribute exists or is None
    if not hasattr(poisoning_cfg, 'data_distribution') or poisoning_cfg.data_distribution is None:
        # Assign an empty dictionary. This is the simplest fix.
        poisoning_cfg.data_distribution = {}

    return config


# --- Main Execution Block (MODIFIED to use initialization utility) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step11_heterogeneity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_heterogeneity_scenarios()
    all_generated_configs = 0
    modality = HETEROGENEITY_SETUP["modality_name"]

    print("\n--- Generating Configuration Files for Step 11 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        static_grid = scenario.parameter_grid.copy()
        bias_type = static_grid.pop("_bias_type")[0]
        defense_name = scenario.name.split('_')[-2]

        # Loop through each alpha value (the biased alpha)
        for alpha in DIRICHLET_ALPHAS_TO_SWEEP:

            # --- LOGIC FIX START ---
            # Determine Strategy AND Alpha for both sides

            if bias_type == "market_wide":
                # Both sides biased -> Both use Dirichlet with the sweep alpha
                seller_strat = "dirichlet"
                seller_alpha_key = alpha

                buyer_strat = "dirichlet"
                buyer_alpha_key = alpha

            elif bias_type == "buyer_only":
                # Sellers are Uniform -> Use IID (Fixes robustness)
                seller_strat = "iid"
                seller_alpha_key = UNIFORM_ALPHA

                # Buyer is Biased -> Use Dirichlet
                buyer_strat = "dirichlet"
                buyer_alpha_key = alpha

            elif bias_type == "seller_only":
                # Sellers are Biased -> Use Dirichlet
                seller_strat = "dirichlet"
                seller_alpha_key = alpha

                # Buyer is Uniform -> Use IID (Fixes FLTrust Crash)
                buyer_strat = "iid"
                buyer_alpha_key = UNIFORM_ALPHA

            # SPECIAL CASE: If the sweep alpha itself is 100 (Control Group),
            # force everything to IID to ensure a clean baseline.
            if alpha >= 100.0:
                seller_strat = "iid"
                buyer_strat = "iid"
            # --- LOGIC FIX END ---

            # 2. Create the specific grid for this combination
            current_grid = static_grid.copy()

            # --- CONFIGURATION INJECTION ---

            # Seller Config
            current_grid[f"data.{modality}.strategy"] = [seller_strat]
            current_grid[f"data.{modality}.dirichlet_alpha"] = [seller_alpha_key]

            # Buyer Config
            current_grid[f"data.{modality}.buyer_strategy"] = [buyer_strat]
            current_grid[f"data.{modality}.buyer_dirichlet_alpha"] = [buyer_alpha_key]

            # OPTIONAL: Boost Buyer Ratio for CIFAR-100 to ensure FLTrust stability
            # (Uncomment if you want to be 100% sure FLTrust has enough data)
            # current_grid[f"data.{modality}.buyer_ratio"] = [0.2]

            # --- END CONFIGURATION ---

            # 3. Define unique output path
            hp_suffix = f"alpha_{alpha}"
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            # 4. Create and Generate
            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers,
                parameter_grid=current_grid
            )

            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)

            # --- FIX APPLICATION: Initialize the missing object BEFORE modifiers run ---
            # This is the fix for: 'PoisoningConfig' object has no attribute 'data_distribution'
            modified_base_config = initialize_adversary_data_distribution(modified_base_config)

            # Set the defense aggregation config name manually for uniqueness
            set_nested_attr(modified_base_config, "aggregation.aggregation_name", defense_name)

            for modifier in temp_scenario.modifiers:
                modified_base_config = modifier(modified_base_config)

            # The generator will now apply grid parameters successfully.
            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 11 (Heterogeneity Impact Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")