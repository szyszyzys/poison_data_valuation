# FILE: generate_step11_heterogeneity.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, get_tuned_defense_params,
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config, use_image_backdoor_attack

try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.5, 0.1]
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE
UNIFORM_ALPHA = 100.0  # "IID"

# --- NEW: Buyer Ratio Sweep ---
# We test varying amounts of root data: 1%, 5%, 10%, 20%
# This helps isolate if FLTrust failure is due to data SCARCITY or HETEROGENEITY.
BUYER_RATIOS_TO_SWEEP = [0.01, 0.05, 0.1, 0.2]
DEFAULT_BUYER_RATIO = 0.1  # Used for the alpha sweep to keep it controlled

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


def initialize_adversary_data_distribution(config: AppConfig) -> AppConfig:
    """Ensures adversary config has data_distribution field."""
    poisoning_cfg = config.adversary_seller_config.poisoning
    if not hasattr(poisoning_cfg, 'data_distribution') or poisoning_cfg.data_distribution is None:
        poisoning_cfg.data_distribution = {}
    return config


def generate_heterogeneity_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 11: Heterogeneity Impact Scenarios ---")
    scenarios = []
    modality = HETEROGENEITY_SETUP["modality_name"]
    model_cfg_name = HETEROGENEITY_SETUP["model_config_name"]

    for bias_type in BIAS_TYPES:
        for defense_name in DEFENSES_TO_TEST:

            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",
                default_attack_type_for_tuning="backdoor"
            )
            if not tuned_defense_params and defense_name != "fedavg":
                continue

            def create_setup_modifier(current_params=tuned_defense_params, defense=defense_name):
                def modifier(config: AppConfig) -> AppConfig:
                    # Golden HPs
                    golden_hp_key = f"{model_cfg_name}"
                    training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)

                    # Defense HPs
                    if current_params:
                        for key, value in current_params.items():
                            set_nested_attr(config, key, value)
                    if defense == "skymask":
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # Attack Fixed Config
                    config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
                    config = HETEROGENEITY_SETUP["attack_modifier"](config)
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                    # Disable Valuation
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier

            # --- PARAMETER GRID ---
            parameter_grid = {
                HETEROGENEITY_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [HETEROGENEITY_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
            }

            scenario_name = f"step11_{bias_type}_{defense_name}_{HETEROGENEITY_SETUP['dataset_name']}"
            parameter_grid["_bias_type"] = [bias_type]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=HETEROGENEITY_SETUP["base_config_factory"],
                modifiers=[create_setup_modifier(), HETEROGENEITY_SETUP["dataset_modifier"]],
                parameter_grid=parameter_grid
            )
            scenarios.append(scenario)

    return scenarios


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

        # =======================================================================
        # EXPERIMENT 1: Alpha Sweep (Heterogeneity) - Fixed Buyer Ratio
        # =======================================================================
        for alpha in DIRICHLET_ALPHAS_TO_SWEEP:
            # Logic for Strategies (Same as before)
            if bias_type == "market_wide":
                seller_strat = "dirichlet";
                seller_alpha = alpha
                buyer_strat = "dirichlet";
                buyer_alpha = alpha
            elif bias_type == "buyer_only":
                seller_strat = "iid";
                seller_alpha = UNIFORM_ALPHA
                buyer_strat = "dirichlet";
                buyer_alpha = alpha
            elif bias_type == "seller_only":
                seller_strat = "dirichlet";
                seller_alpha = alpha
                buyer_strat = "iid";
                buyer_alpha = UNIFORM_ALPHA

            if alpha >= 100.0:
                seller_strat = "iid";
                buyer_strat = "iid"

            # Sync Adversary
            adv_strat = seller_strat
            adv_alpha = seller_alpha

            current_grid = static_grid.copy()
            current_grid[f"data.{modality}.strategy"] = [seller_strat]
            current_grid[f"data.{modality}.dirichlet_alpha"] = [seller_alpha]
            current_grid[f"data.{modality}.buyer_strategy"] = [buyer_strat]
            current_grid[f"data.{modality}.buyer_dirichlet_alpha"] = [buyer_alpha]

            # Fix Buyer Ratio for this sweep
            current_grid[f"data.{modality}.buyer_ratio"] = [DEFAULT_BUYER_RATIO]

            current_grid["adversary_seller_config.poisoning.data_distribution.strategy"] = [adv_strat]
            current_grid["adversary_seller_config.poisoning.data_distribution.dirichlet_alpha"] = [adv_alpha]

            hp_suffix = f"alpha_{alpha}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            # Generate
            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=scenario.modifiers,
                parameter_grid=current_grid
            )
            base_config = temp_scenario.base_config_factory()
            mod_config = initialize_adversary_data_distribution(copy.deepcopy(base_config))
            set_nested_attr(mod_config, "aggregation.aggregation_name", defense_name)
            for modifier in temp_scenario.modifiers: mod_config = modifier(mod_config)

            task_configs += generator.generate(mod_config, temp_scenario)

        # =======================================================================
        # EXPERIMENT 2: Buyer Ratio Sweep (Data Scarcity) - Fixed Alpha=0.5
        # =======================================================================
        # Only run this sweep for "Seller-Only Bias" to see if FLTrust can be fixed
        if bias_type == "seller_only":
            print(f"   -> Adding Buyer Ratio Sweep for {defense_name} (Seller-Only Bias)")

            fixed_alpha = 0.5  # Moderate heterogeneity

            for ratio in BUYER_RATIOS_TO_SWEEP:
                current_grid = static_grid.copy()

                # Fixed Heterogeneity Settings
                current_grid[f"data.{modality}.strategy"] = ["dirichlet"]
                current_grid[f"data.{modality}.dirichlet_alpha"] = [fixed_alpha]
                current_grid[f"data.{modality}.buyer_strategy"] = ["iid"]

                # VARY BUYER RATIO
                current_grid[f"data.{modality}.buyer_ratio"] = [ratio]

                # Sync Adversary
                current_grid["adversary_seller_config.poisoning.data_distribution.strategy"] = ["dirichlet"]
                current_grid["adversary_seller_config.poisoning.data_distribution.dirichlet_alpha"] = [fixed_alpha]

                hp_suffix = f"ratio_sweep_{ratio}"
                current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    modifiers=scenario.modifiers,
                    parameter_grid=current_grid
                )
                base_config = temp_scenario.base_config_factory()
                mod_config = initialize_adversary_data_distribution(copy.deepcopy(base_config))
                set_nested_attr(mod_config, "aggregation.aggregation_name", defense_name)
                for modifier in temp_scenario.modifiers: mod_config = modifier(mod_config)

                task_configs += generator.generate(mod_config, temp_scenario)

        all_generated_configs += task_configs

    print(f"\n✅ Step 11 Config Generation Complete! ({all_generated_configs} files)")