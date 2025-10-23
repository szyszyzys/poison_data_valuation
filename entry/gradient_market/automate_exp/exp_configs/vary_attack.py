# FILE: entry/gradient_market/automate_exp/run_benchmark.py (New File)

import copy
import sys
from typing import Callable

# --- (Imports remain the same) ---
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_image_backdoor_attack
)

# from entry.gradient_market.automate_exp.scenarios import use_trec_config # Add if needed
try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Use Golden Training Parameters (Same as before) ---
GOLDEN_PARAMS_PER_MODALITY = {
    "image": {"learning_rate": 0.01, "local_epochs": 2},  # Replace with actual values
    # Add other modalities if testing them
}

# --- !!! IMPORTANT: SET YOUR TUNED DEFENSE PARAMETERS HERE !!! ---
# (Values found from step3_defense_tuning.py)
TUNED_DEFENSE_PARAMS = {
    "fltrust": {
        "aggregation.method": "fltrust",
        "aggregation.clip_norm": 5.0,  # Example: Replace with your best value
    },
    "martfl": {
        "aggregation.method": "martfl",
        "aggregation.martfl.change_base": True,  # Example: Replace
        "aggregation.martfl.clip": True,  # Example: Replace
        "aggregation.clip_norm": 5.0,  # Example: Replace
        "aggregation.martfl.initial_baseline": "buyer",  # Example: Replace
        "aggregation.martfl.max_k": 5,  # Example: Replace
    },
    "skymask": {
        "aggregation.method": "skymask",
        "aggregation.skymask.clip": True,
        "aggregation.clip_norm": 10.0,
        "aggregation.skymask.mask_epochs": 20,  # Example: Replace
        "aggregation.skymask.mask_lr": 0.01,  # Example: Replace
        "aggregation.skymask.mask_threshold": 0.7,  # Example: Replace
        "aggregation.skymask.mask_clip": 1.0,  # Example: Replace
        # Note: sm_model_type will be set based on the combo
    },
    "fedavg": {  # Include FedAvg as the undefended baseline
        "aggregation.method": "fedavg",
    }
}

# --- Attack Parameters to SWEEP --- ⚔️
# Vary the percentage of attackers and how much they poison
ATTACK_ADV_RATES_TO_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Include 0.0 (benign baseline)
ATTACK_POISON_RATES_TO_SWEEP = [0.2, 0.5, 0.8, 1.0]

NUM_SEEDS_PER_CONFIG = 3

# --- Valuation Settings (Keep Enabled for Analysis) ---
VALUATION_PARAMS = {
    "valuation.run_similarity": [True],
    "valuation.run_influence": [True],
    "valuation.run_loo": [True],
    "valuation.loo_frequency": [10],
    "valuation.run_kernelshap": [True],
    "valuation.kernelshap_frequency": [20],
    "valuation.kernelshap_samples": [32]
}

# --- Define ONE Model/Dataset combination for the benchmark ---
BENCHMARK_COMBO = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",
    "model_structure": "resnet18",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_resnet18",
    "attack_modifier": use_image_backdoor_attack,
    "dataset_modifier": use_cifar10_config,
    "sm_model_type": "resnet18",  # For SkyMask
}


# --- Function to apply Golden Training Parameters ONLY ---
# (Attack params are now swept in the grid)
def apply_benchmark_setup(config: AppConfig, modality: str, dataset_modifier: Callable,
                          attack_modifier: Callable) -> AppConfig:
    """Applies dataset modifier, golden training params, and attack modifier."""
    config = dataset_modifier(config)  # Apply dataset specifics first

    # Apply Golden Training Parameters
    if modality in GOLDEN_PARAMS_PER_MODALITY:
        params = GOLDEN_PARAMS_PER_MODALITY[modality]
        config.training.learning_rate = params["learning_rate"]
        config.training.local_epochs = params["local_epochs"]
        print(f"  Applied Golden Training Params for {modality}")
    else:
        print(f"  Warning: No Golden Params for modality '{modality}'")

    # Apply the base attack structure (poison rate is set by grid)
    config = attack_modifier(config)
    print("  Applied base attack modifier")
    return config


# --- Main Execution Block ---
if __name__ == "__main__":

    output_dir = "./configs_generated/step4_benchmark"  # New directory
    generator = ExperimentGenerator(output_dir)
    all_benchmark_scenarios = []

    print("\n--- Generating Defense Benchmark Scenarios (Step 4) ---")

    combo_config = BENCHMARK_COMBO
    modality = combo_config["modality_name"]
    print(
        f"\n-- Processing Benchmark for: Modality={modality}, Dataset={combo_config['dataset_name']}, Model={combo_config['model_structure']}")

    # Iterate through each DEFENSE method (using its tuned params)
    for defense_name, fixed_defense_params in TUNED_DEFENSE_PARAMS.items():

        # Skip incompatible defenses
        if modality != "image" and defense_name == "skymask":
            print(f"   Skipping {defense_name} benchmark for non-image modality.")
            continue

        scenario_name = f"benchmark_{defense_name}_{modality}_{combo_config['dataset_name']}_{combo_config['model_structure']}"
        print(f"  - Defining scenario for {defense_name}: {scenario_name}")


        # Define the setup modifier lambda
        def create_setup_modifier(current_combo_config):
            def setup_modifier_inner(config: AppConfig) -> AppConfig:
                return apply_benchmark_setup(
                    config,
                    current_combo_config["modality_name"],
                    current_combo_config["dataset_modifier"],
                    current_combo_config["attack_modifier"]
                )

            return setup_modifier_inner


        current_setup_modifier = create_setup_modifier(combo_config)

        # Build the parameter grid for this defense
        full_parameter_grid = {
            # Fixed experiment setup
            "experiment.dataset_name": [combo_config["dataset_name"]],
            "experiment.model_structure": [combo_config["model_structure"]],
            combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.global_rounds": [100],  # Or your desired length
            "experiment.use_early_stopping": [False],  # Must be OFF for benchmark

            # Fixed defense parameters (from TUNED_DEFENSE_PARAMS)
            **fixed_defense_params,

            # Sweep Attack Parameters
            "experiment.adv_rate": ATTACK_ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": ATTACK_POISON_RATES_TO_SWEEP,

            # Valuation settings (fixed)
            **VALUATION_PARAMS,
        }

        # Add sm_model_type specifically for SkyMask runs
        if defense_name == "skymask":
            full_parameter_grid["aggregation.skymask.sm_model_type"] = [combo_config["sm_model_type"]]
        # Ensure sm_model_type isn't present for other defenses
        elif "aggregation.skymask.sm_model_type" in full_parameter_grid:
            del full_parameter_grid["aggregation.skymask.sm_model_type"]

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=combo_config["base_config_factory"],
            modifiers=[current_setup_modifier],  # Applies dataset, golden train params, attack modifier
            parameter_grid=full_parameter_grid
        )
        all_benchmark_scenarios.append(scenario)

    # --- Generate Config Files (Same logic as before) ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    # ... (rest of the generation loop is identical) ...
    for scenario in all_benchmark_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)  # Apply setup
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All defense benchmark configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments using configs in '{output_dir}'.")
    print("2. Aggregate results.")
    print("3. Analyze results to create robustness plots:")
    print("   - Plot 'test_acc' vs 'adv_rate' (line per defense, separate plots per poison_rate).")
    print("   - Plot 'test_asr' vs 'adv_rate' (line per defense, separate plots per poison_rate).")
    print("   - Plot 'FPR' vs 'adv_rate' (line per defense, separate plots per poison_rate).")
    print("   - Analyze valuation metrics (from seller_metrics.csv) vs attack strength.")
