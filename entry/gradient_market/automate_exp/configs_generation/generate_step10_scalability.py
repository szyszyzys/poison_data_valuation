# FILE: generate_step10_scalability.py

import sys
import copy
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    # DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, # Use specific fixed rates below
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES, ALL_DEFENSES,
    create_fixed_params_modifier,  # Use the standard helper
    enable_valuation, get_tuned_defense_params
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config # Example
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
# Evaluate how defense performance (utility and robustness) scales as the
# total number of sellers in the marketplace increases. This experiment uses
# GOLDEN training parameters, TUNED defense parameters, and a FIXED PROPORTION
# (percentage) of adversaries, along with a fixed attack strength. It helps
# assess the practical applicability of defenses in larger markets. ⬆️

# ==============================================================================
# --- Scalability Parameters ---
# ==============================================================================
# ## USER ACTION ##: Define the marketplace sizes to test
MARKETPLACE_SIZES = [10, 30, 50, 100] # Example: Small to large

# ## USER ACTION ##: Define the fixed adversary rate (percentage)
FIXED_ADV_RATE = 0.3 # e.g., 30% attackers at all scales

# ## USER ACTION ##: Define the fixed attack strength
FIXED_ATTACK_POISON_RATE = 0.5 # e.g., 50% poison rate, match defense tuning

# --- Focus Setup for Scalability Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
SCALABILITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10", # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn", # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    "attack_modifier": use_image_backdoor_attack # Standard attack type
}

# --- Defenses to Test ---
# ## USER ACTION ##: Select defenses for scalability testing
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"] # Or a subset

# --- Helper to apply fixed params (similar to previous) ---
# Ensures Golden Training + Tuned Defense HPs + Fixed Attack Strength are set
# Note: adv_rate and n_sellers are set in the grid
def create_fixed_params_modifier_scalability(
    modality: str,
    defense_params: Dict[str, Any],
    attack_modifier: Callable[[AppConfig], AppConfig],
    model_config_name: str
) -> Callable[[AppConfig], AppConfig]:
    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(model_config_name)
        if training_params:
            for key, value in training_params.items(): set_nested_attr(config, key, value)
        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items(): set_nested_attr(config, key, value)
        # 3. Apply the fixed attack type and strength
        config = attack_modifier(config)
        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)
        # config.adversary_seller_config.sybil.is_sybil = True # Optional fixed Sybil
        # 4. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
        # 5. Ensure Non-IID Seller data (standard setup)
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        # 6. Turn off valuation unless specifically desired for scalability
        config.valuation.run_influence = False
        config.valuation.run_loo = False
        config.valuation.run_kernelshap = False
        return config
    return modifier

# ==============================================================================
# --- MAIN CONFIG GENERATION FUNCTION ---
# ==============================================================================
def generate_scalability_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying n_sellers."""
    print("\n--- Generating Step 10: Scalability Scenarios (Fixed Rate) ---")
    scenarios = []
    modality = SCALABILITY_SETUP["modality_name"]
    model_cfg_name = SCALABILITY_SETUP["model_config_name"]
    print(f"Setup: {SCALABILITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Adv Rate: {FIXED_ADV_RATE*100}%")

    for defense_name in DEFENSES_TO_TEST:
        if defense_name not in TUNED_DEFENSE_PARAMS:
             print(f"  Skipping {defense_name}: No tuned parameters found.")
             continue
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")

        # --- Create the modifier to fix Training HPs, Defense HPs, and Attack Strength ---
        fixed_params_modifier = create_fixed_params_modifier_scalability(
            modality,
            tuned_defense_params,
            SCALABILITY_SETUP["attack_modifier"],
            model_cfg_name
        )

        # --- Define the parameter grid (Sweeps n_sellers, fixes adv_rate) ---
        parameter_grid = {
            SCALABILITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [SCALABILITY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True], # Keep early stopping
            "experiment.patience": [10],
            # --- Fixed Attack Rate ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            # --- Swept Parameter ---
            "experiment.n_sellers": MARKETPLACE_SIZES,
        }

        # --- Create the Scenario ---
        scenario_name = f"step10_scalability_{defense_name}_{SCALABILITY_SETUP['dataset_name']}_{model_cfg_name.split('_')[-1]}" # Shorter suffix

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=SCALABILITY_SETUP["base_config_factory"],
            modifiers=[fixed_params_modifier, SCALABILITY_SETUP["dataset_modifier"]], # Apply all fixed settings
            parameter_grid=parameter_grid # Sweep ONLY n_sellers
        )
        scenarios.append(scenario)

    return scenarios

# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step10_scalability"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_scalability_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 10 ---")
    # --- Standard Generator Loop ---
    # The generator applies modifiers first, then expands the grid (n_sellers).
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, attack strength/type, non-iid)
        for modifier in scenario.modifiers:
             modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (the n_sellers sweep)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 10 (Scalability Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results by plotting 'n_sellers' vs. 'test_acc'/'backdoor_asr' for each defense.")
    print("   -> Check if accuracy degrades or ASR increases at larger scales.")