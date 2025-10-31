# FILE: generate_step11_heterogeneity.py

import copy
import sys
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,  # Fixed attack strength
    # Use the standard helper
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
# Evaluate how the DEGREE of data heterogeneity (Non-IID level) impacts the
# performance (utility and robustness) of TUNED defense mechanisms.
# This experiment uses GOLDEN training parameters and a FIXED attack scenario,
# while varying the Dirichlet alpha value used for data partitioning among sellers.
# Helps understand how sensitive defenses are to different levels of data skew. 🌍

# ==============================================================================
# --- Heterogeneity Parameters ---
# ==============================================================================
# ## USER ACTION ##: Define the Dirichlet alpha values to sweep
# High alpha = More IID-like distribution
# Low alpha  = More Non-IID (skewed) distribution
DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.5, 0.1]  # Example range

# --- Fixed Attack Settings ---
# Use the same strength as defense tuning for consistency
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE

# --- Focus Setup for Heterogeneity Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
HETEROGENEITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR10",  # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn",  # lowercase, use your best model
    "dataset_modifier": use_cifar10_config,
    "attack_modifier": use_image_backdoor_attack  # Standard attack type
}

# --- Defenses to Test ---
# ## USER ACTION ##: Select defenses for heterogeneity testing
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"]  # Or a subset


# --- Helper to apply fixed params (similar to previous) ---
# Ensures Golden Training + Tuned Defense HPs + Fixed Attack Strength/Type are set
# Note: Data strategy/alpha are set in the grid for this experiment
def create_fixed_params_modifier_heterogeneity(
        modality: str,
        defense_params: Dict[str, Any],
        attack_modifier: Callable[[AppConfig], AppConfig],
        model_config_name: str
) -> Callable[[AppConfig], AppConfig]:
    def modifier(config: AppConfig) -> AppConfig:

        # 1. Apply Golden Training HPs (USING THE CORRECT KEY)
        training_params = GOLDEN_TRAINING_PARAMS.get(model_config_name) # <-- CORRECT

        if training_params:
            for key, value in training_params.items(): set_nested_attr(config, key, value)
        else:
            # Add a warning in case you add a new model
            print(f"!!!!!!!!!! WARNING !!!!!!!!!!!")
            print(f"No Golden HPs found for model: '{model_config_name}'")

        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items(): set_nested_attr(config, key, value)

        # 3. Apply the fixed attack type and strength
        config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
        config = attack_modifier(config)
        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

        # 4. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            # This logic is also now correct, as it uses model_config_name
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

        # 5. Data distribution is set in the grid, ensure strategy is compatible
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")  # Force dirichlet

        # 6. Turn off valuation
        config.valuation.run_influence = False
        config.valuation.run_loo = False
        config.valuation.run_kernelshap = False
        return config

    return modifier

# ==============================================================================
# --- MAIN CONFIG GENERATION FUNCTION ---
# ==============================================================================
def generate_heterogeneity_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying Dirichlet alpha."""
    print("\n--- Generating Step 11: Heterogeneity Impact Scenarios ---")
    scenarios = []
    modality = HETEROGENEITY_SETUP["modality_name"]
    model_cfg_name = HETEROGENEITY_SETUP["model_config_name"]
    print(f"Setup: {HETEROGENEITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Attack")

    for defense_name in DEFENSES_TO_TEST:
        if defense_name not in TUNED_DEFENSE_PARAMS:
            print(f"  Skipping {defense_name}: No tuned parameters found.")
            continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        # --- Create the modifier to fix Training HPs, Defense HPs, and Attack ---
        fixed_params_modifier = create_fixed_params_modifier_heterogeneity(
            modality,
            tuned_defense_params,
            HETEROGENEITY_SETUP["attack_modifier"],
            model_cfg_name
        )

        # --- Define the parameter grid (Sweeps dirichlet_alpha) ---
        parameter_grid = {
            HETEROGENEITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [HETEROGENEITY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            f"data.{modality}.strategy": ["dirichlet"],  # Strategy is fixed
            # --- Swept Parameter ---
            f"data.{modality}.dirichlet_alpha": DIRICHLET_ALPHAS_TO_SWEEP,
        }

        # --- Create the Scenario ---
        scenario_name = f"step11_heterogeneity_{defense_name}_{HETEROGENEITY_SETUP['dataset_name']}_{model_cfg_name.split('_')[-1]}"  # Shorter suffix

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=HETEROGENEITY_SETUP["base_config_factory"],
            modifiers=[fixed_params_modifier, HETEROGENEITY_SETUP["dataset_modifier"]],  # Apply all fixed settings
            parameter_grid=parameter_grid  # Sweep ONLY dirichlet_alpha
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step11_heterogeneity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_heterogeneity_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 11 ---")
    # --- Standard Generator Loop ---
    # The generator applies modifiers first, then expands the grid (dirichlet_alpha).
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, attack type/strength)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (the alpha sweep)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 11 (Heterogeneity Impact Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results by plotting 'dirichlet_alpha' vs. 'test_acc'/'backdoor_asr' for each defense.")
    print("   -> Check how performance degrades as alpha decreases (more Non-IID).")
