
# FILE: generate_step8_buyer_attacks.py

import sys
import copy
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    # No DEFAULT_ADV_RATE needed here as seller attacks are off
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES, ALL_DEFENSES,
    create_fixed_params_modifier, # Use the standard helper
    enable_valuation
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, use_buyer_dos_attack, \
    use_buyer_starvation_attack, use_buyer_erosion_attack, use_buyer_class_exclusion_attack, \
    use_buyer_oscillating_attack, use_buyer_orthogonal_pivot_attack  # Example
# Import needed attack modifiers
try:
    from common.gradient_market_configs import AppConfig, PoisonType, BuyerAttackConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose
# Evaluate defense robustness against attacks originating from the BUYER/Aggregator.
# This experiment tests how well TUNED defenses handle various buyer-side
# manipulations (DoS, Starvation, Erosion, Class Exclusion, Oscillating Objectives)
# using GOLDEN training parameters and Non-IID seller data. Crucial for assessing
# security in a marketplace where the central entity might be compromised. 🛒🛡️

# --- Focus Setup for Buyer Attack Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
BUYER_ATTACK_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10", # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn", # lowercase (or resnet18 if preferred)
    "dataset_modifier": use_cifar10_config,
    # No seller attack modifier needed here
}

# --- Buyer Attack Configurations to Test ---
# ## USER ACTION ##: Adjust parameters within these calls as needed
BUYER_ATTACK_CONFIGS = [
    ("dos", use_buyer_dos_attack()),
    ("starvation", use_buyer_starvation_attack(target_classes=[0, 1])), # Example target
    ("erosion", use_buyer_erosion_attack()),
    ("class_exclusion_neg", use_buyer_class_exclusion_attack(exclude_classes=[7, 8, 9], gradient_scale=1.2)), # Example exclude
    ("class_exclusion_pos", use_buyer_class_exclusion_attack(target_classes=[0, 1, 2], gradient_scale=1.0)), # Example target
    ("oscillating_binary", use_buyer_oscillating_attack(strategy="binary_flip", period=5)), # Example period
    ("oscillating_random", use_buyer_oscillating_attack(strategy="random_walk", subset_size=3)), # Example size
    ("oscillating_drift", use_buyer_oscillating_attack(strategy="adversarial_drift", drift_rounds=60, classes_a=[0, 1])), # Example drift
    ("orthogonal_pivot_legacy", use_buyer_orthogonal_pivot_attack(target_seller_id="bn_5")), # Optional legacy
]

# --- Defenses to Test ---
# Focus on defenses potentially affected by buyer behavior (e.g., reference gradient)
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl"] # SkyMask less directly affected?

# --- Helper to apply fixed Golden Training & Tuned Defense HPs ---
# (And ensure NO seller attacks)
def create_fixed_params_modifier_buyer_atk(
    modality: str,
    defense_params: Dict[str, Any],
    model_config_name: str
) -> Callable[[AppConfig], AppConfig]:
    # --- (Implementation from previous answer - applies Golden HPs, Tuned Def HPs, SkyMask type, Non-IID data) ---
    def modifier(config: AppConfig) -> AppConfig:
        training_params = GOLDEN_TRAINING_PARAMS.get(modality)
        if training_params:
            for key, value in training_params.items(): set_nested_attr(config, key, value)
        for key, value in defense_params.items(): set_nested_attr(config, key, value)
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        # --- Explicitly Disable Seller Attacks ---
        config.experiment.adv_rate = 0.0
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        # Deactivate other seller attacks (adaptive, mimicry, etc.) if they exist in base config
        if hasattr(config.adversary_seller_config, 'adaptive_attack'): config.adversary_seller_config.adaptive_attack.is_active = False
        if hasattr(config.adversary_seller_config, 'mimicry_attack'): config.adversary_seller_config.mimicry_attack.is_active = False
        # --- Turn off valuation (usually not the focus here, but can enable if needed) ---
        config.valuation.run_influence = False
        config.valuation.run_loo = False
        config.valuation.run_kernelshap = False
        return config
    return modifier

# ==============================================================================
# --- MAIN CONFIG GENERATION FUNCTION ---
# ==============================================================================
def generate_buyer_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against various buyer attacks."""
    print("\n--- Generating Step 8: Buyer Attack Scenarios ---")
    scenarios = []
    modality = BUYER_ATTACK_SETUP["modality_name"]
    model_cfg_name = BUYER_ATTACK_SETUP["model_config_name"]

    for defense_name in DEFENSES_TO_TEST:
        if defense_name not in TUNED_DEFENSE_PARAMS:
             print(f"  Skipping {defense_name}: No tuned parameters found.")
             continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Processing Defense: {defense_name}")

        # 1. Create modifier for Golden Training + Tuned Defense HPs + No Seller Attack
        fixed_params_modifier = create_fixed_params_modifier_buyer_atk(
            modality,
            tuned_defense_params,
            model_cfg_name
        )

        # 2. Loop through buyer attack types defined above
        for attack_tag, buyer_attack_modifier in BUYER_ATTACK_CONFIGS:
            print(f"  -- Buyer Attack Type: {attack_tag}")

            # 3. Define the base grid (fixed parameters for this run)
            grid = {
                BUYER_ATTACK_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [BUYER_ATTACK_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                # Seller adv_rate is set to 0.0 by the fixed_params_modifier
            }

            # 4. Create the Scenario
            scenario_name = f"step8_buyer_attack_{attack_tag}_{defense_name}_{BUYER_ATTACK_SETUP['dataset_name']}"

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=BUYER_ATTACK_SETUP["base_config_factory"],
                modifiers=[
                    fixed_params_modifier, # Sets Golden Train, Tuned Def, No Seller Atk, Non-IID
                    BUYER_ATTACK_SETUP["dataset_modifier"],
                    buyer_attack_modifier # Enables the specific buyer attack
                ],
                parameter_grid=grid # Grid only contains fixed exp params
            )
            scenarios.append(scenario)

    return scenarios

# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step8_buyer_attacks"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_buyer_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 8 ---")
    # --- Standard Generator Loop ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, NO seller attack, buyer attack type)
        for modifier in scenario.modifiers:
             modified_base_config = modifier(modified_base_config)
        # Generator just applies the fixed grid parameters
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 8 (Buyer Attack Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Implement/Verify the MaliciousBuyerProxy logic for all attack types.")
    print(f"3. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"4. Analyze results: Compare performance (Acc, Seller Selection Rates if applicable)")
    print(f"   across defenses for each buyer attack type.")