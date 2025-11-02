# FILE: generate_step9_competitor_mimicry.py

import sys
import copy
from pathlib import Path
from typing import List, Callable, Dict, Any

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    # DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, # AdvRate swept here
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES, ALL_DEFENSES,
    create_fixed_params_modifier,  # Use the standard helper
    enable_valuation, get_tuned_defense_params
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar10_config, \
    use_competitor_mimicry_attack  # Example
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
# Evaluate defense robustness against Competitor Mimicry attacks.
# This experiment tests how well TUNED defenses handle adversaries specifically
# trying to steal market share (increase selection rate) from a target seller
# by submitting similar gradients. This focuses on economic sabotage rather than
# global model poisoning. Uses GOLDEN training parameters. 🎭

# --- Attack Parameters to Sweep ---
# Sweep the adversary rate to see how multiple mimics amplify the effect
ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4] # Percentage of sellers engaging in mimicry

# --- Mimicry Attack Settings ---
# ## USER ACTION ##: Confirm these settings
MIMICRY_STRATEGY = "noisy_copy" # Strategy used by mimics
TARGET_SELLER_ID = "bn_0" # Target the first benign seller (often high quality)
NOISE_SCALE = 0.03
OBSERVATION_ROUNDS = 5

# --- Focus Setup for Mimicry Analysis ---
# ## USER ACTION ##: Choose one representative setup (model/dataset)
MIMICRY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR10", # lowercase
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn",
    "dataset_modifier": use_cifar10_config,
    # No base poisoning attack modifier needed, mimicry IS the attack behavior
}

# --- Defenses to Test ---
# Focus on defenses that might be vulnerable to similarity-based attacks
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl"] # SkyMask less relevant?

# --- Helper to apply fixed Golden Training & Tuned Defense HPs ---
# (And ensure NO other seller attacks interfere)
def create_fixed_params_modifier_mimicry(
    modality: str,
    defense_params: Dict[str, Any],
    model_config_name: str
) -> Callable[[AppConfig], AppConfig]:
    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(modality)
        if training_params:
            for key, value in training_params.items(): set_nested_attr(config, key, value)
        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items(): set_nested_attr(config, key, value)
        # 3. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
        # 4. Ensure Non-IID Seller data
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        # --- 5. Explicitly Disable Other Seller Attacks ---
        # AdvRate is set in the grid, but ensure poisoning/sybil/adaptive off
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        if hasattr(config.adversary_seller_config, 'adaptive_attack'): config.adversary_seller_config.adaptive_attack.is_active = False
        # Ensure buyer attack is off
        config.buyer_attack_config.is_active = False
        return config
    return modifier

# ==============================================================================
# --- MAIN CONFIG GENERATION FUNCTION ---
# ==============================================================================
def generate_competitor_mimicry_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against competitor mimicry."""
    print("\n--- Generating Step 9: Competitor Mimicry Scenarios ---")
    scenarios = []
    modality = MIMICRY_SETUP["modality_name"]
    model_cfg_name = MIMICRY_SETUP["model_config_name"]

    for defense_name in DEFENSES_TO_TEST:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")

        # 1. Create modifier for Golden Training + Tuned Defense HPs + No Other Attacks
        fixed_params_modifier = create_fixed_params_modifier_mimicry(
            modality,
            tuned_defense_params,
            model_cfg_name
        )

        # 2. Create the mimicry attack modifier with specific settings
        mimicry_modifier = use_competitor_mimicry_attack(
            target_seller_id=TARGET_SELLER_ID,
            strategy=MIMICRY_STRATEGY,
            noise_scale=NOISE_SCALE,
            observation_rounds=OBSERVATION_ROUNDS
        )

        # 3. Define the parameter grid (Sweeps adv_rate)
        grid = {
            MIMICRY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [MIMICRY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True], # Keep early stopping
            "experiment.patience": [10],
            "experiment.adv_rate": ADV_RATES_TO_SWEEP, # Sweep adversary rate
        }

        # 4. Create the Scenario
        scenario_name = f"step9_comp_mimicry_{MIMICRY_STRATEGY}_{defense_name}_{MIMICRY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=MIMICRY_SETUP["base_config_factory"],
            modifiers=[
                fixed_params_modifier, # Sets Golden Train, Tuned Def, Non-IID, No Other Atk
                MIMICRY_SETUP["dataset_modifier"],
                mimicry_modifier, # Enables the specific mimicry attack
                lambda config: enable_valuation(
                    config,
                    influence=True,
                    loo=True,
                    loo_freq=10,
                    kernelshap=False
                )
            ],
            parameter_grid=grid # Grid sweeps adv_rate
        )
        scenarios.append(scenario)

    return scenarios

# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step9_competitor_mimicry"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_competitor_mimicry_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 9 ---")
    # --- Standard Generator Loop ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, mimicry attack, valuation)
        for modifier in scenario.modifiers:
             modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (the adv_rate sweep)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 9 (Competitor Mimicry Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Implement/Verify the CompetitorMimicrySeller logic.")
    print(f"3. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"4. Analyze results: Plot Selection Rate and Valuation Score (Influence/LOO)")
    print(f"   of the target seller ('{TARGET_SELLER_ID}') and the average mimicry attacker vs. 'adv_rate' for each defense.")