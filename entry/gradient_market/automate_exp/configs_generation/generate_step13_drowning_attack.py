# FILE: generate_step13_drowning_attack.py

import copy
from pathlib import Path
from typing import List, Callable, Dict, Any

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig
# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.config_generator import set_nested_attr, ExperimentGenerator
from entry.gradient_market.automate_exp.scenarios import use_cifar10_config, Scenario, use_drowning_attack

## Purpose
# Evaluate the "Targeted Drowning Attack" against clustering-based defenses.
# This attack uses Sybils to poison the market centroid, making a specific
# benign seller appear as an outlier to get them rejected.
# This tests the vulnerability of MartFL, using FLTrust as a control.

# --- Attack Parameters ---
FIXED_ADV_RATE = 0.3  # 30% of the market are Sybil attackers
TARGET_VICTIM_ID = "bn_0"  # The benign seller to attack
ATTACK_STRENGTH = 1.0  # Repulsion strength (alpha)

# --- Focus Setup ---
DROWNING_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR10",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar10_cnn",
    "dataset_modifier": use_cifar10_config,
}

# --- Defenses to Test ---
# MartFL is the target, FLTrust is the control
DEFENSES_TO_TEST = ["martfl", "fltrust"]


# --- Helper to apply fixed params (and NO poisoning) ---
def create_fixed_params_modifier_drowning(
        modality: str,
        defense_params: Dict[str, Any],
        model_config_name: str
) -> Callable[[AppConfig], AppConfig]:
    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(model_config_name)
        if training_params:
            for key, value in training_params.items(): set_nested_attr(config, key, value)
        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items(): set_nested_attr(config, key, value)
        # 3. Ensure Non-IID Seller data
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        # 4. Explicitly Disable Poisoning (Drowning is the attack)
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        # 5. Disable other attacks
        config.buyer_attack_config.is_active = False
        if hasattr(config.adversary_seller_config, 'adaptive_attack'):
            config.adversary_seller_config.adaptive_attack.is_active = False
        # 6. Turn off valuation
        config.valuation.run_influence = False
        config.valuation.run_loo = False
        config.valuation.run_kernelshap = False
        return config

    return modifier


# --- Main Generation Function ---
def generate_drowning_attack_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 13: Targeted Drowning Attack Scenarios ---")
    scenarios = []
    modality = DROWNING_SETUP["modality_name"]
    model_cfg_name = DROWNING_SETUP["model_config_name"]

    for defense_name in DEFENSES_TO_TEST:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"  # Use standard HPs
        )
        print(f"-- Processing Defense: {defense_name} (Victim: {TARGET_VICTIM_ID})")

        # 1. Create modifier for Golden Training + Tuned Defense HPs + No Poisoning
        fixed_params_modifier = create_fixed_params_modifier_drowning(
            modality,
            tuned_defense_params,
            model_cfg_name
        )

        # 2. Create the drowning attack modifier
        drowning_modifier = use_drowning_attack(
            target_victim_id=TARGET_VICTIM_ID,
            attack_strength=ATTACK_STRENGTH
        )

        # 3. Define the parameter grid (fixed attack rate)
        grid = {
            DROWNING_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [DROWNING_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [False],  # Must run full rounds
            "experiment.global_rounds": [100],  # Run for 100-200 rounds
            "experiment.adv_rate": [FIXED_ADV_RATE],
        }

        # 4. Create the Scenario
        scenario_name = f"step13_drowning_{defense_name}_{TARGET_VICTIM_ID}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=DROWNING_SETUP["base_config_factory"],
            modifiers=[
                fixed_params_modifier,
                DROWNING_SETUP["dataset_modifier"],
                drowning_modifier  # Enables the drowning attack
            ],
            parameter_grid=grid
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step13_drowning_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_drowning_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 13 ---")
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 13 (Targeted Drowning Attack) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"2. Analyze results: Plot the 'selection_rate' of seller '{TARGET_VICTIM_ID}'")
    print(f"   over time (rounds) for MartFL vs. FLTrust.")
    print(f"3. Expectation: Rate for MartFL -> 0, Rate for FLTrust -> stays high.")
