# FILE: generate_step7_adaptive_attack.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE,  # Use default adversary rate
    IMAGE_DEFENSES,
    # create_fixed_params_modifier,  <-- REMOVED BUGGY HELPER
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS  # <-- ADDED FOR FIX
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config  # Example
from entry.gradient_market.automate_exp.scenarios import Scenario, use_adaptive_attack, \
    use_cifar100_config  # Example

# Import needed attack modifiers
try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# ... (Constants are all correct) ...
ADAPTIVE_MODES_TO_TEST = ["gradient_manipulation", "data_manipulation"]
ADAPTIVE_THREAT_MODELS_TO_TEST = ["black_box", "gradient_inversion", "oracle"]
EXPLORATION_ROUNDS = 30
ADAPTIVE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}


# === THIS IS THE CORRECTED FUNCTION ===
def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against adaptive attackers."""
    print("\n--- Generating Step 7: Adaptive Attack Scenarios ---")
    scenarios = []
    modality = ADAPTIVE_SETUP["modality_name"]
    model_cfg_name = ADAPTIVE_SETUP["model_config_name"]
    current_defenses = IMAGE_DEFENSES

    for defense_name in current_defenses:
        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")
        if not tuned_defense_params:
            print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
            continue

        # === FIX 1: Create the setup modifier INSIDE the loop ===
        # This correctly applies BOTH Step 2.5 and Step 3 HPs
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # Apply Golden Training HPs (from Step 2.5)
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")
                if current_defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # Apply Tuned Defense HPs (from Step 3)
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)

                # Apply other fixed settings
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- Loop over threat models FIRST ---
        for threat_model in ADAPTIVE_THREAT_MODELS_TO_TEST:
            print(f"  -- Threat Model: {threat_model}")

            for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
                print(f"    -- Adaptive Mode: {adaptive_mode}")

                if threat_model != "black_box" and adaptive_mode == "data_manipulation":
                    print(f"       Skipping data_manipulation for {threat_model} (N/A)")
                    continue

                adaptive_modifier = use_adaptive_attack(
                    mode=adaptive_mode,
                    threat_model=threat_model,
                    exploration_rounds=EXPLORATION_ROUNDS
                )

                # === FIX 2: Define unique name AND save path ===
                scenario_name = f"step7_adaptive_{threat_model}_{adaptive_mode}_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"
                unique_save_path = f"./results/{scenario_name}"  # This is the key fix

                grid = {
                    ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
                    "experiment.adv_rate": [DEFAULT_ADV_RATE],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    "experiment.save_path": [unique_save_path]  # <-- ADDED
                }

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
                    modifiers=[
                        setup_modifier_func,  # <-- Use the new, correct modifier
                        ADAPTIVE_SETUP["dataset_modifier"],
                        adaptive_modifier,
                        lambda config: enable_valuation(
                            config,
                            influence=True,
                            loo=True,
                            loo_freq=10,
                            kernelshap=False
                        )
                    ],
                    parameter_grid=grid
                )
                scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (This is now correct) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step7_adaptive_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_adaptive_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 7 ---")

    # This loop is now correct because the unique save path
    # is ALREADY in the scenario's parameter_grid.
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 7 (Adaptive Attack Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Implement/Verify the AdaptiveAttackerSeller logic.")
    print(f"3. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"4. Analyze results.")
