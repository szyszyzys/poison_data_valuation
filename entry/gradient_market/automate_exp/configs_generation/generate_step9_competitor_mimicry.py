# FILE: generate_step9_competitor_mimicry.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    enable_valuation, get_tuned_defense_params
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_competitor_mimicry_attack, use_cifar100_config

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# ... (Constants are all correct) ...
ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4]
MIMICRY_STRATEGY = "noisy_copy"
TARGET_SELLER_ID = "bn_3"
NOISE_SCALE = 0.03
OBSERVATION_ROUNDS = 5
MIMICRY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"]


def generate_competitor_mimicry_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against competitor mimicry."""
    print("\n--- Generating Step 9: Competitor Mimicry Scenarios ---")
    scenarios = []
    modality = MIMICRY_SETUP["modality_name"]
    model_cfg_name = MIMICRY_SETUP["model_config_name"]

    for defense_name in DEFENSES_TO_TEST:
        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",  # Use default
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")
        if not tuned_defense_params:
            print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
            continue

        # === FIX 2: Create the setup modifier INSIDE the loop ===
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Apply Golden Training HPs (from Step 2.5)
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
                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # 3. Apply other fixed settings
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

                # 4. Explicitly Disable Other Seller Attacks
                config.adversary_seller_config.poisoning.type = PoisonType.NONE
                config.adversary_seller_config.sybil.is_sybil = True
                config.buyer_attack_config.is_active = False
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # 2. Create the mimicry attack modifier
        mimicry_modifier = use_competitor_mimicry_attack(
            target_seller_id=TARGET_SELLER_ID,
            strategy=MIMICRY_STRATEGY,
            noise_scale=NOISE_SCALE,
            observation_rounds=OBSERVATION_ROUNDS
        )

        # 3. Define the parameter grid (FIXED, no sweeps)
        grid = {
            MIMICRY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [MIMICRY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            # adv_rate will be set by the main loop
        }

        # 4. Create the Scenario
        scenario_name = f"step9_comp_mimicry_{MIMICRY_STRATEGY}_{defense_name}_{MIMICRY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=MIMICRY_SETUP["base_config_factory"],
            modifiers=[
                setup_modifier_func,  # <-- Use the new, correct modifier
                MIMICRY_SETUP["dataset_modifier"],
                mimicry_modifier,
                lambda config: enable_valuation(
                    config,
                    influence=True,
                    loo=True,
                    loo_freq=10,
                    kernelshap=False
                )
            ],
            parameter_grid=grid  # Grid does NOT sweep adv_rate
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (FIXED) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step9_competitor_mimicry"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_competitor_mimicry_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 9 ---")

    # === FIX 1: Manual loop to set unique save path for each adv_rate ===
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid
        static_grid = scenario.parameter_grid.copy()

        # Loop through each adv_rate
        for adv_rate in ADV_RATES_TO_SWEEP:

            # 1. Create the specific grid for this combination
            current_grid = static_grid.copy()
            current_grid["experiment.adv_rate"] = [adv_rate]

            # 2. Define unique output path
            hp_suffix = f"adv_rate_{adv_rate}"
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

    print(f"\n✅ Step 9 (Competitor Mimicry Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Implement/Verify the CompetitorMimicrySeller logic.")
    print(f"3. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"4. Analyze results.")
