# FILE: generate_step13_drowning_attack.py

import copy
from pathlib import Path
from typing import List, Callable

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig
# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,  # <-- ADDED
    NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.config_generator import set_nested_attr, ExperimentGenerator
from entry.gradient_market.automate_exp.scenarios import Scenario, use_drowning_attack, \
    use_cifar100_config


def use_sybil_drowning_attack(
        target_victim_id: str,
        attack_strength: float
) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier that enables the SybilCoordinator with the
    Targeted Drowning Attack strategy.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Enable the Sybil Coordinator
        config.sybil_coordinator.is_active = True

        # 2. Tell the coordinator to use 'drowning' as the default strategy
        #    for all registered Sybil sellers.
        config.sybil_coordinator.sybil_config.gradient_default_mode = "drowning"

        # 3. Create and set the specific config for the DrowningStrategy.
        #    This matches the SybilDrowningConfig object we created.
        drowning_strat_config = {
            "victim_id": target_victim_id,
            "attack_strength": attack_strength
        }

        # 4. Set this config dict in the coordinator's config tree
        set_nested_attr(
            config,
            "sybil_coordinator.sybil_config.strategy_configs.drowning",
            drowning_strat_config
        )

        # 5. (CRITICAL) Ensure the "adversaries" (defined by adv_rate)
        #    are actually Sybils. This tells the system to register
        #    them with the SybilCoordinator.
        config.adversary_seller_config.is_sybil = True

        # 6. Disable their "normal" poisoning, as the coordinator
        #    will be providing their gradient.
        config.adversary_seller_config.poisoning.type = PoisonType.NONE

        return config

    return modifier


FIXED_ADV_RATE = 0.3
TARGET_VICTIM_ID = "bn_0"
ATTACK_STRENGTH = 1.0
DROWNING_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}
DEFENSES_TO_TEST = ["martfl", "fltrust", "skymask"]


# === THIS IS THE CORRECTED FUNCTION ===
def generate_drowning_attack_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 13: Targeted Drowning Attack Scenarios ---")
    scenarios = []
    modality = DROWNING_SETUP["modality_name"]
    model_cfg_name = DROWNING_SETUP["model_config_name"]

    for defense_name in DEFENSES_TO_TEST:
        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name} (Victim: {TARGET_VICTIM_ID})")
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
                golden_hp_key = f"{current_defense_name}_{current_model_cfg_name}_local_clip"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # 2. Apply Tuned Defense HPs (from Step 3)
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)

                # 3. Apply other fixed settings
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

                # 4. Explicitly Disable Poisoning (Drowning is the attack)
                config.adversary_seller_config.poisoning.type = PoisonType.NONE
                config.buyer_attack_config.is_active = False

                # 5. Turn off valuation
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # 2. Create the drowning attack modifier
        drowning_modifier = use_sybil_drowning_attack(
            target_victim_id=TARGET_VICTIM_ID,
            attack_strength=ATTACK_STRENGTH
        )

        # === FIX 1: Define name and save_path BEFORE the grid ===
        scenario_name = f"step13_drowning_{defense_name}_{TARGET_VICTIM_ID}"
        unique_save_path = f"./results/{scenario_name}"

        # 3. Define the parameter grid (fixed attack rate)
        grid = {
            DROWNING_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [DROWNING_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [False],
            "experiment.global_rounds": [100],
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "experiment.save_path": [unique_save_path]  # <-- ADDED
        }

        # 4. Create the Scenario
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=DROWNING_SETUP["base_config_factory"],
            modifiers=[
                setup_modifier_func,  # <-- Use the new, correct modifier
                DROWNING_SETUP["dataset_modifier"],
                drowning_modifier
            ],
            parameter_grid=grid
        )
        scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (This is now correct) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step13_drowning_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_drowning_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 13 ---")

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

    print(f"\n✅ Step 13 (Targeted Drowning Attack) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"2. Analyze results: Plot the 'selection_rate' of seller '{TARGET_VICTIM_ID}'")
    print(f"   over time (rounds) for MartFL vs. FLTrust.")
    print(f"3. Expectation: Rate for MartFL -> 0, Rate for FLTrust -> stays high.")
