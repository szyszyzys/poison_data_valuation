# FILE: generate_step7_adaptive_attack.py

import copy
import sys
from pathlib import Path
from typing import List, Callable

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE,
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS
)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
ADAPTIVE_MODES_TO_TEST = ["gradient_manipulation", "data_poisoning"]
ADAPTIVE_THREAT_MODELS_TO_TEST = ["black_box", "gradient_inversion", "oracle"]
EXPLORATION_ROUNDS = 30
MIMIC_STRENGTH = 0.5  # Alpha for Oracle/Inversion attacks

ADAPTIVE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}


# --- Helper Function: Adaptive Attack Modifier ---
def use_adaptive_attack(
        mode: str,
        threat_model: str = "black_box",
        exploration_rounds: int = 30,
        with_backdoor: bool = False  # Default False for pure manipulation/evasion
) -> Callable[[AppConfig], AppConfig]:
    """ Returns a modifier to enable and configure the adaptive attacker. """

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Activate Adaptive Logic
        adv_cfg = config.adversary_seller_config.adaptive_attack
        adv_cfg.is_active = True
        adv_cfg.attack_mode = mode
        adv_cfg.threat_model = threat_model
        adv_cfg.exploration_rounds = exploration_rounds

        # 2. Handle Poisoning vs Pure Manipulation
        if with_backdoor:
            # Goal: Inject Backdoor while evading detection
            config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
            config.adversary_seller_config.poisoning.poison_rate = 0.5
        else:
            # Goal: Manipulate selection (Free Rider / Untargeted Noise)
            # This disables the 'stealthy_blend' strategy in the class logic,
            # forcing the Bandit to learn 'reduce_norm' or 'add_noise'.
            config.adversary_seller_config.poisoning.type = PoisonType.NONE
        adv_cfg.noise_level = 0.1  # Amount of Gaussian noise to add
        adv_cfg.scale_factor = 0.5  # Factor for reducing norm (Free riding)

        # 3. Sync "Stealthy Blend" Params (used by class even if poison is None)
        blend_cfg = config.adversary_seller_config.drowning_attack
        blend_cfg.mimicry_rounds = exploration_rounds
        blend_cfg.attack_intensity = 0.3
        blend_cfg.replacement_strategy = "layer_wise"

        # 4. Set Oracle/Inversion Params
        if threat_model in ["oracle", "gradient_inversion"]:
            adv_cfg.mimic_strength = MIMIC_STRENGTH

        # 5. Deactivate other standalone attacks
        config.adversary_seller_config.sybil.is_sybil = False

        return config

    return modifier


def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against adaptive attackers."""
    print("\n--- Generating Step 7: Adaptive Attack Scenarios ---")
    scenarios = []
    modality = ADAPTIVE_SETUP["modality_name"]
    model_cfg_name = ADAPTIVE_SETUP["model_config_name"]
    current_defenses = ["martfl"]  # Add "skymask", "fltrust" etc. as needed

    for defense_name in current_defenses:
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

        # --- Common Setup Modifier ---
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # Apply Golden HPs
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # Apply Defense Specifics
                if current_defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # Apply Tuned Defense HPs
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)

                # Apply Data Distribution
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- SCENARIO 1: BASELINE (No Attack) ---
        print("    -- Adaptive Mode: 0. Baseline (No Attack)")
        baseline_scenario_name = f"step7_baseline_no_attack_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"
        baseline_save_path = f"./results/{baseline_scenario_name}"

        baseline_grid = {
            ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
            "experiment.adv_rate": [0.0],  # 0% Adversaries
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.save_path": [baseline_save_path]
        }

        scenarios.append(Scenario(
            name=baseline_scenario_name,
            base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
            modifiers=[
                setup_modifier_func,
                ADAPTIVE_SETUP["dataset_modifier"],
                lambda config: enable_valuation(
                    config, influence=True, loo=True, loo_freq=10, kernelshap=False
                )
            ],
            parameter_grid=baseline_grid
        ))

        # --- SCENARIO 2: ADAPTIVE ATTACKS ---
        for threat_model in ADAPTIVE_THREAT_MODELS_TO_TEST:
            print(f"  -- Threat Model: {threat_model}")

            for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
                print(f"    -- Adaptive Mode: {adaptive_mode}")

                if threat_model != "black_box" and adaptive_mode == "data_poisoning":
                    print(f"       Skipping data_poisoning for {threat_model} (N/A)")
                    continue

                # Determine if we want backdoors enabled or pure manipulation
                # For this experiment, we assume "pure manipulation" (disabled poison)
                # based on your request. Set to True if you want stealthy backdoors.
                USE_BACKDOOR = False

                adaptive_modifier = use_adaptive_attack(
                    mode=adaptive_mode,
                    threat_model=threat_model,
                    exploration_rounds=EXPLORATION_ROUNDS,
                    with_backdoor=USE_BACKDOOR
                )

                scenario_name = f"step7_adaptive_{threat_model}_{adaptive_mode}_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"
                unique_save_path = f"./results/{scenario_name}"

                grid = {
                    ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
                    "experiment.adv_rate": [DEFAULT_ADV_RATE],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    "experiment.save_path": [unique_save_path]
                }

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
                    modifiers=[
                        setup_modifier_func,
                        ADAPTIVE_SETUP["dataset_modifier"],
                        adaptive_modifier,  # Applies adaptive settings
                        lambda config: enable_valuation(
                            config, influence=True, loo=True, loo_freq=10, kernelshap=False
                        )
                    ],
                    parameter_grid=grid
                )
                scenarios.append(scenario)

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step7_adaptive_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_adaptive_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 7 ---")

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
