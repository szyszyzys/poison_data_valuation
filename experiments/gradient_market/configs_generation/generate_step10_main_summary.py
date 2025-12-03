# FILE: generate_step12_main_summary.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    # create_fixed_params_modifier,  <-- REMOVED
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS  # <-- ADDED
)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_cifar100_config, use_trec_config, use_image_backdoor_attack,
    use_text_backdoor_attack
)
from experiments.gradient_market.automate_exp import TEXAS100_TRIGGER, use_tabular_backdoor_with_trigger, \
    TEXAS100_TARGET_LABEL, get_base_tabular_config, PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE
MAIN_SUMMARY_TARGETS = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL)},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL)},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR10",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_cnn",
     "dataset_modifier": use_cifar10_config, "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config, "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config, "attack_modifier": use_text_backdoor_attack},
]


# === THIS IS THE CORRECTED FUNCTION ===
def generate_main_summary_scenarios() -> List[Scenario]:
    """Generates the main benchmark comparison configs with valuation."""
    print("\n--- Generating Step 12: Main Summary Scenarios (with Valuation) ---")
    scenarios = []
    for target in MAIN_SUMMARY_TARGETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES

        for defense_name in current_defenses:

            # Get Tuned HPs (from Step 3)
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",  # Use default
                default_attack_type_for_tuning="backdoor"
            )
            print(f"  - Defense: {defense_name}")
            # This is the correct check:
            if not tuned_defense_params:
                print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
                continue

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
                    if "skymask" in current_defense_name:
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # 2. Apply Tuned Defense HPs (from Step 3)
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                    # 3. Apply other fixed settings
                    set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            valuation_modifier = lambda config: enable_valuation(
                config,
                influence=True,
                loo=True, loo_freq=5,
                kernelshap=True, kshap_freq=5,
                kshap_samples=500
            )

            scenario_name = f"step12_main_summary_{defense_name}_{modality}_{target['dataset_name']}_{model_cfg_name.split('_')[-1]}"
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                target["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [target["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
                "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path]  # <-- ADDED
            }

            all_modifiers = [
                setup_modifier_func,
                target["dataset_modifier"],
                target["attack_modifier"],  # <-- THE FIX
                valuation_modifier
            ]
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=target["base_config_factory"],
                modifiers=all_modifiers,
                parameter_grid=grid
            )
            scenarios.append(scenario)
    return scenarios


# --- Main Execution Block (This is now correct) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step12_main_summary"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_main_summary_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 12 ---")

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

    print(f"\n✅ Step 12 (Main Summary) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results: Create summary tables/plots.")
    print(f"4. Analyze valuation results: Compare Influence/LOO/KernelSHAP scores.")
