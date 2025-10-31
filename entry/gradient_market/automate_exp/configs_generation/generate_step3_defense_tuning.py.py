# FILE: generate_step3_defense_tuning.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES  # Uses the basic one
)
from entry.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_image_backdoor_attack, use_label_flipping_attack, use_cifar100_config,
    use_trec_config, use_text_backdoor_attack
)
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL, PURCHASE100_TARGET_LABEL, PURCHASE100_TRIGGER

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
TUNING_GRIDS = {
    "fltrust": {
        "aggregation.clip_norm": [1.0, 5.0, 10.0, 20.0, None],
    },
    "martfl": {
        "aggregation.martfl.max_k": [3, 5, 7, 10],
        "aggregation.clip_norm": [5.0, 10.0, 20.0, None],
    },
    "skymask": {
        "aggregation.skymask.mask_epochs": [10, 20, 50],
        "aggregation.skymask.mask_lr": [0.01, 0.001],
        "aggregation.skymask.mask_threshold": [0.5, 0.7, 0.9],
        "aggregation.clip_norm": [10.0],  # Typically keep fixed or sweep separately
    }
}

ATTACK_TYPES_TO_TUNE = ["backdoor", "labelflip"]

# --- All Models/Datasets Combinations for Tuning ---
# Use all combinations from Step 1 to ensure tuned parameters are relevant.
TUNING_TARGETS_STEP3 = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    # ## USER ACTION ##: Add Purchase100 trigger/label if needed
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR10",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_cnn",
     "dataset_modifier": use_cifar10_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    # {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10",
    #  "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_resnet18",
    #  "dataset_modifier": use_cifar10_config, "backdoor_attack_modifier": use_image_backdoor_attack,
    #  "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    # {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar100",
    #  "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_resnet18",
    #  "dataset_modifier": use_cifar100_config, "backdoor_attack_modifier": use_image_backdoor_attack,
    #  "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config, "backdoor_attack_modifier": use_text_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
]


def generate_defense_tuning_scenarios() -> List[Scenario]:
    """Generates configs to tune defense HPs under fixed attacks."""
    print("\n--- Generating Step 3: Defense Tuning Scenarios ---")
    scenarios = []
    for target in TUNING_TARGETS_STEP3:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        for attack_type in ATTACK_TYPES_TO_TUNE:
            attack_modifier_key = f"{attack_type}_attack_modifier"
            if attack_modifier_key not in target: continue
            attack_modifier = target[attack_modifier_key]
            print(f"  -- Attack Type: {attack_type}")

            # Modifier to apply Golden HPs and Fixed Attack Settings
            def create_setup_modifier(current_modifier=attack_modifier):
                # Closure to capture the correct attack modifier
                def modifier(config: AppConfig) -> AppConfig:

                    # --- ADD THIS DEBUG BLOCK ---
                    print("\n" + "=" * 20 + " DEBUGGING " + "=" * 20)
                    print(f"Attempting to get HPs for model_cfg_name: '{model_cfg_name}'")
                    # --- END DEBUG BLOCK ---

                    # Apply Golden HPs from common utils
                    training_params = GOLDEN_TRAINING_PARAMS.get(model_cfg_name)  # <-- Your fix

                    if training_params:
                        # --- ADD THIS DEBUG BLOCK ---
                        print(f"SUCCESS: Found HPs: {training_params}")
                        print("=" * 51 + "\n")
                        # --- END DEBUG BLOCK ---
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        # --- ADD THIS DEBUG BLOCK ---
                        print(f"FAILURE: No key found for '{model_cfg_name}'!")
                        print(f"Available keys in GOLDEN_TRAINING_PARAMS are: {GOLDEN_TRAINING_PARAMS.keys()}")
                        print("=" * 51 + "\n")
                        # --- END DEBUG BLOCK ---
                        print(f"  WARNING: No Golden HPs found for model '{model_cfg_name}'!")

                    # Apply Fixed Attack Strength and Type
                    config.experiment.adv_rate = DEFAULT_ADV_RATE
                    config = current_modifier(config)  # Sets attack type (backdoor/labelflip)
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                    # Ensure Sybil is OFF for pure defense tuning (unless intended otherwise)
                    config.adversary_seller_config.sybil.is_sybil = False

                    # Ensure Non-IID data distribution
                    set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

                    # Turn off valuation for this tuning step
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # Determine relevant defenses for this modality
            current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES
            for defense_name in current_defenses:
                # Skip FedAvg as it has no parameters to tune here
                if defense_name == "fedavg": continue
                # Skip if no tuning grid defined for this defense
                if defense_name not in TUNING_GRIDS: continue

                defense_grid_to_sweep = TUNING_GRIDS[defense_name]
                print(f"    - Defense: {defense_name}")

                # Base grid fixes model/dataset/attack strength
                base_grid = {
                    target["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [target["dataset_name"]],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "aggregation.method": [defense_name],  # Fix the defense method
                    "experiment.use_early_stopping": [True],  # Use early stopping
                    "experiment.patience": [10],
                }
                # Add SkyMask model type if needed
                if defense_name == "skymask":
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    base_grid["aggregation.skymask.sm_model_type"] = [model_struct]

                # Full grid combines base with the defense HP sweep
                full_parameter_grid = {**base_grid, **defense_grid_to_sweep}

                scenarios.append(Scenario(
                    name=f"step3_tune_{defense_name}_{attack_type}_{modality}_{target['dataset_name']}_{model_cfg_name}",
                    base_config_factory=target["base_config_factory"],
                    modifiers=[setup_modifier_func, target["dataset_modifier"]],
                    parameter_grid=full_parameter_grid  # Sweep defense HPs
                ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step3_defense_tuning"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_defense_tuning_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 3 ---")
    # --- Standard Generator Loop (Sweeps Grid Internally) ---
    # The generator applies modifiers first, then expands the grid.
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden HPs, fixed attack, non-iid data)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator expands the parameter grid (the defense HPs being tuned)
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 3 (Defense Tuning) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS in config_common_utils.py is correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(
        f"3. Analyze results using analyze_results.py for each group (e.g., analyze results for '...tune_fltrust_backdoor_image_cifar10_resnet18').")
    print("4. Find the best defense HPs (good Acc, low ASR) for each defense/attack_type/model/dataset combo.")
    print(
        "5. Record these winning HPs -> TUNED_DEFENSE_PARAMS in config_common_utils.py (choose the best overall or note variations).")
