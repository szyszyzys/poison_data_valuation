# FILE: generate_step12_main_summary.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Common Utils (Update path if needed)
from config_common_utils import (
    TUNED_DEFENSE_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,  # Standard attack strength
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES, create_fixed_params_modifier,  # Use the standard helper
    enable_valuation, use_sybil_attack_strategy  # Use the valuation helper
)
# Base Configs & Modifiers (Update path if needed)
from entry.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_cifar100_config, use_trec_config, use_image_backdoor_attack,
    use_text_backdoor_attack
)
from entry.gradient_market.automate_exp.tbl_new import TEXAS100_TRIGGER, use_tabular_backdoor_with_trigger, \
    TEXAS100_TARGET_LABEL, get_base_tabular_config

# Import needed attack modifiers
# ## USER ACTION ##: Ensure these import paths are correct and select standard attack

try:
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose
# Generate the core benchmark results comparing all TUNED defenses across multiple
# datasets and models under a standard, FIXED attack scenario (e.g., backdoor + Sybil mimic).
# This experiment uses GOLDEN training parameters and enables comprehensive VALUATION metrics
# (Influence, LOO, KernelSHAP) for analyzing fairness and economic aspects.
# This often forms the main results figure/table of the paper. 📊🏆

# --- Standard Attack Configuration ---
# ## USER ACTION ##: Define the standard attack for the main comparison
STANDARD_SYBIL_STRATEGY = "mimic"  # Example: Use mimic Sybil coordination
STANDARD_ATTACK_MODIFIERS = {
    "image": [use_image_backdoor_attack, use_sybil_attack_strategy(STANDARD_SYBIL_STRATEGY)],
    "tabular": [use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
                use_sybil_attack_strategy(STANDARD_SYBIL_STRATEGY)],  # Add specific trigger
    "text": [use_text_backdoor_attack, use_sybil_attack_strategy(STANDARD_SYBIL_STRATEGY)],
}
# Fixed attack strength for the main summary
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE

# --- All Models/Datasets Combinations for Main Summary ---
# Use all combinations from Step 1
MAIN_SUMMARY_TARGETS = [
    # Tabular
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg},
    # {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "purchase100", "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline", "dataset_modifier": lambda cfg: cfg}, # Add if needed
    # Image
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_cnn",
     "dataset_modifier": use_cifar10_config},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar10",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_resnet18",
     "dataset_modifier": use_cifar10_config},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "cifar100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_resnet18",
     "dataset_modifier": use_cifar100_config},
    # Text
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "trec",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config},  # Use trec modifier
]


def generate_main_summary_scenarios() -> List[Scenario]:
    """Generates the main benchmark comparison configs with valuation."""
    print("\n--- Generating Step 12: Main Summary Scenarios (with Valuation) ---")
    scenarios = []
    for target in MAIN_SUMMARY_TARGETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        # Determine relevant defenses and attack modifiers for this modality
        current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES
        attack_modifiers = STANDARD_ATTACK_MODIFIERS.get(modality, [])
        if not attack_modifiers:
            print(f"   WARNING: No standard attack modifiers defined for modality '{modality}'. Skipping attack.")

        for defense_name in current_defenses:
            if defense_name not in TUNED_DEFENSE_PARAMS:
                print(f"  Skipping {defense_name}: No tuned parameters found.")
                continue
            tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
            print(f"  - Defense: {defense_name}")

            # --- Create the modifier for Golden Training + Tuned Defense HPs ---
            fixed_params_modifier = create_fixed_params_modifier(
                modality, tuned_defense_params, model_cfg_name, apply_noniid=True
            )

            # --- Create modifier to enable comprehensive valuation ---
            # ## USER ACTION ##: Adjust frequencies/methods based on computational budget
            valuation_modifier = lambda config: enable_valuation(
                config,
                influence=True,  # Fast, run every round
                loo=True, loo_freq=10,  # Slower, run periodically
                kernelshap=True, kshap_freq=20,  # Slowest, run periodically
                kshap_samples=500  # Number of samples for KernelSHAP approximation
            )

            # --- Define the grid (fixed parameters for this run) ---
            grid = {
                target["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [target["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
                "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                # Sybil settings are applied by the attack_modifier if included
            }

            # --- Create the Scenario ---
            # Combine all necessary modifiers
            all_modifiers = [fixed_params_modifier, target["dataset_modifier"]] + attack_modifiers + [
                valuation_modifier]

            scenario_name = f"step12_main_summary_{defense_name}_{modality}_{target['dataset_name']}_{model_cfg_name.split('_')[-1]}"  # Consistent naming

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=target["base_config_factory"],
                modifiers=all_modifiers,
                parameter_grid=grid  # Grid only contains fixed exp params
            )
            scenarios.append(scenario)
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step12_main_summary"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_main_summary_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 12 ---")
    # --- Standard Generator Loop ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        # Apply modifiers (sets golden train, tuned defense, standard attack, valuation)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)
        # Generator just applies the fixed grid parameters
        num_gen = generator.generate(modified_base_config, scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 12 (Main Summary) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results: Create summary tables/plots comparing Acc/ASR across all defenses/datasets/models.")
    print(
        f"4. Analyze valuation results: Compare Influence/LOO/KernelSHAP scores for benign vs. malicious sellers across defenses.")
    print(f"   -> Assess fairness and potential for value inflation.")
