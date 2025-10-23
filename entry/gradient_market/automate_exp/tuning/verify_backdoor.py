# FILE: entry/gradient_market/automate_exp/verify_backdoor.py

import copy
import sys
from typing import Callable

from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import use_image_backdoor_attack, use_text_backdoor_attack, Scenario
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL

try:
    # Import the necessary config classes
    from common.gradient_market_configs import AppConfig, PoisonType
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr


except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your project structure allows importing these components.")
    print("Ensure base config factories have been updated with Golden Parameters.")
    sys.exit(1)

# --- Fixed Attack Settings for Verification ---
# Use the standard attack strength you plan for your main benchmark
ATTACK_ADV_RATE = 0.3
ATTACK_POISON_RATE = 1.0  # Use a high rate here to ensure the pattern works

NUM_SEEDS_PER_CONFIG = 3  # Run a few seeds to ensure consistency


# --- Function to apply Attack Settings ---
def apply_fixed_attack(config: AppConfig, attack_modifier: Callable) -> AppConfig:
    """Applies the fixed attack parameters and pattern modifier."""
    config.experiment.adv_rate = ATTACK_ADV_RATE
    config = attack_modifier(config)  # Apply pattern/type modifier
    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", ATTACK_POISON_RATE)
    # Ensure Sybil is OFF for this baseline check, unless mimicry is part of your basic backdoor
    config.adversary_seller_config.sybil.is_sybil = False
    print(f"  Applied Fixed Attack: adv_rate={ATTACK_ADV_RATE}, poison_rate={ATTACK_POISON_RATE}")
    return config


# --- Define the specific Model/Dataset combinations to verify attack on ---
# Should match the combinations you tuned in Step 1 and will use in Step 3/4
MODELS_DATASETS_TO_VERIFY = [
    # --- Tabular ---
    {
        "modality_name": "tabular",
        "base_config_factory": get_base_tabular_config,
        "dataset_name": "texas100",
        "model_structure": "mlp",
        "model_config_param_key": "experiment.tabular_model_config_name",
        "model_config_name": "mlp_texas100_baseline",
        "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
    },
    # --- Image ---
    {
        "modality_name": "image",
        "base_config_factory": get_base_image_config,
        "dataset_name": "cifar10",
        "model_structure": "resnet18",
        "model_config_param_key": "experiment.image_model_config_name",
        "model_config_name": "cifar10_resnet18",
        "attack_modifier": use_image_backdoor_attack,
        # Assumes standard backdoor pattern is configured within this modifier or base config
    },
    # --- Text ---
    {
        "modality_name": "text",
        "base_config_factory": get_base_text_config,
        "dataset_name": "trec",
        "model_structure": "textcnn",
        "model_config_param_key": "experiment.text_model_config_name",
        "model_config_name": "textcnn_trec_baseline",
        "attack_modifier": use_text_backdoor_attack,  # Assumes standard backdoor pattern is configured
    },
    # Add other combinations from your TUNING_CONFIGS in tune_baselines.py if needed
]

# --- Main Execution Block ---
if __name__ == "__main__":

    output_dir = "./configs_generated/step2_verify_backdoor"
    generator = ExperimentGenerator(output_dir)

    all_verification_scenarios = []

    print("\n--- Generating Backdoor Verification Scenarios (Step 2) ---")

    # Iterate through each model/dataset combination
    for combo_config in MODELS_DATASETS_TO_VERIFY:
        modality = combo_config["modality_name"]
        scenario_name = f"step2_verify_{modality}_{combo_config['dataset_name']}_{combo_config['model_structure']}"
        print(f"-- Defining scenario: {scenario_name}")


        # Create a modifier function that applies the attack settings
        def setup_modifier(config: AppConfig) -> AppConfig:
            # Need to use the correct attack modifier for this specific combo
            return apply_fixed_attack(config, combo_config["attack_modifier"])


        # Parameter grid is minimal, just sets fixed parameters
        parameter_grid = {
            "experiment.dataset_name": [combo_config["dataset_name"]],
            "experiment.model_structure": [combo_config["model_structure"]],
            combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
            "aggregation.method": ["fedavg"],  # <<< Use FedAvg (no defense)
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            # Ensure early stopping is ON
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
        }

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=combo_config["base_config_factory"],  # Use factory with Golden Params
            modifiers=[setup_modifier],  # Apply the fixed attack via modifier
            parameter_grid=parameter_grid  # No sweeping here
        )
        all_verification_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_verification_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()

        # Apply the attack modifier
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # Generate uses the modified base and applies the grid (which just sets fixed params here)
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All backdoor verification configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments using configs in '{output_dir}'.")
    print("2. Aggregate results.")
    print("3. Analyze results: Check if `test_asr` is high (e.g., >80-90%) for all runs.")
    print("   - If YES: Your patterns are effective! Proceed to Step 3 (Tune Defenses).")
    print("   - If NO: Adjust the backdoor pattern parameters (e.g., trigger size, content,")
    print("     poison rate) in your base configs/modifiers and RE-RUN this script and experiments.")
