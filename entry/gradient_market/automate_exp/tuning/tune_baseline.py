# FILE: entry/gradient_market/automate_exp/tune_baselines.py

import sys
from typing import Callable

from common.enums import PoisonType
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config, use_cifar10_config
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config

# --- Assuming your config generator and base factories are importable ---
# Adjust these import paths based on your project structure
try:
    # Import the necessary config classes
    from common.gradient_market_configs import AppConfig, ExperimentConfig, TrainingConfig, \
        DataConfig, AggregationConfig, ServerAttackConfig, AdversarySellerConfig, \
        DebugConfig, BuyerAttackConfig

    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your project structure allows importing these components.")
    print("Verify base config factories (tabular, image, text) exist and use SGD.")
    sys.exit(1)

# --- Parameters to Sweep ---
# These are the learning rates and local epochs we want to test
LEARNING_RATES_TO_SWEEP = [0.05, 0.01, 0.005, 0.001]
LOCAL_EPOCHS_TO_SWEEP = [1, 2, 5]
NUM_SEEDS_PER_CONFIG = 3  # Run each hyperparameter combination 3 times


# --- Function to generate tuning scenario for a given modality ---
def generate_fedavg_tuning_scenario_for_modality(
        modality_name: str,
        base_config_factory: Callable[[], AppConfig],
        dataset_name: str,
        model_structure: str,
        model_config_param_key: str,
        model_config_name: str,
        dataset_modifier: Callable[[AppConfig], AppConfig]
) -> Scenario:
    """Creates the tuning scenario object for a specific modality."""
    scenario_name = f"step1_tune_fedavg_{modality_name}_{dataset_name}_{model_structure}"
    print(f"-- Defining scenario: {scenario_name}")

    parameter_grid = {
        "experiment.dataset_name": [dataset_name],
        "experiment.model_structure": [model_structure],
        model_config_param_key: [model_config_name],
        "aggregation.method": ["fedavg"],
        "experiment.adv_rate": [0.0],
        "n_samples": [NUM_SEEDS_PER_CONFIG],
        "experiment.use_early_stopping": [True],
        "experiment.patience": [10],

        "training.learning_rate": LEARNING_RATES_TO_SWEEP,
        "training.local_epochs": LOCAL_EPOCHS_TO_SWEEP,

        f"data.{modality_name}.strategy": ["dirichlet"],
        f"data.{modality_name}.dirichlet_alpha": [0.5],  # Example alpha, adjust if needed

        f"data.{modality_name}.buyer_strategy": ["iid"],
        f"data.{modality_name}.buyer_ratio": [0.1],  # Or your standard buyer ratio
    }

    # Add modality-specific fixed parameters (e.g., dataset_type)
    parameter_grid[f"experiment.dataset_type"] = [modality_name]  # Simplified

    def ensure_benign(config: AppConfig) -> AppConfig:
        config.experiment.adv_rate = 0.0
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        config.buyer_attack_config.is_active = False
        return config

    return Scenario(
        name=scenario_name,
        base_config_factory=base_config_factory,
        modifiers=[ensure_benign, dataset_modifier],
        parameter_grid=parameter_grid
    )


# --- Main Execution Block ---
if __name__ == "__main__":

    TUNING_CONFIGS = [

        # --- Tabular Tuning ---
        {
            "modality_name": "tabular",
            "base_config_factory": get_base_tabular_config,
            "dataset_name": "Texas100",
            "model_structure": "mlp",
            "model_config_param_key": "experiment.tabular_model_config_name",
            "model_config_name": "mlp_texas100_baseline",  # Assumed name
            "dataset_modifier": lambda cfg: cfg,  # No specific modifier needed if base is Texas100
        },
        {
            "modality_name": "tabular",
            "base_config_factory": get_base_tabular_config,
            "dataset_name": "Purchase100",
            "model_structure": "mlp",  # Assuming MLP, as ResNet is unusual for tabular. Adjust if needed.
            "model_config_param_key": "experiment.tabular_model_config_name",
            "model_config_name": "mlp_purchase100_baseline",  # Assumed name
            "dataset_modifier": lambda cfg: cfg,  # No specific modifier needed if base is Texas100
        },

        # --- Image Tuning ---
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR10",
            "model_structure": "cnn",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "CIFAR10_cnn",
            "dataset_modifier": use_cifar10_config,  # Pass the function itself
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR10",
            "model_structure": "resnet18",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "CIFAR10_resnet18",
            "dataset_modifier": use_cifar10_config,  # Pass the function itself
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR100",
            "model_structure": "cnn",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "CIFAR100_cnn",
            "dataset_modifier": use_cifar100_config,  # Pass the function itself
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR100",
            "model_structure": "resnet18",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "CIFAR100_resnet18",
            "dataset_modifier": use_cifar100_config,  # Pass the function itself
        },

        # --- Text Tuning ---
        {
            "modality_name": "text",
            "base_config_factory": get_base_text_config,
            "dataset_name": "TREC",
            "model_structure": "textcnn",  # Assuming 'text_cnn' is the structure name
            "model_config_param_key": "experiment.text_model_config_name",  # Assuming this key name
            "model_config_name": "textcnn_trec_baseline",  # Assumed name
            "dataset_modifier": lambda cfg: cfg,  # No specific modifier needed if base is TRE
        },
    ]
    # --- Output Directory for Configs ---
    output_dir = "./configs_generated/step1_fedavg_tuning"
    generator = ExperimentGenerator(output_dir)

    all_tuning_scenarios = []

    # --- Generate Scenarios for Each Modality ---
    print("\n--- Generating Tuning Scenarios (Step 1: Benign FedAvg Baseline) ---")
    for config in TUNING_CONFIGS:
        scenario = generate_fedavg_tuning_scenario_for_modality(**config)
        all_tuning_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_tuning_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        # Apply the ensure_benign modifier
        for modifier in scenario.modifiers:
            base_config = modifier(base_config)

        # The generate method handles the parameter grid and saves files
        num_generated = generator.generate(base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All tuning configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments using the configs in '{output_dir}'. Use your existing experiment runner.")
    print("2. Aggregate results using your aggregation script (e.g., aggregate_experiment_results.py).")
    print("3. Analyze the aggregated CSV and individual training_log.csv files to find the")
    print("   'Golden Training Parameters' (best stable lr and local_epochs) for EACH modality.")
    print("4. Hard-code these golden parameters into your base config factories for future steps.")
