# FILE: entry/gradient_market/automate_exp/tune_baselines.py

import sys
from typing import Callable

from common.enums import PoisonType
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config, use_cifar10_config
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config

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

# We define separate learning rate grids for each optimizer
ADAM_LRS_TO_SWEEP = [0.001, 0.0005, 0.0001]
SGD_LRS_TO_SWEEP = [0.1, 0.05, 0.01]  # SGD typically needs larger LRs

OPTIMIZERS_TO_SWEEP = ["Adam", "SGD"]
LOCAL_EPOCHS_TO_SWEEP = [2, 5]

NUM_SEEDS_PER_CONFIG = 1  # Run each hyperparameter combination 3 times


# --- Function to generate tuning scenario for a given modality ---
def generate_fedavg_tuning_scenario_for_modality(
        modality_name: str,
        base_config_factory: Callable[[], AppConfig],
        dataset_name: str,
        model_structure: str,
        model_config_param_key: str,
        model_config_name: str,
        dataset_modifier: Callable[[AppConfig], AppConfig],
        data_setting: str  # <-- 1. ADD THIS NEW ARGUMENT
) -> Scenario:
    # 2. MAKE THE SCENARIO NAME UNIQUE based on data setting
    scenario_name = f"step1_tune_fedavg_{modality_name}_{dataset_name}_{model_structure}_{data_setting}"
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

        # These will be looped over in the main block
        "training.optimizer": OPTIMIZERS_TO_SWEEP,
        "training.local_epochs": LOCAL_EPOCHS_TO_SWEEP,
        # Note: learning_rate is NOT here, it's handled in the main loop

        # (Other params like buyer_strategy are the same)
        f"data.{modality_name}.buyer_strategy": ["iid"],
        f"data.{modality_name}.buyer_ratio": [0.1],
    }

    # 3. SET DATA STRATEGY BASED ON THE NEW ARGUMENT
    if data_setting == "noniid":
        parameter_grid[f"data.{modality_name}.strategy"] = ["dirichlet"]
        parameter_grid[f"data.{modality_name}.dirichlet_alpha"] = [0.5]
    elif data_setting == "iid":
        parameter_grid[f"data.{modality_name}.strategy"] = ["iid"]
        # No dirichlet_alpha needed
    else:
        raise ValueError(f"Unknown data_setting: '{data_setting}'. Must be 'iid' or 'noniid'.")

    # Add modality-specific fixed parameters (e.g., dataset_type)
    parameter_grid[f"experiment.dataset_type"] = [modality_name]

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
            "model_config_name": "mlp_texas100_baseline",
            "dataset_modifier": lambda cfg: cfg,
        },
        {
            "modality_name": "tabular",
            "base_config_factory": get_base_tabular_config,
            "dataset_name": "Purchase100",
            "model_structure": "mlp",
            "model_config_param_key": "experiment.tabular_model_config_name",
            "model_config_name": "mlp_purchase100_baseline",
            "dataset_modifier": lambda cfg: cfg,
        },

        # --- Image Tuning ---
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR10",  # <-- lowercase
            "model_structure": "flexiblecnn",  # <-- (Optional clarity fix)
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "cifar10_cnn",  # <-- lowercase
            "dataset_modifier": use_cifar10_config,
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR10",  # <-- lowercase
            "model_structure": "resnet18",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "cifar10_resnet18",  # <-- lowercase
            "dataset_modifier": use_cifar10_config,
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR100",  # <-- lowercase
            "model_structure": "flexiblecnn",  # <-- (Optional clarity fix)
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "cifar100_cnn",  # <-- lowercase
            "dataset_modifier": use_cifar100_config,
        },
        {
            "modality_name": "image",
            "base_config_factory": get_base_image_config,
            "dataset_name": "CIFAR100",  # <-- lowercase
            "model_structure": "resnet18",
            "model_config_param_key": "experiment.image_model_config_name",
            "model_config_name": "cifar100_resnet18",  # <-- lowercase
            "dataset_modifier": use_cifar100_config,
        },
        # --- Text Tuning ---
        {
            "modality_name": "text",
            "base_config_factory": get_base_text_config,
            "dataset_name": "TREC",
            "model_structure": "textcnn",
            "model_config_param_key": "experiment.text_model_config_name",
            "model_config_name": "textcnn_trec_baseline",
            "dataset_modifier": lambda cfg: cfg,
        },
    ]

    # --- Output Directory for Configs ---
    output_dir = "./configs_generated/step1_fedavg_tuning_nolocalclip"
    generator = ExperimentGenerator(output_dir)

    all_tuning_scenarios = []

    # --- Generate Scenarios for Each Modality ---
    print("\n--- Generating Tuning Scenarios (Step 1: Benign FedAvg Baseline) ---")
    for config_params in TUNING_CONFIGS:
        # For each model, create an IID and a Non-IID scenario
        for data_setting in ["iid", "noniid"]:
            # Create a fresh copy of the parameters
            scenario_params = config_params.copy()

            # Add the new data_setting argument
            scenario_params["data_setting"] = data_setting

            # Call your modified function
            scenario = generate_fedavg_tuning_scenario_for_modality(**scenario_params)
            all_tuning_scenarios.append(scenario)
    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0

    # --- START OF CRITICAL FIX ---
    # We must manually loop over the optimizers to assign the correct LR grid

    for scenario in all_tuning_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")

        # Get the lists of parameters to sweep from the grid
        optimizers_to_sweep = scenario.parameter_grid.get("training.optimizer", ["Adam"])
        epochs_to_sweep = scenario.parameter_grid.get("training.local_epochs", [1])

        # Get all the *other* static parameters from the grid
        static_grid_params = {
            key: value for key, value in scenario.parameter_grid.items()
            if
            key not in ["training.optimizer", "training.learning_rate", "training.local_epochs", "experiment.save_path"]
        }

        num_generated_for_scenario = 0

        # Loop through every combination
        for optimizer in optimizers_to_sweep:

            # Select the correct LR grid based on the optimizer
            if optimizer == "Adam":
                lrs_to_sweep = ADAM_LRS_TO_SWEEP
            else:  # Assumes "SGD"
                lrs_to_sweep = SGD_LRS_TO_SWEEP

            for lr in lrs_to_sweep:
                for epochs in epochs_to_sweep:

                    # 1. Create a new grid for *only this combination*
                    new_grid = static_grid_params.copy()
                    new_grid["training.optimizer"] = [optimizer]
                    new_grid["training.learning_rate"] = [lr]
                    new_grid["training.local_epochs"] = [epochs]

                    # 2. CREATE THE UNIQUE SAVE PATH
                    # This is the path your *results* will be saved to.
                    unique_save_path = f"./new_results_nolocalclip/{scenario.name}/opt_{optimizer}_lr_{lr}_epochs_{epochs}"

                    # 3. Add this unique path to the new grid
                    new_grid["experiment.save_path"] = [unique_save_path]

                    # 4. Create a temporary Scenario object for this single run
                    #    The name determines the config *filename*
                    temp_scenario_name = f"{scenario.name}/opt_{optimizer}_lr_{lr}_epochs_{epochs}"
                    temp_scenario = Scenario(
                        name=temp_scenario_name,
                        base_config_factory=scenario.base_config_factory,
                        modifiers=scenario.modifiers,
                        parameter_grid=new_grid
                    )

                    # 5. Get the base config and apply modifiers
                    base_config = temp_scenario.base_config_factory()
                    for modifier in temp_scenario.modifiers:
                        base_config = modifier(base_config)

                    # 6. Generate the single config file for this combo
                    num_generated = generator.generate(base_config, temp_scenario)
                    num_generated_for_scenario += num_generated

        total_configs += num_generated_for_scenario
        print(f"  Generated {num_generated_for_scenario} config files for this scenario.")

    # --- END OF CRITICAL FIX ---

    print(f"\n✅ All tuning configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments using the configs in '{output_dir}'.")
    print(f"2. Analyze the results, which will now be in unique folders under './results/...'")
