# FILE: entry/gradient_market/automate_exp/tune_defenses.py

import copy
import sys
from typing import Callable

from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import use_image_backdoor_attack, use_text_backdoor_attack
from entry.gradient_market.automate_exp.tbl_new import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL, Scenario

try:
    # Import the necessary config classes
    from common.gradient_market_configs import AppConfig, PoisonType  # Import PoisonType too
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your project structure allows importing these components.")
    sys.exit(1)

# --- !!! IMPORTANT: SET YOUR GOLDEN PARAMETERS HERE !!! ---
# Replace these placeholders with the values found in Step 1
GOLDEN_PARAMS_PER_MODALITY = {
    "tabular": {
        "learning_rate": 0.005,  # Example value
        "local_epochs": 1,  # Example value
    },
    "image": {
        "learning_rate": 0.01,  # Example value
        "local_epochs": 2,  # Example value
    },
    "text": {
        "learning_rate": 0.005,  # Example value
        "local_epochs": 1,  # Example value
    }
}

# --- Fixed Attack Settings for Tuning ---
# Choose ONE representative attack scenario for tuning the defenses
# This should be strong enough to challenge the defenses.
FIXED_ATTACK_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5  # Example: Tune against a 50% poison rate

NUM_SEEDS_PER_CONFIG = 3  # Number of seeds for tuning runs

# --- Defense Hyperparameter Grids ---
TUNING_GRIDS = {
    "fltrust": {
        "aggregation.method": ["fltrust"],
        # Tune clip_norm (enables server-side clipping)
        "aggregation.clip_norm": [1.0, 5.0, 10.0, 20.0],
        # Optionally test without clipping by adding None or 0.0
        # "aggregation.clip_norm": [None, 1.0, 5.0, 10.0],
    },
    "martfl": {
        "aggregation.method": ["martfl"],
        # Core MartFL parameters to tune
        "aggregation.martfl.change_base": [True, False],
        "aggregation.martfl.clip": [True],  # Keep server-side clipping ON
        "aggregation.clip_norm": [5.0, 10.0],  # Tune the clip value used when clip=True
        # Optionally add initial_baseline and max_k if you implemented those
        # "aggregation.martfl.initial_baseline": ["buyer", "bn_0"],
        # "aggregation.martfl.max_k": [5, 10],
    },
    "skymask": {
        "aggregation.method": ["skymask"],
        "aggregation.skymask.clip": [True],  # Keep server-side clipping ON
        "aggregation.clip_norm": [10.0],  # Keep fixed initially, maybe tune later
        # Core SkyMask parameters
        "aggregation.skymask.mask_epochs": [10, 20, 50],
        "aggregation.skymask.mask_lr": [0.01, 0.001],
        "aggregation.skymask.mask_threshold": [0.5, 0.7],
        # "aggregation.skymask.mask_clip": [1.0], # Keep fixed initially
    }
    # Add grids for other defenses if needed
}


# --- Function to apply Golden Parameters and Attack ---
def apply_tuning_setup(config: AppConfig, modality: str, attack_modifier: Callable) -> AppConfig:
    """Applies golden training params and the fixed attack."""
    # 1. Apply Golden Training Parameters
    if modality in GOLDEN_PARAMS_PER_MODALITY:
        params = GOLDEN_PARAMS_PER_MODALITY[modality]
        config.training.learning_rate = params["learning_rate"]
        config.training.local_epochs = params["local_epochs"]
        print(f"  Applied Golden Params for {modality}: lr={params['learning_rate']}, E={params['local_epochs']}")
    else:
        print(f"  Warning: No Golden Params defined for modality '{modality}', using defaults.")

    # 2. Apply Attack Settings (Fixed for Tuning)
    config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
    # Apply the specific poisoning setup via the modifier
    config = attack_modifier(config)
    # Ensure the poison rate is set correctly within the poisoning config
    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)
    # Ensure Sybil is ON if needed for your standard attack
    # config.adversary_seller_config.sybil.is_sybil = True
    # config.adversary_seller_config.sybil.gradient_default_mode = "mimic"

    print(f"  Applied Fixed Attack: adv_rate={FIXED_ATTACK_ADV_RATE}, poison_rate={FIXED_ATTACK_POISON_RATE}")
    return config


# --- Define the specific Model/Dataset combinations to tune defenses on ---
# Use the same list as in tune_baselines.py or a subset if tuning is too long
MODELS_DATASETS_TO_TUNE = [
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
        "attack_modifier": use_image_backdoor_attack,  # Assumes standard backdoor
    },
    # --- Text ---
    {
        "modality_name": "text",
        "base_config_factory": get_base_text_config,
        "dataset_name": "trec",
        "model_structure": "textcnn",
        "model_config_param_key": "experiment.text_model_config_name",
        "model_config_name": "textcnn_trec_baseline",
        "attack_modifier": use_text_backdoor_attack,  # Assumes standard backdoor
    },
    # Add other combinations if necessary
]

# --- Main Execution Block ---
if __name__ == "__main__":

    output_dir = "./configs_generated/step3_defense_tuning"
    generator = ExperimentGenerator(output_dir)

    all_defense_tuning_scenarios = []

    print("\n--- Generating Defense Tuning Scenarios (Step 3) ---")

    # Iterate through each model/dataset combination
    for combo_config in MODELS_DATASETS_TO_TUNE:
        modality = combo_config["modality_name"]
        print(
            f"\n-- Processing Modality: {modality}, Dataset: {combo_config['dataset_name']}, Model: {combo_config['model_structure']}")

        # Iterate through each defense method to tune
        for defense_name, defense_grid in TUNING_GRIDS.items():

            # Skip SkyMask for text data if it's not compatible
            if modality != "image" and defense_name == "skymask":
                print(f"   Skipping {defense_name} tuning for text modality.")
                continue

            scenario_name = f"step3_tune_{defense_name}_{modality}_{combo_config['dataset_name']}"
            print(f"  - Defining scenario for {defense_name}: {scenario_name}")

            # Create the base config using the factory
            base_cfg_instance = combo_config["base_config_factory"]()


            # Create a modifier function that applies BOTH golden params and attack
            def setup_modifier(config: AppConfig) -> AppConfig:
                # Need to use the correct modality and attack modifier for this specific combo
                return apply_tuning_setup(config, combo_config["modality_name"], combo_config["attack_modifier"])


            # Combine the base fixed parameters with the defense-specific grid
            full_parameter_grid = {
                # Fixed parameters for this model/dataset
                "experiment.dataset_name": [combo_config["dataset_name"]],
                "experiment.model_structure": [combo_config["model_structure"]],
                combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "aggregation.skymask.sm_model_type": [
                    combo_config["model_structure"]] if defense_name == "skymask" else [None],

                # The defense method and its specific tuning grid
                **defense_grid
            }

            # Need to remove None values if a defense doesn't use a specific param
            # For example, Skymask grid sets its own sm_model_type, others don't need it.
            if defense_name != "skymask":
                if "aggregation.skymask.sm_model_type" in full_parameter_grid:
                    del full_parameter_grid["aggregation.skymask.sm_model_type"]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=combo_config["base_config_factory"],  # Use original factory
                modifiers=[setup_modifier],  # Apply golden params + attack via modifier
                parameter_grid=full_parameter_grid
            )
            all_defense_tuning_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_defense_tuning_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()

        # IMPORTANT: Apply modifiers BEFORE generating to set golden params/attack
        modified_base_config = copy.deepcopy(base_config)  # Avoid modifying original base
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # Generate uses the modified base and applies the grid
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All defense tuning configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments using the configs in '{output_dir}'.")
    print("2. Aggregate the results.")
    print("3. Analyze results for EACH defense to find the best hyperparameters")
    print("   (lowest ASR with good accuracy) under the fixed attack.")
    print("4. Hard-code these 'Tuned Defense Parameters' into your main benchmark scenarios (Step 4).")
