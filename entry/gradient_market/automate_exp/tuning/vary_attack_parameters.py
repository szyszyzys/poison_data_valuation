import sys
from typing import Callable, Dict, List, Any

# --- Imports from your project ---
# (Ensure these paths are correct for your structure)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import (
    use_image_backdoor_attack, use_text_backdoor_attack,
    use_cifar10_config, use_cifar100_config
)
from entry.gradient_market.automate_exp.tbl_new import (
    get_base_tabular_config, use_tabular_backdoor_with_trigger,
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL,
    Scenario  # Import Scenario
)

try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your project structure allows importing these components.")
    sys.exit(1)

# ==============================================================================
# --- 1. USER ACTION: Define your "Golden" Training HPs (from Step 1 IID Tune) ---
# ==============================================================================
# Fill these with the *single best* HP set you found for each modality.
GOLDEN_TRAINING_PARAMS = {
    "tabular": {
        "training.optimizer": "Adam",
        "training.learning_rate": 0.001,  # Example: Use your tuned value
        "training.local_epochs": 5,      # Example: Use your tuned value
    },
    "image": {
        "training.optimizer": "SGD",
        "training.learning_rate": 0.01,   # Example: Use your tuned value
        "training.local_epochs": 5,       # Example: Use your tuned value
    },
    "text": {
        "training.optimizer": "Adam",
        "training.learning_rate": 0.0005, # Example: Use your tuned value
        "training.local_epochs": 2,       # Example: Use your tuned value
    }
}

# ==============================================================================
# --- 2. USER ACTION: Define your "Tuned" Defense HPs (from Step 3 Defense Tune) ---
# ==============================================================================
# Fill these with the *single best* HP set you found for each defense.
TUNED_DEFENSE_PARAMS = {
    "fedavg": {
        "aggregation.method": "fedavg",
    },
    "fltrust": {
        "aggregation.method": "fltrust",
        "aggregation.clip_norm": 10.0, # Example: Use your tuned value
    },
    "martfl": {
        "aggregation.method": "martfl",
        "aggregation.martfl.max_k": 5, # Example: Use your tuned value
        "aggregation.clip_norm": 10.0, # Example: Use your tuned value
    },
    "skymask": {
        "aggregation.method": "skymask",
        "aggregation.skymask.mask_epochs": 20, # Example: Use your tuned value
        "aggregation.skymask.mask_lr": 0.01, # Example: Use your tuned value
        "aggregation.skymask.mask_threshold": 0.7, # Example: Use your tuned value
        "aggregation.clip_norm": 10.0, # Example: Use your tuned value
    }
}

# ==============================================================================
# --- 3. USER ACTION: Define the Attack Parameters to Sweep ---
# ==============================================================================
# Here, we sweep the adversary rate and keep poison rate high,
# but you can swap this (e.g., sweep poison_rate and fix adv_rate).
ATTACK_PARAMETER_GRID = {
    "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
    "adversary_seller_config.poisoning.poison_rate": [1.0], # Fix poison rate to max
}

# --- Seeds for this run ---
NUM_SEEDS_PER_CONFIG = 3

# ==============================================================================
# --- 4. USER ACTION: Define the Representative Models to Test ---
# ==============================================================================
# (This is your subset - perfect for the paper)
MODELS_TO_TEST = [
    {
        "modality_name": "tabular",
        "base_config_factory": get_base_tabular_config,
        "dataset_name": "texas100",
        "model_config_param_key": "experiment.tabular_model_config_name",
        "model_config_name": "mlp_texas100_baseline",
        "dataset_modifier": lambda cfg: cfg,
        "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
    },
    {
        "modality_name": "image",
        "base_config_factory": get_base_image_config,
        "dataset_name": "cifar10",
        "model_config_param_key": "experiment.image_model_config_name",
        "model_config_name": "cifar10_resnet18", # Good choice: use your best model
        "dataset_modifier": use_cifar10_config,
        "attack_modifier": use_image_backdoor_attack,
    },
]

# --- Helper function to apply all fixed parameters ---
def create_fixed_params_modifier(
    modality: str,
    defense_params: Dict[str, Any],
    attack_modifier: Callable[[AppConfig], AppConfig],
    model_config_name: str # Need this for SkyMask
) -> Callable[[AppConfig], AppConfig]:
    """
    Creates a single modifier function that sets all the fixed HPs
    (training, defense) and the attack type.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(modality)
        if training_params:
            for key, value in training_params.items():
                set_nested_attr(config, key, value)

        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items():
            set_nested_attr(config, key, value)

        # 3. Apply the base attack modifier (e.g., enable backdoor)
        config = attack_modifier(config)

        # 4. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            # Infer model structure from the config name string
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

        return config

    return modifier

# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":

    output_dir = "./configs_generated/step5_attack_sensitivity"
    generator = ExperimentGenerator(output_dir)
    all_scenarios = []

    print("\n--- Generating Attack Sensitivity Scenarios (Step 5) ---")

    # --- NEW: Define Attack Types to test ---
    ATTACK_TYPES_TO_TEST = ["backdoor", "labelflip"]

    for combo_config in MODELS_TO_TEST:
        modality = combo_config["modality_name"]

        for defense_name, defense_params in TUNED_DEFENSE_PARAMS.items():

            # --- NEW: Loop over Attack Types ---
            for attack_type in ATTACK_TYPES_TO_TEST:

                # Skip incompatible combos (e.g., SkyMask for non-image)
                if modality != "image" and defense_name == "skymask":
                    print(f"Skipping SkyMask for non-image modality: {modality}")
                    continue

                print(f"-- Generating: Def={defense_name}, AttackType={attack_type}, Dataset={combo_config['dataset_name']}")

                # 1. Select the correct attack modifier for this type
                modifier_key = f"{attack_type}_attack_modifier" # e.g., "backdoor_attack_modifier"
                if modifier_key not in combo_config:
                    print(f"   WARNING: No '{modifier_key}' defined for {combo_config['dataset_name']}. Skipping.")
                    continue
                selected_attack_modifier = combo_config[modifier_key]

                # 2. Create the all-in-one modifier (passing the selected attack func)
                modifier_func = create_fixed_params_modifier(
                    modality,
                    defense_params,
                    selected_attack_modifier, # Use the modifier for this attack type
                    combo_config["model_config_name"]
                )

                # 3. Define the base parameters for this scenario (same as before)
                base_params = {
                    "experiment.dataset_name": [combo_config["dataset_name"]],
                    combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    f"data.{modality}.strategy": ["dirichlet"], # Ensure Non-IID
                    f"data.{modality}.dirichlet_alpha": [0.5],
                }

                # 4. Combine base params with the sweeping ATTACK STRENGTH grid
                full_parameter_grid = {
                    **base_params,
                    **ATTACK_PARAMETER_GRID # Sweep adv_rate / poison_rate
                }

                # 5. Create the Scenario (Update name to include attack type)
                scenario_name = f"step5_atk_sens_{defense_name}_{attack_type}_{combo_config['dataset_name']}"

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=combo_config["base_config_factory"],
                    modifiers=[modifier_func, combo_config["dataset_modifier"]], # Apply all fixed HPs
                    parameter_grid=full_parameter_grid # Sweep attack STRENGTH params
                )
                all_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()

        # The generator will apply the modifiers (to set fixed HPs)
        # and then iterate over the parameter_grid (the attack HPs),
        # creating a unique config file for each combination.
        num_generated = generator.generate(base_config, scenario)

        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All attack sensitivity configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments: python run_parallel.py --configs_dir {output_dir}")
    print("2. Analyze the results by plotting 'adv_rate' vs. 'test_acc' and 'backdoor_asr' for each defense.")