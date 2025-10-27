import sys
from typing import Callable, Dict, List, Any
import copy # Import copy

# --- Imports from your project ---
# (Ensure these paths are correct for your structure)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from entry.gradient_market.automate_exp.scenarios import (
    use_image_backdoor_attack, use_text_backdoor_attack,
    use_cifar10_config, use_cifar100_config,
    Scenario # Import Scenario
)
from entry.gradient_market.automate_exp.tbl_new import (
    get_base_tabular_config, use_tabular_backdoor_with_trigger,
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL,
)

try:
    from common.gradient_market_configs import AppConfig
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)


# --- 1. Import or Define GOLDEN_TRAINING_PARAMS ---
# (Must contain the best optimizer, LR, epochs per modality)
GOLDEN_TRAINING_PARAMS = {
    "image": {
        "training.optimizer": "SGD",
        "training.learning_rate": 0.01,
        "training.local_epochs": 5,
        "training.momentum": 0.9,
        "training.weight_decay": 5e-4,
    },
    # Add tabular and text if needed
}

# --- 2. Import or Define TUNED_DEFENSE_PARAMS ---
# (Focus on defenses potentially using buyer data)
TUNED_DEFENSE_PARAMS = {
    # FedAvg is a baseline, doesn't use buyer data but good for comparison
    "fedavg":    {"aggregation.method": "fedavg"},
    # FLTrust explicitly uses buyer data as root dataset
    "fltrust":   {"aggregation.method": "fltrust", "aggregation.clip_norm": 10.0},
    # MartFL *might* use buyer data if initial_baseline='buyer'
    "martfl":    {
        "aggregation.method": "martfl",
        "aggregation.martfl.max_k": 5,
        "aggregation.clip_norm": 10.0,
        # "aggregation.martfl.initial_baseline": "buyer", # Add if testing this variant
    },
    # Include others if relevant
}

# --- 3. Fixed Attack Settings ---
FIXED_ATTACK_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5 # Match defense tune

# ==============================================================================
# --- 4. USER ACTION: Define the Buyer Data Parameters to Sweep ---
# ==============================================================================
BUYER_PARAMETER_GRID = {
    # Sweep the fraction of total data held by the buyer
    "data.{modality}.buyer_ratio": [0.01, 0.05, 0.1, 0.2],
    # Test both IID and Non-IID buyer data (if relevant)
    "data.{modality}.buyer_strategy": ["iid", "dirichlet"],
    # If using 'dirichlet' for buyer, set alpha (e.g., 0.5 for skewed)
    "data.{modality}.buyer_dirichlet_alpha": [0.5],
}

# --- Seeds for this run ---
NUM_SEEDS_PER_CONFIG = 3

# --- 5. Define the Representative Models to Test ---
# (Keep this focused, e.g., one model per modality)
MODELS_TO_TEST = [
    {
        "modality_name": "image",
        "base_config_factory": get_base_image_config,
        "dataset_name": "cifar10",
        "model_config_param_key": "experiment.image_model_config_name",
        "model_config_name": "cifar10_resnet18",
        "dataset_modifier": use_cifar10_config,
        "attack_modifier": use_image_backdoor_attack,
    },
    # Add tabular/text if desired
]

# --- Helper to apply fixed params ---
# (Similar to step5, ensures Golden Training + Tuned Defense + Fixed Attack)
def create_fixed_params_modifier_buyer(
    modality: str,
    defense_params: Dict[str, Any],
    attack_modifier: Callable[[AppConfig], AppConfig],
    model_config_name: str
) -> Callable[[AppConfig], AppConfig]:

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(modality)
        if training_params:
            for key, value in training_params.items():
                set_nested_attr(config, key, value)

        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items():
            set_nested_attr(config, key, value)

        # 3. Apply the fixed attack type and strength
        config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
        config = attack_modifier(config)
        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)
        # config.adversary_seller_config.sybil.is_sybil = True # Optional

        # 4. Set SkyMask model type if needed (though SkyMask less relevant here)
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

        # 5. Ensure Seller data remains Non-IID
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

        return config
    return modifier


# ==============================================================================
# --- 6. MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":

    output_dir = "./configs_generated/step6_buyer_data_impact"
    generator = ExperimentGenerator(output_dir)

    all_scenarios = []

    print("\n--- Generating Buyer Data Impact Scenarios (Step 6) ---")

    for combo_config in MODELS_TO_TEST:
        modality = combo_config["modality_name"]

        # Only test relevant defenses (or all for comparison)
        for defense_name, defense_params in TUNED_DEFENSE_PARAMS.items():

            print(f"-- Generating: Def={defense_name}, Dataset={combo_config['dataset_name']}")

            # 1. Create the modifier to fix Training, Defense, Attack, Seller Data
            fixed_params_modifier = create_fixed_params_modifier_buyer(
                modality,
                defense_params,
                combo_config["attack_modifier"],
                combo_config["model_config_name"]
            )

            # 2. Define the base parameters (fixed for this group)
            base_params = {
                "experiment.dataset_name": [combo_config["dataset_name"]],
                combo_config["model_config_param_key"]: [combo_config["model_config_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                # Seller data is fixed to Non-IID by the modifier
            }

            # 3. Dynamically create the BUYER sweep grid for this modality
            buyer_sweep_grid = {}
            for key_template, values in BUYER_PARAMETER_GRID.items():
                key = key_template.format(modality=modality)
                buyer_sweep_grid[key] = values

            # --- Handle conditional alpha for buyer ---
            # Remove buyer alpha if buyer strategy doesn't include 'dirichlet'
            if "dirichlet" not in buyer_sweep_grid.get(f"data.{modality}.buyer_strategy", []):
                 alpha_key = f"data.{modality}.buyer_dirichlet_alpha"
                 if alpha_key in buyer_sweep_grid:
                     del buyer_sweep_grid[alpha_key]


            # 4. Combine base params with the BUYER sweep grid
            full_parameter_grid = {
                **base_params,
                **buyer_sweep_grid
            }

            # 5. Create the Scenario
            scenario_name = f"step6_buyer_impact_{defense_name}_{combo_config['dataset_name']}"

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=combo_config["base_config_factory"],
                modifiers=[fixed_params_modifier, combo_config["dataset_modifier"]], # Apply fixed settings
                parameter_grid=full_parameter_grid # Sweep ONLY buyer params
            )
            all_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()

        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All buyer data impact configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments: python run_parallel.py --configs_dir {output_dir}")
    print("2. Analyze the results by plotting 'buyer_ratio' vs. 'test_acc'/'backdoor_asr'")
    print("   (potentially separate plots for 'buyer_strategy=iid' vs 'buyer_strategy=dirichlet').")