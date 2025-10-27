import sys
from typing import Callable, Dict, List, Any
import copy # Import copy

# --- Imports from your project ---
# (Ensure these paths are correct for your structure)
from entry.gradient_market.automate_exp.base_configs import get_base_image_config # Example
from entry.gradient_market.automate_exp.scenarios import (
    use_image_backdoor_attack, # Example attack
    use_cifar10_config,        # Example dataset config
    Scenario
)
from common.gradient_market_configs import AppConfig, PoisonType # Ensure needed configs imported
from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr

# --- 1. Import or Define GOLDEN_TRAINING_PARAMS ---
# --- REMEMBER TO FILL THESE IN ---
GOLDEN_TRAINING_PARAMS = {
    "image": {
        "training.optimizer": "SGD", "training.learning_rate": 0.01, "training.local_epochs": 5,
        "training.momentum": 0.9, "training.weight_decay": 5e-4,
    },
    # Add other modalities if testing scalability for them
}

# --- 2. Import or Define TUNED_DEFENSE_PARAMS ---
# --- REMEMBER TO FILL THESE IN ---
TUNED_DEFENSE_PARAMS = {
    "fedavg":    {"aggregation.method": "fedavg"},
    "fltrust":   {"aggregation.method": "fltrust", "aggregation.clip_norm": 10.0},
    "martfl":    {"aggregation.method": "martfl", "aggregation.martfl.max_k": 5, "aggregation.clip_norm": 10.0},
    "skymask":   { # Add all relevant tuned SkyMask params
        "aggregation.method": "skymask", "aggregation.skymask.mask_epochs": 20,
        "aggregation.clip_norm": 10.0,
        # ... fill other tuned SkyMask params ...
    },
}
# Define defenses to test scalability for
DEFENSES_TO_TEST = ["fedavg", "fltrust", "martfl", "skymask"]

# ==============================================================================
# --- 3. USER ACTION: Define Scalability Parameters ---
# ==============================================================================
# Sweep marketplace sizes
MARKETPLACE_SIZES = [10, 30, 50, 100] # Adjust as needed (e.g., [10, 50, 100])

# Fixed adversary rate (percentage) across all scales
FIXED_ADV_RATE = 0.3 # 30% attackers

# Fixed attack strength
FIXED_ATTACK_POISON_RATE = 0.5 # Match defense tune

NUM_SEEDS_PER_CONFIG = 3

# --- 4. Define the Representative Model/Dataset ---
# (Choose one good representative case)
SCALABILITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "cifar10",
    "model_config_suffix": "resnet18", # Use your best model here
    "dataset_modifier": use_cifar10_config,
    "attack_modifier": use_image_backdoor_attack, # Use a standard attack
    "model_config_param_key": "experiment.image_model_config_name",
}

# --- Helper to apply fixed params ---
# (Ensures Golden Training + Tuned Defense HPs + Fixed Attack Rate/Strength are set)
def create_fixed_params_modifier_scalability(
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

        # 3. Apply the fixed attack type and strength (RATE is set in grid)
        config = attack_modifier(config)
        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)
        # config.adversary_seller_config.sybil.is_sybil = True # Optional fixed Sybil

        # 4. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

        # 5. Ensure Non-IID Seller data (standard setup)
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

        return config
    return modifier

# ==============================================================================
# --- MAIN SCALABILITY CONFIG GENERATION FUNCTION ---
# ==============================================================================
def generate_scalability_fixed_rate_scenarios() -> List[Scenario]:
    """
    Generates scenarios testing TUNED defenses under FIXED attack rate/strength
    and GOLDEN training, while varying the total number of sellers (n_sellers).
    Focuses on one representative model/dataset setup.
    """
    scenarios = []
    modality = SCALABILITY_SETUP["modality_name"]
    dataset_name = SCALABILITY_SETUP["dataset_name"]
    model_config_suffix = SCALABILITY_SETUP["model_config_suffix"]
    model_config_name = f"{dataset_name}_{model_config_suffix}"

    print(f"\n--- Generating Scalability Scenarios (Fixed Rate) ---")
    print(f"Setup: {dataset_name} {model_config_suffix}, Fixed Adv Rate: {FIXED_ADV_RATE*100}%")

    for defense_name in DEFENSES_TO_TEST:
        if defense_name not in TUNED_DEFENSE_PARAMS: continue
        tuned_defense_params = TUNED_DEFENSE_PARAMS[defense_name]
        print(f"-- Defense: {defense_name}")

        # --- Create the modifier to fix Training HPs, Defense HPs, and Attack ---
        fixed_params_modifier = create_fixed_params_modifier_scalability(
            modality,
            tuned_defense_params,
            SCALABILITY_SETUP["attack_modifier"],
            model_config_name
        )

        # --- Define the parameter grid (Sweeps n_sellers, fixes adv_rate) ---
        parameter_grid = {
            SCALABILITY_SETUP["model_config_param_key"]: [model_config_name],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ADV_RATE], # Fixed Percentage
            "experiment.n_sellers": MARKETPLACE_SIZES, # Swept Parameter
            # Add other fixed experiment params if needed (e.g., global rounds)
            # "experiment.global_rounds": [100],
        }

        # Create the Scenario
        scenario_name = f"scalability_fixed_rate_{defense_name}_{dataset_name}_{model_config_suffix}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=SCALABILITY_SETUP["base_config_factory"],
            modifiers=[fixed_params_modifier, SCALABILITY_SETUP["dataset_modifier"]], # Apply all fixed settings
            parameter_grid=parameter_grid # Sweep ONLY n_sellers
        )
        scenarios.append(scenario)

    return scenarios

# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    output_dir = "./configs_generated/scalability_fixed_rate" # New directory
    generator = ExperimentGenerator(output_dir)

    scalability_scenarios = generate_scalability_fixed_rate_scenarios()

    print("\n--- Generating Scalability Configuration Files ---")
    total_configs = 0
    for scenario in scalability_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()

        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # Generate applies the modified base and sweeps over n_sellers
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All scalability (fixed rate) configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run the experiments: python run_parallel.py --configs_dir {output_dir}")
    print("2. Analyze results by plotting 'n_sellers' vs. 'test_acc'/'backdoor_asr' for each defense.")