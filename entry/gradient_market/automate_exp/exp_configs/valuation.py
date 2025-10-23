# FILE: entry/gradient_market/automate_exp/generate_valuation_scenarios.py (New File)

import sys
from typing import List

# --- Import your necessary components ---
from entry.gradient_market.automate_exp.base_configs import get_base_image_config
from entry.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar10_config, use_image_backdoor_attack, use_sybil_attack
)

try:
    from common.gradient_market_configs import AppConfig  # Ensure AppConfig is importable
    from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Define the single combination to focus on ---
FOCUSED_DATASET = "cifar10"
FOCUSED_MODEL = "resnet18"
FOCUSED_MODIFIER = use_cifar10_config
FOCUSED_BASE_FACTORY = get_base_image_config
FOCUSED_SM_MODEL_TYPE = "resnet18"  # SkyMask needs this

# --- Define defenses to test ---
# Include 'fedavg' as a baseline (no defense)
AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl', 'skymask']

# --- Fixed Attack Settings ---
# Use a moderate attack, suitable for benchmarking defenses
FIXED_ATTACK_PARAMS = {
    "experiment.adv_rate": [0.3],  # 30% adversaries
    "adversary_seller_config.poisoning.poison_rate": [0.5],  # 50% poison rate
}

# --- Valuation Settings (Enable Everything) ---
VALUATION_PARAMS = {
    "valuation.run_similarity": [True],  # Default, always on
    "valuation.run_influence": [True],  # Enable Influence Function
    "valuation.run_loo": [True],  # Enable Leave-One-Out
    "valuation.loo_frequency": [10],  # Run LOO every 10 rounds
    "valuation.run_kernelshap": [True],  # Enable KernelSHAP
    "valuation.kernelshap_frequency": [20],  # Run KernelSHAP every 20 rounds
    "valuation.kernelshap_samples": [32]  # Sample size for KernelSHAP
}

# --- Shared Parameters ---
SHARED_PARAMS = {
    "experiment.dataset_name": [FOCUSED_DATASET],
    "experiment.model_structure": [FOCUSED_MODEL],
    "experiment.image_model_config_name": [f"{FOCUSED_DATASET}_{FOCUSED_MODEL}"],
    "aggregation.skymask.sm_model_type": [FOCUSED_SM_MODEL_TYPE],  # Needed for SkyMask
    "n_samples": [3],  # Run a few seeds for reliability
    "experiment.global_rounds": [100],  # Set a reasonable number of rounds
    "experiment.use_early_stopping": [False],  # Turn OFF for benchmarking
}


def generate_focused_valuation_scenarios() -> List[Scenario]:
    """
    Generates scenarios focused on ONE dataset/model, testing multiple defenses
    with ALL valuation methods enabled. Assumes defenses are already tuned.
    """
    scenarios = []

    scenario = Scenario(
        name=f"valuation_focused_{FOCUSED_DATASET}_{FOCUSED_MODEL}",
        base_config_factory=FOCUSED_BASE_FACTORY,
        modifiers=[
            FOCUSED_MODIFIER,
            use_image_backdoor_attack,  # Configure your standard backdoor
            use_sybil_attack('mimic')  # Assuming mimicry attack
        ],
        parameter_grid={
            **SHARED_PARAMS,
            **FIXED_ATTACK_PARAMS,
            **VALUATION_PARAMS,
            "aggregation.method": AGGREGATORS_TO_TEST,  # Sweep through defenses

            # --- IMPORTANT: Add Tuned Defense Params Here ---
            # You MUST include the "Golden Parameters" found during tuning.
            # Example (replace with your actual tuned values):
            "aggregation.clip_norm": [5.0],  # Example: Best clip norm for FLTrust/MartFL
            "aggregation.martfl.change_base": [True],  # Example: Best setting for MartFL
            "aggregation.martfl.max_k": [5],  # Example: Best setting for MartFL
            "aggregation.skymask.mask_threshold": [0.7],  # Example: Best setting for SkyMask
            # Add other tuned params (mask_epochs, mask_lr etc. for SkyMask)
        }
    )
    scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    output_dir = "./configs_generated/step4_valuation_benchmark"
    generator = ExperimentGenerator(output_dir)
    all_scenarios = generate_focused_valuation_scenarios()

    print("\n--- Generating Focused Valuation Scenarios (Step 4) ---")
    total_configs = 0
    for scenario in all_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        # Apply static modifiers (dataset, attack)
        modified_base_config = base_config
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # Generate uses the modified base and applies the grid
        # (which includes defense sweep + valuation settings)
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All valuation benchmark configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run these experiments. NOTE: They will be slower due to LOO/KernelSHAP.")
    print("2. Analyze the 'seller_metrics.csv' and 'training_log.csv' files.")
    print("   - Compare `price_paid`, `influence_score`, `marginal_contrib_loo`, `kernelshap_score`.")
    print("   - Analyze FPR and ADR for each defense.")
