# scenarios.py

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config


# --- Define the structure of a Scenario ---
@dataclass
class Scenario:
    """A declarative representation of an experimental scenario."""
    name: str
    base_config_factory: Callable[[], AppConfig]
    modifiers: List[Callable[[AppConfig], AppConfig]] = field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)


# --- Define Reusable Modifier Functions ---

# Dataset Modifiers
def use_celeba_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CelebA dataset."""
    config.experiment.dataset_name = "CelebA"
    config.data.image.property_skew.property_key = "Smiling"
    return config


def use_fmnist_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the Fashion-MNIST dataset."""
    config.experiment.dataset_name = "FMNIST"

    # Define the property based on class labels for the skew.
    # Classes 0-4 are apparel (T-shirt, Trouser, Pullover, Dress, Coat).
    config.data.image.property_skew.property_key = "class_in_[0,1,2,3,4]"

    return config


def use_camelyon_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the Camelyon16 dataset."""
    config.experiment.dataset_name = "Camelyon16"
    config.data.image.property_skew.property_key = "tumor"
    return config


# --- UPDATED: Replaced the single backdoor modifier with two specific ones ---
def use_image_backdoor_attack(config: AppConfig) -> AppConfig:
    """Modifier to enable an image backdoor attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
    return config


def use_text_backdoor_attack(config: AppConfig) -> AppConfig:
    """Modifier to enable a text backdoor attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.TEXT_BACKDOOR
    return config


def use_sybil_amplify(config: AppConfig) -> AppConfig:
    """Modifier to enable the Sybil Amplify attack."""
    # Note: Sybil can be paired with any attack, here we default to image backdoor
    config = use_image_backdoor_attack(config)
    config.adversary_seller_config.sybil.is_sybil = True
    config.adversary_seller_config.sybil.role_config = {"amplify": 1.0}
    return config


ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']

# ==============================================================================
# ALL_SCENARIOS
# ==============================================================================
ALL_SCENARIOS = [
    # ==========================================================================
    # == Experiment Group 1: Varying Adversary Rate (Fixed Poison Rate) ==
    # ==========================================================================

    # --- IMAGE (CelebA) ---
    # Scenario(
    #     name="poison_vary_adv_rate_celeba",
    #     base_config_factory=get_base_image_config,
    #     modifiers=[use_celeba_config, use_image_backdoor_attack],
    #     parameter_grid={
    #         # Iterate through all 4 aggregation methods
    #         "experiment.aggregation_method": ALL_AGGREGATORS,
    #         # Fix the poison rate for this experiment group
    #         "adversary_seller_config.poisoning.poison_rate": [0.3],
    #         # Sweep the adversary rate
    #         "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #     }
    # ),

    # # --- TEXT (AG_NEWS) ---
    # Scenario(
    #     name="poison_vary_adv_rate_agnews",
    #     base_config_factory=get_base_text_config,
    #     modifiers=[use_text_backdoor_attack],
    #     parameter_grid={
    #         # Iterate through all 4 aggregation methods
    #         "experiment.aggregation_method": ALL_AGGREGATORS,
    #         # Fix the poison rate for this experiment group
    #         "adversary_seller_config.poisoning.poison_rate": [0.3],
    #         # Sweep the adversary rate
    #         "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #     }
    # ),

    # ==========================================================================
    # == Experiment Group 2: Varying Poison Rate (Fixed Adversary Rate) ==
    # ==========================================================================

    # --- IMAGE (CelebA) ---
    # Scenario(
    #     name="poison_vary_poison_rate_celeba",
    #     base_config_factory=get_base_image_config,
    #     modifiers=[use_celeba_config, use_image_backdoor_attack],
    #     parameter_grid={
    #         # Iterate through all 4 aggregation methods
    #         "experiment.aggregation_method": ALL_AGGREGATORS,
    #         # Fix the adversary rate for this experiment group
    #         "experiment.adv_rate": [0.3],
    #         # Sweep the local data poison rate
    #         "adversary_seller_config.poisoning.poison_rate": [0.1, 0.3, 0.5, 0.7, 1.0],
    #     }
    # ),

    # # --- TEXT (AG_NEWS) ---
    # Scenario(
    #     name="poison_vary_poison_rate_agnews",
    #     base_config_factory=get_base_text_config,
    #     modifiers=[use_text_backdoor_attack],
    #     parameter_grid={
    #         # Iterate through all 4 aggregation methods
    #         "experiment.aggregation_method": ALL_AGGREGATORS,
    #         # Fix the adversary rate for this experiment group
    #         "experiment.adv_rate": [0.3],
    #         # Sweep the local data poison rate
    #         "adversary_seller_config.poisoning.poison_rate": [0.1, 0.3, 0.5, 0.7, 1.0],
    #     }
    # ),
]

ALL_SCENARIOS.extend([
    # ==========================================================================
    # == Scenario for Offline Privacy Analysis Logging ==
    # ==========================================================================

    Scenario(
        name="privacy_analysis_logging_fmnist_lenet",
        base_config_factory=get_base_image_config,
        modifiers=[use_fmnist_config],  # Or any other dataset modifier
        parameter_grid={
            "n_samples": [1],
            "experiment.model_structure": ["lenet"],  # <-- Only one model
            # Use a standard, non-robust aggregator to see the raw leakage
            "aggregation.method": ["fedavg"],
            "experiment.dataset_name": ["fmnist"],  # <-- Only one model

            # --- Key Settings for Logging ---
            # Turn ON gradient saving
            "debug.save_individual_gradients": [True],
            # Save the gradient from EVERY round
            "debug.gradient_save_frequency": [1],

            # --- Ensure NO Attacks are Active ---
            # Turn OFF client-side poisoning
            "adversary_seller_config.poisoning.type": [PoisonType.NONE],
            # Turn OFF server-side attacks during the run
            "server_attack_config.attack_name": ['none'],

            # This is the parameter you will vary in your experiments
            "training.batch_size": [64],
        }
    ),

    Scenario(
        name="privacy_analysis_robust_aggregators_fmnist_lenet",
        base_config_factory=get_base_image_config,
        modifiers=[use_fmnist_config],
        parameter_grid={
            # --- Use a robust aggregator ---
            "n_samples": [1],
            "aggregation.method": ["fltrust", "martfl"],  # This will create runs for all three
            "experiment.model_structure": ["lenet"],  # <-- Only one model dataset_name
            "experiment.dataset_name": ["fmnist"],  # <-- Only one model
            "debug.save_individual_gradients": [True],
            "debug.gradient_save_frequency": [1],
            "adversary_seller_config.poisoning.type": [PoisonType.NONE],
            "server_attack_config.attack_name": ['none'],

            # --- Make one client particularly vulnerable ---
            "training.batch_size": [1],
        }
    )

])


def use_small_subset(config: AppConfig) -> AppConfig:
    """A modifier to drastically reduce dataset size for quick tests."""
    config.experiment.use_subset = True
    return config


smoke_test_scenario = Scenario(
    name="smoke_test_image",
    base_config_factory=get_base_image_config,
    modifiers=[use_celeba_config, use_small_subset, use_image_backdoor_attack],
    parameter_grid={
        # Use a small number of rounds and sellers
        "experiment.global_rounds": [2],
        "experiment.n_sellers": [4],
        "training.local_epochs": [1],

        # Use a robust aggregator to test its logic
        "aggregation.method": ["martfl"],

        # Activate one malicious client
        "experiment.adv_rate": [0.25],
        "adversary_seller_config.poisoning.poison_rate": [0.5],

        # Initialize the server-side attacker to make sure it doesn't crash
        "server_attack_config.attack_name": ['gradient_inversion'],

        # Test the gradient saving feature
        "debug.save_individual_gradients": [True],
        "debug.gradient_save_frequency": [1],
    }
)

smoke_test_text_scenario = Scenario(
    name="smoke_test_text",
    base_config_factory=get_base_text_config,  # <-- Use the text base config
    modifiers=[use_text_backdoor_attack],  # Use an attack to test that path
    parameter_grid={
        # Minimal settings for a fast run
        "experiment.dataset_name": ["AG_NEWS"],
        "experiment.global_rounds": [2],
        "experiment.n_sellers": [4],
        "experiment.adv_rate": [0.5],

        # --- CRITICAL: Use a small subset of the data ---
        "experiment.use_subset": [True],
        "experiment.subset_size": [150],

        "experiment.device": ["cpu"]
    }
)

# ALL_SCENARIOS.append(smoke_test_scenario)
#
# ALL_SCENARIOS.append(smoke_test_text_scenario)
