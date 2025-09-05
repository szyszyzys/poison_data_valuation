# scenarios.py

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

from common.enums import PoisonType
# Make sure your config schemas and enums are accessible from a central place
from common.gradient_market_configs import AppConfig
# Import your base config factories from your main runner script
from generate_configs import get_base_image_config, get_base_text_config


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


def use_camelyon_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the Camelyon16 dataset."""
    config.experiment.dataset_name = "Camelyon16"
    config.data.image.property_skew.property_key = "tumor"
    return config


# Attack Modifiers
def use_backdoor_attack(config: AppConfig) -> AppConfig:
    """Modifier to enable a standard backdoor attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.BACKDOOR
    return config


def use_sybil_amplify(config: AppConfig) -> AppConfig:
    """Modifier to enable the Sybil Amplify attack."""
    config = use_backdoor_attack(config)  # Sybil builds on backdoor
    config.adversary_seller_config.sybil.is_sybil = True
    config.adversary_seller_config.sybil.role_config = {"amplify": 1.0}
    return config

ALL_SCENARIOS = [
    # --- IMAGE: Baseline Scenarios (No Attack) ---
    Scenario(
        name="baseline_celeba",
        base_config_factory=get_base_image_config,  # Link to the image base
        modifiers=[use_celeba_config],
        parameter_grid={"experiment.aggregation_method": ['fedavg', 'krum']}
    ),
    Scenario(
        name="baseline_camelyon",
        base_config_factory=get_base_image_config,  # Link to the image base
        modifiers=[use_camelyon_config],
        parameter_grid={"experiment.aggregation_method": ['fedavg', 'krum']}
    ),

    # --- TEXT: Baseline Scenario (AG_NEWS) ---
    Scenario(
        name="baseline_agnews",
        base_config_factory=get_base_text_config,  # Link to the text base
        modifiers=[],  # No dataset modifier needed if base is already AG_NEWS
        parameter_grid={"experiment.aggregation_method": ['fedavg', 'median']}
    ),

    # --- IMAGE: Backdoor Attack Scenarios ---
    Scenario(
        name="backdoor_celeba",
        base_config_factory=get_base_image_config,
        modifiers=[use_celeba_config, use_backdoor_attack],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.2, 0.5]
        }
    ),
    Scenario(
        name="backdoor_camelyon",
        base_config_factory=get_base_image_config,
        modifiers=[use_camelyon_config, use_backdoor_attack],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "adversary_seller_config.poisoning.poison_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 1]
        }
    ),

    # --- TEXT: Backdoor Attack Scenarios ---
    Scenario(
        name="backdoor_agnews",
        base_config_factory=get_base_text_config,
        modifiers=[use_backdoor_attack],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "adversary_seller_config.poisoning.text_backdoor_params.trigger_content": ["order", "movie", "apple"]
        }
    ),

    # --- IMAGE: Sybil Amplify Attack Scenarios ---
    Scenario(
        name="sybil_amplify_celeba",
        base_config_factory=get_base_image_config,
        modifiers=[use_celeba_config, use_sybil_amplify],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "adversary_seller_config.sybil.strategy_configs.amplify.factor": [2.0, 5.0, 10.0]
        }
    ),
]
