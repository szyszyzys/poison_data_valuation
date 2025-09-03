# scenarios.py

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

# Make sure your config schemas and enums are accessible
from common.enums import PoisonType
from common.gradient_market_configs import AppConfig


# --- Define the structure of a Scenario ---
@dataclass
class Scenario:
    """A declarative representation of an experimental scenario."""
    name: str
    modifiers: List[Callable[[AppConfig], AppConfig]] = field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)


# --- Define Reusable Modifier Functions ---
# These functions encapsulate a specific change to the configuration.

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


# --- Define Your List of Scenarios to Generate ---
# This is now the single source of truth for all your experiments.

ALL_SCENARIOS = [
    # --- Baseline Scenarios (No Attack) ---
    Scenario(
        name="baseline_celeba",
        modifiers=[use_celeba_config],
        parameter_grid={"experiment.aggregation_method": ['fedavg', 'krum']}
    ),
    Scenario(
        name="baseline_camelyon",
        modifiers=[use_camelyon_config],
        parameter_grid={"experiment.aggregation_method": ['fedavg', 'krum']}
    ),

    # --- Backdoor Attack Scenarios ---
    Scenario(
        name="backdoor_celeba",
        modifiers=[use_celeba_config, use_backdoor_attack],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.2, 0.5]
        }
    ),
    Scenario(
        name="backdoor_camelyon",
        modifiers=[use_camelyon_config, use_backdoor_attack],
        parameter_grid={
            "experiment.adv_rate": [0.1, 0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.2, 0.5]
        }
    ),

    # --- Sybil Amplify Attack Scenarios ---
    Scenario(
        name="sybil_amplify_celeba",
        modifiers=[use_celeba_config, use_sybil_amplify],
        parameter_grid={
            "experiment.adv_rate": [0.2, 0.4],
            "adversary_seller_config.sybil.strategy_configs.amplify.factor": [2.0, 5.0, 10.0]
        }
    ),
]
