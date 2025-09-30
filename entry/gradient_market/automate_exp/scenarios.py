from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Tuple

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


# --- 1. Define Reusable Modifier Functions for New Datasets ---

def use_cifar10_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-10 dataset."""
    config.experiment.dataset_name = "CIFAR10"
    # Property for data skew: classes 0-4 (vehicles) vs 5-9 (animals)
    config.data.image.property_skew.property_key = "class_in_[0,1,8,9]"  # airplane, automobile, ship, truck
    return config


def use_cifar100_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-100 dataset."""
    config.experiment.dataset_name = "CIFAR100"
    # Property for data skew: first 50 classes vs. last 50
    config.data.image.property_skew.property_key = f"class_in_{list(range(50))}"
    return config


def use_trec_config(config: AppConfig) -> AppConfig:
    """Modifier for the TREC dataset (AG_NEWS is the default for text)."""
    config.experiment.dataset_name = "TREC"
    return config


# Attack Modifiers (can be reused from your original script)
def use_image_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
    return config


def use_text_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.TEXT_BACKDOOR
    return config


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier function that enables a specific Sybil attack strategy.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Enable the Sybil module
        config.adversary_seller_config.sybil.is_sybil = True
        # Set the specific strategy for non-selected Sybils
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config

    return modifier


# --- 2. Main Function to Generate All Scenarios ---

def generate_attack_impact_scenarios() -> List[Scenario]:
    """
    Generates a list of scenarios to test the impact of attacks against
    different defense mechanisms, now iterating through specified models.
    """
    scenarios = []

    # --- Define Experiment Parameters ---
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]
    POISON_RATES_TO_SWEEP = [0.1, 0.3, 0.5, 0.7, 1.0]

    # --- NEW: Define the models to test for image datasets ---
    IMAGE_MODELS_TO_TEST = ["cnn", "resnet18"]

    IMAGE_DATASETS: List[Tuple[str, Callable]] = [
        ("cifar10", use_cifar10_config),
        ("cifar100", use_cifar100_config),
    ]
    TEXT_DATASETS: List[Tuple[str, Callable]] = [
        ("trec", use_trec_config),
    ]

    # --- Group 1: Varying Adversary Rate (Fixed Poison Rate) ---
    for dataset_name, modifier in IMAGE_DATASETS:
        # --- NEW: Loop through the desired models ---
        for model_name in IMAGE_MODELS_TO_TEST:
            # Construct the "recipe" name (e.g., "cifar10_resnet18")
            model_config_name = f"{dataset_name}_{model_name}"

            scenarios.append(Scenario(
                # Update the scenario name to be more descriptive
                name=f"impact_vary_adv_rate_{dataset_name}_{model_name}",
                base_config_factory=get_base_image_config,
                modifiers=[modifier, use_image_backdoor_attack],
                parameter_grid={
                    # --- FIX: Use the new config name key ---
                    "experiment.image_model_config_name": [model_config_name],
                    "experiment.aggregation_method": ALL_AGGREGATORS,
                    "adversary_seller_config.poisoning.poison_rate": [0.5],
                    "experiment.adv_rate": ADV_RATES_TO_SWEEP,
                }
            ))

    # Text scenarios remain the same as they don't use the new model config system
    for name, modifier in TEXT_DATASETS:
        scenarios.append(Scenario(
            name=f"impact_vary_adv_rate_{name}",
            base_config_factory=get_base_text_config,
            modifiers=[modifier, use_text_backdoor_attack],
            parameter_grid={
                "experiment.aggregation_method": ALL_AGGREGATORS,
                "adversary_seller_config.poisoning.poison_rate": [0.5],
                "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            }
        ))

    # --- Group 2: Varying Poison Rate (Fixed Adversary Rate) ---
    for dataset_name, modifier in IMAGE_DATASETS:
        for model_name in IMAGE_MODELS_TO_TEST:
            model_config_name = f"{dataset_name}_{model_name}"

            scenarios.append(Scenario(
                name=f"impact_vary_poison_rate_{dataset_name}_{model_name}",
                base_config_factory=get_base_image_config,
                modifiers=[modifier, use_image_backdoor_attack],
                parameter_grid={
                    "experiment.image_model_config_name": [model_config_name],
                    "experiment.aggregation_method": ALL_AGGREGATORS,
                    "experiment.adv_rate": [0.3],
                    "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP,
                }
            ))

    # Text scenarios remain the same
    for name, modifier in TEXT_DATASETS:
        scenarios.append(Scenario(
            name=f"impact_vary_poison_rate_{name}",
            base_config_factory=get_base_text_config,
            modifiers=[modifier, use_text_backdoor_attack],
            parameter_grid={
                "experiment.aggregation_method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP,
            }
        ))

    return scenarios


def generate_sybil_impact_scenarios() -> List[Scenario]:
    """
    Generates scenarios to isolate the impact of Sybil coordination strategies.
    The underlying backdoor attack is fixed.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic', 'pivot', 'knock_out']

    # --- Baseline Scenario (Poisoning Attack WITHOUT Sybil Coordination) ---
    scenarios.append(Scenario(
        name="sybil_baseline_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack],
        parameter_grid={
            "experiment.aggregation_method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],  # Fixed adversary rate
            "adversary_seller_config.poisoning.poison_rate": [0.5],  # Fixed poison rate
            "adversary_seller_config.sybil.is_sybil": [False],  # Explicitly OFF
        }
    ))

    # --- Scenarios for each Sybil Strategy ---
    for strategy in SYBIL_STRATEGIES:
        scenarios.append(Scenario(
            name=f"sybil_{strategy}_cifar10",
            base_config_factory=get_base_image_config,
            # Chain the modifiers: set dataset, set base attack, THEN set sybil strategy
            modifiers=[
                use_cifar10_config,
                use_image_backdoor_attack,
                use_sybil_attack(strategy)  # Use the new modifier
            ],
            parameter_grid={
                "experiment.aggregation_method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],  # Fixed adversary rate
                "adversary_seller_config.poisoning.poison_rate": [0.5],  # Fixed poison rate
            }
        ))

    return scenarios


def generate_data_heterogeneity_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test the impact of Non-IID data distributions
    on attack and defense performance.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']

    # Define the levels of data skew to test via Dirichlet alpha
    # High alpha = similar data (IID), Low alpha = skewed data (Non-IID)
    DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.1]

    scenarios.append(Scenario(
        name="heterogeneity_impact_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack],
        parameter_grid={
            # The two key variables for this experiment
            "experiment.aggregation_method": ALL_AGGREGATORS,
            "data.image.property_skew.dirichlet_alpha": DIRICHLET_ALPHAS_TO_SWEEP,

            # Tell the partitioner to use the Dirichlet strategy
            "data.image.strategy": ["dirichlet"],

            # Use fixed attack parameters to isolate the effect of data skew
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
        }
    ))

    return scenarios


# --- 3. Update the Final List to Include the New Scenarios ---
ALL_SCENARIOS = generate_attack_impact_scenarios()
ALL_SCENARIOS.extend(generate_sybil_impact_scenarios())
ALL_SCENARIOS.extend(generate_data_heterogeneity_scenarios())

# You can print the names of generated scenarios to verify
if __name__ == '__main__':
    print(f"Generated {len(ALL_SCENARIOS)} scenarios:")
    for s in ALL_SCENARIOS:
        print(f"  - {s.name}")
