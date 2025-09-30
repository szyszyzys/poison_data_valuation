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


# --- Reusable Modifier Functions ---

def use_cifar10_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-10 dataset."""
    config.experiment.dataset_name = "CIFAR10"
    config.data.image.property_skew.property_key = "class_in_[0,1,8,9]"
    return config


def use_cifar100_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-100 dataset."""
    config.experiment.dataset_name = "CIFAR100"
    config.data.image.property_skew.property_key = f"class_in_{list(range(50))}"
    return config


def use_trec_config(config: AppConfig) -> AppConfig:
    """Modifier for the TREC dataset."""
    config.experiment.dataset_name = "TREC"
    return config


def use_image_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
    return config


def use_text_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.TEXT_BACKDOOR
    return config


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """Returns a modifier function that enables a specific Sybil attack strategy."""
    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.sybil.is_sybil = True
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config
    return modifier


# --- Generator Functions for Scenarios ---

def generate_attack_impact_scenarios() -> List[Scenario]:
    """Generates scenarios to test the impact of attacks against different defenses."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]
    POISON_RATES_TO_SWEEP = [0.1, 0.3, 0.5, 0.7, 1.0]
    IMAGE_MODELS_TO_TEST = ["cnn", "resnet18"]
    IMAGE_DATASETS = [("cifar10", use_cifar10_config), ("cifar100", use_cifar100_config)]
    # --- FIX: Use the correct modifier for the TREC dataset ---
    TEXT_DATASETS = [("trec", use_trec_config)]

    for group_name, sweep_params in [
        ("vary_adv_rate", {"experiment.adv_rate": ADV_RATES_TO_SWEEP, "adversary_seller_config.poisoning.poison_rate": [0.5]}),
        ("vary_poison_rate", {"experiment.adv_rate": [0.3], "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP})
    ]:
        for dataset_name, modifier in IMAGE_DATASETS:
            for model_name in IMAGE_MODELS_TO_TEST:
                model_config_name = f"{dataset_name}_{model_name}"
                scenarios.append(Scenario(
                    name=f"impact_{group_name}_{dataset_name}_{model_name}",
                    base_config_factory=get_base_image_config,
                    modifiers=[modifier, use_image_backdoor_attack],
                    parameter_grid={
                        "experiment.image_model_config_name": [model_config_name],
                        "experiment.model_structure": [model_name],
                        "experiment.aggregation_method": ALL_AGGREGATORS,
                        **sweep_params
                    }
                ))
        for dataset_name, modifier in TEXT_DATASETS:
            scenarios.append(Scenario(
                name=f"impact_{group_name}_{dataset_name}",
                base_config_factory=get_base_text_config,
                modifiers=[modifier, use_text_backdoor_attack],
                parameter_grid={
                    "experiment.aggregation_method": ALL_AGGREGATORS,
                    **sweep_params
                }
            ))
    return scenarios


def generate_sybil_impact_scenarios() -> List[Scenario]:
    """Generates scenarios to isolate the impact of Sybil coordination strategies."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic', 'pivot', 'knock_out']
    IMAGE_MODELS_TO_TEST = ["cnn", "resnet18"]

    for model_name in IMAGE_MODELS_TO_TEST:
        model_config_name = f"cifar10_{model_name}"
        # --- Baseline Scenario (Poisoning Attack WITHOUT Sybil Coordination) ---
        scenarios.append(Scenario(
            name=f"sybil_baseline_cifar10_{model_name}",
            base_config_factory=get_base_image_config,
            modifiers=[use_cifar10_config, use_image_backdoor_attack],
            parameter_grid={
                "experiment.image_model_config_name": [model_config_name],
                "experiment.model_structure": [model_name],
                "experiment.aggregation_method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.5],
                "adversary_seller_config.sybil.is_sybil": [False],
            }
        ))
        # --- Scenarios for each Sybil Strategy ---
        for strategy in SYBIL_STRATEGIES:
            scenarios.append(Scenario(
                name=f"sybil_{strategy}_cifar10_{model_name}",
                base_config_factory=get_base_image_config,
                modifiers=[use_cifar10_config, use_image_backdoor_attack, use_sybil_attack(strategy)],
                parameter_grid={
                    "experiment.image_model_config_name": [model_config_name],
                    "experiment.model_structure": [model_name],
                    "experiment.aggregation_method": ALL_AGGREGATORS,
                    "experiment.adv_rate": [0.3],
                    "adversary_seller_config.poisoning.poison_rate": [0.5],
                }
            ))
    return scenarios


def generate_data_heterogeneity_scenarios() -> List[Scenario]:
    """Generates scenarios to test the impact of Non-IID data distributions."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.1]
    IMAGE_MODELS_TO_TEST = ["cnn", "resnet18"]

    for model_name in IMAGE_MODELS_TO_TEST:
        model_config_name = f"cifar10_{model_name}"
        scenarios.append(Scenario(
            name=f"heterogeneity_impact_cifar10_{model_name}",
            base_config_factory=get_base_image_config,
            modifiers=[use_cifar10_config, use_image_backdoor_attack],
            parameter_grid={
                "experiment.image_model_config_name": [model_config_name],
                "experiment.model_structure": [model_name],
                "experiment.aggregation_method": ALL_AGGREGATORS,
                "data.image.property_skew.dirichlet_alpha": DIRICHLET_ALPHAS_TO_SWEEP,
                "data.image.strategy": ["dirichlet"],
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.5],
            }
        ))
    return scenarios


# --- Create the final list by calling all generator functions ---
ALL_SCENARIOS = generate_attack_impact_scenarios()
ALL_SCENARIOS.extend(generate_sybil_impact_scenarios())
ALL_SCENARIOS.extend(generate_data_heterogeneity_scenarios())

# You can print the names of generated scenarios to verify
if __name__ == '__main__':
    print(f"Generated {len(ALL_SCENARIOS)} scenarios:")
    for s in ALL_SCENARIOS:
        print(f"  - {s.name}")

