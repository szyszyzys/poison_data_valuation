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


def generate_main_summary_figure_scenarios() -> List[Scenario]:
    """
    Generates a broad set of results for a main summary figure.
    This scenario fixes the attack parameters and evaluates various
    datasets, models, and aggregators, respecting their data modality.
    """
    scenarios = []

    # --- FIX: Define separate aggregator lists ---
    IMAGE_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']
    TEXT_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']  # Excludes SkyMask

    fixed_attack_params = {
        "experiment.adv_rate": [0.3],
        "adversary_seller_config.poisoning.poison_rate": [0.5],
    }

    IMAGE_DATASETS_TO_TEST = [
        ("cifar10", use_cifar10_config),
        ("cifar100", use_cifar100_config)
    ]

    # --- Image Scenarios ---
    for dataset_name, dataset_modifier in IMAGE_DATASETS_TO_TEST:
        for model_name in ["cnn", "resnet18"]:
            scenarios.append(Scenario(
                name=f"main_summary_{dataset_name}_{model_name}",
                base_config_factory=get_base_image_config,
                modifiers=[dataset_modifier, use_image_backdoor_attack, use_sybil_attack('mimic')],
                parameter_grid={
                    "experiment.image_model_config_name": [f"{dataset_name}_{model_name}"],
                    "experiment.model_structure": [model_name],
                    # Use the correct list
                    "aggregation.method": IMAGE_AGGREGATORS,
                    **fixed_attack_params
                }
            ))

    # --- Text Scenario (TREC) ---
    scenarios.append(Scenario(
        name="main_summary_trec",
        base_config_factory=get_base_text_config,
        modifiers=[use_trec_config, use_text_backdoor_attack],
        parameter_grid={
            # Use the correct list
            "aggregation.method": TEXT_AGGREGATORS,
            **fixed_attack_params
        }
    ))
    return scenarios


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
        ("vary_adv_rate",
         {"experiment.adv_rate": ADV_RATES_TO_SWEEP, "adversary_seller_config.poisoning.poison_rate": [0.5]}),
        ("vary_poison_rate",
         {"experiment.adv_rate": [0.3], "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP})
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
                        "aggregation.method": ALL_AGGREGATORS,
                        **sweep_params
                    }
                ))
        for dataset_name, modifier in TEXT_DATASETS:
            scenarios.append(Scenario(
                name=f"impact_{group_name}_{dataset_name}",
                base_config_factory=get_base_text_config,
                modifiers=[modifier, use_text_backdoor_attack],
                parameter_grid={
                    "aggregation.method": ALL_AGGREGATORS,
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
                "aggregation.method": ALL_AGGREGATORS,
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
                    "aggregation.method": ALL_AGGREGATORS,
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
                "aggregation.method": ALL_AGGREGATORS,
                "data.image.property_skew.dirichlet_alpha": DIRICHLET_ALPHAS_TO_SWEEP,
                "data.image.strategy": ["dirichlet"],
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.5],
            }
        ))
    return scenarios


def generate_oracle_vs_buyer_bias_scenarios() -> List[Scenario]:
    """
    Generates scenarios to compare the Oracle baseline (validation set reference)
    against the standard Biased Buyer baseline.
    """
    scenarios = []
    # Include SkyMask as it's discussed in the paper
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # This scenario uses a fixed, challenging attack setting
    fixed_attack_params = {
        "experiment.adv_rate": [0.3],
        "adversary_seller_config.poisoning.poison_rate": [0.5],
    }

    scenarios.append(Scenario(
        name="oracle_vs_buyer_bias_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack, use_sybil_attack('mimic')],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": ALL_AGGREGATORS,
            # This is the key parameter we sweep over
            "aggregation.root_gradient_source": ["buyer", "validation"],
            **fixed_attack_params
        }
    ))
    return scenarios


def generate_buyer_data_impact_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test the impact of the buyer's local data size
    on the effectiveness of various defense aggregators.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # This scenario uses a fixed, challenging attack setting
    fixed_attack_params = {
        "experiment.adv_rate": [0.3],
        "adversary_seller_config.poisoning.poison_rate": [0.5],
    }

    scenarios.append(Scenario(
        name="buyer_data_impact_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack, use_sybil_attack('mimic')],
        parameter_grid={
            # --- Fixed Parameters for this Experiment ---
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.root_gradient_source": ["buyer"],  # We only care about the buyer's data here

            # --- Swept Parameters ---
            "aggregation.method": ALL_AGGREGATORS,  # Test against all defenses

            # ✅ THIS IS THE KEY: Sweep over the buyer's data percentage
            # It will test the defenses when the buyer has 1%, 5%, 10%, and 20% of the data.
            "data.image.buyer_config.buyer_percentage": [0.01, 0.05, 0.10, 0.20],

            # --- Attack Parameters ---
            **fixed_attack_params
        }
    ))
    return scenarios


def generate_adv_rate_trend_scenarios() -> List[Scenario]:
    """
    A lean set of scenarios to show the trend of ASR vs. Adversary Rate for MartFL.
    """
    scenarios = []
    # Add a 0.0 baseline for a clean comparison point
    ADV_RATES_TO_SWEEP = [0.0, 0.2, 0.4]

    scenarios.append(Scenario(
        name="trend_adv_rate_martfl_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": ['martfl'],  # Only test MartFL for this trend
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": [0.5],
        }
    ))
    return scenarios


def use_label_flipping_attack(config: AppConfig) -> AppConfig:
    """Modifier to set up for a simple label-flipping attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    # For a simple label-flipping attack, adversaries often flip all their data
    config.adversary_seller_config.poisoning.poison_rate = 1.0
    return config


def generate_label_flipping_scenarios() -> List[Scenario]:
    """
    Generates simple scenarios to test defenses against a label-flipping attack.
    """
    scenarios = []
    AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    fixed_attack_params = {
        "experiment.adv_rate": [0.3],  # Set a fixed adversary rate
    }

    # --- Image Scenario (CIFAR-10) ---
    for model_name in ["cnn", "resnet18"]:
        scenarios.append(Scenario(
            name=f"label_flip_cifar10_{model_name}",
            base_config_factory=get_base_image_config,
            modifiers=[use_cifar10_config, use_label_flipping_attack],
            parameter_grid={
                "experiment.image_model_config_name": [f"cifar10_{model_name}"],
                "experiment.model_structure": [model_name],
                "aggregation.method": AGGREGATORS,
                **fixed_attack_params
            }
        ))

    # --- Text Scenario (TREC) ---
    scenarios.append(Scenario(
        name="label_flip_trec",
        base_config_factory=get_base_text_config,
        modifiers=[use_trec_config, use_label_flipping_attack],
        parameter_grid={
            "aggregation.method": AGGREGATORS,
            **fixed_attack_params
        }
    ))
    return scenarios


def generate_sybil_selection_rate_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test a Sybil strategy aimed at maximizing selection rate.
    """
    scenarios = []
    # Test against defenses that are vulnerable to this
    AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    # See how the strategy performs as the number of Sybils increases
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4]

    # --- Baseline: Poisoning attack WITHOUT Sybil coordination ---
    scenarios.append(Scenario(
        name="selection_rate_baseline_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "adversary_seller_config.sybil.is_sybil": [False],  # Baseline
        }
    ))

    # --- Test Scenario: The "Cluster Creation" Sybil strategy ---
    scenarios.append(Scenario(
        name="selection_rate_cluster_cifar10_cnn",
        base_config_factory=get_base_image_config,
        # Assume 'cluster' is a new strategy you'll implement
        modifiers=[use_cifar10_config, use_image_backdoor_attack, use_sybil_attack('cluster')],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": [0.5],
        }
    ))
    return scenarios


ALL_SCENARIOS = []
# 1. The core new experiment comparing the Oracle vs. Biased Buyer setups
ALL_SCENARIOS.extend(generate_oracle_vs_buyer_bias_scenarios())

# 2. A lean, focused trend analysis for adversary rate
ALL_SCENARIOS.extend(generate_adv_rate_trend_scenarios())

# 3. The focused deep-dive into different Sybil attack methods
ALL_SCENARIOS.extend(generate_sybil_impact_scenarios())

# 4. The focused analysis of data heterogeneity (Non-IID) impact
ALL_SCENARIOS.extend(generate_data_heterogeneity_scenarios())

ALL_SCENARIOS.extend(generate_main_summary_figure_scenarios())
ALL_SCENARIOS.extend(generate_label_flipping_scenarios())
ALL_SCENARIOS.extend(generate_sybil_selection_rate_scenarios())
ALL_SCENARIOS.extend(generate_buyer_data_impact_scenarios())

# You can print the names of generated scenarios to verify
if __name__ == '__main__':
    print(f"Generated {len(ALL_SCENARIOS)} focused scenarios:")
    for s in ALL_SCENARIOS:
        print(f"  - {s.name}")
