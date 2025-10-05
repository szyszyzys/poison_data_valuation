# FILE: entry/gradient_market/automate_exp/scenarios.py
# Add this code before the final "ALL_SCENARIOS" list is created.

from typing import List, Callable

import torch

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, AggregationConfig, DebugConfig, TabularDataConfig, DataConfig, \
    AdversarySellerConfig, ServerAttackConfig, TrainingConfig, ExperimentConfig
from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator
from entry.gradient_market.automate_exp.scenarios import Scenario

def get_base_tabular_config() -> AppConfig:
    """Creates the default, base AppConfig for TABULAR-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="Texas100",  # Default to a known tabular dataset
            model_structure="None",  # Not used for tabular, which uses tabular_model_config_name
            aggregation_method="fedavg",
            global_rounds=100,
            n_sellers=10,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset_type="tabular",  # CRITICAL: Set dataset type to tabular
            evaluations=["clean"],
            evaluation_frequency=10,
            tabular_model_config_name="mlp_texas100_baseline" # Default model config
        ),
        training=TrainingConfig(
            local_epochs=5,
            batch_size=128,
            learning_rate=0.001
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            tabular=TabularDataConfig(
                buyer_ratio=0.1,
                strategy="dirichlet",
                property_skew={'dirichlet_alpha': 0.5}
            )
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42,
        n_samples=5,
        aggregation=AggregationConfig(
            method="fedavg"
        )
    )

def use_tabular_backdoor_attack(config: AppConfig) -> AppConfig:
    """Modifier for a backdoor attack on tabular data."""
    # NOTE: You will need to add TABULAR_BACKDOOR to your PoisonType enum
    # and implement the attack logic in your training pipeline.
    config.adversary_seller_config.poisoning.type = PoisonType.TABULAR_BACKDOOR
    return config


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """Returns a modifier function that enables a specific Sybil attack strategy."""

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.sybil.is_sybil = True
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config

    return modifier


def generate_tabular_attack_impact_scenarios() -> List[Scenario]:
    """Generates scenarios to test the impact of attacks on tabular data."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES = [0.1, 0.3, 0.5]
    TABULAR_EXPERIMENTS = [
        ("texas100", ["mlp_texas100_baseline", "resnet_texas100_baseline"]),
        ("purchase100", ["mlp_purchase100_baseline", "resnet_purchase100_baseline"]),
    ]

    for dataset_name, model_configs in TABULAR_EXPERIMENTS:
        for model_config in model_configs:
            scenarios.append(Scenario(
                name=f"attack_impact_{dataset_name}_{model_config.split('_')[0]}",
                base_config_factory=get_base_tabular_config,
                modifiers=[use_tabular_backdoor_attack],
                parameter_grid={
                    "experiment.dataset_name": [dataset_name],
                    "experiment.tabular_model_config_name": [model_config],
                    "aggregation.method": ALL_AGGREGATORS,
                    "experiment.adv_rate": ADV_RATES,
                    "adversary_seller_config.poisoning.poison_rate": [0.5]
                }
            ))
    return scenarios


def generate_tabular_sybil_impact_scenarios() -> List[Scenario]:
    """Generates scenarios to test Sybil coordination on tabular data."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic', 'pivot', 'knock_out']

    # We will test this on the Texas100 dataset with the MLP model
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    # Baseline: Attack is active, but no Sybil coordination
    scenarios.append(Scenario(
        name=f"sybil_baseline_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_attack],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "adversary_seller_config.sybil.is_sybil": [False]
        }
    ))
    # Sybil Scenarios: Attack is active WITH Sybil coordination
    for strategy in SYBIL_STRATEGIES:
        scenarios.append(Scenario(
            name=f"sybil_{strategy}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_attack, use_sybil_attack(strategy)],
            parameter_grid={
                "experiment.dataset_name": [dataset_name],
                "experiment.tabular_model_config_name": [model_config],
                "aggregation.method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.5],
            }
        ))
    return scenarios


def generate_tabular_data_heterogeneity_scenarios() -> List[Scenario]:
    """Generates scenarios to test the impact of Non-IID tabular data."""
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    DIRICHLET_ALPHAS = [100.0, 1.0, 0.1]

    # We will test this on the Purchase100 dataset with the ResNet model
    dataset_name = "purchase100"
    model_config = "resnet_purchase100_baseline"

    scenarios.append(Scenario(
        name=f"heterogeneity_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[],  # No attack is active, to purely test heterogeneity
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "data.tabular.strategy": ["dirichlet"],
            "data.tabular.property_skew": [{"dirichlet_alpha": alpha} for alpha in DIRICHLET_ALPHAS]
        }
    ))
    return scenarios


# --- Create the final list by calling all generator functions ---
ALL_TABULAR_SCENARIOS = []
ALL_TABULAR_SCENARIOS.extend(generate_tabular_attack_impact_scenarios())
ALL_TABULAR_SCENARIOS.extend(generate_tabular_sybil_impact_scenarios())
ALL_TABULAR_SCENARIOS.extend(generate_tabular_data_heterogeneity_scenarios())


def main():
    """Generates all configurations defined in scenarios.py."""
    output_dir = "./configs_generated/tabular"
    generator = ExperimentGenerator(output_dir)

    # The loop is now simpler and more powerful
    for scenario in ALL_TABULAR_SCENARIOS:
        # Get the correct base config for THIS specific scenario
        base_config = scenario.base_config_factory()
        # Generate all variations
        generator.generate(base_config, scenario)

    print(f"\n✅ All configurations have been generated in '{output_dir}'")


if __name__ == "__main__":
    main()
