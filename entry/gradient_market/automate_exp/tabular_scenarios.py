# FILE: entry/gradient_market/automate_exp/tabular_scenarios.py

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

import torch

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, AggregationConfig, DebugConfig, TabularDataConfig, DataConfig, \
    AdversarySellerConfig, ServerAttackConfig, TrainingConfig, ExperimentConfig


# --- Define the structure of a Scenario ---
@dataclass
class Scenario:
    """A declarative representation of an experimental scenario."""
    name: str
    base_config_factory: Callable[[], AppConfig]
    modifiers: List[Callable[[AppConfig], AppConfig]] = field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)


# --- Base Configuration Factory ---

def get_base_tabular_config() -> AppConfig:
    """Creates the default, base AppConfig for TABULAR-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="Texas100",
            model_structure="None",
            aggregation_method="fedavg",
            global_rounds=100,
            n_sellers=10,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset_type="tabular",
            evaluations=["clean"],
            evaluation_frequency=10,
            tabular_model_config_name="mlp_texas100_baseline"
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
        debug=DebugConfig(),
        seed=42,
        n_samples=3,
        aggregation=AggregationConfig(method="fedavg")
    )


# --- Reusable Modifier Functions ---

def use_tabular_backdoor_attack(config: AppConfig) -> AppConfig:
    """Modifier for a backdoor attack on tabular data."""
    # NOTE: You will need to add TABULAR_BACKDOOR to your PoisonType enum
    # and implement the attack logic. A label flip is a good alternative.
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    return config


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """Returns a modifier function that enables a specific Sybil attack strategy."""

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.sybil.is_sybil = True
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config

    return modifier


# --- FOCUSED SCENARIO GENERATORS ---

def generate_tabular_oracle_vs_buyer_bias_scenarios() -> List[Scenario]:
    """
    Generates the core Oracle vs. Buyer Bias scenario for a representative tabular dataset.
    This is the most important new experiment for showing generalizability.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']

    scenarios.append(Scenario(
        name="oracle_vs_buyer_bias_texas100",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_attack, use_sybil_attack('mimic')],
        parameter_grid={
            "experiment.dataset_name": ["texas100"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "aggregation.root_gradient_source": ["buyer", "validation"],
        }
    ))
    return scenarios


def generate_tabular_sybil_impact_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test Sybil coordination on tabular data.
    This provides a valuable deep-dive.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic', 'pivot', 'knock_out']
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
    """
    Generates scenarios to test the impact of Non-IID tabular data UNDER ATTACK.
    This is a refined, more robust test.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    DIRICHLET_ALPHAS = [100.0, 1.0, 0.1]
    dataset_name = "purchase100"
    model_config = "resnet_purchase100_baseline"

    scenarios.append(Scenario(
        name=f"heterogeneity_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_attack],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "data.tabular.strategy": ["dirichlet"],
            "data.tabular.property_skew": [{"dirichlet_alpha": alpha} for alpha in DIRICHLET_ALPHAS]
        }
    ))
    return scenarios


# --- Create the final, focused list by calling the new and refined generator functions ---
ALL_TABULAR_SCENARIOS = []
ALL_TABULAR_SCENARIOS.extend(generate_tabular_oracle_vs_buyer_bias_scenarios())
ALL_TABULAR_SCENARIOS.extend(generate_tabular_sybil_impact_scenarios())
ALL_TABULAR_SCENARIOS.extend(generate_tabular_data_heterogeneity_scenarios())

# This part can be in a separate runner script, but is included here for completeness
if __name__ == "__main__":
    # You would need to import your ExperimentGenerator class
    # from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator

    output_dir = "./configs_generated/tabular_new"
    # generator = ExperimentGenerator(output_dir)

    print(f"✅ The following {len(ALL_TABULAR_SCENARIOS)} tabular scenarios are defined:")
    for scenario in ALL_TABULAR_SCENARIOS:
        print(f"  - {scenario.name}")
    print(f"\nRun the main generator script to create config files in '{output_dir}'")