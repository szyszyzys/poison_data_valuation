# FILE: entry/gradient_market/automate_exp/tabular_scenarios.py

import torch
from dataclasses import dataclass, field
from typing import Any, List, Callable, Dict

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, AggregationConfig, TabularDataConfig, DataConfig, \
    AdversarySellerConfig, ServerAttackConfig, TrainingConfig, ExperimentConfig
from entry.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr


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
            evaluation_frequency=10,
            tabular_model_config_name="mlp_texas100_baseline"
        ),
        training=TrainingConfig(local_epochs=5, batch_size=128, learning_rate=0.001),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            tabular=TabularDataConfig(
                buyer_ratio=0.1,
                strategy="dirichlet",
                property_skew={'dirichlet_alpha': 0.5}
            )
        ),
        aggregation=AggregationConfig(method="fedavg"),
        seed=42,
        n_samples=3
    )


# --- CRITICAL: DEFINE YOUR TRIGGER PATTERNS HERE ---
# You MUST verify that these feature names match your datasets.
TEXAS100_TRIGGER = {"feature_0": 99.0, "feature_1": -99.0}
PURCHASE100_TRIGGER = {"feature_0": 99.0, "feature_1": -99.0}


# --- Reusable Modifier Functions ---
def use_tabular_backdoor_with_trigger(trigger_conditions: Dict[str, Any]) -> Callable[[AppConfig], AppConfig]:
    """Returns a modifier to enable the tabular backdoor attack with specific trigger conditions."""

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.poisoning.type = PoisonType.TABULAR_BACKDOOR
        trigger_key = "adversary_seller_config.poisoning.tabular_backdoor_params.active_attack_params.trigger_conditions"
        set_nested_attr(config, trigger_key, trigger_conditions)
        return config

    return modifier


def use_tabular_label_flipping_attack(config: AppConfig) -> AppConfig:
    """Modifier for a simple label-flipping attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    config.adversary_seller_config.poisoning.poison_rate = 1.0
    return config


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """Modifier to enable a specific Sybil attack strategy."""

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.sybil.is_sybil = True
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config

    return modifier


# --- Scenario Generator Functions (All Updated) ---
def generate_tabular_oracle_vs_buyer_bias_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    scenarios.append(Scenario(
        name="oracle_vs_buyer_bias_texas100",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER), use_sybil_attack('mimic')],
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
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic', 'pivot', 'knock_out']
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    scenarios.append(Scenario(
        name=f"sybil_baseline_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "adversary_seller_config.sybil.is_sybil": [False]
        }
    ))
    for strategy in SYBIL_STRATEGIES:
        scenarios.append(Scenario(
            name=f"sybil_{strategy}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER), use_sybil_attack(strategy)],
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
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    DIRICHLET_ALPHAS = [100.0, 1.0, 0.1]
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"  # Assuming a model config for purchase100 exists
    scenarios.append(Scenario(
        name=f"heterogeneity_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "data.tabular.strategy": ["dirichlet"],
            "data.tabular.property_skew.dirichlet_alpha": DIRICHLET_ALPHAS
        }
    ))
    return scenarios


def generate_tabular_attack_impact_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.3, 0.5]
    POISON_RATES_TO_SWEEP = [0.1, 0.5, 1.0]
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"
    for group_name, sweep_params in [
        ("vary_adv_rate",
         {"experiment.adv_rate": ADV_RATES_TO_SWEEP, "adversary_seller_config.poisoning.poison_rate": [0.5]}),
        ("vary_poison_rate",
         {"experiment.adv_rate": [0.3], "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP})
    ]:
        scenarios.append(Scenario(
            name=f"impact_{group_name}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
            parameter_grid={
                "experiment.dataset_name": [dataset_name],
                "experiment.tabular_model_config_name": [model_config],
                "aggregation.method": ALL_AGGREGATORS,
                **sweep_params
            }
        ))
    return scenarios


def generate_tabular_label_flipping_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    scenarios.append(Scenario(
        name="label_flip_texas100",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_label_flipping_attack],
        parameter_grid={
            "experiment.dataset_name": ["texas100"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
        }
    ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    output_dir = "./configs_generated/tabular_new_with_trigger"  # Changed output dir to avoid overwriting old results
    generator = ExperimentGenerator(output_dir)

    ALL_TABULAR_SCENARIOS = []
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_oracle_vs_buyer_bias_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_sybil_impact_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_data_heterogeneity_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_attack_impact_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_label_flipping_scenarios())

    for scenario in ALL_TABULAR_SCENARIOS:
        base_config = scenario.base_config_factory()
        generator.generate(base_config, scenario)

    print(f"\n✅ All configurations for {len(ALL_TABULAR_SCENARIOS)} tabular scenarios have been generated.")
    print(f"   Configs saved to: {output_dir}")