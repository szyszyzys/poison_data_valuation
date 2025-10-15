# FILE: entry/gradient_market/automate_exp/generate_focused_tabular_configs.py

import torch
from dataclasses import dataclass, field
from typing import Any, List, Callable, Dict

# Make sure these imports match your project structure
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
            model_structure="None",  # Not used for tabular, but required by class
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
        n_samples=3,
    )


# --- Reusable Modifier Functions ---
def use_tabular_backdoor_with_trigger(trigger_conditions: Dict[str, Any]) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the tabular backdoor attack with specific trigger conditions.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Set the attack type to activate the correct adversary class
        config.adversary_seller_config.poisoning.type = PoisonType.TABULAR_BACKDOOR

        # 2. Set the specific trigger pattern for the attack
        trigger_key = "adversary_seller_config.poisoning.tabular_backdoor_params.active_attack_params.trigger_conditions"
        set_nested_attr(config, trigger_key, trigger_conditions)

        return config

    return modifier


def use_tabular_label_flipping_attack(config: AppConfig) -> AppConfig:
    """Modifier to set up for a simple label-flipping attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    config.adversary_seller_config.poisoning.poison_rate = 1.0
    return config


# --- Main Scenario Generator ---
def generate_focused_tabular_scenarios() -> List[Scenario]:
    """
    Generates a focused set of scenarios for tabular data, including an active backdoor attack.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']

    # --- IMPORTANT: Define your backdoor trigger here ---
    # The feature names (e.g., "feature_0") MUST match the actual column names in your dataset.
    # The values (e.g., 99.0) should be unusual or out-of-distribution to create a strong trigger.
    TEXAS100_TRIGGER = {
        "feature_0": 99.0,
        "feature_1": -99.0
    }

    # --- Scenario 1: Backdoor Attack (Now Active) ---
    scenarios.append(Scenario(
        name="tabular_backdoor_attack",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
        parameter_grid={
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- Scenario 2: Label-Flipping Attack ---
    scenarios.append(Scenario(
        name="tabular_label_flip_attack",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_label_flipping_attack],
        parameter_grid={
            "experiment.adv_rate": [0.3],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    output_dir = "./configs_generated/tabular_focused"
    generator = ExperimentGenerator(output_dir)

    all_scenarios = generate_focused_tabular_scenarios()

    for scenario in all_scenarios:
        base_config = scenario.base_config_factory()
        generator.generate(base_config, scenario)

    print(f"\n✅ Generated all configurations for {len(all_scenarios)} focused tabular scenarios.")
    print(f"   Configs saved to: {output_dir}")