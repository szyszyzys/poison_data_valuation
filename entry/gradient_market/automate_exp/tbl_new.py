# FILE: entry/gradient_market/automate_exp/tabular_scenarios.py

from dataclasses import dataclass, field
from typing import Any, List, Callable, Dict

import torch

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, AggregationConfig, TabularDataConfig, DataConfig, \
    AdversarySellerConfig, ServerAttackConfig, TrainingConfig, ExperimentConfig, DebugConfig
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
            model_structure="mlp",  # <--- CRITICAL FIX 1 (was "None")
            aggregation_method="fedavg",
            global_rounds=80,
            n_sellers=10,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset_type="tabular",
            eval_frequency=10,
            evaluations=["clean", "poison"],  # <--- CRITICAL FIX 2 (added "poison")
            tabular_model_config_name="mlp_texas100_baseline"
        ),
        # training=TrainingConfig(local_epochs=2, batch_size=64, learning_rate=0.0001,),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            optimizer="SGD",  # Set optimizer to SGD
            learning_rate=0.01,  # Use a standard SGD learning rate (0.0001 is very low)
            momentum=0.9,  # Add momentum (important for SGD)
            weight_decay=0.0001,  # Optional: small L2 regularization
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
        aggregation=AggregationConfig(method="fedavg"),
        seed=42,
        n_samples=3,
        debug=DebugConfig(),
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


def generate_tabular_main_summary_scenarios() -> List[Scenario]:
    """
    Generates scenarios for the main summary figure, covering both tabular datasets
    with a fixed attack setting and sweeping core aggregators.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']  # Core aggregators for comparison

    # --- Standard Fixed Attack Settings (adjust if needed) ---
    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.3
    SYBIL_STRATEGY = 'mimic'  # Assuming 'mimic' is your standard for main figs
    # ---------------------------------------------------------

    # --- Scenario for Texas100 (MLP) ---
    scenarios.append(Scenario(
        name="main_summary_texas100",  # Match naming convention
        base_config_factory=get_base_tabular_config,
        modifiers=[
            use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER),
            use_sybil_attack(SYBIL_STRATEGY)  # Include Sybil if standard for main figs
        ],
        parameter_grid={
            "experiment.dataset_name": ["texas100"],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
            # Add other fixed params if necessary (e.g., root_gradient_source)
        }
    ))

    # --- Scenario for Purchase100 (ResNet) ---
    scenarios.append(Scenario(
        name="main_summary_purchase100",  # Match naming convention
        base_config_factory=get_base_tabular_config,  # Still uses the base
        modifiers=[
            use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER),
            use_sybil_attack(SYBIL_STRATEGY)  # Include Sybil if standard for main figs
        ],
        parameter_grid={
            "experiment.dataset_name": ["purchase100"],
            "experiment.model_structure": ["resnet"],  # <<< Specify correct model
            "experiment.tabular_model_config_name": ["resnet_purchase100_baseline"],  # <<< Specify correct config
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
            # Add other fixed params if necessary
        }
    ))

    return scenarios


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

def generate_tabular_sybil_impact_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic']
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    scenarios.append(Scenario(
        name=f"sybil_baseline_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.3],
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
                "experiment.model_structure": ["mlp"],
                "experiment.tabular_model_config_name": [model_config],
                "aggregation.method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.3],
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
         {"experiment.adv_rate": ADV_RATES_TO_SWEEP, "adversary_seller_config.poisoning.poison_rate": [0.3]}),
        ("vary_poison_rate",
         {"experiment.adv_rate": [0.3], "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP})
    ]:
        scenarios.append(Scenario(
            name=f"impact_{group_name}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER)],
            parameter_grid={
                "experiment.dataset_name": [dataset_name],
                "experiment.model_structure": ["mlp"],
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
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
        }
    ))
    return scenarios


# --- NEW SCALABILITY TEST ---
def generate_tabular_scalability_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test aggregator performance as the number of sellers
    (i.e., scalability) increases, using the user's specified list.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']

    # Sweep the total number of sellers
    N_SELLERS_TO_SWEEP = [30, 50, 70, 100]

    # Use a fixed attack setting
    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.3

    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    scenarios.append(Scenario(
        name=f"scalability_attack_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER), use_sybil_attack('mimic')],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": [model_config],

            # --- The Key Sweep Parameters ---
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.n_sellers": N_SELLERS_TO_SWEEP,

            # --- Fixed Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
        }
    ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    output_dir = "./configs_generated/tabular_fixed"
    generator = ExperimentGenerator(output_dir)

    ALL_TABULAR_SCENARIOS = []
    # --- ADD THIS LINE (e.g., at the beginning) ---
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_main_summary_scenarios())
    # ----------------------------------------------
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_sybil_impact_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_attack_impact_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_label_flipping_scenarios())
    ALL_TABULAR_SCENARIOS.extend(generate_tabular_scalability_scenarios())

    for scenario in ALL_TABULAR_SCENARIOS:
        base_config = scenario.base_config_factory()
        generator.generate(base_config, scenario)

    print(f"\n✅ All configurations for {len(ALL_TABULAR_SCENARIOS)} tabular scenarios have been generated.")
    print(f"   Configs saved to: {output_dir}")
