from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config


def disable_all_seller_attacks(config: AppConfig) -> AppConfig:
    """Modifier to ensure a clean environment with only benign sellers."""
    config.experiment.adv_rate = 0.0  # No adversarial sellers
    config.adversary_seller_config.poisoning.type = PoisonType.NONE
    config.adversary_seller_config.sybil.is_sybil = False
    # Add any other seller-side attacks you might have
    # config.adversary_seller_config.drowning_attack.is_active = False
    # config.adversary_seller_config.adaptive_attack.is_active = False
    return config


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
            sm_model_type = 'flexiblecnn' if model_name == 'cnn' else 'resnet18'
            scenarios.append(Scenario(
                name=f"main_summary_{dataset_name}_{model_name}",
                base_config_factory=get_base_image_config,
                modifiers=[dataset_modifier, use_image_backdoor_attack, use_sybil_attack('mimic')],
                parameter_grid={
                    "experiment.image_model_config_name": [f"{dataset_name}_{model_name}"],
                    "experiment.model_structure": [model_name],
                    # Use the correct list
                    "aggregation.method": IMAGE_AGGREGATORS,
                    "aggregation.sm_model_type": [sm_model_type],
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
            "aggregation.sm_model_type": ["flexiblecnn"],
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


def use_adaptive_attack(mode: str, exploration_rounds: int = 30) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the adaptive attacker in a specific mode.
    It also disables other attack types to create a clean test environment.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Activate the adaptive attack
        config.adversary_seller_config.adaptive_attack.is_active = True
        config.adversary_seller_config.adaptive_attack.attack_mode = mode
        config.adversary_seller_config.adaptive_attack.exploration_rounds = exploration_rounds

        # Deactivate other attacks to prevent interference
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        return config

    return modifier


def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test if the AdaptiveAttackerSeller can learn
    to increase its selection rate by fooling different defenses.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']
    ATTACK_MODES_TO_TEST = ["gradient_manipulation", "data_poisoning"]

    for mode in ATTACK_MODES_TO_TEST:
        scenarios.append(Scenario(
            name=f"adaptive_evasion_{mode}_cifar10_cnn",
            base_config_factory=get_base_image_config,
            # Use the new modifier to set up the attack
            modifiers=[
                use_cifar10_config,
                use_adaptive_attack(mode=mode, exploration_rounds=30)
            ],
            parameter_grid={
                # --- Fixed Parameters for a focused test ---
                "experiment.image_model_config_name": ["cifar10_cnn"],
                "experiment.model_structure": ["cnn"],
                "experiment.adv_rate": [0.3],  # 30% of sellers are the adaptive attacker

                # --- Swept Parameter ---
                # Test how the attacker performs against each defense
                "aggregation.method": AGGREGATORS_TO_TEST,
            }
        ))

    return scenarios


def use_drowning_attack(mimicry_rounds: int, drift_factor: float) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the Targeted Drowning (Centroid Poisoning) attack.
    It also disables other attack types to prevent interference.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Activate the drowning attack with specified parameters
        adv_cfg = config.adversary_seller_config.drowning_attack
        adv_cfg.is_active = True
        adv_cfg.mimicry_rounds = mimicry_rounds
        adv_cfg.drift_factor = drift_factor

        # Deactivate other attacks to ensure the experiment is isolated
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        config.adversary_seller_config.adaptive_attack.is_active = False
        return config

    return modifier


# ============================================================================
# 🆕 NEW: Competitor Mimicry Attack (Replaces Drowning Attack)
# ============================================================================

def use_competitor_mimicry_attack(
        target_seller_id: str = "seller_0",
        strategy: str = "noisy_copy",
        noise_scale: float = 0.03,
        observation_rounds: int = 5
) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the Competitor Mimicry attack.
    This attack is simpler and more effective than the drowning attack.

    Args:
        target_seller_id: Which seller to mimic (steal market share from)
        strategy: "exact_copy", "noisy_copy", "scaled_copy", or "averaged_history"
        noise_scale: Amount of noise for noisy_copy/scaled_copy strategies
        observation_rounds: Rounds to observe target before attacking
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Activate the mimicry attack
        adv_cfg = config.adversary_seller_config.mimicry_attack
        adv_cfg.is_active = True
        adv_cfg.target_seller_id = target_seller_id
        adv_cfg.strategy = strategy
        adv_cfg.noise_scale = noise_scale
        adv_cfg.observation_rounds = observation_rounds

        # Deactivate other attacks
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        config.adversary_seller_config.adaptive_attack.is_active = False
        config.adversary_seller_config.drowning_attack.is_active = False
        return config

    return modifier


def generate_competitor_mimicry_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test the CompetitorMimicrySeller's ability
    to steal market share from a high-quality target seller.

    Success Metric: Target's selection rate drops while attacker's increases.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']
    MIMICRY_STRATEGIES = ["noisy_copy"]
    ADV_RATES = [0.2, 0.3, 0.4]  # See how multiple attackers amplify the effect

    for strategy in MIMICRY_STRATEGIES:
        scenarios.append(Scenario(
            name=f"competitor_mimicry_{strategy}_cifar10_cnn",
            base_config_factory=get_base_image_config,
            modifiers=[
                use_cifar10_config,
                use_competitor_mimicry_attack(
                    target_seller_id="bn_0",  # Target the first (often high-quality) seller
                    strategy=strategy,
                    noise_scale=0.03,
                    observation_rounds=5
                )
            ],
            parameter_grid={
                "experiment.image_model_config_name": ["cifar10_cnn"],
                "experiment.model_structure": ["cnn"],
                "aggregation.method": AGGREGATORS_TO_TEST,
                "experiment.adv_rate": ADV_RATES,
            }
        ))

    return scenarios


# ============================================================================
# 🔧 KEEP: Drowning Attack (Now marked as legacy comparison)
# ============================================================================


def generate_drowning_attack_scenarios() -> List[Scenario]:
    """
    [LEGACY] Generates scenarios to test the DrowningAttackerSeller.
    Kept for comparison purposes.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4]

    scenarios.append(Scenario(
        name="drowning_attack_cifar10_cnn_legacy",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_drowning_attack(mimicry_rounds=10, drift_factor=0.1)
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
        }
    ))

    return scenarios


# ============================================================================
# 🆕 NEW: Enhanced Buyer Attack Scenarios
# ============================================================================

def use_buyer_dos_attack() -> Callable[[AppConfig], AppConfig]:
    """Enable buyer DoS attack (zero-gradient)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "dos"
        return config

    return modifier


def use_buyer_starvation_attack(target_classes: List[int]) -> Callable[[AppConfig], AppConfig]:
    """Enable buyer starvation attack (focus on specific classes)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "starvation"
        config.buyer_attack_config.starvation_classes = target_classes
        return config

    return modifier


def use_buyer_erosion_attack() -> Callable[[AppConfig], AppConfig]:
    """Enable buyer trust erosion attack (random gradients)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "erosion"
        return config

    return modifier


def use_buyer_class_exclusion_attack(
        exclude_classes: List[int] = None,
        target_classes: List[int] = None,
        gradient_scale: float = 1.0
) -> Callable[[AppConfig], AppConfig]:
    """
    🆕 Enable buyer class-based exclusion attack.
    More realistic replacement for orthogonal_pivot.

    Args:
        exclude_classes: Classes to exclude from buyer baseline (negative selection)
        target_classes: Classes to include in buyer baseline (positive selection)
        gradient_scale: Amplification factor for bias (>1.0 increases bias)
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "class_exclusion"

        if exclude_classes:
            config.buyer_attack_config.exclusion_exclude_classes = exclude_classes
        if target_classes:
            config.buyer_attack_config.exclusion_target_classes = target_classes

        config.buyer_attack_config.exclusion_gradient_scale = gradient_scale
        config.buyer_attack_config.num_classes = 10  # Adjust based on dataset
        return config

    return modifier


def use_buyer_oscillating_attack(
        strategy: str = "binary_flip",
        period: int = 2,
        classes_a: List[int] = None,
        classes_b: List[int] = None,
        subset_size: int = 3,
        drift_rounds: int = 50
) -> Callable[[AppConfig], AppConfig]:
    """
    🆕 Enable buyer oscillating objective attack.

    Args:
        strategy: "binary_flip", "rotating", "random_walk", or "adversarial_drift"
        period: Rounds before switching objectives (for binary_flip/rotating)
        classes_a: Phase A classes (for binary_flip/adversarial_drift)
        classes_b: Phase B classes (for binary_flip)
        subset_size: Number of random classes per round (for random_walk)
        drift_rounds: Total rounds for drift cycle (for adversarial_drift)
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "oscillating"
        config.buyer_attack_config.oscillation_strategy = strategy
        config.buyer_attack_config.oscillation_period = period
        config.buyer_attack_config.num_classes = 10  # Adjust based on dataset

        if classes_a:
            config.buyer_attack_config.oscillation_classes_a = classes_a
        else:
            config.buyer_attack_config.oscillation_classes_a = [0, 1, 2, 3, 4]

        if classes_b:
            config.buyer_attack_config.oscillation_classes_b = classes_b
        else:
            config.buyer_attack_config.oscillation_classes_b = [5, 6, 7, 8, 9]

        config.buyer_attack_config.oscillation_subset_size = subset_size
        config.buyer_attack_config.oscillation_drift_total_rounds = drift_rounds

        return config

    return modifier


def use_buyer_orthogonal_pivot_attack(target_seller_id: str) -> Callable[[AppConfig], AppConfig]:
    """
    [LEGACY] Enable buyer orthogonal pivot attack.
    NOTE: Consider using class_exclusion_attack for more realistic exclusion.
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "orthogonal_pivot"
        config.buyer_attack_config.target_seller_id = target_seller_id
        return config

    return modifier


def generate_buyer_attack_scenarios() -> List[Scenario]:
    """
    🔧 UPDATED: Generates scenarios to test ALL buyer-side attacks.
    Now includes the new class_exclusion and oscillating attacks.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']

    # --- Attack 1: DoS ---
    scenarios.append(Scenario(
        name="buyer_attack_dos_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, disable_all_seller_attacks, use_buyer_dos_attack()],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- Attack 2: Starvation ---
    scenarios.append(Scenario(
        name="buyer_attack_starvation_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_starvation_attack(target_classes=[0, 1])
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- Attack 3: Trust Erosion ---
    scenarios.append(Scenario(
        name="buyer_attack_erosion_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, disable_all_seller_attacks, use_buyer_erosion_attack()],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- 🆕 Attack 4: Class-Based Exclusion (Exclude sellers with classes 7,8,9) ---
    scenarios.append(Scenario(
        name="buyer_attack_class_exclusion_negative_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_class_exclusion_attack(exclude_classes=[7, 8, 9], gradient_scale=1.2)
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- 🆕 Attack 5: Class-Based Exclusion (Only favor sellers with classes 0,1,2) ---
    scenarios.append(Scenario(
        name="buyer_attack_class_exclusion_positive_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_class_exclusion_attack(target_classes=[0, 1, 2], gradient_scale=1.0)
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- 🆕 Attack 6: Oscillating (Binary Flip) ---
    scenarios.append(Scenario(
        name="buyer_attack_oscillating_binary_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="binary_flip",
                period=2,
                classes_a=[0, 1, 2, 3, 4],
                classes_b=[5, 6, 7, 8, 9]
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- 🆕 Attack 7: Oscillating (Random Walk) ---
    scenarios.append(Scenario(
        name="buyer_attack_oscillating_random_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="random_walk",
                subset_size=3
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- 🆕 Attack 8: Oscillating (Adversarial Drift) ---
    scenarios.append(Scenario(
        name="buyer_attack_oscillating_drift_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="adversarial_drift",
                drift_rounds=60,
                classes_a=[0, 1]  # Final poisoned focus
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    # --- [LEGACY] Attack 9: Orthogonal Pivot (kept for comparison) ---
    scenarios.append(Scenario(
        name="buyer_attack_orthogonal_pivot_legacy_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_orthogonal_pivot_attack(target_seller_id="bn_5")
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
        }
    ))

    return scenarios


# ============================================================================
# 🆕 NEW: Comparative Analysis Scenarios
# ============================================================================

def generate_attack_comparison_scenarios() -> List[Scenario]:
    """
    Generates scenarios to directly compare old vs. new attack effectiveness.
    Useful for validating that the new attacks are improvements.
    """
    scenarios = []

    # Compare Drowning vs. Competitor Mimicry
    scenarios.append(Scenario(
        name="comparison_drowning_vs_mimicry_martfl",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": ["martfl"],  # Focus on MartFL as it's most vulnerable
            "experiment.adv_rate": [0.3],
            # Sweep between the two attack types
            "adversary_seller_config.drowning_attack.is_active": [True, False],
            "adversary_seller_config.mimicry_attack.is_active": [False, True],
        }
    ))

    # Compare Orthogonal Pivot vs. Class Exclusion
    scenarios.append(Scenario(
        name="comparison_orthogonal_vs_class_exclusion_fltrust",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, disable_all_seller_attacks],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": ["fltrust"],  # Focus on FLTrust
            # Sweep between buyer attack types
            "buyer_attack_config.is_active": [True],
            "buyer_attack_config.attack_type": ["orthogonal_pivot", "class_exclusion"],
            "buyer_attack_config.exclusion_exclude_classes": [[7, 8, 9]],
        }
    ))

    return scenarios


# ============================================================================
# Updated ALL_SCENARIOS List
# ============================================================================
# ============================================================================
# 🆕 NEW: Attack Scalability Analysis (Rate-Based)
# ============================================================================

def generate_attack_scalability_scenarios_OLD() -> List[Scenario]:
    """
    Generates scenarios to test how attack effectiveness scales with marketplace size.
    Uses a FIXED adversary rate (30%) to maintain consistent attacker proportion.

    Key Research Questions:
    1. Does 30% attackers remain equally effective as the market grows?
    2. How does market size affect the absolute impact of the same proportion of attackers?
    3. Do defenses become more/less robust at larger scales?

    Tests both seller-side (Competitor Mimicry) and buyer-side (Class Exclusion) attacks.
    """
    scenarios = []

    # Sweep marketplace sizes from small (10 sellers) to large (100 sellers)
    # With adv_rate=0.3: 10 sellers = 3 attackers, 100 sellers = 30 attackers
    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]

    # Fixed adversary rate across all scales
    FIXED_ADV_RATE = 0.3  # 30% of sellers are attackers

    # Focus on defenses most vulnerable to these attacks
    AGGREGATORS_TO_TEST = ['fltrust', 'martfl']

    # --- Scenario 1: Seller-Side Attack Scalability (Competitor Mimicry) ---
    scenarios.append(Scenario(
        name="scalability_competitor_mimicry_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",  # Always target the first seller
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Adversary Rate (30% at all scales) ---
            "experiment.adv_rate": [FIXED_ADV_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            # Keep these fixed for consistency
            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 2: Buyer-Side Attack Scalability (Class Exclusion) ---
    scenarios.append(Scenario(
        name="scalability_buyer_class_exclusion_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,  # Pure buyer attack, no seller interference
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],  # Exclude sellers with these classes
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            # Keep these fixed
            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 3: Oscillating Buyer Attack Scalability ---
    # This is interesting because buyer attacks should be scale-independent
    scenarios.append(Scenario(
        name="scalability_buyer_oscillating_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="binary_flip",
                period=2,
                classes_a=[0, 1, 2, 3, 4],
                classes_b=[5, 6, 7, 8, 9]
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 4: Combined Attack Scalability (Both attacks active) ---
    scenarios.append(Scenario(
        name="scalability_combined_attacks_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            # Enable BOTH attacks simultaneously
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            ),
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Adversary Rate ---
            "experiment.adv_rate": [FIXED_ADV_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 OPTIONAL: Multi-Rate Comparison (if you want to compare rates later)
# ============================================================================

def generate_attack_scalability_multirate_scenarios() -> List[Scenario]:
    """
    OPTIONAL: Compare how different adversary rates perform at multiple scales.
    Use this if you want to study the interaction between rate and scale.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 30, 50, 100]  # Fewer sizes to keep experiments manageable
    ADVERSARY_RATES = [0.1, 0.2, 0.3, 0.4]  # Compare multiple rates

    scenarios.append(Scenario(
        name="scalability_multirate_competitor_mimicry_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- SWEEP BOTH: Size and Rate ---
            "experiment.n_sellers": MARKETPLACE_SIZES,
            "experiment.adv_rate": ADVERSARY_RATES,

            # --- Focus on most vulnerable defense ---
            "aggregation.method": ["martfl"],

            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


def generate_attack_scalability_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test how attack effectiveness scales with marketplace size.
    Uses BACKDOOR attack for sellers (matching main_summary_figure style).
    Uses a FIXED adversary rate (30%) to maintain consistent attacker proportion.

    Key Research Questions:
    1. Does 30% backdoor attackers remain equally effective as the market grows?
    2. How does market size affect backdoor success rate (ASR)?
    3. Do defenses become more/less robust at larger scales?

    Tests both seller-side (Backdoor + Sybil) and buyer-side (Class Exclusion) attacks.
    """
    scenarios = []

    # Sweep marketplace sizes from small (10 sellers) to large (100 sellers)
    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]

    # Fixed attack parameters (matching main_summary style)
    FIXED_ADV_RATE = 0.3  # 30% of sellers are attackers
    FIXED_POISON_RATE = 0.5  # 50% of attacker's data is poisoned

    # Focus on defenses most vulnerable to these attacks
    IMAGE_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # --- Scenario 1: Seller-Side Backdoor Attack Scalability ---
    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,  # Image backdoor (10x10 white patch)
            use_sybil_attack('mimic')  # Sybil coordination to evade detection
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.sm_model_type": ["flexiblecnn"],  # For SkyMask

            # Keep these fixed for consistency
            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 2: Backdoor Attack Scalability with ResNet (More Complex Model) ---
    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar10_resnet18",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_resnet18"],
            "experiment.model_structure": ["resnet18"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.sm_model_type": ["resnet18"],

            "experiment.num_rounds": [100],

        }
    ))

    # --- Scenario 3: Buyer-Side Attack Scalability (Class Exclusion) ---
    # This tests if buyer attacks remain scale-independent
    scenarios.append(Scenario(
        name="scalability_buyer_class_exclusion_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,  # Pure buyer attack, no seller interference
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],  # Exclude sellers with these classes
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.sm_model_type": ["flexiblecnn"],

            # Keep these fixed
            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 4: Buyer Oscillating Attack Scalability ---
    scenarios.append(Scenario(
        name="scalability_buyer_oscillating_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="binary_flip",
                period=2,
                classes_a=[0, 1, 2, 3, 4],
                classes_b=[5, 6, 7, 8, 9]
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.sm_model_type": ["flexiblecnn"],

            "experiment.num_rounds": [100],
        }
    ))

    # --- Scenario 5: Combined Attack Scalability (Backdoor + Class Exclusion) ---
    scenarios.append(Scenario(
        name="scalability_combined_backdoor_buyer_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic'),
            # ALSO enable buyer attack
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Seller Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.sm_model_type": ["flexiblecnn"],

            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 OPTIONAL: Text Dataset Scalability (TREC)
# ============================================================================

def generate_text_scalability_scenarios() -> List[Scenario]:
    """
    Test scalability on text datasets (TREC) with text backdoor attack.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 20, 30, 50]  # Fewer sizes for text (smaller dataset)
    TEXT_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']  # No SkyMask for text

    scenarios.append(Scenario(
        name="scalability_backdoor_trec",
        base_config_factory=get_base_text_config,
        modifiers=[
            use_trec_config,
            use_text_backdoor_attack  # Text trigger (e.g., "cf" token)
        ],
        parameter_grid={
            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": TEXT_AGGREGATORS,

            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 NEW: Extreme Scale Test (Stress Testing)
# ============================================================================

def generate_extreme_scale_scenarios() -> List[Scenario]:
    """
    Stress test: Can backdoor attacks work in VERY large marketplaces (200-500 sellers)?
    Maintains 30% adversary rate even at extreme scales.
    """
    scenarios = []

    EXTREME_SIZES = [100, 200, 300, 500]
    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.5

    # Test only the most vulnerable defense (faster experiments)
    scenarios.append(Scenario(
        name="extreme_scale_backdoor_martfl",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- Extreme marketplace sizes ---
            "experiment.n_sellers": EXTREME_SIZES,

            # --- Fixed: 30% adversary rate, 50% poison ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- Fixed: Most vulnerable defense ---
            "aggregation.method": ["martfl"],

            # Reduce rounds for faster experiments
            "experiment.num_rounds": [50],
        }
    ))

    scenarios.append(Scenario(
        name="extreme_scale_buyer_class_exclusion_fltrust",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            "experiment.n_sellers": EXTREME_SIZES,
            "aggregation.method": ["fltrust"],

            "experiment.num_rounds": [50],
        }
    ))

    return scenarios


def generate_baseline_scalability_scenarios() -> List[Scenario]:
    """
    Test how defenses perform at different scales WITHOUT any attacks.
    Essential baseline for comparison.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    scenarios.append(Scenario(
        name="scalability_baseline_no_attack_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks  # Pure benign
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- NO ATTACKS ---
            "experiment.adv_rate": [0.0],

            # --- Test all defenses ---
            "aggregation.method": ALL_AGGREGATORS,
            "aggregation.sm_model_type": ["flexiblecnn"],

            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# Add to ALL_SCENARIOS
# ============================================================================
def generate_cifar100_scalability_scenarios() -> List[Scenario]:
    """Test scalability on more complex dataset (CIFAR-100)"""
    scenarios = []

    MARKETPLACE_SIZES = [10, 30, 50, 100]  # Fewer sizes for efficiency

    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar100_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar100_config,  # Different dataset
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar100_cnn"],
            "experiment.model_structure": ["cnn"],
            "experiment.n_sellers": MARKETPLACE_SIZES,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.5],
            "aggregation.method": ['fedavg', 'fltrust', 'martfl'],  # Fewer for speed
            "experiment.num_rounds": [100],
        }
    ))

    return scenarios


# Add to ALL_SCENARIOS
ALL_SCENARIOS = []
ALL_SCENARIOS.extend(generate_cifar100_scalability_scenarios())

# ... [Keep all your existing scenario generators] ...

# 🆕 Add scalability scenarios with BACKDOOR attacks
ALL_SCENARIOS.extend(generate_attack_scalability_scenarios())
# Add to ALL_SCENARIOS:
ALL_SCENARIOS.extend(generate_baseline_scalability_scenarios())

# 🆕 OPTIONAL: Add text scalability
ALL_SCENARIOS.extend(generate_text_scalability_scenarios())

# 🆕 Add extreme scale testing
ALL_SCENARIOS.extend(generate_extreme_scale_scenarios())

ALL_SCENARIOS.extend(generate_oracle_vs_buyer_bias_scenarios())
ALL_SCENARIOS.extend(generate_adv_rate_trend_scenarios())
ALL_SCENARIOS.extend(generate_sybil_impact_scenarios())
ALL_SCENARIOS.extend(generate_data_heterogeneity_scenarios())
ALL_SCENARIOS.extend(generate_main_summary_figure_scenarios())
ALL_SCENARIOS.extend(generate_label_flipping_scenarios())
ALL_SCENARIOS.extend(generate_sybil_selection_rate_scenarios())
ALL_SCENARIOS.extend(generate_buyer_data_impact_scenarios())

ALL_SCENARIOS.extend(generate_competitor_mimicry_scenarios())  # Replaces/augments drowning
ALL_SCENARIOS.extend(generate_buyer_attack_scenarios())  # Updated with new attacks

# 3. 🔧 LEGACY: Keep for comparison
ALL_SCENARIOS.extend(generate_drowning_attack_scenarios())  # Marked as legacy

# 4. 🆕 NEW: Comparative analysis
ALL_SCENARIOS.extend(generate_attack_comparison_scenarios())  # Direct comparisons

ALL_SCENARIOS.extend(generate_adaptive_attack_scenarios())
if __name__ == '__main__':
    print(f"Generated {len(ALL_SCENARIOS)} focused scenarios:")
    for s in ALL_SCENARIOS:
        print(f"  - {s.name}")

# total number of sellers
