import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List, Union, Callable, Literal

from torch.utils.data import Dataset

from common.enums import TextTriggerLocation, ImageTriggerType, ImageTriggerLocation, PoisonType, LabelFlipMode, \
    ImageBackdoorAttackName, TextBackdoorAttackName

logger = logging.getLogger("Configs")


@dataclass
class MimicryAttackConfig:
    """Configuration for Direct Competitor Mimicry Attack"""
    is_active: bool = False
    target_seller_id: str = "bn_3"  # Which seller to mimic
    observation_rounds: int = 3  # How many rounds to observe before attacking
    noise_scale: float = 0.05  # Noise level for noisy_copy strategy
    strategy: str = "noisy_copy"  # Options: "exact_copy", "noisy_copy", "scaled_copy", "averaged_history"


@dataclass
class DebugConfig:
    """Configuration for debugging and detailed logging."""
    save_individual_gradients: bool = False
    gradient_save_frequency: int = 10  # Save every round by default if enabled


@dataclass
class ExperimentConfig:
    """Primary configuration for the overall experiment run."""
    dataset_name: str
    model_structure: str
    global_rounds: int
    n_sellers: int
    adv_rate: float
    device: str
    eval_frequency: int = 10
    compute_gradient_similarity: bool = True  # Add this line
    save_path: str = "./results"
    num_classes: int = 0  # Default to 0, will be set dynamically at runtime
    use_subset: bool = False
    subset_size: int = 3000  # Number of samples to use in the subset
    dataset_type: str = "text"
    evaluations: List[str] = field(default_factory=lambda: ["clean"])
    image_model_config_name: str = "cifar10_cnn"
    tabular_model_config_name: str = "mlp_texas100_baseline"  # Default model config to use
    use_early_stopping: bool = True
    patience: int = 10


@dataclass
class TrainingConfig:
    """Holds parameters specifically for local seller training."""
    local_epochs: int = 2  # Common default for FL
    batch_size: int = 64  # Common default batch size
    learning_rate: float = 0.001  # Sensible default, especially if Adam is common
    optimizer: str = "Adam"  # Default optimizer
    momentum: float = 0.9  # Only used by SGD
    weight_decay: float = 0.0001
    eps: float = 1e-4


@dataclass
class BackdoorSimpleDataPoisonParams:
    target_label: int = 0
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    trigger_shape: Tuple[int, int] = (10, 10)
    strength: float = 1
    pattern_channel: int = 3


@dataclass
class ImageBackdoorParams:
    """Container for all possible image backdoor attack configurations."""
    # This field determines which of the sub-configurations is active
    attack_name: ImageBackdoorAttackName = ImageBackdoorAttackName.SIMPLE_DATA_POISON
    # It holds an instance of every possible sub-configuration
    simple_data_poison_params: BackdoorSimpleDataPoisonParams = field(default_factory=BackdoorSimpleDataPoisonParams)

    @property
    def active_attack_params(self) -> Union[BackdoorSimpleDataPoisonParams]:
        """
        Returns the active parameter object based on the 'attack_name' field.
        """
        if self.attack_name == ImageBackdoorAttackName.SIMPLE_DATA_POISON:
            return self.simple_data_poison_params
        raise ValueError(f"Unknown image backdoor attack name: {self.attack_name}")


@dataclass
class TextBackdoorParams:
    """Parameters specific to a TEXT backdoor attack."""
    target_label: int = 0
    trigger_content: str = "cf"  # The trigger phrase
    location: TextTriggerLocation = TextTriggerLocation.END
    attack_name: TextBackdoorAttackName = TextBackdoorAttackName.SIMPLE_DATA_POISON


@dataclass
class BuyerAttackConfig:
    """Configuration for malicious buyer attacks"""
    is_active: bool = False
    attack_type: str = "none"  # "dos", "starvation", "erosion", "class_exclusion", "oscillating", "orthogonal_pivot"
    target_seller_id = "bn_5"
    # --- Starvation Attack ---
    starvation_classes: List[int] = field(default_factory=list)

    # --- 🆕 Class-Based Exclusion ---
    exclusion_target_classes: List[int] = field(default_factory=list)
    exclusion_exclude_classes: List[int] = field(default_factory=list)
    exclusion_gradient_scale: float = 1.0

    # --- 🆕 Oscillating Objective ---
    oscillation_strategy: str = "binary_flip"  # "binary_flip", "rotating", "random_walk", "adversarial_drift"
    oscillation_period: int = 2
    oscillation_classes_a: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    oscillation_classes_b: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    oscillation_class_subsets: List[List[int]] = field(default_factory=lambda: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    num_classes: int = 10  # 🆕 ADD THIS (should match dataset)
    oscillation_subset_size: int = 3
    oscillation_drift_total_rounds: int = 50


@dataclass
class TabularFeatureTriggerParams:
    """Parameters for a feature-based trigger backdoor attack."""
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)


# --- Step 2: Create the container, just like for image and text ---

class TabularBackdoorAttackName(Enum):
    """Enum to define the different types of tabular backdoor attacks."""
    FEATURE_TRIGGER = "feature_trigger"


@dataclass
class TabularBackdoorParams:
    """Container for all possible tabular backdoor attack configurations."""
    # This field determines which of the sub-configurations is active
    attack_name: TabularBackdoorAttackName = TabularBackdoorAttackName.FEATURE_TRIGGER
    target_label: int = 0

    # It holds an instance of every possible sub-configuration
    feature_trigger_params: TabularFeatureTriggerParams = field(default_factory=TabularFeatureTriggerParams)

    @property
    def active_attack_params(self) -> Union[TabularFeatureTriggerParams]:
        """
        Returns the active parameter object based on the 'attack_name' field.
        """
        if self.attack_name == TabularBackdoorAttackName.FEATURE_TRIGGER:
            return self.feature_trigger_params
        raise ValueError(f"Unknown tabular backdoor attack name: {self.attack_name}")


@dataclass
class LabelFlipParams:
    """Parameters specific to the label-flipping poisoning attack."""
    target_label: int = 0
    mode: LabelFlipMode = LabelFlipMode.FIXED_TARGET


@dataclass
class PoisoningConfig:
    """Configuration for client-side data poisoning attacks."""
    type: PoisonType = PoisonType.NONE
    poison_rate: float = 0.1
    image_backdoor_params: ImageBackdoorParams = field(default_factory=ImageBackdoorParams)
    text_backdoor_params: TextBackdoorParams = field(default_factory=TextBackdoorParams)
    tabular_backdoor_params: TabularBackdoorParams = field(default_factory=TabularBackdoorParams)
    label_flip_params: LabelFlipParams = field(default_factory=LabelFlipParams)

    @property
    def active_params(self) -> Union[  # Add all possible return types here
        BackdoorSimpleDataPoisonParams,
        TextBackdoorParams,
        TabularBackdoorParams,
        LabelFlipParams,
        None
    ]:
        """
        Returns the active parameter object based on the main 'type' field.
        """
        if self.type == PoisonType.IMAGE_BACKDOOR:
            return self.image_backdoor_params.active_attack_params

        # --- FIX 2: Made text_backdoor consistent with the container pattern ---
        elif self.type == PoisonType.TEXT_BACKDOOR:
            return self.text_backdoor_params

        # --- FIX 3: Added the missing case for tabular backdoor ---
        elif self.type == PoisonType.TABULAR_BACKDOOR:
            return self.tabular_backdoor_params

        elif self.type == PoisonType.LABEL_FLIP:
            return self.label_flip_params

        elif self.type == PoisonType.NONE:
            return None

        raise ValueError(f"Unknown poison type: {self.type}")


# --- Create a specific parameter class for GIA ---
@dataclass
class GradientInversionParams:
    """Parameters for the Gradient Inversion Attack."""
    frequency: int = 10
    victim_strategy = 'random'
    fixed_victim_idx: int = 0
    lrs_to_try: List[float] = field(default_factory=lambda: [0.1, 0.01])
    base_attack_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerAttackConfig:
    """Configuration for all server-side privacy attacks."""
    # This is the main switch: 'none', 'gradient_inversion', etc.
    attack_name: str = 'none'

    gradient_inversion_params: GradientInversionParams = field(default_factory=GradientInversionParams)


@dataclass
class DrowningAttackConfig:
    """Configuration for the Targeted Drowning (Centroid Poisoning) Attack."""
    is_active: bool = False
    mimicry_rounds: int = 10  # Number of rounds to act honestly to build trust
    drift_factor: float = 0.1  # How much to shift the gradient each drift round


@dataclass
class SybilDrowningConfig:
    """Configuration for the Targeted Drowning Attack."""
    target_victim_id: str = "bn_3"
    attack_strength: float = 1.0


@dataclass
class SybilConfig:
    """Configuration for Sybil attack coordination and behavior."""
    is_sybil: bool = False  # Is Sybil coordination active?
    benign_rounds: int = 0  # Rounds before Sybils start manipulating (0 = immediate attack)

    # --- Strategy Control ---
    # Specifies the primary manipulation strategy used by Sybils.
    # Options: "mimic", "pivot", "knock_out", "oracle_blend", "systematic_probe"
    # If None, might fall back to role-based strategies (less common now).
    gradient_default_mode: Optional[str] = "mimic"

    # --- Oracle Blend Specific ---
    # Blending factor alpha for oracle_blend strategy.
    # G_submit = alpha * G_mal + (1 - alpha) * G_oracle_centroid
    oracle_blend_alpha: float = 0.1  # Example: 10% malicious intent

    # --- Historical Mimicry Specific ---
    # Number of past rounds to consider for calculating historical centroid.
    history_window_size: int = 5  # Reduced default example

    # --- Role Assignment (Less critical if gradient_default_mode is set) ---
    # Defines proportions for dynamic role assignment (attacker, hybrid, explorer).
    # Used if adaptive_role_assignment is called and gradient_default_mode is None.
    role_config: Dict[str, float] = field(default_factory=lambda: {'attacker': 0.2, 'explorer': 0.4})

    # --- Strategy-Specific Parameters ---
    # Holds keyword arguments for specific strategy classes (e.g., MimicStrategy).
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Triggering (Keep if used elsewhere) ---
    # Controls how/when the attack activates (e.g., 'static' after benign_rounds).
    trigger_mode: str = "static"

    # --- REMOVED ---
    detection_threshold: float = 0.8  # No longer needed with simplified phase logic


@dataclass
class VocabConfig:
    """Configuration for building the torchtext vocabulary."""
    min_freq: int = 1
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"


@dataclass
class PropertySkewParams:
    """Parameters for the 'property-skew' image data splitting strategy."""
    property_key: str = "Blond_Hair"
    num_high_prevalence_clients: int = 2
    num_security_attackers: int = 2
    high_prevalence_ratio: float = 0.8
    low_prevalence_ratio: float = 0.1
    standard_prevalence_ratio: float = 0.4
    dirichlet_alpha: float = 1.0  # <--- ADD THIS LINE


@dataclass
class TextPropertySkewParams:
    """Parameters for the 'property-skew' text data splitting strategy."""
    property_key: str = "software"  # The keyword/phrase to search for
    num_high_prevalence_clients: int = 2
    num_security_attackers: int = 2
    high_prevalence_ratio: float = 0.8
    low_prevalence_ratio: float = 0.1
    standard_prevalence_ratio: float = 0.4


@dataclass
class AdaptiveAttackConfig:
    """Configuration for the adaptive learning attacker."""
    is_active: bool = False

    # --- General Learning Parameters ---

    # Set a valid default and use Literal for type safety
    threat_model: Literal["black_box", "gradient_inversion", "oracle"] = "black_box"

    # The primary mode of attack
    attack_mode: Literal["gradient_manipulation", "data_poisoning"] = "gradient_manipulation"

    exploration_rounds: int = 20

    # --- Threat Model Specific ---

    # [FIX] ADDED: Required by Oracle and Gradient Inversion models
    mimic_strength: float = 0.5

    # --- Gradient Manipulation Strategies ---

    # [FIX] RENAMED "scale_up" to "reduce_norm" to match your class
    gradient_strategies: List[str] = field(
        default_factory=lambda: ["honest", "reduce_norm", "add_noise", "stealthy_blend"]
    )
    scale_factor: float = 0.5  # A value < 1.0 matches "reduce_norm"
    noise_level: float = 0.01

    # --- Data Poisoning Strategies ---

    data_strategies: List[str] = field(
        default_factory=lambda: ["honest", "subsample_clean"]
    )

    # [FIX] ADDED: Required by "subsample_clean" strategy
    subset_ratio: float = 0.5

@dataclass
class AdversarySellerConfig:
    """A unified profile for an adversarial seller."""
    name: str = "benign"
    poisoning: PoisoningConfig = field(default_factory=PoisoningConfig)
    sybil: SybilConfig = field(default_factory=SybilConfig)
    adaptive_attack: AdaptiveAttackConfig = field(default_factory=AdaptiveAttackConfig)
    drowning_attack: DrowningAttackConfig = field(default_factory=DrowningAttackConfig)
    mimicry_attack: MimicryAttackConfig = field(default_factory=MimicryAttackConfig)  # <-- Add this


@dataclass
class TextDataConfig:
    """All settings related to a text dataset source."""
    vocab: VocabConfig = VocabConfig()
    strategy: str = "dirichlet"  # Strategy for SELLERS (Changed default)
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS (Added)

    property_skew: Optional[TextPropertySkewParams] = None  # Keep if used

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1  # Renamed from buyer_config for consistency
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Image Data Configuration ---
@dataclass
class ImageDataConfig:
    """All settings related to an image dataset source."""
    strategy: str = "dirichlet"  # Strategy for SELLERS (Defaulting to Dirichlet)
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS
    property_skew: Optional[PropertySkewParams] = None  # Keep if you use property skew sometimes

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Tabular Data Configuration ---
@dataclass
class TabularDataConfig:
    """All settings related to a tabular dataset source."""
    dataset_config_path: str = "configs/tabular_datasets.yaml"
    model_config_dir: str = "model/configs"

    strategy: str = "dirichlet"  # Strategy for SELLERS (Changed default from 'iid')
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS (Added, assuming you have this param)
    property_skew: Optional[PropertySkewParams] = None  # Keep if you use property skew sometimes

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Main Data Configuration ---
@dataclass
class DataConfig:
    """Holds configuration for one type of data source."""
    text: Optional[TextDataConfig] = TextDataConfig()
    image: Optional[ImageDataConfig] = ImageDataConfig()
    tabular: Optional[TabularDataConfig] = TabularDataConfig()
    num_workers: int = 2  # Keep num_workers here if it applies globally


@dataclass
class RuntimeDataConfig:
    """Holds runtime data objects passed to sellers."""
    dataset: Dataset
    num_classes: int
    collate_fn: Optional[Callable] = None


# --- These are RUNTIME configs, not loaded from YAML. Keeping them is correct. ---
@dataclass
class LabelFlipConfig:
    """Configuration for the LabelFlipGenerator."""
    num_classes: int
    attack_mode: str = "fixed_target"
    target_label: int = 0


@dataclass
class BackdoorImageConfig:
    """Configuration for an image backdoor trigger."""
    target_label: int
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    channels: int = 3
    trigger_size: Tuple[int, int] = (5, 5)
    blend_alpha: float = 0.2
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    randomize_location: bool = False


@dataclass
class BackdoorTextConfig:
    """Configuration for the BackdoorTextGenerator."""
    vocab: Any  # Expects an object with a .stoi attribute
    target_label: int
    trigger_content: str = "cf"
    location: TextTriggerLocation = TextTriggerLocation.END
    max_seq_len: Optional[int] = None


@dataclass
class BackdoorTabularConfig:
    """Configuration for a feature-based tabular backdoor attack."""
    target_label: int
    # Defines the trigger, e.g., {'feature_name_A': 1.0, 'feature_name_B': 0.0}
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MartFLParams:
    """Parameters specific to the martFL aggregator."""
    # --- Existing Parameters ---
    change_base: bool = True
    # If True, rotates baseline seller based on kappa score.
    # If False, always uses the initial_baseline (like FLTrust).

    clip: bool = True
    # If True, uses server-side clipping with the clip_norm value
    # defined in the main AggregationConfig before similarity calculation.

    # --- New Tunable Parameters ---
    initial_baseline: str = "buyer"
    # Specifies the ID used as the baseline anchor in the *first* round
    # (and subsequent rounds if change_base is False).
    # Common values: "buyer" or a specific client ID like "bn_3".

    max_k: int = 10
    # The maximum number of clusters ('k') that the k-means algorithm
    # (in _cluster_and_score_martfl) will consider when detecting outliers.


@dataclass
class SkymaskParams:
    """Parameters specific to the Skymask aggregator."""
    clip: bool = True
    sm_model_type: str = 'cnn'
    mask_epochs: int = 20
    mask_lr: float = 1e-4
    mask_clip: float = 1.0
    mask_threshold: float = 0.5


@dataclass
class AggregationConfig:
    """Top-level configuration for aggregation."""
    # This is the main switch to select the algorithm
    method: str = "martfl"

    # A common parameter used by multiple methods
    clip_norm: Optional[float] = 0.01

    # --- Nested parameter objects for each strategy ---
    martfl: MartFLParams = field(default_factory=MartFLParams)
    skymask: SkymaskParams = field(default_factory=SkymaskParams)


@dataclass
class ValuationConfig:
    """Configuration for valuation methods."""

    # --- Default, Always-On Methods ---
    run_similarity: bool = True
    # ^ Note: This is for config clarity. The manager
    # will always run this cheap, default evaluator.

    # --- Optional Expensive Methods ---
    run_influence: bool = False  # Run fast influence function
    run_loo: bool = False  # Run per-round Leave-One-Out
    loo_frequency: int = 10  # Run LOO every N rounds

    # --- NEW KERNELSHAP (LINEAR MODEL) ---
    run_kernelshap: bool = False  # Run KernelSHAP approximation
    kernelshap_frequency: int = 10
    kernelshap_samples: int = 32  # Number of coalitions to sample (e.g., 2*N + 8)


@dataclass
class AppConfig:
    """Top-level configuration for the entire experiment run."""
    experiment: ExperimentConfig
    training: TrainingConfig
    server_attack_config: ServerAttackConfig
    adversary_seller_config: AdversarySellerConfig
    data: DataConfig
    debug: DebugConfig
    aggregation: AggregationConfig
    seed: int = 42
    n_samples: int = 3
    data_root: str = "./data"
    use_cache: bool = True
    buyer_attack_config: BuyerAttackConfig = field(default_factory=BuyerAttackConfig)
    valuation: ValuationConfig = field(default_factory=ValuationConfig)

    # def __post_init__(self):
    #     """
    #     Performs validation and adjustments after the config object is created.
    #     """
    #     logger.info("Performing post-initialization validation on AppConfig...")
    #
    #     poison_cfg = self.adversary_seller_config.poisoning
    #     sybil_cfg = self.adversary_seller_config.sybil
    #
    #     # --- Define Attack States ---
    #     is_poisoning = poison_cfg.type.value != 'none'
    #     is_sybil = sybil_cfg.is_sybil
    #     adv_rate_is_set = self.experiment.adv_rate > 0
    #
    #     # Define Sybil attacks that are *intentionally* non-poisoning
    #     sabotage_strategies = {'drowning'}
    #     is_sabotage_attack = is_sybil and sybil_cfg.gradient_default_mode in sabotage_strategies
    #
    #     # --- Check 1: (The one you asked about) ---
    #     # "adv_rate > 0" but "poisoning = none"
    #     # This is ONLY a bug if it's NOT a Sybil attack.
    #     if adv_rate_is_set and not is_poisoning and not is_sybil:
    #         logger.warning(
    #             f"Poisoning type is 'none' and not a Sybil attack, but adv_rate is {self.experiment.adv_rate}. "
    #             f"Forcing adv_rate to 0."
    #         )
    #         self.experiment.adv_rate = 0.0
    #     # (This check will now correctly do *nothing* for your Drowning attack)
    #
    #     # --- Check 2: (The one from the previous turn) ---
    #     # "is_sybil = True" but "poisoning = none"
    #     # This is ONLY a bug if it's NOT a known sabotage attack.
    #     if is_sybil and not is_poisoning and not is_sabotage_attack:
    #         logger.warning(
    #             f"Sybil attack is enabled (strategy: '{sybil_cfg.gradient_default_mode}') "
    #             f"but NO poisoning is active. This may be an error "
    #             f"(unless this strategy is a non-poisoning evasion attack)."
    #         )
    #

# --- A. Define the Configuration Class ---
# This is the 'SybilDrowningConfig' your coordinator code was missing.
@dataclass
class SybilDrowningConfig:
    """Config for the Targeted Drowning Attack."""
    # The seller_id of the benign competitor to target
    victim_id: str = "bn_3"
    # The 'alpha' from your paper's formula [cite: 530]
    attack_strength: float = 1.0
