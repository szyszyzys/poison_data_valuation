import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List, Union, Callable

from torch.utils.data import Dataset

from common.enums import TextTriggerLocation, ImageTriggerType, ImageTriggerLocation, PoisonType, LabelFlipMode, \
    VictimStrategy, ImageBackdoorAttackName, TextBackdoorAttackName

logger = logging.getLogger("Configs")


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
    aggregation_method: str
    global_rounds: int
    n_sellers: int
    adv_rate: float
    device: str
    eval_frequency: int = 10
    compute_gradient_similarity: bool = False  # Add this line
    save_path: str = "./results"
    num_classes: int = 0  # Default to 0, will be set dynamically at runtime
    use_subset: bool = False
    subset_size: int = 3000  # Number of samples to use in the subset
    dataset_type: str = "text"
    evaluation_frequency: int = 1
    evaluations: List[str] = field(default_factory=lambda: ["clean"])
    image_model_config_name: str = "cifar10_cnn"
    tabular_model_config_name: str = "mlp_texas100_baseline"  # Default model config to use


@dataclass
class TrainingConfig:
    """Holds parameters for local seller training."""
    local_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "Adam"
    use_early_stopping: bool = False
    patience: int = 10


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
class TabularFeatureTriggerParams:
    """Parameters for a feature-based trigger backdoor attack."""
    target_label: int = 1
    # Trigger is a dictionary of {'feature_name': value},
    # e.g., {'age': 65, 'job_title': 'retired'}
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
    victim_strategy: VictimStrategy = VictimStrategy.RANDOM
    fixed_victim_idx: int = 0
    lrs_to_try: List[float] = field(default_factory=lambda: [0.1, 0.01])
    base_attack_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerAttackConfig:
    """Configuration for all server-side privacy attacks."""
    # This is the main switch: 'none', 'gradient_inversion', etc.
    attack_name: str = 'none'

    # Nested parameters for each possible attack
    gradient_inversion_params: GradientInversionParams = field(default_factory=GradientInversionParams)
    # You could add others here in the future
    # membership_inference: MembershipInferenceParams = field(default_factory=MembershipInferenceParams)


@dataclass
class SybilConfig:
    """Configuration for Sybil attack coordination and behavior."""
    is_sybil: bool = False
    detection_threshold: float = 0.8
    benign_rounds: int = 0
    gradient_default_mode: str = "mimic"
    trigger_mode: str = "static"
    history_window_size: int = 10
    role_config: Dict[str, float] = field(default_factory=dict)
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DiscoverySplitParams:
    """Parameters for the 'discovery' text data splitting strategy."""
    buyer_percentage: float = 0.05
    discovery_quality: float = 0.3
    buyer_data_mode: str = "biased"
    buyer_bias_type: str = "dirichlet"
    buyer_dirichlet_alpha: float = 0.2


@dataclass  # --- FIXED: Added the @dataclass decorator ---
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
class TextDataConfig:
    """All settings related to a text dataset source."""
    vocab: VocabConfig
    strategy: str = "discovery"
    discovery: Optional[DiscoverySplitParams] = None
    property_skew: Optional[TextPropertySkewParams] = None
    buyer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageDataConfig:
    """All settings related to an image dataset source."""
    strategy: str = "property-skew"
    property_skew: Optional[PropertySkewParams] = None
    buyer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveAttackConfig:
    """Configuration for the adaptive learning attacker."""
    is_active: bool = False
    # The primary mode of attack: either manipulate the gradient or poison the data
    attack_mode: Literal["gradient_manipulation", "data_poisoning"] = "gradient_manipulation"

    # --- General Learning Parameters ---
    exploration_rounds: int = 20

    # --- Gradient Manipulation Strategies (if mode is 'gradient_manipulation') ---
    gradient_strategies: List[str] = field(
        default_factory=lambda: ["honest", "scale_up", "add_noise"]
    )
    scale_factor: float = 1.5
    noise_level: float = 0.01

    # --- Data Poisoning Strategies (if mode is 'data_poisoning') ---
    # These map to poison types you already have configured
    data_strategies: List[str] = field(
        default_factory=lambda: ["honest", "label_flip"]  # e.g., 'label_flip', 'image_backdoor'
    )


@dataclass
class AdversarySellerConfig:
    """A unified profile for an adversarial seller."""
    name: str = "benign"
    poisoning: PoisoningConfig = field(default_factory=PoisoningConfig)
    sybil: SybilConfig = field(default_factory=SybilConfig)
    adaptive_attack: AdaptiveAttackConfig = field(default_factory=AdaptiveAttackConfig)


@dataclass
class TabularDataConfig:
    dataset_config_path: str = "configs/tabular_datasets.yaml"
    model_config_dir: str = "model/configs"
    buyer_ratio: float = 0.1
    strategy: str = "iid"  # Add 'strategy' field
    property_skew: Dict[str, Any] = field(default_factory=dict)  # Add 'property_skew' field


@dataclass
class DataConfig:
    """Holds configuration for one type of data source."""
    text: Optional[TextDataConfig] = None
    image: Optional[ImageDataConfig] = None
    num_workers = 2
    tabular: Optional[TabularDataConfig] = None


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
    change_base: bool = True
    clip: bool = True


@dataclass
class SkymaskParams:
    """Parameters specific to the Skymask aggregator."""
    clip: bool = True
    sm_model_type: str = 'None'
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
    clip_norm: float = 0.01

    # --- Nested parameter objects for each strategy ---
    martfl: MartFLParams = field(default_factory=MartFLParams)
    skymask: SkymaskParams = field(default_factory=SkymaskParams)


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

    def __post_init__(self):
        """
        Performs validation and adjustments after the config object is created.
        """
        logger.info("Performing post-initialization validation on AppConfig...")

        poison_cfg = self.adversary_seller_config.poisoning
        sybil_cfg = self.adversary_seller_config.sybil

        # If attacks are off, ensure adv_rate is zero.
        if poison_cfg.type.value == 'none' and self.experiment.adv_rate > 0:
            logger.warning(
                f"Poisoning type is 'none', but adv_rate is {self.experiment.adv_rate}. Forcing adv_rate to 0."
            )
            self.experiment.adv_rate = 0.0

        # If sybil is on but poisoning is off, issue a warning.
        if sybil_cfg.is_sybil and poison_cfg.type.value == 'none':
            logger.warning(
                "Sybil attack is enabled, but poisoning type is 'none'. Ensure this is intended."
            )
