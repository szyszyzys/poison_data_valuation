from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import Dataset

from common.enums import TextTriggerLocation, ImageTriggerType, ImageTriggerLocation


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
    save_path: str = "./results"
    num_classes: int = 0  # Default to 0, will be set dynamically at runtime


@dataclass
class TrainingConfig:
    """Holds parameters for local seller training."""
    local_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "Adam"


@dataclass
class ImageBackdoorParams:
    """Parameters specific to an IMAGE backdoor attack."""
    target_label: int = 0
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    strength: float = 0.2  # e.g., blend_alpha


@dataclass
class TextBackdoorParams:
    """Parameters specific to a TEXT backdoor attack."""
    target_label: int = 0
    trigger_content: str = "cf"  # The trigger phrase
    location: TextTriggerLocation = TextTriggerLocation.END


@dataclass
class LabelFlipParams:
    """Parameters specific to the label-flipping poisoning attack."""
    target_label: int = 0
    mode: str = "fixed_target"


@dataclass
class PoisoningConfig:
    """
    Configuration for client-side data poisoning attacks.
    Specifies the type and parameters for the attack.
    """
    type: str = "none"  # 'none', 'backdoor', or 'label_flip'
    poison_rate: float = 0.1
    # Nested parameters for each attack type
    image_backdoor_params: ImageBackdoorParams = field(default_factory=ImageBackdoorParams)
    text_backdoor_params: TextBackdoorParams = field(default_factory=TextBackdoorParams)
    label_flip_params: LabelFlipParams = field(default_factory=LabelFlipParams)


@dataclass
class PrivacyConfig:
    """Configuration for server-side privacy attacks (e.g., GIA)."""
    perform_gia: bool = False
    gia_frequency: int = 10


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
class AdversarySellerConfig:
    """
    A unified profile for an adversarial seller.
    This composes all configs related to adversarial behavior.
    """
    # Contains settings like type ('backdoor', 'label_flip') and poison_rate
    poisoning: PoisoningConfig = field(default_factory=PoisoningConfig)

    # Contains settings like is_sybil, roles, and strategies
    sybil: SybilConfig = field(default_factory=SybilConfig)


@dataclass
class DataConfig:
    """Configuration for a seller's local data."""
    dataset: Dataset
    num_classes: int


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
class PropertySkewConfig:
    """Configuration for the property-skew partitioning strategy."""
    property_key: str
    num_high_prevalence_clients: int
    num_security_attackers: int
    high_prevalence_ratio: float
    low_prevalence_ratio: float
    standard_prevalence_ratio: float


@dataclass
class DataPartitionConfig:
    """Configuration for data partitioning and splitting."""
    strategy: str = "property-skew"
    buyer_config: Dict[str, Any] = field(default_factory=dict)
    partition_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """
    A single, top-level dataclass that composes all other configurations.
    This object directly maps to the structure of your YAML file.
    """
    experiment: ExperimentConfig
    training: TrainingConfig
    privacy: PrivacyConfig
    adversary_seller_config: AdversarySellerConfig
    data_partition: DataPartitionConfig
    seed: int = 42
    n_samples: int = 1
