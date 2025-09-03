from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from common.enums import TextTriggerLocation, ImageTriggerType, ImageTriggerLocation, PoisonType, LabelFlipMode


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
    mode: LabelFlipMode = LabelFlipMode.FIXED_TARGET


@dataclass
class PoisoningConfig:
    """Configuration for client-side data poisoning attacks."""
    type: PoisonType = PoisonType.NONE
    poison_rate: float = 0.1
    image_backdoor_params: ImageBackdoorParams = field(default_factory=ImageBackdoorParams)
    text_backdoor_params: TextBackdoorParams = field(default_factory=TextBackdoorParams)
    label_flip_params: LabelFlipParams = field(default_factory=LabelFlipParams)


@dataclass
class ServerPrivacyConfig:
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


@dataclass
class TextDataConfig:
    """All settings related to a text dataset source."""
    vocab: VocabConfig
    strategy: str = "discovery"
    discovery: Optional[DiscoverySplitParams] = None
    buyer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageDataConfig:
    """All settings related to an image dataset source."""
    strategy: str = "property-skew"
    property_skew: Optional[PropertySkewParams] = None
    buyer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdversarySellerConfig:
    """A unified profile for an adversarial seller."""
    poisoning: PoisoningConfig = field(default_factory=PoisoningConfig)
    sybil: SybilConfig = field(default_factory=SybilConfig)


@dataclass
class DataConfig:
    """Holds configuration for one type of data source."""
    text: Optional[TextDataConfig] = None
    image: Optional[ImageDataConfig] = None


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


# --- REMOVED: Deleted the duplicate and redundant class definitions ---


@dataclass
class AppConfig:
    """Top-level configuration for the entire experiment run."""
    experiment: ExperimentConfig
    training: TrainingConfig
    server_privacy: ServerPrivacyConfig
    adversary_seller_config: AdversarySellerConfig
    data: DataConfig
    seed: int = 42
    n_samples: int = 1
    data_root: str = "./data"
    use_cache: bool = True
