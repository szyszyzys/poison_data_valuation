# config_loader.py
import logging

import yaml
from dacite import from_dict, Config as DaciteConfig

# Import your full AppConfig and its components
from common.gradient_market_configs import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> AppConfig:
    """
    Loads a YAML configuration file directly into a nested AppConfig dataclass.

    This is the modern replacement for a manual parser.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Use a library like dacite to automatically and safely cast the dict to your dataclasses
    cfg = from_dict(
        data_class=AppConfig,
        data=config_dict,
        config=DaciteConfig(cast=[bool, int, float, str])  # Helps with type conversion
    )

    # --- Handling Special Logic ---
    # This logic is now part of the loading process, not a separate parsing step.

    poison_cfg = cfg.adversary_seller_config.poisoning
    sybil_cfg = cfg.adversary_seller_config.sybil

    # If attacks are off, ensure adv_rate is zero.
    if poison_cfg.type.value == 'none':  # Assuming PoisonType is an Enum
        if cfg.experiment.adv_rate > 0:
            logger.warning(
                "Poisoning type is 'none', but 'adv_rate' is non-zero. Forcing adv_rate to 0."
            )
            cfg.experiment.adv_rate = 0.0

    # If sybil is on but poisoning is off, issue a warning.
    if sybil_cfg.is_sybil and poison_cfg.type.value == 'none':
        logger.warning(
            "Sybil attack is enabled, but poisoning type is 'none'. Ensure this is intended."
        )

    logger.info(f"Successfully loaded and validated config for experiment: {config_dict.get('experiment_id', 'N/A')}")
    return cfg
