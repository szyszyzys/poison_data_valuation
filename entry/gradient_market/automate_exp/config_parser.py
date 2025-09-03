# config_loader.py
import logging

import yaml
from dacite import from_dict, Config as DaciteConfig, MissingValueError, WrongTypeError

# Import your full AppConfig and its components
from common.gradient_market_configs import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> AppConfig:
    """
    Loads a YAML config file into a validated AppConfig dataclass.
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Dacite automatically handles the conversion from dict to your nested dataclasses
        cfg = from_dict(
            data_class=AppConfig,
            data=config_dict,
            config=DaciteConfig(cast=[bool, int, float, str])
        )

        # The __post_init__ method in AppConfig is automatically called here.

        logger.info("Successfully loaded and validated config.")
        return cfg

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except (MissingValueError, WrongTypeError) as e:
        logger.error(f"Error parsing config file '{config_path}': {e}")
        logger.error("Please check that your YAML file matches the structure of the AppConfig dataclasses.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the config: {e}")
        raise
