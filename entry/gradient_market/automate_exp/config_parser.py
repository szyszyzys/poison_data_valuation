# config_parser.py
import logging
from argparse import Namespace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parses a nested config dictionary into a flat dictionary of keyword arguments
    suitable for a target experiment function.
    """
    # Maps YAML key paths (dot-separated) to the desired function argument name.
    KEY_MAP = {
        "data_split.num_sellers": "n_sellers",
        "output.final_save_path": "save_path",
        "attack.trigger_type": "backdoor_trigger_type",
        "attack.poison_strength": "backdoor_poison_strength",
    }

    # Keys that should be bundled into the legacy 'args' Namespace object.
    NAMESPACE_KEYS = [
        "gradient_manipulation_mode",
        "bkd_loc",
        "is_sybil",
        "clip",
        "remove_baseline",
    ]

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        self.config = config
        self.parsed_args = {}

    def _flatten_dict(self, d: dict, parent_key: str = ''):
        """Recursively flattens a nested dictionary."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def _apply_special_logic(self):
        """Handles conditional logic based on the parsed config."""
        # If the main 'attack' section is disabled, ensure adv_rate is zero.
        if not self.parsed_args.get("attack.enabled", False):
            if self.parsed_args.get("adv_rate", 0.0) > 0:
                logger.warning(
                    "Attack is disabled in config, but 'adv_rate' is non-zero. Forcing adv_rate to 0."
                )
                self.parsed_args["adv_rate"] = 0.0
            # Set attack_type to None if not enabled
            self.parsed_args["attack_type"] = "None"

        # If sybil is on but attack is off, issue a warning.
        if self.parsed_args.get("is_sybil", False) and not self.parsed_args.get("attack.enabled", False):
            logger.warning(
                "Sybil attack is enabled ('is_sybil': True), but the main attack section is disabled. Ensure this is intended."
            )

    def parse(self) -> dict:
        """
        Executes the full parsing workflow.

        Returns:
            A dictionary of keyword arguments for the experiment function.
        """
        # 1. Flatten the entire nested dictionary
        flat_config = self._flatten_dict(self.config)

        # 2. Map and rename keys, creating the base argument dictionary
        self.parsed_args = {}
        for flat_key, value in flat_config.items():
            # Use the mapped key if it exists, otherwise use the final part of the key
            arg_key = self.KEY_MAP.get(flat_key, flat_key.split('.')[-1])
            self.parsed_args[arg_key] = value

        # 3. Apply any special conditional logic
        self._apply_special_logic()

        # 4. Create the 'args' Namespace for legacy compatibility
        args_namespace = Namespace()
        for key in self.NAMESPACE_KEYS:
            # Set attribute on namespace and remove from main args dict
            if key in self.parsed_args:
                setattr(args_namespace, key, self.parsed_args.pop(key))
        self.parsed_args['args'] = args_namespace

        logger.info(f"Successfully parsed config for experiment: {self.config.get('experiment_id', 'N/A')}")
        return self.parsed_args
