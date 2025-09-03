# config_generator.py

import copy
import itertools
from dataclasses import asdict
from pathlib import Path
from types import NoneType

import numpy as np
import yaml

from common.gradient_market_configs import AppConfig
from scenarios import Scenario


# --- YAML Dumper for handling None and Numpy types gracefully ---
class CustomDumper(yaml.SafeDumper):
    """A custom YAML dumper to correctly handle None and numpy types."""

    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

    def represent_numpy(self, data):
        if isinstance(data, np.integer): return self.represent_int(int(data))
        if isinstance(data, np.floating): return self.represent_float(float(data))
        if isinstance(data, np.ndarray): return self.represent_list(data.tolist())
        if isinstance(data, np.bool_): return self.represent_bool(bool(data))
        return self.represent_data(data)


CustomDumper.add_representer(NoneType, CustomDumper.represent_none)
for numpy_type in (np.integer, np.floating, np.ndarray, np.bool_):
    CustomDumper.add_multi_representer(numpy_type, CustomDumper.represent_numpy)


# --- Helper function to safely set nested dataclass attributes ---
def set_nested_attr(obj: object, attr_path: str, value):
    """Safely sets a nested attribute on an object using dot notation."""
    keys = attr_path.split('.')
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)


class ExperimentGenerator:
    """Generates configuration files by modifying a base AppConfig object."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, base_config: AppConfig, scenario: Scenario):
        """
        Generates all config files for a given experimental scenario.
        """
        print(f"\n--- Generating '{scenario.name}' Configs ---")

        # 1. Apply all base modifiers for the scenario
        scenario_config = copy.deepcopy(base_config)
        for modifier_func in scenario.modifiers:
            scenario_config = modifier_func(scenario_config)

        # 2. Create all combinations from the parameter grid
        keys, values = zip(*scenario.parameter_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # 3. Generate a file for each unique combination
        for params in param_combinations:
            final_config = copy.deepcopy(scenario_config)

            # Apply the specific parameters for this run
            for key, value in params.items():
                set_nested_attr(final_config, key, value)

            # IMPROVED: Generate a descriptive experiment ID from the swept parameters
            exp_id = self._generate_exp_id(params)

            # Set the final save path for results
            save_path = self.output_dir / scenario.name / exp_id
            final_config.experiment.save_path = str(save_path)

            # Convert the final dataclass object to a dictionary for saving
            config_dict = asdict(final_config)

            # Save the config file
            file_path = save_path / "config.yaml"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, Dumper=CustomDumper, sort_keys=False, indent=2)
            print(f"  Saved config: {file_path}")

    def _generate_exp_id(self, params: dict) -> str:
        """Creates a human-readable ID from the parameters being swept."""
        if not params:
            return "default"

        parts = []
        for key, value in sorted(params.items()):
            # Abbreviate common keys for concise filenames
            short_key = key.split('.')[-1]
            if "adv_rate" in short_key: short_key = "adv"
            if "poison_rate" in short_key: short_key = "pr"

            # Format value nicely (e.g., 0.1 becomes 0p1)
            value_str = f"{value:g}".replace('.', 'p') if isinstance(value, float) else str(value)
            parts.append(f"{short_key}-{value_str}")
        return "_".join(parts)
