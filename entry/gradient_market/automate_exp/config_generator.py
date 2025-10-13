# config_generator.py

import copy
import itertools
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from types import NoneType

import numpy as np
import yaml

from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.scenarios import Scenario


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

    def represent_enum(self, data):
        """Tells YAML how to represent any Enum: by using its value."""
        return self.represent_scalar('tag:yaml.org,2002:str', data.value)


CustomDumper.add_representer(NoneType, CustomDumper.represent_none)
CustomDumper.add_multi_representer(Enum, CustomDumper.represent_enum)
for numpy_type in (np.integer, np.floating, np.ndarray, np.bool_):
    CustomDumper.add_multi_representer(numpy_type, CustomDumper.represent_numpy)


# --- Helper function to safely set nested dataclass attributes ---
# def set_nested_attr(obj: object, attr_path: str, value):
#     """Safely sets a nested attribute on an object using dot notation."""
#     keys = attr_path.split('.')
#     for key in keys[:-1]:
#         obj = getattr(obj, key)
#     setattr(obj, keys[-1], value)
def set_nested_attr(obj: Any, key: str, value: Any):
    """
    Sets a nested attribute on an object or a key in a nested dict
    using a dot-separated key.
    """
    keys = key.split('.')
    current_obj = obj

    # Traverse to the second-to-last object in the path
    for k in keys[:-1]:
        current_obj = getattr(current_obj, k)

    # Get the final key/attribute to be set
    final_key = keys[-1]

    # --- THIS IS THE CRITICAL LOGIC ---
    # Check if the object we need to modify is a dictionary
    if isinstance(current_obj, dict):
        # If it's a dict, use item assignment (e.g., my_dict['key'] = value)
        current_obj[final_key] = value
    else:
        # Otherwise, use attribute assignment (e.g., my_obj.key = value)
        setattr(current_obj, final_key, value)


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
        print(f"  Found {len(param_combinations)} parameter combinations for this scenario.")

        # 3. Generate a file for each unique combination
        for params in param_combinations:
            final_config = copy.deepcopy(scenario_config)

            # Apply the specific parameters for this run
            for key, value in params.items():
                set_nested_attr(final_config, key, value)

            # --- THIS IS THE UPDATED LOGIC ---
            # 1. Generate the descriptive run name from the FINAL config object.
            run_name = self.create_run_name(final_config)

            # 2. Create the full, unique path for this run's results.
            save_path = self.output_dir / scenario.name / run_name

            # 3. Update the config object to be self-aware of its save path.
            final_config.experiment.save_path = str(save_path)

            # 4. Set the path for the config file itself inside the unique directory.
            file_path = save_path / "config.yaml"
            # -----------------------------------

            # Your existing saving logic is correct
            file_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = asdict(final_config)
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, Dumper=CustomDumper, sort_keys=False, indent=2)
            print(f"  Saved config to: {file_path}")

    @staticmethod
    def create_run_name(config: AppConfig) -> str:
        """
        Generates a descriptive and unique name for a single experiment run
        from its final configuration.
        """
        # 1. Model Architecture
        model_name = config.experiment.model_structure.lower()

        # 2. Aggregation Method
        agg_method = config.aggregation.method.lower()

        # 3. Adversary Attack Type
        attack_type = config.adversary_seller_config.poisoning.type.name.lower()
        if attack_type == 'none':
            attack_str = "no_attack"
        else:
            # Include the adversary rate if an attack is active
            adv_rate_str = f"{config.experiment.adv_rate:g}".replace('.', 'p')
            attack_str = f"{attack_type}_adv-{adv_rate_str}"

        # 4. Random Seed
        seed = f"seed-{config.seed}"

        # Assemble the final name
        run_name = f"model-{model_name}_agg-{agg_method}_{attack_str}_{seed}"

        return run_name
