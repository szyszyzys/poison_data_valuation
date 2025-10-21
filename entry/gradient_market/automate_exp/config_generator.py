# config_generator.py

import copy
import itertools
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from types import NoneType
from typing import Any

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
        from its final configuration. Captures all swept parameters.
        """
        parts = []

        # 1. Dataset
        dataset = config.experiment.dataset_name.lower()
        parts.append(f"ds-{dataset}")

        # 2. Model Architecture
        model = config.experiment.model_structure.lower()
        parts.append(f"model-{model}")

        # 3. Aggregation Method
        agg = config.aggregation.method.lower()
        parts.append(f"agg-{agg}")

        # 4. Root Gradient Source (if not default 'buyer')
        root_source = config.aggregation.root_gradient_source
        if root_source != "buyer":
            parts.append(f"root-{root_source}")

        # 5. Seller Attack Information
        seller_attack_parts = []

        # 5a. Poisoning Attack
        poison_type = config.adversary_seller_config.poisoning.type.name.lower()
        if poison_type != 'none':
            adv_rate = f"{config.experiment.adv_rate:g}".replace('.', 'p')
            poison_rate = f"{config.adversary_seller_config.poisoning.poison_rate:g}".replace('.', 'p')
            seller_attack_parts.append(f"{poison_type}")
            seller_attack_parts.append(f"adv-{adv_rate}")
            seller_attack_parts.append(f"poison-{poison_rate}")

            # 5b. Sybil Coordination
            if config.adversary_seller_config.sybil.is_sybil:
                sybil_mode = config.adversary_seller_config.sybil.gradient_default_mode
                seller_attack_parts.append(f"sybil-{sybil_mode}")

        # 5c. Other Seller Attacks
        if config.adversary_seller_config.adaptive_attack.is_active:
            mode = config.adversary_seller_config.adaptive_attack.attack_mode
            seller_attack_parts.append(f"adaptive-{mode}")

        if config.adversary_seller_config.drowning_attack.is_active:
            seller_attack_parts.append("drowning")

        if config.adversary_seller_config.mimicry_attack.is_active:
            strategy = config.adversary_seller_config.mimicry_attack.strategy
            target = config.adversary_seller_config.mimicry_attack.target_seller_id
            seller_attack_parts.append(f"mimicry-{strategy}-target-{target}")

        # Join seller attack parts
        if seller_attack_parts:
            parts.append("_".join(seller_attack_parts))
        else:
            parts.append("no-seller-attack")

        # 6. Buyer Attack Information
        if config.buyer_attack_config.is_active:
            buyer_attack = config.buyer_attack_config.attack_type
            buyer_parts = [f"buyer-{buyer_attack}"]

            # Add attack-specific details
            if buyer_attack == "starvation":
                classes = config.buyer_attack_config.starvation_classes
                buyer_parts.append(f"classes-{'-'.join(map(str, classes))}")

            elif buyer_attack == "class_exclusion":
                if hasattr(config.buyer_attack_config, 'exclusion_exclude_classes') and \
                        config.buyer_attack_config.exclusion_exclude_classes:
                    classes = config.buyer_attack_config.exclusion_exclude_classes
                    buyer_parts.append(f"exclude-{'-'.join(map(str, classes))}")
                elif hasattr(config.buyer_attack_config, 'exclusion_target_classes') and \
                        config.buyer_attack_config.exclusion_target_classes:
                    classes = config.buyer_attack_config.exclusion_target_classes
                    buyer_parts.append(f"target-{'-'.join(map(str, classes))}")

            elif buyer_attack == "oscillating":
                strategy = config.buyer_attack_config.oscillation_strategy
                period = config.buyer_attack_config.oscillation_period
                buyer_parts.append(f"{strategy}-p{period}")

            elif buyer_attack == "orthogonal_pivot":
                target = config.buyer_attack_config.target_seller_id
                buyer_parts.append(f"target-{target}")

            parts.append("_".join(buyer_parts))

        # 7. Data Distribution Parameters (if non-default)
        if hasattr(config.data, 'image'):
            # Dirichlet alpha (for heterogeneity experiments)
            if config.data.image.strategy == "dirichlet":
                alpha = config.data.image.property_skew.dirichlet_alpha
                if alpha != 1.0:  # Only include if non-default
                    alpha_str = f"{alpha:g}".replace('.', 'p')
                    parts.append(f"alpha-{alpha_str}")

            # Buyer data percentage (for buyer_data_impact experiments)
            if hasattr(config.data.image, 'buyer_config'):
                buyer_pct = config.data.image.buyer_config.buyer_percentage
                if buyer_pct != 0.1:  # Only include if non-default
                    pct_str = f"{buyer_pct:g}".replace('.', 'p')
                    parts.append(f"buyerdata-{pct_str}")

        # 8. Marketplace Size (for scalability experiments)
        n_sellers = config.experiment.n_sellers
        if n_sellers != 20:  # Only include if non-default
            parts.append(f"sellers-{n_sellers}")

        # 9. Random Seed
        seed = config.seed
        parts.append(f"seed-{seed}")

        # Join all parts with underscores
        run_name = "_".join(parts)

        # Sanitize for filesystem (replace problematic characters)
        run_name = run_name.replace('[', '').replace(']', '').replace(',', '-')

        return run_name
