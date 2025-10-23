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

from common.enums import PoisonType, VictimStrategy
from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.scenarios import Scenario


class CustomDumper(yaml.SafeDumper):
    """A custom YAML dumper to correctly handle None, numpy types, and enums."""

    def represent_none(self, _):
        """Represents None as an empty string (or choose 'null')."""
        return self.represent_scalar('tag:yaml.org,2002:null', '')  # or 'null'

    def represent_numpy(self, data):
        """Represents various numpy types as standard Python types."""
        if isinstance(data, np.integer): return self.represent_int(int(data))
        if isinstance(data, np.floating): return self.represent_float(float(data))
        if isinstance(data, np.ndarray): return self.represent_list(data.tolist())
        if isinstance(data, np.bool_): return self.represent_bool(bool(data))
        # Add fallback for other numpy types if needed, or let default handle it
        return super().represent_data(data)

    def represent_enum(self, data):
        """Tells YAML how to represent any Enum: by using its NAME as a string."""
        return self.represent_scalar('tag:yaml.org,2002:str', data.name)


CustomDumper.add_representer(NoneType, CustomDumper.represent_none)

# --- MODIFY REGISTRATION ---
# Keep the base Enum representer (as a fallback)
CustomDumper.add_representer(Enum, CustomDumper.represent_enum)
# ADD specific representers for the exact Enum classes you use
CustomDumper.add_representer(PoisonType, CustomDumper.represent_enum)
CustomDumper.add_representer(VictimStrategy, CustomDumper.represent_enum)
# Add specific lines for any other Enum types used in your AppConfig
# --- END MODIFICATION ---
# For NumPy types (using add_multi_representer for specific types)
for numpy_type in (np.integer, np.floating, np.ndarray, np.bool_):
    # Ensure you add representers for the specific types, not the base class np.generic
    CustomDumper.add_representer(numpy_type, CustomDumper.represent_numpy)


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

    # --- ADDED A RETURN TYPE HINT ---
    def generate(self, base_config: AppConfig, scenario: Scenario) -> int:
        """
        Generates all config files for a given experimental scenario.

        Returns:
            int: The number of configuration files generated for this scenario.
        """
        print(f"\n--- Generating '{scenario.name}' Configs ---")

        # 1. Apply all base modifiers for the scenario
        scenario_config = copy.deepcopy(base_config)
        for modifier_func in scenario.modifiers:
            scenario_config = modifier_func(scenario_config)

        # 2. Create all combinations from the parameter grid
        keys, values = zip(*scenario.parameter_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # --- STORE THE COUNT ---
        num_generated = len(param_combinations)
        print(f"  Found {num_generated} parameter combinations for this scenario.")

        # 3. Generate a file for each unique combination
        count_saved = 0  # Keep track of successfully saved files
        for i, params in enumerate(param_combinations):
            final_config = copy.deepcopy(scenario_config)

            # Apply the specific parameters for this run
            for key, value in params.items():
                try:
                    set_nested_attr(final_config, key, value)
                except (AttributeError, KeyError) as e:
                    print(f"  ❌ Error setting {key}={value} for combo {i}: {e}. Skipping this combo.")
                    num_generated -= 1  # Decrement count if setting fails
                    continue  # Skip to the next combination

            # --- Assuming self.create_run_name exists ---
            try:
                run_name = self.create_run_name(final_config)
            except Exception as e:
                print(f"  ❌ Error creating run name for combo {i}: {e}. Using index instead.")
                run_name = f"combo_{i}"

            save_path = self.output_dir / scenario.name / run_name
            final_config.experiment.save_path = str(save_path)
            file_path = save_path / "config.yaml"

            try:
                print(f"DEBUG: Attempting to save {file_path}")
                print(f"DEBUG: Using Dumper: {CustomDumper}")
                # Check if the Enum representer is actually registered on this dumper instance
                if Enum in CustomDumper.yaml_representers:
                    print(f"DEBUG: Enum representer IS registered for CustomDumper.")
                else:
                    print(f"DEBUG: WARNING! Enum representer NOT registered for CustomDumper!")

                # Optionally, print the specific value causing issues if you know it
                try:
                    problem_value = final_config.adversary_seller_config.poisoning.type
                    print(f"DEBUG: Value of poison type: {problem_value} (Type: {type(problem_value)})")
                except Exception as e_debug:
                    print(f"DEBUG: Could not access poison type for debug print: {e_debug}")
                # --- END DEBUG PRINTS ---
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # ... (mkdir logic) ...
                config_dict = asdict(final_config)
                with open(file_path, 'w') as f:
                    # --- ENSURE THIS LINE HAS Dumper=CustomDumper ---
                    yaml.dump(config_dict, f, Dumper=CustomDumper, sort_keys=False, indent=2)
                    # --- END VERIFICATION ---
                count_saved += 1
            except Exception as e:
                print(f"  ❌ Error saving config file {file_path}: {e}")
                num_generated -= 1  # Decrement if saving fails

        print(f"  Successfully saved {count_saved} / {len(param_combinations)} config files.")

        # --- RETURN THE COUNT ---
        return num_generated

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
        if config.data.image:
            # Dirichlet alpha (for heterogeneity experiments)
            if config.data.image.strategy == "dirichlet":
                alpha = config.data.image.property_skew.dirichlet_alpha
                if alpha != 1.0:  # Only include if non-default
                    alpha_str = f"{alpha:g}".replace('.', 'p')
                    parts.append(f"alpha-{alpha_str}")

            # Buyer data percentage (for buyer_data_impact experiments)
            buyer_pct = config.data.image.discovery.buyer_percentage
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
