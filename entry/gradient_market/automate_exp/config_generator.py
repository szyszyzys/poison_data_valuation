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

from common.enums import PoisonType, VictimStrategy, ImageBackdoorAttackName, ImageTriggerType, ImageTriggerLocation, \
    TextTriggerLocation, TextBackdoorAttackName, LabelFlipMode
from common.gradient_market_configs import AppConfig, TabularBackdoorAttackName
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
        return self.represent_scalar('tag:yaml.org,2002:str', data.value)

CustomDumper.add_representer(NoneType, CustomDumper.represent_none)

# --- MODIFY REGISTRATION ---
# Keep the base Enum representer (as a fallback)
CustomDumper.add_representer(Enum, CustomDumper.represent_enum)
# ADD specific representers for the exact Enum classes you use
CustomDumper.add_representer(PoisonType, CustomDumper.represent_enum)
CustomDumper.add_representer(VictimStrategy, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageBackdoorAttackName, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageTriggerType, CustomDumper.represent_enum)
CustomDumper.add_representer(TextTriggerLocation, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageTriggerLocation, CustomDumper.represent_enum)

CustomDumper.add_representer(TextBackdoorAttackName, CustomDumper.represent_enum)
CustomDumper.add_representer(TabularBackdoorAttackName, CustomDumper.represent_enum)

CustomDumper.add_representer(LabelFlipMode, CustomDumper.represent_enum)


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

        # 1. Apply all base modifiers
        scenario_config = copy.deepcopy(base_config)
        for modifier_func in scenario.modifiers:
            scenario_config = modifier_func(scenario_config)

        # 2. Create all combinations from the parameter grid
        keys, values = zip(*scenario.parameter_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        num_generated_total = len(param_combinations)
        print(f"  Found {num_generated_total} parameter combinations for this scenario.")

        count_saved = 0
        for i, params in enumerate(param_combinations):
            final_config = copy.deepcopy(scenario_config)

            # Apply the specific parameters for this run
            for key, value in params.items():
                try:
                    set_nested_attr(final_config, key, value)
                except (AttributeError, KeyError) as e:
                    print(f"  ❌ Error setting {key}={value} for combo {i}: {e}. Skipping.")
                    continue

            # 3. Create the descriptive run name
            try:
                run_name = self.create_run_name(final_config)
            except Exception as e:
                print(f"  ❌ Error creating run name for combo {i}: {e}. Using index instead.")
                run_name = f"combo_{i}"

            # 4. --- START OF FIX ---
            #    Define SEPARATE paths for results and configs

            # This is where the EXPERIMENT RESULTS will be saved
            results_save_path = Path("./results") / scenario.name / run_name
            final_config.experiment.save_path = str(results_save_path)

            # This is where the CONFIG FILE will be saved
            config_file_path = self.output_dir / scenario.name / run_name / "config.yaml"
            # --- END OF FIX ---

            # 5. Save the config file
            try:
                config_file_path.parent.mkdir(parents=True, exist_ok=True)
                config_dict = asdict(final_config)
                with open(config_file_path, 'w') as f:
                    # Make sure you have CustomDumper imported and registered!
                    yaml.dump(config_dict, f, Dumper=CustomDumper, sort_keys=False, indent=2)
                count_saved += 1
            except Exception as e:
                print(f"  ❌ Error saving config file {config_file_path}: {e}")

        print(f"  Successfully saved {count_saved} / {num_generated_total} config files.")
        return count_saved

    @staticmethod
    def create_run_name(config: AppConfig) -> str:
        """
        Generates a descriptive and unique name for a single experiment run
        from its final configuration. Captures all swept parameters.
        """
        parts = []

        # 1. Dataset
        parts.append(f"ds-{config.experiment.dataset_name.lower()}")

        # 2. Model Architecture
        parts.append(f"model-{config.experiment.model_structure.lower()}")

        # 3. Aggregation Method
        parts.append(f"agg-{config.aggregation.method.lower()}")

        # 5. Seller Attack Information
        seller_attack_parts = []
        poison_cfg = config.adversary_seller_config.poisoning
        sybil_cfg = config.adversary_seller_config.sybil

        poison_type = poison_cfg.type.name.lower()
        if poison_type != 'none':
            adv_rate = f"{config.experiment.adv_rate:g}".replace('.', 'p')
            poison_rate = f"{poison_cfg.poison_rate:g}".replace('.', 'p')
            seller_attack_parts.append(f"{poison_type}")
            seller_attack_parts.append(f"adv-{adv_rate}")
            seller_attack_parts.append(f"poison-{poison_rate}")

            if sybil_cfg.is_sybil:
                sybil_mode = sybil_cfg.gradient_default_mode or "unknown"
                seller_attack_parts.append(f"sybil-{sybil_mode}")

        if config.adversary_seller_config.adaptive_attack.is_active:
            mode = config.adversary_seller_config.adaptive_attack.attack_mode
            seller_attack_parts.append(f"adaptive-{mode}")
        # ... (other attacks are correct) ...

        if seller_attack_parts:
            parts.append("_".join(seller_attack_parts))
        else:
            parts.append("no-seller-attack")

        # 6. Buyer Attack (Your logic here is fine)
        if config.buyer_attack_config.is_active:
            # ... (your existing buyer attack logic) ...
            pass # Placeholder

        # 7. --- START OF FIX ---
        #    Data Distribution Parameters (Modality-Agnostic)
        modality_data_config = None
        if config.data.image:
            modality_data_config = config.data.image
            modality_name = "image"
        elif config.data.text:
            modality_data_config = config.data.text
            modality_name = "text"
        elif config.data.tabular:
            modality_data_config = config.data.tabular
            modality_name = "tabular"

        if modality_data_config:
            # Check strategy and alpha
            if modality_data_config.strategy == "dirichlet":
                # Check your default non-iid alpha, e.g., 0.5
                default_alpha = 0.5
                alpha = modality_data_config.dirichlet_alpha
                if alpha != default_alpha:
                    alpha_str = f"{alpha:g}".replace('.', 'p')
                    parts.append(f"alpha-{alpha_str}")

            # Check non-default buyer ratio
            default_buyer_ratio = 0.1
            buyer_pct = modality_data_config.buyer_ratio
            if buyer_pct != default_buyer_ratio:
                pct_str = f"{buyer_pct:g}".replace('.', 'p')
                parts.append(f"buyerdata-{pct_str}")
        # --- END OF FIX ---

        # 8. Marketplace Size
        n_sellers = config.experiment.n_sellers
        default_n_sellers = 10 # Or whatever your default is
        if n_sellers != default_n_sellers:
            parts.append(f"sellers-{n_sellers}")

        # 9. Random Seed
        parts.append(f"seed-{config.seed}")

        # Join and sanitize
        run_name = "_".join(parts)
        run_name = run_name.replace('[', '').replace(']', '').replace(',', '-')

        return run_name