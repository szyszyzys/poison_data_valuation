import copy
import itertools
from pathlib import Path
from types import NoneType

import numpy as np
import torch
import yaml


class CustomDumper(yaml.SafeDumper):
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

# --- Updated Base Configuration Template ---
BASE_CONFIG_TEMPLATE = {
    'exp_name': 'new',
    'dataset_name': 'CIFAR',
    'model_structure': 'Simple_CNN',
    'aggregation_method': 'fedavg',
    'global_rounds': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_samples': 10,
    'data_split': {
        'num_sellers': 30, 'adv_rate': 0.0, 'buyer_percentage': 0.02,
        'data_split_mode': 'discovery',
        'dm_params': {'discovery_quality': 0.3, 'buyer_data_mode': 'unbiased'},
        'normalize_data': True, 'data_path': './data',
    },
    'training': {
        'batch_size': 64,
        'local_training_params': {'local_epochs': 2, 'optimizer': 'Adam', 'learning_rate': 0.001},
        'clip': 10.0, 'early_stopping_patience': 20
    },
    'attack': {
        'type': 'none',
        'poison_rate': 0.2,
        'image_backdoor_params': {
            'target_label': 0, 'trigger_type': 'blended_patch',
            'location': 'bottom_right', 'strength': 1.0,
        },
        'text_backdoor_params': {
            "target_label": 0,
            "trigger_content": "indeed remarked",
            "location": "end"
        },
        'label_flip_params': {'mode': 'fixed_target', 'target_label': 0},
    },
    'sybil': {
        'is_sybil': False,
        'benign_rounds': 0,
        'detection_threshold': 0.8,
        'gradient_default_mode': 'mimic',
        'trigger_mode': 'static',
        'history_window_size': 10,
        'role_config': {'mimic': 1.0},
        'strategy_configs': {
            'stealth': {'scale_factor': 0.5},
            'amplify': {'factor': 2.0}
        }
    },
    'privacy_attack': {
        'perform_gradient_inversion': False,
        'attack_victim_strategy': 'fixed',
        'attack_fixed_victim_idx': 0
    },
    'data_partition': {
        'strategy': "property-skew",
        'buyer_config': {
            'root_set_fraction': 0.2
        },
        'partition_params': {
            'property_key': 'tumor',
            'num_high_prevalence_clients': 5,
            'num_security_attackers': 5,
            'high_prevalence_ratio': 0.8,
            'low_prevalence_ratio': 0.1,
            'standard_prevalence_ratio': 0.4,
        }
    },
    'output': {'save_path_base': './experiment_results'}
}


# --- ConfigGenerator Class (Slightly improved ID generation) ---
class ConfigGenerator:
    """A class to generate experiment configurations from declarative scenarios."""

    def __init__(self, base_template, output_dir):
        self.base_template = base_template
        self.output_dir = Path(output_dir)
        self.dataset_channels = {'CIFAR': 3, 'FMNIST': 1, 'CelebA': 3, 'Camelyon16': 3}
        self.default_lrs = {'CIFAR': 0.001, 'FMNIST': 0.001, 'CelebA': 0.001, 'Camelyon16': 0.001}

    def _set_nested_key(self, config_dict, key_path, value):
        keys = key_path.split('.')
        d = config_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _generate_exp_id(self, scenario_name, params):
        key_map = {
            "dataset_name": "ds", "aggregation_method": "agg", "data_split.adv_rate": "adv",
            "attack.poison_rate": "pr", "sybil.strategy_configs.amplify.factor": "amp"
        }
        parts = [scenario_name]
        for key, value in sorted(params.items()):
            short_key = key_map.get(key, key.replace('.', '_'))
            value_str = f"{value:g}".replace('.', 'p') if isinstance(value, float) else str(value)
            parts.append(f"{short_key}-{value_str}")
        return "_".join(parts)

    def generate(self, scenario_name, parameter_grid, base_overrides):
        print(f"\n--- Generating '{scenario_name}' Configs ---")
        keys, values = zip(*parameter_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in param_combinations:
            config = copy.deepcopy(self.base_template)
            for key, value in base_overrides.items():
                self._set_nested_key(config, key, value)
            for key, value in params.items():
                self._set_nested_key(config, key, value)

            exp_id = self._generate_exp_id(scenario_name, params)
            config['experiment_id'] = exp_id

            ds = config['dataset_name']
            lr = self.default_lrs.get(ds, 0.001)
            self._set_nested_key(config, 'training.local_training_params.learning_rate', lr)
            self._set_nested_key(config, 'data_split.normalize_data', self.dataset_channels.get(ds) is not None)

            save_path = self.output_dir / scenario_name / exp_id
            self._set_nested_key(config, 'output.final_save_path', str(save_path))

            file_path = save_path / "config.yaml"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                yaml.dump(config, f, Dumper=CustomDumper, sort_keys=False, indent=2)
            print(f"  Saved config: {file_path}")


# --- Main Execution with Updated Scenarios ---
def main():
    """Defines and runs all experimental scenarios."""
    CONFIG_OUTPUT_DIRECTORY = "./configs_generated"
    generator = ConfigGenerator(BASE_CONFIG_TEMPLATE, CONFIG_OUTPUT_DIRECTORY)

    SCENARIOS = {
        "baseline": {
            "base_overrides": {
                "attack.type": "none",
                "sybil.is_sybil": False,
            },
            "parameter_grid": {
                "dataset_name": ['CelebA', 'Camelyon16'],
                "aggregation_method": ['fedavg', 'krum'],
            }
        },
        "backdoor_attack": {
            "base_overrides": {
                "attack.type": "backdoor",
                "data_split.num_sellers": 10,
            },
            "parameter_grid": {
                "dataset_name": ['CelebA', 'Camelyon16'],
                "data_split.adv_rate": [0.1, 0.3],
                "attack.poison_rate": [0.1, 0.4]
            }
        },
        "sybil_amplify_attack": {
            "base_overrides": {
                "attack.type": "backdoor",
                "attack.poison_rate": 0.2,
                "sybil.is_sybil": True,
                "sybil.role_config": {"amplify": 1.0},  # All sybils use the amplify strategy
                "data_split.num_sellers": 10
            },
            "parameter_grid": {
                "dataset_name": ['CelebA', 'Camelyon16'],
                "data_split.adv_rate": [0.1, 0.2],
                # Sweep over the amplification factor
                "sybil.strategy_configs.amplify.factor": [2.0, 5.0, 10.0]
            }
        },
    }

    for name, scenario in SCENARIOS.items():
        generator.generate(name, scenario['parameter_grid'], scenario['base_overrides'])

    print(f"\n✅ Configuration generation finished in '{CONFIG_OUTPUT_DIRECTORY}'")


if __name__ == "__main__":
    main()
