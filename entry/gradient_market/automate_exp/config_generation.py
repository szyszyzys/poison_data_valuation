# generate_configs.py
import copy
import itertools
import os

import numpy as np  # Used for linspace if needed
import yaml

# --- Configuration Templates ---

BASE_CONFIG_TEMPLATE = {
    # --- Top Level Params for backdoor_attack ---
    'dataset_name': 'CIFAR',  # Default, override per experiment
    'model_structure': 'SimpleCNN',  # Default, override per experiment
    'aggregation_method': 'fedavg',  # Default, override per experiment
    'global_rounds': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Detect device

    # --- Data Split Info (Passed to get_data_set) ---
    'data_split': {
        'num_sellers': 10,
        'adv_rate': 0.0,  # Default: no adversaries
        'buyer_percentage': 0.02,
        'data_split_mode': 'NonIID',  # Options: NonIID, IID, discovery
        'dirichlet_alpha': 0.5,  # Used if data_split_mode="NonIID"
        'dm_params': {  # Used if data_split_mode="discovery"
            'discovery_quality': 0.7,  # Example default
            'buyer_data_mode': 'random',  # Example default ('random' or 'biased')
            # 'buyer_bias_distribution': {0: 0.8, 1: 0.1, ...} # Add if buyer_data_mode='biased'
        },
        'normalize_data': True,
        'data_path': './data',
    },

    # --- Training Params ---
    'training': {
        'batch_size': 64,
        'local_training_params': {  # Passed to sellers/buyer
            'local_epochs': 5,
            'optimizer': 'Adam',
            'learning_rate': 0.001,  # Adjust per dataset/model maybe
            # Add other params like weight_decay if needed
        },
        'clip': 10.0,  # Gradient clipping (from args.clip)
        'early_stopping_patience': 20,  # Example
        'early_stopping_min_delta': 0.01,  # Example
        'early_stopping_monitor': 'acc',  # Example ('acc' or 'loss')
    },

    # --- Federated Learning / Aggregation Params ---
    'federated_learning': {
        'change_base': True,  # Passed to Aggregator (related to martfl?)
        'remove_baseline': False,  # From args.remove_baseline (related to martfl?)
        # --- Add params specific to other aggregators if needed ---
        # 'skymask_param': value,
        # 'fltrust_param': value,
    },

    # --- Attack Params ---
    'attack': {
        'enabled': False,  # Default: No attack
        'scenario': 'backdoor',  # This whole function is about backdoor
        'backdoor_target_label': 0,
        'trigger_type': 'blended_patch',  # e.g., blended_patch, pixel_pattern, sig
        'trigger_rate': 0.1,  # Portion of malicious client data poisoned
        'poison_strength': 1.0,  # Strength scaling
        'bkd_loc': 'bottom_right',  # From args.bkd_loc
        'gradient_manipulation_mode': 'default',  # From args.gradient_manipulation_mode
    },

    # --- Sybil Params ---
    'sybil': {
        'is_sybil': False,  # Default: Not a sybil attack (from args.is_sybil)
        'benign_rounds': 5,
        'sybil_mode': 'default',
        'alpha': 1.0,
        'amplify_factor': 1.0,
        'cost_scale': 1.0,
        'trigger_mode': 'always_on',  # Or 'adaptive', etc.
    },

    # --- Output ---
    'output': {
        'save_path_base': './experiment_results',  # Base directory for saving results
        # Specific save path will be constructed using experiment_id
    }
}

# --- Model Configs per Dataset (Simplified) ---
# You might need more details (layers, etc.) depending on model structure definition
MODEL_CONFIGS = {
    'CIFAR': 'SimpleCNN', 'CIFAR10': 'SimpleCNN',
    'FMNIST': 'SimpleCNN',  # Or SimpleMLP
    'MNIST': 'SimpleMLP',
    'AG_NEWS': 'SimpleLSTM',
    'TREC': 'SimpleLSTM',
}
DATASET_CHANNELS = {
    'CIFAR': 3, 'CIFAR10': 3,
    'FMNIST': 1,
    'MNIST': 1,
    'AG_NEWS': None,  # Not applicable
    'TREC': None,  # Not applicable
}
DEFAULT_LRS = {  # Example: Adjust learning rates per dataset
    'CIFAR': 0.001, 'CIFAR10': 0.001,
    'FMNIST': 0.001,
    'MNIST': 0.001,
    'AG_NEWS': 0.0005,
    'TREC': 0.0005,
}


# --- Helper Function ---
def save_config(config_dict, path):
    """Saves a dictionary as a YAML file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert numpy types for YAML compatibility if they sneak in
        def numpy_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False, indent=2, default_dumper=yaml.Dumper,
                      default_representer=lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:null',
                                                                                       '') if data is None else dumper.represent_data(
                          data))

        print(f"  Saved config: {path}")
    except Exception as e:
        print(f"  Error saving config {path}: {e}")


# --- Generation Functions ---

def generate_baseline_configs(output_dir):
    """Baselines: No attack, vary dataset and split method."""
    print("\n--- Generating Baseline Configs ---")
    datasets = ['CIFAR', 'FMNIST']
    split_methods = ['discovery']  # Add 'discovery' if desired

    for ds in datasets:
        for split in split_methods:
            config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
            exp_id = f"baseline_{ds.lower()}_{split.lower()}"
            if split == 'NonIID': exp_id += f"_a{config['data_split']['dirichlet_alpha']}"

            config['experiment_id'] = exp_id
            config['dataset_name'] = ds
            config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
            config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
            config['data_split']['data_split_mode'] = split
            config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)  # Normalize vision only

            # Disable attack and sybil
            config['attack']['enabled'] = False
            config['sybil']['is_sybil'] = False

            # Configure results path
            results_path = os.path.join(config['output']['save_path_base'], "baselines", exp_id)
            config['output']['final_save_path'] = results_path  # Runner script uses this

            file_path = os.path.join(output_dir, "baselines", f"{exp_id}.yaml")
            save_config(config, file_path)


def generate_attack_configs(output_dir):
    """Vary attack params: adv_rate, aggregation, maybe trigger/target."""
    print("\n--- Generating Attack Configs ---")
    datasets = ['CIFAR']  # Focus on one dataset for this example
    adv_rates = [0.1, 0.3]
    aggregations = ['fedavg', 'martfl']  # Add others like 'median', 'trimmed_mean', 'skymask', 'fltrust'
    target_labels = [0]  # Could vary this too
    trigger_types = ['blended_patch']  # Could vary

    for ds, rate, agg, target, trigger in itertools.product(datasets, adv_rates, aggregations, target_labels,
                                                            trigger_types):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        rate_pct = int(rate * 100)
        exp_id = f"attack_{ds.lower()}_{agg.lower()}_adv{rate_pct}pct_t{target}_{trigger}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        config['aggregation_method'] = agg
        config['data_split']['adv_rate'] = rate
        config['data_split']['num_sellers'] = 10  # Keep constant for this example

        # Enable and configure attack
        config['attack']['enabled'] = True
        config['attack']['backdoor_target_label'] = target
        config['attack']['trigger_type'] = trigger
        # Use default trigger rate, strength, location, grad manip mode - vary these in other experiments if needed

        # Disable sybil unless specifically testing sybil+backdoor
        config['sybil']['is_sybil'] = False

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "attack_comparison", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "attack_comparison", f"{exp_id}.yaml")
        save_config(config, file_path)


def generate_sybil_configs(output_dir):
    """Focus on varying Sybil parameters."""
    print("\n--- Generating Sybil Attack Configs ---")
    datasets = ['CIFAR']
    adv_rates = [0.3]  # Fix adversary rate
    aggregations = ['fedavg', 'martfl']  # Compare how Sybil affects different aggregators
    amplify_factors = [1.0, 5.0, 10.0]  # Vary amplification

    for ds, rate, agg, amplify in itertools.product(datasets, adv_rates, aggregations, amplify_factors):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        rate_pct = int(rate * 100)
        exp_id = f"sybil_{ds.lower()}_{agg.lower()}_adv{rate_pct}pct_amp{amplify}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        config['aggregation_method'] = agg
        config['data_split']['adv_rate'] = rate
        config['data_split']['num_sellers'] = 10

        # Enable backdoor attack (Sybil often works via backdoor pattern)
        config['attack']['enabled'] = True
        config['attack']['backdoor_target_label'] = 0  # Example target
        config['attack']['trigger_type'] = 'blended_patch'  # Example trigger

        # Enable and configure Sybil
        config['sybil']['is_sybil'] = True
        config['sybil']['amplify_factor'] = amplify
        # Use default benign_rounds, sybil_mode, alpha, cost_scale, trigger_mode - vary these in other experiments

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "sybil_comparison", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "sybil_comparison", f"{exp_id}.yaml")
        save_config(config, file_path)


def generate_discovery_configs(output_dir):
    """Compare discovery split method."""
    print("\n--- Generating Discovery Split Configs ---")
    datasets = ['CIFAR']  # Discovery might be more interesting with complex data
    qualities = [0.3, 0.7, 0.95]  # Low, Medium, High quality simulation
    buyer_modes = ['unbiased', 'unbiased']  # Add 'biased' if construct_buyer_set supports it well

    for ds, quality, buyer_mode in itertools.product(datasets, qualities, buyer_modes):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        exp_id = f"discovery_{ds.lower()}_q{quality}_{buyer_mode}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        # Set split method to discovery
        config['data_split']['data_split_mode'] = 'discovery'
        config['data_split']['dm_params']['discovery_quality'] = quality
        config['data_split']['dm_params']['buyer_data_mode'] = buyer_mode
        # Remove dirichlet_alpha if not used by discovery split
        if 'dirichlet_alpha' in config['data_split']: del config['data_split']['dirichlet_alpha']
        # Ensure buyer percentage is set if needed by discovery logic
        config['data_split']['buyer_percentage'] = 0.02  # Example

        # No attack for this comparison
        config['attack']['enabled'] = False
        config['sybil']['is_sybil'] = False

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "discovery_split", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "discovery_split", f"{exp_id}.yaml")
        save_config(config, file_path)


# --- Main Execution ---
if __name__ == "__main__":
    # Need torch to check cuda availability in template
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not found. 'device' will default to 'cpu'.")
        # Manually set device in template if torch is unavailable
        BASE_CONFIG_TEMPLATE['device'] = 'cpu'

    CONFIG_OUTPUT_DIRECTORY = "./configs_generated"  # Directory to save generated configs

    print(f"Generating configuration files in: {CONFIG_OUTPUT_DIRECTORY}")

    # Generate specific experiment groups
    generate_baseline_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_attack_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_sybil_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_discovery_configs(CONFIG_OUTPUT_DIRECTORY)
    # Add calls to generate other experiment groups as needed

    print("\nConfiguration generation finished.")
    print(f"Check the '{CONFIG_OUTPUT_DIRECTORY}' directory.")
    print("NOTE: Review generated configs, especially algorithm/attack-specific params marked TODO or using defaults.")
