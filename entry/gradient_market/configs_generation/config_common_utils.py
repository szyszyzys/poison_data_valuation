# ==============================================================================
# --- 1. USER ACTION: Define Golden Training HPs (from Step 1 IID Tune) ---
# ==============================================================================
# --- CRITICAL: FILL THESE WITH YOUR ACTUAL RESULTS ---
from typing import Dict, Any, Callable, Optional

from common_utils.constants.enums import PoisonType
from marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig

GOLDEN_TRAINING_PARAMS = {
    # --- IMAGE MODELS ---
    "cifar10_cnn": {
        # Best: 0.8248 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "cifar10_resnet18": {
        # Best: 0.8092 acc
        "training.optimizer": "SGD", "training.learning_rate": 0.1, "training.local_epochs": 2,
        "training.momentum": 0.9, "training.weight_decay": 5e-4,  # (Assuming standard SGD params)
    },
    "cifar100_cnn": {
        # Best: 0.5536 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "cifar100_resnet18": {
        # NOTE: No resnet18 cifar100 results were in your table.
        # We will re-use the flexiblecnn HPs as a placeholder.
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },

    # --- TABULAR MODELS ---
    "mlp_texas100_baseline": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "mlp_purchase100_baseline": {
        # Best: 0.6002 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },

    # --- TEXT MODELS ---
    "textcnn_trec_baseline": {
        # Best: 0.7985 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0
    }
}

# In config_common_utils.py

# This new structure is keyed by:
# {defense_name}_{model_config_name}_{attack_type}
# You MUST fill this with your results from Step 3 analysis

TUNED_DEFENSE_PARAMS = {
    "fedavg_cifar100_cnn_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_cifar100_cnn_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_cifar10_cnn_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_cifar10_cnn_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_texas100_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_texas100_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_textcnn_trec_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_textcnn_trec_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fltrust_cifar100_cnn_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_cifar100_cnn_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_cifar10_cnn_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_cifar10_cnn_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_texas100_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_texas100_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_textcnn_trec_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_textcnn_trec_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "martfl_cifar100_cnn_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "martfl_cifar100_cnn_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_cifar10_cnn_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0},
    "martfl_cifar10_cnn_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_texas100_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_texas100_baseline_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 7},
    "martfl_textcnn_trec_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "martfl_textcnn_trec_baseline_labelflip": {'aggregation.method': 'martfl',  'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "skymask_cifar100_cnn_backdoor": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar100_cnn_labelflip": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar10_cnn_backdoor": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar10_cnn_labelflip": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar100_cnn_backdoor": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar100_cnn_labelflip": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar10_cnn_backdoor": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar10_cnn_labelflip": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},

}

def get_tuned_defense_params(
        defense_name: str,
        model_config_name: str,
        attack_state: str,
        explicit_attack_type: Optional[str] = None,
        default_attack_type_for_tuning: str = "backdoor"
) -> Optional[Dict[str, Any]]:  # <-- Return Optional
    """
    Intelligently retrieves the correct tuned defense parameters from the
    global TUNED_DEFENSE_PARAMS dictionary.
    """

    if defense_name == "fedavg":
        return {"aggregation.method": "fedavg"}

    # --- THIS IS THE NEW, CORRECTED LOGIC ---
    attack_type_key = default_attack_type_for_tuning  # Start with the default

    if explicit_attack_type:
        # If Step 5 or 12 passes "backdoor" or "labelflip", use it.
        attack_type_key = explicit_attack_type
    elif attack_state == "no_attack":
        # For a no_attack run (like in Step 4), use the default.
        attack_type_key = default_attack_type_for_tuning
    # If attack_state is "with_attack" and no explicit_type,
    # it will also (correctly) use the default.
    # --- END NEW LOGIC ---

    # Build the specific key
    tuned_params_key = f"{defense_name}_{model_config_name}_{attack_type_key}"

    if tuned_params_key not in TUNED_DEFENSE_PARAMS:
        print(f"!!!!!!!!!! FATAL WARNING !!!!!!!!!!!")
        print(f"  Could not find tuned params for key: '{tuned_params_key}'")
        print(f"  Please check your TUNED_DEFENSE_PARAMS in config_common_utils.py")

        # --- FIX 2: Return None. Do NOT fall back. ---
        return None

    return TUNED_DEFENSE_PARAMS[tuned_params_key]

# This helper list can now be simplified
ALL_DEFENSES = ["fedavg", "fltrust", "martfl", "skymask", "skymask_small"]
IMAGE_DEFENSES = ["fltrust", "martfl", "skymask"]
TEXT_TABULAR_DEFENSES = ["fltrust", "martfl"]

# ==============================================================================
# --- 3. Define Shared Parameters & Modifiers ---
# ==============================================================================
NUM_SEEDS_PER_CONFIG = 3
DEFAULT_ADV_RATE = 0.3
DEFAULT_POISON_RATE = 0.5  # Match defense tuning


def disable_all_attacks(config: AppConfig) -> AppConfig:
    """
    This modifier disables all attack flags across the entire configuration
    to create a purely benign (non-adversarial) experiment setting.
    """

    # 1. Set main adversary rate to 0
    config.experiment.adv_rate = 0.0

    # 2. Disable all Adversarial Seller attacks
    adv_seller_cfg = config.adversary_seller_config
    adv_seller_cfg.poisoning.type = PoisonType.NONE
    adv_seller_cfg.sybil.is_sybil = False
    adv_seller_cfg.adaptive_attack.is_active = False
    adv_seller_cfg.drowning_attack.is_active = False
    adv_seller_cfg.mimicry_attack.is_active = False

    # 3. Disable all Server attacks (e.g., gradient inversion)
    config.server_attack_config.attack_name = "none"

    # 4. Disable all Buyer attacks
    config.buyer_attack_config.is_active = False
    config.buyer_attack_config.attack_type = "none"

    return config

# --- Valuation Config Helper ---
def enable_valuation(config: AppConfig, influence: bool = True, loo: bool = False, kernelshap: bool = False,
                     loo_freq: int = 10, kshap_freq: int = 20,
                     kshap_samples: int = 500) -> AppConfig:  # <-- ADD kshap_samples HERE

    config.valuation.run_influence = influence
    config.valuation.run_loo = loo
    config.valuation.run_kernelshap = kernelshap
    config.valuation.loo_frequency = loo_freq
    config.valuation.kernelshap_frequency = kshap_freq

    # Add this line to actually use the new parameter
    config.valuation.kernelshap_samples = kshap_samples

    return config


def use_sybil_attack_strategy(strategy: str, **kwargs) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier function that enables Sybil attack with a specific strategy
    and optional strategy-specific parameters.

    Args:
        strategy: "oracle_blend", "systematic_probe", "mimic", "pivot", etc.
        **kwargs: Additional parameters for the strategy (e.g., blend_alpha for oracle)
    """

    def modifier(config: AppConfig) -> AppConfig:
        sybil_cfg = config.adversary_seller_config.sybil
        sybil_cfg.is_sybil = True
        sybil_cfg.gradient_default_mode = strategy

        # Add any extra strategy-specific parameters
        # Example: Oracle blending factor
        if strategy == "oracle_blend":
            sybil_cfg.oracle_blend_alpha = kwargs.get("blend_alpha", 0.1)  # Default 10% malicious

        # Example: Parameters for systematic probing (if needed in config)
        # if strategy == "systematic_probe":
        #     sybil_cfg.probe_num_variations = kwargs.get("num_probes", 5)
        #     sybil_cfg.probe_noise_scale = kwargs.get("probe_scale", 0.01)

        # Ensure base poisoning is active (Sybil modifies *how* poison is delivered)
        # Assuming the attack modifier (e.g., use_image_backdoor_attack) is also applied
        # If not, you might need to set config.adversary_seller_config.poisoning.type here too.

        return config

    return modifier
