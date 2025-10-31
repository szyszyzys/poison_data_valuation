# ==============================================================================
# --- 1. USER ACTION: Define Golden Training HPs (from Step 1 IID Tune) ---
# ==============================================================================
# --- CRITICAL: FILL THESE WITH YOUR ACTUAL RESULTS ---
from typing import Dict, Any, Callable

from common.enums import PoisonType
from common.gradient_market_configs import AppConfig
from entry.gradient_market.automate_exp.config_generator import set_nested_attr


GOLDEN_TRAINING_PARAMS = {
    # --- IMAGE MODELS ---
    "cifar10_flexiblecnn": {
        # Best: 0.8248 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "cifar10_resnet18": {
        # Best: 0.8092 acc
        "training.optimizer": "SGD", "training.learning_rate": 0.1, "training.local_epochs": 2,
        "training.momentum": 0.9, "training.weight_decay": 5e-4, # (Assuming standard SGD params)
    },
    "cifar100_flexiblecnn": {
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
        # Best: 0.6250 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.0001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "mlp_purchase100_baseline": {
        # Best: 0.6002 acc
        "training.optimizer": "Adam", "training.learning_rate": 0.0005, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },

    # --- TEXT MODELS ---
    "textcnn_trec_baseline": {
        # Best: 0.7985 acc
        "training.optimizer": "SGD", "training.learning_rate": 0.05, "training.local_epochs": 5,
        "training.momentum": 0.9, "training.weight_decay": 0.0001
    }
}

TUNED_DEFENSE_PARAMS = {
    "fedavg": {"aggregation.method": "fedavg"},
    "fltrust": {"aggregation.method": "fltrust", "aggregation.clip_norm": 10.0},
    "martfl": {"aggregation.method": "martfl", "aggregation.martfl.max_k": 5, "aggregation.clip_norm": 10.0},
    "skymask": {  # Add all relevant tuned SkyMask params
        "aggregation.method": "skymask", "aggregation.skymask.mask_epochs": 20,
        "aggregation.skymask.mask_lr": 0.01, "aggregation.skymask.mask_threshold": 0.7,
        "aggregation.clip_norm": 10.0
    },
}
# Define which defenses are compatible with which modality
ALL_DEFENSES = list(TUNED_DEFENSE_PARAMS.keys())
IMAGE_DEFENSES = ["fedavg", "fltrust", "martfl", "skymask"]
TEXT_TABULAR_DEFENSES = ["fedavg", "fltrust", "martfl"]  # Exclude SkyMask

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

# --- Helper to apply fixed Golden Training & Tuned Defense HPs ---
def create_fixed_params_modifier(
        modality: str,
        defense_params: Dict[str, Any],
        model_config_name: str,
        apply_noniid: bool = True  # Flag to control data distribution
) -> Callable[[AppConfig], AppConfig]:
    def modifier(config: AppConfig) -> AppConfig:
        # 1. Apply Golden Training HPs
        training_params = GOLDEN_TRAINING_PARAMS.get(model_config_name)
        if training_params:
            for key, value in training_params.items():
                set_nested_attr(config, key, value)
        # 2. Apply Tuned Defense HPs
        for key, value in defense_params.items():
            set_nested_attr(config, key, value)
        # 3. Set SkyMask model type if needed
        if defense_params.get("aggregation.method") == "skymask":
            model_struct = "resnet18" if "resnet" in model_config_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
        # 4. Ensure Correct Data Distribution
        if apply_noniid:
            set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
            set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        else:  # Apply IID
            set_nested_attr(config, f"data.{modality}.strategy", "iid")
            # Remove alpha if it exists from base config
            if hasattr(config.data, modality) and hasattr(getattr(config.data, modality), 'dirichlet_alpha'):
                delattr(getattr(config.data, modality), 'dirichlet_alpha')

        return config

    return modifier


# --- Valuation Config Helper ---
def enable_valuation(config: AppConfig, influence: bool = True, loo: bool = False, kernelshap: bool = False,
                     loo_freq: int = 10, kshap_freq: int = 20) -> AppConfig:
    config.valuation.run_influence = influence
    config.valuation.run_loo = loo
    config.valuation.run_kernelshap = kernelshap
    config.valuation.loo_frequency = loo_freq
    config.valuation.kernelshap_frequency = kshap_freq
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
            sybil_cfg.oracle_blend_alpha = kwargs.get("blend_alpha", 0.1) # Default 10% malicious

        # Example: Parameters for systematic probing (if needed in config)
        # if strategy == "systematic_probe":
        #     sybil_cfg.probe_num_variations = kwargs.get("num_probes", 5)
        #     sybil_cfg.probe_noise_scale = kwargs.get("probe_scale", 0.01)

        # Ensure base poisoning is active (Sybil modifies *how* poison is delivered)
        # Assuming the attack modifier (e.g., use_image_backdoor_attack) is also applied
        # If not, you might need to set config.adversary_seller_config.poisoning.type here too.

        return config
    return modifier

