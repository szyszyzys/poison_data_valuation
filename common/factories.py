import logging
from typing import Callable, Optional, Dict, Any

from torch import nn
from torch.utils.data import Dataset

# Import the specific generator classes needed for the generic poisoner
from attack.attack_gradient_market.poison_attack.attack_utils import (
    LabelFlipGenerator, PoisonGenerator
)
from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, RuntimeDataConfig
from common.gradient_market_configs import LabelFlipConfig
# Import the specific seller classes
from marketplace.seller.gradient_seller import (
    GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator,
    AdvancedPoisoningAdversarySeller
)


class SellerFactory:
    """Handles creating and configuring different seller types from a unified AppConfig."""

    # Map the ENUM directly to the class for better type safety
    ADVERSARY_CLASS_MAP = {
        PoisonType.IMAGE_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.TEXT_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.TABULAR_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.LABEL_FLIP: AdvancedPoisoningAdversarySeller,
    }

    def __init__(self, cfg: AppConfig, model_factory: Callable[[], nn.Module], num_classes: int, **kwargs):
        """Initializes the factory with the main application config and runtime args."""
        self.cfg = cfg
        self.model_factory = model_factory
        self.num_classes = num_classes  # Store it
        self.runtime_kwargs = kwargs

    def create_seller(self,
                      seller_id: str,
                      dataset: Dataset,
                      is_adversary: bool,
                      sybil_coordinator: SybilCoordinator,
                      collate_fn: Callable = None):
        """Creates a seller instance, assembling configs and dependencies on the fly."""
        data_cfg = RuntimeDataConfig(
            dataset=dataset,
            num_classes=self.num_classes,  # Use it here
            collate_fn=collate_fn
        )
        training_cfg = self.cfg.training
        base_kwargs = {
            "seller_id": seller_id,
            "data_config": data_cfg,
            "training_config": training_cfg,
            "model_factory": self.model_factory,
            "save_path": self.cfg.experiment.save_path,
            "device": self.cfg.experiment.device,
        }

        if not is_adversary:
            return GradientSeller(**base_kwargs, **self.runtime_kwargs)

        # --- UPDATED: Adversary Creation Logic ---
        attack_type = self.cfg.adversary_seller_config.poisoning.type
        AdversaryClass = self.ADVERSARY_CLASS_MAP.get(attack_type)

        if not AdversaryClass:
            logging.warning(f"Attack type '{attack_type.value}' is invalid. Creating a benign seller.")
            return GradientSeller(**base_kwargs, **self.runtime_kwargs)

        # ** THE CORE CHANGE IS HERE **
        # The factory now knows which sellers need a pre-built generator
        # and which ones build their own.
        if AdversaryClass is AdvancedBackdoorAdversarySeller:
            # For the backdoor seller, we do NOT pass a poison_generator.
            # Instead, we pass the `model_type` it needs to build its own.
            logging.info(f"Creating AdvancedBackdoorAdversarySeller for {seller_id}")
            # 1. Make a copy of kwargs to avoid modifying the original.
            kwargs_for_seller = self.runtime_kwargs.copy()
            # 2. Safely remove the conflicting key.
            kwargs_for_seller.pop('model_type', None)

            return AdvancedBackdoorAdversarySeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                sybil_coordinator=sybil_coordinator,
                model_type=self.cfg.experiment.dataset_type,
                **kwargs_for_seller  # 3. Unpack the cleaned dictionary
            )

        elif AdversaryClass is AdvancedPoisoningAdversarySeller:
            # For the generic poisoner (e.g., label-flip), we DO create
            # and pass in the poison_generator.
            logging.info(f"Creating AdvancedPoisoningAdversarySeller for {seller_id}")
            poison_generator = create_generic_poison_generator(self.cfg, **self.runtime_kwargs)
            if not poison_generator:
                logging.warning(f"Could not create generator for '{attack_type.value}'. Creating benign seller.")
                return GradientSeller(**base_kwargs, **self.runtime_kwargs)

            return AdvancedPoisoningAdversarySeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                sybil_coordinator=sybil_coordinator,
                poison_generator=poison_generator,
                **self.runtime_kwargs
            )

        # Fallback for any unhandled adversary types
        logging.warning(f"Unhandled adversary class '{AdversaryClass.__name__}'. Creating benign seller.")
        return GradientSeller(**base_kwargs, **self.runtime_kwargs)


def create_generic_poison_generator(cfg: AppConfig, **kwargs: Dict[str, Any]) -> Optional[PoisonGenerator]:
    """
    A simplified factory that creates generators for generic poisoning attacks.
    ** Backdoor logic has been removed, as that is now handled inside the
    ** AdvancedBackdoorAdversarySeller class itself.
    """
    poison_cfg = cfg.adversary_seller_config.poisoning
    active_params = poison_cfg.active_params

    if not active_params:
        return None

    if poison_cfg.type == PoisonType.LABEL_FLIP:
        generator_cfg = LabelFlipConfig(
            num_classes=cfg.experiment.num_classes,
            attack_mode=active_params.mode.value,
            target_label=active_params.target_label
        )
        return LabelFlipGenerator(generator_cfg)

    # This function would be extended for other *generic* attacks like noise, etc.
    logging.warning(f"Unhandled poison type in generic generator factory: {poison_cfg.type}")
    return None
