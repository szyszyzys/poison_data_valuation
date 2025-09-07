import logging
from typing import Callable, Optional
from typing import Dict, Any

import torch.nn as nn
from torch.utils.data import Dataset

from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorImageGenerator, LabelFlipGenerator, \
    PoisonGenerator
from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorTextGenerator
from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, BackdoorTextConfig, BackdoorImageConfig, \
    LabelFlipConfig, RuntimeDataConfig
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator, \
    AdvancedPoisoningAdversarySeller


class SellerFactory:
    """Handles creating and configuring different seller types from a unified AppConfig."""

    ADVERSARY_CLASS_MAP = {
        "image_backdoor": AdvancedBackdoorAdversarySeller,
        "text_backdoor": AdvancedBackdoorAdversarySeller,
        "label_flip": AdvancedPoisoningAdversarySeller,
    }

    def __init__(self, cfg: AppConfig, model_factory: Callable[[], nn.Module], **kwargs):
        """Initializes the factory with the main application config and runtime args."""
        self.cfg = cfg
        self.model_factory = model_factory
        self.runtime_kwargs = kwargs  # For passing runtime objects like vocab

    def create_seller(self,
                      seller_id: str,
                      dataset: Dataset,
                      is_adversary: bool,
                      sybil_coordinator: SybilCoordinator,
                      collate_fn: Callable = None):
        """Creates a seller instance, assembling configs and dependencies on the fly."""
        data_cfg = RuntimeDataConfig(
            dataset=dataset,
            num_classes=self.cfg.experiment.num_classes,
            collate_fn=collate_fn
        )
        training_cfg = self.cfg.training

        if not is_adversary:
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
                save_path=self.cfg.experiment.save_path,
                device=self.cfg.experiment.device,
                **self.runtime_kwargs
            )

        # --- Adversary Creation Logic ---
        attack_type = self.cfg.adversary_seller_config.poisoning.type
        AdversaryClass = self.ADVERSARY_CLASS_MAP.get(attack_type.value)

        # Call the central factory function
        poison_generator = create_poison_generator(self.cfg, **self.runtime_kwargs)

        if not AdversaryClass or not poison_generator:
            logging.warning(
                f"Attack type '{attack_type.value}' is invalid or 'none'. Creating a benign seller for {seller_id}.")
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
                save_path=self.cfg.experiment.save_path,
                device=self.cfg.experiment.device,
                **self.runtime_kwargs
            )

        return AdversaryClass(
            seller_id=seller_id,
            data_config=data_cfg,
            training_config=training_cfg,
            model_factory=self.model_factory,
            adversary_config=self.cfg.adversary_seller_config,
            sybil_coordinator=sybil_coordinator,
            poison_generator=poison_generator,
            save_path=self.cfg.experiment.save_path,
            device=self.cfg.experiment.device,
            **self.runtime_kwargs
        )


def create_poison_generator(cfg: AppConfig, **kwargs: Dict[str, Any]) -> Optional[PoisonGenerator]:
    """
    A centralized factory function to create the correct poison generator based on the AppConfig.

    Args:
        cfg: The main application configuration.
        **kwargs: Catches runtime arguments like 'vocab' needed for text attacks.
    """
    poison_cfg = cfg.adversary_seller_config.poisoning
    active_params = poison_cfg.active_params

    if not active_params:
        return None

    if poison_cfg.type == PoisonType.IMAGE_BACKDOOR:
        generator_cfg = BackdoorImageConfig(
            target_label=active_params.target_label,
            trigger_type=active_params.trigger_type,
            location=active_params.location,
            blend_alpha=active_params.strength,
            channels=active_params.pattern_channel,
            trigger_size=active_params.trigger_shape
        )
        return BackdoorImageGenerator(generator_cfg)

    elif poison_cfg.type == PoisonType.TEXT_BACKDOOR:
        generator_cfg = BackdoorTextConfig(
            vocab=kwargs.get("vocab"),
            target_label=active_params.target_label,
            trigger_content=active_params.trigger_content,
            location=active_params.location
        )
        return BackdoorTextGenerator(generator_cfg)

    elif poison_cfg.type == PoisonType.LABEL_FLIP:
        generator_cfg = LabelFlipConfig(
            num_classes=cfg.experiment.num_classes,
            attack_mode=active_params.mode.value,
            target_label=active_params.target_label
        )
        return LabelFlipGenerator(generator_cfg)

    logging.warning(f"Unknown or unhandled poison type: {poison_cfg.type}")
    return None
