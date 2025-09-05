# log_utils.py (or results_logger.py)

import argparse
import copy
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorImageGenerator, LabelFlipGenerator, \
    PoisonGenerator
from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorTextGenerator
from common.datasets.image_data_processor import get_image_dataset
from common.datasets.text_data_processor import get_text_data_set
from common.enums import PoisonType
from common.gradient_market_configs import AppConfig, BackdoorTextConfig, BackdoorImageConfig, \
    LabelFlipConfig, RuntimeDataConfig
from common.utils import FederatedEarlyStopper
from entry.gradient_market.automate_exp.config_parser import load_config
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market.utils import FederatedEvaluator
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator, \
    AdvancedPoisoningAdversarySeller
from model.utils import get_text_model, get_image_model


class SellerFactory:
    """Handles creating and configuring different seller types from a unified AppConfig."""

    ADVERSARY_CLASS_MAP = {
        "backdoor": AdvancedBackdoorAdversarySeller,
        "label_flip": AdvancedPoisoningAdversarySeller,
    }

    def __init__(self, cfg: AppConfig, model_factory: Callable[[], nn.Module], **kwargs):
        """Initializes the factory with the main application config and runtime args."""
        self.cfg = cfg
        self.model_factory = model_factory
        self.runtime_kwargs = kwargs  # For passing runtime objects like vocab

    def _create_poison_generator(self) -> Optional[PoisonGenerator]:
        """Internal factory method to create the correct poison generator based on the AppConfig."""
        poison_cfg = self.cfg.adversary_seller_config.poisoning

        is_text = self.cfg.data.text is not None

        if poison_cfg.type == PoisonType.BACKDOOR:
            if is_text:
                params = poison_cfg.text_backdoor_params
                generator_cfg = BackdoorTextConfig(
                    vocab=self.runtime_kwargs.get("vocab"),
                    target_label=params.target_label,
                    trigger_content=params.trigger_content,
                    location=params.location
                )
                return BackdoorTextGenerator(generator_cfg)
            else:  # Image
                params = poison_cfg.image_backdoor_params
                generator_cfg = BackdoorImageConfig(
                    target_label=params.target_label,
                    trigger_type=params.trigger_type,
                    blend_alpha=params.strength,
                    location=params.location
                    # channels and trigger_size are assumed to have defaults in BackdoorImageConfig
                )
                return BackdoorImageGenerator(generator_cfg)

        elif poison_cfg.type == PoisonType.LABEL_FLIP:
            params = poison_cfg.label_flip_params
            generator_cfg = LabelFlipConfig(
                num_classes=self.cfg.experiment.num_classes,
                attack_mode=params.mode.value,  # Use .value to get the string for the generator
                target_label=params.target_label
            )
            return LabelFlipGenerator(generator_cfg)

        return None  # Return None if poison type is 'none'

    def create_seller(self,
                      seller_id: str,
                      dataset: Dataset,
                      is_adversary: bool,
                      sybil_coordinator: SybilCoordinator):
        """Creates a seller instance, assembling configs and dependencies on the fly."""
        data_cfg = RuntimeDataConfig(
            dataset=dataset,
            num_classes=self.cfg.experiment.num_classes
        )
        training_cfg = self.cfg.training

        if not is_adversary:
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
                device=self.cfg.experiment.device,
                **self.runtime_kwargs
            )

        # --- Adversary Creation Logic ---
        attack_type = self.cfg.adversary_seller_config.poisoning.type

        AdversaryClass = self.ADVERSARY_CLASS_MAP.get(attack_type.value)
        poison_generator = self._create_poison_generator()

        if not AdversaryClass or not poison_generator:
            logging.warning(
                f"Attack type '{attack_type.value}' is invalid or 'none'. Creating a benign seller for {seller_id}.")
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
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
            device=self.cfg.experiment.device,
            **self.runtime_kwargs
        )


def setup_data_and_model(cfg: AppConfig):
    """Loads dataset and creates a model factory from the AppConfig."""
    dataset_name = cfg.experiment.dataset_name
    is_text = 'text' in dataset_name.lower()

    if is_text:
        ### FIXED ###
        # Renamed 'client_loaders' to 'seller_loaders' for consistency.
        _, seller_loaders, test_loader, classes, vocab, pad_idx = get_text_data_set(cfg)
        num_classes = len(classes)
        model_init_cfg = {"num_classes": num_classes, "vocab_size": len(vocab), "padding_idx": pad_idx}
        model_factory = lambda: get_text_model(model_name=cfg.experiment.model_structure, **model_init_cfg)
        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx, "model_type": "text"}
    else:  # Image
        buyer_loader, seller_loaders, test_loader, stats = get_image_dataset(cfg)
        if test_loader:
            full_dataset = test_loader.dataset.dataset
        elif buyer_loader:
            full_dataset = buyer_loader.dataset.dataset
        else:
            # Fallback to the first available seller if others don't exist
            first_seller_loader = next(iter(seller_loaders.values()), None)
            if not first_seller_loader:
                raise ValueError("No data loaders available to determine dataset properties.")
            full_dataset = first_seller_loader.dataset.dataset

        num_classes = len(np.unique(full_dataset.targets))
        model_init_cfg = {"num_classes": num_classes}
        model_factory = lambda: get_image_model(model_name=cfg.experiment.model_structure, **model_init_cfg)
        seller_extra_args = {"model_type": "image"}

    # Dynamically update num_classes in the config object for runtime use
    cfg.experiment.num_classes = num_classes
    logging.info(f"Data loaded for '{dataset_name}'. Number of classes: {cfg.experiment.num_classes}")

    # Now this return statement works for both text and image datasets
    return seller_loaders, test_loader, model_factory, seller_extra_args


def initialize_sellers(cfg: AppConfig, marketplace, client_loaders, model_factory, seller_extra_args,
                       sybil_coordinator):
    """Creates and registers all sellers in the marketplace using the factory."""
    ### UPDATED ###
    # The 'attack_generator' argument is gone. The factory handles it all.
    logging.info("--- Initializing Sellers ---")
    n_adversaries = int(cfg.experiment.n_sellers * cfg.experiment.adv_rate)
    adversary_ids = list(client_loaders.keys())[:n_adversaries]

    seller_factory = SellerFactory(cfg, model_factory, **seller_extra_args)

    for cid, loader in client_loaders.items():
        is_adv = cid in adversary_ids
        seller = seller_factory.create_seller(
            seller_id=f"{'adv' if is_adv else 'bn'}_{cid}",
            dataset=loader.dataset,
            is_adversary=is_adv,
            sybil_coordinator=sybil_coordinator
        )
        marketplace.register_seller(seller.seller_id, seller)

        if is_adv and cfg.adversary_seller_config.sybil.is_sybil:
            sybil_coordinator.register_seller(seller)

    logging.info(f"Initialized {cfg.experiment.n_sellers} sellers ({n_adversaries} adversaries).")


def run_training_loop(cfg: AppConfig, marketplace, test_loader, loss_fn, sybil_coordinator):
    """Runs the main federated training rounds."""
    logging.info("\n--- Starting Federated Training Rounds ---")
    early_stopper = FederatedEarlyStopper(patience=20, monitor='acc')

    for gr in range(cfg.experiment.global_rounds):
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} Start ===============")
        sybil_coordinator.prepare_for_new_round()

        round_record, _ = marketplace.train_federated_round(
            round_number=gr,
            test_dataloader_global=test_loader,
            loss_fn=loss_fn,
            sybil_coordinator=sybil_coordinator
        )

        if early_stopper.update(round_record.get("perf_global", {}).get('acc', 0)):
            logging.info(f"Early stopping at round {gr + 1}.")
            break

        sybil_coordinator.on_round_end()
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} End ===============")


def run_attack(cfg: AppConfig):
    """Orchestrates the entire experiment from a single config object."""
    logging.info(f"--- Starting Experiment: {cfg.experiment.dataset_name} | {cfg.experiment.model_structure} ---")

    # 1. Data and Model Setup
    client_loaders, test_loader, model_factory, seller_extra_args = setup_data_and_model(cfg)

    # 2. FL Component Initialization
    aggregator = Aggregator(cfg.experiment.aggregation_method, model_factory())
    loss_fn = nn.CrossEntropyLoss()
    evaluator = FederatedEvaluator(loss_fn, device=cfg.experiment.device)
    sybil_coordinator = SybilCoordinator(cfg.adversary_seller_config.sybil, aggregator)

    # Get a sample batch to determine input shape (your method is perfect)
    sample_data, _ = next(iter(test_loader))
    input_shape = tuple(sample_data.shape[1:])

    # 3. Marketplace Initialization
    # The marketplace now receives the full config and all its dependencies
    marketplace = DataMarketplaceFederated(
        cfg=cfg,
        aggregator=aggregator,
        evaluator=evaluator,
        sellers={},
        input_shape=input_shape
    )

    # 4. Seller Initialization
    initialize_sellers(cfg, marketplace, client_loaders, model_factory, seller_extra_args, sybil_coordinator)

    # 5. Federated Training Loop
    # Note: run_training_loop no longer needs the loss_fn or sybil_coordinator,
    # as those are handled within the marketplace or sellers.
    run_training_loop(cfg, marketplace, test_loader)
    logging.info("--- Experiment Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    cli_args = parser.parse_args()

    app_config = load_config(cli_args.config)

    initial_seed = app_config.seed
    for i in range(app_config.n_samples):
        run_cfg = copy.deepcopy(app_config)
        current_seed = initial_seed + i
        # set_seed(current_seed)

        run_save_path = Path(run_cfg.experiment.save_path) / f"run_{i}_seed_{current_seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        run_cfg.experiment.save_path = str(run_save_path)

        logging.info(f"\n{'=' * 20} Starting Run {i + 1}/{app_config.n_samples} (Seed: {current_seed}) {'=' * 20}")
        run_attack(run_cfg)


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
