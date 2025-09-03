# log_utils.py (or results_logger.py)

import argparse
import logging
from pathlib import Path
from typing import Callable

import torch.nn as nn
from torch.utils.data import Dataset

from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorImageGenerator, LabelFlipGenerator
from attack.attack_gradient_market.poison_attack.attack_utils import BackdoorTextGenerator
from common.gradient_market_configs import ExperimentConfig, TrainingConfig, DataConfig, AppConfig
from entry.gradient_market.backdoor_attack import FederatedEarlyStopper, load_config
from marketplace.market.markplace_gradient import DataMarketplaceFederated, MarketplaceConfig
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator, \
    AdvancedPoisoningAdversarySeller
from marketplace.utils.gradient_market_utils.data_processor import AttackConfig, setup_federated_marketplace
from marketplace.utils.gradient_market_utils.text_data_processor import get_text_data_set
from model.utils import get_text_model, get_image_model


class SellerFactory:
    """Handles creating and configuring different seller types from a unified AppConfig."""

    # A map to make adding new adversary types easy and clean
    ADVERSARY_CLASS_MAP = {
        "backdoor": AdvancedBackdoorAdversarySeller,
        "label_flip": AdvancedPoisoningAdversarySeller,
    }

    def __init__(self, cfg: AppConfig, model_factory: Callable[[], nn.Module], **kwargs):
        """Initializes the factory with the main application config."""
        self.cfg = cfg
        self.model_factory = model_factory
        self.kwargs = kwargs  # For passing runtime objects like vocab and model_type

    def create_seller(self,
                      seller_id: str,
                      dataset: Dataset,
                      is_adversary: bool,
                      sybil_coordinator: SybilCoordinator):
        """
        Creates a seller instance, assembling the necessary config objects on the fly.
        """
        # --- 1. Assemble the common config objects all sellers need ---
        data_cfg = DataConfig(
            dataset=dataset,
            num_classes=self.cfg.experiment.num_classes
        )

        # The TrainingConfig is already perfectly structured in AppConfig
        training_cfg = self.cfg.training

        # --- 2. Create a benign seller if not an adversary ---
        if not is_adversary:
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
                device=self.cfg.experiment.device,
                **self.kwargs
            )

        # --- 3. Logic for creating an adversary ---

        # Determine which adversary class to use from the config
        attack_type = self.cfg.adversary_seller_config.poisoning.type
        AdversaryClass = self.ADVERSARY_CLASS_MAP.get(attack_type)

        if not AdversaryClass:
            # Fallback to a benign seller if the attack type is unknown or 'none'
            logging.warning(f"No adversary class found for attack type '{attack_type}'. Creating a benign seller.")
            return GradientSeller(
                seller_id=seller_id,
                data_config=data_cfg,
                training_config=training_cfg,
                model_factory=self.model_factory,
                device=self.cfg.experiment.device,
                **self.kwargs
            )

        # The adversary class gets the same base configs plus the specific AdversarySellerConfig
        return AdversaryClass(
            seller_id=seller_id,
            data_config=data_cfg,
            training_config=training_cfg,
            model_factory=self.model_factory,
            adversary_config=self.cfg.adversary_seller_config,  # Pass the whole object
            sybil_coordinator=sybil_coordinator,
            device=self.cfg.experiment.device,
            **self.kwargs
        )


# ==============================================================================
# SECTION 3: UNIFIED EXPERIMENT FUNCTION
# ==============================================================================

def setup_data_and_model(exp_cfg: ExperimentConfig, train_cfg: TrainingConfig):
    """
    Loads the appropriate dataset and creates a model factory based on the experiment config.

    Returns:
        A tuple containing loaders, model factory, class information, and any extra arguments.
    """
    is_text = get_domain(exp_cfg.dataset_name) == 'text'  # Use a helper for robustness

    if is_text:
        loader_args = {"batch_size": train_cfg.batch_size, "num_sellers": exp_cfg.n_sellers}
        _, client_loaders, test_loader, classes, vocab, pad_idx = get_text_data_set(**loader_args)
        model_init_cfg = {"num_classes": len(classes), "vocab_size": len(vocab), "padding_idx": pad_idx}
        model_factory = lambda: get_text_model(dataset_name=exp_cfg.dataset_name, **model_init_cfg)
        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx, "model_type": "text"}
    else:  # Image
        loader_args = {"batch_size": train_cfg.batch_size, "num_sellers": exp_cfg.n_sellers}
        _, client_loaders, _, test_loader, classes = get_data_set(**loader_args)
        model_init_cfg = {"num_classes": len(classes)}
        model_factory = lambda: get_image_model(model_name=exp_cfg.model_name, **model_init_cfg)
        seller_extra_args = {"model_type": "image"}

    logging.info(f"Data loaded for '{exp_cfg.dataset_name}'. Number of classes: {len(classes)}")
    return client_loaders, test_loader, model_factory, model_init_cfg, seller_extra_args, len(classes)


def create_attack_generator(exp_cfg: ExperimentConfig, atk_cfg: AttackConfig, num_classes: int, **kwargs):
    """
    Initializes and returns the appropriate data poisoning attack generator.
    """
    if exp_cfg.attack_type == "backdoor":
        if get_domain(exp_cfg.dataset_name) == 'text':
            return BackdoorTextGenerator(
                kwargs.get("vocab"),
                atk_cfg.backdoor_target_label,
                atk_cfg.backdoor_trigger_content
            )
        else:  # Image
            return BackdoorImageGenerator(atk_cfg.backdoor_target_label)

    elif exp_cfg.attack_type == "label_flip":
        return LabelFlipGenerator(
            num_classes,
            atk_cfg.label_flip_mode,
            atk_cfg.label_flip_target_label
        )
    return None  # For benign experiments


def initialize_sellers(cfg: AppConfig, marketplace, client_loaders, model_factory, model_init_cfg, seller_extra_args,
                       attack_generator, sybil_coordinator):
    """Creates and registers all benign and adversarial sellers in the marketplace."""
    logging.info("--- Initializing Sellers ---")

    n_adversaries = int(cfg.experiment.n_sellers * cfg.experiment.adv_rate)
    adversary_ids = list(client_loaders.keys())[:n_adversaries]

    # The SellerFactory now takes the main config object
    seller_factory = SellerFactory(cfg, model_factory, model_init_cfg, **seller_extra_args)

    for cid, loader in client_loaders.items():
        is_adv = cid in adversary_ids
        seller_type = "Adversary" if is_adv else "Benign"

        seller = seller_factory.create_seller(
            seller_id=f"{'adv' if is_adv else 'bn'}_{cid}",
            dataset=loader.dataset,
            is_adversary=is_adv,
            attack_generator=attack_generator,
            sybil_coordinator=sybil_coordinator
        )
        marketplace.register_seller(seller.seller_id, seller)

        if is_adv and cfg.sybil.is_sybil:
            sybil_coordinator.register_seller(seller)

    logging.info(f"Initialized {cfg.experiment.n_sellers} sellers ({n_adversaries} adversaries).")
    return marketplace, sybil_coordinator


def run_training_loop(cfg: AppConfig, marketplace, test_loader, loss_fn, sybil_coordinator, attack_generator):
    """Runs the main federated training rounds, handling early stopping."""
    logging.info("\n--- Starting Federated Training Rounds ---")
    early_stopper = FederatedEarlyStopper(patience=20, monitor='acc')

    for gr in range(cfg.experiment.global_rounds):
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} Start ===============")
        sybil_coordinator.prepare_for_new_round()

        round_record, _ = marketplace.train_federated_round(
            round_number=gr,
            test_dataloader_global=test_loader,
            loss_fn=loss_fn,
            sybil_coordinator=sybil_coordinator,
            backdoor_generator=attack_generator if cfg.poisoning.type == 'backdoor' else None
        )

        if early_stopper.update(round_record.get("perf_global", {}).get('acc', 0)):
            logging.info(f"Early stopping at round {gr + 1}.")
            break

        sybil_coordinator.on_round_end()
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} End ===============")


def run_attack(cfg: AppConfig):
    """
    Orchestrates the poisoning attack experiment by calling modular helper functions.
    """
    logging.info(f"--- Starting Poisoning Attack ---")
    logging.info(
        f"Dataset: {cfg.experiment.dataset_name} | Attack: {cfg.poisoning.type} | Model: {cfg.experiment.model_structure}")

    # --- 1. Data and Model Setup ---
    # Note: These helpers would also be updated to accept 'cfg'
    client_loaders, test_loader, model_factory, model_init_cfg, seller_extra_args, num_classes = setup_data_and_model(
        cfg)

    # --- 2. Attack Generator Setup ---
    attack_generator = create_attack_generator(cfg, num_classes, **seller_extra_args)

    # --- 3. FL Component Initialization ---
    initial_model = model_factory()
    aggregator = Aggregator(cfg, initial_model)  # Aggregator can also be simplified to take cfg

    # Dynamically get input shape and create marketplace config
    sample_data, _ = next(iter(test_loader))
    marketplace_config = MarketplaceConfig(
        save_path=cfg.experiment.save_path,
        dataset_name=cfg.experiment.dataset_name,
        input_shape=tuple(sample_data.shape[1:]),
        num_classes=num_classes,
        privacy_attack_config=cfg.privacy
    )
    marketplace = DataMarketplaceFederated(aggregator, marketplace_config)

    sybil_coordinator = SybilCoordinator(cfg, aggregator, attack_generator)  # SybilCoordinator can also take cfg
    loss_fn = nn.CrossEntropyLoss()

    # --- 4. Seller Initialization (using the new helper) ---
    marketplace, sybil_coordinator = initialize_sellers(
        cfg, marketplace, client_loaders, model_factory,
        model_init_cfg, seller_extra_args, attack_generator, sybil_coordinator
    )

    # --- 5. Federated Training Loop (using the new helper) ---
    run_training_loop(cfg, marketplace, test_loader, loss_fn, sybil_coordinator, attack_generator)

    # --- 6. Save Final Results ---
    logging.info("Training finished. Saving final logs...")


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    cli_args = parser.parse_args()

    config_dict = load_config(cli_args.config)
    if not config_dict:
        return  # Exit if config loading failed

    # --- Automatically map the entire dictionary to your nested dataclasses ---
    try:
        # The main, unified config object
        cfg = from_dict(data_class=AppConfig, data=config_dict)
    except Exception as e:
        logging.error(f"Error parsing config into dataclasses: {e}")
        return

    # Loop for multiple runs with different seeds
    initial_seed = cfg.seed
    for i in range(cfg.n_samples):
        # Create a deep copy of the config for this specific run to avoid mutation issues
        run_cfg = copy.deepcopy(cfg)
        current_seed = initial_seed + i
        # set_seed(current_seed) # Assuming you have this function

        # Dynamically set the save path for this specific run inside its own config copy
        run_save_path = Path(run_cfg.experiment.save_path) / f"run_{i}_seed_{current_seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        run_cfg.experiment.save_path = str(run_save_path)

        logging.info(f"\n--- Starting Run {i} (Seed: {current_seed}) ---")

        # Pass the single, comprehensive config object for this run
        run_attack(run_cfg)


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
