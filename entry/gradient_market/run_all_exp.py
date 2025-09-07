# log_utils.py (or results_logger.py)

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import torch
import torch.nn as nn

from common.datasets.dataset import get_image_dataset, get_text_dataset
from common.evaluators import create_evaluators
from common.factories import SellerFactory
from common.gradient_market_configs import AppConfig
from common.utils import FederatedEarlyStopper
from entry.gradient_market.automate_exp.config_parser import load_config
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.gradient_seller import SybilCoordinator
from model.utils import get_text_model, get_image_model


def setup_data_and_model(cfg: AppConfig):
    """Loads dataset and creates a model factory from the AppConfig."""
    dataset_name = cfg.experiment.dataset_name

    is_text = cfg.experiment.dataset_type == "text"

    if is_text:
        processed_data = get_text_dataset(cfg)
        buyer_loader = processed_data.buyer_loader
        seller_loaders = processed_data.seller_loaders
        test_loader = processed_data.test_loader
        num_classes = processed_data.num_classes
        vocab = processed_data.vocab
        pad_idx = processed_data.pad_idx

        model_init_cfg = {"num_classes": num_classes, "vocab_size": len(vocab), "padding_idx": pad_idx,
                          "dataset_name": dataset_name}
        model_factory = lambda: get_text_model(model_name=cfg.experiment.model_structure, **model_init_cfg)
        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx, "model_type": "text"}
    else:  # Image
        buyer_loader, seller_loaders, test_loader, stats, num_classes = get_image_dataset(cfg)

        dataset_name_lower = cfg.experiment.dataset_name.lower()
        in_channels = 3 if dataset_name_lower in ["celeba", "camelyon16"] else 1
        model_init_cfg = {"num_classes": num_classes, "in_channels": in_channels}
        model_factory = lambda: get_image_model(model_name=cfg.experiment.model_structure, **model_init_cfg)
        seller_extra_args = {"model_type": "image"}

    cfg.experiment.num_classes = num_classes
    logging.info(f"Data loaded for '{dataset_name}'. Number of classes: {cfg.experiment.num_classes}")

    return buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args


def initialize_sellers(cfg: AppConfig, marketplace, client_loaders, model_factory, seller_extra_args,
                       sybil_coordinator):
    """Creates and registers all sellers in the marketplace using the factory."""
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


def run_final_evaluation_and_logging(
        cfg: AppConfig,
        final_model: nn.Module,
        results_buffer: List[Dict],
        test_loader,
        evaluators
):
    """Performs a final evaluation and saves all experiment artifacts."""
    logging.info("\n--- Final Evaluation and Logging ---")

    # 1. Save the round-by-round training log
    if results_buffer:
        results_df = pd.DataFrame(results_buffer)
        log_path = Path(cfg.experiment.save_path) / "training_log.csv"
        results_df.to_csv(log_path, index=False)
        logging.info(f"Full training log saved to {log_path}")

    # 2. Perform a final, high-quality evaluation on the trained model
    logging.info("Performing final evaluation on the trained model...")
    final_metrics = {}
    for evaluator in evaluators:
        metrics = evaluator.evaluate(final_model, test_loader)
        final_metrics.update(metrics)
    logging.info(f"Final Model Performance: {final_metrics}")

    # Optionally, save the final metrics to a file as well
    final_metrics_path = Path(cfg.experiment.save_path) / "final_metrics.json"
    with open(final_metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    logging.info(f"Final metrics saved to {final_metrics_path}")

    # 3. Save the final model state
    model_path = Path(cfg.experiment.save_path) / "final_model.pth"
    torch.save(final_model.state_dict(), model_path)
    logging.info(f"Final model state dictionary saved to {model_path}")


def run_training_loop(
        cfg: AppConfig,
        marketplace,
        test_loader,
        evaluators,
        sybil_coordinator
) -> Tuple[nn.Module, List[Dict]]:
    """
    Runs the main federated training rounds and returns the final model and results.
    """
    logging.info("\n--- Starting Federated Training Rounds ---")
    early_stopper = FederatedEarlyStopper(patience=20, monitor='acc')
    results_buffer = []

    eval_freq = cfg.experiment.evaluation_frequency

    for gr in range(cfg.experiment.global_rounds):
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} Start ===============")
        sybil_coordinator.prepare_for_new_round()

        round_record, _ = marketplace.train_federated_round(
            round_number=gr,
            ground_truth_dict={}
        )

        # Perform periodic evaluation
        is_last_round = (gr + 1) == cfg.experiment.global_rounds
        if (gr + 1) % eval_freq == 0 or is_last_round:
            logging.info(f"--- Performing Evaluation for Round {gr + 1} ---")
            global_model = marketplace.aggregator.strategy.global_model

            all_metrics = {}
            for evaluator in evaluators:
                metrics = evaluator.evaluate(global_model, test_loader)
                all_metrics.update(metrics)

            # UPDATED: Merge the marketplace's round_record with the evaluation metrics
            # This creates a complete log for the round.
            log_entry = {
                "acc": all_metrics.get("acc"),
                "loss": all_metrics.get("loss"),
                "asr": all_metrics.get("asr"),
            }
            log_entry.update(round_record)  # Merge in keys like 'round', 'duration_sec', etc.

            results_buffer.append(log_entry)
            logging.info(f"Evaluation Metrics: {log_entry}")

            if early_stopper.update(all_metrics.get('acc', 0)):
                logging.info(f"Early stopping at round {gr + 1}.")
                break

        sybil_coordinator.on_round_end()
        logging.info(f"============= Round {gr + 1}/{cfg.experiment.global_rounds} End ===============")

    # Return the final trained model and the log of results
    final_model = marketplace.aggregator.strategy.global_model
    return final_model, results_buffer


def run_attack(cfg: AppConfig):
    """Orchestrates the entire experiment from a single config object."""
    logging.info(f"--- Starting Experiment: {cfg.experiment.dataset_name} | {cfg.experiment.model_structure} ---")

    # 1. Data and Model Setup
    buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args = setup_data_and_model(cfg)

    global_model = model_factory().to(cfg.experiment.device)
    logging.info(f"Global model created and moved to {cfg.experiment.device}")

    # 3. Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # 2. FL Component Initialization
    aggregator = Aggregator(
        global_model=global_model,
        device=torch.device(cfg.experiment.device),
        loss_fn=loss_fn,
        buyer_data_loader=buyer_loader,
        agg_config=cfg.aggregation
    )

    loss_fn = nn.CrossEntropyLoss()
    sybil_coordinator = SybilCoordinator(cfg.adversary_seller_config.sybil, aggregator)
    evaluators = create_evaluators(cfg, cfg.experiment.device, **seller_extra_args)

    # Get a sample batch to determine input shape (your method is perfect)
    sample_data = None
    if test_loader:
        try:
            # Try to get a sample from the test loader first
            sample_data, _ = next(iter(test_loader))
        except StopIteration:
            logging.warning("Test loader is available but empty.")

    # If no sample was retrieved, try getting one from a seller loader as a fallback
    if sample_data is None:
        logging.warning("No test data found. Using a sample from a seller loader for initialization.")
        for loader in seller_loaders.values():
            if loader:
                try:
                    sample_data, _ = next(iter(loader))
                    break  # Success!
                except StopIteration:
                    continue  # This seller's loader is empty, try the next

    if sample_data is None:
        raise RuntimeError("Could not retrieve a sample data batch from any available loader.")

    input_shape = tuple(sample_data.shape[1:])

    # 3. Marketplace Initialization
    # The marketplace now receives the full config and all its dependencies
    marketplace = DataMarketplaceFederated(
        cfg=cfg,
        aggregator=aggregator,
        sellers={},
        input_shape=input_shape
    )

    # 4. Seller Initialization
    initialize_sellers(cfg, marketplace, seller_loaders, model_factory, seller_extra_args, sybil_coordinator)

    # 5. Federated Training Loop
    final_model, results_buffer = run_training_loop(
        cfg, marketplace, test_loader, evaluators, sybil_coordinator
    )

    # 6. Final Evaluation and Artifact Saving
    # UPDATED: Call the new function to handle all post-training tasks.
    run_final_evaluation_and_logging(
        cfg, final_model, results_buffer, test_loader, evaluators
    )

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
