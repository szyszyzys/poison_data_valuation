# log_utils.py (or results_logger.py)

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

from common.datasets.dataset import get_image_dataset, get_text_dataset
from common.datasets.tabular_data_processor import get_dataset_tabular
from common.datasets.text_data_processor import collate_batch
from common.evaluators import create_evaluators
from common.factories import SellerFactory
from common.gradient_market_configs import AppConfig
from common.utils import FederatedEarlyStopper, set_seed
from entry.gradient_market.automate_exp.config_parser import load_config
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.gradient_seller import SybilCoordinator
from model.image_model import ImageModelFactory
from model.model_configs import get_image_model_config
from model.utils import get_text_model


def setup_data_and_model(cfg: AppConfig):
    """Loads dataset and creates a model factory from the AppConfig."""
    dataset_name = cfg.experiment.dataset_name
    dataset_type = cfg.experiment.dataset_type
    collate_fn = None

    if dataset_type == "text":
        processed_data = get_text_dataset(cfg)
        buyer_loader = processed_data.buyer_loader
        seller_loaders = processed_data.seller_loaders
        test_loader = processed_data.test_loader
        num_classes = processed_data.num_classes
        vocab = processed_data.vocab
        pad_idx = processed_data.pad_idx
        collate_fn = lambda batch: collate_batch(batch, padding_value=pad_idx)

        model_init_cfg = {"num_classes": num_classes, "vocab_size": len(vocab), "padding_idx": pad_idx,
                          "dataset_name": dataset_name}
        model_factory = lambda: get_text_model(model_name=cfg.experiment.model_structure, **model_init_cfg)
        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx}

    elif dataset_type == "image":
        buyer_loader, seller_loaders, test_loader, stats, num_classes = get_image_dataset(cfg)

        image_model_config = get_image_model_config(cfg.experiment.image_model_config_name)
        logging.info(f"DEBUG: Intended model config name from cfg: '{cfg.experiment.image_model_config_name}'")
        image_model_config = get_image_model_config(cfg.experiment.image_model_config_name)
        logging.info(f"DEBUG: Loaded model config for '{image_model_config.model_name}' with recipe '{image_model_config.config_name}'")

        # 2. Determine other parameters needed for model creation
        in_channels = 3 # CIFAR datasets have 3 channels
        sample_data, _ = next(iter(test_loader))
        image_size = tuple(sample_data.shape[2:])

        # 3. The model_factory uses the config object loaded by name
        model_factory = lambda: ImageModelFactory.create_model(
            # The model_name now comes from the loaded recipe, ensuring consistency
            model_name=image_model_config.model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            config=image_model_config
        )

        seller_extra_args = {}
    elif dataset_type == "tabular":
        # --- NEW LOGIC FOR TABULAR DATA ---
        with open("configs/tabular_datasets.yaml", 'r') as f:
            all_tabular_configs = yaml.safe_load(f)
        d_cfg = all_tabular_configs[dataset_name]

        df, categorical_cols = get_dataset_tabular(config=d_cfg)

        # Preprocess: One-hot encode categorical features and scale numerical ones
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df.drop(columns=[d_cfg['target_column']])
        y = df[d_cfg['target_column']]

        numerical_cols = X.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        # --- TODO: Replace this simple split with your buyer/seller logic ---
        # For now, we split into a single "seller" training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=cfg.seed
        )
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Create DataLoaders
        # NOTE: This is a simplified example. You'll need to create multiple seller_loaders.
        seller_loaders = {0: DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)}
        buyer_loader = None # No buyer data in this simple split
        test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

        # --- TODO: Define a model factory for your tabular models ---
        # Example: model_factory = lambda: YourMLPModel(input_features=X_tensor.shape[1], ...)
        model_factory = None

        seller_extra_args = {}
        collate_fn = None
        num_classes = len(torch.unique(y_tensor))

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    cfg.experiment.num_classes = num_classes
    logging.info(f"Data loaded for '{dataset_name}'. Number of classes: {cfg.experiment.num_classes}")

    logging.info(f"Data loaded for '{dataset_name}'. Number of classes: {cfg.experiment.num_classes}")

    return buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args, collate_fn, num_classes


def initialize_sellers(cfg: AppConfig, marketplace, client_loaders, model_factory, seller_extra_args,
                       sybil_coordinator, collate_fn, num_classes: int):  # <-- Accept num_classes
    """Creates and registers all sellers in the marketplace using the factory."""
    logging.info("--- Initializing Sellers ---")
    n_adversaries = int(cfg.experiment.n_sellers * cfg.experiment.adv_rate)
    adversary_ids = list(client_loaders.keys())[:n_adversaries]

    # --- FIX: Pass num_classes to the factory's constructor ---
    seller_factory = SellerFactory(cfg, model_factory, num_classes=num_classes, **seller_extra_args)

    for cid, loader in client_loaders.items():
        is_adv = cid in adversary_ids
        seller = seller_factory.create_seller(
            seller_id=f"{'adv' if is_adv else 'bn'}_{cid}",
            dataset=loader.dataset,
            is_adversary=is_adv,
            sybil_coordinator=sybil_coordinator,
            collate_fn=collate_fn
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

    # --- 1. NEW: Add Caching Logic at the Beginning ---
    save_path = Path(cfg.experiment.save_path)
    # This success marker should be the very last file created by a successful run.
    success_marker = save_path / "final_metrics.json"

    if success_marker.exists():
        logging.info(f"✅ Results for this configuration already exist. Skipping re-run.")
        logging.info(f"   - Cached results can be found at: {save_path}")
        return  # <-- Exit the function early, skipping all computation

    # --- If no cache is found, the original function proceeds as normal ---

    logging.info(f"--- Starting Experiment: {cfg.experiment.dataset_name} | {cfg.experiment.model_structure} ---")
    logging.info(f"   - Results will be saved to: {save_path}")

    # 2. Data and Model Setup
    buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args, collate_fn, num_classes = \
        setup_data_and_model(cfg)

    global_model = model_factory().to(cfg.experiment.device)
    logging.info(f"Global model created and moved to {cfg.experiment.device}")
    logging.info(f"--- Global Model Architecture ---\n{global_model}")

    # 3. FL Component Initialization (renumbered for clarity)
    loss_fn = nn.CrossEntropyLoss()
    aggregator = Aggregator(
        global_model=global_model,
        device=torch.device(cfg.experiment.device),
        loss_fn=loss_fn,
        buyer_data_loader=buyer_loader,
        agg_config=cfg.aggregation
    )
    sybil_coordinator = SybilCoordinator(cfg.adversary_seller_config.sybil, aggregator)
    evaluators = create_evaluators(cfg, cfg.experiment.device, **seller_extra_args)

    # 4. Marketplace Initialization
    # ... (the rest of your function remains exactly the same) ...
    sample_data = None
    if test_loader:
        try:
            sample_data, _ = next(iter(test_loader))
        except StopIteration:
            logging.warning("Test loader is available but empty.")
    if sample_data is None:
        logging.warning("No test data found. Using a sample from a seller loader for initialization.")
        for loader in seller_loaders.values():
            if loader:
                try:
                    sample_data, _ = next(iter(loader))
                    break
                except StopIteration:
                    continue
    if sample_data is None:
        raise RuntimeError("Could not retrieve a sample data batch from any available loader.")
    input_shape = tuple(sample_data.shape[1:])
    marketplace = DataMarketplaceFederated(
        cfg=cfg,
        aggregator=aggregator,
        sellers={},
        input_shape=input_shape
    )

    # 5. Seller Initialization
    initialize_sellers(cfg, marketplace, seller_loaders, model_factory, seller_extra_args,
                       sybil_coordinator, collate_fn, num_classes)

    # 6. Federated Training Loop
    final_model, results_buffer = run_training_loop(
        cfg, marketplace, test_loader, evaluators, sybil_coordinator
    )

    # 7. Final Evaluation and Artifact Saving
    run_final_evaluation_and_logging(
        cfg, final_model, results_buffer, test_loader, evaluators
    )
    for sid, seller in marketplace.sellers.items():
        seller.save_round_history_csv()

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
        set_seed(current_seed)

        run_save_path = Path(run_cfg.experiment.save_path) / f"run_{i}_seed_{current_seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        run_cfg.experiment.save_path = str(run_save_path)

        logging.info(f"\n{'=' * 20} Starting Run {i + 1}/{app_config.n_samples} (Seed: {current_seed}) {'=' * 20}")
        run_attack(run_cfg)


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
