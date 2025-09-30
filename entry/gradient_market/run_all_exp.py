# log_utils.py (or results_logger.py)

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
import torch.nn as nn

from common.datasets.dataset import get_image_dataset, get_text_dataset
from common.datasets.text_data_processor import collate_batch
from common.evaluators import create_evaluators
from common.factories import SellerFactory
from common.gradient_market_configs import AppConfig
from common.utils import set_seed
from entry.gradient_market.automate_exp.config_parser import load_config
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.gradient_seller import SybilCoordinator
from model.image_model import ImageModelFactory, validate_model_factory
from model.model_configs import get_image_model_config
from model.utils import get_text_model


def setup_data_and_model(cfg: AppConfig):
    """
    Loads dataset and creates a model factory from the AppConfig.

    Returns factories with frozen parameters to prevent config mutation issues.
    """
    logging.info("=" * 60)
    logging.info("Setting up data and model...")
    logging.info("=" * 60)

    dataset_name = cfg.experiment.dataset_name
    is_text = cfg.experiment.dataset_type == "text"
    collate_fn = None

    if is_text:
        # Text dataset setup
        logging.info("Loading text dataset...")
        processed_data = get_text_dataset(cfg)
        buyer_loader = processed_data.buyer_loader
        seller_loaders = processed_data.seller_loaders
        test_loader = processed_data.test_loader
        num_classes = processed_data.num_classes
        vocab = processed_data.vocab
        pad_idx = processed_data.pad_idx

        collate_fn = lambda batch: collate_batch(batch, padding_value=pad_idx)

        # Freeze parameters for text model factory
        frozen_num_classes = int(num_classes)
        frozen_vocab_size = len(vocab)
        frozen_pad_idx = int(pad_idx)
        frozen_model_name = str(cfg.experiment.model_structure)
        frozen_dataset_name = str(dataset_name)

        model_init_cfg = {
            "num_classes": frozen_num_classes,
            "vocab_size": frozen_vocab_size,
            "padding_idx": frozen_pad_idx,
            "dataset_name": frozen_dataset_name
        }

        # Create frozen factory
        def model_factory():
            return get_text_model(model_name=frozen_model_name, **model_init_cfg)

        # Validate text model factory
        try:
            test_model = model_factory()
            logging.info(f"Text model factory validated:")
            logging.info(f"  - Model: {frozen_model_name}")
            logging.info(f"  - Vocab size: {frozen_vocab_size}")
            logging.info(f"  - Classes: {frozen_num_classes}")
            logging.info(f"  - Parameters: {sum(p.numel() for p in test_model.parameters()):,}")
            del test_model
        except Exception as e:
            raise RuntimeError(f"Text model factory validation failed: {e}")

        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx}

    else:
        logging.info("Loading image dataset...")

        buyer_loader, seller_loaders, test_loader, stats, num_classes = get_image_dataset(cfg)

        if not seller_loaders or len(seller_loaders) == 0:
            logging.error("get_image_dataset returned empty seller_loaders!")

            logging.error(f"Config values:")

            logging.error(f"  - n_sellers: {cfg.experiment.n_sellers}")

            logging.error(f"  - dataset_name: {cfg.experiment.dataset_name}")

            logging.error(f"  - data_distribution: {getattr(cfg.experiment, 'data_distribution', 'N/A')}")

            raise ValueError(

                "get_image_dataset returned empty seller_loaders. "

                "Check your data partitioning configuration."

            )

        logging.info(f"Seller loaders created: {len(seller_loaders)}")

        logging.info(f"Seller IDs: {list(seller_loaders.keys())}")

        # Validate each loader has data

        empty_loaders = []

        for sid, loader in seller_loaders.items():

            if loader is None or len(loader.dataset) == 0:
                empty_loaders.append(sid)

        if empty_loaders:
            logging.error(f"Found {len(empty_loaders)} empty loaders: {empty_loaders}")

            raise ValueError(f"Sellers have no data: {empty_loaders}")
            # Load image model configuration
        config_name = cfg.experiment.image_model_config_name
        logging.info(f"Loading image model config: '{config_name}'")

        image_model_config = get_image_model_config(config_name)
        logging.info(
            f"Loaded config for '{image_model_config.model_name}' with recipe '{image_model_config.config_name}'")

        # Determine model parameters
        in_channels = 3  # CIFAR datasets have 3 channels
        sample_data, _ = next(iter(test_loader))
        image_size = tuple(sample_data.shape[2:])

        logging.info(f"Image model parameters:")
        logging.info(f"  - Input channels: {in_channels}")
        logging.info(f"  - Image size: {image_size}")
        logging.info(f"  - Classes: {num_classes}")

        # Create frozen factory using the new method
        model_factory = ImageModelFactory.create_factory(
            model_name=image_model_config.model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            config=image_model_config  # Will be deep copied inside create_factory
        )

        # Validate factory creates consistent models
        validate_model_factory(model_factory, num_tests=3)

        seller_extra_args = {}

    # Update config with final num_classes
    cfg.experiment.num_classes = num_classes

    logging.info("=" * 60)
    logging.info(f"Data and model setup complete:")
    logging.info(f"  - Dataset: {dataset_name}")
    logging.info(f"  - Type: {'text' if is_text else 'image'}")
    logging.info(f"  - Classes: {num_classes}")
    logging.info(f"  - Sellers: {len(seller_loaders)}")
    logging.info(f"  - Test samples: {len(test_loader.dataset) if test_loader else 0}")
    logging.info("=" * 60)

    return buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args, collate_fn, num_classes


def generate_marketplace_report(save_path: Path, marketplace):
    """Generate comprehensive marketplace analysis report."""

    report = {
        'experiment_summary': {
            'total_sellers': len(marketplace.sellers),
            'total_rounds': marketplace.current_round,
            'adversary_rate': sum(1 for s in marketplace.sellers.values() if 'adv' in s.seller_id) / len(
                marketplace.sellers)
        },
        'seller_summaries': {}
    }

    # Per-seller summary
    for sid, seller in marketplace.sellers.items():
        selection_history = getattr(seller, 'selection_history', [])
        reward_history = getattr(seller, 'reward_history', [])

        report['seller_summaries'][sid] = {
            'type': 'adversary' if 'adv' in sid else 'benign',
            'selection_rate': sum(1 for h in selection_history if h['selected']) / len(
                selection_history) if selection_history else 0,
            'outlier_rate': sum(1 for h in selection_history if h.get('outlier', False)) / len(
                selection_history) if selection_history else 0,
            'total_reward': sum(r['reward'] for r in reward_history) if reward_history else 0
        }

    # Save report
    with open(save_path / "marketplace_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    logging.info(f"Marketplace report saved to {save_path / 'marketplace_report.json'}")


def save_marketplace_analysis_data(save_path: Path, round_records: List[Dict]):
    """
    Save marketplace data in analysis-ready format.
    Creates multiple CSVs for different aspects of analysis.
    """

    # 1. Round-level aggregate metrics
    round_df = pd.DataFrame([{
        'round': r['round'],
        'timestamp': r['timestamp'],
        'duration_sec': r['duration_sec'],
        'selection_rate': r.get('selection_rate', 0),
        'outlier_rate': r.get('outlier_rate', 0),
        'avg_gradient_norm': r.get('avg_gradient_norm', 0),
        'adversary_detection_rate': r.get('adversary_detection_rate', 0),
        'false_positive_rate': r.get('false_positive_rate', 0)
    } for r in round_records])
    round_df.to_csv(save_path / "round_aggregates.csv", index=False)

    # 2. Per-seller per-round metrics
    seller_round_records = []
    for r in round_records:
        round_num = r['round']
        for key, value in r.items():
            if key.startswith('seller_') and '_' in key[7:]:
                parts = key.split('_', 2)
                if len(parts) == 3:
                    _, sid, metric = parts
                    seller_round_records.append({
                        'round': round_num,
                        'seller_id': sid,
                        'metric': metric,
                        'value': value
                    })

    if seller_round_records:
        seller_df = pd.DataFrame(seller_round_records)
        # Pivot for easier analysis
        seller_pivot = seller_df.pivot_table(
            index=['round', 'seller_id'],
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        seller_pivot.to_csv(save_path / "seller_round_metrics.csv", index=False)

    # 3. Selection history
    selection_records = []
    for r in round_records:
        for sid in r.get('selected_seller_ids', []):
            selection_records.append({
                'round': r['round'],
                'seller_id': sid,
                'selected': True,
                'outlier': False
            })
        for sid in r.get('outlier_seller_ids', []):
            selection_records.append({
                'round': r['round'],
                'seller_id': sid,
                'selected': False,
                'outlier': True
            })

    if selection_records:
        pd.DataFrame(selection_records).to_csv(
            save_path / "selection_history.csv", index=False
        )


def initialize_sellers(
        cfg: AppConfig,
        marketplace,
        client_loaders,
        model_factory,
        seller_extra_args,
        sybil_coordinator,
        collate_fn,
        num_classes: int
):
    """
    Creates and registers all sellers in the marketplace using the factory.

    Args:
        cfg: Application configuration
        marketplace: DataMarketplaceFederated instance
        client_loaders: Dict mapping client_id -> DataLoader
        model_factory: Factory function that creates model instances
        seller_extra_args: Extra arguments for seller creation
        sybil_coordinator: Coordinator for Sybil attacks
        collate_fn: Collate function for data loading
        num_classes: Number of output classes
    """
    logging.info("=" * 60)
    logging.info("🏪 Initializing Sellers")
    logging.info("=" * 60)

    # Validate inputs
    if not client_loaders:
        raise ValueError("client_loaders is empty! Cannot create sellers.")

    n_sellers = len(client_loaders)
    n_adversaries = int(n_sellers * cfg.experiment.adv_rate)

    # Validate adversary count
    if n_adversaries > n_sellers:
        logging.warning(
            f"⚠️  Requested {n_adversaries} adversaries but only {n_sellers} sellers available. "
            f"Capping at {n_sellers}."
        )
        n_adversaries = n_sellers

    logging.info(f"Configuration:")
    logging.info(f"  - Total sellers: {n_sellers}")
    logging.info(f"  - Adversary rate: {cfg.experiment.adv_rate:.1%}")
    logging.info(f"  - Adversaries: {n_adversaries}")
    logging.info(f"  - Benign: {n_sellers - n_adversaries}")
    logging.info(f"  - Sybil enabled: {cfg.adversary_seller_config.sybil.is_sybil}")

    # Select adversary IDs (first n_adversaries)
    all_client_ids = list(client_loaders.keys())
    adversary_ids = all_client_ids[:n_adversaries]

    logging.info(f"\n📋 Adversary IDs: {adversary_ids}")

    # Validate model_factory creates valid models
    try:
        test_model = model_factory()
        test_params = list(test_model.parameters())
        logging.info(f"\n🔍 Model factory validation:")
        logging.info(f"  - Parameters: {len(test_params)}")
        logging.info(f"  - Total params: {sum(p.numel() for p in test_params):,}")
        logging.info(f"  - First param shape: {test_params[0].shape}")
        del test_model  # Clean up
    except Exception as e:
        logging.error(f"❌ Model factory validation failed: {e}")
        raise ValueError(f"Invalid model_factory: {e}")

    # Create seller factory
    seller_factory = SellerFactory(
        cfg=cfg,
        model_factory=model_factory,
        num_classes=num_classes,
        **seller_extra_args
    )

    # Track creation statistics
    created_adversaries = 0
    created_benign = 0
    registered_sybils = 0
    failed_creations = 0

    # Create and register sellers
    logging.info(f"\n🏗️  Creating sellers...")

    for cid, loader in client_loaders.items():
        is_adv = cid in adversary_ids
        seller_type = "adversary" if is_adv else "benign"
        seller_id = f"{'adv' if is_adv else 'bn'}_{cid}"

        try:
            # Validate loader has data
            if loader.dataset is None or len(loader.dataset) == 0:
                logging.error(f"  ❌ {seller_id}: Empty dataset! Skipping.")
                failed_creations += 1
                continue

            # Create seller
            seller = seller_factory.create_seller(
                seller_id=seller_id,
                dataset=loader.dataset,
                is_adversary=is_adv,
                sybil_coordinator=sybil_coordinator,
                collate_fn=collate_fn
            )

            # Validate seller was created properly
            if seller is None:
                logging.error(f"  ❌ {seller_id}: Factory returned None! Skipping.")
                failed_creations += 1
                continue

            # Validate seller has model_factory
            if not hasattr(seller, 'model_factory') or seller.model_factory is None:
                logging.error(f"  ❌ {seller_id}: No model_factory attribute! Skipping.")
                failed_creations += 1
                continue

            # Validate seller's model matches expected architecture
            try:
                seller_model = seller.model_factory()
                seller_params = list(seller_model.parameters())
                if len(seller_params) != len(test_params):
                    logging.error(
                        f"  ❌ {seller_id}: Model architecture mismatch! "
                        f"Expected {len(test_params)} params, got {len(seller_params)}"
                    )
                    failed_creations += 1
                    continue
                del seller_model
            except Exception as e:
                logging.error(f"  ❌ {seller_id}: Model validation failed: {e}")
                failed_creations += 1
                continue

            # Register seller in marketplace
            marketplace.register_seller(seller.seller_id, seller)

            if is_adv:
                created_adversaries += 1
            else:
                created_benign += 1

            # Register as Sybil if applicable
            # Only register if: 1) is adversary, 2) Sybil globally enabled, 3) seller is Sybil
            if is_adv and cfg.adversary_seller_config.sybil.is_sybil:
                # Check if this specific seller should be a Sybil
                # (In case you want fine-grained control later)
                is_sybil = getattr(seller, 'is_sybil', True)  # Default to True for all advs

                if is_sybil:
                    sybil_coordinator.register_seller(seller)
                    registered_sybils += 1

            logging.info(
                f"  ✅ {seller_id} ({seller_type}): "
                f"{len(loader.dataset)} samples"
                f"{' [SYBIL]' if (is_adv and registered_sybils) else ''}"
            )

        except Exception as e:
            logging.error(f"  ❌ {seller_id}: Creation failed: {e}", exc_info=True)
            failed_creations += 1

    # Summary
    logging.info("=" * 60)
    logging.info("📊 Seller Initialization Summary:")
    logging.info(f"  - Total created: {created_adversaries + created_benign}/{n_sellers}")
    logging.info(f"  - Adversaries: {created_adversaries}/{n_adversaries}")
    logging.info(f"  - Benign: {created_benign}/{n_sellers - n_adversaries}")
    logging.info(f"  - Registered Sybils: {registered_sybils}")
    logging.info(f"  - Failed: {failed_creations}")
    logging.info("=" * 60)

    # Validate we created enough sellers
    total_created = created_adversaries + created_benign
    if total_created == 0:
        raise RuntimeError("❌ Failed to create any sellers!")

    if total_created < n_sellers * 0.5:  # Less than 50% succeeded
        logging.warning(
            f"⚠️  Only created {total_created}/{n_sellers} sellers ({total_created / n_sellers:.1%}). "
            f"Experiment may not run as expected."
        )

    # Verify marketplace state
    registered_count = len(marketplace.sellers)
    if registered_count != total_created:
        logging.error(
            f"❌ Mismatch: Created {total_created} sellers but marketplace has {registered_count}!"
        )

    logging.info(f"✅ Seller initialization complete!\n")


def run_final_evaluation_and_logging(
        cfg: AppConfig,
        final_model: nn.Module,
        results_buffer: List[Dict],  # Should be removed - use incremental saving
        test_loader,
        evaluators
):
    """Performs final evaluation and saves experiment artifacts."""
    logging.info("=" * 60)
    logging.info("Final Evaluation and Logging")
    logging.info("=" * 60)

    save_path = Path(cfg.experiment.save_path)

    # NOTE: training_log.csv should already exist from incremental saves
    # This is just a validation check
    log_path = save_path / "training_log.csv"
    if not log_path.exists():
        logging.warning("training_log.csv not found! Reconstructing from buffer...")
        if results_buffer:
            save_dataframe_atomic(pd.DataFrame(results_buffer), log_path)

    # Final evaluation
    logging.info("Performing final evaluation...")
    final_metrics = {}

    for evaluator in evaluators:
        try:
            metrics = evaluator.evaluate(final_model, test_loader)
            final_metrics.update(metrics)
        except Exception as e:
            logging.error(f"Evaluator {evaluator.__class__.__name__} failed: {e}")

    logging.info(f"Final metrics: {final_metrics}")

    # Add metadata
    final_metrics['timestamp'] = time.time()
    final_metrics['completed_rounds'] = len(results_buffer) if results_buffer else 'unknown'

    # Save final metrics atomically
    save_json_atomic(final_metrics, save_path / "final_metrics.json")

    # Save final model atomically
    save_model_atomic(final_model.state_dict(), save_path / "final_model.pth")

    logging.info("Final evaluation complete")


def save_json_atomic(data, filepath):
    """Save JSON with atomic write to prevent corruption."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        suffix='.tmp',
        prefix=filepath.stem
    )
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(temp_path, filepath)
        logging.debug(f"Saved (atomic): {filepath}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save {filepath}: {e}")


def save_dataframe_atomic(df, filepath):
    """Save DataFrame with atomic write."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_path = filepath.with_suffix('.tmp')
    try:
        df.to_csv(temp_path, index=False)
        shutil.move(temp_path, filepath)
    except:
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_model_atomic(state_dict, filepath):
    """Save model with atomic write."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_path = filepath.with_suffix('.tmp')
    try:
        torch.save(state_dict, temp_path)
        shutil.move(temp_path, filepath)
    except:
        if temp_path.exists():
            temp_path.unlink()
        raise


def run_training_loop(cfg, marketplace, test_loader, evaluators, sybil_coordinator):
    """Training loop with incremental saving."""
    save_path = Path(cfg.experiment.save_path)
    log_path = save_path / "training_log.csv"

    # Initialize CSV with headers
    if not log_path.exists():
        pd.DataFrame(columns=['round', 'duration_sec', 'num_selected', 'num_outliers']).to_csv(
            log_path, index=False
        )
    round_records = []
    for round_num in range(1, cfg.experiment.rounds + 1):
        round_record, agg_grad = marketplace.train_federated_round(
            round_number=round_num,
            ground_truth_dict={}
        )
        round_records.append(round_record)
        save_round_incremental(round_record, save_path)

        # Incremental save - append to CSV
        df_round = pd.DataFrame([round_record])
        df_round.to_csv(log_path, mode='a', header=False, index=False)

        # Evaluate periodically
        if round_num % cfg.experiment.eval_frequency == 0:
            eval_metrics = {}
            for evaluator in evaluators:
                metrics = evaluator.evaluate(marketplace.aggregator.strategy.global_model, test_loader)
                eval_metrics.update(metrics)

            # Save evaluation checkpoint
            eval_path = save_path / "evaluations" / f"round_{round_num}.json"
            save_json_atomic(eval_metrics, eval_path)

        # Save seller histories incrementally
        save_marketplace_analysis_data(save_path, round_records)
        generate_marketplace_report(save_path, marketplace)

        # Seller summaries
        for sid, seller in marketplace.sellers.items():
            seller.save_marketplace_summary()

    return marketplace.aggregator.strategy.global_model, []  # Empty buffer since we saved incrementally


def save_round_incremental(round_record, save_path):
    log_path = Path(save_path) / "training_log.csv"
    df = pd.DataFrame([round_record])

    # Append to existing file or create new
    if log_path.exists():
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)


def run_attack(cfg: AppConfig):
    """
    Orchestrates the entire experiment from a single config object.

    Note: Caching is handled by the experiment orchestrator, not here.
    This function always runs the full experiment when called.

    Args:
        cfg: Application configuration containing all experiment parameters
    """
    save_path = Path(cfg.experiment.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info(f"🚀 Starting Experiment")
    logging.info(f"   Dataset: {cfg.experiment.dataset_name}")
    logging.info(f"   Model: {cfg.experiment.model_structure}")
    logging.info(f"   Device: {cfg.experiment.device}")
    logging.info(f"   Save Path: {save_path}")
    logging.info("=" * 80)

    try:
        # 1. Save configuration snapshot for reproducibility
        _save_config_for_reproducibility(cfg, save_path)

        # 2. Data and Model Setup
        buyer_loader, seller_loaders, test_loader, model_factory, seller_extra_args, collate_fn, num_classes = \
            setup_data_and_model(cfg)

        global_model = model_factory().to(cfg.experiment.device)
        logging.info(f"✅ Global model created and moved to {cfg.experiment.device}")
        logging.info(f"--- Global Model Architecture ---\n{global_model}")

        # 3. FL Component Initialization
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
        sample_data = _get_sample_data(test_loader, seller_loaders)
        input_shape = tuple(sample_data.shape[1:])

        marketplace = DataMarketplaceFederated(
            cfg=cfg,
            aggregator=aggregator,
            sellers={},
            input_shape=input_shape
        )

        # 5. Seller Initialization
        initialize_sellers(
            cfg, marketplace, seller_loaders, model_factory,
            seller_extra_args, sybil_coordinator, collate_fn, num_classes
        )

        # 6. Federated Training Loop
        logging.info("🏋️ Starting federated training...")
        final_model, results_buffer = run_training_loop(
            cfg, marketplace, test_loader, evaluators, sybil_coordinator
        )

        # 7. Final Evaluation and Artifact Saving
        logging.info("📊 Running final evaluation and saving results...")
        run_final_evaluation_and_logging(
            cfg, final_model, results_buffer, test_loader, evaluators
        )

        # 8. Save seller-specific results
        for sid, seller in marketplace.sellers.items():
            seller.save_round_history_csv()

        # 9. Mark experiment as successfully completed
        _mark_experiment_success(save_path)

        logging.info("=" * 80)
        logging.info("✅ Experiment Finished Successfully")
        logging.info(f"   Results saved to: {save_path}")
        logging.info("=" * 80)

        return results_buffer

    except Exception as e:
        logging.error("=" * 80)
        logging.error(f"❌ Experiment Failed: {e}")
        logging.error("=" * 80)
        # Mark as failed (optional)
        _mark_experiment_failed(save_path, str(e))
        raise  # Re-raise to propagate error to orchestrator


# ==============================================================================
# Helper Functions
# ==============================================================================

def _save_config_for_reproducibility(cfg: AppConfig, save_path: Path):
    """
    Save the configuration to disk for reproducibility and debugging.
    """
    import json
    try:
        config_path = save_path / "config_snapshot.json"
        with open(config_path, 'w') as f:
            # Assuming your AppConfig has a to_dict() method
            # If not, you might need to use dataclasses.asdict() or similar
            if hasattr(cfg, 'to_dict'):
                json.dump(cfg.to_dict(), f, indent=2)
            elif hasattr(cfg, '__dict__'):
                # Fallback: try to serialize the object's dictionary
                json.dump(cfg.__dict__, f, indent=2, default=str)
            else:
                logging.warning("Could not serialize config - no to_dict() method available")

        logging.info(f"📝 Configuration snapshot saved to: {config_path}")
    except Exception as e:
        logging.warning(f"⚠️  Could not save config snapshot: {e}")


def _get_sample_data(test_loader, seller_loaders):
    """
    Get a sample batch of data for shape inference.
    Tries test_loader first, then falls back to seller loaders.
    """
    sample_data = None

    # Try test loader first
    if test_loader:
        try:
            sample_data, _ = next(iter(test_loader))
            logging.info("✅ Sample data obtained from test loader")
        except StopIteration:
            logging.warning("⚠️  Test loader is available but empty")

    # Fall back to seller loaders
    if sample_data is None:
        logging.info("🔍 No test data found. Trying seller loaders...")
        for sid, loader in seller_loaders.items():
            if loader:
                try:
                    sample_data, _ = next(iter(loader))
                    logging.info(f"✅ Sample data obtained from seller {sid} loader")
                    break
                except StopIteration:
                    continue

    if sample_data is None:
        raise RuntimeError(
            "❌ Could not retrieve a sample data batch from any available loader. "
            "Please check your data loaders."
        )

    return sample_data


def _mark_experiment_success(save_path: Path):
    """
    Create a success marker file to indicate the experiment completed successfully.
    This is used by the orchestrator for caching.
    """
    success_marker = save_path / ".success"
    try:
        success_marker.touch()
        logging.info(f"✅ Success marker created: {success_marker}")
    except Exception as e:
        logging.warning(f"⚠️  Could not create success marker: {e}")


def _mark_experiment_failed(save_path: Path, error_message: str):
    """
    Create a failure marker with error information for debugging.
    """
    try:
        failed_marker = save_path / ".failed"
        with open(failed_marker, 'w') as f:
            f.write(f"Experiment failed with error:\n{error_message}\n")
        logging.info(f"❌ Failure marker created: {failed_marker}")
    except Exception as e:
        logging.warning(f"⚠️  Could not create failure marker: {e}")


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
