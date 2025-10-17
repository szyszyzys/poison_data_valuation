import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common.enums import ServerAttackMode
from common.gradient_market_configs import AppConfig, ServerAttackConfig
from entry.gradient_market.privacy_attack import GradientInversionAttacker
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.gradient_seller import GradientSeller
from marketplace.seller.seller import BaseSeller


@dataclass
class MarketplaceConfig:
    """Configuration for the DataMarketplaceFederated."""
    save_path: str
    dataset_name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    privacy_attack_config: ServerAttackConfig = field(default_factory=ServerAttackConfig)


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self, cfg: AppConfig, aggregator: Aggregator, sellers: dict,
                 input_shape: tuple, SellerClass: type, validation_loader, model_factory, num_classes: int,
                 attacker=None, buyer_seller: GradientSeller = None, oracle_seller: GradientSeller = None):
        """
        Initializes the marketplace with all necessary components and the main config.
        """
        self.cfg = cfg  # Store the main config object
        self.aggregator = aggregator
        self.sellers = sellers
        self.attacker = attacker  # For server-side privacy attacks
        self.consecutive_failed_rounds = 0
        self.SellerClass = SellerClass  # <-- Add this
        self.model_factory = model_factory
        self.validation_loader = validation_loader
        self.buyer_seller: GradientSeller = buyer_seller
        self.oracle_seller: GradientSeller = oracle_seller
        self.num_classes = num_classes

        # Conditionally initialize the privacy attacker
        if self.cfg.server_attack_config.attack_name == ServerAttackMode.GRADIENT_INVERSION:
            # 2. Use the correct variable name: 'self.cfg' not 'self.config'
            self.attacker = GradientInversionAttacker(
                attack_config=self.cfg.server_attack_config,
                model_template=aggregator.strategy.global_model,
                device=aggregator.device,
                # 3. Use the correct path for the save directory
                save_dir=self.cfg.experiment.save_path,

                # 4. Use the correct paths for other experiment parameters
                dataset_name=self.cfg.experiment.dataset_name,
                input_shape=input_shape,  # Passed in, as it's determined at runtime
                num_classes=self.cfg.experiment.num_classes
            )
            logging.info("Initialized GradientInversionAttacker.")

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """Registers a new seller in the marketplace."""
        self.sellers[seller_id] = seller
        logging.info(f"Marketplace: Registered seller '{seller_id}'. Total sellers: {len(self.sellers)}")

    @property
    def global_model(self) -> torch.nn.Module:
        """Provides convenient, direct access to the global model."""
        return self.aggregator.strategy.global_model

    def train_federated_round(
            self,
            round_number: int,
            global_model,  # <-- FIX: Add missing arguments
            validation_loader,  # <-- Add this for clarity
            ground_truth_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict, Any]:
        """Orchestrates a single federated round with the 'virtual seller' design."""
        round_start_time = time.time()
        logging.info(f"--- Round {round_number} Started ---")

        # === 2. Collect Gradients from the Real Marketplace ===
        gradients_dict, seller_ids, seller_stats_list = self._get_current_market_gradients()

        # === 1. Compute Root Gradients using Virtual Sellers ===
        logging.info("🛒 Creating virtual 'Buyer Seller' to compute root gradient...")
        buyer_root_gradient, _ = self.buyer_seller.get_gradient_for_upload(
            global_model,
            all_seller_gradients=gradients_dict,
            target_seller_id=getattr(self.cfg.buyer_attack_config, 'target_seller_id', None)  # Example for config
        )
        oracle_root_gradient, _ = self.oracle_seller.get_gradient_for_upload(global_model)
        if buyer_root_gradient is None:
            logging.error("Virtual buyer failed to compute a gradient, which is a fatal error for this round.")
            # This indicates a severe problem (like a CUDA crash) that should stop the experiment.
            raise RuntimeError("Buyer root gradient is None, indicating a fatal error in local training.")

        # === 3. Sanitize ALL Gradients Before Use ===
        target_device = self.aggregator.device

        sanitized_gradients = {}

        # Get the shapes and dtypes of the global model's parameters ONCE
        param_meta = [(p.shape, p.dtype) for p in self.global_model.parameters()]

        for sid, grad_list in gradients_dict.items():
            # Check if the whole gradient list is missing or invalid
            if grad_list is None or len(grad_list) != len(param_meta):
                logging.error(f"Seller '{sid}' returned an invalid gradient list. Replacing with zeros.")
                # Create a zero-gradient that matches the model structure
                sanitized_gradients[sid] = [
                    torch.zeros(shape, dtype=dtype, device=target_device)
                    for shape, dtype in param_meta
                ]
                continue  # Move to the next seller

            # If the list is valid, sanitize each tensor inside it
            corrected_list = []
            for i, tensor in enumerate(grad_list):
                if tensor is None:
                    logging.warning(
                        f"Sanitizer: Found None in gradient from seller '{sid}' at index {i}. Replacing with zeros.")
                    shape, dtype = param_meta[i]
                    corrected_list.append(torch.zeros(shape, dtype=dtype, device=target_device))
                elif tensor.device != target_device:
                    logging.warning(
                        f"Sanitizer: Correcting device for seller '{sid}' grad: {tensor.device} -> {target_device}")
                    corrected_list.append(tensor.to(target_device))
                else:
                    corrected_list.append(tensor)
            sanitized_gradients[sid] = corrected_list

        # Also sanitize the root gradient (no need for the None check if you trust the buyer)
        sanitized_root_gradient = [
            (tensor.to(target_device) if tensor.device != target_device else tensor)
            for tensor in buyer_root_gradient
        ]

        # Sanitize the oracle's root gradient (MOVED HERE)
        sanitized_oracle_gradient = [
            (tensor.to(target_device) if tensor is not None and tensor.device != target_device else tensor)
            for tensor in oracle_root_gradient
        ]

        # Perform privacy attack (optional)
        attack_log = None
        if self.attacker and self.attacker.should_run(round_number):
            attack_log = self.attacker.execute(round_number, gradients_dict, seller_ids, ground_truth_dict)
        agg_grad, selected_ids, outlier_ids, aggregation_stats = self.aggregator.aggregate(
            global_epoch=round_number,
            seller_updates=sanitized_gradients,  # Use the sanitized dictionary
            root_gradient=sanitized_root_gradient,  # Use the sanitized root gradient
            buyer_data_loader=self.aggregator.buyer_data_loader
        )

        # === NEW: Calculate marketplace metrics ===
        marketplace_metrics = self._compute_marketplace_metrics(
            round_number=round_number,
            gradients_dict=sanitized_gradients,
            seller_ids=seller_ids,
            selected_ids=selected_ids,
            outlier_ids=outlier_ids,
            aggregation_stats=aggregation_stats,
            seller_stats_list=seller_stats_list,
            oracle_root_gradient=sanitized_oracle_gradient  # Pass the oracle gradient for logging
        )

        # In train_federated_round, after the `if agg_grad:` check
        if agg_grad:
            self.consecutive_failed_rounds = 0  # Reset on success
        else:
            self.consecutive_failed_rounds += 1
            logging.warning(
                f"Round failed to produce an update. Consecutive failures: {self.consecutive_failed_rounds}")

        max_failures = getattr(self.cfg.training, 'max_consecutive_failures', 5)
        if self.consecutive_failed_rounds >= max_failures:
            logging.error(f"Stopping experiment after {max_failures} consecutive failed rounds.")
            raise RuntimeError("Halting due to persistent aggregation failures.")

        # Save individual gradients if needed
        if self.cfg.debug.save_individual_gradients:
            if round_number % self.cfg.debug.gradient_save_frequency == 0:
                self._save_round_gradients(round_number, gradients_dict, agg_grad)

        # Create comprehensive round record
        duration = time.time() - round_start_time
        round_record = {
            "round": round_number,
            "timestamp": time.time(),
            "duration_sec": duration,

            # Basic stats
            "num_total_sellers": len(seller_ids),
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),

            # Selection details
            "selected_seller_ids": selected_ids,
            "outlier_seller_ids": outlier_ids,
            "buyer_attack_active": self.cfg.buyer_attack_config.is_active,
            "buyer_attack_type": self.cfg.buyer_attack_config.attack_type if self.cfg.buyer_attack_config.is_active else "none",
            # Attack info
            "attack_performed": bool(attack_log),
            "attack_victim": attack_log.get('victim_id') if attack_log else None,
            "attack_success": attack_log.get('success') if attack_log else None,

            # Marketplace metrics
            **marketplace_metrics
        }

        # Merge aggregator stats
        if aggregation_stats:
            round_record.update(aggregation_stats)

        # Notify sellers of round end
        for sid, seller in self.sellers.items():
            seller.round_end_process(
                round_number,
                was_selected=(sid in selected_ids),
                was_outlier=(sid in outlier_ids),
                marketplace_metrics=marketplace_metrics.get(f'seller_{sid}', {})
            )

        logging.info(f"--- Round {round_number} Ended (Duration: {duration:.2f}s) ---")

        return round_record, agg_grad

    def _save_round_gradients(self, round_number: int, gradients_dict: Dict, agg_grad: List[torch.Tensor]):
        """
        Save individual seller gradients and aggregated gradient for debugging/analysis.
        This is the implementation of the method called in train_federated_round.
        """
        grad_save_dir = Path(self.cfg.experiment.save_path) / "individual_gradients" / f"round_{round_number}"
        grad_save_dir.mkdir(parents=True, exist_ok=True)

        # Save individual seller gradients
        saved_count = 0
        for sid, grad in gradients_dict.items():
            if grad is not None and len(grad) > 0:
                try:
                    torch.save(grad, grad_save_dir / f"{sid}_grad.pt")
                    saved_count += 1
                except Exception as e:
                    logging.error(f"Failed to save gradient for {sid}: {e}")

        logging.info(f"Saved {saved_count}/{len(gradients_dict)} individual gradients to {grad_save_dir}")

        # Save aggregated gradient
        if agg_grad is not None and len(agg_grad) > 0:
            try:
                torch.save(agg_grad, grad_save_dir / "aggregated_grad.pt")
                logging.info(f"Saved aggregated gradient")
            except Exception as e:
                logging.error(f"Failed to save aggregated gradient: {e}")

    def _compute_marketplace_metrics(
            self,
            round_number: int,
            gradients_dict: Dict,
            seller_ids: List[str],
            selected_ids: List[str],
            outlier_ids: List[str],
            aggregation_stats: Dict,
            seller_stats_list: List[Dict],
            oracle_root_gradient: List[torch.Tensor]
    ) -> Dict:
        """
        Compute comprehensive marketplace metrics for analysis.

        Returns dict with both aggregate metrics and per-seller metrics.
        """
        metrics = {}

        # === 1. Selection Analysis ===
        metrics['selection_rate'] = len(selected_ids) / len(seller_ids) if seller_ids else 0
        metrics['outlier_rate'] = len(outlier_ids) / len(seller_ids) if seller_ids else 0

        # Per-seller selection info
        for sid in seller_ids:
            prefix = f'seller_{sid}_'
            metrics[f'{prefix}selected'] = sid in selected_ids
            metrics[f'{prefix}outlier'] = sid in outlier_ids

        # === 2. Gradient Quality Metrics ===
        if gradients_dict:
            gradient_norms = {}
            gradient_similarities = {}

            for sid, grad in gradients_dict.items():
                if grad is None:
                    continue

                # Compute L2 norm
                norm = sum(torch.norm(g).item() ** 2 for g in grad) ** 0.5
                gradient_norms[sid] = norm
                metrics[f'seller_{sid}_gradient_norm'] = norm

            # Aggregate gradient statistics
            norms_list = list(gradient_norms.values())
            if norms_list:
                metrics['avg_gradient_norm'] = np.mean(norms_list)
                metrics['std_gradient_norm'] = np.std(norms_list)
                metrics['min_gradient_norm'] = np.min(norms_list)
                metrics['max_gradient_norm'] = np.max(norms_list)

            # Compute pairwise cosine similarities (expensive, do sparingly)
            if self.cfg.experiment.compute_gradient_similarity and round_number % 5 == 0:
                similarities = self._compute_gradient_similarities(gradients_dict)
                metrics['avg_gradient_similarity'] = np.mean(similarities) if similarities else 0
                metrics['gradient_similarity_matrix'] = similarities  # Full matrix for detailed analysis

            if hasattr(self.aggregator.strategy,
                       'root_gradient') and self.aggregator.strategy.root_gradient is not None:
                g_buyer_root_flat = torch.cat([g.flatten() for g in self.aggregator.strategy.root_gradient])
                for sid, grad in gradients_dict.items():
                    if grad is None: continue
                    g_seller_flat = torch.cat([g.flatten() for g in grad])
                    sim_score = torch.nn.functional.cosine_similarity(g_buyer_root_flat.unsqueeze(0),
                                                                      g_seller_flat.unsqueeze(0)).item()
                    metrics[f'seller_{sid}_sim_to_buyer_root'] = sim_score

            # 2. Calculate Similarity to the Oracle Root Gradient (for analysis)
            if oracle_root_gradient is not None:
                g_oracle_root_flat = torch.cat([g.flatten() for g in oracle_root_gradient])
                for sid, grad in gradients_dict.items():
                    if grad is None: continue
                    g_seller_flat = torch.cat([g.flatten() for g in grad])
                    sim_score = torch.nn.functional.cosine_similarity(g_oracle_root_flat.unsqueeze(0),
                                                                      g_seller_flat.unsqueeze(0)).item()
                    metrics[f'seller_{sid}_sim_to_oracle_root'] = sim_score

        # === 3. Seller Contribution Metrics ===
        # These come from aggregation_stats if your aggregator provides them
        if aggregation_stats and 'seller_weights' in aggregation_stats:
            for sid, weight in aggregation_stats['seller_weights'].items():
                metrics[f'seller_{sid}_weight'] = weight

        # === 4. Data Quality Indicators ===
        for sid, stats in zip(seller_ids, seller_stats_list):
            if stats:
                metrics[f'seller_{sid}_train_loss'] = stats.get('train_loss')
                metrics[f'seller_{sid}_num_samples'] = stats.get('num_samples', 0)
                metrics[f'seller_{sid}_upload_bytes'] = stats.get('upload_bytes', 0)

        # === 5. Adversary Detection Metrics ===
        # Track known adversaries vs detected outliers
        known_adversaries = [sid for sid in seller_ids if 'adv' in sid]
        detected_adversaries = [sid for sid in outlier_ids if 'adv' in sid]
        benign_outliers = [sid for sid in outlier_ids if 'bn' in sid]

        metrics['num_known_adversaries'] = len(known_adversaries)
        metrics['num_detected_adversaries'] = len(detected_adversaries)
        metrics['num_benign_outliers'] = len(benign_outliers)
        metrics['adversary_detection_rate'] = (
            len(detected_adversaries) / len(known_adversaries)
            if known_adversaries else 0
        )
        metrics['false_positive_rate'] = (
            len(benign_outliers) / (len(seller_ids) - len(known_adversaries))
            if (len(seller_ids) - len(known_adversaries)) > 0 else 0
        )

        return metrics

    def _compute_gradient_similarities(self, gradients_dict: Dict) -> List[float]:
        """Compute pairwise cosine similarities between gradients."""
        seller_ids = list(gradients_dict.keys())
        similarities = []

        for i, sid1 in enumerate(seller_ids):
            for sid2 in seller_ids[i + 1:]:
                grad1 = gradients_dict[sid1]
                grad2 = gradients_dict[sid2]

                if grad1 is None or grad2 is None:
                    continue

                # Flatten and compute cosine similarity
                flat1 = torch.cat([g.flatten() for g in grad1])
                flat2 = torch.cat([g.flatten() for g in grad2])

                similarity = torch.nn.functional.cosine_similarity(
                    flat1.unsqueeze(0),
                    flat2.unsqueeze(0)
                ).item()
                similarities.append(similarity)

        return similarities

    def _get_current_market_gradients(self) -> Tuple[Dict, List, List]:
        """Collects gradients and stats from all sellers with detailed debugging."""
        logging.info("=" * 60)
        logging.info("📦 Collecting gradients from all sellers...")
        logging.info("=" * 60)

        gradients_dict = OrderedDict()
        seller_ids, seller_stats_list = [], []

        # Get expected parameters from global model
        global_params = list(self.aggregator.strategy.global_model.parameters())
        expected_param_count = len(global_params)

        logging.info(f"Global model info:")
        logging.info(f"  - Expected param count: {expected_param_count}")
        logging.info(f"  - First param shape: {global_params[0].shape if global_params else 'N/A'}")
        logging.info(f"  - Device: {global_params[0].device if global_params else 'N/A'}")
        logging.info(f"\nProcessing {len(self.sellers)} sellers...")

        success_count = 0
        fail_count = 0

        for sid, seller in self.sellers.items():
            logging.info(f"\n{'─' * 60}")
            logging.info(f"🔍 Seller: {sid}")

            # Check seller state
            try:
                is_active = getattr(seller, 'is_active', True)
                has_model = hasattr(seller, 'model') and seller.model is not None

                logging.info(f"  Status:")
                logging.info(f"    - Active: {is_active}")
                logging.info(f"    - Has model: {has_model}")

                if has_model:
                    seller_param_count = sum(1 for _ in seller.model.parameters())
                    logging.info(f"    - Seller model param count: {seller_param_count}")

                if not is_active:
                    logging.warning(f"  ⚠️  Seller {sid} is inactive, skipping")
                    fail_count += 1
                    continue

            except Exception as e:
                logging.error(f"  ❌ Error checking seller state: {e}")
                fail_count += 1
                continue

            # Try to get gradient
            try:
                logging.info(f"  Calling get_gradient_for_upload()...")
                grad, stats = seller.get_gradient_for_upload(self.aggregator.strategy.global_model)

                # Detailed gradient inspection
                logging.info(f"  Gradient inspection:")
                logging.info(f"    - Returned grad is None: {grad is None}")
                logging.info(f"    - Returned stats: {stats}")

                if grad is None:
                    logging.warning(f"  ⚠️  Seller {sid} returned None gradient")
                    fail_count += 1
                    continue

                # Check gradient type
                logging.info(f"    - Gradient type: {type(grad)}")

                if not isinstance(grad, (list, tuple)):
                    logging.error(f"  ❌ Gradient is not a list/tuple, got {type(grad)}")
                    fail_count += 1
                    continue

                # Check gradient length
                grad_length = len(grad)
                logging.info(f"    - Gradient length: {grad_length}")
                logging.info(f"    - Expected length: {expected_param_count}")

                if grad_length == 0:
                    logging.error(f"  ❌ Gradient is empty list")
                    fail_count += 1
                    continue

                if grad_length != expected_param_count:
                    logging.error(f"  ❌ LENGTH MISMATCH: got {grad_length}, expected {expected_param_count}")
                    fail_count += 1
                    continue

                # Check gradient content
                logging.info(f"    - First element type: {type(grad[0])}")

                if hasattr(grad[0], 'shape'):
                    logging.info(f"    - First element shape: {grad[0].shape}")
                    logging.info(f"    - Expected first shape: {global_params[0].shape}")

                    if grad[0].shape != global_params[0].shape:
                        logging.error(f"  ❌ SHAPE MISMATCH at first parameter!")
                        fail_count += 1
                        continue

                if isinstance(grad[0], torch.Tensor):
                    logging.info(f"    - First element dtype: {grad[0].dtype}")
                    logging.info(f"    - First element device: {grad[0].device}")
                    logging.info(f"    - First element has NaN: {torch.isnan(grad[0]).any().item()}")
                    logging.info(f"    - First element has Inf: {torch.isinf(grad[0]).any().item()}")
                    logging.info(f"    - First element mean: {grad[0].mean().item():.6e}")
                    logging.info(f"    - First element std: {grad[0].std().item():.6e}")

                    # Check if gradient is all zeros
                    is_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in grad[:3])  # Check first 3
                    if is_zero:
                        logging.warning(f"  ⚠️  Warning: First few gradients appear to be all zeros")

                # Validate all parameters
                all_valid = True
                for i, g in enumerate(grad):
                    if not isinstance(g, (torch.Tensor, np.ndarray)):
                        logging.error(f"  ❌ Parameter {i} is not a tensor/array: {type(g)}")
                        all_valid = False
                        break

                    if hasattr(g, 'shape') and g.shape != global_params[i].shape:
                        logging.error(f"  ❌ Shape mismatch at parameter {i}: {g.shape} vs {global_params[i].shape}")
                        all_valid = False
                        break

                if not all_valid:
                    logging.error(f"  ❌ Gradient validation failed")
                    fail_count += 1
                    continue

                # Success!
                gradients_dict[sid] = grad
                seller_ids.append(sid)
                seller_stats_list.append(stats)
                success_count += 1

                logging.info(f"  ✅ Gradient accepted from seller {sid}")

            except AttributeError as e:
                logging.error(f"  ❌ AttributeError (missing method?): {e}")
                logging.error(f"     Seller methods: {[m for m in dir(seller) if not m.startswith('_')]}")
                fail_count += 1

            except Exception as e:
                logging.error(f"  ❌ Exception getting gradient from seller {sid}: {e}", exc_info=True)
                fail_count += 1

        # Summary
        logging.info("\n" + "=" * 60)
        logging.info("📊 Gradient Collection Summary:")
        logging.info(f"  - Total sellers: {len(self.sellers)}")
        logging.info(f"  - Successful: {success_count} ✅")
        logging.info(f"  - Failed: {fail_count} ❌")
        logging.info(f"  - Success rate: {success_count / len(self.sellers) * 100:.1f}%")
        logging.info(f"  - Collected seller IDs: {seller_ids}")
        logging.info("=" * 60)

        # Final validation
        if not gradients_dict:
            logging.error("⚠️⚠️⚠️  NO GRADIENTS COLLECTED! ⚠️⚠️⚠️")
            logging.error("This will cause aggregation to fail!")
            logging.error("\nPossible causes:")
            logging.error("  1. All sellers returned None")
            logging.error("  2. Length/shape mismatches")
            logging.error("  3. Sellers not properly initialized")
            logging.error("  4. Training didn't happen")
            logging.error("\nCheck the detailed logs above for specific errors per seller.")

        return gradients_dict, seller_ids, seller_stats_list
