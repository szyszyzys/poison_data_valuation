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
                 input_shape: tuple, attacker=None):
        """
        Initializes the marketplace with all necessary components and the main config.
        """
        self.cfg = cfg  # Store the main config object
        self.aggregator = aggregator
        self.sellers = sellers
        self.attacker = attacker  # For server-side privacy attacks

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

    def train_federated_round(
            self,
            round_number: int,
            ground_truth_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict, Any]:
        """Orchestrates a single, config-driven round of federated learning."""
        round_start_time = time.time()
        logging.info("=" * 80)
        logging.info(f"🚀 Round {round_number} Started")
        logging.info("=" * 80)

        # 1. Collect gradients from all active sellers
        logging.info("📦 Step 1: Collecting gradients from sellers...")
        gradients_dict, seller_ids, _ = self._get_current_market_gradients()

        # === CRITICAL DEBUG POINT 1: Inspect what sellers returned ===
        logging.info(f"📊 Gradient Collection Summary:")
        logging.info(f"   - Total sellers: {len(seller_ids)}")
        logging.info(f"   - Seller IDs: {seller_ids}")
        logging.info(f"   - Gradients dict keys: {list(gradients_dict.keys())}")
        logging.info(f"   - Gradients dict size: {len(gradients_dict)}")

        # Debug each seller's gradient
        for sid in seller_ids:
            grad = gradients_dict.get(sid)
            logging.info(f"\n🔍 Seller {sid} gradient inspection:")
            logging.info(f"   - Present in dict: {sid in gradients_dict}")
            logging.info(f"   - Value is None: {grad is None}")
            logging.info(f"   - Value type: {type(grad)}")

            if grad is not None:
                if isinstance(grad, (list, tuple)):
                    logging.info(f"   - Length: {len(grad)}")
                    if len(grad) > 0:
                        logging.info(f"   - First element type: {type(grad[0])}")
                        if hasattr(grad[0], 'shape'):
                            logging.info(f"   - First element shape: {grad[0].shape}")
                        if hasattr(grad[0], 'dtype'):
                            logging.info(f"   - First element dtype: {grad[0].dtype}")
                        # Check for NaN/Inf
                        if isinstance(grad[0], torch.Tensor):
                            logging.info(f"   - First element has NaN: {torch.isnan(grad[0]).any()}")
                            logging.info(f"   - First element has Inf: {torch.isinf(grad[0]).any()}")
                    else:
                        logging.warning(f"   ⚠️  Empty gradient list!")
                elif isinstance(grad, dict):
                    logging.info(f"   - Dict keys: {list(grad.keys())}")
                else:
                    logging.warning(f"   ⚠️  Unexpected gradient type: {type(grad)}")
            else:
                logging.error(f"   ❌ Gradient is None!")

        # Check if any gradients are actually present
        valid_gradients = {k: v for k, v in gradients_dict.items() if v is not None and len(v) > 0}
        logging.info(f"\n✅ Valid gradients: {len(valid_gradients)}/{len(gradients_dict)}")

        if len(valid_gradients) == 0:
            logging.error("❌ NO VALID GRADIENTS COLLECTED! Investigating sellers...")
            # Debug sellers directly
            for sid, seller in self.sellers.items():
                logging.error(f"   Seller {sid}:")
                logging.error(f"     - Active: {getattr(seller, 'is_active', 'N/A')}")
                logging.error(f"     - Has model: {hasattr(seller, 'model')}")
                if hasattr(seller, 'model') and seller.model is not None:
                    logging.error(f"     - Model params count: {sum(1 for _ in seller.model.parameters())}")

        # 2. Perform privacy attack (optional)
        attack_log = None
        if self.attacker and self.attacker.should_run(round_number):
            logging.info("🎭 Step 2: Executing privacy attack...")
            attack_log = self.attacker.execute(round_number, gradients_dict, seller_ids, ground_truth_dict)
            if attack_log:
                logging.info(f"   Attack completed: victim={attack_log.get('victim_id')}")

        # === CRITICAL DEBUG POINT 2: Before aggregation ===
        logging.info("\n🔄 Step 3: Starting aggregation...")
        logging.info(f"   Input to aggregator:")
        logging.info(f"     - Round: {round_number}")
        logging.info(f"     - Gradients dict keys: {list(gradients_dict.keys())}")
        logging.info(f"     - Number of items: {len(gradients_dict)}")

        # Detailed inspection before aggregation
        for sid, grad in list(gradients_dict.items())[:3]:  # Show first 3 for brevity
            logging.info(f"     - {sid}: type={type(grad)}, "
                         f"len={len(grad) if hasattr(grad, '__len__') else 'N/A'}, "
                         f"is_none={grad is None}")

        try:
            agg_grad, selected_ids, outlier_ids, aggregation_stats = self.aggregator.aggregate(
                global_epoch=round_number,
                seller_updates=gradients_dict
            )

            # === CRITICAL DEBUG POINT 3: After aggregation ===
            logging.info(f"\n✅ Aggregation completed:")
            logging.info(f"   - Selected sellers: {selected_ids}")
            logging.info(f"   - Outlier sellers: {outlier_ids}")
            logging.info(f"   - Aggregated gradient is None: {agg_grad is None}")
            if agg_grad is not None:
                logging.info(f"   - Aggregated gradient length: {len(agg_grad)}")
                if len(agg_grad) > 0:
                    logging.info(f"   - First tensor shape: {agg_grad[0].shape}")
                    logging.info(f"   - First tensor mean: {agg_grad[0].mean().item():.6f}")
                    logging.info(f"   - First tensor std: {agg_grad[0].std().item():.6f}")
            logging.info(f"   - Aggregation stats: {aggregation_stats}")

        except Exception as e:
            logging.error(f"❌ AGGREGATION FAILED: {e}", exc_info=True)
            raise

        # 4. Update global model
        if agg_grad:
            logging.info("📥 Step 4: Applying gradient to global model...")
            try:
                self.aggregator.apply_gradient(agg_grad)
                logging.info("   ✅ Gradient applied successfully")
            except Exception as e:
                logging.error(f"   ❌ Failed to apply gradient: {e}", exc_info=True)
                raise
        else:
            logging.warning("⚠️  Step 4: No gradient to apply (agg_grad is None/empty)")

        # 5. Save individual gradients (This logic is already correct)
        if self.cfg.debug.save_individual_gradients:
            if round_number % self.cfg.debug.gradient_save_frequency == 0:
                logging.info(f"💾 Step 5: Saving gradients to disk...")
                grad_save_dir = Path(self.cfg.experiment.save_path) / "individual_gradients" / f"round_{round_number}"
                grad_save_dir.mkdir(parents=True, exist_ok=True)

                saved_count = 0
                for sid, grad in gradients_dict.items():
                    if grad is not None:
                        try:
                            torch.save(grad, grad_save_dir / f"{sid}_grad.pt")
                            saved_count += 1
                        except Exception as e:
                            logging.error(f"   Failed to save gradient for {sid}: {e}")

                logging.info(f"   Saved {saved_count}/{len(gradients_dict)} individual gradients to {grad_save_dir}")

                if agg_grad is not None:
                    try:
                        torch.save(agg_grad, grad_save_dir / f"aggregated_grad.pt")
                        logging.info(f"   Saved aggregated gradient")
                    except Exception as e:
                        logging.error(f"   Failed to save aggregated gradient: {e}")

        # 6. Create a simple record of the round's events.
        duration = time.time() - round_start_time
        round_record = {
            "round": round_number,
            "duration_sec": duration,
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),
            "attack_performed": bool(attack_log),
            "attack_victim": attack_log.get('victim_id') if attack_log else None
        }

        # 3b. Merge the detailed aggregator stats into the round record
        if aggregation_stats:
            round_record.update(aggregation_stats)

        # 7. Notify sellers of round end
        logging.info("📢 Step 6: Notifying sellers of round end...")
        for sid, seller in self.sellers.items():
            try:
                seller.round_end_process(round_number, (sid in selected_ids))
            except Exception as e:
                logging.error(f"   Failed to notify seller {sid}: {e}")

        logging.info("=" * 80)
        logging.info(f"✅ Round {round_number} Completed (Duration: {round_record['duration_sec']:.2f}s)")
        logging.info(f"   Selected: {len(selected_ids)}, Outliers: {len(outlier_ids)}")
        logging.info("=" * 80)

        return round_record, agg_grad

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