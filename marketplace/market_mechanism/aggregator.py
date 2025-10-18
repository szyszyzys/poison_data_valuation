import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.gradient_market_configs import AggregationConfig
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator
from marketplace.market_mechanism.aggregators.fedavg import FedAvgAggregator
from marketplace.market_mechanism.aggregators.fltrust import FLTrustAggregator
from marketplace.market_mechanism.aggregators.martfl import MartflAggregator
from marketplace.market_mechanism.aggregators.skymask import SkymaskAggregator

logger = logging.getLogger("Aggregator")


class Aggregator:
    """
    Acts as a factory and orchestrator for different aggregation strategies.
    """

    def __init__(self,
                 global_model: nn.Module,
                 device: torch.device,
                 loss_fn: nn.Module,
                 buyer_data_loader: Optional[DataLoader],
                 agg_config: AggregationConfig):

        self.strategy: BaseAggregator

        strategy_map = {
            "fedavg": FedAvgAggregator,
            "fltrust": FLTrustAggregator,
            "martfl": MartflAggregator,
            "skymask": SkymaskAggregator,
        }

        method = agg_config.method
        if method not in strategy_map:
            raise NotImplementedError(f"Aggregation method '{method}' is not implemented.")

        strategy_params = getattr(agg_config, method, {})
        strategy_kwargs = asdict(strategy_params) if strategy_params else {}

        StrategyClass = strategy_map[method]
        self.strategy = StrategyClass(
            global_model=global_model,
            device=device,
            loss_fn=loss_fn,
            buyer_data_loader=buyer_data_loader,
            clip_norm=agg_config.clip_norm,
            **strategy_kwargs
        )
        self.device = device
        self.buyer_data_loader = buyer_data_loader
        # Cache this for efficiency and consistency
        self.expected_num_params = len(list(self.strategy.global_model.parameters()))
        logger.info(f"Global model has {self.expected_num_params} parameters")

    def _validate_and_standardize_updates(self, updates: Dict[str, list]) -> Dict[str, List[torch.Tensor]]:
        """
        A more robust standardization that also validates shapes against the global model.
        """
        if not updates:
            logger.error("❌ Received empty updates dictionary")
            return {}

        logger.info(f"📊 Validating updates from {len(updates)} sellers")

        standardized = {}
        global_model_params = list(self.strategy.global_model.parameters())

        # Debug: Log what we're expecting
        logger.debug(f"Expected number of parameters: {len(global_model_params)}")

        for seller_id, update_list in updates.items():
            # --- ENHANCED DEBUGGING ---
            logger.debug(f"Processing seller {seller_id}:")
            logger.debug(f"  - Update is None: {update_list is None}")
            logger.debug(f"  - Update type: {type(update_list)}")

            if update_list is None:
                logger.warning(f"⚠️  Seller {seller_id} returned None. Skipping.")
                continue

            if not isinstance(update_list, (list, tuple)):
                logger.warning(
                    f"⚠️  Seller {seller_id} update is not a list/tuple (type: {type(update_list)}). Skipping.")
                continue

            logger.debug(f"  - Update length: {len(update_list)}")
            logger.debug(f"  - Expected length: {len(global_model_params)}")

            # Check if empty
            if len(update_list) == 0:
                logger.warning(f"⚠️  Seller {seller_id} provided an empty update list. Skipping.")
                continue

            # Check length mismatch
            if len(update_list) != len(global_model_params):
                logger.warning(
                    f"⚠️  Seller {seller_id} update length mismatch: "
                    f"got {len(update_list)}, expected {len(global_model_params)}. Skipping."
                )
                # Debug: Show first few items to understand structure
                logger.debug(f"  - First update item type: {type(update_list[0]) if update_list else 'N/A'}")
                if update_list and hasattr(update_list[0], 'shape'):
                    logger.debug(f"  - First update item shape: {update_list[0].shape}")
                continue

            # Validate and convert each parameter
            try:
                seller_tensors = []
                valid = True

                for i, p_update in enumerate(update_list):
                    # Convert to tensor if needed
                    if isinstance(p_update, np.ndarray):
                        tensor_update = torch.from_numpy(p_update).float().to(self.strategy.device)
                    elif isinstance(p_update, torch.Tensor):
                        tensor_update = p_update.float().to(self.strategy.device)
                    else:
                        logger.error(
                            f"❌ Seller {seller_id} param {i}: unexpected type {type(p_update)}. Skipping seller."
                        )
                        valid = False
                        break

                    # Validate shape
                    expected_shape = global_model_params[i].shape
                    if tensor_update.shape != expected_shape:
                        logger.error(
                            f"❌ SHAPE MISMATCH for seller {seller_id} at parameter {i}:\n"
                            f"   Expected: {expected_shape}\n"
                            f"   Got:      {tensor_update.shape}\n"
                            f"   Skipping seller."
                        )
                        valid = False
                        break

                    seller_tensors.append(tensor_update)

                if valid:
                    standardized[seller_id] = seller_tensors
                    logger.debug(f"✅ Seller {seller_id} update validated successfully")

            except Exception as e:
                logger.error(f"❌ Exception validating seller {seller_id}: {e}", exc_info=True)

        logger.info(f"✅ Validated {len(standardized)}/{len(updates)} seller updates")
        return standardized

    def aggregate(self, global_epoch: int, seller_updates: Dict, root_gradient: Optional[List[torch.Tensor]] = None,
                  **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        """
        Standardizes updates and delegates the aggregation to the selected strategy.
        Now consistently returns 4 values.
        """
        logger.info(f"🔄 Starting aggregation for epoch {global_epoch}")
        logger.info(f"   Received updates from {len(seller_updates)} sellers: {list(seller_updates.keys())}")

        s_updates_tensor = self._validate_and_standardize_updates(seller_updates)

        if not s_updates_tensor:
            logger.error("❌ No valid seller updates after standardization. Aborting aggregation.")
            logger.error(f"   Original sellers: {list(seller_updates.keys())}")

            # Return zero gradients
            zero_grad = [torch.zeros_like(p) for p in self.strategy.global_model.parameters()]
            return zero_grad, [], list(seller_updates.keys()), {}

        logger.info(f"✅ Proceeding with {len(s_updates_tensor)} valid updates")

        # === 2. Build the arguments for the strategy call ===
        strategy_args = {
            'global_epoch': global_epoch,
            'seller_updates': s_updates_tensor,
            **kwargs  # Pass through any other miscellaneous args
        }

        if isinstance(self.strategy, (FLTrustAggregator, MartflAggregator, SkymaskAggregator)):
            if root_gradient is None:
                # This is a critical error if the wrong configuration is used.
                raise ValueError(f"{self.strategy.__class__.__name__} requires a 'root_gradient', but received None.")
            strategy_args['root_gradient'] = root_gradient

        # === 4. Call the strategy with the prepared arguments ===
        return self.strategy.aggregate(**strategy_args)

    def apply_gradient(self, aggregated_gradient: List[torch.Tensor]):
        self.strategy.apply_gradient(aggregated_gradient)
