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

    def _validate_and_standardize_updates(self, updates: Dict[str, list]) -> Dict[str, List[torch.Tensor]]:
        """
        A more robust standardization that also validates shapes against the global model.
        """
        standardized = {}
        global_model_params = list(self.strategy.global_model.parameters())

        for seller_id, update_list in updates.items():
            if not update_list or len(update_list) != len(global_model_params):
                logger.warning(f"Seller {seller_id} provided an invalid or empty update. Skipping.")
                continue

            try:
                seller_tensors = []
                valid = True
                for i, p_update in enumerate(update_list):
                    tensor_update = (torch.from_numpy(p_update) if isinstance(p_update, np.ndarray) else p_update).float().to(self.strategy.device)

                    # --- Key Validation Step ---
                    if tensor_update.shape != global_model_params[i].shape:
                        logger.error(
                            f"SHAPE MISMATCH for seller {seller_id} at parameter {i}. "
                            f"Global model shape: {global_model_params[i].shape}, "
                            f"Seller update shape: {tensor_update.shape}. Skipping seller."
                        )
                        valid = False
                        break
                    seller_tensors.append(tensor_update)

                if valid:
                    standardized[seller_id] = seller_tensors

            except Exception as e:
                logger.error(f"Could not convert or validate update for seller {seller_id}: {e}")

        return standardized

    def aggregate(self, global_epoch: int, seller_updates: Dict, **kwargs) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        """
        Standardizes updates and delegates the aggregation to the selected strategy.
        Now consistently returns 4 values.
        """
        s_updates_tensor = self._validate_and_standardize_updates(seller_updates)

        if not s_updates_tensor:
            logger.error("No valid seller updates after standardization. Aborting aggregation.")
            zero_grad = [torch.zeros_like(p) for p in self.strategy.global_model.parameters()]

            return zero_grad, [], list(seller_updates.keys()), {}

        return self.strategy.aggregate(
            global_epoch=global_epoch,
            seller_updates=s_updates_tensor,
            **kwargs
        )

    def apply_gradient(self, aggregated_gradient: List[torch.Tensor]):
        self.strategy.apply_gradient(aggregated_gradient)

