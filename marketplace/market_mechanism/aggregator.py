import logging
from dataclasses import asdict
from typing import Dict, List, Optional

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

        # Map strings to the strategy classes
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
        # Convert dataclass to dict if it exists
        strategy_kwargs = asdict(strategy_params) if strategy_params else {}

        # Instantiate the chosen strategy with its specific parameters
        StrategyClass = strategy_map[method]
        self.strategy = StrategyClass(
            global_model=global_model,
            device=device,
            loss_fn=loss_fn,
            buyer_data_loader=buyer_data_loader,
            clip_norm=agg_config.clip_norm,
            **strategy_kwargs  # <-- Unpack the specific params here
        )

    def _standardize_updates(self, updates: Dict[str, list]) -> Dict[str, List[torch.Tensor]]:
        # This pre-processing step can stay in the main orchestrator.
        standardized = {}
        for seller_id, update_list in updates.items():
            if not update_list:
                logger.warning(f"Seller {seller_id} provided an empty update. Skipping.")
                continue
            try:
                standardized[seller_id] = [
                    (torch.from_numpy(p) if isinstance(p, np.ndarray) else p).float().to(self.strategy.device)
                    for p in update_list
                ]
            except Exception as e:
                logger.error(f"Could not convert update for seller {seller_id} to tensor: {e}")
        return standardized

    def aggregate(self, global_epoch: int, seller_updates: Dict, **kwargs):
        """
        Standardizes updates and delegates the aggregation to the selected strategy.
        """
        s_updates_tensor = self._standardize_updates(seller_updates)

        if not s_updates_tensor:
            logger.error("No valid seller updates after standardization. Aborting aggregation.")
            zero_grad = [torch.zeros_like(p) for p in self.strategy.global_model.parameters()]
            return zero_grad, [], list(seller_updates.keys())

        # Delegate the call to the strategy instance
        return self.strategy.aggregate(
            global_epoch=global_epoch,
            seller_updates=s_updates_tensor,
            **kwargs
        )
