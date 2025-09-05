import logging
from typing import Dict, List, Tuple

import torch

from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str]]:
        logger.info(f"--- FedAvg Aggregation (Epoch {global_epoch}) ---")
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        num_valid_sellers = len(seller_updates)

        if num_valid_sellers == 0:
            return aggregated_gradient, [], list(seller_updates.keys())

        for update in seller_updates.values():
            for agg_grad, upd_grad in zip(aggregated_gradient, update):
                agg_grad.add_(upd_grad, alpha=1 / num_valid_sellers)

        seller_ids = list(seller_updates.keys())
        return aggregated_gradient, seller_ids, []
