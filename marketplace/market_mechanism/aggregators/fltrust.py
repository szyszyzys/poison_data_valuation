import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from common.utils import clip_gradient_update, flatten_tensor
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger(__name__)


class FLTrustAggregator(BaseAggregator):
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str]]:
        logger.info(f"--- FLTrust Aggregation (Epoch {global_epoch}) ---")
        clip = kwargs.get('clip', False)

        # 1. Compute trusted baseline and process all updates
        trust_gradient = self._compute_trust_gradient()
        processed_updates = {sid: clip_gradient_update(upd, self.clip_norm) if clip else upd for sid, upd in
                             seller_updates.items()}
        trust_gradient = clip_gradient_update(trust_gradient, self.clip_norm) if clip else trust_gradient

        # 2. Flatten for similarity calculation
        flat_seller_updates = {sid: flatten_tensor(upd) for sid, upd in processed_updates.items()}
        flat_trust_gradient = flatten_tensor(trust_gradient)

        # 3. Compute trust scores
        trust_norm = torch.norm(flat_trust_gradient) + 1e-9
        seller_ids = list(flat_seller_updates.keys())
        seller_updates_stack = torch.stack(list(flat_seller_updates.values()))
        cos_sim = F.cosine_similarity(seller_updates_stack, flat_trust_gradient.unsqueeze(0), dim=1)
        trust_scores = torch.relu(cos_sim)

        # 4. Normalize scores and aggregate
        total_trust = trust_scores.sum()
        weights = trust_scores / total_trust if total_trust > 1e-9 else torch.zeros_like(trust_scores)

        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            for agg_grad, upd_grad in zip(aggregated_gradient, processed_updates[sid]):
                agg_grad.add_(upd_grad, alpha=weights[i])

        # 5. Scale and finalize
        for agg_grad in aggregated_gradient:
            agg_grad.mul_(trust_norm)

        selected_sids = [sid for i, sid in enumerate(seller_ids) if weights[i] > 0]
        outlier_sids = [sid for i, sid in enumerate(seller_ids) if weights[i] == 0]

        return aggregated_gradient, selected_sids, outlier_sids
