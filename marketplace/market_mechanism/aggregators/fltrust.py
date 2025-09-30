# In your aggregator file (e.g., marketplace/market/aggregator.py)
from typing import Dict, Tuple, List, Any

import torch

from common.utils import clip_gradient_update, flatten_tensor
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator, logger


class FLTrustAggregator(BaseAggregator):
    # --- FIX 1: Update the return signature to include the new stats dictionary ---
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

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

        # --- FIX 2: Create the stats dictionary for logging ---
        aggregation_stats = {
            "fltrust_trust_norm": trust_norm.item(),
            "fltrust_avg_score": trust_scores.mean().item(),
            # Log the score for each individual seller
            **{f"fltrust_score_{sid}": score.item() for sid, score in zip(seller_ids, trust_scores)}
        }

        # 4. Normalize scores and aggregate
        total_trust = trust_scores.sum()
        weights = trust_scores / total_trust if total_trust > 1e-9 else torch.zeros_like(trust_scores)

        # Log the final weight for each seller
        aggregation_stats.update({f"fltrust_weight_{sid}": w.item() for sid, w in zip(seller_ids, weights)})

        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            for agg_grad, upd_grad in zip(aggregated_gradient, processed_updates[sid]):
                agg_grad.add_(upd_grad, alpha=weights[i].item())  # Use .item() for scalar alpha

        # 5. Scale and finalize
        for agg_grad in aggregated_gradient:
            agg_grad.mul_(trust_norm.item())  # Use .item() for scalar multiplication

        selected_sids = [sid for i, sid in enumerate(seller_ids) if weights[i] > 0]
        outlier_sids = [sid for i, sid in enumerate(seller_ids) if weights[i] == 0]

        # Add summary stats
        aggregation_stats["fltrust_num_selected"] = len(selected_sids)
        aggregation_stats["fltrust_num_rejected"] = len(outlier_sids)

        # --- FIX 3: Return the new stats dictionary ---
        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
