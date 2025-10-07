import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any

from common.utils import clip_gradient_update, flatten_tensor
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator, logger


class FLTrustAggregator(BaseAggregator):

    def __init__(self, global_model, device, loss_fn, buyer_data_loader, clip_norm):
        super().__init__(global_model, device, loss_fn, buyer_data_loader, clip_norm)
        self.root_gradient = None  # Holds the root gradient for the current round

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        """
        FLTrust aggregation with comprehensive marketplace metrics.
        """
        logger.info(f"=== FLTrust Aggregation (Round {global_epoch}) ===")
        logger.info(f"Processing {len(seller_updates)} seller updates")

        # --- CHANGE: Get the pre-computed root gradient from kwargs ---
        self.root_gradient = kwargs.get('root_gradient')
        if self.root_gradient is None:
            raise ValueError("FLTrustAggregator requires a 'root_gradient' to be passed.")

        # 2. Process and clip all updates
        processed_updates = {}
        for sid, upd in seller_updates.items():
            if clip:
                processed_updates[sid] = clip_gradient_update(upd, self.clip_norm)
            else:
                processed_updates[sid] = upd

        if clip:
            trust_gradient = clip_gradient_update(trust_gradient, self.clip_norm)

        # 3. Flatten for similarity calculation
        flat_seller_updates = {sid: flatten_tensor(upd) for sid, upd in processed_updates.items()}
        flat_trust_gradient = flatten_tensor(trust_gradient)
        trust_norm = torch.norm(flat_trust_gradient)

        logger.info(f"Trust gradient norm: {trust_norm.item():.4f}")

        # 4. Compute trust scores (cosine similarity with ReLU)
        seller_ids = list(flat_seller_updates.keys())
        seller_updates_stack = torch.stack(list(flat_seller_updates.values()))

        cos_sim = F.cosine_similarity(
            seller_updates_stack,
            flat_trust_gradient.unsqueeze(0),
            dim=1
        )
        trust_scores = torch.relu(cos_sim)  # Negative similarities become 0

        logger.info(f"Trust scores - Mean: {trust_scores.mean().item():.4f}, "
                    f"Std: {trust_scores.std().item():.4f}, "
                    f"Min: {trust_scores.min().item():.4f}, "
                    f"Max: {trust_scores.max().item():.4f}")

        # 5. Initialize comprehensive stats dictionary
        aggregation_stats = {
            # Method metadata
            'aggregation_method': 'fltrust',
            'round': global_epoch,
            'clip_enabled': clip,
            'clip_norm': self.clip_norm if clip else None,

            # Trust baseline metrics
            'trust_gradient_norm': trust_norm.item(),

            # Trust score statistics
            'avg_trust_score': trust_scores.mean().item(),
            'std_trust_score': trust_scores.std().item(),
            'min_trust_score': trust_scores.min().item(),
            'max_trust_score': trust_scores.max().item(),
            'median_trust_score': trust_scores.median().item(),

            # Per-seller raw scores
            'trust_scores': {sid: score.item() for sid, score in zip(seller_ids, trust_scores)},

            # Per-seller cosine similarities (before ReLU)
            'cosine_similarities': {sid: sim.item() for sid, sim in zip(seller_ids, cos_sim)}
        }

        # 6. Normalize trust scores to get aggregation weights
        total_trust = trust_scores.sum()

        if total_trust > 1e-9:
            weights = trust_scores / total_trust
        else:
            logger.warning("Total trust score near zero! Using uniform weights.")
            weights = torch.ones_like(trust_scores) / len(trust_scores)

        # Log per-seller weights
        aggregation_stats['seller_weights'] = {
            sid: w.item() for sid, w in zip(seller_ids, weights)
        }
        aggregation_stats['total_trust_sum'] = total_trust.item()

        # 7. Weighted aggregation
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]

        for i, sid in enumerate(seller_ids):
            weight = weights[i].item()

            if weight > 0:
                for agg_grad, seller_grad in zip(aggregated_gradient, processed_updates[sid]):
                    agg_grad.add_(seller_grad, alpha=weight)

                logger.debug(f"Seller {sid}: trust={trust_scores[i].item():.4f}, weight={weight:.4f}")

        # 8. Scale by trust norm (FLTrust paper's normalization)
        for agg_grad in aggregated_gradient:
            agg_grad.mul_(trust_norm.item())

        # 9. Determine selected vs outlier sellers
        selected_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] == 0]

        logger.info(f"Selected: {len(selected_ids)} sellers")
        logger.info(f"Rejected (outliers): {len(outlier_ids)} sellers")

        if outlier_ids:
            logger.info(f"Outlier IDs: {outlier_ids}")

        # 10. Add summary statistics
        aggregation_stats.update({
            'num_sellers': len(seller_ids),
            'num_selected': len(selected_ids),
            'num_outliers': len(outlier_ids),
            'selection_rate': len(selected_ids) / len(seller_ids) if seller_ids else 0,
            'outlier_rate': len(outlier_ids) / len(seller_ids) if seller_ids else 0,

            # Identify adversary detection
            'known_adversaries': [sid for sid in seller_ids if 'adv' in sid],
            'detected_adversaries': [sid for sid in outlier_ids if 'adv' in sid],
            'missed_adversaries': [sid for sid in selected_ids if 'adv' in sid],
            'false_positives': [sid for sid in outlier_ids if 'bn' in sid],
        })

        # Detection performance
        num_known_adv = len(aggregation_stats['known_adversaries'])
        num_detected_adv = len(aggregation_stats['detected_adversaries'])
        num_benign = len(seller_ids) - num_known_adv
        num_false_pos = len(aggregation_stats['false_positives'])

        aggregation_stats['adversary_detection_rate'] = (
            num_detected_adv / num_known_adv if num_known_adv > 0 else 0
        )
        aggregation_stats['false_positive_rate'] = (
            num_false_pos / num_benign if num_benign > 0 else 0
        )

        logger.info(f"Adversary detection: {num_detected_adv}/{num_known_adv} "
                    f"({aggregation_stats['adversary_detection_rate']:.1%})")
        logger.info(f"False positives: {num_false_pos}/{num_benign} "
                    f"({aggregation_stats['false_positive_rate']:.1%})")

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats
