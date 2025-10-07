import copy
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import clip_gradient_update, flatten_tensor
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator
from marketplace.utils.gradient_market_utils.clustering import optimal_k_gap, kmeans

logger = logging.getLogger("Aggregator")


def _cluster_and_score_martfl(similarities: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster similarities and score inliers.
    Returns scores and clustering metadata.
    """
    clustering_info = {}

    if len(similarities) < 2:
        clustering_info['num_clusters'] = 1
        clustering_info['cluster_method'] = 'insufficient_data'
        return np.ones_like(similarities), clustering_info

    sim_reshaped = similarities.reshape(-1, 1)
    max_k = min(len(similarities) - 1, 10)

    if max_k < 2:
        clustering_info['num_clusters'] = 1
        clustering_info['cluster_method'] = 'max_k_too_small'
        return np.ones_like(similarities), clustering_info

    # Find optimal k using gap statistic
    k = optimal_k_gap(sim_reshaped, k_max=max_k)
    clustering_info['optimal_k'] = k
    clustering_info['max_k_considered'] = max_k

    # Perform k-means clustering
    labels, centers = kmeans(x=sim_reshaped, k=k)

    # Identify inlier cluster (highest similarity center)
    inlier_label = np.argmax(centers)
    clustering_info['inlier_cluster_id'] = int(inlier_label)
    clustering_info['cluster_centers'] = centers
    clustering_info['cluster_labels'] = labels

    # Score: keep original similarity for inliers, zero for outliers
    scores = np.where(labels == inlier_label, similarities, 0.0)

    # Additional stats
    clustering_info['num_inliers'] = int(np.sum(labels == inlier_label))
    clustering_info['num_outliers'] = int(np.sum(labels != inlier_label))

    return scores, clustering_info


class MartflAggregator(BaseAggregator):

    def __init__(self, clip: bool, change_base: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip
        self.change_base = change_base
        self.baseline_id = "buyer"
        self.root_gradient = None # Will hold the buyer's root gradient for the round
        logger.info(f"MartflAggregator initialized:")
        logger.info(f"  - clip_gradients: {self.clip}")
        logger.info(f"  - change_baseline: {self.change_base}")
        logger.info(f"  - initial_baseline: {self.baseline_id}")

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        """
        martFL aggregation with comprehensive marketplace metrics.

        Uses clustering on cosine similarities to detect outliers.
        Optionally rotates the baseline seller each round.
        """
        logger.info(f"=== martFL Aggregation (Round {global_epoch}) ===")
        logger.info(f"Processing {len(seller_updates)} seller updates")
        logger.info(f"Current baseline: {self.baseline_id}")
        self.root_gradient = kwargs.get('root_gradient')
        buyer_data_loader = kwargs.get('buyer_data_loader')

        seller_ids = list(seller_updates.keys())

        # 1. Flatten and optionally clip updates
        flat_updates = {}
        for sid, upd in seller_updates.items():
            if self.clip:
                clipped = clip_gradient_update(upd, self.clip_norm)
                flat_updates[sid] = flatten_tensor(clipped)
            else:
                flat_updates[sid] = flatten_tensor(upd)

        # 2. Get or compute baseline
        if self.baseline_id and self.baseline_id in seller_updates:
            baseline_update_flat = flat_updates[self.baseline_id]
            baseline_source = 'seller'
            logger.info(f"Using seller '{self.baseline_id}' as baseline")
        else:
            # --- CHANGE: Use the pre-computed root gradient ---
            self.baseline_id = "buyer"
            if self.root_gradient is None:
                raise ValueError("MartflAggregator requires a 'root_gradient' when baseline is 'buyer'.")

            trust_gradient = self.root_gradient  # Use the passed-in gradient
            # --- REMOVED: Call to self._compute_trust_gradient() is gone ---

            if self.clip:
                trust_gradient = clip_gradient_update(trust_gradient, self.clip_norm)
            baseline_update_flat = flatten_tensor(trust_gradient)
            baseline_source = 'buyer_trust'
            logger.info("Using pre-computed buyer's trust gradient as baseline")

        baseline_norm = torch.norm(baseline_update_flat).item()

        # 3. Compute cosine similarities to baseline
        seller_updates_stack = torch.stack(list(flat_updates.values()))
        similarities = F.cosine_similarity(
            baseline_update_flat.unsqueeze(0),
            seller_updates_stack,
            dim=1
        )

        logger.info(f"Similarities - Mean: {similarities.mean().item():.4f}, "
                    f"Std: {similarities.std().item():.4f}, "
                    f"Min: {similarities.min().item():.4f}, "
                    f"Max: {similarities.max().item():.4f}")

        # 4. Initialize stats dictionary
        aggregation_stats = {
            'aggregation_method': 'martfl',
            'round': global_epoch,
            'clip_enabled': self.clip,
            'clip_norm': self.clip_norm if self.clip else None,
            'change_baseline_enabled': self.change_base,

            # Baseline info
            'baseline_id': self.baseline_id,
            'baseline_source': baseline_source,
            'baseline_norm': baseline_norm,

            # Raw similarities
            'cosine_similarities': {
                sid: sim.item() for sid, sim in zip(seller_ids, similarities)
            },
            'avg_similarity': similarities.mean().item(),
            'std_similarity': similarities.std().item(),
            'min_similarity': similarities.min().item(),
            'max_similarity': similarities.max().item(),
        }

        # 5. Cluster and score
        if self.baseline_id in seller_ids:
            baseline_idx = seller_ids.index(self.baseline_id)
            # Create a list of indices for all non-baseline sellers
            other_indices = [i for i in range(len(seller_ids)) if i != baseline_idx]
            # Create a tensor of similarities for clustering that excludes the baseline's perfect 1.0 score
            similarities_for_clustering = similarities[other_indices]
        else:
            # The baseline is the buyer, so all sellers are included in clustering
            baseline_idx = -1
            similarities_for_clustering = similarities

        # 6. Cluster and score using only the non-baseline sellers
        # This prevents the baseline's self-similarity of 1.0 from skewing the result
        inlier_scores_others, clustering_info = _cluster_and_score_martfl(
            similarities_for_clustering.cpu().numpy()
        )

        # 7. Reconstruct the full scores array, giving the baseline a perfect score by default
        if baseline_idx != -1:
            inlier_scores = np.zeros(len(seller_ids), dtype=float)
            # The baseline is always an inlier with its original similarity score (which is 1.0)
            inlier_scores[baseline_idx] = similarities[baseline_idx].item()
            # Fill in the scores for the other sellers
            inlier_scores[other_indices] = inlier_scores_others
        else:
            inlier_scores = inlier_scores_others

        # Add clustering metadata
        aggregation_stats['clustering'] = clustering_info

        logger.info(f"Clustering results:")
        logger.info(f"  - Optimal k: {clustering_info.get('optimal_k', 'N/A')}")
        logger.info(f"  - Inliers: {clustering_info.get('num_inliers', 'N/A')}")
        logger.info(f"  - Outliers: {clustering_info.get('num_outliers', 'N/A')}")

        # 6. Convert to weights
        weights = torch.tensor(inlier_scores, dtype=torch.float, device=self.device)
        total_weight = weights.sum()

        if total_weight > 1e-9:
            weights /= total_weight
        else:
            logger.warning("Total weight near zero! Setting all weights to 0")
            weights.zero_()

        # Log per-seller scores and weights
        aggregation_stats['inlier_scores'] = {
            sid: float(score) for sid, score in zip(seller_ids, inlier_scores)
        }
        aggregation_stats['seller_weights'] = {
            sid: w.item() for sid, w in zip(seller_ids, weights)
        }
        aggregation_stats['total_weight'] = total_weight.item()

        # 7. Aggregate gradients
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]

        for i, sid in enumerate(seller_ids):
            weight = weights[i].item()
            if weight > 0:
                for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                    agg_grad.add_(upd_grad, alpha=weight)
                logger.debug(f"Seller {sid}: similarity={similarities[i].item():.4f}, "
                             f"score={inlier_scores[i]:.4f}, weight={weight:.4f}")

        # 8. Optionally select new baseline for next round
        if self.change_base:
            next_baseline_id, baseline_selection_info = self._select_new_baseline_martfl(
                seller_updates, seller_ids, inlier_scores
            )
            aggregation_stats['next_baseline_id'] = next_baseline_id
            aggregation_stats['baseline_selection'] = baseline_selection_info

            logger.info(f"Baseline rotation: {self.baseline_id} -> {next_baseline_id}")
            if baseline_selection_info:
                logger.info(f"  Selection based on kappa: {baseline_selection_info.get('best_kappa', 'N/A'):.4f}")

            self.baseline_id = next_baseline_id

        # 9. Determine selected vs outlier sellers
        selected_ids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] == 0]

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

            # Adversary detection
            'known_adversaries': [sid for sid in seller_ids if 'adv' in sid],
            'detected_adversaries': [sid for sid in outlier_ids if 'adv' in sid],
            'missed_adversaries': [sid for sid in selected_ids if 'adv' in sid],
            'false_positives': [sid for sid in outlier_ids if 'bn' in sid],
        })

        # Detection metrics
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

    def _select_new_baseline_martfl(
            self,
            seller_updates: Dict,
            seller_ids: List[str],
            inlier_scores: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select new baseline seller based on kappa score evaluation.

        Returns:
            - new_baseline_id: ID of selected baseline seller
            - selection_info: Metadata about the selection process
        """
        candidate_ids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] > 0]

        selection_info = {
            'num_candidates': len(candidate_ids),
            'candidate_ids': candidate_ids,
            'evaluation_metric': 'kappa',
            'kappa_scores': {}
        }

        if not candidate_ids:
            logger.warning("No candidates for baseline selection, using 'buyer'")
            selection_info['reason'] = 'no_candidates'
            return "buyer", selection_info

        # Evaluate each candidate
        best_seller = "buyer"
        max_kappa = float('-inf')

        for sid in candidate_ids:
            # Create temporary model with this seller's update
            temp_model = self._get_model_from_update(seller_updates[sid])

            # Evaluate on buyer's data
            _, _, kappa, _ = self._evaluate_model(temp_model)
            selection_info['kappa_scores'][sid] = kappa

            if kappa > max_kappa:
                max_kappa = kappa
                best_seller = sid

        selection_info['best_kappa'] = max_kappa
        selection_info['selected_seller'] = best_seller

        return best_seller, selection_info

    def _get_model_from_update(self, update: List[torch.Tensor]) -> nn.Module:
        """Create a temporary model with the update applied."""
        temp_model = copy.deepcopy(self.global_model)
        with torch.no_grad():
            for model_p, update_p in zip(temp_model.parameters(), update):
                model_p.add_(update_p)
        return temp_model
