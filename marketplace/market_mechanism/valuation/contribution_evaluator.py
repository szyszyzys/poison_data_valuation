import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

from common.gradient_market_configs import AppConfig


class ContributionEvaluator:
    """
    Handles the valuation of seller contributions for a single round.

    This class takes the state of the marketplace after gradient collection
    and computes various contribution scores for each seller.
    """

    def __init__(self, cfg: AppConfig):
        """
        Initializes the evaluator.

        Args:
            cfg: The main AppConfig object, used to check for settings
                 like compute_gradient_similarity.
        """
        self.cfg = cfg
        logging.info("ContributionEvaluator initialized.")

    def _flatten_and_calc_similarity(self, grad1: List[torch.Tensor],
                                     grad2: List[torch.Tensor]) -> Optional[float]:
        """Helper to safely flatten two gradients and compute cosine similarity."""
        if not grad1 or not grad2:
            return None
        try:
            # Move to CPU for similarity calculation to avoid CUDA sync issues
            flat1 = torch.cat([g.detach().cpu().flatten() for g in grad1])
            flat2 = torch.cat([g.detach().cpu().flatten() for g in grad2])

            # Handle potential all-zero gradients
            if torch.all(flat1 == 0) or torch.all(flat2 == 0):
                return 0.0

            return torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0),
                flat2.unsqueeze(0)
            ).item()
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return None

    def evaluate_round(
            self,
            round_number: int,
            seller_gradients: Dict[str, List[torch.Tensor]],
            seller_stats: Dict[str, Dict[str, Any]],
            oracle_gradient: Optional[List[torch.Tensor]],
            buyer_gradient: Optional[List[torch.Tensor]],
            aggregated_gradient: Optional[List[torch.Tensor]],
            aggregation_stats: Dict[str, Any],
            selected_ids: List[str],
            outlier_ids: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Computes all contribution scores for the current round.

        Returns:
            Tuple[
                Dict[str, Dict[str, Any]],  # seller_valuations
                Dict[str, Any]             # aggregate_metrics
            ]
        """
        seller_ids = list(seller_gradients.keys())
        seller_valuations = {sid: {} for sid in seller_ids}
        aggregate_metrics = {}

        # --- 1. Basic Stats & Market-Based Valuation ---
        for sid in seller_ids:
            stats = seller_stats.get(sid, {})
            seller_valuations[sid]['selected'] = sid in selected_ids
            seller_valuations[sid]['outlier'] = sid in outlier_ids
            seller_valuations[sid]['train_loss'] = stats.get('train_loss')
            seller_valuations[sid]['num_samples'] = stats.get('num_samples', 0)

            # Market-Based (Implicit) Valuation
            seller_valuations[sid]['selection_score'] = 1.0 if (sid in selected_ids) else 0.0

            # Market-Based (Weight) Valuation
            if 'seller_weights' in aggregation_stats:
                weight = aggregation_stats['seller_weights'].get(sid, 0.0)
                seller_valuations[sid]['aggregation_weight'] = weight

        # --- 2. Gradient Norm (Effort/Magnitude) ---
        gradient_norms = []
        for sid, grad in seller_gradients.items():
            if grad:
                norm = sum(torch.norm(g).item() ** 2 for g in grad) ** 0.5
                seller_valuations[sid]['gradient_norm'] = norm
                gradient_norms.append(norm)

        if gradient_norms:
            aggregate_metrics['avg_gradient_norm'] = np.mean(gradient_norms)
            aggregate_metrics['std_gradient_norm'] = np.std(gradient_norms)
            aggregate_metrics['min_gradient_norm'] = np.min(gradient_norms)
            aggregate_metrics['max_gradient_norm'] = np.max(gradient_norms)

        # --- 3. Similarity-Based Valuation (Quality & Relevance) ---
        sims_to_oracle = []
        sims_to_buyer = []
        sims_to_aggregate = []

        for sid, grad in seller_gradients.items():
            # a) Oracle Similarity (True Quality)
            oracle_sim = self._flatten_and_calc_similarity(grad, oracle_gradient)
            seller_valuations[sid]['sim_to_oracle'] = oracle_sim
            if oracle_sim is not None:
                sims_to_oracle.append(oracle_sim)

            # b) Buyer Similarity (Task Relevance)
            buyer_sim = self._flatten_and_calc_similarity(grad, buyer_gradient)
            seller_valuations[sid]['sim_to_buyer'] = buyer_sim
            if buyer_sim is not None:
                sims_to_buyer.append(buyer_sim)

            # --- c) CGSV (Cosine Gradient Shapley Value) ---
            # This is the "Gradient Shapley" approximation.
            # It measures contribution as the alignment with the
            # final aggregated consensus gradient.
            cgsv_score = self._flatten_and_calc_similarity(grad, aggregated_gradient)
            seller_valuations[sid]['sim_to_aggregate_cgsv'] = cgsv_score
            if cgsv_score is not None:
                sims_to_aggregate.append(cgsv_score)
            # -----------------------------------------------
        # Aggregate similarity stats
        if sims_to_oracle:
            aggregate_metrics['avg_sim_to_oracle'] = np.mean(sims_to_oracle)
            aggregate_metrics['std_sim_to_oracle'] = np.std(sims_to_oracle)
        if sims_to_buyer:
            aggregate_metrics['avg_sim_to_buyer'] = np.mean(sims_to_buyer)
            aggregate_metrics['std_sim_to_buyer'] = np.std(sims_to_buyer)
        if sims_to_aggregate:
            aggregate_metrics['avg_sim_to_aggregate_cgsv'] = np.mean(sims_to_aggregate)
            aggregate_metrics['std_sim_to_aggregate_cgsv'] = np.std(sims_to_aggregate)
        # --- 4. Adversary Detection Metrics ---
        known_adversaries = [sid for sid in seller_ids if 'adv' in sid]
        detected_adversaries = [sid for sid in outlier_ids if 'adv' in sid]
        benign_outliers = [sid for sid in outlier_ids if 'bn' in sid]

        aggregate_metrics['num_known_adversaries'] = len(known_adversaries)
        aggregate_metrics['num_detected_adversaries'] = len(detected_adversaries)
        aggregate_metrics['num_benign_outliers'] = len(benign_outliers)
        aggregate_metrics['adversary_detection_rate'] = (
            len(detected_adversaries) / len(known_adversaries)
            if known_adversaries else (1.0 if not detected_adversaries else 0.0)  # 1.0 if 0/0
        )
        non_adv_count = len(seller_ids) - len(known_adversaries)
        aggregate_metrics['false_positive_rate'] = (
            len(benign_outliers) / non_adv_count
            if non_adv_count > 0 else 0.0
        )

        # --- 5. Add Pricing (Example) ---
        # This is where you'd implement your pricing model.
        # Example: Pay-for-Quality (proportional to oracle similarity)
        BASE_PRICE_PER_ROUND = 0.1  # Example: 10 cents
        for sid in seller_ids:
            price = 0.0
            if seller_valuations[sid]['selected']:
                oracle_sim = seller_valuations[sid].get('sim_to_oracle', 0.0)
                # Pay based on quality, don't pay for negative contributions
                price = BASE_PRICE_PER_ROUND * max(0, oracle_sim)
            seller_valuations[sid]['price_paid'] = price

        logging.info("Contribution evaluation complete.")
        return seller_valuations, aggregate_metrics
