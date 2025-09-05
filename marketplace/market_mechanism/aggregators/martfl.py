import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from common.utils import clip_gradient_update, flatten_tensor
from marketplace.market_mechanism.aggregators.base_aggregator import BaseAggregator
from marketplace.utils.gradient_market_utils.clustering import optimal_k_gap, kmeans

logger = logging.getLogger("Aggregator")


class MartflAggregator(BaseAggregator):
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str]]:
        logger.info(f"--- martFL Aggregation (Epoch {global_epoch}) ---")
        clip = kwargs.get("clip", False)  # Defaults to False if not provided
        change_base = kwargs.get("change_base", True)  # Defaults to True if not provided
        seller_ids = list(seller_updates.keys())
        flat_updates = {sid: flatten_tensor(clip_gradient_update(upd, self.clip_norm) if clip else upd) for sid, upd in
                        seller_updates.items()}

        if self.baseline_id and self.baseline_id in seller_updates:
            baseline_update_flat = flat_updates[self.baseline_id]
        else:
            self.baseline_id = "buyer"
            trust_gradient = self._compute_trust_gradient()
            baseline_update_flat = flatten_tensor(
                clip_gradient_update(trust_gradient, self.clip_norm) if clip else trust_gradient)

        similarities = F.cosine_similarity(baseline_update_flat.unsqueeze(0), torch.stack(list(flat_updates.values())),
                                           dim=1)
        inlier_scores = self._cluster_and_score_martfl(similarities.cpu().numpy())
        weights = torch.tensor(inlier_scores, dtype=torch.float, device=self.device)
        total_weight = weights.sum()

        if total_weight > 1e-9:
            weights /= total_weight
        else:
            weights.zero_()

        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                agg_grad.add_(upd_grad, alpha=weights[i])

        if change_base: self.baseline_id = self._select_new_baseline_martfl(seller_updates, seller_ids,
                                                                            inlier_scores)

        selected_sids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] > 0]
        outlier_sids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] == 0]
        return aggregated_gradient, selected_sids, outlier_sids

    def _cluster_and_score_martfl(self, similarities: np.ndarray) -> np.ndarray:
        if len(similarities) < 2: return np.ones_like(similarities)
        sim_reshaped = similarities.reshape(-1, 1)
        max_k = min(len(similarities) - 1, 10)
        if max_k < 2: return np.ones_like(similarities)
        k = optimal_k_gap(sim_reshaped, k_max=max_k)
        labels, centers = kmeans(x=sim_reshaped, k=k)
        inlier_label = np.argmax(centers)
        return np.where(labels == inlier_label, similarities, 0.0)

    def _select_new_baseline_martfl(self, seller_updates: Dict, seller_ids: List[str], inlier_scores: np.ndarray):
        candidate_ids = [sid for i, sid in enumerate(seller_ids) if inlier_scores[i] > 0]
        if not candidate_ids: return "buyer"
        best_seller, max_kappa = "buyer", float('-inf')
        for sid in candidate_ids:
            temp_model = self._get_model_from_update(seller_updates[sid])
            _, _, kappa, _ = self._evaluate_model(temp_model)
            if kappa > max_kappa:
                max_kappa, best_seller = kappa, sid
        return best_seller

    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float, float, float]:
        # (This method remains the same as before)
        model.to(self.device);
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []
        if not self.buyer_data_loader: return 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for data, labels in self.buyer_data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                total_loss += self.loss_fn(outputs, labels).item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += data.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        if total_samples == 0: return 0.0, 0.0, 0.0, 0.0
        acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        kappa = cohen_kappa_score(all_labels, all_preds)
        return avg_loss, acc, kappa, 0.0  # F1 placeholder

    def _get_model_from_update(self, update: List[torch.Tensor]) -> nn.Module:
        temp_model = copy.deepcopy(self.global_model)
        with torch.no_grad():
            for model_p, update_p in zip(temp_model.parameters(), update):
                model_p.add_(update_p)
        return temp_model
