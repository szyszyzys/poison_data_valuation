import copy
import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Abstract base class for all aggregation strategies."""

    def __init__(self, global_model: nn.Module, device: torch.device,
                 loss_fn: nn.Module, buyer_data_loader: DataLoader, clip_norm: float):
        self.global_model = global_model
        self.device = device
        self.loss_fn = loss_fn
        self.buyer_data_loader = buyer_data_loader
        self.clip_norm = clip_norm
        self._buyer_data_iter = None  # Initialize iterator as None

    @abstractmethod
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]], **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str]]:
        """
        The core aggregation logic. Must be implemented by all subclasses.
        Returns the aggregated gradient, a list of selected seller IDs, and a list of outlier IDs.
        """
        raise NotImplementedError

    def apply_gradient(self, aggregated_gradient: List[torch.Tensor], learning_rate: float = 1.0):
        """
        Applies the provided aggregated gradient to the global model.

        Args:
            aggregated_gradient (List[torch.Tensor]): The gradient to apply.
            learning_rate (float): The server-side learning rate. Defaults to 1.0,
                                   assuming learning rate is handled on the client or
                                   incorporated into the gradient itself (e.g., FedAvg).
        """
        if not aggregated_gradient:
            logger.warning("apply_gradient called with an empty or None gradient. Skipping model update.")
            return

        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), aggregated_gradient):
                # The update rule is: param = param - learning_rate * grad
                # The in-place equivalent is param.add_(grad, alpha=-learning_rate)
                param.add_(grad, alpha=-learning_rate)

    def _get_model_from_update(self, update: List[torch.Tensor]) -> nn.Module:
        temp_model = copy.deepcopy(self.global_model)
        with torch.no_grad():
            for model_p, update_p in zip(temp_model.parameters(), update):
                model_p.add_(update_p)
        return temp_model

    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float, float, float]:
        """Evaluates a model using the buyer's data loader."""
        model.to(self.device)
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
        return avg_loss, acc, kappa, 0.0
