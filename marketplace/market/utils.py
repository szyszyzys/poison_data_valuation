from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from attack.evaluation.evaluation_backdoor import evaluate_attack_performance


class FederatedEvaluator:
    """Encapsulates all logic for evaluating a federated model."""

    def __init__(self, loss_fn: nn.Module, device: str):
        self.loss_fn = loss_fn
        self.device = device

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model accuracy and loss."""
        model.to(self.device)
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                total_loss += self.loss_fn(outputs, y).item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)
        return {"acc": total_correct / total_samples, "loss": total_loss / total_samples}

    def evaluate_backdoor_asr(self, model: nn.Module, test_loader: DataLoader, backdoor_generator,
                              target_label) -> float:
        """Evaluates the Attack Success Rate for a backdoor attack."""
        poison_metrics = evaluate_attack_performance(
            model, test_loader, self.device, backdoor_generator, target_label
        )
        return poison_metrics.get("attack_success_rate")
