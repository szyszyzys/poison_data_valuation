# in marketplace/market/evaluation/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from attack.evaluation.evaluation_backdoor import evaluate_attack_performance
from common.gradient_market_configs import AppConfig


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation strategies."""

    def __init__(self, cfg: AppConfig, device: str, **kwargs):
        self.cfg = cfg
        self.device = device
        self.runtime_kwargs = kwargs  # Store any extra args

    @abstractmethod
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Runs the evaluation and returns a dictionary of metrics."""
        raise NotImplementedError


class CleanEvaluator(BaseEvaluator):
    """Evaluates standard model performance (accuracy and loss)."""

    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device, **kwargs)
        # FIX: Instantiate the loss function here to make the class self-contained.
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        if len(test_loader.dataset) == 0:
            logging.warning("CleanEvaluator: Test loader is empty. Returning zero metrics.")
            return {"acc": 0.0, "loss": 0.0}

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

        acc = total_correct / total_samples
        loss = total_loss / total_samples
        return {"acc": acc, "loss": loss}


# in marketplace/market/evaluation/base.py

# ... (other imports)
from marketplace.seller.gradient_seller import AdvancedBackdoorAdversarySeller  # 1. ADD THIS IMPORT


class BackdoorEvaluator(BaseEvaluator):
    """A specialized evaluator for measuring the Attack Success Rate (ASR) of backdoor attacks."""

    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device, **kwargs)

        # --- THIS IS THE UPDATED LOGIC ---
        # Only try to create a generator if a backdoor attack is actually active.
        if 'backdoor' in cfg.adversary_seller_config.poisoning.type:
            # 2. Call the static method directly from the seller class
            self.backdoor_generator = AdvancedBackdoorAdversarySeller._create_poison_generator(
                adv_cfg=self.cfg.adversary_seller_config,
                model_type=self.cfg.experiment.dataset_type,  # 'image' or 'text'
                **self.runtime_kwargs
            )
        else:
            self.backdoor_generator = None
            logging.warning("BackdoorEvaluator initialized, but no backdoor attack is configured.")

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        if not self.backdoor_generator:
            return {}  # If no generator, do nothing.

        # The rest of your evaluate method is already correct
        poison_cfg = self.cfg.adversary_seller_config.poisoning
        active_params = poison_cfg.active_params
        if not active_params:
            return {}
        target_label = active_params.target_label

        metrics = evaluate_attack_performance(
            model, test_loader, self.device, self.backdoor_generator, target_label
        )
        return {"asr": metrics.get("attack_success_rate", 0.0)}


EVALUATOR_MAP = {
    "clean": CleanEvaluator,
    "backdoor": BackdoorEvaluator,
}


def create_evaluators(cfg: AppConfig, device: str, **kwargs: Dict[str, Any]) -> List[BaseEvaluator]:
    """Creates a list of evaluator instances based on the config."""
    evaluator_list = []
    evaluator_names = cfg.experiment.evaluations

    for name in evaluator_names:
        EvaluatorClass = EVALUATOR_MAP.get(name)
        if EvaluatorClass:
            # FIX: Pass the **kwargs to the constructor.
            evaluator_list.append(EvaluatorClass(cfg, device, **kwargs))
        else:
            logging.warning(f"Unknown evaluator '{name}' specified in config.")

    return evaluator_list
