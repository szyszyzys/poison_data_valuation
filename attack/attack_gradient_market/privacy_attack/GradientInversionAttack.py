import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

# Assumes attack_base.py and attack_utils.py are in the same directory
from attack.attack_gradient_market.privacy_attack import BaseAttacker
from common.attack_utils import (DATASET_STATS, evaluate_reconstruction_metrics,
                                 visualize_reconstruction)


# --- 1. Configuration ---

@dataclass
class GIAConfig:
    """Configuration for the Gradient Inversion Attack."""
    num_images: int = 1
    iterations: int = 2000
    lrs_to_try: List[float] = field(default_factory=lambda: [0.1, 0.01])
    label_type: str = 'optimize'
    loss_type: str = 'cosine'
    init_type: str = 'gaussian'
    regularization_weight: float = 1e-4
    log_interval: Optional[int] = 500
    save_visuals: bool = True
    save_dir: str = "./attack_visualizations/gia"


# --- 2. Refactored Attacker Class ---

class GradientInversionAttacker(BaseAttacker):
    """
    Inherits from BaseAttacker to perform Gradient Inversion.
    This is a server-side attack to reconstruct client data from gradients.
    """

    def __init__(
            self,
            config: GIAConfig,
            model_template: nn.Module,
            dataset_name: str,
            input_shape: Tuple[int, ...],
            num_classes: int,
            device: str = 'cpu',
    ):
        super().__init__(config, device)
        self.model_template = model_template
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        if self.dataset_name not in DATASET_STATS:
            raise ValueError(f"Stats for '{self.dataset_name}' not found in DATASET_STATS.")

    def train(self, *args, **kwargs):
        """GIA does not require a pre-trained attack model."""
        pass

    def _reconstruct(
            self,
            target_gradient: List[torch.Tensor],
            lr: float,
            ground_truth_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The core optimization loop for gradient inversion."""
        model = copy.deepcopy(self.model_template).eval().to(self.device)
        target_gradient = [g.to(self.device) for g in target_gradient]
        stats = DATASET_STATS[self.dataset_name]
        normalize = transforms.Normalize(mean=stats['mean'], std=stats['std'])

        # Initialize dummy data
        if self.config.init_type == 'gaussian':
            dummy_data = torch.randn(self.config.num_images, *self.input_shape, device=self.device, requires_grad=True)
        else:  # 'random'
            dummy_data = torch.rand(self.config.num_images, *self.input_shape, device=self.device, requires_grad=True)

        # Initialize dummy labels and optimizer
        if self.config.label_type == 'optimize':
            dummy_labels_param = torch.randn(self.config.num_images, self.num_classes, device=self.device,
                                             requires_grad=True)
            params_to_optimize = [dummy_data, dummy_labels_param]
        else:  # 'ground_truth'
            if ground_truth_labels is None:
                raise ValueError("Ground truth labels required for label_type='ground_truth'")
            fixed_labels = ground_truth_labels.to(self.device)
            params_to_optimize = [dummy_data]

        optimizer = optim.Adam(params_to_optimize, lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_loss, best_data, best_labels = float('inf'), None, None

        for it in range(self.config.iterations):
            optimizer.zero_grad()
            model.zero_grad()

            normalized_data = normalize(dummy_data)

            if self.config.label_type == 'optimize':
                current_labels = torch.argmax(F.softmax(dummy_labels_param, dim=-1), dim=-1)
            else:
                current_labels = fixed_labels

            output = model(normalized_data)
            task_loss = criterion(output, current_labels)
            dummy_gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=False)

            # Gradient matching loss
            if self.config.loss_type == 'cosine':
                grad_loss = 1 - F.cosine_similarity(torch.cat([g.view(-1) for g in dummy_gradient]),
                                                    torch.cat([g.view(-1) for g in target_gradient]),
                                                    dim=0, eps=1e-8)
            else:  # 'l2'
                grad_loss = sum(torch.sum((dg - tg) ** 2) for dg, tg in zip(dummy_gradient, target_gradient))

            # Total Variation regularization
            tv_loss = 0
            if self.config.regularization_weight > 0 and dummy_data.dim() == 4:
                diff1 = dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:]
                diff2 = dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]
                tv_loss = torch.norm(diff1, p=2) + torch.norm(diff2, p=2)

            total_loss = grad_loss + self.config.regularization_weight * tv_loss
            total_loss.backward()
            optimizer.step()
            dummy_data.data.clamp_(0, 1)

            if grad_loss.item() < best_loss:
                best_loss = grad_loss.item()
                best_data = dummy_data.detach().clone()
                best_labels = current_labels.detach().clone()

            if self.config.log_interval and it % self.config.log_interval == 0:
                logging.info(
                    f"  Iter: {it:5d} | Grad Loss: {grad_loss.item():.4f} | TV Loss: {tv_loss.item():.4f if isinstance(tv_loss, torch.Tensor) else tv_loss:.4f}")

        return best_data, best_labels

    def execute(
            self,
            *,
            data_for_attack: Dict[str, Any],
            ground_truth_data: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        target_gradient = data_for_attack.get('gradient')
        if target_gradient is None:
            raise ValueError("'gradient' not found in data_for_attack dictionary.")

        logging.info(f"Starting Gradient Inversion Attack...")
        start_time = time.time()

        best_result_log, best_psnr = None, -float('inf')

        # Hyperparameter tuning for learning rate
        for lr in self.config.lrs_to_try:
            logging.info(f"--- Trying LR: {lr} ---")

            gt_labels = ground_truth_data.get("labels") if ground_truth_data else None

            reconstructed_images, reconstructed_labels = self._reconstruct(
                target_gradient=target_gradient, lr=lr, ground_truth_labels=gt_labels
            )

            current_log = {"learning_rate": lr}

            # Evaluate if ground truth is provided
            if ground_truth_data and "images" in ground_truth_data and "labels" in ground_truth_data:
                metrics = evaluate_reconstruction_metrics(
                    reconstructed_images, ground_truth_data["images"],
                    reconstructed_labels, ground_truth_data["labels"]
                )
                current_log["metrics"] = metrics
                current_psnr = metrics.get("psnr", -1)

                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_result_log = current_log

                    if self.config.save_visuals:
                        save_dir = Path(self.config.save_dir)
                        victim_id = kwargs.get("victim_id", "unknown")
                        round_num = kwargs.get("round", 0)
                        filename = f"round_{round_num}_victim_{victim_id}_lr_{lr}.png"
                        visualize_reconstruction(
                            reconstructed_images, ground_truth_data["images"],
                            reconstructed_labels, ground_truth_data["labels"],
                            metrics, save_dir / filename
                        )
            else:
                # If no ground truth, just save the first result
                if best_result_log is None:
                    best_result_log = current_log

        total_duration = time.time() - start_time
        if best_result_log:
            best_result_log["duration_sec"] = total_duration
            best_lr = best_result_log.get("learning_rate", "N/A")
            logging.info(f"GIA Complete. Best PSNR: {best_psnr:.2f} (with LR: {best_lr})")
            return best_result_log

        return {"status": "failed", "duration_sec": total_duration}


# --- 3. Example Usage ---
if __name__ == '__main__':
    # This example demonstrates how to use the refactored attacker class.
    # It simulates a victim's data and gradient to attack.

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, num_classes)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Simulate victim's data and gradient
    victim_model = SimpleCNN().to(DEVICE)
    victim_images = torch.rand(1, 3, 32, 32, device=DEVICE)
    victim_labels = torch.randint(0, 10, (1,), device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    output = victim_model(victim_images)
    loss = criterion(output, victim_labels)
    victim_gradient = torch.autograd.grad(loss, victim_model.parameters())
    victim_gradient = [g.detach() for g in victim_gradient]

    # 2. Initialize and execute the attacker
    print("\n--- Initializing and executing the GIA attacker ---")
    gia_config = GIAConfig(iterations=100, log_interval=50)  # Fewer iterations for quick demo
    attacker = GradientInversionAttacker(
        config=gia_config,
        model_template=SimpleCNN(),
        dataset_name='CIFAR10',
        input_shape=(3, 32, 32),
        num_classes=10,
        device=DEVICE
    )

    results = attacker.execute(
        data_for_attack={'gradient': victim_gradient},
        ground_truth_data={'images': victim_images, 'labels': victim_labels},
        victim_id="demo_victim",
        round=1
    )

    print("\n--- ATTACK RESULTS ---")
    if "metrics" in results:
        print(f"Best Tuned LR: {results.get('learning_rate')}")
        for key, val in results["metrics"].items():
            print(f"{key.upper()}: {val:.4f}")
