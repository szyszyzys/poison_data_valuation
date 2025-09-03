import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from marketplace.seller.gradient_seller import estimate_byte_size
from marketplace.seller.seller import BaseSeller
from model.utils import local_training_and_get_gradient

## 2. The Refactored GradientSeller Class
# This version is clean, decoupled, and robust.

class GradientSeller(BaseSeller):
    """
    A seller that participates in federated learning by providing gradient updates.

    This version is fully decoupled from model creation via a factory pattern
    and uses configuration objects for clarity and ease of use.
    """

    def __init__(self,
                 seller_id: str,
                 data_config: DataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        super().__init__(
            seller_id=seller_id,
            dataset=data_config.dataset,
            save_path=save_path,
            device=device,
            **kwargs  # Pass any remaining BaseSeller args like pricing
        )
        self.data_config = data_config
        self.training_config = training_config
        self.model_factory = model_factory

        # --- State Attributes ---
        # Cleanly manage the state from the last computation
        self.selected_last_round = False
        self.last_computed_gradient: Optional[List[torch.Tensor]] = None
        self.last_training_stats: Optional[Dict[str, Any]] = None

    def get_gradient_for_upload(self, global_model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes and returns the gradient update and training statistics.

        This is the primary method for the federated learning coordinator to call.
        """
        try:
            # 1. Create a fresh model instance using the injected factory.
            #    This ensures no state leaks from previous rounds.
            local_model = self.model_factory()
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(self.device)
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to prepare model: {e}", exc_info=True)
            return None, {'error': 'Model preparation failed.'}

        # 2. Delegate the actual training and gradient calculation.
        gradient, stats = self._compute_local_grad(local_model)

        # 3. Update the seller's internal state for logging or inspection.
        self.last_computed_gradient = gradient
        self.last_training_stats = stats

        return gradient, stats

    def _compute_local_grad(self, model_to_train: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Handles the local training loop and gradient computation."""
        start_time = time.time()

        if not self.data_config.dataset or len(self.data_config.dataset) == 0:
            logging.warning(f"[{self.seller_id}] Dataset is empty. Returning zero gradient.")
            zero_grad = [torch.zeros_like(p, device=self.device) for p in model_to_train.parameters()]
            return zero_grad, {'train_loss': None, 'compute_time_ms': 0}

        # Create DataLoader using the provided data configuration
        data_loader = DataLoader(
            self.data_config.dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=self.data_config.collate_fn
        )

        try:
            # The core training logic is in an external, reusable function
            grad_tensors, _, _, _, avg_loss = local_training_and_get_gradient(
                model=model_to_train,
                train_loader=data_loader,
                device=self.device,
                local_epochs=self.training_config.epochs,
                lr=self.training_config.learning_rate,
            )

            stats = {
                'train_loss': avg_loss,
                'compute_time_ms': (time.time() - start_time) * 1000,
                'upload_bytes': estimate_byte_size(grad_tensors) if grad_tensors else 0
            }
            return grad_tensors, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] Error in training loop: {e}", exc_info=True)
            return None, {'error': 'Training failed.'}

    def save_local_model(self, model_instance: nn.Module) -> None:
        """Saves the state dictionary of a given model instance."""
        save_file_path = Path(self.save_path) / f"local_model_{self.seller_id}.pt"
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model_instance.state_dict(), save_file_path)
            logging.info(f"[{self.seller_id}] Saved model to {save_file_path}")
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to save model: {e}")

    def load_local_model(self) -> nn.Module:
        """
        Loads a model from disk. Uses the factory to get the correct
        architecture before loading the state.
        """
        load_file_path = Path(self.save_path) / f"local_model_{self.seller_id}.pt"

        # Always create a new model instance to ensure correct architecture
        model = self.model_factory()
        model.to(self.device)

        if not load_file_path.exists():
            logging.warning(f"[{self.seller_id}] No saved model found at {load_file_path}. Returning a new instance.")
            return model

        try:
            model.load_state_dict(torch.load(load_file_path, map_location=self.device))
            logging.info(f"[{self.seller_id}] Loaded model from {load_file_path}")
        except Exception as e:
            logging.error(f"[{self.seller_id}] Could not load model: {e}. Returning a new instance.")

        return model

    def round_end_process(self, round_number: int, is_selected: bool) -> None:
        """Cleans up the state at the end of a federated round."""
        self.selected_last_round = is_selected
        # You could add logging logic here using self.last_training_stats if needed
        # For example:
        if is_selected:
            logging.info(f"[{self.seller_id}] Round {round_number} complete. Stats: {self.last_training_stats}")

        # Reset state for the next round
        self.last_computed_gradient = None
        self.last_training_stats = None