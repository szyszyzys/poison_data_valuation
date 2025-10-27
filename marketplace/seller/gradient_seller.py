import collections
import copy
import csv
import json
import logging
import os
import random
import sys
import time
# Add these class definitions as well
from abc import ABC, abstractmethod
from collections import abc  # abc.Mapping for general dicts
from dataclasses import field, dataclass
from pathlib import Path
from typing import Any, Callable, Set
from typing import Dict
from typing import List, Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from attack.attack_gradient_market.poison_attack.attack_utils import PoisonGenerator, BackdoorImageGenerator, \
    BackdoorTextGenerator, BackdoorTabularGenerator
from common.enums import ImageTriggerType, ImageTriggerLocation, PoisonType
from common.gradient_market_configs import AdversarySellerConfig, BackdoorImageConfig, BackdoorTextConfig, SybilConfig, \
    RuntimeDataConfig, TrainingConfig, BackdoorTabularConfig
from common.utils import unflatten_tensor, flatten_tensor
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.seller import BaseSeller
from model.utils import local_training_and_get_gradient


class BaseGradientStrategy(ABC):
    """Abstract base class for all gradient manipulation strategies."""

    @abstractmethod
    def manipulate(self, current_grad: torch.Tensor, avg_grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MimicStrategy(BaseGradientStrategy):
    """Mimics the average gradient by blending it with the current one."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def manipulate(self, current_grad: torch.Tensor, avg_grad: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * current_grad + self.alpha * avg_grad


class PivotStrategy(BaseGradientStrategy):
    """Completely replaces the current gradient with the average one."""

    def manipulate(self, current_grad: torch.Tensor, avg_grad: torch.Tensor) -> torch.Tensor:
        return avg_grad.clone()


def diagnose_parameter_issue(model):
    """Deep diagnostic of parameter issues."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"\n❌ Problem with parameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Dtype: {param.dtype}")
            print(f"  Device: {param.device}")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Is leaf: {param.is_leaf}")
            print(f"  NaN count: {torch.isnan(param).sum().item()}")
            print(f"  Inf count: {torch.isinf(param).sum().item()}")
            print(f"  First few values: {param.flatten()[:10]}")

            # Try to see what's in the actual layer
            layer_idx = name.split('.')[1] if '.' in name else None
            if layer_idx:
                layer = dict(model.named_modules()).get(f"features.{layer_idx}")
                print(f"  Layer type: {type(layer)}")
                if isinstance(layer, nn.BatchNorm2d):
                    print(f"  BatchNorm running_mean has NaN: {torch.isnan(layer.running_mean).any()}")
                    print(f"  BatchNorm running_var has NaN: {torch.isnan(layer.running_var).any()}")


class KnockOutStrategy(BaseGradientStrategy):
    """A more aggressive version of the Mimic strategy."""

    def __init__(self, alpha: float = 0.5):
        # Uses a higher amplification factor for the blend
        self.alpha_knock = min(alpha * 2, 1.0)

    def manipulate(self, current_grad: torch.Tensor, avg_grad: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha_knock) * current_grad + self.alpha_knock * avg_grad


def estimate_byte_size(data: Any) -> int:
    """
    Recursively estimates the size in bytes of potentially complex data structures
    containing primarily numpy arrays or torch tensors. Handles basic types poorly.

    Args:
        data: The data structure (dict, list, tensor, array, etc.).

    Returns:
        Estimated size in bytes. Returns 0 for None or empty structures.
    """
    total_size = 0

    if data is None:
        return 0

    # --- Handle Iterables (Lists, Tuples, Sets) ---
    if isinstance(data, (list, tuple, set)):
        if not data: return 0  # Empty iterable
        for item in data:
            total_size += estimate_byte_size(item)  # Recurse

    # --- Handle Mappings (Dictionaries, OrderedDict) ---
    elif isinstance(data, abc.Mapping):  # Handles dict, OrderedDict, etc.
        if not data: return 0  # Empty mapping
        # Iterate through values as keys are usually small strings
        for value in data.values():
            total_size += estimate_byte_size(value)  # Recurse on values

    # --- Handle Numpy Arrays ---
    elif isinstance(data, np.ndarray):
        total_size = data.nbytes

    # --- Handle Torch Tensors ---
    elif isinstance(data, torch.Tensor):
        # element_size() gives bytes per element (e.g., float32=4)
        total_size = data.element_size() * data.numel()

    # --- Handle Basic Python Numeric/Bool Types (Approximate) ---
    # Often negligible compared to large arrays/tensors, but can include
    elif isinstance(data, (int, float, bool, np.integer, np.floating, np.bool_)):
        # np.dtype(...).itemsize is more accurate than sys.getsizeof for raw size
        try:
            total_size = np.dtype(type(data)).itemsize
        except TypeError:
            # Fallback for standard python types if numpy dtype fails
            total_size = sys.getsizeof(data)  # Includes Python object overhead
            # Or just use fixed estimates: 8 for int/float, 1 for bool?
            # total_size = 8 if isinstance(data, (int, float)) else 1

    # --- Handle Other Types ---
    # Add elif blocks here for other data types you expect (e.g., strings)
    # elif isinstance(data, str):
    #    total_size = sys.getsizeof(data) # Includes overhead

    # --- Unhandled Type ---
    else:
        # You might want to log a warning for unexpected types
        # logging.warning(f"estimate_byte_size encountered unhandled type: {type(data)}. Size may be inaccurate.")
        # Attempt sys.getsizeof as a fallback, but it includes Python overhead
        try:
            total_size = sys.getsizeof(data)
        except TypeError:
            total_size = 0  # Cannot determine size

    return total_size


def validate_and_fix_model_initialization(model: nn.Module) -> bool:
    """
    Check for and fix NaN/Inf values in model parameters.
    Returns True if model is valid, otherwise raises a RuntimeError.
    """
    problematic_params = []

    # First pass: detect issues
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logging.error(f"❌ NaN/Inf detected in parameter '{name}' during initialization!")
            problematic_params.append(name)

    if not problematic_params:
        return True

    logging.warning(
        f"🔄 NaN/Inf detected. Attempting stable reinitialization for {len(problematic_params)} parameters..."
    )

    # -------------------- START OF FIX --------------------
    #
    # We will iterate over model.modules() directly instead of using model.apply()
    # This is more explicit and fixes the bug.
    #
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # USE UNIFORM: This is bounded and cannot create Inf
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # --- THIS IS THE CRITICAL BUG FIX ---
        # Your old code had nn.BatchNorm2d, but the model uses nn.BatchNorm1d
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            if hasattr(m, 'running_mean'):
                m.running_mean.zero_()
            if hasattr(m, 'running_var'):
                m.running_var.fill_(1)

        elif isinstance(m, nn.Conv2d):
            # Also handle Conv2d just in case
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # -------------------- END OF FIX --------------------

    # Verify fix worked
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            error_msg = f"❌ CRITICAL: Still NaN/Inf in '{name}' after stable reinitialization. This is unrecoverable."
            logging.error(error_msg)
            # This is what's raising your error
            raise RuntimeError(error_msg)

    logging.info("✅ Successfully fixed all NaN/Inf parameters with stable initialization.")
    return True


class GradientSeller(BaseSeller):
    """
    A seller that participates in federated learning by providing gradient updates.

    This version is decoupled from model creation via a factory pattern and uses
    configuration objects for clarity, making it a robust base class.
    """

    def __init__(self,
                 seller_id: str,
                 data_config: RuntimeDataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        kwargs.pop("model_type", None)
        kwargs.pop("vocab", None)
        kwargs.pop("pad_idx", None)

        # Now it's safe to pass the remaining kwargs (like pricing) to the parent
        super().__init__(
            seller_id=seller_id,
            dataset=data_config.dataset,
            save_path=save_path,
            device=device,
            **kwargs
        )
        self.data_config = data_config
        self.training_config = training_config
        self.model_factory = model_factory
        # --- State Attributes ---
        self.last_computed_gradient: Optional[List[torch.Tensor]] = None
        self.last_training_stats: Optional[Dict[str, Any]] = None
        self.selected_last_round = False
        self.save_path = Path(save_path)
        self.seller_specific_path = self.save_path / self.seller_id
        self.seller_specific_path.mkdir(parents=True, exist_ok=True)  # Ensure seller's work dir exists
        self.selection_history = []  # Track selection per round
        self.performance_history = []  # Track contribution to global model
        self.reward_history = []  # Track hypothetical rewards

        self.train_loader = DataLoader(
            dataset=self.data_config.dataset,
            batch_size=self.training_config.batch_size,
            collate_fn=self.data_config.collate_fn,  # Handles text/image
            shuffle=True,
            num_workers=2,  # Optional: for efficiency
            pin_memory=True if self.device == "cuda" else False  # Optional
        )

    def get_gradient_for_upload(self,
                                all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
                                target_seller_id: str = None) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes and returns the gradient update and training statistics.
        This is the primary method for the federated learning coordinator to call.
        """
        try:
            # Create a fresh model instance with current global weights
            # The stateful factory automatically loads global_model.state_dict()
            local_model = self.model_factory().to(self.device)
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to prepare model: {e}", exc_info=True)
            return None, {'error': 'Model preparation failed.'}

        # Delegate the actual training and gradient calculation
        gradient, stats = self._compute_local_grad(local_model, self.data_config.dataset)

        # Update the seller's internal state for logging or inspection
        self.last_computed_gradient = gradient
        self.last_training_stats = stats

        return gradient, stats

    def save_latest_round(self):
        """Save only the most recent round (incremental save)."""
        if not self.federated_round_history:
            return

        history_csv = self.seller_specific_path / "history" / "round_history.csv"
        history_csv.parent.mkdir(parents=True, exist_ok=True)

        latest_record = self.federated_round_history[-1]

        # Flatten record for CSV
        csv_row = {
            'event_type': latest_record.get('event_type'),
            'round': latest_record.get('round'),
            'timestamp': latest_record.get('timestamp'),
            'was_selected': latest_record.get('was_selected')
        }

        if latest_record.get('training_stats'):
            for key, value in latest_record['training_stats'].items():
                csv_row[f'training_stats_{key}'] = value

        df = pd.DataFrame([csv_row])

        # Append or create
        if history_csv.exists():
            df.to_csv(history_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(history_csv, mode='w', header=True, index=False)

    def _compute_local_grad(
            self,
            model_to_train: nn.Module,
            dataset_to_use: Optional[Dataset] = None
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Handles the local training loop and gradient computation.

        Returns:
            Tuple of (gradient_list, stats_dict)
            - gradient_list: List of tensors matching model parameters, or None on failure
            - stats_dict: Dictionary containing training statistics
        """
        logging.info(f"\n[{self.seller_id}] Starting local training")
        logging.info(f"  Dataset size: {len(self.dataset)}")
        logging.info(f"  Batch size: {self.training_config.batch_size}")
        logging.info(f"  Local epochs: {self.training_config.local_epochs}")

        start_time = time.time()

        # Use provided dataset or fall back to self.dataset
        if dataset_to_use is None:
            if not hasattr(self, 'dataset') or self.dataset is None:
                logging.error(f"[{self.seller_id}] ❌ No dataset available!")
                return None, {'error': 'No dataset available'}
            dataset_to_use = self.dataset

        # Validate dataset
        if not dataset_to_use or len(dataset_to_use) == 0:
            logging.warning(f"[{self.seller_id}] ⚠️  Dataset is empty. Returning zero gradient.")
            zero_grad = [torch.zeros_like(p) for p in model_to_train.parameters()]
            return zero_grad, {
                'train_loss': 0.0,
                'compute_time_ms': 0,
                'upload_bytes': estimate_byte_size(zero_grad),
                'num_samples': 0
            }

        # if not validate_and_fix_model_initialization(model_to_train):
        #     logging.error(f"[{self.seller_id}] ❌ Model has unfixable NaN/Inf values!")
        #     return None, {'error': 'Model initialization contains NaN/Inf'}

        logging.info(f"[{self.seller_id}] Training on {len(dataset_to_use)} samples...")

        # Create DataLoader with proper collate function handling
        collate_fn = getattr(self.data_config, 'collate_fn', None) if hasattr(self, 'data_config') else None

        try:
            data_loader = DataLoader(
                dataset_to_use,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,  # Important for multiprocessing compatibility
                pin_memory=False,  # Avoid issues with CUDA in multiprocessing
                drop_last=False
            )

            # Validate DataLoader
            if len(data_loader) == 0:
                logging.warning(f"[{self.seller_id}] ⚠️  DataLoader is empty after creation!")
                zero_grad = [torch.zeros_like(p) for p in model_to_train.parameters()]
                return zero_grad, {
                    'train_loss': 0.0,
                    'compute_time_ms': 0,
                    'upload_bytes': estimate_byte_size(zero_grad),
                    'num_samples': 0
                }

        except Exception as e:
            logging.error(f"[{self.seller_id}] ❌ Failed to create DataLoader: {e}", exc_info=True)
            return None, {'error': f'DataLoader creation failed: {str(e)}'}

        # Perform local training
        try:
            # --- START: Simplified Parameter Usage ---
            # Directly use parameters from self.training_config

            local_epochs_to_use = self.training_config.local_epochs
            batch_size_to_use = self.training_config.batch_size  # Used above
            lr_to_use = self.training_config.learning_rate
            optimizer_to_use = self.training_config.optimizer  # Ensure correct attribute name
            momentum_to_use = self.training_config.momentum
            weight_decay_to_use = self.training_config.weight_decay

            logging.info(f"[{self.seller_id}] Starting local training...")
            logging.info(f"  - Epochs: {local_epochs_to_use}")
            logging.info(f"  - Batch size: {batch_size_to_use}")
            logging.info(f"  - Learning rate: {lr_to_use} (Source: TrainingConfig)")
            logging.info(f"  - Optimizer: {optimizer_to_use} (Source: TrainingConfig)")
            logging.info(f"  - Momentum: {momentum_to_use}")
            logging.info(f"  - Weight Decay: {weight_decay_to_use}")
            logging.info(f"  - EPS: {self.training_config.eps}")
            logging.info(f"  - Device: {self.device}")

            grad_tensors, avg_loss = local_training_and_get_gradient(
                model=model_to_train,
                train_loader=data_loader,
                device=self.device,
                local_epochs=local_epochs_to_use,
                lr=lr_to_use,
                opt_str=optimizer_to_use,
                eps=self.training_config.eps,
                momentum=momentum_to_use,
                weight_decay=weight_decay_to_use
            )
            # --- END: Simplified Parameter Usage ---
            compute_time = (time.time() - start_time) * 1000
            # === CRITICAL: Validate returned gradient ===
            if grad_tensors is None:
                logging.error(f"[{self.seller_id}] ❌ Training function returned None gradient!")
                return None, {'error': 'Training returned None gradient'}

            if not isinstance(grad_tensors, (list, tuple)):
                logging.error(f"[{self.seller_id}] ❌ Gradient is not a list/tuple: {type(grad_tensors)}")
                return None, {'error': f'Invalid gradient type: {type(grad_tensors)}'}

            if len(grad_tensors) == 0:
                logging.error(f"[{self.seller_id}] ❌ Gradient list is empty!")
                return None, {'error': 'Empty gradient list'}

            # Validate gradient matches model
            model_params = list(model_to_train.parameters())
            if len(grad_tensors) != len(model_params):
                logging.error(
                    f"[{self.seller_id}] ❌ Gradient length mismatch: "
                    f"got {len(grad_tensors)}, expected {len(model_params)}"
                )
                return None, {'error': f'Gradient length mismatch: {len(grad_tensors)} vs {len(model_params)}'}

            # Validate each tensor
            for i, (grad_tensor, model_param) in enumerate(zip(grad_tensors, model_params)):
                if not isinstance(grad_tensor, torch.Tensor):
                    logging.error(f"[{self.seller_id}] ❌ Gradient[{i}] is not a tensor: {type(grad_tensor)}")
                    return None, {'error': f'Gradient[{i}] is not a tensor'}

                if grad_tensor.shape != model_param.shape:
                    logging.error(
                        f"[{self.seller_id}] ❌ Shape mismatch at param {i}: "
                        f"gradient {grad_tensor.shape} vs model {model_param.shape}"
                    )
                    return None, {'error': f'Shape mismatch at param {i}'}

                # Check for NaN/Inf
                if torch.isnan(grad_tensor).any():
                    logging.error(f"[{self.seller_id}] ❌ NaN detected in gradient[{i}]!")
                    return None, {'error': f'NaN in gradient[{i}]'}

                if torch.isinf(grad_tensor).any():
                    logging.error(f"[{self.seller_id}] ❌ Inf detected in gradient[{i}]!")
                    return None, {'error': f'Inf in gradient[{i}]'}

            # All validation passed!
            logging.info(f"[{self.seller_id}] ✅ Training completed successfully")
            logging.info(f"  - Average loss: {avg_loss:.4f}")
            logging.info(f"  - Compute time: {compute_time:.2f}ms")
            logging.info(f"  - Gradient params: {len(grad_tensors)}")

            # Compute gradient statistics
            grad_norm = sum(torch.norm(g).item() ** 2 for g in grad_tensors) ** 0.5

            stats = {
                'train_loss': avg_loss,
                'compute_time_ms': compute_time,
                'upload_bytes': estimate_byte_size(grad_tensors),
                'num_samples': len(dataset_to_use),
                'gradient_norm': grad_norm
            }

            return grad_tensors, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] ❌ Error in training loop: {e}", exc_info=True)

            # Return zero gradients instead of None to prevent cascading failures
            logging.warning(f"[{self.seller_id}] Returning zero gradients due to training failure")
            zero_grad = [torch.zeros_like(p) for p in model_to_train.parameters()]

            return zero_grad, {  # This is a robust return value
                'error': str(e),
                'train_loss': float('nan'),
                'compute_time_ms': (time.time() - start_time) * 1000,
                'upload_bytes': estimate_byte_size(zero_grad)
            }

    def save_local_model(self, model_instance: nn.Module) -> None:
        """Saves the state dictionary of a given model instance."""
        save_file_path = self.save_path / f"local_model_{self.seller_id}.pt"
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
        load_file_path = self.save_path / f"local_model_{self.seller_id}.pt"
        model = self.model_factory().to(self.device)

        if not load_file_path.exists():
            logging.warning(f"[{self.seller_id}] No saved model found. Returning new instance.")
            return model

        try:
            model.load_state_dict(torch.load(load_file_path, map_location=self.device))
            logging.info(f"[{self.seller_id}] Loaded model from {load_file_path}")
        except Exception as e:
            logging.error(f"[{self.seller_id}] Could not load model: {e}. Returning new instance.")

        return model

    def save_round_history_csv(self, subdirectory: str = "history") -> None:
        """Saves the federated round history for this seller to a CSV file in a seller-specific subdirectory."""
        history_dir = self.seller_specific_path / subdirectory
        os.makedirs(history_dir, exist_ok=True)
        file_name = history_dir / f"round_history.csv"  # Fixed name within seller's history dir

        if not self.federated_round_history:
            logging.info(f"[{self.seller_id}] No round history data to save for CSV.")
            return

        # Determine fieldnames for the CSV header
        first_record = self.federated_round_history[0]
        fieldnames = ['event_type', 'round', 'timestamp', 'was_selected']

        if first_record.get('training_stats'):
            for key in first_record['training_stats'].keys():
                fieldnames.append(f'training_stats_{key}')

        try:
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.federated_round_history:
                    csv_row = {
                        'event_type': record.get('event_type'),
                        'round': record.get('round'),
                        'timestamp': record.get('timestamp'),
                        'was_selected': record.get('was_selected')
                    }
                    if record.get('training_stats'):
                        for key, value in record['training_stats'].items():
                            csv_row[f'training_stats_{key}'] = value
                    writer.writerow(csv_row)
            logging.info(f"[{self.seller_id}] Round history saved successfully to {file_name}")
        except IOError as e:
            logging.error(f"[{self.seller_id}] Error saving round history to {file_name}: {e}")
        except Exception as e:
            logging.error(f"[{self.seller_id}] An unexpected error occurred while saving round history: {e}")

    def round_end_process(
            self,
            round_number: int,
            was_selected: bool,
            was_outlier: bool = False,
            marketplace_metrics: Dict = None
    ):
        """Enhanced round end processing with marketplace tracking."""

        # Record selection
        self.selection_history.append({
            'round': round_number,
            'selected': was_selected,
            'outlier': was_outlier,
            'timestamp': time.time()
        })

        # Calculate hypothetical reward based on contribution
        if marketplace_metrics:
            reward = self._calculate_reward(marketplace_metrics, was_selected)
            self.reward_history.append({
                'round': round_number,
                'reward': reward,
                'cumulative_reward': sum(r['reward'] for r in self.reward_history) + reward
            })

        # Save incrementally
        self.save_latest_round()

    def _calculate_reward(self, metrics: Dict, was_selected: bool) -> float:
        """
        Calculate seller reward based on contribution.
        This is a placeholder - implement your actual reward mechanism.
        """
        if not was_selected:
            return 0.0

        # Example: Reward based on gradient quality and model improvement
        base_reward = 1.0

        # Bonus for good gradient quality
        norm = metrics.get('gradient_norm', 0)
        if norm > 0:
            # Normalize by average (if available)
            quality_bonus = 0.5  # Placeholder
        else:
            quality_bonus = 0

        return base_reward + quality_bonus

    def save_marketplace_summary(self):
        """Save seller's marketplace participation summary."""
        summary = {
            'seller_id': self.seller_id,
            'total_rounds': len(self.selection_history),
            'times_selected': sum(1 for h in self.selection_history if h['selected']),
            'times_outlier': sum(1 for h in self.selection_history if h['outlier']),
            'selection_rate': sum(1 for h in self.selection_history if h['selected']) / len(
                self.selection_history) if self.selection_history else 0,
            'total_reward': sum(r['reward'] for r in self.reward_history) if self.reward_history else 0,
            'avg_reward_per_round': np.mean([r['reward'] for r in self.reward_history]) if self.reward_history else 0
        }

        summary_path = self.seller_specific_path / "marketplace_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


@dataclass
class ClientState:
    seller_obj: GradientSeller  # The actual seller instance
    selection_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=20))  # Example maxlen
    selection_rate: float = 0.0
    rounds_participated: int = 0
    phase: str = "benign"  # "benign" or "attack"
    role: Optional[str] = "hybrid"  # "attacker", "explorer", or "hybrid"


class SybilCoordinator:
    def __init__(self, sybil_cfg: SybilConfig, aggregator: Aggregator):
        self.sybil_cfg = sybil_cfg
        self.aggregator = aggregator
        self.device = aggregator.device

        self.clients: Dict[str, ClientState] = collections.OrderedDict()

        # Initialize strategies from config
        self.strategies: Dict[str, BaseGradientStrategy] = {
            'mimic': MimicStrategy(**self.sybil_cfg.strategy_configs.get('mimic', {})),
            'pivot': PivotStrategy(),
            'knock_out': KnockOutStrategy(**self.sybil_cfg.strategy_configs.get('knock_out', {})),
        }

        # State Tracking
        self.cur_round = 0
        self.start_atk = False
        self.selected_gradients: Dict[str, torch.Tensor] = {}
        self.selected_history: collections.deque = collections.deque(
            maxlen=self.sybil_cfg.history_window_size
        )
        self.selection_patterns = {}

        self.known_strategies: Set[str] = set(self.strategies.keys()) | {"oracle_blend", "systematic_probe"}

    def register_seller(self, seller: GradientSeller) -> None:
        """Register a malicious seller with the coordinator."""
        if not hasattr(seller, 'seller_id'):
            raise AttributeError("Seller object must have a 'seller_id' attribute")
        self.clients[seller.seller_id] = ClientState(seller_obj=seller)

    def apply_manipulation(self,
                           current_round_gradients: Dict[str, List[torch.Tensor]],
                           all_sellers: Dict[str, GradientSeller],
                           current_root_gradient: Optional[List[torch.Tensor]] = None,
                           global_epoch=-1, buyer_data_loader=None) -> Dict[str, List[torch.Tensor]]:
        if not self.start_atk:
            logging.debug("[SybilCoordinator] Not in attack phase - no manipulation.")
            return current_round_gradients

        manipulated_gradients = copy.deepcopy(current_round_gradients)
        sybil_ids = set(self.clients.keys())
        benign_ids = set(current_round_gradients.keys()) - sybil_ids

        # --- 1. Determine which strategies are needed ---
        strategies_in_use = set()
        active_sybils = []
        for sybil_id in sybil_ids:
            if sybil_id in manipulated_gradients and self.clients[sybil_id].phase == "attack":
                strategy_name = self._get_strategy_for_client(self.clients[sybil_id])
                strategies_in_use.add(strategy_name)
                active_sybils.append(sybil_id)

        if not active_sybils:
            logging.debug("[SybilCoordinator] No active Sybils needing manipulation.")
            return current_round_gradients

        logging.info(
            f"[SybilCoordinator] Applying manipulations for {len(active_sybils)} active Sybils. Strategies: {strategies_in_use}")

        # --- 2. Calculate Oracle Information (if needed) ---
        oracle_centroid_flat: Optional[torch.Tensor] = None
        if "oracle_blend" in strategies_in_use:
            benign_gradients_dict = {
                sid: manipulated_gradients[sid] for sid in benign_ids if sid in manipulated_gradients
            }
            if benign_gradients_dict and current_root_gradient is not None:  # Ensure root gradient is available
                logging.debug("[SybilCoordinator] Running hypothetical aggregation for Oracle...")
                try:
                    _hypo_agg_grad, hypothetical_selected_ids, _hypo_outliers, _hypo_stats = self.aggregator.aggregate(
                        global_epoch=global_epoch,
                        seller_updates=benign_gradients_dict,
                        root_gradient=current_root_gradient,
                        buyer_data_loader = buyer_data_loader
                    )

                    if hypothetical_selected_ids:
                        selected_benign_grads_flat = [
                            self._ensure_tensor(benign_gradients_dict[sid])
                            for sid in hypothetical_selected_ids if sid in benign_gradients_dict
                        ]
                        if selected_benign_grads_flat:
                            oracle_centroid_flat = torch.mean(torch.stack(selected_benign_grads_flat), dim=0)
                            logging.info(
                                f"[SybilCoordinator] Oracle centroid calculated from {len(selected_benign_grads_flat)} hypothetically selected benign gradients.")
                        else:
                            logging.warning(
                                "[SybilCoordinator] Oracle calc failed: No valid benign grads after hypothetical selection.")
                    else:
                        logging.warning(
                            "[SybilCoordinator] Oracle calc failed: Aggregator hypothetically selected no benign clients.")
                except Exception as e:
                    logging.error(
                        f"[SybilCoordinator] Error during hypothetical aggregation for Oracle: {e}. Oracle attack will fail.",
                        exc_info=True)
            elif not benign_gradients_dict:
                logging.warning("[SybilCoordinator] Oracle calc skipped: No benign gradients.")
            elif current_root_gradient is None:
                logging.warning("[SybilCoordinator] Oracle calc skipped: Root gradient not provided.")

        historical_centroid_flat: Optional[torch.Tensor] = None
        needs_historical = any(
            s in self.strategies for s in strategies_in_use)  # Check if any known historical strategy is needed
        if needs_historical:
            if self.selection_patterns and "centroid" in self.selection_patterns:
                historical_centroid_flat = self.selection_patterns["centroid"]
                logging.debug("[SybilCoordinator] Using historical centroid for mimic/pivot/knock_out.")
            else:
                logging.warning(
                    "[SybilCoordinator] Historical strategies requested but no historical centroid available yet.")

        # --- 4. Loop through ACTIVE Sybils and Manipulate ---
        manipulated_count = 0
        for sybil_id in active_sybils:
            client_state = self.clients[sybil_id]
            strategy_name = self._get_strategy_for_client(client_state)

            original_malicious_grad_list = current_round_gradients[sybil_id]  # Use original input
            original_shapes = [g.shape for g in original_malicious_grad_list]
            original_malicious_flat = self._ensure_tensor(original_malicious_grad_list)

            manipulated_grad_flat: Optional[torch.Tensor] = None

            # --- Apply Oracle Blend ---
            if strategy_name == "oracle_blend":
                if oracle_centroid_flat is not None:
                    alpha = self.sybil_cfg.oracle_blend_alpha  # Get alpha from config
                    manipulated_grad_flat = alpha * original_malicious_flat + (1.0 - alpha) * oracle_centroid_flat
                    logging.debug(f"   Oracle Blending applied to {sybil_id} (alpha={alpha})")
                else:
                    logging.warning(
                        f"   Oracle Blend for {sybil_id} failed (no oracle centroid). Submitting original malicious gradient.")
                    manipulated_grad_flat = original_malicious_flat  # Fallback

            # --- Apply Historical Strategies ---
            elif strategy_name in self.strategies:
                if historical_centroid_flat is not None:
                    strategy_obj = self.strategies[strategy_name]
                    try:
                        manipulated_grad_flat = strategy_obj.manipulate(original_malicious_flat,
                                                                        historical_centroid_flat)
                        logging.debug(f"   Historical Strategy '{strategy_name}' applied to {sybil_id}")
                    except Exception as e:
                        logging.error(
                            f"   Error applying historical strategy {strategy_name} to {sybil_id}: {e}. Submitting original.",
                            exc_info=False)
                        manipulated_grad_flat = original_malicious_flat  # Fallback
                else:
                    logging.warning(
                        f"   Historical Strategy '{strategy_name}' for {sybil_id} failed (no historical centroid). Submitting original.")
                    manipulated_grad_flat = original_malicious_flat  # Fallback

            # --- Placeholder for Systematic Probing ---
            elif strategy_name == "systematic_probe":
                # TODO: Implement systematic probing logic.
                # This is complex and stateful. Might involve:
                # 1. Assigning a specific probe target based on sybil_id or internal state.
                # 2. Generating the probe gradient (e.g., perturbing a base gradient).
                # 3. Requires tracking results in update_post_selection.
                logging.warning(
                    f"   Systematic Probing for {sybil_id} not implemented. Submitting original malicious gradient.")
                manipulated_grad_flat = original_malicious_flat  # Placeholder

            # --- Unknown Strategy ---
            else:
                logging.error(
                    f"   Unknown strategy '{strategy_name}' for {sybil_id}. Submitting original malicious gradient.")
                manipulated_grad_flat = original_malicious_flat  # Fallback

            # --- Unflatten and Store ---
            if manipulated_grad_flat is not None:
                manipulated_grad_list = unflatten_tensor(manipulated_grad_flat, original_shapes)
                manipulated_gradients[sybil_id] = manipulated_grad_list
                manipulated_count += 1
                # Update seller's cache (if available)
                sybil_seller = all_sellers.get(sybil_id)
                if sybil_seller: sybil_seller.last_computed_gradient = manipulated_grad_list

        logging.info(
            f"[SybilCoordinator] Successfully manipulated {manipulated_count}/{len(active_sybils)} active Sybil gradients.")
        return manipulated_gradients

    def _get_strategy_for_client(self, client_state: ClientState) -> str:
        """Determines the strategy based on config or role."""
        # 1. Global override from config (if valid)
        config_strategy = self.sybil_cfg.gradient_default_mode
        if config_strategy and config_strategy in self.known_strategies:
            return config_strategy
        # 2. Role-based strategy
        return self._get_strategy_for_role(client_state.role)

    def update_post_selection(self, selected_ids: List[str], all_sellers: Dict):
        """
        Update sybil coordinator state after selection.

        Args:
            selected_ids: List of seller IDs that were selected
            all_sellers: Dictionary of all seller objects
        """
        logging.info(f"[SybilCoordinator] Updating state after selection")

        # Update client states based on selection
        self.update_client_states(selected_ids)

        # Collect HONEST gradients from selected sellers for pattern learning
        selected_honest_gradients = {}
        for sid in selected_ids:
            seller = all_sellers.get(sid)
            if seller and hasattr(seller, 'get_honest_gradient'):
                honest_grad = seller.get_honest_gradient()
                if honest_grad:
                    selected_honest_gradients[sid] = honest_grad

        # Update historical patterns for future rounds
        if selected_honest_gradients:
            self.update_historical_patterns(selected_honest_gradients)
            logging.info(f"   Updated patterns with {len(selected_honest_gradients)} honest gradients")

        # Log sybil performance
        sybil_selected = [sid for sid in selected_ids if sid in self.clients]
        if sybil_selected:
            success_rate = len(sybil_selected) / len(self.clients) if self.clients else 0
            logging.info(f"   🎯 Sybil success: {len(sybil_selected)}/{len(self.clients)} "
                         f"selected ({success_rate:.1%})")
            logging.info(f"      Selected sybils: {sybil_selected}")
        else:
            logging.info(f"   No sybils selected this round")

        # End of round cleanup
        self.on_round_end()

    def _get_strategy_for_role(self, role: str) -> str:
        """Map sybil role to manipulation strategy."""
        strategy_map = {
            "attacker": "mimic",  # Strongly mimic winners
            "hybrid": "mimic",  # Also mimic
            "explorer": "pivot"  # Try different approaches
        }
        return strategy_map.get(role, "mimic")

    def get_client_with_highest_selection_rate(self) -> str:
        """Returns the client ID with the highest selection rate."""
        best_client = None
        max_rate = -1.0
        for cid, client_info in self.clients.items():
            if client_info.selection_rate > max_rate:
                max_rate = client_info.selection_rate
                best_client = cid
        return best_client

    def update_client_states(self, selected_client_ids: List[str]) -> None:
        """Update the state of each client based on the latest selection results."""
        for cid, client_state in self.clients.items():
            was_selected = cid in selected_client_ids
            client_state.selection_history.append(was_selected)
            client_state.selection_rate = sum(client_state.selection_history) / len(client_state.selection_history)
            client_state.rounds_participated += 1

            if (client_state.rounds_participated >= self.sybil_cfg.benign_rounds and
                    client_state.selection_rate > self.sybil_cfg.detection_threshold):
                client_state.phase = "attack"
            else:
                client_state.phase = "benign"

    def update_historical_patterns(self, selected_gradients: Dict[str, torch.Tensor]) -> None:
        """Update the global history of selected gradients and re-analyze patterns."""
        if not selected_gradients:
            return

        self.selected_history.append(selected_gradients)
        self._analyze_selection_patterns()

    def _analyze_selection_patterns(self) -> None:
        """Analyze stored selected gradients to compute a centroid and average cosine similarity."""
        all_selected_flat = [
            grad.flatten() for round_dict in self.selected_history for grad in round_dict.values()
        ]

        if not all_selected_flat:
            self.selection_patterns = {}
            return

        all_tensor = torch.stack(all_selected_flat)
        centroid = torch.mean(all_tensor, dim=0)

        avg_sim = 0.0
        if len(all_tensor) > 1:
            sims = F.cosine_similarity(all_tensor, centroid.unsqueeze(0))
            avg_sim = torch.mean(sims).item()

        self.selection_patterns = {"centroid": centroid, "avg_similarity": avg_sim}

    def adaptive_role_assignment(self) -> None:
        """Dynamically reassign roles based on selection rates."""
        if not self.clients:
            return

        sorted_clients = sorted(self.clients.items(), key=lambda item: item[1].selection_rate, reverse=True)
        num_clients = len(sorted_clients)

        role_config = self.sybil_cfg.role_config
        attacker_cutoff = int(role_config.get('attacker', 0.2) * num_clients)
        explorer_cutoff = int((1 - role_config.get('explorer', 0.4)) * num_clients)

        for i, (cid, client_state) in enumerate(sorted_clients):
            if i < attacker_cutoff:
                client_state.role = "attacker"
            elif i >= explorer_cutoff:
                client_state.role = "explorer"
            else:
                client_state.role = "hybrid"

    def _ensure_tensor(self, gradient: Union[torch.Tensor, List, np.ndarray]) -> torch.Tensor:
        """Ensure that the provided gradient is a single flattened tensor on the correct device."""
        if isinstance(gradient, list):
            flat_tensors = []
            for g in gradient:
                if isinstance(g, torch.Tensor):
                    flat_tensors.append(g.flatten().to(self.device))
                elif isinstance(g, np.ndarray):
                    flat_tensors.append(torch.from_numpy(g).flatten().to(self.device))
                else:
                    raise TypeError(f"Unsupported gradient element type: {type(g)}")
            return torch.cat(flat_tensors)
        elif isinstance(gradient, torch.Tensor):
            return gradient.flatten().to(self.device)
        elif isinstance(gradient, np.ndarray):
            return torch.from_numpy(gradient).flatten().to(self.device)
        else:
            raise TypeError(f"Unsupported gradient type: {type(gradient)}")

    def collect_selected_gradients(self, selected_client_ids: List[str]) -> None:
        """
        Collects CACHED gradients from selected sellers.
        Does NOT recompute - uses gradients already computed in the training round.
        """
        self.selected_gradients = {}

        for cid in selected_client_ids:
            if cid in self.clients:
                seller = self.clients[cid].seller_obj

                # ✅ Get CACHED gradient (no recomputation!)
                gradient_tensors = seller.get_cached_gradient()

                if gradient_tensors is not None:
                    self.selected_gradients[cid] = self._ensure_tensor(gradient_tensors)
                else:
                    logging.warning(f"[SybilCoordinator] Seller {cid} has no cached gradient")

    def get_selected_average(self) -> Optional[torch.Tensor]:
        """Compute the average gradient of all selected sellers."""
        if not self.selected_gradients:
            return None
        gradients = list(self.selected_gradients.values())
        return torch.mean(torch.stack(gradients), dim=0)

    def update_nonselected_gradient(self, current_gradient: Union[torch.Tensor, List[torch.Tensor]],
                                    strategy: Optional[str] = None) -> List[torch.Tensor]:
        """Update a non-selected gradient using a specified strategy object."""
        strat_name = strategy or self.sybil_cfg.gradient_default_mode
        avg_grad = self.get_selected_average()

        # If no selected gradients exist, return original
        if avg_grad is None:
            if isinstance(current_gradient, list):
                return current_gradient
            return [self._ensure_tensor(current_gradient)]

        strategy_obj = self.strategies.get(strat_name)
        if not strategy_obj:
            raise ValueError(f"Strategy '{strat_name}' not found or configured.")

        # Store original shapes
        if isinstance(current_gradient, list):
            original_shapes = [g.shape for g in current_gradient]
        else:
            original_shapes = [current_gradient.shape]

        # Flatten, manipulate, unflatten
        current_grad_tensor = self._ensure_tensor(current_gradient)
        new_grad = strategy_obj.manipulate(current_grad_tensor, avg_grad)

        return unflatten_tensor(new_grad, original_shapes)

    def prepare_for_new_round(self) -> None:
        """Prepares state for the next round and handles dynamic triggers."""
        self.cur_round += 1
        if self.cur_round >= self.sybil_cfg.benign_rounds:
            self.start_atk = True

    def on_round_end(self) -> None:
        """Clear round-specific state."""
        self.update_historical_patterns(self.selected_gradients)
        self.adaptive_role_assignment()
        self.selected_gradients = {}


class PoisonedDataset(Dataset):
    """
    A wrapper dataset that applies a poison generator to an original dataset
    on the fly, based on a specified poison rate and data format.
    """

    def __init__(self,
                 original_dataset: Dataset,
                 poison_generator: Optional[PoisonGenerator],
                 poison_rate: float = 0.0,
                 data_format: str = 'image'):  # <-- ADD THIS NEW ARGUMENT

        if not (0.0 <= poison_rate <= 1.0):
            raise ValueError("Poison rate must be between 0.0 and 1.0")

        self.original_dataset = original_dataset
        self.poison_generator = poison_generator
        self.poison_rate = poison_rate
        self.data_format = data_format  # <-- STORE IT

        # Pre-determine which indices to poison for consistency
        n_poison = int(len(original_dataset) * poison_rate)
        all_indices = list(range(len(original_dataset)))
        random.shuffle(all_indices)
        self.poison_indices = set(all_indices[:n_poison])

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        original_sample = self.original_dataset[index]

        # Unpack based on format
        # This part is fine
        if self.data_format == 'text':
            label, data = original_sample
        else:  # image or tabular
            data, label = original_sample

        # Apply poison if needed
        if self.poison_generator and index in self.poison_indices:
            # Assuming the generator returns data and a new label
            poisoned_data, poisoned_label = self.poison_generator.apply(data, label)

            # FIX: Ensure the returned label is a tensor
            return poisoned_data, torch.tensor(poisoned_label, dtype=torch.long)

        # FIX: For the non-poisoned path, also ensure the label is a tensor!
        # This guarantees consistency for the DataLoader.
        return data, torch.tensor(label, dtype=torch.long)


class AdvancedPoisoningAdversarySeller(GradientSeller):
    """
    An adversary that performs data poisoning using a provided generator
    and can participate in a Sybil group.
    """

    def __init__(self,
                 seller_id: str,
                 # Assumes you use the RuntimeDataConfig from our previous discussion
                 data_config: RuntimeDataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 adversary_config: AdversarySellerConfig,
                 # The seller now RECEIVES the generator from the factory
                 poison_generator: Optional[PoisonGenerator],
                 device: str = "cpu",
                 **kwargs: Any):

        super().__init__(
            seller_id=seller_id,
            data_config=data_config,
            training_config=training_config,
            model_factory=model_factory,
            device=device,
            **kwargs
        )
        self.adversary_config = adversary_config
        self.selected_last_round = False

        # The seller simply stores the generator it was given.
        self.poison_generator = poison_generator

    def get_honest_gradient(self) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes and returns the gradient based *only* on the clean dataset.
        This is used by the SybilCoordinator for learning patterns.
        """
        logging.debug(f"[{self.seller_id}] Calculating HONEST gradient...")
        try:
            local_model = self.model_factory().to(self.device)
            # --- Always use the clean dataset ---
            clean_dataset = self.dataset

            gradient, stats = self._compute_local_grad(
                model_to_train=local_model,
                dataset_to_use=clean_dataset
            )

            # Basic validation
            if gradient is None:
                logging.error(f"[{self.seller_id}] ❌ _compute_local_grad returned None during honest calculation!")
                return None, {}
            if not isinstance(gradient, (list, tuple)) or len(gradient) == 0:
                logging.error(f"[{self.seller_id}] ❌ Invalid honest gradient format: {type(gradient)}")
                return None, {}

            logging.debug(f"[{self.seller_id}] ✅ Honest gradient computed.")
            # Optionally cache this if needed frequently, but likely okay to recompute.
            return gradient, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] ❌ Exception in get_honest_gradient: {e}", exc_info=True)
            return None, {}

    # --- END NEW METHOD ---
    def get_gradient_for_upload(self,
                                all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
                                target_seller_id: str = None) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Overrides the base method to implement poisoning logic.
        Returns the potentially poisoned gradient.
        """
        logging.info(f"[{self.seller_id}] Getting gradient for upload (potentially poisoned)...")

        try:
            local_model = self.model_factory().to(self.device)

            # Select dataset based on attack phase/config
            if self.poison_generator:
                logging.info(f"[{self.seller_id}] 🎭 Using poisoned dataset for upload gradient")
                dataset_for_training = PoisonedDataset(
                    original_dataset=self.dataset,
                    poison_generator=self.poison_generator,
                    poison_rate=self.adversary_config.poisoning.poison_rate
                )
            else:
                logging.info(f"[{self.seller_id}] 😇 Using clean data for upload gradient (no poison generator)")
                dataset_for_training = self.dataset

            # --- Calculate the gradient to be uploaded ---
            gradient_to_upload, stats = self._compute_local_grad(
                model_to_train=local_model,
                dataset_to_use=dataset_for_training
            )

            # --- Basic Validation ---
            if gradient_to_upload is None:
                logging.error(f"[{self.seller_id}] ❌ _compute_local_grad returned None for upload!")
                self.last_computed_gradient = None  # Clear cache
                self.last_training_stats = {}
                return None, {}
            if not isinstance(gradient_to_upload, (list, tuple)) or len(gradient_to_upload) == 0:
                logging.error(f"[{self.seller_id}] ❌ Invalid upload gradient format: {type(gradient_to_upload)}")
                self.last_computed_gradient = None  # Clear cache
                self.last_training_stats = {}
                return None, {}

            logging.info(f"[{self.seller_id}] ✅ Upload gradient computed: {len(gradient_to_upload)} parameters")

            # --- Cache the result ---
            self.last_computed_gradient = gradient_to_upload
            self.last_training_stats = stats

            return gradient_to_upload, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] ❌ Exception in get_gradient_for_upload: {e}", exc_info=True)
            self.last_computed_gradient = None  # Clear cache on error
            self.last_training_stats = {}
            return None, {}

    # --- You might also need this helper if not inherited ---
    def get_cached_gradient(self) -> Optional[List[torch.Tensor]]:
        """Returns the last computed gradient."""
        return self.last_computed_gradient


class TriggeredSubsetDataset(Dataset):
    """
    Wraps a dataset to dynamically apply a backdoor trigger to a subset of
    samples at retrieval time. This is memory-efficient as it avoids
    duplicating the dataset.

    Handles both vision (Tensor[C,H,W], int) and text datasets.
    """

    def __init__(
            self,
            original_dataset: Dataset,
            trigger_indices: np.ndarray,
            target_label: int,
            backdoor_generator: PoisonGenerator,
            device: str,
            label_first: bool = False
    ):
        self.original_dataset = original_dataset
        self.trigger_indices_set: Set[int] = set(map(int, trigger_indices))
        self.target_label = int(target_label)
        self.backdoor_generator = backdoor_generator
        self.device = device
        self.label_first = label_first

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """
        Retrieves an item, applies a trigger if the index is targeted,
        and ensures the output is a standardized (data, label) tuple on the correct device.
        """
        # 1. Get raw data and label from the original dataset
        original_data, original_label = self.original_dataset[idx]
        if self.label_first:
            data, label = original_label, original_data
        else:
            data, label = original_data, original_label

        # 2. If the index is targeted, apply the poison using the unified interface
        if idx in self.trigger_indices_set:
            # The generator handles all the logic for images, text, etc.
            data, label = self.backdoor_generator.apply(data, label)

        # 3. Standardize output and move to the correct device at the end
        # This ensures both clean and poisoned samples are handled consistently.
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)

        final_label = torch.tensor(label, dtype=torch.long, device=self.device)

        return data, final_label


class AdvancedBackdoorAdversarySeller(AdvancedPoisoningAdversarySeller):
    """
    A backdoor adversary that creates a specific backdoor generator and then
    delegates all attack logic to its parent, AdvancedPoisoningAdversarySeller.

    Its sole responsibility is to configure the correct poisoning "tool" for the job.
    """

    def __init__(self,
                 seller_id: str,
                 data_config: RuntimeDataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 adversary_config: AdversarySellerConfig,
                 model_type: str,  # 'image' or 'text'
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        # 1. Create the specific poison generator for this backdoor attack.
        #    This is the primary job of this specialized class.
        backdoor_generator = self._create_poison_generator(
            adversary_config, model_type, device, **kwargs
        )

        # 2. Call the PARENT's constructor. It receives the fully configured
        #    generator and will handle all the logic for when and how to use it.
        super().__init__(
            seller_id=seller_id,
            data_config=data_config,
            training_config=training_config,
            model_factory=model_factory,
            adversary_config=adversary_config,
            poison_generator=backdoor_generator,  # Pass the created generator up!
            save_path=save_path,
            device=device,
            **kwargs
        )
        logging.info(
            f"[{self.seller_id}] Initialized as AdvancedBackdoorAdversarySeller "
            f"with a '{type(backdoor_generator).__name__}'."
        )

    @staticmethod
    def _create_poison_generator(adv_cfg: AdversarySellerConfig, model_type: str, device: str,
                                 **kwargs: Any) -> PoisonGenerator:
        """Factory method to create the correct backdoor generator from configuration."""
        poison_cfg = adv_cfg.poisoning

        try:
            poison_type = PoisonType(poison_cfg.type)
        except ValueError:
            raise ValueError(f"Invalid poison type in config: '{poison_cfg.type}'.")

        if 'backdoor' not in poison_type.value:
            raise ValueError(f"This factory only supports backdoor types, but got '{poison_type.value}'.")

        if model_type == 'image':
            params = poison_cfg.image_backdoor_params.simple_data_poison_params
            backdoor_image_cfg = BackdoorImageConfig(
                target_label=params.target_label,
                trigger_type=ImageTriggerType(params.trigger_type),
                location=ImageTriggerLocation(params.location)
            )
            return BackdoorImageGenerator(backdoor_image_cfg, device=device)

        elif model_type == 'text':
            params = poison_cfg.text_backdoor_params
            vocab = kwargs.get('vocab')
            if not vocab:
                raise ValueError("Text backdoor generator requires 'vocab' to be provided in kwargs.")

            backdoor_text_cfg = BackdoorTextConfig(
                vocab=vocab,
                target_label=params.target_label,
                trigger_content=params.trigger_content,
                location=params.location
            )
            return BackdoorTextGenerator(backdoor_text_cfg)

        # --- THIS IS THE CORRECTED TABULAR LOGIC ---
        elif model_type == 'tabular':
            logging.debug("Factory: Creating BackdoorTabularGenerator.")

            # 1. Get the MAIN params object, which holds the target_label
            main_params = poison_cfg.tabular_backdoor_params

            # 2. Get the NESTED params object, which holds the trigger
            trigger_params = main_params.feature_trigger_params

            feature_to_idx = kwargs.get('feature_to_idx')
            if not feature_to_idx:
                raise ValueError("Tabular backdoor generator requires 'feature_to_idx' in kwargs.")

            # 3. Build the config correctly from the two different sources
            backdoor_tabular_cfg = BackdoorTabularConfig(
                target_label=main_params.target_label,  # <-- Get label from main object
                trigger_conditions=trigger_params.trigger_conditions  # <-- Get trigger from nested object
            )
            return BackdoorTabularGenerator(backdoor_tabular_cfg, feature_to_idx)
        # --- END FIX ---

        else:
            raise ValueError(f"Unsupported model_type for backdoor: {model_type}")


class AdaptiveAttackerSeller(AdvancedPoisoningAdversarySeller):
    """
    Adaptive adversary simulating three threat models:

    Threat Model 1 - ORACLE (Strongest):
        - Knows the centroid of selected gradients from previous round
        - Can precisely target the "center" of what gets selected
        - Unrealistic but upper bound on attack success

    Threat Model 2 - GRADIENT INVERSION (Moderate):
        - Knows the aggregated gradient from previous round
        - Can infer favorable data distributions via gradient analysis
        - Realistic for curious participants who observe model updates

    Threat Model 3 - BLACK BOX (Weakest/Most Realistic):
        - Only knows: was my gradient selected? (binary feedback)
        - Learns through trial-and-error which strategies work
        - Most realistic: standard FL participant with selection feedback
    """

    def __init__(self, seller_id: str, data_config: RuntimeDataConfig,
                 training_config: TrainingConfig, model_factory: Callable[[], nn.Module],
                 adversary_config: AdversarySellerConfig, device: str = "cpu", **kwargs):

        super().__init__(seller_id=seller_id, data_config=data_config,
                         training_config=training_config, model_factory=model_factory,
                         adversary_config=adversary_config, poison_generator=None,
                         device=device, **kwargs)

        self.adv_cfg = adversary_config.adaptive_attack
        if not self.adv_cfg.is_active:
            raise ValueError("AdaptiveAttackerSeller requires is_active=True")

        # Determine threat model
        self.threat_model = self.adv_cfg.threat_model  # "oracle", "gradient_inversion", "black_box"

        # Adaptive learning state
        self.phase = "exploration"
        self.strategy_history = collections.deque(maxlen=100)
        self.current_strategy = "honest"
        self.best_strategy = "honest"
        self.round_counter = 0

        # Strategy pool based on attack mode
        if self.adv_cfg.attack_mode == "gradient_manipulation":
            self.base_strategies = ["honest", "reduce_norm", "add_noise"]
        elif self.adv_cfg.attack_mode == "data_manipulation":
            # Will be dynamically determined based on dataset
            self.base_strategies = ["honest", "subsample_clean"]
            self._add_class_based_strategies()
        else:
            raise ValueError(f"Invalid attack_mode: {self.adv_cfg.attack_mode}")

        # Threat-model-specific state
        self.previous_centroid = None  # For oracle
        self.previous_aggregate = None  # For gradient inversion
        self.selection_history = []  # For black box

        logging.info(f"[{self.seller_id}] Initialized AdaptiveAttacker")
        logging.info(f"  Threat Model: {self.threat_model}")
        logging.info(f"  Attack Mode: {self.adv_cfg.attack_mode}")
        logging.info(f"  Strategies: {self.base_strategies}")

    def _add_class_based_strategies(self):
        """Add class-specific strategies if we can determine dataset classes."""
        try:
            if hasattr(self.dataset, 'classes'):
                num_classes = len(self.dataset.classes)
            elif hasattr(self.dataset, 'targets'):
                num_classes = len(set(self.dataset.targets))
            else:
                return  # Can't determine classes

            # Add focus/exclude strategies for each class
            for c in range(num_classes):
                self.base_strategies.extend([f"focus_class_{c}", f"exclude_class_{c}"])
        except:
            pass  # If we can't determine classes, just use base strategies

    # ========================================================================
    # THREAT MODEL 1: ORACLE (Centroid Knowledge)
    # ========================================================================
    def _apply_oracle_attack(self, gradient: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Oracle: Knows the centroid of selected gradients from last round.
        Strategy: Move gradient toward the previous centroid.
        """
        if self.previous_centroid is None:
            logging.debug(f"[{self.seller_id}] Oracle: No previous centroid, using honest")
            return gradient

        flat_grad = flatten_tensor(gradient)
        centroid = self.previous_centroid.to(self.device)

        if flat_grad.numel() != centroid.numel():
            logging.warning(f"[{self.seller_id}] Oracle: Dimension mismatch, using honest")
            return gradient

        # Move toward centroid while preserving gradient norm
        alpha = self.adv_cfg.mimic_strength  # e.g., 0.3
        original_norm = torch.norm(flat_grad)
        direction_to_centroid = centroid - flat_grad

        # Nudge toward centroid
        manipulated = flat_grad + alpha * direction_to_centroid

        # Preserve original norm (important for not being too obvious)
        manipulated = manipulated * (original_norm / (torch.norm(manipulated) + 1e-9))

        logging.debug(f"[{self.seller_id}] Oracle: Moved gradient toward centroid (alpha={alpha})")
        return unflatten_tensor(manipulated, [p.shape for p in gradient])

    # ========================================================================
    # THREAT MODEL 2: GRADIENT INVERSION
    # ========================================================================
    def _apply_gradient_inversion_attack(self, gradient: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Gradient Inversion: Knows last round's aggregate gradient.
        Strategy: Analyze aggregate to infer favorable data distribution,
                 then manipulate current gradient to match that pattern.
        """
        if self.previous_aggregate is None:
            logging.debug(f"[{self.seller_id}] GradInv: No previous aggregate, using honest")
            return gradient

        flat_grad = flatten_tensor(gradient)
        aggregate = self.previous_aggregate.to(self.device)

        if flat_grad.numel() != aggregate.numel():
            logging.warning(f"[{self.seller_id}] GradInv: Dimension mismatch, using honest")
            return gradient

        # Simulate "inversion" by mimicking the aggregate's characteristics
        # In practice, this could involve:
        # 1. Matching gradient statistics (mean, variance per layer)
        # 2. Gradient matching optimization
        # 3. Pattern analysis to infer data distribution

        # Simple approach: blend gradient toward aggregate
        alpha = self.adv_cfg.mimic_strength  # e.g., 0.2
        original_norm = torch.norm(flat_grad)

        # Weighted average toward aggregate's direction
        aggregate_norm = torch.norm(aggregate)
        normalized_aggregate = aggregate / (aggregate_norm + 1e-9)
        normalized_grad = flat_grad / (original_norm + 1e-9)

        blended_direction = (1 - alpha) * normalized_grad + alpha * normalized_aggregate
        manipulated = blended_direction * original_norm

        logging.debug(f"[{self.seller_id}] GradInv: Blended toward aggregate (alpha={alpha})")
        return unflatten_tensor(manipulated, [p.shape for p in gradient])

    # ========================================================================
    # THREAT MODEL 3: BLACK BOX
    # ========================================================================
    def _select_black_box_strategy(self) -> str:
        """
        Black Box: Only knows selection outcomes (binary feedback).
        Uses multi-armed bandit approach to learn best strategy.
        """
        if self.phase == "exploration":
            # Random exploration with slight bias toward untried strategies
            untried = [s for s in self.base_strategies
                       if s not in [strat for _, strat, _ in self.strategy_history]]
            if untried:
                return random.choice(untried)
            return random.choice(self.base_strategies)

        # Exploitation: Use UCB (Upper Confidence Bound) algorithm
        if not self.strategy_history:
            return "honest"

        strategy_stats = collections.defaultdict(lambda: {'attempts': 0, 'successes': 0})
        for _, strategy, selected in self.strategy_history:
            strategy_stats[strategy]['attempts'] += 1
            if selected:
                strategy_stats[strategy]['successes'] += 1

        total_attempts = sum(s['attempts'] for s in strategy_stats.values())

        # UCB1 formula
        best_score = -1
        best_strategy = "honest"

        for strategy in self.base_strategies:
            stats = strategy_stats[strategy]
            if stats['attempts'] == 0:
                return strategy  # Prioritize untried strategies

            success_rate = stats['successes'] / stats['attempts']
            exploration_bonus = np.sqrt(2 * np.log(total_attempts) / stats['attempts'])
            ucb_score = success_rate + 0.5 * exploration_bonus  # 0.5 is exploration parameter

            if ucb_score > best_score:
                best_score = ucb_score
                best_strategy = strategy

        logging.debug(f"[{self.seller_id}] BlackBox: Selected '{best_strategy}' (UCB score: {best_score:.3f})")
        return best_strategy

    def _apply_black_box_data_strategy(self, strategy: str) -> Dataset:
        """Apply data manipulation based on learned strategy."""
        if strategy == "honest":
            return self.dataset

        elif strategy == "subsample_clean":
            subset_ratio = self.adv_cfg.subset_ratio
            subset_size = max(1, int(len(self.dataset) * subset_ratio))
            indices = random.sample(range(len(self.dataset)), subset_size)
            return Subset(self.dataset, indices)

        elif strategy.startswith("focus_class_") or strategy.startswith("exclude_class_"):
            return self._apply_class_filter_strategy(strategy)

        else:
            logging.warning(f"[{self.seller_id}] Unknown strategy '{strategy}', using honest")
            return self.dataset

    def _apply_class_filter_strategy(self, strategy: str) -> Dataset:
        """Filter dataset by class."""
        try:
            action, _, class_label = strategy.rpartition('_')
            class_label = int(class_label)

            # Extract targets
            if hasattr(self.dataset, 'targets'):
                targets = np.array(self.dataset.targets)
            elif isinstance(self.dataset, Subset):
                base_targets = np.array(self.dataset.dataset.targets)
                targets = base_targets[self.dataset.indices]
            else:
                targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

            # Filter indices
            if action == "focus_class":
                indices = np.where(targets == class_label)[0]
            else:  # exclude_class
                indices = np.where(targets != class_label)[0]

            if len(indices) == 0:
                return self.dataset

            logging.debug(f"[{self.seller_id}] Applied '{strategy}': {len(indices)} samples")
            return Subset(self.dataset, indices.tolist())

        except Exception as e:
            logging.warning(f"[{self.seller_id}] Failed to apply '{strategy}': {e}")
            return self.dataset

    # ========================================================================
    # MAIN GRADIENT GENERATION
    # ========================================================================
    def get_gradient_for_upload(self, all_seller_gradients=None, target_seller_id=None):
        """Compute gradient using threat-model-specific strategy."""
        self.round_counter += 1

        # --- Phase Transition ---
        if self.phase == "exploration" and self.round_counter > self.adv_cfg.exploration_rounds:
            self.phase = "exploitation"
            if self.threat_model == "black_box":
                self._compute_best_black_box_strategy()  # Determine best strategy after exploration

        # --- Strategy Selection (only relevant for Black Box) ---
        if self.threat_model == "black_box":
            self.current_strategy = self._select_black_box_strategy()
        else:
            # Oracle and GradInv use fixed manipulation logic, not strategy selection
            self.current_strategy = "threat_model_specific"

        logging.info(f"[{self.seller_id}][{self.phase}][{self.threat_model}] "
                     f"Round {self.round_counter}: strategy='{self.current_strategy}'")

        # --- Prepare Dataset ---
        dataset_for_training = self.dataset  # Default to clean
        # Apply data strategy ONLY if Black Box + Data Manipulation mode
        if self.threat_model == "black_box" and self.adv_cfg.attack_mode == "data_manipulation":
            dataset_for_training = self._apply_black_box_data_strategy(self.current_strategy)

        # --- Compute Base Gradient ---
        try:
            local_model = self.model_factory().to(self.device)
            # Base gradient is computed on clean data if gradient manip mode,
            # or potentially modified data if black box + data manip mode.
            base_gradient, stats = self._compute_local_grad(local_model, dataset_for_training)

            if base_gradient is None:
                # Add strategy info even on failure
                stats = stats or {}
                stats.update({'threat_model': self.threat_model, 'attack_strategy': self.current_strategy,
                              'attack_phase': self.phase})
                return None, stats

            stats['threat_model'] = self.threat_model
            stats['attack_strategy'] = self.current_strategy  # Log the decided strategy
            stats['attack_phase'] = self.phase

        except Exception as e:
            logging.error(f"[{self.seller_id}] Gradient computation failed: {e}")
            return None, {'error': 'Gradient computation failed', 'threat_model': self.threat_model,
                          'attack_strategy': self.current_strategy, 'attack_phase': self.phase}

        # --- Apply Gradient Manipulation (If Applicable) ---
        final_gradient = base_gradient  # Start with the computed base

        if self.adv_cfg.attack_mode == "gradient_manipulation":
            if self.threat_model == "oracle":
                final_gradient = self._apply_oracle_attack(base_gradient)
            elif self.threat_model == "gradient_inversion":
                final_gradient = self._apply_gradient_inversion_attack(base_gradient)
            elif self.threat_model == "black_box":
                # --- Apply the chosen black-box gradient strategy ---
                final_gradient = self._apply_black_box_gradient_manipulation(base_gradient, self.current_strategy)
            # (No 'else' needed, final_gradient remains base_gradient if threat model doesn't match)

        # Cache and Return
        self.last_computed_gradient = final_gradient
        self.last_training_stats = stats
        return final_gradient, stats

    # --- NEW HELPER FOR BLACK BOX GRADIENT STRATEGIES ---
    def _apply_black_box_gradient_manipulation(self, gradient: List[torch.Tensor], strategy: str) -> List[torch.Tensor]:
        """Applies simple black-box manipulations like noise or scaling."""
        if strategy == "honest":
            return gradient

        flat_grad = flatten_tensor(gradient).clone().detach()
        manipulated_flat_grad = flat_grad
        log_msg = f"[{self.seller_id}] BlackBox: Applying gradient strategy '{strategy}'"

        if strategy == "add_orthogonal_noise":
            # ... (implementation from previous answer) ...
            noise = torch.randn_like(flat_grad)
            proj_noise_on_grad = (torch.dot(noise, flat_grad) / (torch.dot(flat_grad, flat_grad) + 1e-9)) * flat_grad
            orthogonal_noise = noise - proj_noise_on_grad
            scaled_noise = orthogonal_noise * self.adv_cfg.noise_level * torch.norm(flat_grad) / (
                    torch.norm(orthogonal_noise) + 1e-9)
            manipulated_flat_grad = flat_grad + scaled_noise
            log_msg += "."
        elif strategy == "reduce_norm":
            manipulated_flat_grad = flat_grad * self.adv_cfg.scale_factor  # Use scale_factor from config
            log_msg += f" (scaling by {self.adv_cfg.scale_factor})."
        # Add other simple black-box strategies here if needed
        else:
            # If strategy isn't 'honest' or known manip, it might be a data strategy name
            # passed incorrectly, or just 'threat_model_specific'. Return original.
            if strategy not in ["threat_model_specific", "honest"]:
                logging.warning(f"[{self.seller_id}] Unknown black-box gradient strategy '{strategy}'. Using honest.")
            return gradient

        logging.debug(log_msg)
        original_shapes = [p.shape for p in gradient]
        return unflatten_tensor(manipulated_flat_grad, original_shapes)

    def _compute_best_black_box_strategy(self):
        """Determine best strategy after exploration phase (black box only)."""
        if not self.strategy_history:
            self.best_strategy = "honest"
            return

        strategy_success = collections.defaultdict(lambda: {'attempts': 0, 'successes': 0})
        for _, strategy, selected in self.strategy_history:
            strategy_success[strategy]['attempts'] += 1
            if selected:
                strategy_success[strategy]['successes'] += 1

        success_rates = {
            s: stats['successes'] / stats['attempts']
            for s, stats in strategy_success.items()
            if stats['attempts'] > 0
        }

        if success_rates:
            self.best_strategy = max(success_rates, key=success_rates.get)
            logging.info(f"[{self.seller_id}] Exploration complete. "
                         f"Success rates: {success_rates}. Best: '{self.best_strategy}'")

    # ========================================================================
    # ROUND END PROCESSING
    # ========================================================================
    def round_end_process(self, round_number: int, was_selected: bool,
                          marketplace_metrics: Dict = None, **kwargs):
        """Record outcome and update threat-model-specific state."""
        super().round_end_process(round_number=round_number, was_selected=was_selected,
                                  marketplace_metrics=marketplace_metrics, **kwargs)

        # Update selection history
        if self.threat_model == "black_box" and self.phase == "exploration":
            self.strategy_history.append((round_number, self.current_strategy, was_selected))
            logging.debug(f"[{self.seller_id}] BlackBox: Recorded {self.current_strategy} -> {was_selected}")

        # Store information for next round based on threat model
        if marketplace_metrics:
            # Oracle: Store centroid of selected gradients
            if self.threat_model == "oracle" and 'selected_centroid_flat' in marketplace_metrics:
                self.previous_centroid = marketplace_metrics['selected_centroid_flat'].clone().detach().cpu()
                logging.debug(f"[{self.seller_id}] Oracle: Stored centroid")

            # Gradient Inversion: Store aggregated gradient
            if self.threat_model == "gradient_inversion" and 'final_aggregated_gradient_flat' in marketplace_metrics:
                self.previous_aggregate = marketplace_metrics['final_aggregated_gradient_flat'].clone().detach().cpu()
                logging.debug(f"[{self.seller_id}] GradInv: Stored aggregate gradient")


class DrowningAttackerSeller(GradientSeller):
    """
    Stealthy gradient manipulation attack that maintains selection probability
    while subtly pulling the aggregate toward a malicious objective.

    Strategy:
    1. Mimicry Phase: Learn the "shape" of selected gradients
    2. Attack Phase: Submit gradients that:
       - Match statistical properties of honest gradients (norm, direction)
       - Contain subtle malicious components in specific layers
       - Stay within selection threshold

    This is more realistic than orthogonal drift because it:
    - Maintains similarity to honest gradients
    - Targets specific vulnerable layers
    - Adapts based on selection feedback
    """

    def __init__(self, *args, adversary_config: AdversarySellerConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_cfg = adversary_config.gradient_replacement_attack
        if not self.adv_cfg.is_active:
            raise ValueError("GradientReplacementAttacker requires is_active=True")

        self.phase = "mimicry"
        self.round_counter = 0

        # Learning state
        self.honest_gradient_stats = {
            'mean_norm': None,
            'layer_norms': {},
            'direction_estimate': None
        }
        self.selection_history = []

        # Attack state
        self.target_layers = self.adv_cfg.target_layers  # e.g., ['fc.weight', 'fc.bias']
        self.attack_intensity = self.adv_cfg.attack_intensity  # e.g., 0.1

        logging.info(f"[{self.seller_id}] Initialized GradientReplacementAttacker")
        logging.info(f"  Mimicry Rounds: {self.adv_cfg.mimicry_rounds}")
        logging.info(f"  Target Layers: {self.target_layers if self.target_layers else 'All'}")
        logging.info(f"  Attack Intensity: {self.attack_intensity}")

    def _update_honest_gradient_stats(self, gradient: List[torch.Tensor]):
        """Learn characteristics of honest gradients during mimicry."""
        flat_grad = flatten_tensor(gradient)
        grad_norm = torch.norm(flat_grad).item()

        # Update running average of norm
        if self.honest_gradient_stats['mean_norm'] is None:
            self.honest_gradient_stats['mean_norm'] = grad_norm
        else:
            beta = 0.9
            self.honest_gradient_stats['mean_norm'] = (
                    beta * self.honest_gradient_stats['mean_norm'] +
                    (1 - beta) * grad_norm
            )

        # Store direction estimate (EMA of normalized gradient)
        normalized_grad = flat_grad / (grad_norm + 1e-9)
        if self.honest_gradient_stats['direction_estimate'] is None:
            self.honest_gradient_stats['direction_estimate'] = normalized_grad.detach().cpu()
        else:
            beta = 0.9
            current_estimate = self.honest_gradient_stats['direction_estimate'].to(self.device)
            updated = beta * current_estimate + (1 - beta) * normalized_grad
            self.honest_gradient_stats['direction_estimate'] = updated.detach().cpu()

        logging.debug(f"[{self.seller_id}] Updated stats: mean_norm={self.honest_gradient_stats['mean_norm']:.4f}")

    def _compute_malicious_gradient(self) -> List[torch.Tensor]:
        """
        Compute gradient on malicious objective (backdoor, poisoning, etc.).
        This is what we actually want to inject into the aggregate.
        """
        if self.adv_cfg.attack_type == "backdoor" and hasattr(self, 'backdoor_dataset'):
            # Compute gradient on backdoor task
            model = self.model_factory().to(self.device)
            backdoor_grad, _ = self._compute_local_grad(model, self.backdoor_dataset)
            return backdoor_grad

        elif self.adv_cfg.attack_type == "untargeted_poisoning":
            # Compute gradient that degrades model performance
            model = self.model_factory().to(self.device)
            # Option 1: Negative gradient
            honest_grad, _ = self._compute_local_grad(model, self.dataset)
            return [-g for g in honest_grad]

        elif self.adv_cfg.attack_type == "targeted_poisoning":
            # Compute gradient toward specific misclassification
            # This requires specialized poisoned dataset
            model = self.model_factory().to(self.device)
            if hasattr(self, 'poisoned_dataset'):
                poison_grad, _ = self._compute_local_grad(model, self.poisoned_dataset)
                return poison_grad

        # Fallback: return honest gradient
        model = self.model_factory().to(self.device)
        grad, _ = self._compute_local_grad(model, self.dataset)
        return grad

    def _identify_vulnerable_layers(self, gradient: List[torch.Tensor]) -> List[int]:
        """
        Identify which layers to inject malicious gradients into.
        Strategy: Target layers with high magnitude or final layers.
        """
        if self.target_layers:
            # User specified target layers by name
            # This requires model structure knowledge
            return []  # Would need to map layer names to indices

        # Automatic selection: target last few layers (most impactful)
        num_layers = len(gradient)
        num_target = max(1, int(num_layers * 0.2))  # Target top 20% of layers

        # Select layers with largest gradient norms
        layer_norms = [torch.norm(g).item() for g in gradient]
        sorted_indices = sorted(range(num_layers), key=lambda i: layer_norms[i], reverse=True)

        return sorted_indices[:num_target]

    def _create_stealthy_malicious_gradient(self,
                                            honest_gradient: List[torch.Tensor],
                                            malicious_gradient: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Blend honest and malicious gradients while maintaining:
        1. Overall norm similar to honest gradient
        2. Direction close enough to pass selection
        3. Malicious component in vulnerable layers
        """
        # Flatten for analysis
        honest_flat = flatten_tensor(honest_gradient)
        malicious_flat = flatten_tensor(malicious_gradient)

        honest_norm = torch.norm(honest_flat)
        honest_direction = honest_flat / (honest_norm + 1e-9)

        # Get expected norm from learning
        if self.honest_gradient_stats['mean_norm']:
            target_norm = self.honest_gradient_stats['mean_norm']
        else:
            target_norm = honest_norm.item()

        # Strategy 1: Layer-wise replacement (most effective)
        if self.adv_cfg.replacement_strategy == "layer_wise":
            vulnerable_layers = self._identify_vulnerable_layers(honest_gradient)

            blended_gradient = []
            for i, (h_grad, m_grad) in enumerate(zip(honest_gradient, malicious_gradient)):
                if i in vulnerable_layers:
                    # Replace with malicious gradient in this layer
                    alpha = self.attack_intensity  # e.g., 0.2
                    layer_blend = (1 - alpha) * h_grad + alpha * m_grad
                    blended_gradient.append(layer_blend)
                    logging.debug(f"[{self.seller_id}] Injected into layer {i} (alpha={alpha})")
                else:
                    # Keep honest gradient
                    blended_gradient.append(h_grad)

            # Adjust overall norm to match expected
            blended_flat = flatten_tensor(blended_gradient)
            current_norm = torch.norm(blended_flat)
            scale = target_norm / (current_norm.item() + 1e-9)

            blended_gradient = [g * scale for g in blended_gradient]

        # Strategy 2: Global blending with norm preservation
        elif self.adv_cfg.replacement_strategy == "global_blend":
            alpha = self.attack_intensity

            # Blend in gradient space
            blended_flat = (1 - alpha) * honest_flat + alpha * malicious_flat

            # Ensure we maintain similarity to honest direction
            blended_direction = blended_flat / (torch.norm(blended_flat) + 1e-9)
            cosine_sim = torch.dot(blended_direction, honest_direction)

            similarity_threshold = self.adv_cfg.similarity_threshold  # e.g., 0.95
            if cosine_sim < similarity_threshold:
                # Reduce alpha to maintain similarity
                # Binary search for maximum alpha
                alpha_low, alpha_high = 0.0, alpha
                for _ in range(10):
                    alpha_mid = (alpha_low + alpha_high) / 2
                    test_blend = (1 - alpha_mid) * honest_flat + alpha_mid * malicious_flat
                    test_dir = test_blend / (torch.norm(test_blend) + 1e-9)
                    test_sim = torch.dot(test_dir, honest_direction)

                    if test_sim >= similarity_threshold:
                        alpha_low = alpha_mid
                    else:
                        alpha_high = alpha_mid

                alpha = alpha_low
                blended_flat = (1 - alpha) * honest_flat + alpha * malicious_flat
                logging.debug(f"[{self.seller_id}] Reduced alpha to {alpha:.3f} for similarity")

            # Scale to target norm
            blended_flat = blended_flat * (target_norm / (torch.norm(blended_flat) + 1e-9))

            # Unflatten
            original_shapes = [g.shape for g in honest_gradient]
            blended_gradient = unflatten_tensor(blended_flat, original_shapes)

        else:
            logging.warning(f"[{self.seller_id}] Unknown strategy, using honest gradient")
            blended_gradient = honest_gradient

        return blended_gradient

    def get_gradient_for_upload(self, **kwargs) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Generate gradient for this round."""
        self.round_counter += 1

        # Phase transition
        if self.phase == "mimicry" and self.round_counter > self.adv_cfg.mimicry_rounds:
            self.phase = "attack"
            logging.info(f"[{self.seller_id}] Transitioning to Attack Phase")
            logging.info(f"  Learned mean norm: {self.honest_gradient_stats['mean_norm']:.4f}")

        # Compute honest gradient
        try:
            model = self.model_factory().to(self.device)
            honest_gradient, stats = self._compute_local_grad(model, self.dataset)
            if honest_gradient is None:
                return None, stats
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to compute honest gradient: {e}")
            return None, {'error': str(e)}

        stats['attack_phase'] = self.phase

        # MIMICRY PHASE
        if self.phase == "mimicry":
            self._update_honest_gradient_stats(honest_gradient)
            logging.info(f"[{self.seller_id}][Mimicry] Round {self.round_counter}: Learning")
            return honest_gradient, stats

        # ATTACK PHASE
        else:
            # Compute malicious gradient
            malicious_gradient = self._compute_malicious_gradient()

            # Create stealthy blend
            attack_gradient = self._create_stealthy_malicious_gradient(
                honest_gradient,
                malicious_gradient
            )

            # Verify similarity
            honest_flat = flatten_tensor(honest_gradient)
            attack_flat = flatten_tensor(attack_gradient)

            cosine_sim = torch.nn.functional.cosine_similarity(
                honest_flat.unsqueeze(0),
                attack_flat.unsqueeze(0)
            ).item()

            norm_ratio = torch.norm(attack_flat) / (torch.norm(honest_flat) + 1e-9)

            logging.info(f"[{self.seller_id}][Attack] Round {self.round_counter}: "
                         f"cosine_sim={cosine_sim:.3f}, norm_ratio={norm_ratio:.3f}")

            stats['cosine_similarity'] = cosine_sim
            stats['norm_ratio'] = norm_ratio.item()
            stats['attack_type'] = self.adv_cfg.attack_type

            return attack_gradient, stats

    def round_end_process(self, round_number: int, was_selected: bool,
                          marketplace_metrics: Dict = None, **kwargs):
        """Track selection outcomes to adapt attack."""
        super().round_end_process(round_number, was_selected, marketplace_metrics, **kwargs)

        self.selection_history.append({
            'round': round_number,
            'phase': self.phase,
            'selected': was_selected
        })

        # Adapt attack intensity based on selection rate
        if self.phase == "attack" and len(self.selection_history) >= 5:
            recent_selections = [h['selected'] for h in self.selection_history[-5:]
                                 if h['phase'] == 'attack']
            if recent_selections:
                selection_rate = sum(recent_selections) / len(recent_selections)

                # If selection rate drops, reduce attack intensity
                if selection_rate < 0.5 and self.attack_intensity > 0.05:
                    self.attack_intensity *= 0.9
                    logging.info(f"[{self.seller_id}] Reduced attack intensity to {self.attack_intensity:.3f}")
                # If selection rate is high, can increase intensity
                elif selection_rate > 0.8 and self.attack_intensity < 0.5:
                    self.attack_intensity *= 1.1
                    logging.info(f"[{self.seller_id}] Increased attack intensity to {self.attack_intensity:.3f}")
