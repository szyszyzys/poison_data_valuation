import collections
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
from torch.utils.data import DataLoader, Dataset

from attack.attack_gradient_market.poison_attack.attack_utils import PoisonGenerator, BackdoorImageGenerator, \
    BackdoorTextGenerator, BackdoorTabularGenerator
from common.enums import ImageTriggerType, ImageTriggerLocation, PoisonType
from common.gradient_market_configs import AdversarySellerConfig, BackdoorImageConfig, BackdoorTextConfig, SybilConfig, \
    RuntimeDataConfig, TrainingConfig, BackdoorTabularConfig
from common.utils import unflatten_tensor
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

    def register_seller(self, seller: GradientSeller) -> None:
        """Register a malicious seller with the coordinator."""
        if not hasattr(seller, 'seller_id'):
            raise AttributeError("Seller object must have a 'seller_id' attribute")
        self.clients[seller.seller_id] = ClientState(seller_obj=seller)

    def apply_manipulation(self, gradients_dict: Dict, all_sellers: Dict) -> Dict:
        """
        Apply sybil manipulation based on historical patterns.

        Args:
            gradients_dict: Dictionary of {seller_id: gradient}
            all_sellers: Dictionary of {seller_id: seller_object}

        Returns:
            Updated gradients_dict with manipulated sybil gradients
        """
        if not self.start_atk:
            logging.info("[SybilCoordinator] Not in attack phase yet - no manipulation")
            return gradients_dict

        if not self.selection_patterns:
            logging.info("[SybilCoordinator] No historical patterns yet - sybils submit honest gradients")
            return gradients_dict

        avg_similarity = self.selection_patterns.get('avg_similarity', 0)
        logging.info(f"[SybilCoordinator] Applying manipulation (historical similarity: {avg_similarity:.4f})")

        manipulated_count = 0
        for sybil_id in self.clients:
            if sybil_id not in gradients_dict:
                continue

            client_state = self.clients[sybil_id]

            # Only manipulate if in attack phase
            if client_state.phase == "attack":
                # Get strategy based on role
                strategy = self._get_strategy_for_role(client_state.role)

                # Get honest gradient from seller
                seller = all_sellers.get(sybil_id)
                if not seller or not hasattr(seller, 'get_honest_gradient'):
                    continue

                original_grad = seller.get_honest_gradient()
                if not original_grad:
                    continue

                # Manipulate based on historical winners
                manipulated_grad = self.update_nonselected_gradient(
                    current_gradient=original_grad,
                    strategy=strategy
                )

                # Replace in gradients dict
                gradients_dict[sybil_id] = manipulated_grad

                # Update seller's cache
                seller.last_computed_gradient = manipulated_grad

                manipulated_count += 1
                logging.debug(f"   Manipulated {sybil_id} (role: {client_state.role}, strategy: {strategy})")

        logging.info(f"[SybilCoordinator] Manipulated {manipulated_count} sybil gradients")
        return gradients_dict

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

    def get_gradient_for_upload(self,
                                all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
                                target_seller_id: str = None) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Overrides the base method to implement poisoning and Sybil logic.
        Returns gradient as a list of tensors matching global_model parameters.
        """
        logging.info(f"[{self.seller_id}] Getting gradient for upload...")

        try:
            local_model = self.model_factory().to(self.device)

            # Debug logging (only for specific sellers to reduce noise)
            if self.seller_id in ["adv_0", "bn_4"]:
                logging.info(f"--- Local Model Architecture for {self.seller_id} ---")
                logging.info(f"Number of parameters: {sum(p.numel() for p in local_model.parameters())}")
                logging.info(f"Parameter shapes: {[p.shape for p in local_model.parameters()]}")

            # Select dataset based on attack phase
            if self.poison_generator:
                logging.info(f"[{self.seller_id}] 🎭 Attack phase: using poisoned dataset")
                dataset_for_training = PoisonedDataset(
                    original_dataset=self.dataset,
                    poison_generator=self.poison_generator,
                    poison_rate=self.adversary_config.poisoning.poison_rate
                )
            else:
                logging.info(f"[{self.seller_id}] 😇 Benign phase: using clean data")
                dataset_for_training = self.dataset

            base_gradient, stats = self._compute_local_grad(
                model_to_train=local_model,  # ✅ Use local_model!
                dataset_to_use=dataset_for_training
            )

            if base_gradient is None:
                logging.error(f"[{self.seller_id}] ❌ _compute_local_grad returned None!")
                return None, {}

            if not isinstance(base_gradient, (list, tuple)):
                logging.error(f"[{self.seller_id}] ❌ Gradient is not a list/tuple: {type(base_gradient)}")
                return None, {}

            if len(base_gradient) == 0:
                logging.error(f"[{self.seller_id}] ❌ Gradient is empty!")
                return None, {}

            logging.info(f"[{self.seller_id}] ✅ Base gradient validated: {len(base_gradient)} parameters")

            # Return base gradient for normal sellers or selected Sybils
            logging.info(f"[{self.seller_id}] ✅ Returning base gradient")
            return base_gradient, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] ❌ Exception in get_gradient_for_upload: {e}", exc_info=True)
            return None, {}


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


class AdaptiveAttackerSeller(GradientSeller):
    """
    An adaptive adversary that learns the best strategy to get selected.
    It can operate in two modes:
    1. Gradient Manipulation: Directly alters the computed gradient.
    2. Data Poisoning: Changes its dataset distribution before training.
    """

    def __init__(self, *args, adversary_config: AdversarySellerConfig,
                 poison_generator_factory: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_cfg = adversary_config.adaptive_attack
        self.adversary_config = adversary_config
        if not self.adv_cfg.is_active:
            raise ValueError("AdaptiveAttackerSeller created but is_active is False in config.")

        self.poison_generator_factory = poison_generator_factory
        self.phase = "exploration"
        self.strategy_history = []
        self.current_strategy = "honest"
        self.best_strategy = "honest"
        self.round_counter = 0

        logging.info(f"[{self.seller_id}] Initialized as AdaptiveAttacker in '{self.adv_cfg.attack_mode}' mode.")
        logging.info(f" -> Exploring for {self.adv_cfg.exploration_rounds} rounds.")

    def _apply_gradient_manipulation(self, gradient: List[torch.Tensor], strategy: str) -> List[torch.Tensor]:
        """Applies a manipulation directly to the gradient tensors."""
        if strategy == "honest":
            return gradient

        manipulated_grad = []
        for tensor in gradient:
            if strategy == "scale_up":
                manipulated_grad.append(tensor * self.adv_cfg.scale_factor)
            elif strategy == "scale_down":
                manipulated_grad.append(tensor / self.adv_cfg.scale_factor)
            elif strategy == "add_noise":
                noise = torch.randn_like(tensor) * self.adv_cfg.noise_level
                manipulated_grad.append(tensor + noise)
            else:
                manipulated_grad.append(tensor)
        return manipulated_grad

    def _get_poisoned_dataset(self, strategy: str) -> Dataset:
        """Creates a poisoned dataset based on the chosen strategy."""
        if strategy == "honest" or not self.poison_generator_factory:
            return self.dataset

        # Use the factory to create a poison generator for the chosen strategy
        poison_generator = self.poison_generator_factory(poison_type_str=strategy)
        if poison_generator:
            return PoisonedDataset(
                original_dataset=self.dataset,
                poison_generator=poison_generator,
                poison_rate=self.adversary_config.poisoning.poison_rate
            )
        return self.dataset  # Fallback to clean data

    def _determine_best_strategy(self):
        """Analyzes history to find the most successful strategy."""
        if not self.strategy_history:
            self.best_strategy = "honest"
            return

        strategy_success = collections.defaultdict(int)
        strategy_attempts = collections.defaultdict(int)

        for _, strategy, was_selected in self.strategy_history:
            strategy_attempts[strategy] += 1
            if was_selected:
                strategy_success[strategy] += 1

        success_rates = {s: strategy_success[s] / strategy_attempts[s] for s in strategy_attempts}

        if success_rates:
            self.best_strategy = max(success_rates, key=success_rates.get)

        logging.info(
            f"[{self.seller_id}] Exploration complete. Success Rates: {dict(success_rates)}. Best Strategy: '{self.best_strategy}'")

    def get_gradient_for_upload(self,
                                all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
                                target_seller_id: str = None) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes gradient using adaptive strategy.
        The stateful factory automatically provides model with current global weights.
        """
        self.round_counter += 1

        if self.phase == "exploration" and self.round_counter > self.adv_cfg.exploration_rounds:
            self.phase = "exploitation"
            self._determine_best_strategy()

        # 1. Choose a strategy for this round
        strategy_pool = (self.adv_cfg.gradient_strategies
                         if self.adv_cfg.attack_mode == "gradient_manipulation"
                         else self.adv_cfg.data_strategies)
        if self.phase == "exploration":
            self.current_strategy = random.choice(strategy_pool)
        else:
            self.current_strategy = self.best_strategy

        logging.info(
            f"[{self.seller_id}][{self.phase.capitalize()}] Round {self.round_counter}: "
            f"Using strategy '{self.current_strategy}'")

        # 2. Prepare dataset based on attack mode
        dataset_for_training = self.dataset
        if self.adv_cfg.attack_mode == "data_poisoning":
            dataset_for_training = self._get_poisoned_dataset(self.current_strategy)

        # 3. Create model with current global weights (factory handles this)
        try:
            local_model = self.model_factory().to(self.device)
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to create model: {e}", exc_info=True)
            return None, {'error': 'Model creation failed.'}

        # 4. Compute the base gradient
        base_gradient, stats = self._compute_local_grad(local_model, dataset_for_training)
        if base_gradient is None:
            return None, stats

        stats['attack_strategy'] = self.current_strategy
        stats['attack_phase'] = self.phase

        # 5. If in gradient manipulation mode, apply the manipulation
        if self.adv_cfg.attack_mode == "gradient_manipulation":
            final_gradient = self._apply_gradient_manipulation(base_gradient, self.current_strategy)
            return final_gradient, stats

        # Otherwise, manipulation was done via poisoned dataset
        return base_gradient, stats

    def round_end_process(self, round_number: int, was_selected: bool, **kwargs):
        """Records the outcome of the strategy used."""
        super().round_end_process(round_number=round_number, was_selected=was_selected, **kwargs)
        if self.phase == "exploration":
            self.strategy_history.append((round_number, self.current_strategy, was_selected))


class DrowningAttackerSeller(GradientSeller):
    """
    Implements the Targeted Drowning (Centroid Poisoning) Attack.

    This seller first acts honestly to gain the aggregator's trust, then
    slowly and consistently drifts its gradient away from the honest direction.
    This is designed to poison the memory of adaptive defenses like MartFL,
    eventually making honest sellers appear to be outliers.
    """

    def __init__(self, *args, adversary_config: AdversarySellerConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_cfg = adversary_config.drowning_attack
        if not self.adv_cfg.is_active:
            raise ValueError("DrowningAttackerSeller created but is_active is False in config.")

        # --- Attack State ---
        self.phase = "mimicry"  # Can be "mimicry" or "drift"
        self.round_counter = 0
        self.drift_vector: Optional[List[torch.Tensor]] = None

        logging.info(
            f"[{self.seller_id}] Initialized as DrowningAttacker. "
            f"Mimicking for {self.adv_cfg.mimicry_rounds} rounds."
        )

    def get_gradient_for_upload(self,
                                all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
                                target_seller_id: str = None) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        self.round_counter += 1

        # --- Phase Transition Logic ---
        if self.phase == "mimicry" and self.round_counter > self.adv_cfg.mimicry_rounds:
            self.phase = "drift"
            logging.info(f"[{self.seller_id}] Transitioning to Drift Phase.")

        # 1. Always compute the honest gradient first.
        # This represents the gradient of the honest, high-quality target we are mimicking.
        honest_gradient, stats = super().get_gradient_for_upload()
        if honest_gradient is None:
            return None, stats

        stats['attack_phase'] = self.phase

        # 2. If in the mimicry phase, act honestly to build trust.
        if self.phase == "mimicry":
            logging.info(f"[{self.seller_id}][Mimicry] Round {self.round_counter}: Submitting honest gradient.")
            return honest_gradient, stats

        # 3. If in the drift phase, apply the consistent poisoning.
        else:  # self.phase == "drift"
            # Initialize the drift direction on the first drift round
            if self.drift_vector is None:
                logging.info(f"[{self.seller_id}] Initializing a random, consistent drift vector.")
                self.drift_vector = [
                    torch.randn_like(p, device=self.device) for p in honest_gradient
                ]

            # Apply the drift
            drifted_gradient = []
            for i, tensor in enumerate(honest_gradient):
                # Add the scaled drift vector to the honest gradient
                drift_update = self.drift_vector[i] * self.adv_cfg.drift_factor
                drifted_gradient.append(tensor + drift_update)

            logging.info(f"[{self.seller_id}][Drift] Round {self.round_counter}: Submitting drifted gradient.")
            return drifted_gradient, stats


class CompetitorMimicrySeller(GradientSeller):
    """
    Implements the Direct Competitor Mimicry Attack.

    This seller identifies a high-quality target competitor and submits gradients
    that closely mimic the target's updates with small perturbations. The goal is
    to evade detection while stealing market share from the target.

    Success Metric: Target's selection rate drops (e.g., 80% → 40%) while
    attacker gains similar selection rate (0% → 40%).
    """

    def __init__(self, *args, adversary_config: AdversarySellerConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_cfg = adversary_config.mimicry_attack
        if not self.adv_cfg.is_active:
            raise ValueError("CompetitorMimicrySeller created but is_active is False in config.")

        # --- Attack Configuration ---
        self.target_seller_id = self.adv_cfg.target_seller_id  # Who to mimic
        self.noise_scale = self.adv_cfg.noise_scale  # How much noise to add (e.g., 0.05)
        self.mimicry_strategy = self.adv_cfg.strategy  # "exact_copy", "noisy_copy", or "scaled_copy"

        # --- Attack State ---
        self.round_counter = 0
        self.observation_rounds = self.adv_cfg.observation_rounds  # Learn target's pattern first (e.g., 3)
        self.target_gradient_history = []  # Store recent target gradients for averaging

        logging.info(
            f"[{self.seller_id}] Initialized as CompetitorMimicry attacker. "
            f"Target: {self.target_seller_id}, Strategy: {self.mimicry_strategy}, "
            f"Noise: {self.noise_scale}, Observing for {self.observation_rounds} rounds."
        )

    def get_gradient_for_upload(
            self,
            all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
            target_seller_id: str = None
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes gradient that mimics a target competitor.
        The stateful factory automatically provides model with current global weights.
        """
        self.round_counter += 1

        # 1. Compute our own honest gradient (for fallback and stats)
        # Note: super().get_gradient_for_upload() creates its own model via factory
        honest_gradient, stats = super().get_gradient_for_upload()
        if honest_gradient is None:
            return None, stats

        stats['attack_type'] = 'competitor_mimicry'
        stats['target_id'] = self.target_seller_id

        # 2. Check if we have access to target's gradient
        if all_seller_gradients is None or self.target_seller_id not in all_seller_gradients:
            logging.warning(
                f"[{self.seller_id}] Round {self.round_counter}: "
                f"Cannot access target {self.target_seller_id}'s gradient. "
                f"Falling back to honest gradient."
            )
            stats['fallback_reason'] = 'target_unavailable'
            return honest_gradient, stats

        target_gradient = all_seller_gradients[self.target_seller_id]

        # 3. Observation Phase: Learn target's gradient pattern
        if self.round_counter <= self.observation_rounds:
            self.target_gradient_history.append([t.clone() for t in target_gradient])
            logging.info(
                f"[{self.seller_id}][Observation] Round {self.round_counter}/{self.observation_rounds}: "
                f"Learning target's pattern. Submitting honest gradient."
            )
            stats['attack_phase'] = 'observation'
            return honest_gradient, stats

        # 4. Attack Phase: Mimic the target
        stats['attack_phase'] = 'active_mimicry'

        if self.mimicry_strategy == "exact_copy":
            # Strategy 1: Exact copy (most aggressive, easiest to detect if defense checks for duplicates)
            malicious_gradient = [t.clone() for t in target_gradient]
            logging.info(f"[{self.seller_id}][Attack] Submitting exact copy of target's gradient.")

        elif self.mimicry_strategy == "noisy_copy":
            # Strategy 2: Add small Gaussian noise (more stealthy)
            malicious_gradient = []
            for tensor in target_gradient:
                noise = torch.randn_like(tensor) * self.noise_scale * tensor.abs().mean()
                malicious_gradient.append(tensor + noise)
            logging.info(
                f"[{self.seller_id}][Attack] Submitting noisy copy "
                f"(noise_scale={self.noise_scale})."
            )

        elif self.mimicry_strategy == "scaled_copy":
            # Strategy 3: Scale magnitude slightly (preserves direction, changes norm)
            scale_factor = 1.0 + torch.randn(1).item() * self.noise_scale  # e.g., 1.0 ± 0.05
            malicious_gradient = [tensor * scale_factor for tensor in target_gradient]
            logging.info(
                f"[{self.seller_id}][Attack] Submitting scaled copy "
                f"(scale={scale_factor:.3f})."
            )

        elif self.mimicry_strategy == "averaged_history":
            # Strategy 4: Average of target's recent gradients (smooths out noise)
            if len(self.target_gradient_history) > 0:
                malicious_gradient = []
                for param_idx in range(len(target_gradient)):
                    # Average this parameter across historical observations
                    param_history = [g[param_idx] for g in self.target_gradient_history]
                    avg_param = torch.stack(param_history).mean(dim=0)
                    # Add current target gradient with weight
                    blended = 0.7 * target_gradient[param_idx] + 0.3 * avg_param
                    malicious_gradient.append(blended)
                logging.info(
                    f"[{self.seller_id}][Attack] Submitting averaged gradient from "
                    f"{len(self.target_gradient_history)} historical observations."
                )
            else:
                # Fallback if no history
                malicious_gradient = [t.clone() for t in target_gradient]

        else:
            raise ValueError(f"Unknown mimicry strategy: {self.mimicry_strategy}")

        # 5. Update history (rolling window)
        max_history = 5
        self.target_gradient_history.append([t.clone() for t in target_gradient])
        if len(self.target_gradient_history) > max_history:
            self.target_gradient_history.pop(0)

        # 6. Compute similarity for logging
        with torch.no_grad():
            target_flat = torch.cat([t.flatten() for t in target_gradient])
            malicious_flat = torch.cat([t.flatten() for t in malicious_gradient])
            cosine_sim = F.cosine_similarity(target_flat.unsqueeze(0), malicious_flat.unsqueeze(0))
            stats['cosine_similarity_to_target'] = cosine_sim.item()

        logging.info(
            f"[{self.seller_id}][Attack] Round {self.round_counter}: "
            f"Cosine similarity to target: {stats['cosine_similarity_to_target']:.4f}"
        )

        return malicious_gradient, stats
