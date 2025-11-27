import collections
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
from typing import List, Dict, Any, Optional, Callable, Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from attack.attack_gradient_market.poison_attack.attack_utils import PoisonGenerator, BackdoorImageGenerator, \
    BackdoorTextGenerator, BackdoorTabularGenerator
from common.enums import ImageTriggerType, ImageTriggerLocation, PoisonType
from common.gradient_market_configs import AdversarySellerConfig, BackdoorImageConfig, BackdoorTextConfig, SybilConfig, \
    RuntimeDataConfig, TrainingConfig, BackdoorTabularConfig, SybilDrowningConfig
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


class BuyerCollusionStrategy(BaseGradientStrategy):
    """
    The Sybil simply copies the Buyer's root gradient.
    This ensures Cosine Similarity = 1.0 against the buyer's baseline.
    """

    def __init__(self, noise_scale=1e-6):
        self.noise_scale = noise_scale

    def manipulate(self, original_malicious_flat: torch.Tensor,
                   root_gradient_flat: torch.Tensor) -> torch.Tensor:

        if root_gradient_flat is None:
            # Fallback if no root gradient is provided (shouldn't happen in this attack mode)
            return original_malicious_flat

        # Copy the buyer's gradient
        sybil_gradient = root_gradient_flat.clone()

        # Optional: Add insignificant noise to avoid bit-exact duplicate detection
        # (Some defenses filter out updates that are mathematically identical)
        if self.noise_scale > 0:
            noise = torch.randn_like(sybil_gradient) * self.noise_scale
            sybil_gradient += noise

        return sybil_gradient


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
            marketplace_metrics: Dict = None,
            **kwargs
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


class DrowningContext:
    def __init__(self,
                 victim_gradient: torch.Tensor,
                 benign_centroid: torch.Tensor):
        self.victim_gradient = victim_gradient
        self.benign_centroid = benign_centroid


# --- C. Define the Base Strategy Class ---
# Your coordinator imports 'BaseGradientStrategy', so let's define it.
class BaseGradientStrategy:
    """Abstract base class for all Sybil gradient manipulation strategies."""

    def manipulate(self,
                   original_malicious_flat: torch.Tensor,
                   reference_centroid_flat: torch.Tensor) -> torch.Tensor:
        """
        Manipulates a Sybil's gradient based on a reference centroid.
        This is used for 'mimic', 'pivot', etc.
        """
        raise NotImplementedError


# --- D. The Drowning Strategy Implementation ---
class DrowningStrategy(BaseGradientStrategy):
    """
    Implements the Targeted Drowning Attack .

    This strategy is unique: it does not 'manipulate' the Sybil's
    original gradient. Instead, it calculates a *repulsion gradient*
    that all Sybils assigned to this strategy will submit.
    """

    def __init__(self, config: SybilDrowningConfig):
        self.config = config
        self.victim_id = config.victim_id
        self.attack_strength = config.attack_strength
        logging.info(
            f"[DrowningStrategy] Initialized. "
            f"Victim: {self.victim_id}, Strength: {self.attack_strength}"
        )

    def calculate_repulsion_gradient(self,
                                     context: DrowningContext) -> Optional[torch.Tensor]:
        """
        Calculates the single repulsion gradient for all Sybils [cite: 528-532].
        """
        if (context.victim_gradient is None or
                context.benign_centroid is None):
            logging.warning("[DrowningStrategy] Missing victim or centroid.")
            return None

        g_victim_flat = context.victim_gradient
        g_centroid_flat = context.benign_centroid

        # The formula from your paper[cite: 530]:
        # g_attack = g_centroid - alpha * (g_victim - g_centroid)
        g_attack_flat = g_centroid_flat - self.attack_strength * \
                        (g_victim_flat - g_centroid_flat)

        return g_attack_flat

    def manipulate(self, *args, **kwargs) -> torch.Tensor:
        """
        This method is not used by the DrowningStrategy, as it replaces
        the gradient entirely, rather than manipulating it.
        """
        raise NotImplementedError(
            "DrowningStrategy uses calculate_repulsion_gradient, not manipulate."
        )


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
            'drowning': DrowningStrategy(
                SybilDrowningConfig(**self.sybil_cfg.strategy_configs.get('drowning', {}))
            ),
            'collusion': BuyerCollusionStrategy(noise_scale=1e-5)
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

        root_grad_flat = None
        if "collusion" in strategies_in_use or "buyer_collusion" in strategies_in_use:
            if current_root_gradient:
                root_grad_flat = self._ensure_tensor(current_root_gradient)
            elif self.selection_patterns and "centroid" in self.selection_patterns:
                # Fallback to historical centroid if root gradient missing
                root_grad_flat = self.selection_patterns["centroid"]
                logging.warning("[SybilCoordinator] Root gradient missing for collusion. Using historical centroid.")

        drowning_repulsion_grad: Optional[torch.Tensor] = None
        if "drowning" in strategies_in_use:
            # Type-check to ensure we have the right strategy object
            if isinstance(self.strategies.get('drowning'), DrowningStrategy):
                drowning_strategy: DrowningStrategy = self.strategies['drowning']
                victim_id = drowning_strategy.victim_id

                # 1. Find Victim Gradient
                victim_grad_list = manipulated_gradients.get(victim_id)
                if not victim_grad_list:
                    logging.warning(f"[SybilCoordinator] Drowning attack failed: Victim '{victim_id}' not found.")
                else:
                    # 2. Find Benign Centroid (EXCLUDING victim)
                    benign_grads_for_drowning = [
                        self._ensure_tensor(manipulated_gradients[sid])
                        for sid in benign_ids
                        if sid in manipulated_gradients and sid != victim_id
                    ]

                    if not benign_grads_for_drowning:
                        logging.warning(f"[SybilCoordinator] Drowning attack failed: No other benign gradients found.")
                    else:
                        benign_centroid_flat = torch.mean(torch.stack(benign_grads_for_drowning), dim=0)
                        victim_grad_flat = self._ensure_tensor(victim_grad_list)

                        # 3. Calculate the Repulsion Gradient
                        context = DrowningContext(victim_gradient=victim_grad_flat,
                                                  benign_centroid=benign_centroid_flat)
                        drowning_repulsion_grad = drowning_strategy.calculate_repulsion_gradient(context)

                        if drowning_repulsion_grad is not None:
                            logging.info(f"[SybilCoordinator] Calculated Drowning repulsion gradient.")
                        else:
                            logging.warning(f"[SybilCoordinator] DrowningStrategy failed to calculate gradient.")
            else:
                logging.error("[SybilCoordinator] 'drowning' strategy not correctly configured as DrowningStrategy.")

        # --- 2. Calculate Oracle Information (if needed) ---
        oracle_centroid_flat: Optional[torch.Tensor] = None
        if "oracle_blend" in strategies_in_use:
            benign_gradients_dict = {
                sid: manipulated_gradients[sid] for sid in benign_ids if sid in manipulated_gradients
            }
            if benign_gradients_dict and current_root_gradient is not None:  # Ensure root gradient is available
                logging.info("[SybilCoordinator] Running hypothetical aggregation for Oracle...")
                try:
                    _hypo_agg_grad, hypothetical_selected_ids, _hypo_outliers, _hypo_stats = self.aggregator.aggregate(
                        global_epoch=global_epoch,
                        seller_updates=benign_gradients_dict,
                        root_gradient=current_root_gradient,
                        buyer_data_loader=buyer_data_loader
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
                logging.info("[SybilCoordinator] Using historical centroid for mimic/pivot/knock_out.")
            else:
                logging.warning(
                    "[SybilCoordinator] Historical strategies requested but no historical centroid available yet.")

        # --- 4. Loop through ACTIVE Sybils and Manipulate ---
        manipulated_count = 0
        for sybil_id in active_sybils:
            client_state = self.clients[sybil_id]
            strategy_name = "oracle_blend" if "oracle_blend" in strategies_in_use else self._get_strategy_for_client(
                client_state)
            print(
                f"[SybilCoordinator] Current strategy_name: {strategy_name}, oracle_centroid_flat exist: {oracle_centroid_flat is not None}")

            original_malicious_grad_list = current_round_gradients[sybil_id]
            original_shapes = [g.shape for g in original_malicious_grad_list]
            original_malicious_flat = self._ensure_tensor(original_malicious_grad_list)

            manipulated_grad_flat: Optional[torch.Tensor] = None

            if strategy_name == "collusion":
                if root_grad_flat is not None:
                    # Note: Ensure key in self.strategies matches initialization
                    # If you initialized it as 'collusion', use 'collusion'.
                    # If 'buyer_collusion', use that.
                    strategy_key = 'collusion'
                    strategy_obj = self.strategies[strategy_key]

                    manipulated_grad_flat = strategy_obj.manipulate(
                        original_malicious_flat,
                        root_grad_flat
                    )
                    logging.info(f"   Buyer Collusion applied to {sybil_id}")
                else:
                    logging.warning(f"   Buyer Collusion failed (No root gradient). Submitting original.")
                    manipulated_grad_flat = original_malicious_flat
            elif strategy_name == "drowning":
                if drowning_repulsion_grad is not None:
                    # This Sybil's gradient is REPLACED with the common
                    # repulsion gradient, regardless of its original payload.
                    manipulated_grad_flat = drowning_repulsion_grad
                    logging.debug(f"   Drowning attack gradient REPLACED for {sybil_id}")
                else:
                    logging.warning(
                        f"   Drowning attack for {sybil_id} failed (no repulsion gradient). Submitting original.")
                    manipulated_grad_flat = original_malicious_flat  # Fallback
            # --- Apply Oracle Blend ---
            elif strategy_name == "oracle_blend":
                if oracle_centroid_flat is not None:
                    alpha = self.sybil_cfg.oracle_blend_alpha  # Get alpha from config
                    manipulated_grad_flat = alpha * original_malicious_flat + (1.0 - alpha) * oracle_centroid_flat
                    logging.info(f"   Oracle Blending applied to {sybil_id} (alpha={alpha})")
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

                expected_numel = sum(torch.prod(torch.tensor(s)) for s in original_shapes)
                if manipulated_grad_flat.numel() != expected_numel:
                    logging.error(
                        f"   FATAL: Manipulated gradient for {sybil_id} (strategy '{strategy_name}') "
                        f"has wrong numel ({manipulated_grad_flat.numel()}) vs expected ({expected_numel}). "
                        f"Submitting original."
                    )
                    manipulated_grad_list = original_malicious_grad_list  # Fallback
                else:
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
        selected_benign_gradients = {}
        for sid in selected_ids:
            # Check if this ID is for a seller that is NOT a Sybil
            if sid not in self.clients:
                seller = all_sellers.get(sid)

                # Get the gradient this benign seller just submitted
                # (This gradient was cached in the GradientSeller class)
                if seller and hasattr(seller, 'last_computed_gradient'):
                    benign_grad = seller.last_computed_gradient
                    if benign_grad:
                        # Ensure it's a flat tensor for the pattern analysis
                        selected_benign_gradients[sid] = self._ensure_tensor(benign_grad)

        # Update historical patterns for future rounds
        if selected_benign_gradients:
            # Pass the DICTIONARY of flat tensors
            self.update_historical_patterns(selected_benign_gradients)
            logging.info(f"   Updated patterns with {len(selected_benign_gradients)} benign gradients")
        else:
            logging.warning("   No selected benign gradients found to update historical patterns.")
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

            if self.start_atk:
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
        """Analyze stored selected gradients to compute a centroid..."""
        all_selected_flat = [
            flat_grad  # <-- FIX: grad is ALREADY the flat tensor
            for round_dict in self.selected_history
            for flat_grad in round_dict.values()
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

    def prepare_for_new_round(self) -> None:
        """Prepares state for the next round and handles dynamic triggers."""
        self.cur_round += 1
        if self.cur_round >= self.sybil_cfg.benign_rounds:
            self.start_atk = True

    def on_round_end(self) -> None:
        """Clear round-specific state."""
        self.adaptive_role_assignment()
        self.selected_gradients = {}


class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, poison_generator, poison_rate, data_format='image'):
        self.original_dataset = original_dataset
        self.poison_generator = poison_generator
        self.data_format = data_format  # Track data format

        if not (0.0 <= poison_rate <= 1.0):
            raise ValueError("Poison rate must be between 0.0 and 1.0")

        # --- THIS IS THE FIX ---
        # The logic for _select_indices is now moved directly here.
        n_poison = int(len(original_dataset) * poison_rate)
        all_indices = list(range(len(original_dataset)))
        random.shuffle(all_indices)
        self.poison_indices = set(all_indices[:n_poison])
        # -----------------------

    def __len__(self):
        # Always return the full, original length.
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get the original data
        original_sample = self.original_dataset[idx]

        # Unpack based on format
        if self.data_format == 'text':
            label, data = original_sample
        else:  # image or tabular
            data, label = original_sample

        # Check if this specific index should be poisoned.
        if self.poison_generator and idx in self.poison_indices:
            # If yes, poison it
            poisoned_data, poisoned_label = self.poison_generator.apply(data, label)
            # Ensure label is a tensor for consistency
            return poisoned_data, torch.tensor(poisoned_label, dtype=torch.long)
        else:
            # If no, return the original data
            # Also ensure label is a tensor for consistency
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
                                 **kwargs: Any) -> Optional[PoisonGenerator]:
        poison_cfg = adv_cfg.poisoning

        if not poison_cfg:
            return None

        try:
            poison_type = PoisonType(poison_cfg.type)

            # Reuse the robust logic from AdvancedBackdoorAdversarySeller
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
                backdoor_text_cfg = BackdoorTextConfig(
                    vocab=vocab,
                    target_label=params.target_label,
                    trigger_content=params.trigger_content,
                    location=params.location
                )
                return BackdoorTextGenerator(backdoor_text_cfg)

            elif model_type == 'tabular':
                feature_to_idx = kwargs.get('feature_to_idx')
                main_params = poison_cfg.tabular_backdoor_params
                backdoor_tabular_cfg = BackdoorTabularConfig(
                    target_label=main_params.target_label,
                    trigger_conditions=main_params.feature_trigger_params.trigger_conditions
                )
                return BackdoorTabularGenerator(backdoor_tabular_cfg, feature_to_idx)

        except Exception as e:
            logging.error(f"AdaptiveAttacker factory failed: {e}")
            return None


class AdaptiveAttackerSeller(AdvancedPoisoningAdversarySeller):
    """
    Adaptive Adversary focused PURELY on maximizing Selection Rate.
    It uses a Bandit to learn the data distribution that the Aggregator prefers.
    """

    def __init__(self, seller_id: str, data_config: Any,
                 training_config: Any, model_factory: Callable[[], nn.Module],
                 adversary_config: Any,
                 model_type: str,
                 device: str = "cpu",
                 validation_loader: DataLoader = None,
                 # Kept for compatibility, though not strictly needed for Black Box
                 **kwargs):

        super().__init__(seller_id=seller_id, data_config=data_config,
                         training_config=training_config, model_factory=model_factory,
                         adversary_config=adversary_config, poison_generator=None,
                         device=device, **kwargs)

        self.adv_cfg = adversary_config.adaptive_attack
        self.model_type = model_type
        self.device = device
        self.kwargs = kwargs

        # Bandit State
        self.threat_model = "black_box"  # We are probing the black box aggregator
        self.phase = "exploration"
        self.strategy_history = collections.deque(maxlen=200)
        self.round_counter = 0

        # --- SELECTION OPTIMIZATION STRATEGIES ---
        # 1. honest: Baseline behavior
        # 2. balance_classes: Fixes Non-IID skew (Looks like Global Mean)
        # 3. easy_samples: Reduces Variance (Looks Stable/High Confidence)
        self.base_strategies = ["honest", "balance_classes", "easy_samples"]

        # 4. Add Class Probing (Check if specific classes are preferred)
        self._add_class_based_strategies()

        logging.info(f"[{self.seller_id}] Initialized Reputation Farmer. Strategies: {len(self.base_strategies)}")

    def _add_class_based_strategies(self):
        """
        Adds strategies to focus on specific classes, BUT only if the client
        actually possesses data for that class.
        """
        try:
            targets = self._get_targets()

            if targets is not None:
                # 1. Identify which classes are actually present locally
                # (e.g., in Non-IID, this client might only have classes [0, 1, 2])
                present_classes = set(targets)

                # 2. Only add strategies for THOSE classes
                count = 0
                for c in present_classes:
                    self.base_strategies.append(f"focus_class_{c}")
                    count += 1

                logging.info(
                    f"[{self.seller_id}] Smart Init: Added {count} class-focus strategies (ignored missing classes).")
            else:
                logging.warning(f"[{self.seller_id}] Could not extract targets. Skipping class strategies.")

        except Exception as e:
            logging.warning(f"[{self.seller_id}] Error initializing class strategies: {e}")

    def get_gradient_for_upload(self, all_seller_gradients=None, target_seller_id=None):
        self.round_counter += 1
        stats = {}
        local_model = self.model_factory().to(self.device)

        # 1. Update Phase (Exploration -> Exploitation)
        if self.phase == "exploration" and self.round_counter > self.adv_cfg.exploration_rounds:
            self.phase = "exploitation"

        # 2. Bandit Selects Best Data Strategy
        self.current_strategy = self._select_black_box_strategy()
        stats['attack_strategy'] = self.current_strategy

        # 3. Apply Data Strategy (The Core Logic)
        # We manipulate the dataset to look "better" to the aggregator
        dataset_for_training = self._apply_black_box_data_strategy(self.current_strategy)

        # 4. Compute Gradient
        # We train normally, but on the manipulated (clean) data
        final_gradient, train_stats = self._compute_local_grad(local_model, dataset_for_training)

        # 5. Safety Check: Zero Gradients get banned
        if final_gradient is None or all(torch.norm(g) == 0 for g in final_gradient):
            logging.warning(
                f"[{self.seller_id}] Strategy '{self.current_strategy}' produced Zero Gradient. Fallback to honest.")
            final_gradient, train_stats = self._compute_local_grad(local_model, self.dataset)

        stats.update(train_stats)

        # Cache for history tracking
        self.last_computed_gradient = final_gradient
        self.last_training_stats = stats

        return final_gradient, stats

    def _select_black_box_strategy(self) -> str:
        # Standard UCB Bandit Logic
        # 1. Try everything once
        tried_strategies = set(hist[1] for hist in self.strategy_history)
        untried = [s for s in self.base_strategies if s not in tried_strategies]
        if untried: return random.choice(untried)

        # 2. Epsilon Greedy
        epsilon = 0.2 if self.phase == "exploration" else 0.05
        if random.random() < epsilon: return random.choice(self.base_strategies)

        # 3. Calculate Scores (UCB)
        strategy_stats = collections.defaultdict(lambda: {'attempts': 0, 'successes': 0})
        for _, strategy, selected in self.strategy_history:
            strategy_stats[strategy]['attempts'] += 1
            if selected: strategy_stats[strategy]['successes'] += 1

        total = sum(s['attempts'] for s in strategy_stats.values())
        best_score = -float('inf')
        best_strat = "honest"

        for strat in self.base_strategies:
            stats = strategy_stats[strat]
            if stats['attempts'] == 0: continue

            # Reward = Selection Rate
            avg_reward = stats['successes'] / stats['attempts']
            bonus = np.sqrt(2 * np.log(total) / stats['attempts'])
            ucb_score = avg_reward + 0.5 * bonus

            if ucb_score > best_score:
                best_score = ucb_score
                best_strat = strat

        return best_strat

    def _apply_black_box_data_strategy(self, strategy: str) -> Dataset:
        if strategy == "honest":
            return self.dataset

        # Hygiene Strategies
        elif strategy == "balance_classes":
            return self._create_balanced_subset()
        elif strategy == "easy_samples":
            return self._create_easy_subset()

        # Probing Strategies
        elif strategy.startswith("focus_class_"):
            return self._apply_class_filter_strategy(strategy)

        return self.dataset

    # --- HELPER METHODS (Same as before) ---

    def _create_balanced_subset(self):
        targets = self._get_targets()
        if targets is None: return self.dataset
        targets = np.array(targets)
        class_counts = np.bincount(targets)
        # Weight = 1 / Freq (Rare classes get picked more)
        weights = 1.0 / np.maximum(class_counts[targets], 1)

        # Oversample to ensure balanced batch
        num_samples = len(self.dataset)
        indices = torch.multinomial(torch.DoubleTensor(weights), num_samples, replacement=True).tolist()
        return Subset(self.dataset, indices)

    def _create_easy_subset(self):
        # Optimization: Only check a small subset to save compute
        subset_indices = random.sample(range(len(self.dataset)), min(len(self.dataset), 512))

        model = self.model_factory().to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = []

        temp_loader = DataLoader(Subset(self.dataset, subset_indices), batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch in temp_loader:
                if len(batch) == 2:
                    d, t = batch
                else:
                    _, d, t = batch
                l = criterion(model(d.to(self.device)), t.to(self.device))
                losses.extend(l.cpu().numpy())

        # Pick lowest loss samples (Cleanest data)
        sorted_args = np.argsort(losses)
        cutoff = int(len(sorted_args) * 0.5)
        final_indices = [subset_indices[i] for i in sorted_args[:cutoff]]
        return Subset(self.dataset, final_indices)

    def _apply_class_filter_strategy(self, strategy):
        try:
            c = int(strategy.rpartition('_')[2])
            targets = np.array(self._get_targets())
            indices = np.where(targets == c)[0]
            if len(indices) == 0: return self.dataset
            return Subset(self.dataset, indices.tolist())
        except:
            return self.dataset

    def _get_targets(self):
        if hasattr(self.dataset, 'targets'): return self.dataset.targets
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'targets'):
            return [self.dataset.dataset.targets[i] for i in self.dataset.indices]
        return None

    def round_end_process(self, round_number: int, was_selected: bool, was_outlier: bool = False,
                          marketplace_metrics: Dict = None, **kwargs):
        """
        Record outcome and update threat-model-specific state.
        Fixed signature to match GradientSeller parent class.
        """
        # 1. Update Bandit History (The specific Adaptive logic)
        # We record: (Round, Strategy Used, Success/Failure)
        self.strategy_history.append((round_number, self.current_strategy, was_selected))

        # 2. Call Parent (Standard logic)
        # Pass arguments explicitly by name to avoid positional mix-ups
        super().round_end_process(
            round_number=round_number,
            was_selected=was_selected,
            was_outlier=was_outlier,
            marketplace_metrics=marketplace_metrics,
            **kwargs
        )


# class AdaptiveAttackerSeller(AdvancedPoisoningAdversarySeller):
#     """
#     Adaptive adversary simulating three threat models.
#
#     This class combines a multi-armed bandit (black-box) to discover *which* strategy
#     to use, with a sophisticated "stealthy_blend" strategy that learns *how much*
#     malice to inject.
#
#     Key Improvements in this version:
#     1. Fixes UCB learning freeze (continues learning even after exploration phase).
#     2. Optimizes gradient computation (prevents double calculation for data attacks).
#     3. Clamps attack intensity to valid ranges.
#     """
#
#     def __init__(self, seller_id: str, data_config: Any,
#                  training_config: Any, model_factory: Callable[[], nn.Module],
#                  adversary_config: Any,
#                  model_type: str,
#                  device: str = "cpu",
#                  # --- FIX 1: Explicitly accept the argument to prevent crash ---
#                  validation_loader: DataLoader = None,
#                  **kwargs):
#
#         super().__init__(seller_id=seller_id, data_config=data_config,
#                          training_config=training_config, model_factory=model_factory,
#                          adversary_config=adversary_config, poison_generator=None,
#                          device=device, **kwargs)
#
#         # --- FIX 2: Store it so the Oracle attack can use it ---
#         self.validation_loader = validation_loader
#
#         # Fallback: Check kwargs just in case
#         if self.validation_loader is None and 'validation_loader' in kwargs:
#             self.validation_loader = kwargs['validation_loader']
#
#         self.adv_cfg = adversary_config.adaptive_attack
#         if not self.adv_cfg.is_active:
#             raise ValueError("AdaptiveAttackerSeller requires is_active=True")
#
#         self.model_type = model_type
#         self.kwargs = kwargs
#
#         self.threat_model = self.adv_cfg.threat_model
#
#         self.phase = "exploration"
#         self.strategy_history = collections.deque(maxlen=200)
#         self.current_strategy = "honest"
#         self.round_counter = 0
#
#         # --- FIX 3: Use the "Selection Boosting" Strategy Pool ---
#         if self.adv_cfg.attack_mode == "gradient_manipulation":
#             # Removed 'add_noise' because it gets rejected by MartFL
#             self.base_strategies = ["honest", "reduce_norm", "stealthy_blend"]
#
#         elif self.adv_cfg.attack_mode == "data_poisoning":
#             # Removed 'subsample_clean' because it creates high variance (rejection)
#             # Added 'balance_classes' and 'easy_samples' to look like the Global Mean (selection)
#             self.base_strategies = ["honest", "balance_classes", "easy_samples"]
#             self._add_class_based_strategies()
#
#         # ... (Rest of resources remains the same) ...
#         self.previous_centroid = None
#         self.previous_aggregate = None
#         self.blend_cfg = adversary_config.drowning_attack
#         self.blend_phase = "mimicry"
#         self.blend_mimicry_rounds = self.blend_cfg.mimicry_rounds
#         self.blend_attack_intensity = self.blend_cfg.attack_intensity
#         self.blend_honest_gradient_stats = {'mean_norm': None, 'direction_estimate': None}
#
#         self.backdoor_dataset: Optional[Dataset] = None
#         self.poisoned_dataset: Optional[Dataset] = None
#         self.layer_name_to_index: Dict[str, int] = {}
#         self.target_layer_indices: Set[int] = set()
#
#         self._initialize_malicious_resources()
#
#         logging.info(f"[{self.seller_id}] Initialized AdaptiveAttacker ({self.threat_model})")
#
#     def _initialize_malicious_resources(self):
#         """Helper to setup malicious datasets and layer maps."""
#         # 1. Create Malicious Datasets
#         try:
#             # We create the generator based on the MAIN poisoning config
#             malicious_generator = self._create_poison_generator(
#                 self.adversary_config, self.model_type, self.device, **self.kwargs
#             )
#
#             if malicious_generator:
#                 poison_type_enum = self.adversary_config.poisoning.type
#
#                 # FIX: Check the POISON TYPE directly, not the blend config
#                 if "backdoor" in poison_type_enum.value:
#                     logging.info(f"[{self.seller_id}] Initializing BACKDOOR dataset.")
#                     self.backdoor_dataset = PoisonedDataset(
#                         original_dataset=self.dataset,
#                         poison_generator=malicious_generator,
#                         poison_rate=1.0,  # Backdoors target 100% of trigger inputs
#                         data_format=self.model_type
#                     )
#                 else:
#                     # Targeted / Untargeted / Label Flip
#                     logging.info(f"[{self.seller_id}] Initializing POISONED dataset.")
#                     self.poisoned_dataset = PoisonedDataset(
#                         original_dataset=self.dataset,
#                         poison_generator=malicious_generator,
#                         poison_rate=self.adversary_config.poisoning.poison_rate,
#                         data_format=self.model_type
#                     )
#             else:
#                 logging.warning(f"[{self.seller_id}] No malicious generator created.")
#
#         except Exception as e:
#             logging.warning(f"[{self.seller_id}] Malicious dataset setup failed: {e}", exc_info=True)
#
#         # 2. Map Layers (Keep existing logic)
#         try:
#             temp_model = self.model_factory()
#             for i, (name, _) in enumerate(temp_model.named_parameters()):
#                 self.layer_name_to_index[name] = i
#             if self.blend_cfg.target_layers:
#                 for layer_name in self.blend_cfg.target_layers:
#                     if layer_name in self.layer_name_to_index:
#                         self.target_layer_indices.add(self.layer_name_to_index[layer_name])
#         except Exception:
#             pass
#
#     def _add_class_based_strategies(self):
#         """Adds strategies like 'focus_class_0', 'exclude_class_1' to the pool."""
#         try:
#             # Detect number of classes
#             if hasattr(self.dataset, 'classes'):
#                 num_classes = len(self.dataset.classes)
#             elif hasattr(self.dataset, 'targets'):
#                 num_classes = len(set(self.dataset.targets))
#             elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'targets'):
#                 # Handle Subset
#                 targets = [self.dataset.dataset.targets[i] for i in self.dataset.indices]
#                 num_classes = len(set(targets))
#             else:
#                 return  # Can't determine classes
#
#             # Add strategies for every single class
#             # The bandit will find which specific class the Aggregator prefers!
#             for c in range(num_classes):
#                 self.base_strategies.append(f"focus_class_{c}")
#                 # Exclude might be useful if one class is "noisy" and causing rejection
#                 self.base_strategies.append(f"exclude_class_{c}")
#
#         except Exception as e:
#             logging.warning(f"Could not add class strategies: {e}")
#
#     # ========================================================================
#     # THREAT MODELS 1 & 2: ORACLE / GRADIENT INVERSION
#     # ========================================================================
#     def _apply_oracle_attack(self, malicious_gradient: List[torch.Tensor]) -> List[torch.Tensor]:
#         if self.validation_loader is None:
#             logging.error(f"[{self.seller_id}] Oracle attack requires validation_loader!")
#             return malicious_gradient
#
#         # 1. Get True Honest Direction (from Validation Data)
#         # Note: We create a temporary model to avoid mutating the main one
#         temp_model = self.model_factory().to(self.device)
#         true_honest_grad, _ = self._compute_local_grad(
#             temp_model,
#             self.validation_loader.dataset
#         )
#
#         honest_flat = flatten_tensor(true_honest_grad)
#         malicious_flat = flatten_tensor(malicious_gradient)
#
#         # 2. Constraint: Target Cosine Similarity
#         target_similarity = 0.95
#
#         h_norm = torch.norm(honest_flat)
#         h_unit = honest_flat / (h_norm + 1e-9)
#
#         # 3. Geometric Projection ("Boundary Riding")
#         # Decompose malicious vector into Parallel (honest) and Orthogonal components
#         parallel_component = torch.dot(malicious_flat, h_unit) * h_unit
#         orthogonal_component = malicious_flat - parallel_component
#
#         # Edge Case: Malicious is already perfectly aligned with Honest
#         if torch.norm(orthogonal_component) < 1e-9:
#             return true_honest_grad
#
#         orth_unit = orthogonal_component / torch.norm(orthogonal_component)
#
#         # 4. Construct the vector at exactly 'theta' degrees
#         import math
#         # Clamp to avoid domain errors if target_similarity > 1.0 due to float precision
#         theta = math.acos(min(max(target_similarity, -1.0), 1.0))
#
#         # New Direction = (Cos(theta) * Honest_Dir) + (Sin(theta) * Malicious_Orth_Dir)
#         boundary_dir = (target_similarity * h_unit) + (math.sin(theta) * orth_unit)
#
#         # Scale: Mimic the honest norm to avoid norm-clipping defenses
#         final_flat = boundary_dir * h_norm
#
#         # --- FIX IS HERE: Use 'malicious_gradient' for shapes ---
#         return unflatten_tensor(final_flat, [p.shape for p in malicious_gradient])
#
#     def _apply_gradient_inversion_attack(self, _unused_gradient_input) -> List[torch.Tensor]:
#         """
#         Gradient Inversion: Simulates having reconstructed honest data.
#         We mix the LOSS, not the gradients. This creates a much more natural vector.
#         """
#         # 1. IDENTIFY MALICIOUS DATASET
#         # We need actual data to calculate the 'malicious loss'.
#         # Check backdoor first, then poisoned (e.g. targeted poisoning).
#         malicious_dataset = self.backdoor_dataset
#         if malicious_dataset is None:
#             malicious_dataset = self.poisoned_dataset
#
#         # SAFETY CHECK: If no malicious data is configured/loaded, we cannot attack.
#         if malicious_dataset is None:
#             logging.warning(f"[{self.seller_id}] GradInv skipped: No malicious dataset found. Returning honest grad.")
#             # Fallback: Compute honest gradient
#             model = self.model_factory().to(self.device)
#             grad, _ = self._compute_local_grad(model, self.dataset)
#             return grad
#
#         # 2. EXTRA KNOWLEDGE: Get 'Inverted' (Honest) Data
#         # In a real attack, this comes from the reconstruction algorithm.
#         try:
#             # Safe iterator handling
#             inverted_iter = iter(self.train_loader)
#             inverted_batch_data, inverted_batch_label = next(inverted_iter)
#         except StopIteration:
#             # Handle edge case of empty loader
#             return self._compute_local_grad(self.model_factory().to(self.device), self.dataset)[0]
#
#         inverted_batch_data = inverted_batch_data.to(self.device)
#         inverted_batch_label = inverted_batch_label.to(self.device)
#
#         model = self.model_factory().to(self.device)
#         model.train()
#
#         # 3. Compute Honest Loss on Inverted Data
#         outputs_honest = model(inverted_batch_data)
#         loss_honest = nn.functional.cross_entropy(outputs_honest, inverted_batch_label)
#
#         # 4. Compute Malicious Loss on Malicious Data
#         # [FIX] Use the safely resolved 'malicious_dataset'
#         try:
#             # Note: Creating a DataLoader inside a loop is inefficient but functional for small batch attacks.
#             # Using shuffle=True ensures we don't just overfit to the first 32 examples.
#             bd_iter = iter(DataLoader(malicious_dataset, batch_size=32, shuffle=True))
#             bd_data, bd_label = next(bd_iter)
#         except Exception as e:
#             logging.error(f"[{self.seller_id}] Failed to load malicious batch: {e}")
#             return self._compute_local_grad(model, self.dataset)[0]
#
#         bd_data = bd_data.to(self.device)
#         bd_label = bd_label.to(self.device)
#
#         outputs_mal = model(bd_data)
#         loss_mal = nn.functional.cross_entropy(outputs_mal, bd_label)
#
#         # 5. Joint Optimization (The "Shadow Imitator")
#         # We minimize: Loss_Honest + lambda * Loss_Malicious
#         lambda_val = 2.0
#         total_loss = loss_honest + lambda_val * loss_mal
#
#         total_loss.backward()
#
#         grad_list = [p.grad.clone() for p in model.parameters()]
#         return grad_list
#
#     def _select_black_box_strategy(self) -> str:
#         """Selects a strategy using UCB1 with epsilon-greedy fallback."""
#
#         # 1. Ensure all strategies are tried at least once
#         tried_strategies = set(hist[1] for hist in self.strategy_history)
#         untried = [s for s in self.base_strategies if s not in tried_strategies]
#         if untried:
#             return random.choice(untried)
#
#         # 2. Epsilon-Greedy exploration (prevents getting stuck in local optima)
#         # Higher epsilon in exploration phase, lower in exploitation
#         epsilon = 0.2 if self.phase == "exploration" else 0.05
#         if random.random() < epsilon:
#             return random.choice(self.base_strategies)
#
#         # 3. UCB Calculation
#         strategy_stats = collections.defaultdict(lambda: {'attempts': 0, 'successes': 0})
#         for _, strategy, selected in self.strategy_history:
#             strategy_stats[strategy]['attempts'] += 1
#             if selected:
#                 strategy_stats[strategy]['successes'] += 1
#
#         total_attempts = sum(s['attempts'] for s in strategy_stats.values())
#         if total_attempts == 0:
#             return "honest"
#
#         best_score = -float('inf')
#         best_strategy = "honest"
#
#         for strategy in self.base_strategies:
#             stats = strategy_stats[strategy]
#             if stats['attempts'] == 0:
#                 continue
#
#             avg_reward = stats['successes'] / stats['attempts']
#             # UCB1 Exploration Term
#             exploration_bonus = np.sqrt(2 * np.log(total_attempts) / stats['attempts'])
#             # 0.5 is the exploration constant (tunable)
#             ucb_score = avg_reward + 0.5 * exploration_bonus
#
#             if ucb_score > best_score:
#                 best_score = ucb_score
#                 best_strategy = strategy
#
#         logging.debug(f"[{self.seller_id}] UCB Selected '{best_strategy}' (Score: {best_score:.3f})")
#         return best_strategy
#
#     def _apply_black_box_data_strategy(self, strategy: str) -> Dataset:
#         """Apply data manipulation based on learned strategy."""
#         if strategy == "honest":
#             return self.dataset
#         elif strategy == "balance_classes":
#             return self._create_balanced_subset()  # Helper function call
#
#         # 3. HYGIENE: EASY SAMPLES (Variance Reduction)
#         elif strategy == "easy_samples":
#             return self._create_easy_subset()  # Helper function call
#         elif strategy == "subsample_clean":
#             subset_ratio = self.adv_cfg.subset_ratio
#             subset_size = max(1, int(len(self.dataset) * subset_ratio))
#             indices = random.sample(range(len(self.dataset)), subset_size)
#             return Subset(self.dataset, indices)
#         elif strategy.startswith("focus_class_") or strategy.startswith("exclude_class_"):
#             return self._apply_class_filter_strategy(strategy)
#         else:
#             return self.dataset
#
#     def _create_balanced_subset(self):
#         # (Logic from previous response)
#         targets = self._get_targets()
#         if targets is None: return self.dataset
#
#         targets = np.array(targets)
#         class_counts = np.bincount(targets)
#         weights = 1.0 / np.maximum(class_counts[targets], 1)
#
#         num_samples = len(self.dataset)
#         balanced_indices = torch.multinomial(torch.DoubleTensor(weights), num_samples, replacement=True)
#         return Subset(self.dataset, balanced_indices.tolist())
#
#     def _create_easy_subset(self):
#         # (Logic from previous response)
#         # Random sample to save time
#         subset_indices = random.sample(range(len(self.dataset)), min(len(self.dataset), 512))
#
#         model = self.model_factory().to(self.device)
#         model.eval()
#         criterion = nn.CrossEntropyLoss(reduction='none')
#         losses = []
#
#         # Quick eval
#         temp_loader = DataLoader(Subset(self.dataset, subset_indices), batch_size=64, shuffle=False)
#         with torch.no_grad():
#             for batch in temp_loader:
#                 if len(batch) == 2:
#                     data, target = batch
#                 else:
#                     _, data, target = batch
#                 data, target = data.to(self.device), target.to(self.device)
#                 loss = criterion(model(data), target)
#                 losses.extend(loss.cpu().numpy())
#
#         # Take bottom 50%
#         sorted_args = np.argsort(losses)
#         cutoff = int(len(sorted_args) * 0.5)
#         final_indices = [subset_indices[i] for i in sorted_args[:cutoff]]
#         return Subset(self.dataset, final_indices)
#
#     def _apply_class_filter_strategy(self, strategy: str) -> Dataset:
#         """Filter dataset by class."""
#         try:
#             action, _, class_label = strategy.rpartition('_')
#             class_label = int(class_label)
#
#             # Determine targets based on dataset type
#             if hasattr(self.dataset, 'targets'):
#                 targets = np.array(self.dataset.targets)
#             elif isinstance(self.dataset, Subset):
#                 base_targets = np.array(self.dataset.dataset.targets)
#                 targets = base_targets[self.dataset.indices]
#             else:
#                 targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
#
#             if action == "focus_class":
#                 indices = np.where(targets == class_label)[0]
#             else:  # exclude_class
#                 indices = np.where(targets != class_label)[0]
#
#             if len(indices) == 0:
#                 return self.dataset
#             return Subset(self.dataset, indices.tolist())
#         except Exception:
#             return self.dataset
#
#     def _apply_black_box_gradient_manipulation(self, gradient: List[torch.Tensor], strategy: str) -> List[torch.Tensor]:
#         """Applies simple black-box manipulations."""
#         if strategy == "honest":
#             return gradient
#
#         flat_grad = flatten_tensor(gradient).clone().detach()
#
#         if strategy == "add_noise":
#             noise = torch.randn_like(flat_grad)
#             scaled_noise = noise * self.adv_cfg.noise_level * torch.norm(flat_grad) / (torch.norm(noise) + 1e-9)
#             manipulated_flat_grad = flat_grad + scaled_noise
#         elif strategy == "reduce_norm":
#             manipulated_flat_grad = flat_grad * self.adv_cfg.scale_factor
#         else:
#             return gradient
#
#         return unflatten_tensor(manipulated_flat_grad, [p.shape for p in gradient])
#
#     # ========================================================================
#     # HELPER METHODS FOR "STEALTHY_BLEND"
#     # ========================================================================
#
#     @staticmethod
#     def _create_poison_generator(adv_cfg, model_type, device, **kwargs):
#         # Implementation depends on your PoisonGenerator class structure
#         # Returning None as placeholder if not defined
#         if hasattr(adv_cfg, 'poisoning') and adv_cfg.poisoning:
#             # Logic to instantiate BackdoorImageGenerator etc. goes here
#             # based on previous snippets provided in context
#             return None
#         return None
#
#     def _update_honest_gradient_stats(self, gradient: List[torch.Tensor]):
#         """Learn characteristics of honest gradients (norm EMA)."""
#         flat_grad = flatten_tensor(gradient)
#         grad_norm = torch.norm(flat_grad).item()
#
#         if self.blend_honest_gradient_stats['mean_norm'] is None:
#             self.blend_honest_gradient_stats['mean_norm'] = grad_norm
#         else:
#             beta = 0.9
#             self.blend_honest_gradient_stats['mean_norm'] = (
#                     beta * self.blend_honest_gradient_stats['mean_norm'] +
#                     (1 - beta) * grad_norm
#             )
#
#     def _get_targets(self):
#         # Robust helper to get targets from any dataset wrapper
#         if hasattr(self.dataset, 'targets'): return self.dataset.targets
#         if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'targets'):
#             return [self.dataset.dataset.targets[i] for i in self.dataset.indices]
#         return None
#
#     def _compute_malicious_gradient(self) -> Optional[List[torch.Tensor]]:
#         """Compute gradient on malicious objective."""
#         model = self.model_factory().to(self.device)
#         if self.blend_cfg.attack_type == "backdoor" and self.backdoor_dataset:
#             backdoor_grad, _ = self._compute_local_grad(model, self.backdoor_dataset)
#             return backdoor_grad
#         elif self.blend_cfg.attack_type == "targeted_poisoning" and self.poisoned_dataset:
#             poison_grad, _ = self._compute_local_grad(model, self.poisoned_dataset)
#             return poison_grad
#         return None
#
#     def _identify_vulnerable_layers(self, gradient: List[torch.Tensor]) -> List[int]:
#         """Identify which layers to inject malicious gradients into."""
#         if self.target_layer_indices:
#             return list(self.target_layer_indices)
#         num_layers = len(gradient)
#         num_target = max(1, int(num_layers * 0.2))
#         layer_norms = [torch.norm(g).item() for g in gradient]
#         sorted_indices = sorted(range(num_layers), key=lambda i: layer_norms[i], reverse=True)
#         return sorted_indices[:num_target]
#
#     def _create_stealthy_malicious_gradient(self, honest_gradient, malicious_gradient):
#         """Blend honest and malicious gradients."""
#         honest_flat = flatten_tensor(honest_gradient)
#         malicious_flat = flatten_tensor(malicious_gradient)
#
#         if honest_flat.numel() != malicious_flat.numel():
#             return honest_gradient
#
#         target_norm = self.blend_honest_gradient_stats.get('mean_norm') or torch.norm(honest_flat).item()
#
#         if self.blend_cfg.replacement_strategy == "layer_wise":
#             vulnerable_layers = self._identify_vulnerable_layers(honest_gradient)
#             blended_gradient = []
#             for i, (h_grad, m_grad) in enumerate(zip(honest_gradient, malicious_gradient)):
#                 if i in vulnerable_layers:
#                     alpha = self.blend_attack_intensity
#                     layer_blend = (1 - alpha) * h_grad + alpha * m_grad
#                     blended_gradient.append(layer_blend)
#                 else:
#                     blended_gradient.append(h_grad.clone())
#
#             # Renormalize entire vector
#             blended_flat = flatten_tensor(blended_gradient)
#             scale = target_norm / (torch.norm(blended_flat).item() + 1e-9)
#             return [g * scale for g in blended_gradient]
#
#         elif self.blend_cfg.replacement_strategy == "global_blend":
#             alpha = self.blend_attack_intensity
#             blended_flat = (1 - alpha) * honest_flat + alpha * malicious_flat
#             blended_flat = blended_flat * (target_norm / (torch.norm(blended_flat) + 1e-9))
#             return unflatten_tensor(blended_flat, [g.shape for g in honest_gradient])
#
#         return honest_gradient
#
#     # ========================================================================
#     # MAIN PIPELINE: GENERATE GRADIENT
#     # ========================================================================
#
#     def get_gradient_for_upload(self, all_seller_gradients=None, target_seller_id=None):
#         self.round_counter += 1
#         stats = {}
#         local_model = self.model_factory().to(self.device)
#
#         # 1. Determine Phase and Strategy
#         if self.phase == "exploration" and self.round_counter > self.adv_cfg.exploration_rounds:
#             self.phase = "exploitation"
#
#         if self.threat_model == "black_box":
#             self.current_strategy = self._select_black_box_strategy()
#         elif self.threat_model == "oracle":
#             self.current_strategy = "oracle_specific"
#         elif self.threat_model == "gradient_inversion":
#             self.current_strategy = "gradient_inversion_specific"
#         else:
#             self.current_strategy = "honest"
#
#         stats['attack_strategy'] = self.current_strategy
#
#         # 2. OPTIMIZATION: Black-Box Data Poisoning
#         # If strategy changes data, compute on that data directly and return.
#         if self.adv_cfg.attack_mode == "data_poisoning" and self.threat_model == "black_box":
#             dataset_for_training = self._apply_black_box_data_strategy(self.current_strategy)
#
#             final_gradient, train_stats = self._compute_local_grad(local_model, dataset_for_training)
#             stats.update(train_stats)
#
#             self.last_computed_gradient = final_gradient
#             self.last_training_stats = stats
#             return final_gradient, stats
#
#         # 3. Base Computation: Honest Gradient
#         # We compute this ONCE. It serves as the "Mask" for attacks or the fallback.
#         honest_gradient, train_stats = self._compute_local_grad(local_model, self.dataset)
#         stats.update(train_stats)
#
#         if honest_gradient is None:
#             stats['error'] = 'Honest gradient computation failed'
#             return None, stats
#
#         final_gradient = honest_gradient  # Default fallback
#
#         # 4. Apply Advanced Attacks
#         if self.threat_model == "oracle":
#             # [FIXED LOGIC]
#             # 1. Compute the MALICIOUS target (Poison/Backdoor)
#             malicious_gradient = self._compute_malicious_gradient()
#
#             if malicious_gradient is not None:
#                 # 2. Project the MALICIOUS target onto the Honest Boundary
#                 final_gradient = self._apply_oracle_attack(malicious_gradient)
#             else:
#                 logging.warning(f"[{self.seller_id}] Oracle attack missing malicious dataset. Sending honest.")
#                 final_gradient = honest_gradient
#
#         elif self.threat_model == "gradient_inversion":
#             # Pass None because this method calculates gradients via Joint Loss Optimization
#             final_gradient = self._apply_gradient_inversion_attack(None)
#
#         elif self.threat_model == "black_box":
#             # Gradient Manipulation Mode
#             if self.current_strategy == "stealthy_blend":
#                 stats['blend_phase'] = self.blend_phase
#
#                 # Check phase transition
#                 if self.blend_phase == "mimicry" and self.round_counter > self.blend_mimicry_rounds:
#                     self.blend_phase = "attack"
#                     logging.info(f"[{self.seller_id}] Blend strategy entering ATTACK phase.")
#
#                 if self.blend_phase == "mimicry":
#                     self._update_honest_gradient_stats(honest_gradient)
#                     final_gradient = honest_gradient
#                 else:
#                     malicious_gradient = self._compute_malicious_gradient()
#                     if malicious_gradient:
#                         final_gradient = self._create_stealthy_malicious_gradient(
#                             honest_gradient, malicious_gradient
#                         )
#             else:
#                 # Standard manipulations (reduce_norm, etc.)
#                 final_gradient = self._apply_black_box_gradient_manipulation(
#                     honest_gradient, self.current_strategy
#                 )
#
#         # Cache and Return
#         stats.update({
#             'threat_model': self.threat_model,
#             'attack_phase': self.phase
#         })
#         self.last_computed_gradient = final_gradient
#         self.last_training_stats = stats
#         return final_gradient, stats
#
#     # ========================================================================
#     # FEEDBACK LOOP
#     # ========================================================================
#
#     def round_end_process(self, round_number: int, was_selected: bool,
#                           marketplace_metrics: Dict = None, **kwargs):
#         """Record outcome and update threat-model-specific state."""
#         super().round_end_process(round_number=round_number, was_selected=was_selected,
#                                   marketplace_metrics=marketplace_metrics, **kwargs)
#
#         # 1. Update UCB Bandit (CRITICAL: Always update, regardless of phase)
#         if self.threat_model == "black_box":
#             self.strategy_history.append((round_number, self.current_strategy, was_selected))
#             logging.debug(f"[{self.seller_id}] History: {self.current_strategy} -> {was_selected}")
#
#         # 2. Update "Stealthy Blend" Intensity
#         if self.current_strategy == "stealthy_blend" and self.blend_phase == "attack":
#             if not was_selected:
#                 # Rejected? Reduce intensity significantly
#                 self.blend_attack_intensity = max(0.01, self.blend_attack_intensity * 0.9)
#                 logging.info(f"[{self.seller_id}] Blend failed. Reduced to {self.blend_attack_intensity:.3f}")
#             else:
#                 # Accepted? Increase intensity slightly to push limits
#                 self.blend_attack_intensity = min(0.99, self.blend_attack_intensity * 1.05)
#
#         # 3. Update Oracle/GradInv Snapshots
#         if marketplace_metrics:
#             if self.threat_model == "oracle" and 'selected_centroid_flat' in marketplace_metrics:
#                 self.previous_centroid = marketplace_metrics['selected_centroid_flat'].clone().detach().cpu()
#             if self.threat_model == "gradient_inversion" and 'final_aggregated_gradient_flat' in marketplace_metrics:
#                 self.previous_aggregate = marketplace_metrics['final_aggregated_gradient_flat'].clone().detach().cpu()


class DrowningAttackerSeller(GradientSeller):
    """
    Stealthy gradient manipulation attack that maintains selection probability
    while subtly pulling the aggregate toward a malicious objective.
    """

    def __init__(self,
                 seller_id: str,  # <-- Added missing *args from parent
                 data_config: RuntimeDataConfig,  # <-- FIX: Added data_config
                 training_config: TrainingConfig,  # <-- Added missing *args
                 model_factory: Callable[[], nn.Module],  # <-- Added missing *args
                 adversary_config: AdversarySellerConfig,
                 model_type: str,  # <-- FIX: Added model_type
                 device: str = "cpu",  # <-- Added missing *args
                 **kwargs: Any):

        # <-- FIX: Call parent __init__ with all required args
        super().__init__(
            seller_id=seller_id,
            data_config=data_config,
            training_config=training_config,
            model_factory=model_factory,
            device=device,
            **kwargs
        )

        self.adv_cfg = adversary_config.gradient_replacement_attack
        if not self.adv_cfg.is_active:
            raise ValueError("DrowningAttackerSeller requires is_active=True")

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
        self.attack_intensity = self.adv_cfg.attack_intensity  # e.g., 0.1

        # --- FIX 1: Initialize Malicious Datasets ---
        self.backdoor_dataset: Optional[Dataset] = None
        self.poisoned_dataset: Optional[Dataset] = None
        self.adversary_config = adversary_config  # Store full config

        try:
            # Create a poison generator based on the *full* adversary config
            # This generator defines our malicious objective
            malicious_generator = self._create_poison_generator(
                self.adversary_config, model_type, device, **kwargs
            )

            # Create the specific datasets this attack will use
            if self.adv_cfg.attack_type == "backdoor":
                # For backdoor, we want a gradient *only* on the backdoor task
                self.backdoor_dataset = PoisonedDataset(
                    original_dataset=self.dataset,
                    poison_generator=malicious_generator,
                    poison_rate=1.0,  # <-- 100% poisoning for the task
                    data_format=model_type  # <-- Pass model_type
                )
                logging.info(f"[{self.seller_id}] Created backdoor dataset.")

            elif self.adv_cfg.attack_type == "targeted_poisoning":
                # For targeted poisoning, we use the poison rate from the config
                self.poisoned_dataset = PoisonedDataset(
                    original_dataset=self.dataset,
                    poison_generator=malicious_generator,
                    poison_rate=self.adversary_config.poisoning.poison_rate,
                    data_format=model_type  # <-- Pass model_type
                )
                logging.info(f"[{self.seller_id}] Created targeted poisoning dataset.")

        except Exception as e:
            logging.warning(
                f"[{self.seller_id}] Could not create malicious dataset for attack type "
                f"'{self.adv_cfg.attack_type}': {e}. Attack may fail."
            )

        # --- FIX 2: Map Target Layer Names to Indices ---
        self.layer_name_to_index: Dict[str, int] = {}
        self.target_layer_indices: Set[int] = set()

        # Instantiate a temporary model to read its parameter names
        try:
            temp_model = self.model_factory()
            for i, (name, _) in enumerate(temp_model.named_parameters()):
                self.layer_name_to_index[name] = i
            del temp_model  # Free memory

            if self.adv_cfg.target_layers:
                for layer_name in self.adv_cfg.target_layers:
                    if layer_name in self.layer_name_to_index:
                        idx = self.layer_name_to_index[layer_name]
                        self.target_layer_indices.add(idx)
                    else:
                        logging.warning(
                            f"[{self.seller_id}] Target layer '{layer_name}' "
                            f"not found in model parameters."
                        )
                logging.info(
                    f"[{self.seller_id}] Mapped target layers to indices: "
                    f"{self.target_layer_indices}"
                )
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to map layer names: {e}")

        logging.info(f"[{self.seller_id}] Initialized DrowningAttacker")
        logging.info(f"  Mimicry Rounds: {self.adv_cfg.mimicry_rounds}")
        logging.info(f"  Target Layers: {self.adv_cfg.target_layers}")
        logging.info(f"  Attack Intensity: {self.attack_intensity}")

    # --- FIX 1 (HELPER): Added this static method ---
    @staticmethod
    def _create_poison_generator(adv_cfg: AdversarySellerConfig, model_type: str, device: str,
                                 **kwargs: Any) -> Optional[PoisonGenerator]:
        """Factory method to create the correct backdoor generator from configuration."""
        poison_cfg = adv_cfg.poisoning
        if not poison_cfg:
            logging.warning("No 'poisoning' config found in AdversarySellerConfig.")
            return None

        try:
            poison_type = PoisonType(poison_cfg.type)
        except ValueError:
            logging.error(f"Invalid poison type in config: '{poison_cfg.type}'.")
            return None

        try:
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
                    raise ValueError("Text backdoor generator requires 'vocab'.")

                backdoor_text_cfg = BackdoorTextConfig(
                    vocab=vocab,
                    target_label=params.target_label,
                    trigger_content=params.trigger_content,
                    location=params.location
                )
                return BackdoorTextGenerator(backdoor_text_cfg)

            elif model_type == 'tabular':
                main_params = poison_cfg.tabular_backdoor_params
                trigger_params = main_params.feature_trigger_params
                feature_to_idx = kwargs.get('feature_to_idx')
                if not feature_to_idx:
                    raise ValueError("Tabular backdoor requires 'feature_to_idx'.")

                backdoor_tabular_cfg = BackdoorTabularConfig(
                    target_label=main_params.target_label,
                    trigger_conditions=trigger_params.trigger_conditions
                )
                return BackdoorTabularGenerator(backdoor_tabular_cfg, feature_to_idx)

            else:
                raise ValueError(f"Unsupported model_type for backdoor: {model_type}")

        except Exception as e:
            logging.error(f"Failed to create poison generator: {e}", exc_info=True)
            return None

    def _update_honest_gradient_stats(self, gradient: List[torch.Tensor]):
        """Learn characteristics of honest gradients during mimicry."""
        # ... (this function was already correct) ...
        flat_grad = flatten_tensor(gradient)
        grad_norm = torch.norm(flat_grad).item()

        if self.honest_gradient_stats['mean_norm'] is None:
            self.honest_gradient_stats['mean_norm'] = grad_norm
        else:
            beta = 0.9
            self.honest_gradient_stats['mean_norm'] = (
                    beta * self.honest_gradient_stats['mean_norm'] +
                    (1 - beta) * grad_norm
            )

        normalized_grad = flat_grad / (grad_norm + 1e-9)
        if self.honest_gradient_stats['direction_estimate'] is None:
            self.honest_gradient_stats['direction_estimate'] = normalized_grad.detach().cpu()
        else:
            beta = 0.9
            current_estimate = self.honest_gradient_stats['direction_estimate'].to(self.device)
            updated = beta * current_estimate + (1 - beta) * normalized_grad
            self.honest_gradient_stats['direction_estimate'] = updated.detach().cpu()

        logging.debug(f"[{self.seller_id}] Updated stats: mean_norm={self.honest_gradient_stats['mean_norm']:.4f}")

    def _compute_malicious_gradient(self) -> Optional[List[torch.Tensor]]:  # <-- FIX: Return Optional
        """
        Compute gradient on malicious objective (backdoor, poisoning, etc.).
        This is what we actually want to inject into the aggregate.
        """
        model = self.model_factory().to(self.device)

        # --- FIX: Use the datasets initialized in __init__ ---
        if self.adv_cfg.attack_type == "backdoor" and self.backdoor_dataset:
            logging.debug(f"[{self.seller_id}] Computing backdoor gradient...")
            backdoor_grad, _ = self._compute_local_grad(model, self.backdoor_dataset)
            return backdoor_grad

        elif self.adv_cfg.attack_type == "untargeted_poisoning":
            logging.debug(f"[{self.seller_id}] Computing untargeted (negative) gradient...")
            honest_grad, _ = self._compute_local_grad(model, self.dataset)
            if honest_grad:
                return [-g for g in honest_grad]
            return None  # <-- FIX

        elif self.adv_cfg.attack_type == "targeted_poisoning" and self.poisoned_dataset:
            logging.debug(f"[{self.seller_id}] Computing targeted poisoning gradient...")
            poison_grad, _ = self._compute_local_grad(model, self.poisoned_dataset)
            return poison_grad

        # Fallback: return honest gradient if attack type is misconfigured
        logging.warning(
            f"[{self.seller_id}] No valid malicious dataset for "
            f"attack_type '{self.adv_cfg.attack_type}'. Falling back to honest gradient."
        )
        grad, _ = self._compute_local_grad(model, self.dataset)
        return grad

    def _identify_vulnerable_layers(self, gradient: List[torch.Tensor]) -> List[int]:
        """
        Identify which layers to inject malicious gradients into.
        Strategy: Target layers from config, or fallback to high-magnitude/final layers.
        """
        # --- FIX 2: Use the pre-computed indices ---
        if self.target_layer_indices:
            logging.debug(f"[{self.seller_id}] Using user-defined target layers: {self.target_layer_indices}")
            return list(self.target_layer_indices)

        # --- Fallback logic (was already correct) ---
        logging.debug(f"[{self.seller_id}] No target layers specified, finding by norm...")
        num_layers = len(gradient)
        num_target = max(1, int(num_layers * 0.2))  # Target top 20% of layers

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
        # ... (this function was already correct) ...
        # (It relies on _identify_vulnerable_layers, which is now fixed)

        honest_flat = flatten_tensor(honest_gradient)
        malicious_flat = flatten_tensor(malicious_gradient)

        # Handle potential size mismatch if malicious task is different
        if honest_flat.numel() != malicious_flat.numel():
            logging.error(f"[{self.seller_id}] Honest/malicious grad size mismatch. Using honest.")
            return honest_gradient

        honest_norm = torch.norm(honest_flat)
        honest_direction = honest_flat / (honest_norm + 1e-9)

        if self.honest_gradient_stats['mean_norm']:
            target_norm = self.honest_gradient_stats['mean_norm']
        else:
            target_norm = honest_norm.item()

        if self.adv_cfg.replacement_strategy == "layer_wise":
            vulnerable_layers = self._identify_vulnerable_layers(honest_gradient)

            blended_gradient = []
            for i, (h_grad, m_grad) in enumerate(zip(honest_gradient, malicious_gradient)):
                if i in vulnerable_layers:
                    alpha = self.attack_intensity
                    layer_blend = (1 - alpha) * h_grad + alpha * m_grad
                    blended_gradient.append(layer_blend)
                else:
                    blended_gradient.append(h_grad.clone())

            blended_flat = flatten_tensor(blended_gradient)
            current_norm = torch.norm(blended_flat)
            scale = target_norm / (current_norm.item() + 1e-9)

            # Re-scale all layers to match the target norm
            # Use a separate loop to avoid modifying tensors in-place during iteration
            final_blended_gradient = [g * scale for g in blended_gradient]
            return final_blended_gradient

        elif self.adv_cfg.replacement_strategy == "global_blend":
            # ... (this logic was correct) ...
            alpha = self.attack_intensity
            blended_flat = (1 - alpha) * honest_flat + alpha * malicious_flat
            blended_direction = blended_flat / (torch.norm(blended_flat) + 1e-9)
            cosine_sim = torch.dot(blended_direction, honest_direction)

            similarity_threshold = self.adv_cfg.similarity_threshold
            if cosine_sim < similarity_threshold:
                alpha_low, alpha_high = 0.0, alpha
                for _ in range(10):  # Binary search
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

            blended_flat = blended_flat * (target_norm / (torch.norm(blended_flat) + 1e-9))
            original_shapes = [g.shape for g in honest_gradient]
            return unflatten_tensor(blended_flat, original_shapes)

        else:
            logging.warning(f"[{self.seller_id}] Unknown strategy, using honest gradient")
            return honest_gradient

    def get_gradient_for_upload(self, **kwargs) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Generate gradient for this round."""
        self.round_counter += 1

        if self.phase == "mimicry" and self.round_counter > self.adv_cfg.mimicry_rounds:
            self.phase = "attack"
            logging.info(f"[{self.seller_id}] Transitioning to Attack Phase")
            logging.info(f"  Learned mean norm: {self.honest_gradient_stats['mean_norm']:.4f}")

        try:
            model = self.model_factory().to(self.device)
            honest_gradient, stats = self._compute_local_grad(model, self.dataset)
            if honest_gradient is None:
                return None, stats
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to compute honest gradient: {e}")
            return None, {'error': str(e)}

        stats['attack_phase'] = self.phase

        if self.phase == "mimicry":
            self._update_honest_gradient_stats(honest_gradient)
            logging.info(f"[{self.seller_id}][Mimicry] Round {self.round_counter}: Learning")
            return honest_gradient, stats

        else:  # Attack Phase
            malicious_gradient = self._compute_malicious_gradient()

            # --- FIX: Handle failure in malicious gradient computation ---
            if malicious_gradient is None:
                logging.error(f"[{self.seller_id}] Failed to compute malicious gradient. Aborting attack this round.")
                return honest_gradient, stats  # Submit honest gradient

            attack_gradient = self._create_stealthy_malicious_gradient(
                honest_gradient,
                malicious_gradient
            )

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
        # ... (this function was already correct) ...
        super().round_end_process(round_number, was_selected, marketplace_metrics, **kwargs)

        self.selection_history.append({
            'round': round_number,
            'phase': self.phase,
            'selected': was_selected
        })

        if self.phase == "attack" and len(self.selection_history) >= 5:
            recent_selections = [h['selected'] for h in self.selection_history[-5:]
                                 if h['phase'] == 'attack']
            if recent_selections:
                selection_rate = sum(recent_selections) / len(recent_selections)

                if selection_rate < 0.5 and self.attack_intensity > 0.05:
                    self.attack_intensity *= 0.9
                    logging.info(f"[{self.seller_id}] Reduced attack intensity to {self.attack_intensity:.3f}")
                elif selection_rate > 0.8 and self.attack_intensity < 0.5:
                    self.attack_intensity *= 1.1
                    logging.info(f"[{self.seller_id}] Increased attack intensity to {self.attack_intensity:.3f}")
