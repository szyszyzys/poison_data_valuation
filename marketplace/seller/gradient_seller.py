import collections
import csv
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
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attack.attack_gradient_market.poison_attack.attack_utils import PoisonGenerator, BackdoorImageGenerator, \
    BackdoorTextGenerator
from common.enums import ImageTriggerType, ImageTriggerLocation
from common.gradient_market_configs import AdversarySellerConfig, BackdoorImageConfig, BackdoorTextConfig, SybilConfig, \
    RuntimeDataConfig, TrainingConfig
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

    def get_gradient_for_upload(self, global_model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes and returns the gradient update and training statistics.
        This is the primary method for the federated learning coordinator to call.
        """
        try:
            # 1. Create a fresh model instance using the injected factory.
            local_model = self.model_factory().to(self.device)
            local_model.load_state_dict(global_model.state_dict())
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to prepare model: {e}", exc_info=True)
            return None, {'error': 'Model preparation failed.'}

        # 2. Delegate the actual training and gradient calculation.
        gradient, stats = self._compute_local_grad(local_model, self.data_config.dataset)

        # 3. Update the seller's internal state for logging or inspection.
        self.last_computed_gradient = gradient
        self.last_training_stats = stats

        return gradient, stats

    def _compute_local_grad(self, model_to_train: nn.Module, dataset_to_use: Optional[Dataset] = None) -> Tuple[
        Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Handles the local training loop and gradient computation."""
        if dataset_to_use is None:
            dataset_to_use = self.data_config.dataset
        start_time = time.time()

        if not dataset_to_use or len(dataset_to_use) == 0:
            logging.warning(f"[{self.seller_id}] Dataset is empty. Returning zero gradient.")
            zero_grad = [torch.zeros_like(p) for p in model_to_train.parameters()]
            return zero_grad, {'train_loss': None, 'compute_time_ms': 0}

        # Create DataLoader using the provided data configuration
        data_loader = DataLoader(
            dataset_to_use,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=self.data_config.collate_fn
        )

        try:
            # The core training logic is in an external, reusable function
            grad_tensors, avg_loss = local_training_and_get_gradient(
                model=model_to_train,
                train_loader=data_loader,
                device=self.device,
                local_epochs=self.training_config.local_epochs,
                lr=self.training_config.learning_rate,
            )

            stats = {
                'train_loss': avg_loss,
                'compute_time_ms': (time.time() - start_time) * 1000,
                'upload_bytes': estimate_byte_size(grad_tensors)
            }
            return grad_tensors, stats

        except Exception as e:
            logging.error(f"[{self.seller_id}] Error in training loop: {e}", exc_info=True)
            return None, {'error': 'Training failed.'}

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

    def round_end_process(self, round_number: int, is_selected: bool) -> None:
        """Logs the outcome of the round and cleans up state."""
        logging.info(f"[{self.seller_id}] Round {round_number} ended. Selected: {is_selected}")

        round_record = {
            'event_type': 'federated_round',
            'round': round_number,
            'timestamp': time.time(),
            'was_selected': is_selected,
            'training_stats': self.last_training_stats
        }
        self.federated_round_history.append(round_record)

        self.last_computed_gradient = None
        self.last_training_stats = None
        self.selected_last_round = is_selected

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
        self.device = aggregator.device  # Get device from a reliable source

        self.clients: Dict[str, ClientState] = collections.OrderedDict()

        # --- Initialize strategies from config ---
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

    def get_client_with_highest_selection_rate(self) -> str:
        """
        Returns the client ID with the highest selection rate.
        If there are no clients, returns None.
        """
        best_client = None
        max_rate = -1.0  # start with a rate lower than any possible selection rate
        for cid, client_info in self.clients.items():
            if client_info.selection_rate > max_rate:
                max_rate = client_info.selection_rate
                best_client = cid
        return best_client

    # ----- Update & Analysis Methods -----
    def update_client_states(self, selected_client_ids: List[str]) -> None:
        """Update the state of each client based on the latest selection results."""
        for cid, client_state in self.clients.items():
            was_selected = cid in selected_client_ids
            client_state.selection_history.append(was_selected)
            client_state.selection_rate = sum(client_state.selection_history) / len(client_state.selection_history)
            client_state.rounds_participated += 1

            # Use parameters from the config object
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
        # This function's internal logic is mostly fine, so it's kept as is.
        # It benefits from self.selected_history being a deque.
        all_selected_flat = [
            grad.flatten() for round_dict in self.selected_history for grad in round_dict.values()
        ]

        if not all_selected_flat:
            self.selection_patterns = {}
            return

        all_tensor = torch.stack(all_selected_flat)
        centroid = torch.mean(all_tensor, dim=0)

        # Simplified similarity calculation
        avg_sim = 0.0
        if len(all_tensor) > 1:
            # Calculate similarity of each gradient to the centroid
            sims = F.cosine_similarity(all_tensor, centroid.unsqueeze(0))
            avg_sim = torch.mean(sims).item()

        self.selection_patterns = {"centroid": centroid, "avg_similarity": avg_sim}

    def adaptive_role_assignment(self) -> None:
        """Dynamically reassign roles based on selection rates."""
        if not self.clients:
            return

        sorted_clients = sorted(self.clients.items(), key=lambda item: item[1].selection_rate, reverse=True)
        num_clients = len(sorted_clients)

        # Use role percentages from the config
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
        """
        Ensure that the provided gradient is a single flattened tensor on the correct device.
        """
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
            return gradient.to(self.device)
        elif isinstance(gradient, np.ndarray):
            return torch.from_numpy(gradient).to(self.device)
        else:
            raise TypeError(f"Unsupported gradient type: {type(gradient)}")

    def collect_selected_gradients(self, selected_client_ids: List[str]) -> None:
        """Collects gradients from the sellers selected in the current round."""
        self.selected_gradients = {}
        base_model = self.aggregator.strategy.global_model.to(self.device)

        for cid in selected_client_ids:
            if cid in self.clients:
                seller = self.clients[cid].seller_obj
                gradient_tensors, _ = seller.get_gradient_for_upload(base_model)

                # Now, pass only the tensors to _ensure_tensor
                if gradient_tensors is not None:
                    self.selected_gradients[cid] = self._ensure_tensor(gradient_tensors)

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

        # If no selected gradients exist, just return the original gradient as a list of tensors.
        if avg_grad is None:
            if isinstance(current_gradient, list):
                return current_gradient  # Already a list of tensors
            # Ensure even a single tensor is returned as a list
            return [self._ensure_tensor(current_gradient)]

        strategy_obj = self.strategies.get(strat_name)
        if not strategy_obj:
            raise ValueError(f"Strategy '{strat_name}' not found or configured.")

        if isinstance(current_gradient, list):
            original_shapes = [g.shape for g in current_gradient]
        else:
            original_shapes = [current_gradient.shape]  # A list with just one shape

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
        # --- MODIFICATION: UNPACK BASED ON FORMAT ---
        original_sample = self.original_dataset[index]

        if self.data_format == 'text':
            # For text, the format is (label, data)
            label, data = original_sample
        else:
            # For images, the format is (data, label)
            data, label = original_sample

        # The rest of the logic remains the same
        if self.poison_generator and index in self.poison_indices:
            return self.poison_generator.apply(data, label)

        # Return in the standard (data, label) format for consistency
        return data, label


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
                 sybil_coordinator: Optional[SybilCoordinator] = None,
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
        self.sybil_coordinator = sybil_coordinator
        self.is_sybil = self.adversary_config.sybil.is_sybil and sybil_coordinator is not None
        self.selected_last_round = False

        # The seller simply stores the generator it was given.
        self.poison_generator = poison_generator

    def get_gradient_for_upload(self, global_model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Overrides the base method to implement poisoning and Sybil logic.
        """
        current_round = self.sybil_coordinator.cur_round if self.is_sybil else float('inf')
        should_attack = current_round >= self.adversary_config.sybil.benign_rounds

        # --- UPDATED: Use the PoisonedDataset wrapper for clean data handling ---
        if should_attack and self.poison_generator:
            logging.info(f"[{self.seller_id}] Attack phase: using poisoned dataset wrapper.")
            dataset_for_training = PoisonedDataset(
                original_dataset=self.dataset,  # self.dataset from GradientSeller
                poison_generator=self.poison_generator,
                poison_rate=self.adversary_config.poisoning.poison_rate
            )
        else:
            logging.info(f"[{self.seller_id}] Benign phase: using clean data.")
            dataset_for_training = self.dataset

        # The base GradientSeller's training method can now be called directly.
        # This assumes train_local_model is the method that computes the gradient.
        base_gradient, stats = self._compute_local_grad(
            model_to_train=global_model,
            dataset_to_use=dataset_for_training
        )
        # Apply Sybil logic (this part remains the same)
        sybil_strategy = self.adversary_config.sybil.gradient_default_mode
        if self.is_sybil and not self.selected_last_round and should_attack:
            logging.info(f"[{self.seller_id}] Sybil client not selected. Using strategy: '{sybil_strategy}'")
            coordinated_gradient = self.sybil_coordinator.update_nonselected_gradient(
                base_gradient, strategy=sybil_strategy
            )
            return coordinated_gradient, stats

        return base_gradient, stats


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
                 sybil_coordinator: Optional[SybilCoordinator] = None,
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        # 1. Create the specific poison generator for this backdoor attack.
        #    This is the primary job of this specialized class.
        backdoor_generator = self._create_poison_generator(
            adversary_config, model_type, **kwargs
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
            sybil_coordinator=sybil_coordinator,
            save_path=save_path,
            device=device,
            **kwargs
        )
        logging.info(
            f"[{self.seller_id}] Initialized as AdvancedBackdoorAdversarySeller "
            f"with a '{type(backdoor_generator).__name__}'."
        )

    # This method is now static as it doesn't depend on instance state ('self').
    # It's a pure factory function that translates config into an object.
    @staticmethod
    def _create_poison_generator(adv_cfg: AdversarySellerConfig, model_type: str, **kwargs: Any) -> PoisonGenerator:
        """Factory method to create the correct backdoor generator from configuration."""
        poison_cfg = adv_cfg.poisoning
        if 'backdoor' not in poison_cfg.type.value:
            raise ValueError(f"This factory only supports backdoor types, but got '{poison_cfg.type.value}'.")

        if model_type == 'image':
            params = poison_cfg.image_backdoor_params.simple_data_poison_params
            backdoor_image_cfg = BackdoorImageConfig(
                target_label=params.target_label,
                trigger_type=ImageTriggerType(params.trigger_type),
                location=ImageTriggerLocation(params.location)
            )
            return BackdoorImageGenerator(backdoor_image_cfg)

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
        else:
            raise ValueError(f"Unsupported model_type for backdoor: {model_type}")
