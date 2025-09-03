import collections
import copy
import logging
import sys
import time
# Add these class definitions as well
from abc import ABC, abstractmethod
from collections import abc  # abc.Mapping for general dicts
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
    LabelFlipGenerator, BackdoorTextGenerator
from common.enums import TriggerType, TriggerLocation
from common.gradient_market_configs import AdversarySellerConfig, BackdoorImageConfig, LabelFlipConfig, \
    BackdoorTextConfig, SybilConfig, RuntimeDataConfig
from common.status_save import ClientState
from marketplace.seller.adversary_gradient_seller import DataConfig, TrainingConfig, GradientSeller
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


class SybilCoordinator:
    def __init__(self, sybil_cfg: SybilConfig, aggregator):
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
        base_model = self.aggregator.global_model.to(self.device)

        for cid in selected_client_ids:
            if cid in self.clients:
                seller = self.clients[cid].seller_obj
                gradient = seller.get_local_gradient(base_model)
                self.selected_gradients[cid] = self._ensure_tensor(gradient)

    def get_selected_average(self) -> Optional[torch.Tensor]:
        """Compute the average gradient of all selected sellers."""
        if not self.selected_gradients:
            return None
        gradients = list(self.selected_gradients.values())
        return torch.mean(torch.stack(gradients), dim=0)

    def update_nonselected_gradient(self, current_gradient, strategy: Optional[str] = None) -> List[np.ndarray]:
        """Update a non-selected gradient using a specified strategy object."""
        strat_name = strategy or self.sybil_cfg.gradient_default_mode
        avg_grad = self.get_selected_average()

        # If no selected gradients exist, just return the original gradient
        if avg_grad is None:
            # Logic to handle both list and single tensor inputs
            if isinstance(current_gradient, list):
                return [g.cpu().numpy() for g in current_gradient]
            return [self._ensure_tensor(current_gradient).cpu().numpy()]

        strategy_obj = self.strategies.get(strat_name)
        if not strategy_obj:
            raise ValueError(f"Strategy '{strat_name}' not found or configured.")

        current_grad_tensor = self._ensure_tensor(current_gradient)
        new_grad = strategy_obj.manipulate(current_grad_tensor, avg_grad)

        return [new_grad.cpu().numpy()]

    def _unflatten_gradient(self, flat_grad: torch.Tensor, original_shapes: List[torch.Size]) -> List[torch.Tensor]:
        """
        Reconstruct a list of tensors with original shapes from a flattened gradient.
        """
        result = []
        offset = 0
        for shape in original_shapes:
            num_elements = torch.prod(torch.tensor(shape)).item()
            tensor_flat = flat_grad[offset:offset + num_elements]
            tensor = tensor_flat.reshape(shape)
            result.append(tensor)
            offset += num_elements
        return result

    def prepare_for_new_round(self) -> None:
        """Prepares state for the next round and handles dynamic triggers."""
        self.cur_round += 1
        if self.cur_round >= self.sybil_cfg.benign_rounds:
            self.start_atk = True

        # Dynamic trigger logic is now decoupled
        if self.sybil_cfg.trigger_mode == "dynamic" and self.start_atk:
            self._execute_dynamic_trigger_update()

    def _execute_dynamic_trigger_update(self):
        """Logic for generating and distributing a new trigger."""
        best_client_id = self.get_client_with_highest_selection_rate()
        if not best_client_id:
            return

        # 1. Have the "best" seller generate a new, optimized trigger
        best_seller = self.clients[best_client_id].seller_obj
        logging.info(f"Coordinator: Tasking seller {best_seller.seller_id} to generate a new trigger.")
        # This method is assumed to exist on the adversary seller class
        new_trigger = best_seller.generate_optimized_trigger(self.aggregator.global_model)

        # 2. Distribute the new trigger to all other sybils
        if new_trigger is not None:
            self.distribute_global_trigger(new_trigger)

    def distribute_global_trigger(self, new_trigger: torch.Tensor) -> None:
        """Distribute a new trigger to all registered malicious sellers."""
        logging.info("Coordinator: Distributing new global trigger to all Sybil sellers.")
        for client_state in self.clients.values():
            # This method is assumed to exist on the adversary seller class
            client_state.seller_obj.update_poison_trigger(new_trigger)

    def on_round_end(self) -> None:
        """Clear round-specific state."""
        self.update_historical_patterns(self.selected_gradients)
        self.adaptive_role_assignment()
        self.selected_gradients = {}

    def distribute_global_trigger(self, new_trigger: torch.Tensor) -> None:
        """
        Update the global trigger maintained by the coordinator.
        This new trigger will be used by all malicious sellers.
        """
        trigger = new_trigger.clone().detach().to(self.device)
        self.backdoor_generator.update_trigger(trigger)
        print("Coordinator: Global trigger updated.")


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
        self.last_computed_gradient: Optional[List[torch.Tensor]] = None
        self.last_training_stats: Optional[Dict[str, Any]] = None

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

    def _compute_local_grad(self, model_to_train: nn.Module, dataset: Dataset) -> Tuple[
        Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Handles the local training loop and gradient computation."""
        start_time = time.time()

        if not dataset or len(dataset) == 0:
            logging.warning(f"[{self.seller_id}] Dataset is empty. Returning zero gradient.")
            zero_grad = [torch.zeros_like(p) for p in model_to_train.parameters()]
            return zero_grad, {'train_loss': None, 'compute_time_ms': 0}

        # Create DataLoader using the provided data configuration
        data_loader = DataLoader(
            dataset,
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

        # NEW: Record the round's events to the seller's history
        round_record = {
            'event_type': 'federated_round',
            'round': round_number,
            'timestamp': time.time(),
            'was_selected': is_selected,
            'training_stats': self.last_training_stats if is_selected else None
        }
        self.federated_round_history.append(round_record)

        # Reset state for the next round (as before)
        self.last_computed_gradient = None
        self.last_training_stats = None


class AdvancedPoisoningAdversarySeller(GradientSeller):
    """
    An adversary that performs data poisoning attacks and can participate
    in a Sybil group, built using a unified AdversaryConfig.
    """

    def __init__(self,
                 seller_id: str,
                 data_config: RuntimeDataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 adversary_config: AdversarySellerConfig,  # Use the unified config
                 sybil_coordinator: Optional[SybilCoordinator] = None,
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        super().__init__(
            seller_id=seller_id,
            data_config=data_config,
            training_config=training_config,
            model_factory=model_factory,
            save_path=save_path,
            device=device,
            **kwargs
        )
        self.adversary_config = adversary_config
        self.sybil_coordinator = sybil_coordinator
        self.is_sybil = self.adversary_config.sybil.is_sybil and sybil_coordinator is not None
        self.selected_last_round = False

        # --- Key Update: Create the poison generator inside the seller ---
        # The seller is now responsible for creating its tools from its config.
        self.poison_generator = self._create_poison_generator()

    def _create_poison_generator(self) -> Optional[PoisonGenerator]:
        """Factory method to create the correct poison generator from the config."""
        poison_cfg = self.adversary_config.poisoning

        if poison_cfg.type == 'none':
            return None
        elif poison_cfg.type == 'backdoor':
            # Create the specific config for the image generator
            backdoor_image_cfg = BackdoorImageConfig(
                target_label=poison_cfg.backdoor_params.target_label,
                trigger_type=TriggerType(poison_cfg.backdoor_params.trigger_type),
                location=TriggerLocation(poison_cfg.backdoor_params.location)
            )
            return BackdoorImageGenerator(backdoor_image_cfg)
        elif poison_cfg.type == 'label_flip':
            # Create the specific config for the label flip generator
            label_flip_cfg = LabelFlipConfig(
                num_classes=self.data_config.num_classes,  # Assuming num_classes is available
                attack_mode=poison_cfg.label_flip_params.mode,
                target_label=poison_cfg.label_flip_params.target_label
            )
            return LabelFlipGenerator(label_flip_cfg)
        else:
            raise ValueError(f"Unknown poison type in config: {poison_cfg.type}")

    def get_gradient_for_upload(self, global_model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Overrides the base method to implement poisoning and Sybil logic.
        """
        # --- Correctly access nested config values ---
        current_round = self.sybil_coordinator.cur_round if self.is_sybil else float('inf')
        should_attack = current_round >= self.adversary_config.sybil.benign_rounds

        if should_attack and self.poison_generator:
            logging.info(f"[{self.seller_id}] Attack phase: generating poisoned data.")
            # Use the generator created during initialization
            dataset_for_training, _ = self.poison_generator.generate_poisoned_dataset(
                original_dataset=self.data_config.dataset,
                poison_rate=self.adversary_config.poisoning.poison_rate
            )
        else:
            logging.info(f"[{self.seller_id}] Benign phase: using clean data.")
            dataset_for_training = self.data_config.dataset

        # --- Improved: Use a helper to avoid changing self.data_config ---
        base_gradient, stats = self._compute_gradient_on_dataset(dataset_for_training, global_model)

        # Apply Sybil logic if applicable
        # --- Removed hardcoded "mimic" strategy ---
        sybil_strategy = self.adversary_config.sybil.gradient_default_mode
        if self.is_sybil and not self.selected_last_round and should_attack:
            logging.info(f"[{self.seller_id}] Sybil client not selected. Using strategy: '{sybil_strategy}'")
            coordinated_gradient = self.sybil_coordinator.update_nonselected_gradient(
                base_gradient, strategy=sybil_strategy
            )
            return coordinated_gradient, stats

        return base_gradient, stats

    def _compute_gradient_on_dataset(self, dataset: Dataset, global_model: nn.Module) -> Tuple[
        Optional[List[torch.Tensor]], Dict[str, Any]]:
        """A helper to compute gradient on a specific dataset without modifying object state."""
        # Create a temporary DataConfig for this computation
        temp_data_config = DataConfig(dataset=dataset, collate_fn=self.data_config.collate_fn)

        # In a real implementation, you might refactor GradientSeller to accept a DataConfig
        # override. For now, this state-swapping is isolated to a helper method.
        original_data_config = self.data_config
        self.data_config = temp_data_config
        gradient, stats = super().get_gradient_for_upload(global_model)
        self.data_config = original_data_config  # Restore state immediately
        return gradient, stats

    def round_end_process(self, round_number: int, is_selected: bool) -> None:
        """Records the outcome of the round for Sybil coordination."""
        self.selected_last_round = is_selected
        logging.info(f"[{self.seller_id}] Round {round_number} ended. Selected: {is_selected}")


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
            backdoor_generator: object,
            device: str,
    ):
        self.original_dataset = original_dataset
        self.trigger_indices_set: Set[int] = set(map(int, trigger_indices))
        self.target_label = int(target_label)
        self.backdoor_generator = backdoor_generator
        self.device = device

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> tuple:
        """

        Retrieves an item, applies a trigger if the index is targeted,
        and ensures the output is a standardized (data, label) tuple on the correct device.
        """
        # 1. Get raw item and handle different data/label orderings
        item_a, item_b = self.original_dataset[idx]
        if isinstance(item_a, int) and not isinstance(item_b, int):
            label, data = item_a, item_b  # Handle (label, data) format
        else:
            data, label = item_a, item_b  # Assume (data, label) format

        is_poisoned = idx in self.trigger_indices_set

        # 2. Apply trigger and update label if the sample is targeted
        if is_poisoned:
            label = self.target_label
            # --- Image Data Branch ---
            if isinstance(data, torch.Tensor) and data.dim() >= 2:  # Robust check for tensors
                data = self.backdoor_generator.apply_trigger_tensor(data.to(self.device))
            # --- Text Data Branch ---
            elif isinstance(data, (list, tuple, torch.Tensor, dict)):
                data = self.backdoor_generator.apply_trigger_text(data, device=self.device)
            else:
                raise TypeError(f"Unsupported data type from dataset: {type(data)}")

        # 3. Ensure data is on the correct device (for non-poisoned samples)
        if not is_poisoned:
            if isinstance(data, dict):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
            else:
                # This handles tensors, lists, and other types that the model expects.
                # We assume the model's forward pass will handle final tensor conversion if needed.
                pass

        # 4. Standardize label to a Tensor and move to device
        final_label = torch.tensor(label, dtype=torch.long, device=self.device)

        return data, final_label


class AdvancedBackdoorAdversarySeller(GradientSeller):
    """
    A generic adversary that injects backdoors (image or text) and can
    coordinate with a Sybil group, driven by a unified AdversaryConfig.
    """

    def __init__(self,
                 seller_id: str,
                 data_config: RuntimeDataConfig,
                 training_config: TrainingConfig,
                 model_factory: Callable[[], nn.Module],
                 adversary_config: AdversarySellerConfig,  # Use the unified config
                 model_type: str,  # 'image' or 'text'
                 sybil_coordinator: Optional[SybilCoordinator] = None,
                 save_path: str = "",
                 device: str = "cpu",
                 **kwargs: Any):

        super().__init__(seller_id, data_config, training_config, model_factory,
                         save_path, device, **kwargs)

        self.adversary_config = adversary_config
        self.model_type = model_type
        self.sybil_coordinator = sybil_coordinator
        self.is_sybil = self.adversary_config.sybil.is_sybil and sybil_coordinator is not None
        self.selected_last_round = False
        if self.model_type == 'image':
            # Use the image-specific params
            params = self.adversary_config.poisoning.image_backdoor_params
        elif self.model_type == 'text':
            # Use the text-specific params
            params = self.adversary_config.poisoning.text_backdoor_params
        self.backdoor_params = params
        # --- Key Update: Create the correct poison generator from the config ---
        self.poison_generator = self._create_poison_generator(**kwargs)

    def _create_poison_generator(self, **kwargs) -> PoisonGenerator:
        """Factory method to create the correct backdoor generator."""
        poison_cfg = self.adversary_config.poisoning
        if poison_cfg.type != 'backdoor':
            raise ValueError("AdvancedBackdoorAdversarySeller only supports 'backdoor' poisoning type.")

        if self.model_type == 'image':
            # Create the specific config for the image generator
            backdoor_image_cfg = BackdoorImageConfig(
                target_label=self.backdoor_params.target_label,
                trigger_type=TriggerType(self.backdoor_params.trigger_type),
                location=TriggerLocation(self.backdoor_params.location)
            )
            return BackdoorImageGenerator(backdoor_image_cfg)

        elif self.model_type == 'text':
            # Create the specific config for the text generator
            backdoor_text_cfg = BackdoorTextConfig(
                vocab=kwargs.get('vocab'),  # Vocab is passed via kwargs from the factory
                target_label=self.backdoor_params.target_label,
                trigger_content=self.backdoor_params.trigger_content,
                location=self.backdoor_params.location
            )
            return BackdoorTextGenerator(backdoor_text_cfg)
        else:
            raise ValueError(f"Unsupported model_type for backdoor: {self.model_type}")

    def get_gradient_for_upload(self, global_model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """Overrides the base method to implement the full adversary strategy."""
        # --- Correctly access nested config values ---
        current_round = self.sybil_coordinator.cur_round if self.is_sybil else float('inf')
        should_attack = current_round >= self.adversary_config.sybil.benign_rounds

        # Assuming 'cmd' or 'single' is defined in backdoor_params now
        mode = self.adversary_config.poisoning.backdoor_params.mode

        if not should_attack:
            mode = "none"
        logging.info(f"[{self.seller_id}] Round {current_round}. Behavior: {mode}.")

        base_model = self.model_factory().to(self.device)
        base_model.load_state_dict(global_model.state_dict())

        if mode == "mixed_data":
            base_gradient, stats = self._compute_single_attack_gradient(base_model)
        elif mode == "combine_gradient":
            base_gradient, stats = self._compute_cmd_attack_gradient(base_model)
        else:
            base_gradient, stats = self._compute_clean_gradient(base_model)

        # Sybil logic remains the same
        return base_gradient, stats

    def _create_poisoned_dataset_view(self) -> TriggeredSubsetDataset:
        """Helper to create the on-the-fly poisoned dataset view."""
        poison_cfg = self.adversary_config.poisoning
        num_samples = len(self.data_config.dataset)
        num_triggered = int(num_samples * poison_cfg.poison_rate)
        trigger_indices = np.random.choice(np.arange(num_samples), size=num_triggered, replace=False)

        # The dataset wrapper uses the generator created in __init__
        return TriggeredSubsetDataset(
            original_dataset=self.data_config.dataset,
            trigger_indices=trigger_indices,
            backdoor_generator=self.poison_generator,  # Pass the live generator object
        )

    def _compute_clean_gradient(self, model: nn.Module) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Computes a gradient on the original, clean dataset."""
        return self._compute_local_grad(model, self.data_config.dataset)

    def _compute_single_attack_gradient(self, model: nn.Module) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Computes one gradient on a dataset with both clean and poisoned samples."""
        poisoned_dataset_view = self._create_poisoned_dataset_view()
        return self._compute_local_grad(model, poisoned_dataset_view)

    def _compute_cmd_attack_gradient(self, model: nn.Module) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Computes separate clean and backdoor gradients and blends them."""
        g_benign, stats_benign = self._compute_clean_gradient(model)

        poisoned_dataset_view = self._create_poisoned_dataset_view()
        g_backdoor, _ = self._compute_local_grad(model, poisoned_dataset_view)

        if not g_benign or not g_backdoor:
            logging.warning(f"[{self.seller_id}] CMD failed; returning clean gradient.")
            return g_benign, stats_benign

        strength = self.backdoor_config.poison_strength
        final_gradient = [
            (1 - strength) * gb + strength * gbp
            for gb, gbp in zip(g_benign, g_backdoor)
        ]
        return final_gradient, stats_benign

    def round_end_process(self, round_number: int, is_selected: bool) -> None:
        """Records the outcome of the round for Sybil coordination."""
        self.selected_last_round = is_selected
        logging.info(f"[{self.seller_id}] Round {round_number} ended. Selected: {is_selected}")

    # --- Trigger Optimization Methods (Assumed to be okay as per "running good" and focus on 'single' mode) ---
    # upload_global_trigger, get_new_trigger, trigger_opt methods would remain here.
    # Make sure they are compatible with the self.device and backdoor_generator used.
    def upload_global_trigger(self, global_model, first_attack=False, lr=0.01, num_steps=50, lambda_param=1):
        logging.info(f"[{self.seller_id}] Broadcasting new trigger...")
        new_trigger = self.get_new_trigger(global_model, first_attack=first_attack, lr=lr, num_steps=num_steps,
                                           lambda_param=lambda_param)
        if self.sybil_coordinator is not None:
            self.sybil_coordinator.update_global_trigger(new_trigger)
        return new_trigger

    def get_new_trigger(self, global_model, first_attack, lr=0.01, num_steps=50, lambda_param=1):
        # (Implementation from your original code)
        # Ensure self.backdoor_generator.get_trigger() and apply_trigger_tensor are robust
        trigger = self.backdoor_generator.get_trigger().detach().to(self.device).requires_grad_(True)
        model = global_model.to(self.device)
        if first_attack:
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=True, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=0.0)
            tmp_trigger = self.trigger_opt(model, tmp_trigger, first_attack=False, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=1.0)
        else:
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=False, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=1.0)
        return tmp_trigger

    def trigger_opt(self, model, trigger, first_attack=False, trigger_lr=0.01, num_steps=50, lambda_param=1.0):
        # (Implementation from your original code - ensure imports like optim, nn, DataLoader are present at file level)
        # For brevity, not re-pasting the full trigger_opt, but it would be here.
        # Key: ensure that self.dataset used inside DataLoader here is the original clean dataset.
        # And self.backdoor_generator.apply_trigger_tensor(data, trigger) is used.
        # Also, the model passed should be on self.device.
        # (The rest of your trigger_opt code as provided previously)
        model.eval()
        if not trigger.requires_grad: trigger = trigger.clone().detach().requires_grad_(True)
        trigger_optimizer = torch.optim.Adam([trigger], lr=trigger_lr)  # Ensure optim is imported
        criterion = torch.nn.CrossEntropyLoss()  # Ensure nn is imported
        # Use self.dataset (original clean data) for trigger optimization source
        dataloader = DataLoader(self.dataset, batch_size=self.local_training_params.get('batch_size', 64), shuffle=True)

        # Phase 1
        if first_attack:
            logging.info("Trigger Opt - Phase 1: Loss alignment")
            for step in range(num_steps):
                # ... (your phase 1 loop)
                # Example of one batch from your code:
                data, _ = next(iter(dataloader))  # Get a batch
                if data.dim() == 3: data = data.unsqueeze(1)
                data = data.to(self.device)
                backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)  # Pass trigger explicitly
                backdoor_labels = torch.full((data.shape[0],), self.target_label, dtype=torch.long, device=self.device)
                outputs = model(backdoored_data)
                loss = criterion(outputs, backdoor_labels)
                trigger_optimizer.zero_grad();
                loss.backward();
                trigger_optimizer.step()
                with torch.no_grad():
                    trigger.clamp_(0, 1)
                if (step + 1) % 10 == 0: logging.info(f"Phase 1 - Step {step + 1}, Loss: {loss.item():.4f}")

        # Phase 2
        logging.info(f"Trigger Opt - Phase 2: Gradient alignment optimization (λ={lambda_param})")
        # Re-initialize dataloader or ensure it can be iterated again if needed, or use a fresh one
        dataloader_phase2 = DataLoader(self.dataset, batch_size=self.local_training_params.get('batch_size', 64),
                                       shuffle=True)
        if not trigger.requires_grad: trigger = trigger.clone().detach().requires_grad_(True)

        for step in range(num_steps):
            # ... (your phase 2 loop)
            # Example of one batch from your code:
            data, label = next(iter(dataloader_phase2))
            if data.dim() == 3: data = data.unsqueeze(1)
            data, label = data.to(self.device), label.to(self.device)
            backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)  # Pass trigger explicitly
            backdoor_labels = torch.full((data.shape[0],), self.target_label, dtype=torch.long, device=self.device)

            model.zero_grad();
            clean_outputs = model(data);
            clean_loss = criterion(clean_outputs, label)
            clean_loss.backward(retain_graph=True)
            clean_grads = {name: param.grad.clone() for name, param in model.named_parameters() if
                           param.grad is not None}

            model.zero_grad();
            backdoor_outputs = model(backdoored_data);
            backdoor_loss_val = criterion(backdoor_outputs, backdoor_labels)
            # If lambda_param is 0, backdoor_loss_val.backward() alone is enough to get grads for backdoor_loss_val
            # If lambda_param is >0 and <1, you need grads from both backdoor_loss_val and gradient_distance
            # If lambda_param is 1, you only need grads from gradient_distance

            # Simplified: calculate combined loss and backprop once if possible.
            # For gradient_distance, you need param.grad from backdoor_loss.backward() first.
            backdoor_loss_val.backward(retain_graph=True)  # Get param.grad for backdoor data

            gradient_distance = torch.tensor(0.0, device=self.device)
            for name, param in model.named_parameters():
                if param.grad is not None and name in clean_grads:
                    gradient_distance += torch.sum((param.grad - clean_grads[name]) ** 2)

            combined_loss = lambda_param * gradient_distance
            if (1 - lambda_param) > 1e-6:  # only add backdoor loss if its weight is meaningful
                combined_loss += (1 - lambda_param) * backdoor_loss_val

            trigger_optimizer.zero_grad()
            # Need to compute gradient of combined_loss w.r.t trigger
            # This requires combined_loss to be a function of trigger.
            # The param.grad values used in gradient_distance are themselves functions of trigger (via backdoored_data).
            # backdoor_loss_val is also a function of trigger.
            if trigger.grad is not None: trigger.grad.zero_()  # Clear previous trigger gradients

            # We need d(combined_loss)/d(trigger).
            # torch.autograd.grad is the right tool here.
            # Ensure that operations contributing to combined_loss are part of the graph involving trigger.
            if combined_loss.requires_grad:  # Check if it's part of a graph that requires grad
                trigger_grads_from_loss = torch.autograd.grad(combined_loss, trigger,
                                                              retain_graph=False)  # retain_graph=False if not needed further
                if trigger_grads_from_loss[0] is not None:
                    trigger.grad = trigger_grads_from_loss[0]
                    trigger_optimizer.step()
                else:
                    logging.warning("Gradient of combined_loss w.r.t trigger is None.")
            else:  # if lambda_param is 0, combined_loss might be just backdoor_loss_val
                if (1 - lambda_param) > 1e-6:  # if backdoor_loss term is active
                    trigger_grads_from_bloss = torch.autograd.grad(backdoor_loss_val, trigger, retain_graph=False)
                    if trigger_grads_from_bloss[0] is not None:
                        trigger.grad = trigger_grads_from_bloss[0]
                        trigger_optimizer.step()
                    else:
                        logging.warning("Gradient of backdoor_loss_val w.r.t trigger is None.")

            with torch.no_grad():
                trigger.clamp_(0, 1)
            if (step + 1) % 10 == 0: logging.info(
                f"Phase 2 - Step {step + 1}, GradDist: {gradient_distance.item():.4f}, BdoorLoss: {backdoor_loss_val.item():.4f}")

        return trigger.detach()
