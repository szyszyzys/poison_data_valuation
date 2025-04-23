# You already have these classes in your project:
# from your_seller_module import GradientSeller, SellerStats
# from train import compute_loss, etc. (if needed)
# from dataset import dataset_output_dim (if needed)
import collections
import copy
import logging
import sys
import time
from collections import abc  # abc.Mapping for general dicts
from typing import Any
from typing import Dict
from typing import List, Optional, Union
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from general_utils.data_utils import list_to_tensor_dataset
from marketplace.seller.seller import BaseSeller
from model.utils import get_image_model, local_training_and_get_gradient


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


# CombinedSybilCoordinator integrates functionalities from both PFedBA_SybilAttack and SybilCoordinator.
class SybilCoordinator:
    def __init__(self,
                 backdoor_generator,
                 detection_threshold: float = 0.8,
                 benign_rounds: int = 3,
                 gradient_default_mode: str = "mimic",
                 trigger_mode: str = "static",
                 alpha: float = 0.5,
                 amplify_factor: float = 2.0,
                 cost_scale: float = 1.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 aggregator=None):
        # PFedBA-related attributes
        self.detection_threshold = detection_threshold
        self.benign_rounds = benign_rounds  # rounds to act benign before switching to attack
        self.selected_history = []  # history of gradients (dicts: seller_id -> gradient)
        self.selection_patterns = {}  # stores computed centroid and average similarity
        self.clients = {}  # maps seller_id to info: role, selection_history, phase, rounds_participated
        self.trigger_mode = trigger_mode
        # SybilCoordinator-related attributes
        self.gradient_default_mode = gradient_default_mode
        self.alpha = alpha
        self.amplify_factor = amplify_factor
        self.cost_scale = cost_scale
        self.device = device
        self.aggregator = aggregator
        self.registered_clients = collections.OrderedDict()  # seller_id -> seller object
        self.selected_gradients = {}  # stores gradients from sellers selected in the last round
        self.cur_round = 0
        self.start_atk = False

        self.backdoor_generator = backdoor_generator

    # ----- Registration Methods -----
    def register_seller(self, seller) -> None:
        """
        Register a malicious seller with the coordinator.
        The seller object must have a unique attribute 'seller_id'.
        Also, add an entry to the local clients dictionary.
        """
        if not hasattr(seller, 'seller_id'):
            raise AttributeError("Seller object must have a 'seller_id' attribute")
        self.registered_clients[seller.seller_id] = seller
        self.clients[seller.seller_id] = {
            "role": "hybrid",  # initial role can be "hybrid"
            "selection_history": [],
            "selection_rate": 0.0,
            "phase": "benign",  # initial phase: benign
            "rounds_participated": 0
        }

    def get_client_with_highest_selection_rate(self) -> str:
        """
        Returns the client ID with the highest selection rate.
        If there are no clients, returns None.
        """
        best_client = None
        max_rate = -1.0  # start with a rate lower than any possible selection rate
        for cid, client_info in self.clients.items():
            if client_info["selection_rate"] > max_rate:
                max_rate = client_info["selection_rate"]
                best_client = cid
        return best_client

    # ----- Update & Analysis Methods -----
    def update_selection_information(self, selected_client_ids: List[str],
                                     client_gradients: dict) -> None:
        """
        Update each registered seller's selection history based on whether
        its update was selected by the server. Also update global selection patterns.
        """
        for cid in self.clients:
            was_selected = cid in selected_client_ids
            self.clients[cid]["selection_history"].append(was_selected)
            history = self.clients[cid]["selection_history"]
            self.clients[cid]["selection_rate"] = sum(history) / len(history)
            self.clients[cid]["rounds_participated"] += 1
            # Switch phase to "attack" if enough rounds have passed and selection rate is high.
            if (self.clients[cid]["rounds_participated"] >= self.benign_rounds and
                    self.clients[cid]["selection_rate"] > 0.8):
                self.clients[cid]["phase"] = "attack"
            else:
                self.clients[cid]["phase"] = "benign"
        # Collect gradients from selected sellers.
        selected_grads = {cid: grad for cid, grad in client_gradients.items() if cid in selected_client_ids}
        self.selected_history.append(selected_grads)
        if len(self.selected_history) > 10:
            self.selected_history.pop(0)
        self._analyze_selection_patterns()

    def _analyze_selection_patterns(self) -> None:
        """
        Analyze stored selected gradients to compute a centroid and average cosine similarity.
        This pattern information is used to adjust non-selected gradients.
        """
        all_selected = []
        for round_dict in self.selected_history:
            for grad in round_dict.values():
                all_selected.append(grad.flatten())
        if not all_selected:
            return
        all_tensor = torch.stack(all_selected)
        centroid = torch.mean(all_tensor, dim=0)
        total_sim = 0.0
        count = 0
        for i in range(len(all_selected)):
            for j in range(i + 1, len(all_selected)):
                sim = F.cosine_similarity(all_selected[i].unsqueeze(0),
                                          all_selected[j].unsqueeze(0))[0]
                total_sim += sim.item()
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.0
        self.selection_patterns = {"centroid": centroid, "avg_similarity": avg_sim}

    def adaptive_role_assignment(self) -> None:
        """
        Dynamically reassign roles to sellers based on their selection rates.
        For example, the top 20% become "attacker", the bottom 40% "explorer", and the remainder "hybrid."
        """
        selection_rates = {cid: self.clients[cid]["selection_rate"] for cid in self.clients}
        sorted_clients = sorted(selection_rates.items(), key=lambda x: x[1], reverse=True)
        num_clients = len(sorted_clients)
        top_cutoff = int(0.2 * num_clients)
        bottom_cutoff = int(0.6 * num_clients)
        for i, (cid, _) in enumerate(sorted_clients):
            if i < top_cutoff:
                self.clients[cid]["role"] = "attacker"
            elif i >= bottom_cutoff:
                self.clients[cid]["role"] = "explorer"
            else:
                self.clients[cid]["role"] = "hybrid"

    # ----- Selected Gradients & Update Methods -----
    def precompute_current_round_gradient(self, selected_info: Optional[dict] = None) -> None:
        """
        Update the internal storage of selected gradients.
        If selected_info is provided, use it; otherwise, query registered sellers.
        """
        self.selected_gradients = {}
        if selected_info:
            for seller_id, gradient in selected_info.items():
                if seller_id in self.registered_clients:
                    self.selected_gradients[seller_id] = self._ensure_tensor(gradient)
            return
        selected_ids = []
        for seller_id, seller in self.registered_clients.items():
            if hasattr(seller, 'selected_last_round') and seller.selected_last_round:
                base_model = copy.deepcopy(self.aggregator.global_model)
                base_model = base_model.to(self.device)
                gradient = seller.get_local_gradient(base_model)
                selected_ids.append(seller_id)
                self.selected_gradients[seller_id] = self._ensure_tensor(gradient)
        print(f"Selected adv sellers in last round: {selected_ids}")

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

    def get_selected_average(self) -> Optional[torch.Tensor]:
        """
        Compute and return the average gradient of all selected sellers.
        """
        if not self.selected_gradients:
            return None
        gradients = list(self.selected_gradients.values())
        gradients = [g.to(self.device) for g in gradients]
        avg_grad = torch.mean(torch.stack(gradients), dim=0)
        return avg_grad

    def update_nonselected_gradient(self,
                                    current_gradient: Union[torch.Tensor, List],
                                    strategy: Optional[str] = None) -> List[np.ndarray]:
        """
        Update the gradient for a non-selected seller based on the average gradient of selected sellers.
        Strategies include "mimic", "pivot", "knock_out", "slowdown", "cost_inflation", "camouflage", etc.
        """
        strat = strategy if strategy is not None else self.gradient_default_mode
        avg_grad = self.get_selected_average()

        # Helper function to safely convert to numpy
        def to_numpy(g):
            if isinstance(g, torch.Tensor):
                return g.cpu().numpy()
            elif isinstance(g, np.ndarray):
                return g
            else:
                raise TypeError(f"Expected torch.Tensor or numpy.ndarray but got {type(g)}.")

        # If no average gradient, just return current_gradient as numpy arrays.
        if avg_grad is None:
            if isinstance(current_gradient, list):
                return [to_numpy(g) for g in current_gradient]
            else:
                return [to_numpy(current_gradient)]

        is_list = isinstance(current_gradient, list)
        original_shapes = None
        if is_list:
            original_shapes = [g.shape for g in current_gradient]
            current_grad_tensor = self._ensure_tensor(current_gradient)
        else:
            current_grad_tensor = current_gradient.to(self.device)

        if strat == "mimic":
            new_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad
        elif strat == "pivot":
            new_grad = avg_grad.clone()
        elif strat == "knock_out":
            alpha_knock = min(self.alpha * 2, 1.0)
            new_grad = (1 - alpha_knock) * current_grad_tensor + alpha_knock * avg_grad
        elif strat == "slowdown":
            new_grad = 0.1 * current_grad_tensor
        elif strat == "cost_inflation":
            new_grad = self.cost_scale * avg_grad
        elif strat == "camouflage":
            aligned_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad
            new_grad = self.amplify_factor * aligned_grad
        else:
            new_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad

        # If the original gradient was a list, unflatten it to match the original shapes.
        if is_list and original_shapes:
            new_grad = self._unflatten_gradient(new_grad, original_shapes)

        # Ensure new_grad is a list (if it's a single tensor or array, wrap it in a list)
        if isinstance(new_grad, (torch.Tensor, np.ndarray)):
            new_grad = [new_grad]

        # Convert each element in the list to a NumPy array.
        return [to_numpy(t) for t in new_grad]

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

    def on_round_start(self) -> None:
        """Operations to be performed at the end of a round."""
        self.cur_round += 1
        if self.trigger_mode == "dynamic" and self.cur_round >= self.benign_rounds:
            # find the seller with the most selection rate
            is_first = self.cur_round == self.benign_rounds
            best_client_id = self.get_client_with_highest_selection_rate()
            best_client = self.registered_clients[best_client_id]
            # use the best seller's local to update the pattern
            best_client.upload_global_trigger(self.aggregator.global_model, first_attack=is_first, lr=0.01,
                                              num_steps=50)
        # save the result after tigger updates
        if self.cur_round >= self.benign_rounds:
            self.start_atk = True
        self.precompute_current_round_gradient()

    # ----- Reset and End-of-Round Handling -----
    def on_round_end(self) -> None:
        """Operations to be performed at the end of a round."""
        self.selected_gradients = {}

    def update_global_trigger(self, new_trigger: torch.Tensor) -> None:
        """
        Update the global trigger maintained by the coordinator.
        This new trigger will be used by all malicious sellers.
        """
        trigger = new_trigger.clone().detach().to(self.device)
        self.backdoor_generator.update_trigger(trigger)
        print("Coordinator: Global trigger updated.")


class GradientSeller(BaseSeller):
    """
    Seller that participates in federated learning by providing gradient updates
    instead of selling raw data.
    """

    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,
                 price_strategy: str = 'uniform',
                 dataset_name: str = 'dataset',
                 base_price: float = 1.0,
                 price_variation: float = 0.2,
                 save_path="",
                 device="cpu",
                 local_epochs=2, local_training_params=None):
        """.
        :param seller_id: Unique ID for the seller.
        :param local_data: The local dataset this seller holds for gradient computation.
        :param price_strategy: If needed, you can still keep a pricing concept or set to 'none'.
        :param base_price:  For some FL-based cost logic, or ignore if not used.
        :param price_variation: Variation factor for generating costs, if relevant.
        """
        super().__init__(
            seller_id=seller_id,
            dataset=local_data,  # We store the local dataset internally.
            price_strategy=price_strategy,
            base_price=base_price,
            price_variation=price_variation, save_path=save_path
            , device=device
        )

        # Possibly store local model parameters or placeholders.
        # E.g., we might keep them in this field after each training round:
        self.dataset_name = dataset_name
        self.local_model_params: Optional[np.ndarray] = None
        self.current_round = 0
        self.selected_last_round = False
        self.local_epochs = local_epochs
        self.local_training_params = local_training_params
        self.recent_metrics = None
        self.cur_upload_gradient_flt = None
        self.cur_local_gradient = None

    def set_local_model_params(self, params: np.ndarray):
        """Set (or update) local model parameters before computing gradient."""
        self.local_model_params = params

    def get_gradient_for_upload(self, global_model=None) -> (torch.Tensor, int):
        """
        Compute the gradient that will be sent to the central server

        :param global_model: Optional global model (will be deep copied)
        :return: Tuple (gradient, flattened_gradient, local_model, eval_results)
        """
        # 1. Determine the base model for local training
        if global_model is not None:
            # Deep copy the provided global model
            base_model = copy.deepcopy(global_model)
            print(f"[{self.seller_id}] Using provided global model.")
        else:
            try:
                # Load previous local model if no global model provided
                base_model = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved model found; using default initialization.")
                base_model = get_image_model(self.dataset_name)  # Create a new model with default initialization

        # Move the model to the correct device
        base_model = base_model.to(self.device)

        # 2. Train locally and obtain the gradient update
        try:
            # Call the MODIFIED local training method
            gradient, gradient_flt, updated_model, local_eval_res, training_stats = self._compute_local_grad(
                base_model, self.dataset  # Pass the actual dataset attribute
            )

            # Ensure training_stats is a dict, even if _compute_local_grad fails partially
            if not isinstance(training_stats, dict):
                logging.warning(f"Seller {self.seller_id}: _compute_local_grad did not return a valid stats dict.")
                training_stats = {'train_loss': None, 'compute_time_ms': None}


        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Error during _compute_local_grad: {e}", exc_info=True)
            # Return None gradient and empty stats on error? Or raise?
            # Returning None gradient signals failure to the marketplace loop
            return None, {'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}

        # Update internal counter
        self.cur_upload_gradient_flt = gradient_flt

        try:
            upload_bytes = estimate_byte_size(gradient)  # Use a helper to estimate size
        except Exception as e:
            logging.warning(f"Seller {self.seller_id}: Could not estimate gradient size: {e}")
            upload_bytes = None  # Indicate failure to estimate

        # 4. Add upload_bytes to the stats dictionary
        training_stats['upload_bytes'] = upload_bytes

        # 5. Return the gradient data and the completed stats dictionary
        #    Make sure the 'gradient' variable holds the data structure you intend to send
        #    (e.g., the dict of weight differences, NOT necessarily gradient_flt)
        return gradient, training_stats

    # def _compute_local_grad(self, base_model, dataset):
    #     """
    #     Train a local model and compute the gradient update.
    #
    #     :param base_model: The initial model for local training (already on correct device)
    #     :param dataset: Local data as a list of (image, label) tuples
    #     :return: Tuple (gradient, flattened_gradient, updated_model, eval_results)
    #     """
    #     # The base_model is already initialized and on the correct device
    #
    #     # Perform local training and get the gradient update
    #     # grad_update, grad_update_flt, local_model, local_eval_res, avg_train_loss
    #     grad_update, grad_update_flt, local_model, local_eval_res, avg_train_loss = local_training_and_get_gradient(
    #         base_model,
    #         list_to_tensor_dataset(dataset),
    #         batch_size=64,
    #         device=self.device,
    #         local_epochs=self.local_training_params["local_epochs"],
    #         lr=self.local_training_params["learning_rate"],
    #     )
    #
    #     # Clean up to help with memory
    #     torch.cuda.empty_cache()
    #
    #     return grad_update, grad_update_flt, local_model, local_eval_res

    def _compute_local_grad(self, base_model, dataset) -> Tuple[Any, Any, Any, Dict, Dict]:
        """
        MODIFIED: Train local model, compute gradient, AND gather training stats.

        Args:
            base_model: Initial model for local training (on correct device).
            dataset: Local data (expects structure usable by list_to_tensor_dataset).

        Returns:
            Tuple (gradient, gradient_flt, updated_model, local_eval_res, training_stats):
                gradient: Calculated gradient (e.g., weight diff).
                gradient_flt: Flattened gradient.
                updated_model: Model after local training.
                local_eval_res: Evaluation results (potentially basic).
                training_stats: {'train_loss': float|None, 'compute_time_ms': float|None}.
        """
        start_time = time.time()  # Start timing

        # Prepare dataset if needed
        try:
            # Use your actual conversion function
            tensor_dataset = list_to_tensor_dataset(dataset)
            if tensor_dataset is None:
                raise ValueError("Dataset conversion returned None.")
        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Failed to convert dataset: {e}", exc_info=True)
            stats_error = {'train_loss': None, 'compute_time_ms': (time.time() - start_time) * 1000}
            return None, None, base_model, {}, stats_error  # Return base model on dataset error

        # Call the MODIFIED external training function which now returns avg_loss
        try:
            # Check if local_training_params are available
            if not hasattr(self, 'local_training_params') or not self.local_training_params:
                raise ValueError("Missing 'local_training_params' attribute on seller.")

            # Call the function - IT MUST NOW RETURN 5 VALUES
            grad_update, grad_update_flt, local_model, local_eval_res, avg_train_loss = local_training_and_get_gradient(
                model=base_model,
                train_dataset=tensor_dataset,  # Pass the converted dataset
                batch_size=self.local_training_params.get('batch_size', 64),  # Use batch size from params
                device=self.device,
                # Ensure key names match your actual params dict
                local_epochs=self.local_training_params.get("epochs",
                                                            self.local_training_params.get("local_epochs", 1)),
                lr=self.local_training_params.get("learning_rate", self.local_training_params.get("lr", 0.01)),
                # Pass other necessary params if local_training_and_get_gradient needs them
            )

        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Error during call to local_training_and_get_gradient: {e}",
                          exc_info=True)
            end_time_error = time.time()
            stats_error = {'train_loss': None, 'compute_time_ms': (end_time_error - start_time) * 1000}
            # Return None for gradient components, keep original model, empty eval, computed stats
            return None, None, base_model, {}, stats_error

        # Calculate compute time
        end_time = time.time()
        compute_time_ms = (end_time - start_time) * 1000

        # Package the stats
        training_stats = {
            'train_loss': avg_train_loss,  # Use the value returned by the training function
            'compute_time_ms': compute_time_ms
        }

        # Clean up GPU memory (keep this if useful)
        torch.cuda.empty_cache()

        # Return all original values PLUS the new stats dict
        return grad_update, grad_update_flt, local_model, local_eval_res, training_stats

    def save_local_model(self, model: torch.nn.Module):
        """
        Save the local model parameters to disk for future rounds.
        """
        # Build the save path based on client_path and seller_id.
        save_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"[{self.seller_id}] Saved local model to {save_path}.")

    def load_local_model(self) -> Dict[str, torch.Tensor]:
        """
        Load the local model parameters from disk.
        """
        load_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
        state_dict = torch.load(load_path, map_location=self.device)
        return state_dict

    def record_federated_round(self,
                               round_number: int,
                               is_selected: bool,
                               final_model_params: Optional[Dict[str, torch.Tensor]] = None):
        """
        Record this seller's participation in a federated round.
        This may include whether it was selected, and (optionally) its final local model parameters.

        :param round_number: The current round index.
        :param is_selected:  Whether this seller's update was selected.
        :param final_model_params: Optionally, the final local model parameters.
        """
        record = {
            'round_number': round_number,
            'timestamp': pd.Timestamp.now().isoformat(),
            'selected': is_selected,
            'gradient': self.cur_upload_gradient_flt,
        }
        self.selected_last_round = is_selected
        # if final_model_params is not None:
        #     # Convert state_dict tensors to lists (or use another serialization as needed).
        #     record['final_model_params'] = {k: v.cpu().numpy().tolist() for k, v in final_model_params.items()}

    def round_end_process(self, round_number,
                          is_selected,
                          final_model_params=None):
        self.reset_current_local_gradient()
        self.record_federated_round(
            round_number,
            is_selected,
            final_model_params)

    def reset_current_local_gradient(self):
        self.cur_local_gradient = None

    # If you don't need the .get_data() returning "X" and "cost", you can override it:
    @property
    def get_data(self):
        """
        Overridden: Typically in FL, we might not 'sell' raw data.
        Return something if your code expects this method, or return empty.
        """
        return {
            "X": None,
            "cost": None,
        }

    @property
    def get_federated_history(self):
        return self.federated_round_history

    @property
    def local_model_path(self):
        return {self.exp_save_path}


class AdvancedPoisoningAdversarySeller(GradientSeller):
    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,
                 target_label: int,
                 poison_generator=None,
                 device: str = 'cpu',
                 poison_rate=0.1,
                 save_path: str = "",
                 local_epochs: int = 2,
                 dataset_name: str = "",
                 local_training_params: Optional[dict] = None,
                 is_sybil: bool = False,
                 benign_rounds=3,
                 sybil_coordinator: Optional['SybilCoordinator'] = None):
        super().__init__(seller_id, local_data, save_path=save_path, device=device,
                         local_epochs=local_epochs, dataset_name=dataset_name,
                         local_training_params=local_training_params)

        self.flip_target_label = target_label
        self.poison_generator = poison_generator

        self.cur_upload_gradient_flt = None
        self.is_sybil = is_sybil
        self.sybil_coordinator = sybil_coordinator
        self.cur_local_gradient = None
        self.selected_last_round = False
        self.benign_rounds = benign_rounds
        # Adversary behaviors registry: maps a mode to a function.
        self.adversary_behaviors = self.simple_flipping
        self.poison_rate = poison_rate

    def get_clean_gradient(self, base_model):
        """
        Compute the gradient on clean (benign) local data.
        """
        gradient, gradient_flt, updated_model, local_eval_res, training_stats = self._compute_local_grad(base_model,
                                                                                                         self.dataset)
        self.recent_metrics = local_eval_res
        return gradient

    def _flip_labels(self, data, fraction: float):
        """
        Insert a stealthy trigger into a fraction of images.
        """
        n = len(data)
        n_trigger = int(n * fraction)
        idxs = np.random.choice(n, size=n_trigger, replace=False)
        backdoor_data, clean_data = [], []
        for i, (img, label) in enumerate(data):
            if i in idxs:
                if self.poison_generator is None:
                    raise NotImplementedError(f"Cannot find the backdoor generator")
                else:
                    triggered_img = self.poison_generator.generate_poisoned_samples()
                backdoor_data.append((triggered_img, self.flip_target_label))
            else:
                clean_data.append((img, label))
        return backdoor_data, clean_data

    def simple_flipping(self, base_model):
        """
        Generates label-flipped data and computes the gradient on this poisoned dataset.
        The "flipping" refers to the label manipulation strategy.
        """
        logging.info(f"Starting simple_flipping attack generation (rate={self.poison_rate})")
        # 1. Generate the dataset with flipped labels using the correct generator
        # The generator handles selecting samples and flipping labels based on its mode.
        poisoned_dataset, original_labels = self.poison_generator.generate_poisoned_dataset(
            original_dataset=self.dataset,  # Pass the clean dataset
            poison_rate=self.poison_rate,
        )
        # poisoned_dataset now contains *all* samples, some with original labels, some flipped.

        # 2. Compute the gradient on the *entire* poisoned dataset
        # The effect comes from the mislabeled samples influencing the gradient calculation.
        logging.info(f"Computing gradient on combined dataset (size={len(poisoned_dataset)})")
        g_combined, g_combined_flt, _, _, _ = self._compute_local_grad(base_model, poisoned_dataset)

        # 3. Get original shapes (assuming g_combined is a list of Tensors/arrays per layer)
        if not g_combined:  # Handle case where model has no parameters / gradient is empty
            logging.warning("Gradient list 'g_combined' is empty.")
            return []  # Or handle appropriately

        # Determine shapes based on type (Tensor or NumPy)
        if isinstance(g_combined[0], torch.Tensor):
            original_shapes = [param.shape for param in g_combined]
        elif isinstance(g_combined[0], np.ndarray):
            original_shapes = [param.shape for param in g_combined]
        else:
            raise TypeError(f"Unexpected gradient type in g_combined: {type(g_combined[0])}")

        # 4. Unflatten the gradient (assuming g_combined_flt is NumPy and unflatten_np works)
        try:
            final_poisoned_gradient_np = unflatten_np(g_combined_flt, original_shapes)
        except NameError:
            raise RuntimeError("Function 'unflatten_np' is not defined.")
        except Exception as e:
            logging.error(f"Error during unflattening: {e}")
            raise  # Re-raise the error

        logging.info("Simple flipping gradient processed and unflattened.")
        # Returns the gradient computed on the label-flipped data, in the original structure (list of NumPy arrays)
        return final_poisoned_gradient_np

    def get_local_gradient(self, global_model=None):
        """
        Compute the local gradient using the selected adversary behavior.
        The behavior is selected via self.gradient_manipulation_mode.
        """
        if self.cur_local_gradient is not None:
            return self.cur_local_gradient

        if global_model is not None:
            base_model = global_model
        else:
            try:
                base_model = self.load_local_model()
            except Exception as e:
                base_model = get_image_model(self.dataset_name)
                base_model = base_model.to(self.device)

        # Select the behavior function from the registry; default to clean gradient.
        if self.sybil_coordinator.start_atk:
            behavior_func = self.simple_flipping
        else:
            behavior_func = self.get_clean_gradient

        # get local gradient
        local_gradient = behavior_func(base_model)
        self.cur_local_gradient = local_gradient
        return local_gradient

    # ============================
    # Coordinator Integration Methods
    # ============================
    def get_gradient_for_upload(self, global_model=None):
        """
        Compute the local gradient for upload.
        If not in a Sybil setting, return the local gradient directly.
        If Sybil and not selected last round, query the coordinator to update the gradient.
        """
        if global_model is not None:
            base_model = copy.deepcopy(global_model)
            print(f"[{self.seller_id}] Using provided global model.")
        else:
            try:
                base_model = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved model found; using default initialization.")
                base_model = get_image_model(self.dataset_name)

        base_model = base_model.to(self.device)
        local_grad = self.get_local_gradient(base_model)
        self.cur_upload_gradient_flt = local_grad

        if not self.is_sybil:
            return local_grad, {}

        # If selected in last round, do not modify gradient.
        if getattr(self, "selected_last_round", False):
            return local_grad, {}

        # Provide information to the coordinator and get an updated gradient.
        coordinated_grad = self._query_coordinator(local_grad)
        self.cur_upload_gradient_flt = coordinated_grad
        return coordinated_grad, {}

    def _query_coordinator(self, local_grad):
        """
        Send the current local gradient to the coordinator and get an updated gradient.
        This is an extension point—different coordinator integration strategies can be implemented here.
        """
        if self.sybil_coordinator is not None:
            # For example, the coordinator might adjust the gradient for non-selected sellers.
            updated_grad = self.sybil_coordinator.update_nonselected_gradient(local_grad)
            return updated_grad
        return local_grad

    def reset_current_local_gradient(self):
        """Reset cached gradient information."""
        self.cur_local_gradient = None
        self.cur_upload_gradient_flt = None

    # ============================
    # Federated Round Reporting
    # ============================
    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Record the result of a federated round.
        """
        record = {
            "round_number": round_number,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_selected": is_selected,
            "gradient": self.cur_upload_gradient_flt,
        }
        self.selected_last_round = is_selected
        # self.federated_round_history.append(record)

    def round_end_process(self, round_number: int, is_selected: bool,
                          final_model_params=None):
        """
        Process the end-of-round tasks: reset gradient cache and record round info.
        """
        self.reset_current_local_gradient()
        self.record_federated_round(round_number, is_selected, final_model_params)


class AdvancedBackdoorAdversarySeller(GradientSeller):
    """
    An advanced seller that supports multiple adversary behaviors.
    This seller can:
      1) Dynamically inject stealthy trigger patterns.
      2) Compute a local gradient that is a blend of benign and backdoor signals.
      3) Optionally align the final gradient with a guessed server gradient.
      4) Integrate with a coordinator to adjust its behavior (e.g., in a Sybil setting).
    """

    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,
                 target_label: int,
                 alpha_align: float = 0.5,
                 trigger_rate: float = 0.1,
                 poison_strength: float = 0.7,
                 clip_value: float = 0.01,
                 trigger_type: str = "blended_patch",
                 backdoor_generator=None,
                 device: str = 'cpu',
                 save_path: str = "",
                 local_epochs: int = 2,
                 dataset_name: str = "",
                 local_training_params: Optional[dict] = None,
                 gradient_manipulation_mode: str = "cmd",
                 is_sybil: bool = False,
                 benign_rounds=3,
                 sybil_coordinator: Optional['SybilCoordinator'] = None):
        super().__init__(seller_id, local_data, save_path=save_path, device=device,
                         local_epochs=local_epochs, dataset_name=dataset_name,
                         local_training_params=local_training_params)

        self.target_label = target_label
        self.alpha_align = alpha_align
        self.poison_strength = poison_strength
        self.clip_value = clip_value
        self.trigger_type = trigger_type
        self.backdoor_generator = backdoor_generator
        self.trigger_rate = trigger_rate

        # Pre-split data: inject triggers into a fraction of the local data.

        self.gradient_manipulation_mode = gradient_manipulation_mode
        self.cur_upload_gradient_flt = None
        self.is_sybil = is_sybil
        self.sybil_coordinator = sybil_coordinator
        self.cur_local_gradient = None
        self.federated_round_history = []
        self.selected_last_round = False
        self.benign_rounds = benign_rounds
        # # Register this seller with the coordinator if available.
        # if self.sybil_coordinator is not None:
        #     self.sybil_coordinator.register_seller(self)

        # Adversary behaviors registry: maps a mode to a function.
        self.adversary_behaviors = {
            "cmd": self.gradient_manipulation_cmd,
            "single": self.gradient_manipulation_single,
            "none": self.get_clean_gradient,
            # New strategies can be added here.
        }

    # ============================
    # Data Injection Methods
    # ============================
    def _inject_triggers(self, data: List[Tuple[torch.Tensor, int]], fraction: float):
        """
        Insert a stealthy trigger into a fraction of images.
        """
        n = len(data)
        n_trigger = int(n * fraction)
        idxs = np.random.choice(n, size=n_trigger, replace=False)
        backdoor_data, clean_data = [], []
        for i, (img, label) in enumerate(data):
            if i in idxs:
                if self.backdoor_generator is None:
                    raise NotImplementedError(f"Cannot find the backdoor generator")
                else:
                    triggered_img = self.backdoor_generator.apply_trigger_tensor(img)
                backdoor_data.append((triggered_img, self.target_label))
            else:
                clean_data.append((img, label))
        return backdoor_data, clean_data

    # ============================
    # Adversary Behavior Methods
    # ============================
    def get_clean_gradient(self, base_model):
        """
        Compute the gradient on clean (benign) local data.
        """
        gradient, gradient_flt, updated_model, local_eval_res, training_stats = self._compute_local_grad(base_model,
                                                                                                         self.dataset)
        self.recent_metrics = local_eval_res
        return gradient

    def gradient_manipulation_cmd(self, base_model):
        """
        Compute a gradient that combines benign and backdoor gradients.
        """
        grad_benign, g_benign_flt, _, _, _ = self._compute_local_grad(base_model, self.dataset)
        original_shapes = [param.shape for param in grad_benign]
        g_backdoor, g_backdoor_flt, _, _, _ = self._compute_local_grad(base_model, self.backdoor_data)
        final_poisoned_flt = ((1 - self.poison_strength) * g_benign_flt +
                              self.poison_strength * g_backdoor_flt)
        self.last_benign_grad = g_benign_flt
        final_poisoned = unflatten_np(final_poisoned_flt, original_shapes)
        return final_poisoned

    def gradient_manipulation_single(self, base_model):
        """
        Compute the gradient on combined (backdoor + clean) data.
        """
        backdoor_data, clean_data = self._inject_triggers(self.dataset, self.trigger_rate)
        g_combined, g_combined_flt, _, _, _ = self._compute_local_grad(base_model, backdoor_data + clean_data)
        original_shapes = [param.shape for param in g_combined]
        final_poisoned = unflatten_np(g_combined_flt, original_shapes)
        return final_poisoned

    def get_local_gradient(self, global_model=None):
        """
        Compute the local gradient using the selected adversary behavior.
        The behavior is selected via self.gradient_manipulation_mode.
        """
        if self.cur_local_gradient is not None:
            return self.cur_local_gradient

        if global_model is not None:
            base_model = global_model
        else:
            try:
                base_model = self.load_local_model()
            except Exception as e:
                base_model = get_image_model(self.dataset_name)
                base_model = base_model.to(self.device)

        # Select the behavior function from the registry; default to clean gradient.
        if self.sybil_coordinator.start_atk:
            behavior_func = self.adversary_behaviors.get(self.gradient_manipulation_mode, self.get_clean_gradient)
        else:
            behavior_func = self.get_clean_gradient

        # get local gradient
        local_gradient = behavior_func(base_model)
        self.cur_local_gradient = local_gradient
        return local_gradient

    # ============================
    # Coordinator Integration Methods
    # ============================
    def get_gradient_for_upload(self, global_model=None):
        """
        Compute the local gradient for upload.
        If not in a Sybil setting, return the local gradient directly.
        If Sybil and not selected last round, query the coordinator to update the gradient.
        """
        if global_model is not None:
            base_model = copy.deepcopy(global_model)
            print(f"[{self.seller_id}] Using provided global model.")
        else:
            try:
                base_model = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved model found; using default initialization.")
                base_model = get_image_model(self.dataset_name)

        base_model = base_model.to(self.device)
        local_grad = self.get_local_gradient(base_model)
        self.cur_upload_gradient_flt = local_grad

        if not self.is_sybil:
            return local_grad, {}

        # If selected in last round, do not modify gradient.
        if getattr(self, "selected_last_round", False):
            return local_grad, {}

        # Provide information to the coordinator and get an updated gradient.
        coordinated_grad = self._query_coordinator(local_grad)
        self.cur_upload_gradient_flt = coordinated_grad
        return coordinated_grad, {}

    def _query_coordinator(self, local_grad):
        """
        Send the current local gradient to the coordinator and get an updated gradient.
        This is an extension point—different coordinator integration strategies can be implemented here.
        """
        if self.sybil_coordinator is not None:
            # For example, the coordinator might adjust the gradient for non-selected sellers.
            updated_grad = self.sybil_coordinator.update_nonselected_gradient(local_grad)
            return updated_grad
        return local_grad

    def reset_current_local_gradient(self):
        """Reset cached gradient information."""
        self.cur_local_gradient = None
        self.cur_upload_gradient_flt = None

    # ============================
    # Federated Round Reporting
    # ============================
    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Record the result of a federated round.
        """
        record = {
            "round_number": round_number,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_selected": is_selected,
            "gradient": self.cur_upload_gradient_flt,
        }
        self.selected_last_round = is_selected

    def round_end_process(self, round_number: int, is_selected: bool,
                          final_model_params=None):
        """
        Process the end-of-round tasks: reset gradient cache and record round info.
        """
        self.reset_current_local_gradient()
        self.record_federated_round(round_number, is_selected, final_model_params)

    def upload_global_trigger(self, global_model, first_attack=False, lr=0.01, num_steps=50, lambda_param=1):
        """
        If this seller is selected and designated as the trigger broadcaster,
        update the trigger and then broadcast the new trigger to all malicious sellers
        via the coordinator.
        """
        print(f"[{self.seller_id}] Broadcasting new trigger...")
        new_trigger = self.get_new_trigger(global_model, first_attack=first_attack, lr=lr, num_steps=num_steps,
                                           lambda_param=lambda_param)
        if self.sybil_coordinator is not None:
            self.sybil_coordinator.update_global_trigger(new_trigger)
        return new_trigger

    def get_new_trigger(self, global_model, first_attack, lr=0.01, num_steps=50, lambda_param=1):
        # For first attack (round 0), use loss alignment then gradient alignment
        # first load the base trigger
        trigger = self.backdoor_generator.get_trigger().detach().to(self.device).requires_grad_(True)
        model = global_model.to(self.device)
        if first_attack:
            # First optimize trigger with both objectives
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=True, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=0.0)

            # Then use gradient alignment
            tmp_trigger = self.trigger_opt(model, tmp_trigger, first_attack=False, trigger_lr=lr,
                                           num_steps=num_steps,
                                           lambda_param=1.0)
        else:
            # For subsequent rounds, only use gradient alignment
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=False, trigger_lr=lr,
                                           num_steps=num_steps,
                                           lambda_param=1.0)

        return tmp_trigger

    def trigger_opt(self, model, trigger, first_attack=False, trigger_lr=0.01, num_steps=50, lambda_param=1.0):
        """
        Optimize the backdoor trigger in two phases:
          Phase 1: Loss alignment (if first_attack is True) minimizes the classification loss
                   on backdoored data.
          Phase 2: Gradient alignment optimizes the trigger such that the difference between
                   the gradients on clean and backdoored data is minimized (weighted by lambda_param).

        Returns the updated trigger.
        """
        # Set model to evaluation mode
        model.eval()

        # (Optional) Ensure trigger has gradients enabled from the start.
        if not trigger.requires_grad:
            trigger = trigger.clone().detach().requires_grad_(True)

        # Create optimizer for the trigger
        trigger_optimizer = optim.Adam([trigger], lr=trigger_lr)
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Phase 1: Loss alignment (only for first attack)
        if first_attack:
            print("Phase 1: Loss alignment")
            for step in range(num_steps):
                total_loss = 0.0
                batches = 0

                for data, _ in dataloader:
                    # If data is missing a channel dimension, add it.
                    if data.dim() == 3:  # [batch, 28, 28]
                        data = data.unsqueeze(1)  # becomes [batch, 1, 28, 28]
                    data = data.to(self.device)

                    batches += 1
                    backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)

                    # Create target labels for the backdoor task for the entire batch
                    backdoor_labels = torch.full((data.shape[0],), self.target_label,
                                                 dtype=torch.long, device=self.device)

                    # Forward pass – minimize classification loss for the backdoor task
                    outputs = model(backdoored_data)
                    loss = criterion(outputs, backdoor_labels)
                    total_loss += loss.item()

                    # Backward pass and optimization
                    trigger_optimizer.zero_grad()
                    loss.backward()
                    trigger_optimizer.step()

                    # Clip trigger values to valid range [0, 1]
                    with torch.no_grad():
                        trigger.clamp_(0, 1)

                    # Limit number of batches per step for efficiency
                    if batches >= 10:
                        break
                if (step + 1) % 10 == 0:
                    print(f"Phase 1 - Step {step + 1}, Loss: {total_loss / batches:.4f}")

        # Phase 2: Gradient alignment optimization
        print(f"Phase 2: Gradient alignment optimization (λ={lambda_param})")
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        # Ensure trigger requires grad in Phase 2
        if not trigger.requires_grad:
            trigger = trigger.clone().detach().requires_grad_(True)

        for step in range(num_steps):
            total_gradient_distance = 0.0
            total_backdoor_loss = 0.0
            batches_processed = 0

            for data, label in dataloader:
                # Ensure proper shape: [batch, 1, 28, 28]
                if data.dim() == 3:
                    data = data.unsqueeze(1)
                data, label = data.to(self.device), label.to(self.device)
                batches_processed += 1

                # Create backdoored data and corresponding target labels
                backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)
                backdoor_labels = torch.full((data.shape[0],), self.target_label,
                                             dtype=torch.long, device=self.device)

                # Compute clean loss and store clean gradients
                model.zero_grad()
                clean_outputs = model(data)
                clean_loss = criterion(clean_outputs, label)
                clean_loss.backward(retain_graph=True)
                clean_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        clean_grads[name] = param.grad.clone()

                # Compute backdoor loss and gradients
                model.zero_grad()
                backdoor_outputs = model(backdoored_data)
                backdoor_loss = criterion(backdoor_outputs, backdoor_labels)
                backdoor_loss.backward(retain_graph=True)

                # Compute gradient distance (L2 norm squared difference)
                gradient_distance = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if param.grad is not None and name in clean_grads:
                        distance = torch.sum((param.grad - clean_grads[name]) ** 2)
                        gradient_distance += distance

                # Combined loss for trigger optimization
                combined_loss = lambda_param * gradient_distance + (1 - lambda_param) * backdoor_loss

                total_gradient_distance += gradient_distance.item()
                total_backdoor_loss += backdoor_loss.item()

                # Optimize trigger using the combined loss
                trigger_optimizer.zero_grad()
                if lambda_param > 0:
                    trigger_grad = torch.autograd.grad(combined_loss, trigger, retain_graph=True)[0]
                else:
                    trigger_grad = torch.autograd.grad(backdoor_loss, trigger)[0]
                trigger.grad = trigger_grad
                trigger_optimizer.step()

                with torch.no_grad():
                    trigger.clamp_(0, 1)

                # Break after processing a few batches for efficiency
                if batches_processed >= 10:
                    break
            if (step + 1) % 10 == 0:
                print(
                    f"Phase 2 - Step {step + 1}, Gradient Distance: {total_gradient_distance / batches_processed:.4f}, "
                    f"Backdoor Loss: {total_backdoor_loss / batches_processed:.4f}")

        # Detach and return the updated trigger
        trigger = trigger.detach()
        return trigger


def global_clip_np(arr, max_norm: float) -> np.ndarray:
    current_norm = np.linalg.norm(arr)
    if current_norm > max_norm:
        scale = max_norm / (current_norm + 1e-8)
        return arr * scale
    return arr


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    flat_params = []
    for key, param in state_dict.items():
        flat_params.append(param.detach().cpu().numpy().ravel())
    return np.concatenate(flat_params)


def unflatten_state_dict(model, flat_params: np.ndarray) -> dict:
    new_state_dict = {}
    pointer = 0
    for key, param in model.state_dict().items():
        numel = param.numel()
        # Slice the flat_params to match this parameter's number of elements.
        param_flat = flat_params[pointer:pointer + numel]
        # Reshape to the original shape.
        new_state_dict[key] = torch.tensor(param_flat.reshape(param.shape), dtype=param.dtype)
        pointer += numel
    return new_state_dict


def unflatten_np(flat_array, shapes):
    """
    Unflatten a 1D NumPy array back into a list of arrays with the provided shapes.

    Parameters:
      flat_array (np.ndarray): The flattened array.
      shapes (list of tuple): List of shapes corresponding to the original arrays.

    Returns:
      arrays (list of np.ndarray): The unflattened arrays.
    """
    arrays = []
    start = 0
    for shape in shapes:
        num_elements = np.prod(shape)
        segment = flat_array[start:start + num_elements]
        arrays.append(segment.reshape(shape))
        start += num_elements
    return arrays
