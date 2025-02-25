import collections
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch

from general_utils.data_utils import list_to_tensor_dataset
from marketplace.seller.seller import BaseSeller
from model.utils import get_model, local_training_and_get_gradient, apply_gradient_update


# You already have these classes in your project:
# from your_seller_module import GradientSeller, SellerStats
# from train import compute_loss, etc. (if needed)
# from dataset import dataset_output_dim (if needed)
class SybilCoordinator:
    """
    A comprehensive SybilCoordinator to coordinate malicious sellers (Sybil identities)
    in a federated learning setup where selection is based solely on gradient similarity.

    The coordinator maintains a registry of malicious sellers and stores the gradients
    reported by those that were selected in the last round. It then provides a method for
    non-selected sellers to update (perturb) their local gradients based on the average
    gradient of the selected ones. This helps the adversaries coordinate their attack.
    """

    def __init__(self, default_mode="mimic", alpha=0.5, amplify_factor=2.0, cost_scale=1.5):
        """
        :param default_mode: Default strategy to use for non-selected gradient updates.
                             Options: "mimic", "pivot", "knock_out", "slowdown", "cost_inflation", "camouflage".
        :param alpha: Alignment factor (0 < alpha <= 1) for blending strategies.
        :param amplify_factor: Factor used in "camouflage" mode to amplify the coordinated gradient.
        :param cost_scale: Scale factor for "cost_inflation" mode.
        """
        self.default_mode = default_mode
        self.alpha = alpha
        self.amplify_factor = amplify_factor
        self.cost_scale = cost_scale

        # Dictionary to store gradients from malicious sellers that were selected last round.
        # Format: { seller_id (str): gradient (torch.Tensor) }
        self.selected_gradients = {}
        # Also keep a registry of seller instances (if needed for additional coordination).
        self.registered_clients = collections.OrderedDict()

    def register_seller(self, seller):
        """
        Register a malicious seller (client). The seller must have a unique seller_id attribute.
        """
        self.registered_clients[seller.seller_id] = seller

    def precompute_current_round_gradient(self, selected_info: dict):
        """
        Update the coordinator with the gradient updates from malicious sellers that were
        selected in the last round.

        :param selected_info: Dictionary mapping seller_id (str) to its gradient (torch.Tensor).
        """
        # precompute the gradient of each local client.
        for cdx, seller in self.registered_clients.items():
            gradient = seller.get_local_gradient()
            # only record the gradient why are selected
            if seller.selected_last_round:
                self.selected_gradients = {cdx: gradient.clone()}

    def get_selected_average(self) -> torch.Tensor:
        """
        Compute and return the average gradient from the selected malicious sellers.

        :return: A torch.Tensor representing the average gradient, or None if no gradients stored.
        """
        if not self.selected_gradients:
            return None
        gradients = list(self.selected_gradients.values())
        avg_grad = torch.mean(torch.stack(gradients), dim=0)
        return avg_grad

    def update_nonselected_gradient(self, current_gradient: torch.Tensor, strategy: str = None) -> torch.Tensor:
        """
        Given a non-selected seller's current gradient, update it based on the average gradient of
        the selected malicious sellers using the specified strategy.

        :param current_gradient: The current local gradient of a malicious seller (torch.Tensor).
        :param strategy: The strategy to use; if None, uses the default mode.
                         Options: "mimic", "pivot", "knock_out", "slowdown", "cost_inflation", "camouflage".
        :return: The updated gradient (torch.Tensor).
        """
        strat = strategy if strategy is not None else self.default_mode
        avg_grad = self.get_selected_average()
        if avg_grad is None:
            # No information available; return the current gradient unchanged.
            return current_gradient

        if strat == "mimic":
            # Blend current gradient with the average.
            new_grad = (1 - self.alpha) * current_gradient + self.alpha * avg_grad
        elif strat == "pivot":
            # Fully replace the current gradient with the average.
            new_grad = avg_grad.clone()
        elif strat == "knock_out":
            # Use a higher alignment factor (e.g., double alpha, capped at 1.0).
            alpha_knock = min(self.alpha * 2, 1.0)
            new_grad = (1 - alpha_knock) * current_gradient + alpha_knock * avg_grad
        elif strat == "slowdown":
            # Scale down the current gradient to stall progress.
            new_grad = 0.1 * current_gradient
        elif strat == "cost_inflation":
            # Scale up the average gradient to inflate the buyer's cost.
            new_grad = self.cost_scale * avg_grad
        elif strat == "camouflage":
            # Align with the average then amplify the result.
            aligned_grad = (1 - self.alpha) * current_gradient + self.alpha * avg_grad
            new_grad = self.amplify_factor * aligned_grad
        else:
            # Default to mimic if unknown strategy.
            new_grad = (1 - self.alpha) * current_gradient + self.alpha * avg_grad

        return new_grad

    def reset(self):
        """
        Reset the stored selected gradients for the next round.
        """
        self.selected_gradients = {}

    def on_round_end(self):
        self.reset()


class GradientSeller(BaseSeller):
    """
    Seller that participates in federated learning by providing gradient updates
    instead of selling raw data.
    """

    def __init__(self,
                 seller_id: str,
                 local_data: List[Tuple[torch.Tensor, int]],
                 price_strategy: str = 'uniform',
                 dataset_name: str = 'dataset',
                 base_price: float = 1.0,
                 price_variation: float = 0.2,
                 save_path="",
                 device="cpu",
                 local_epochs=2, local_training_params=None):
        """
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
        self.cur_gradient = None

    def set_local_model_params(self, params: np.ndarray):
        """Set (or update) local model parameters before computing gradient."""
        self.local_model_params = params

    def get_gradient_for_upload(self, global_params: Optional[Dict[str, torch.Tensor]] = None) -> (torch.Tensor, int):
        """
        Compute the gradient that finally sent to the central server

        :param global_params: Optional global model parameters.
        :return: Tuple (flattened_gradient, data_size)
        """
        # 1. Determine the base parameters for local training.
        if global_params is not None:
            base_params = global_params
        else:
            try:
                base_params = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved local model found; using default initialization.")
                base_params = None  # or load_param("f") as your fallback

        # 2. Train locally starting from base_params, obtain the local gradient update
        gradient, gradient_flt, updated_model, local_eval_res = self._compute_local_grad(base_params, self.dataset)
        self.recent_metrics = local_eval_res
        # 3. Save the updated local model for future rounds.
        # self.save_local_model(updated_model)

        self.current_round += 1
        self.cur_upload_gradient_flt = gradient
        return gradient

    def _compute_local_grad(self, base_params: Dict[str, torch.Tensor],
                            dataset: List[Tuple[torch.Tensor, int]]) -> (torch.Tensor, torch.nn.Module):
        """
        Build and update a local model using the base parameters and local data.
        This function:
          1) Builds a local model using get_model()
          2) Loads the base parameters into the model
          3) Trains the model locally and computes a flattened gradient update.
          4) Returns the flattened gradient and the updated model.

        :param base_params: The initial parameters for the local model.
        :param dataset: Local data as a list of (image, label) tuples.
        :return: Tuple (flattened_gradient, updated_model)
        """
        # Create a new model instance
        model = get_model(self.dataset_name)
        # Load base parameters into the model
        if base_params:
            model.load_state_dict(base_params)
        else:
            self.save_local_model(model)
        model = model.to(self.device)

        # Perform local training and get the flattened gradient update.
        grad_update, grad_update_flt, local_model, local_eval_res = local_training_and_get_gradient(
            model, list_to_tensor_dataset(dataset), batch_size=64, device=self.device,
            local_epochs=self.local_training_params["epochs"], lr=self.local_training_params["lr"]
        )

        # Optionally, you might want to clip the gradient here.
        # flat_update = torch.clamp(flat_update, -self.clip_value, self.clip_value)
        del model
        torch.cuda.empty_cache()
        return grad_update, grad_update_flt, local_model, local_eval_res

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
        self.federated_round_history.append(record)

    def round_end_process(self, round_number,
                          is_selected,
                          final_model_params=None):
        self.reset_current_local_gradient()
        self.record_federated_round(
            round_number,
            is_selected,
            final_model_params)

    def reset_current_local_gradient(self):
        self.cur_gradient = None

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


class AdvancedBackdoorAdversarySeller(GradientSeller):
    """
    A more sophisticated backdoor attacker that:
      1) Dynamically inserts a stealthy trigger pattern into a fraction of images.
      2) Blends the backdoor gradient with the benign gradient.
      3) Aligns the final gradient with a guessed server gradient to remain an inlier.
    """

    def __init__(self,
                 seller_id: str,
                 local_data: List[Tuple[torch.Tensor, int]],
                 target_label: int,
                 alpha_align: float = 0.5,
                 trigger_rate=0.1,
                 poison_strength: float = 0.7,
                 clip_value: float = 0.01,
                 trigger_type: str = "blended_patch",
                 backdoor_generator=None,
                 device='cpu',
                 save_path="",
                 local_epochs=2,
                 dataset_name="",
                 local_training_params=None,
                 gradient_manipulation_mode="cmd",
                 is_sybil=False,
                 sybil_coordinator: SybilCoordinator = None
                 ):
        """
        :param local_data:        List[(image_tensor, label_int)] for the local training set.
        :param target_label:      The label the attacker wants the model to predict for triggered images.
        :param trigger_fraction:  Fraction of local data to be turned into backdoor samples.
        :param alpha_align:       How strongly to align with server guess (0 -> purely backdoor, 1 -> purely guess).
        :param poison_strength:   Weighting factor for combining backdoor and benign gradients.
        :param clip_value:        Max abs value for gradient components (aggregator clamp).
        :param trigger_type:      e.g. "blended_patch", "invisible", "random_noise_patch", etc.
        """
        super().__init__(seller_id, local_data, save_path=save_path, device=device, local_epochs=local_epochs,
                         dataset_name=dataset_name, local_training_params=local_training_params)
        self.target_label = target_label
        self.alpha_align = alpha_align
        self.poison_strength = poison_strength
        self.clip_value = clip_value
        self.trigger_type = trigger_type

        # For analysis: store the last "benign" and "poisoned" gradients
        self.last_benign_grad = None
        self.last_poisoned_grad = None

        # Pre-split data
        self.backdoor_generator = backdoor_generator
        self.backdoor_data, self.clean_data = self._inject_triggers(local_data, trigger_rate)
        self.local_training_params = local_training_params
        self.gradient_manipulation_mode = gradient_manipulation_mode
        self.cur_upload_gradient_flt = None
        self.is_sybil = is_sybil
        self.sybil_coordinator = sybil_coordinator
        self.cur_gradient = None
        self.sybil_coordinator.register_seller(self)

    def _inject_triggers(self, data: List[Tuple[torch.Tensor, int]], fraction: float):
        """
        Insert a small, stealthy pattern into a fraction of images
        and change their label to self.target_label.
        """
        n = len(data)
        n_trigger = int(n * fraction)
        idxs = np.random.choice(n, size=n_trigger, replace=False)

        # We'll build new data lists
        backdoor_data = []
        clean_data = []

        for i, (img, label) in enumerate(data):
            if i in idxs:
                # create a triggered version of 'img'
                if self.backdoor_generator is None:
                    triggered_img = self._apply_stealth_trigger(img)
                else:
                    triggered_img = self.backdoor_generator.apply_trigger_tensor(img)
                backdoor_data.append((triggered_img, self.target_label))
            else:
                clean_data.append((img, label))

        return backdoor_data, clean_data

    def _apply_stealth_trigger(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Modify the input image with a 'stealthy' pattern.
        Examples:
          - Blending a small patch with alpha < 0.2 so it's barely visible
          - Random noise in a corner
          - 'Invisible' triggers using transparency
        We'll do a small 'blended patch' as an example.
        """
        # Let's assume (C,H,W) shape
        # For demonstration, place a small patch in the top-left corner
        # with a random pattern that has low alpha blending
        c, h, w = img_tensor.shape
        patch_size = 4
        patch = torch.rand(c, patch_size, patch_size)  # random pattern
        alpha = 0.2  # blend ratio

        # Inset the patch at top-left
        # shape => [c, patch_size, patch_size]
        triggered_img = img_tensor.clone()
        triggered_img[:, :patch_size, :patch_size] = (
                (1 - alpha) * triggered_img[:, :patch_size, :patch_size] + alpha * patch
        )

        return triggered_img

    def get_gradient_for_upload(self, global_params: Dict[str, torch.Tensor] = None, align_global=False):
        """
        Compute the local gradient and, if this seller is not selected, query the coordinator
        to update the gradient based on coordinated malicious updates.

        :param global_params: Global model parameters (if needed for training).
        :param selected: Boolean flag indicating if this seller was selected in the previous round.
        :return: A tuple (final_gradient, data_size)
        """
        # Step 1: Compute the local malicious gradient.
        local_grad = self.get_local_gradient(global_params)
        # Save the computed gradient.
        # get the gradient, return if not doing sybil attack.
        self.cur_upload_gradient_flt = local_grad
        if not self.is_sybil:
            return local_grad
        # don't perturb as current gradient is selected in previous round
        if self.selected_last_round:
            return local_grad

        # todo perform apply global knowledge, manipluate the local gradient
        # self.sybil_coordinator.todo(local_gradient)
        local_grad = self.sybil_coordinator.update_nonselected_gradient(local_grad)
        self.cur_upload_gradient_flt = local_grad
        return local_grad

    def get_local_gradient(self, global_params: Dict[str, torch.Tensor] = None) -> [np.ndarray]:
        """
        Return a single 'final' gradient that merges:
          1) benign gradient
          2) backdoor gradient
          3) partial alignment with a guessed server gradient.

        Returns a tuple where the first element is the flattened final gradient (np.ndarray)
        and the second element is, for example, the number of gradient segments.
        """
        # if calculated before, return the cached value
        if self.cur_gradient:
            return self.cur_gradient

        if global_params is not None:
            base_params = global_params
        else:
            try:
                base_params = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved local model found; using default initialization.")
                base_params = None  # or load_param("f") as your fallback

        if self.gradient_manipulation_mode == "cmd":
            local_gradient = self.gradient_manipulation_cmd(base_params)
        elif self.gradient_manipulation_mode == "single":
            local_gradient = self.gradient_manipulation_single(base_params)
        elif self.gradient_manipulation_mode == "none":
            local_gradient = self.get_clean_gradient(base_params)
        else:
            raise NotImplementedError(f"No current poison mode: {self.gradient_manipulation_mode}")

        self.cur_gradient = local_gradient
        return local_gradient

    def get_clean_gradient(self, global_params: Optional[Dict[str, torch.Tensor]] = None) -> (torch.Tensor, int):
        """
        Compute the gradient with respect to a base model using this seller's local data.
        If global_params is provided, use them to initialize the local model.
        Otherwise, load the previously saved local model (if available) or fallback to a default.

        :param global_params: Optional global model parameters.
        :return: Tuple (flattened_gradient, data_size)
        """
        # 1. Determine the base parameters for local training.
        if global_params is not None:
            base_params = global_params
        else:
            try:
                base_params = self.load_local_model()
                print(f"[{self.seller_id}] Loaded previous local model.")
            except Exception as e:
                print(f"[{self.seller_id}] No saved local model found; using default initialization.")
                base_params = None  # or load_param("f") as your fallback

        # 2. Train locally starting from base_params, obtain the local gradient update
        gradient, gradient_flt, updated_model, local_eval_res = self._compute_local_grad(base_params, self.dataset)
        self.recent_metrics = local_eval_res
        # 3. Save the updated local model for future rounds.
        # self.save_local_model(updated_model)

        return gradient

    def gradient_manipulation_cmd(self, base_params):
        if self.poison_strength != 1:
            grad_benign_update, g_benign_flt, local_model_benign, local_eval_res_n = self._compute_local_grad(
                base_params,
                self.clean_data)
            original_shapes = [param.shape for param in grad_benign_update]

            # 2) Compute backdoor gradient
            g_backdoor_update, g_backdoor_flt, local_model_malicious, local_eval_res_m = self._compute_local_grad(
                base_params,
                self.backdoor_data)

            # 3) Combine them:
            final_poisoned_flt = (1 - self.poison_strength) * g_benign_flt + (self.poison_strength) * g_backdoor_flt

            self.last_benign_grad = g_benign_flt
        else:
            g_backdoor_update, g_backdoor_flt, local_model_malicious, local_eval_res_m = self._compute_local_grad(
                base_params,
                self.backdoor_data)
            # final_poisoned_flt = g_backdoor_flt
            final_poisoned_flt = g_backdoor_flt
            original_shapes = [param.shape for param in g_backdoor_update]
        # final_poisoned_flt = np.clip(final_poisoned_flt, -self.clip_value, self.clip_value)
        # final_poisoned = global_clip_np(final_poisoned_flt, 1)
        final_poisoned = unflatten_np(final_poisoned_flt, original_shapes)

        return final_poisoned

    def gradient_manipulation_single(self, base_params):
        g_backdoor_update, g_backdoor_flt, local_model_malicious, local_eval_res_m = self._compute_local_grad(
            base_params,
            self.backdoor_data + self.clean_data)
        # final_poisoned_flt = g_backdoor_flt
        # final_poisoned_flt = np.clip(g_backdoor_flt, -self.clip_value, self.clip_value)
        original_shapes = [param.shape for param in g_backdoor_update]
        # # final_poisoned = global_clip_np(final_poisoned_flt, 1)
        # final_poisoned = g_backdoor_update
        final_poisoned = unflatten_np(g_backdoor_flt, original_shapes)
        return final_poisoned

    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Tracks if we were selected. We can store additional info
        about the 'last_benign_grad' or 'last_poisoned_grad' if needed.
        """
        record = {
            "round_number": round_number,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_selected": is_selected,
            'gradient': self.cur_upload_gradient_flt,
        }
        self.selected_last_round = is_selected
        self.federated_round_history.append(record)

    def round_end_process(self, round_number,
                          is_selected,
                          final_model_params=None):
        self.reset_current_local_gradient()
        self.record_federated_round(
            round_number,
            is_selected,
            final_model_params)


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


def update_local_model_from_global(client: GradientSeller, dataset_name, aggregated_gradient):
    s_local_model_dict = client.load_local_model()
    s_local_model = get_model(dataset_name=dataset_name)
    # Load base parameters into the model
    s_local_model.load_state_dict(s_local_model_dict)
    cur_local_model = apply_gradient_update(s_local_model, aggregated_gradient)
    client.save_local_model(cur_local_model)
