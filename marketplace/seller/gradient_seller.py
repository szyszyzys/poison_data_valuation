from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch

from general_utils.data_utils import list_to_tensor_dataset
from marketplace.seller.seller import BaseSeller
from model.utils import get_model, local_training_and_get_gradient


# You already have these classes in your project:
# from your_seller_module import GradientSeller, SellerStats
# from train import compute_loss, etc. (if needed)
# from dataset import dataset_output_dim (if needed)

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

    def set_local_model_params(self, params: np.ndarray):
        """Set (or update) local model parameters before computing gradient."""
        self.local_model_params = params

    def get_gradient(self, global_params: Optional[Dict[str, torch.Tensor]] = None) -> (torch.Tensor, int):
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

        self.current_round += 1
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
            'event_type': 'federated_round',
            'round_number': round_number,
            'timestamp': pd.Timestamp.now().isoformat(),
            'selected': is_selected,
            "metrics_local": self.recent_metrics
        }
        self.selected_last_round = is_selected
        if final_model_params is not None:
            # Convert state_dict tensors to lists (or use another serialization as needed).
            record['final_model_params'] = {k: v.cpu().numpy().tolist() for k, v in final_model_params.items()}
        self.federated_round_history.append(record)

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
                 poison_strength: float = 0.7,
                 clip_value: float = 0.01,
                 trigger_type: str = "blended_patch",
                 backdoor_generator=None,
                 device='cpu',
                 save_path="",
                 local_epochs=2,
                 dataset_name="",
                 local_training_params=None,
                 gradient_manipulation_mode="cmd"
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
        self.backdoor_data, self.clean_data = self._inject_triggers(local_data, poison_strength)
        self.local_training_params = local_training_params
        self.gradient_manipulation_mode = gradient_manipulation_mode

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

    def get_gradient(self, global_params: Dict[str, torch.Tensor] = None, align_global=False) -> [np.ndarray]:
        """
        Return a single 'final' gradient that merges:
          1) benign gradient
          2) backdoor gradient
          3) partial alignment with a guessed server gradient.

        Returns a tuple where the first element is the flattened final gradient (np.ndarray)
        and the second element is, for example, the number of gradient segments.
        """
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
            final_poisoned = self.gradient_manipulation_cmd(base_params)
        elif self.gradient_manipulation_mode == "single":
            final_poisoned = self.gradient_manipulation_single(base_params)
        else:
            raise NotImplementedError(f"No current poison mode: {self.gradient_manipulation_mode}")

        return final_poisoned

    def gradient_manipulation_cmd(self, base_params, previous_selection=None):
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
            final_poisoned_flt = np.clip(g_backdoor_flt, -self.clip_value, self.clip_value)
            original_shapes = [param.shape for param in g_backdoor_update]
        if previous_selection:
            final_poisoned_flt = self.alpha_align * final_poisoned_flt + (1 - self.alpha_align) * previous_selection
        self.last_poisoned_grad = final_poisoned_flt
        final_poisoned = global_clip_np(final_poisoned_flt, 1)
        final_poisoned = unflatten_np(final_poisoned, original_shapes)
        return final_poisoned

    def gradient_manipulation_single(self, base_params):
        g_backdoor_update, g_backdoor_flt, local_model_malicious, local_eval_res_m = self._compute_local_grad(
            base_params,
            self.backdoor_data + self.clean_data)
        # final_poisoned_flt = g_backdoor_flt
        final_poisoned_flt = np.clip(g_backdoor_flt, -self.clip_value, self.clip_value)
        original_shapes = [param.shape for param in g_backdoor_update]
        self.last_poisoned_grad = final_poisoned_flt
        final_poisoned = global_clip_np(final_poisoned_flt, 1)
        final_poisoned = unflatten_np(final_poisoned, original_shapes)
        return final_poisoned

    # def get_gradient(self, global_params: Dict[str, torch.Tensor] = None, align_global=False) -> np.ndarray:
    #     """
    #     Return a single 'final' gradient that merges:
    #       1) benign gradient
    #       2) backdoor gradient
    #       3) partial alignment w/ a guessed server gradient
    #     """
    #     if global_params is not None:
    #         base_params = global_params
    #     else:
    #         try:
    #             base_params = self.load_local_model()
    #             print(f"[{self.seller_id}] Loaded previous local model.")
    #         except Exception as e:
    #             print(f"[{self.seller_id}] No saved local model found; using default initialization.")
    #             model = get_model(self.dataset_name)
    #             base_params = model.state_dict()  # or load_param("f") as your fallback
    #
    #     # 1) Compute benign gradient
    #     if self.poison_strength != 1:
    #         grad_benign_update, g_benign_flt, local_model_benign = self._compute_local_grad(base_params,
    #                                                                                         self.clean_data)
    #
    #         # Example usage:
    #         # Suppose grad_benign_update is a list of tensors (or numpy arrays) that you flattened.
    #         # Get the original shapes:
    #         original_shapes = [param.shape for param in grad_benign_update]
    #
    #         # 2) Compute backdoor gradient
    #         g_backdoor_update, g_backdoor_flt, local_model_malicious = self._compute_local_grad(base_params,
    #                                                                                             self.backdoor_data)
    #
    #         # 3) Combine them:
    #         final_poisoned_flt = (1 - self.poison_strength) * g_benign_flt + (self.poison_strength) * g_backdoor_flt
    #         if align_global:
    #             # 4) Estimate server grad (in black-box, we might guess near zero or track old updates)
    #             server_guess = np.random.randn(final_poisoned_flt.shape[0]) * 0.0001
    #
    #             # 5) final_poisoned = alpha_align * raw_poison + (1 - alpha_align)*server_guess
    #             final_poisoned = self.alpha_align * final_poisoned_flt + (1 - self.alpha_align) * server_guess
    #
    #         # 6) Clip to aggregator’s clamp
    #         final_poisoned_flt = np.clip(final_poisoned_flt, -self.clip_value, self.clip_value)
    #         final_poisoned = unflatten_np(final_poisoned_flt, original_shapes)
    #         self.last_benign_grad = np.clip(g_benign_flt, -self.clip_value, self.clip_value)
    #     else:
    #         g_backdoor_update, g_backdoor_flt, local_model_malicious = self._compute_local_grad(base_params,
    #                                                                                             self.backdoor_data)
    #         final_poisoned_flt = np.clip(g_backdoor_flt, -self.clip_value, self.clip_value)
    #         original_shapes = [param.shape for param in g_backdoor_update]
    #         final_poisoned = unflatten_np(final_poisoned_flt, original_shapes)
    #     # store for analysis
    #     self.last_poisoned_grad = final_poisoned_flt
    #     cur_local_model = get_model(self.dataset_name)
    #     updated_params_flat = flatten_state_dict(base_params) - final_poisoned_flt
    #
    #     # Convert the updated flat parameters back into the model's state dict format.
    #     new_state_dict = unflatten_state_dict(cur_local_model, updated_params_flat)
    #     cur_local_model.load_state_dict(new_state_dict)
    #     # Load the updated parameters into the model.
    #
    #     # Load the updated parameters into the model.
    #     self.save_local_model(cur_local_model)
    #
    #     return final_poisoned

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
            "benign_grad_norm": float(
                np.linalg.norm(self.last_benign_grad)) if self.last_benign_grad is not None else None,
            "poisoned_grad_norm": float(
                np.linalg.norm(self.last_poisoned_grad)) if self.last_poisoned_grad is not None else None
        }
        self.selected_last_round = is_selected
        self.federated_round_history.append(record)


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
# def unflatten_np(flat_array: np.ndarray, param_shapes: list):
#     """
#     Unflatten a 1D numpy array into a list of numpy arrays with specified shapes.
#
#     Parameters
#     ----------
#     flat_array : np.ndarray
#         The flattened array (1D).
#     param_shapes : list of tuple
#         A list of shapes (e.g., [(3, 3), (3,), ...]) corresponding to each parameter.
#
#     Returns
#     -------
#     list of np.ndarray
#         A list of numpy arrays reshaped to the corresponding shapes.
#     """
#     results = []
#     current_pos = 0
#     for shape in param_shapes:
#         # Calculate how many elements this parameter has.
#         num_elements = np.prod(shape)
#         # Extract the segment and reshape it.
#         segment = flat_array[current_pos:current_pos + num_elements].reshape(shape)
#         results.append(segment)
#         current_pos += num_elements
#     return results
