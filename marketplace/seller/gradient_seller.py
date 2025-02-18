from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch

from marketplace.seller.seller import BaseSeller
from model.utils import get_model, load_model, local_training_and_get_gradient


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
                 save_path=""):
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
        )

        # Possibly store local model parameters or placeholders.
        # E.g., we might keep them in this field after each training round:
        self.dataset_name = dataset_name
        self.local_model_params: Optional[np.ndarray] = None

    def set_local_model_params(self, params: np.ndarray):
        """Set (or update) local model parameters before computing gradient."""
        self.local_model_params = params

    def get_gradient(self, global_params: np.ndarray = None) -> (np.ndarray, int):
        """
        Compute the gradient w.r.t. the global model using this seller's local data.

        :param global_params: The current global model parameters.
        :return: (gradient, data_size), for example
        """
        """
        Return a single 'final' gradient that merges:
          1) benign gradient
          2) backdoor gradient
          3) partial alignment w/ a guessed server gradient
        """
        # 1) Compute benign gradient
        if global_params is not None:
            base_param = global_params
        else:
            base_param = load_model(model, path, device)
        gradient = self._compute_local_grad(base_param, self.dataset)
        gradient_np = gradient.cpu().numpy()

        # store for analysis
        # self.last_benign_grad = np.clip(g_benign_np, -self.clip_value, self.clip_value)
        # self.last_poisoned_grad = final_poisoned

        return gradient, len(self.dataset)

    def _compute_local_grad(self, global_params: Dict[str, torch.Tensor],
                            dataset: List[Tuple[torch.Tensor, int]]) -> torch.Tensor:
        """
        In practice:
         1) Build local model from 'global_params'
         2) Run forward/backward on `dataset`
         3) Return flattened gradient
        We do a dummy example here for demonstration.
        """
        model = get_model(self.dataset_name)
        model = load_model(model=model, path=self.client_path, device=self.device)
        flat_update, train_size = local_training_and_get_gradient(model, self.dataset,
                                                                  batch_size=64,
                                                                  device=self.device,
                                                                  local_epochs=self.local_epochs,
                                                                  lr=0.01)

        # 4. Free the model (or let it go out of scope)
        del model  # Free memory if necessary
        return flat_update

    def record_federated_round(self,
                               round_number: int,
                               is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Record the seller's participation in a federated round.
        E.g. whether it computed a gradient, whether it was chosen for the final model,
        and possibly store its local or final model parameters.
        """
        record = {
            'event_type': 'federated_round',
            'round_number': round_number,
            'timestamp': pd.Timestamp.now().isoformat(),
            'selected': is_selected
        }
        if final_model_params is not None:
            # You might store them as a list or as a separate file if large
            record['final_model_params'] = final_model_params.tolist()

        self.federated_round_history.append(record)

        # Optionally update some stats about FL participation.
        # For instance:
        # self.stats.rounds_participated += 1
        # if is_selected:
        #     self.stats.rounds_selected += 1
        # etc.

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
                 trigger_fraction: float = 0.1,
                 alpha_align: float = 0.5,
                 poison_strength: float = 0.7,
                 clip_value: float = 0.01,
                 trigger_type: str = "blended_patch", save_path=""):
        """
        :param local_data:        List[(image_tensor, label_int)] for the local training set.
        :param target_label:      The label the attacker wants the model to predict for triggered images.
        :param trigger_fraction:  Fraction of local data to be turned into backdoor samples.
        :param alpha_align:       How strongly to align with server guess (0 -> purely backdoor, 1 -> purely guess).
        :param poison_strength:   Weighting factor for combining backdoor and benign gradients.
        :param clip_value:        Max abs value for gradient components (aggregator clamp).
        :param trigger_type:      e.g. "blended_patch", "invisible", "random_noise_patch", etc.
        """
        super().__init__(seller_id, local_data, save_path=save_path)
        self.target_label = target_label
        self.trigger_fraction = trigger_fraction
        self.alpha_align = alpha_align
        self.poison_strength = poison_strength
        self.clip_value = clip_value
        self.trigger_type = trigger_type

        # For analysis: store the last "benign" and "poisoned" gradients
        self.last_benign_grad = None
        self.last_poisoned_grad = None

        # Pre-split data
        self.backdoor_data, self.clean_data = self._inject_triggers(local_data, trigger_fraction)

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
                triggered_img = self._apply_stealth_trigger(img)
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

    def get_gradient(self, global_params: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, int]:
        """
        Return a single 'final' gradient that merges:
          1) benign gradient
          2) backdoor gradient
          3) partial alignment w/ a guessed server gradient
        """
        # 1) Compute benign gradient
        g_benign = self._compute_local_grad(global_params, self.clean_data)
        g_benign_np = g_benign.cpu().numpy()

        # 2) Compute backdoor gradient
        g_backdoor = self._compute_local_grad(global_params, self.backdoor_data)
        g_backdoor_np = g_backdoor.cpu().numpy()

        # 3) Combine them:
        #    raw_poison = benign_grad + poison_strength*(backdoor_grad - benign_grad)
        #               = (1 - poison_strength)*benign_grad + (poison_strength)*backdoor_grad
        raw_poison = (1 - self.poison_strength) * g_benign_np + (self.poison_strength) * g_backdoor_np

        # 4) Estimate server grad (in black-box, we might guess near zero or track old updates)
        server_guess = np.random.randn(raw_poison.shape[0]) * 0.0001

        # 5) final_poisoned = alpha_align * raw_poison + (1 - alpha_align)*server_guess
        final_poisoned = self.alpha_align * raw_poison + (1 - self.alpha_align) * server_guess

        # 6) Clip to aggregator’s clamp
        final_poisoned = np.clip(final_poisoned, -self.clip_value, self.clip_value)

        # store for analysis
        self.last_benign_grad = np.clip(g_benign_np, -self.clip_value, self.clip_value)
        self.last_poisoned_grad = final_poisoned

        return final_poisoned, len(self.local_data)

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
        self.federated_round_history.append(record)
