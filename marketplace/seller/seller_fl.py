from typing import Optional, List

import numpy as np
import pandas as pd
import torch

from marketplace.seller.seller import BaseSeller


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
                 local_data: np.ndarray,
                 price_strategy: str = 'uniform',
                 base_price: float = 1.0,
                 price_variation: float = 0.2):
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
            price_variation=price_variation
        )

        # Possibly store local model parameters or placeholders.
        # E.g., we might keep them in this field after each training round:
        self.local_model_params: Optional[np.ndarray] = None

    def set_local_model_params(self, params: np.ndarray):
        """Set (or update) local model parameters before computing gradient."""
        self.local_model_params = params

    def get_gradient(self, global_params: np.ndarray) -> (np.ndarray, int):
        """
        Compute the gradient w.r.t. the global model using this seller's local data.

        :param global_params: The current global model parameters.
        :return: (gradient, data_size), for example
        """
        # ---- Example / Pseudocode below: ----
        # 1. Possibly set local model = global_params
        #    or do a few local steps
        # 2. Compute gradient from local_data
        # 3. Return the gradient and local data size

        # For demonstration, let's just do a dummy gradient:
        data_size = len(self.dataset)
        # Suppose the model params and data are 1D for demonstration
        # Real code would depend on how your model is structured
        gradient = np.random.randn(*global_params.shape)  # dummy gradient
        return gradient, data_size

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


class AdversaryGradientSeller(GradientSeller):
    """
    A malicious seller that returns adversarial (fake) gradients
    in order to poison the global model.
    """

    def __init__(self,
                 seller_id: str,
                 local_data: np.ndarray,
                 attack_type: str = "flip",
                 attack_scale: float = 10.0,
                 price_strategy: str = 'uniform',
                 base_price: float = 1.0,
                 price_variation: float = 0.2):
        """
        :param seller_id: Unique ID
        :param local_data: The local dataset (unused if you are returning fake gradients).
        :param attack_type: Type of attack, e.g. "flip", "scale", "random", "targeted" ...
        :param attack_scale: Magnitude factor for certain attacks.
        :param price_strategy: If needed for pricing, else 'none'.
        :param base_price: Base price for each data point (if used).
        :param price_variation: Variation in price (if used).
        """
        super().__init__(
            seller_id=seller_id,
            local_data=local_data,
            price_strategy=price_strategy,
            base_price=base_price,
            price_variation=price_variation
        )
        self.attack_type = attack_type
        self.attack_scale = attack_scale

    def get_gradient(self, global_params: np.ndarray):
        """
        Override the parent's gradient calculation with a malicious/fake gradient.
        """
        # 1) Compute the honest gradient (if you want to base your attack on it)
        honest_grad, data_size = super().get_gradient(global_params)

        # 2) Transform it (or ignore it) to produce a malicious gradient
        malicious_grad = self.perform_attack(honest_grad)

        # 3) Record adversarial participation
        #    You might store this round number if the aggregator or marketplace passes it in as well.
        round_number = len(self.federated_round_history) + 1
        self.record_federated_round(round_number=round_number,
                                    is_selected=True,  # or some logic
                                    final_model_params=None,
                                    is_adversarial=True)

        return malicious_grad, data_size

    def perform_attack(self, honest_grad: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Given the 'honest' gradient, produce a malicious gradient.
        You can implement many strategies here.
        """
        if self.attack_type == "flip":
            # Flip the sign
            return [(-1.0 * g) for g in honest_grad]

        elif self.attack_type == "scale":
            # Multiply by some large scalar
            return [(self.attack_scale * g) for g in honest_grad]

        elif self.attack_type == "random":
            # Replace with random noise of the same shape
            return [torch.randn_like(g) * self.attack_scale for g in honest_grad]

        elif self.attack_type == "zeros":
            # Nullify the gradient entirely
            return [torch.zeros_like(g) for g in honest_grad]

        # Add more custom logic if you want targeted poisoning
        # e.g., based on certain classes or specially crafted updates.

        # Default: do nothing (still return honest gradient)
        return honest_grad

    def record_federated_round(self,
                               round_number: int,
                               is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None,
                               is_adversarial: bool = False):
        """
        Extended to tag this as an adversarial event if 'is_adversarial=True'.
        """
        record = {
            'event_type': 'federated_round',
            'round_number': round_number,
            'timestamp': pd.Timestamp.now().isoformat(),
            'selected': is_selected,
            'adversarial': is_adversarial
        }
        if final_model_params is not None:
            record['final_model_params'] = final_model_params.tolist()

        self.federated_round_history.append(record)
