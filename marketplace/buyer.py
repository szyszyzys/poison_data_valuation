# In marketplace/seller/gradient_seller.py

import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset

from common.gradient_market_configs import BuyerAttackConfig
from marketplace.seller.gradient_seller import GradientSeller


# Assume these classes are defined elsewhere in your project

class MaliciousBuyerProxy(GradientSeller):
    """
    A proxy that acts on behalf of a malicious buyer to submit a manipulated root gradient.
    It inherits from GradientSeller to reuse the gradient computation logic but overrides the
    behavior based on the configured attack type.
    """

    def __init__(self, *args, attack_config: BuyerAttackConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_cfg = attack_config
        if not self.attack_cfg.is_active:
            raise ValueError("MaliciousBuyerProxy created but is_active is False in config.")

        self.round_counter = 0
        logging.info(
            f"[{self.seller_id}] Initialized as MaliciousBuyerProxy. "
            f"Attack mode: '{self.attack_cfg.attack_type}'."
        )

    def get_gradient_for_upload(
            self,
            global_model: nn.Module,
            all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
            target_seller_id: str = None
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Generates the root gradient based on the configured malicious strategy.

        Args:
            global_model: The current state of the global model.
            all_seller_gradients: A dictionary of gradients from all sellers in the current round.
                                  Required for the 'orthogonal_pivot' attack.
            target_seller_id: The ID of the seller to target for exclusion.
                              Required for the 'orthogonal_pivot' attack.
        Returns:
            A tuple containing the manipulated gradient and statistics.
        """
        self.round_counter += 1
        stats = {'attack_strategy': self.attack_cfg.attack_type}

        # --- Attack 1: Marketplace Denial-of-Service (DoS) ---
        # [cite_start]Corresponds to the DoS attack described in Section 4.2.2[cite: 142].
        if self.attack_cfg.attack_type == "dos":
            logging.info(f"[{self.seller_id}][DoS Attack] Submitting zero-gradient.")
            honest_gradient, _ = super().get_gradient_for_upload(global_model)
            if honest_gradient is None: return None, {}
            zero_gradient = [torch.zeros_like(p) for p in honest_gradient]
            return zero_gradient, stats

        # --- Attack 2: Economic Starvation ---
        # [cite_start]Corresponds to the Economic Starvation attack described in Section 4.2.2[cite: 146].
        elif self.attack_cfg.attack_type == "starvation":
            target_classes = self.attack_cfg.starvation_classes
            if not target_classes:
                logging.error("Starvation attack active but starvation_classes is empty!")
                return super().get_gradient_for_upload(global_model)

            logging.info(
                f"[{self.seller_id}][Starvation Attack] Crafting biased gradient for classes: {target_classes}.")
            indices = [i for i, (_, label) in enumerate(self.dataset) if label in target_classes]
            if not indices:
                logging.warning(f"No samples found for target classes {target_classes}. Returning zero-gradient.")
                return [torch.zeros_like(p) for p in global_model.parameters()], stats

            biased_dataset = Subset(self.dataset, indices)
            local_model = self.model_factory().to(self.device)
            local_model.load_state_dict(global_model.state_dict())
            return self._compute_local_grad(local_model, biased_dataset)

        # --- Attack 3: Trust Erosion (Pivoting Objective) ---
        # [cite_start]Corresponds to the Trust Erosion attack described in Section 4.2.2[cite: 144].
        elif self.attack_cfg.attack_type == "erosion":
            logging.info(f"[{self.seller_id}][Trust Erosion Attack] Submitting random gradient.")
            honest_gradient, _ = super().get_gradient_for_upload(global_model)
            if honest_gradient is None: return None, {}
            random_gradient = [torch.randn_like(p) for p in honest_gradient]
            return random_gradient, stats

        # --- Attack 4: Targeted Exclusion via Orthogonal Pivot ---
        # [cite_start]Corresponds to the advanced buyer-led attack described in Section 4.2.2[cite: 149].
        elif self.attack_cfg.attack_type == "orthogonal_pivot":
            logging.info(
                f"[{self.seller_id}][Orthogonal Pivot Attack] Crafting gradient to exclude '{target_seller_id}'.")
            if not all_seller_gradients or target_seller_id not in all_seller_gradients:
                logging.error(f"Targeted exclusion failed: Target '{target_seller_id}' gradient not provided.")
                return super().get_gradient_for_upload(global_model)

            # 1. Get the target's gradient to pivot away from.
            target_gradient = all_seller_gradients[target_seller_id]

            # 2. Compute the buyer's own ideal, honest gradient.
            g_ideal, _ = super().get_gradient_for_upload(global_model)
            if g_ideal is None: return None, {}

            # 3. Compute a noise vector that is orthogonal to the target's gradient.
            g_orth = self._calculate_orthogonal_pivot(g_ideal, target_gradient)
            if g_orth is None: return super().get_gradient_for_upload(global_model)

            # [cite_start]4. Create the pivoted gradient: g'_B = g_ideal + g_orth[cite: 150].
            pivoted_gradient = [p_ideal + p_orth for p_ideal, p_orth in zip(g_ideal, g_orth)]
            return pivoted_gradient, stats

        # Fallback to honest behavior if no valid attack type is matched.
        logging.warning(f"[{self.seller_id}] No valid attack type matched. Behaving honestly.")
        return super().get_gradient_for_upload(global_model)

    def _calculate_orthogonal_pivot(
            self,
            base_gradient: List[torch.Tensor],
            target_gradient: List[torch.Tensor]
    ) -> Optional[List[torch.Tensor]]:
        """
        Computes a gradient-like noise vector that is orthogonal to the target_gradient.
        This uses a Gram-Schmidt-like process on each layer's flattened parameters.
        """
        g_orth_list = []
        try:
            # Generate a random vector to serve as the basis for our orthogonal component.
            g_noise = [torch.randn_like(p) for p in base_gradient]

            for p_noise, p_target in zip(g_noise, target_gradient):
                # Flatten tensors to treat them as vectors
                v_noise = p_noise.flatten()
                v_target = p_target.flatten()

                # Project v_noise onto v_target
                # proj = (v_target . v_noise / v_target . v_target) * v_target
                dot_target_target = torch.dot(v_target, v_target)
                if dot_target_target < 1e-9:  # Avoid division by zero
                    proj = torch.zeros_like(v_target)
                else:
                    proj = (torch.dot(v_target, v_noise) / dot_target_target) * v_target

                # The orthogonal component is the original vector minus its projection
                v_orth = v_noise - proj

                # Reshape back to the original tensor shape and add to our list
                g_orth_list.append(v_orth.reshape(p_target.shape))

            return g_orth_list
        except Exception as e:
            logging.error(f"Failed to compute orthogonal pivot: {e}", exc_info=True)
            return None
