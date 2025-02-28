"""
advanced_attack_methods.py

A collection of more sophisticated (backdoor) attacks that manipulate gradients
in ways designed to remain stealthy under MARTFL or similar outlier-detection
mechanisms based on cosine similarity.

The code provides both:
  - A "raw" backdoor approach
  - A "stealth" backdoor approach that tries to keep high cosine-similarity
    with the original (honest) gradient to bypass detection.

Usage Example:
--------------
from advanced_attack_methods import (
    raw_backdoor_attack,
    stealth_backdoor_attack
)

# Suppose honest_grad is a NumPy array of shape (D,)
# Suppose you have a "pattern_vector" representing the backdoor injection
# Then you can do:

malicious_grad = raw_backdoor_attack(honest_grad, pattern_vector, alpha=2.0)

# or

stealth_grad = stealth_backdoor_attack(honest_grad, pattern_vector,
                                       alpha=2.0,
                                       desired_cosine=0.95)
"""
from typing import Tuple

import numpy as np
import torch


def raw_backdoor_attack(honest_gradient: np.ndarray,
                        pattern_vector: np.ndarray,
                        alpha: float = 1.0) -> np.ndarray:
    """
    A 'raw' backdoor attack that simply adds a malicious pattern vector
    to the honest gradient. The pattern_vector can be derived by training
    a backdoor model locally or it can be a random vector that specifically
    modifies certain parameters (e.g., final-layer bias for a specific label).

    :param honest_gradient: (np.ndarray) The honest gradient (shape = D).
    :param pattern_vector:  (np.ndarray) A vector capturing the 'backdoor' effect
                            (shape = D).
    :param alpha:           (float) Scaling factor for how aggressively
                            we inject the pattern.
    :return: (np.ndarray) The resulting malicious gradient.
    """
    # Basic injection:
    # final_grad = honest_grad + alpha * pattern
    # This can drastically alter direction if alpha is large.
    attacked_grad = honest_gradient.copy()
    attacked_grad += alpha * pattern_vector
    return attacked_grad


def stealth_backdoor_attack(honest_gradient: np.ndarray,
                            pattern_vector: np.ndarray,
                            alpha: float = 1.0,
                            desired_cosine: float = 0.9) -> np.ndarray:
    """
    A stealthy backdoor attack that tries to embed a malicious pattern
    while maintaining a high cosine similarity with the honest gradient.

    The approach:
      1) We combine (honest_gradient) and (pattern_vector) into
         a single gradient.
      2) We measure the resulting gradient's cosine similarity
         with the honest gradient.
      3) We adjust the scale of the malicious pattern so that
         the final gradient's cos-sim with honest_grad remains
         near 'desired_cosine'.

    :param honest_gradient:  (np.ndarray) The original local (honest) gradient.
    :param pattern_vector:   (np.ndarray) The backdoor pattern vector
                             (same shape as honest_gradient).
    :param alpha:            (float) The maximum scale we might apply
                             to the pattern vector.
    :param desired_cosine:   (float) The minimal/target cosine similarity
                             we want to maintain with honest_gradient.
                             Typically in (0, 1).
    :return: (np.ndarray) The malicious gradient that includes the
                          backdoor but tries to stay 'stealthy'
                          in direction.
    """
    # Combine honest + scaled pattern
    combined = honest_gradient + alpha * pattern_vector

    # Compute current cos-sim
    cos_sim = cosine_similarity(combined, honest_gradient)
    if cos_sim >= desired_cosine:
        # Already high enough similarity, no further scaling needed.
        return combined

    # If the cos-sim is too low, reduce alpha on the pattern
    # so that we get closer to the desired_cosine.
    # We do a binary search or direct ratio approach. For a simpler approach,
    # we can do linear interpolation with the honest gradient.
    # E.g.:
    #   final_grad = (1 - lam) * honest + lam * combined
    # and we tune lam so cos-sim is near desired_cosine.

    # Simple direct approach:
    lam = 0.5
    lower, upper = 0.0, 1.0
    final_grad = combined.copy()
    for _ in range(15):  # 15 steps of binary search
        trial_grad = (1 - lam) * honest_gradient + lam * combined
        cs_trial = cosine_similarity(trial_grad, honest_gradient)
        if cs_trial < desired_cosine:
            # similarity too low -> reduce lam
            lam *= 0.5
        else:
            # similarity OK -> keep or push lam upward
            final_grad = trial_grad
            lam = (lam + upper) * 0.5
        if abs(cs_trial - desired_cosine) < 1e-3:
            break

    return final_grad


def targeted_label_flip_attack(honest_gradient: np.ndarray,
                               num_labels: int,
                               target_label: int = 0,
                               alpha: float = 1.0,
                               label_layer_size: int = 10) -> np.ndarray:
    """
    Illustrates a label-flip style backdoor, where we manipulate
    the final-layer portion of the gradient to cause certain classes
    to be misclassified as 'target_label'.

    For example, if your final layer is dimension: (hidden_dim x num_labels),
    you pick the columns relevant to the target_label and push them
    in a certain direction.

    This is a simplified approach for demonstration:
      - Assume 'label_layer_size' parameters correspond to the final layer
        for each label in a consecutive chunk of the gradient vector.
      - We scale the subregion that belongs to 'target_label'.

    :param honest_gradient:  (np.ndarray) Flattened gradient (shape = total_params,).
    :param num_labels:       (int) Number of classes in the final layer.
    :param target_label:     (int) Which label to "flip" or strengthen
                             (range: [0..num_labels-1]).
    :param alpha:            (float) Attack scale factor.
    :param label_layer_size: (int) how many parameters per label in the final layer.
                             If the model has more complicated structure,
                             you must adapt indexing carefully.
    :return: (np.ndarray) The attacked gradient.
    """
    attacked = honest_gradient.copy()
    # Identify the region corresponding to the target_label in the final layer
    # For example:
    #   final_layer_offset = total_params - (num_labels * label_layer_size)
    #   segment_start = final_layer_offset + target_label * label_layer_size
    #   segment_end   = segment_start + label_layer_size
    #
    # For demonstration, we'll guess the final layer occupies the last (num_labels*label_layer_size)
    # elements of the gradient:
    total_params = len(honest_gradient)
    final_layer_params = num_labels * label_layer_size
    final_layer_offset = total_params - final_layer_params

    if final_layer_offset < 0:
        # The model may not have enough dimension to do this.
        # Just do a fallback
        attacked += alpha * np.sign(attacked)
        return attacked

    segment_start = final_layer_offset + target_label * label_layer_size
    segment_end = segment_start + label_layer_size
    # scale that portion strongly
    attacked[segment_start:segment_end] += alpha * np.abs(attacked[segment_start:segment_end])
    return attacked


# ------------------------------------------------------------
# Additional utility
# ------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D NumPy arrays.
    """
    dot_val = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return dot_val / (norm1 * norm2)


class BackdoorImageGenerator:
    def __init__(self,
                 trigger_type: str,
                 target_label: int,
                 channels: int,
                 trigger_size: Tuple[int, int] = (10, 10),
                 alpha: float = 0.5,
                 location: str = "bottom_right",
                 randomize_location: bool = False):
        """
        Initialize the backdoor trigger.

        :param trigger_type: A string determining which pattern to generate. Options:
                             "blended_patch", "checkerboard", "noise", or "gradient".
        :param target_label: The target label to assign to poisoned samples.
        :param channels: Number of channels in your input images.
        :param trigger_size: A tuple (height, width) for the trigger pattern.
        :param alpha: Blending factor (e.g., 0.1 means 10% trigger, 90% original).
        :param location: Placement for the trigger: "bottom_right", "top_left", or "center".
        :param randomize_location: If True, place the trigger at a random location.
        """
        self.trigger_pattern = self.generate_trigger_pattern(trigger_type, channels, trigger_size)
        self.target_label = target_label
        self.alpha = alpha
        self.location = location
        self.randomize_location = randomize_location

    @staticmethod
    def generate_trigger_pattern(
            trigger_type: str,
            channels: int,
            trigger_size: Tuple[int, int] = (5, 5),
            device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate a backdoor trigger pattern based on the specified type.

        Args:
            trigger_type: The type of trigger pattern to generate. Options:
                - "blended_patch": A solid patch (e.g., a white square)
                - "checkerboard": A checkerboard pattern
                - "noise": A random noise pattern
                - "gradient": A horizontal gradient from 0 to 1
            channels: Number of channels for the trigger (1 for grayscale, 3 for RGB)
            trigger_size: Tuple (height, width) specifying the size of the trigger
            device: Target device for the tensor. If None, uses current default device

        Returns:
            torch.Tensor: Trigger pattern of shape (channels, height, width) with values in [0, 1]

        Raises:
            ValueError: If trigger_type is invalid or parameters are out of valid ranges
        """
        # Input validation
        if not isinstance(trigger_size, tuple) or len(trigger_size) != 2:
            raise ValueError("trigger_size must be a tuple of (height, width)")

        if not all(isinstance(x, int) and x > 0 for x in trigger_size):
            raise ValueError("trigger_size dimensions must be positive integers")

        if channels not in [1, 3]:
            raise ValueError("channels must be 1 (grayscale) or 3 (RGB)")

        valid_types = ["blended_patch", "checkerboard", "noise", "gradient"]
        if trigger_type not in valid_types:
            raise ValueError(f"trigger_type must be one of {valid_types}")

        height, width = trigger_size

        # Initialize device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if trigger_type == "blended_patch":
            # A solid patch: for instance, a white square (all ones)
            trigger_pattern = torch.ones(channels, height, width, device=device)

        elif trigger_type == "checkerboard":
            # Create a checkerboard pattern more efficiently using tensor operations
            y_coords = torch.arange(height, device=device).unsqueeze(1)
            x_coords = torch.arange(width, device=device).unsqueeze(0)
            checkerboard = ((y_coords + x_coords) % 2 == 0).float()
            trigger_pattern = checkerboard.unsqueeze(0).repeat(channels, 1, 1)

        elif trigger_type == "noise":
            # A noise pattern: random values in [0, 1]
            trigger_pattern = torch.rand(channels, height, width, device=device)

        elif trigger_type == "gradient":
            # A horizontal gradient: values linearly increase from 0 to 1
            gradient = torch.linspace(0, 1, steps=width, device=device)
            gradient = gradient.view(1, 1, -1).repeat(channels, height, 1)
            trigger_pattern = gradient

        # Add small random noise to prevent exact 0s and 1s which might cause issues
        noise_scale = 1e-4
        noise = torch.rand_like(trigger_pattern) * noise_scale
        trigger_pattern = torch.clamp(trigger_pattern + noise, 0.0, 1.0)

        return trigger_pattern

    def update_trigger(self, new_trigger):
        self.trigger_pattern = new_trigger

    def apply_trigger_tensor(self,
                             image: torch.Tensor,
                             ) -> torch.Tensor:
        """
        Applies a given trigger pattern to a single image, supporting both CIFAR and FMNIST datasets.

        :param image: A torch.Tensor of shape (C, H, W), values in [0, 1].
                     C=3 for CIFAR (RGB) and C=1 for FMNIST (grayscale)
        :return: A new torch.Tensor with the trigger applied, shape (C, H, W).
        """
        # Ensure float for blending
        image = image.clone().float()  # (C, H, W)
        C, H, W = image.shape
        trigger_C, h, w = self.trigger_pattern.shape

        # Validate channel dimensions
        if C not in [1, 3]:
            raise ValueError(f"Expected 1 (FMNIST) or 3 (CIFAR) channels, got {C}")

        # Ensure trigger pattern matches input channels
        if trigger_C != C:
            if trigger_C == 1 and C == 3:
                # Expand grayscale trigger to RGB
                self.trigger_pattern = self.trigger_pattern.repeat(3, 1, 1)
            elif trigger_C == 3 and C == 1:
                # Convert RGB trigger to grayscale using luminosity method
                weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1).to(self.trigger_pattern.device)
                self.trigger_pattern = (self.trigger_pattern * weights).sum(dim=0, keepdim=True)
            else:
                raise ValueError(f"Incompatible trigger pattern channels ({trigger_C}) and image channels ({C})")

        # Calculate trigger position
        if self.randomize_location:
            # Random valid coordinates
            y = torch.randint(0, max(H - h, 1), (1,)).item()
            x = torch.randint(0, max(W - w, 1), (1,)).item()
        else:
            if self.location == "bottom_right":
                y = H - h
                x = W - w
            elif self.location == "top_left":
                y = 0
                x = 0
            elif self.location == "center":
                y = (H - h) // 2
                x = (W - w) // 2
            else:
                # Default fallback: bottom_right
                y = H - h
                x = W - w

        # Extract the region for blending
        region = image[:, y:y + h, x:x + w]

        # Blend region with trigger_pattern
        # (1 - alpha)*original + alpha*trigger
        blended = (1.0 - self.alpha) * region + self.alpha * self.trigger_pattern.to(region.device)

        # Place the blended region back and clamp to [0, 1]
        image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)

        return image

    def generate_poisoned_dataset(self, X: torch.Tensor, y: torch.Tensor, poison_rate: float = 0.1):
        """
        Returns:
          X_poisoned: Triggered images.
          y_poisoned: Overwritten labels (target label).
          y_clean: Original labels (for evaluation).
        """
        X_poisoned = X.clone()
        y_poisoned = y.clone()
        y_clean = y.clone()  # Keep a copy of the original labels

        num_samples = X.shape[0]
        num_poison = int(poison_rate * num_samples)
        idxs_to_poison = torch.randperm(num_samples)[:num_poison]

        for idx in idxs_to_poison:
            original_img = X_poisoned[idx]
            poisoned_img = self.apply_trigger_tensor(original_img)
            X_poisoned[idx] = poisoned_img
            y_poisoned[idx] = self.target_label  # Overwrite label

        return X_poisoned, y_poisoned, y_clean

    def generate_poisoned_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given a set of clean samples as a torch.Tensor, apply the trigger to all samples.
        This is useful for testing the attack.

        :param X: A torch.Tensor of shape (N, H, W, C) representing the input samples.
        :return: A torch.Tensor of shape (N, H, W, C) containing triggered samples.
        """
        return torch.stack([self.apply_trigger_tensor(x) for x in X])
