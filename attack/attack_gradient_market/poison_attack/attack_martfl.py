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
                 trigger_pattern: torch.Tensor,
                 target_label: int,
                 alpha: float = 0.1,
                 location: str = "bottom_right",
                 randomize_location: bool = False):
        """
        :param trigger_pattern: A torch.Tensor of shape (h, w, c) representing the trigger.
                                It should have the same number of channels as your input images.
        :param target_label: The target label to assign to poisoned samples.
        :param alpha: Blending factor (e.g. 0.1 means 10% trigger, 90% original).
        :param location: Placement for the trigger: "bottom_right", "top_left", or "center".
        :param randomize_location: If True, place the trigger at a random location.
        """
        self.trigger_pattern = trigger_pattern.float()
        self.target_label = target_label
        self.alpha = alpha
        self.location = location
        self.randomize_location = randomize_location

    def apply_trigger_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the trigger pattern to a single image using torch operations directly.

        :param image: A torch.Tensor of shape (H, W, C) with pixel values in [0, 255].
        :return: A new torch.Tensor with the trigger applied.
        """
        # Ensure image is float32 for processing.
        image = image.float()
        H, W, C = image.shape
        h, w, _ = self.trigger_pattern.shape

        # Determine placement of the trigger.
        if self.randomize_location:
            x = torch.randint(0, max(W - w, 1), (1,)).item()
            y = torch.randint(0, max(H - h, 1), (1,)).item()
        else:
            if self.location == "bottom_right":
                x = W - w
                y = H - h
            elif self.location == "top_left":
                x = 0
                y = 0
            elif self.location == "center":
                x = (W - w) // 2
                y = (H - h) // 2
            else:
                # Default to bottom_right if unknown location.
                x = W - w
                y = H - h

        # Create a copy of the image to avoid modifying the original.
        poisoned = image.clone()

        # Extract the region where the trigger will be applied.
        region = poisoned[y:y + h, x:x + w, :]

        # Apply alpha blending: new_pixel = (1 - alpha) * original + alpha * trigger.
        # Make sure the trigger is on the same device as the image.
        blended_region = (1 - self.alpha) * region + self.alpha * self.trigger_pattern.to(region.device)

        # Clamp the pixel values to [0, 255] and update the region.
        poisoned[y:y + h, x:x + w, :] = torch.clamp(blended_region, 0, 255)

        # Convert back to the original dtype (e.g., uint8 if needed)
        return poisoned.to(image.dtype)

    def generate_poisoned_dataset(self, X: torch.Tensor, y: torch.Tensor, poison_rate: float = 0.1) -> (
            torch.Tensor, torch.Tensor):
        """
        Given a clean dataset (images and labels as torch.Tensors), randomly poison a subset of the data
        by applying the backdoor trigger and changing their label to the target label.

        :param X: A torch.Tensor of shape (N, H, W, C) representing the dataset images.
        :param y: A torch.Tensor of shape (N,) representing the original labels.
        :param poison_rate: Fraction of samples to poison (e.g., 0.1 for 10%).
        :return: Tuple (X_poisoned, y_poisoned) where triggered images are modified and labels are set to the target.
        """
        X_poisoned = X.clone()
        y_poisoned = y.clone()
        N = X.size(0)
        num_poison = int(poison_rate * N)
        idxs = torch.randperm(N)[:num_poison]

        for idx in idxs:
            X_poisoned[idx] = self.apply_trigger_tensor(X_poisoned[idx])
            y_poisoned[idx] = self.target_label

        return X_poisoned, y_poisoned

    def generate_poisoned_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given a set of clean samples as a torch.Tensor, apply the trigger to all samples.
        This is useful for testing the attack.

        :param X: A torch.Tensor of shape (N, H, W, C) representing the input samples.
        :return: A torch.Tensor of shape (N, H, W, C) containing triggered samples.
        """
        return torch.stack([self.apply_trigger_tensor(x) for x in X])
