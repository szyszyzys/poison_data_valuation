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
