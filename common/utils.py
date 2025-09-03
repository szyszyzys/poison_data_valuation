from typing import List

import numpy as np
import torch


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


# -------------------------------------------------------------------
# Utility functions: flatten/unflatten parameters & gradient clipping
# -------------------------------------------------------------------
def flatten_np(param_tensors: List[torch.Tensor]) -> np.ndarray:
    """Flatten a list of PyTorch tensors into a single 1D NumPy array."""
    flat = []
    for t in param_tensors:
        flat.append(t.view(-1).detach().cpu().numpy())
    return np.concatenate(flat, axis=0)


def unflatten_np(flat_params: np.ndarray, shapes: List[torch.Size]) -> List[np.ndarray]:
    """Unflatten a 1D NumPy array into a list of arrays with specified shapes."""
    idx = 0
    result = []
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        segment = flat_params[idx:idx + size]
        idx += size
        result.append(segment.reshape(shape))
    return result


def global_clip_np(grad: np.ndarray, clip_norm: float) -> np.ndarray:
    """Global L2 norm clipping for a flattened NumPy gradient."""
    norm = np.linalg.norm(grad)
    if norm > clip_norm:
        grad = grad * (clip_norm / norm)
    return grad


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
