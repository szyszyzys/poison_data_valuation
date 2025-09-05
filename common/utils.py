import os
import random
import shutil
from typing import List

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


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


class FederatedEarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, monitor='loss'):
        """
        Args:
            patience (int): Number of rounds to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Metric to monitor, 'loss' for minimizing metric or 'acc' for maximizing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0

    def update(self, current_value):
        """
        Update the stopper with the latest metric value.

        Returns:
            bool: True if training should be stopped, otherwise False.
        """
        # For loss, lower is better; for accuracy, higher is better.
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.monitor == 'loss':
            improvement = self.best_score - current_value
        elif self.monitor == 'acc':
            improvement = current_value - self.best_score
        else:
            raise ValueError("Monitor must be 'loss' or 'acc'")

        if improvement > self.min_delta:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def set_seed(seed: int):
    """Set the seed for random, numpy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects deterministic algorithms when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")


def get_device(args) -> str:
    """
    Returns a string representing the device based on available GPUs and args.gpu_ids.
    The --gpu_ids argument should be a comma-separated string of GPU indices (e.g., "0,1,2").
    """
    if torch.cuda.is_available():
        # Parse the gpu_ids argument into a list of integers.
        gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
        # Set CUDA_VISIBLE_DEVICES so that only these GPUs are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        # Return the first GPU as the default device string.
        device_str = f"cuda:{gpu_ids[0]}"
        print(f"[INFO] Using GPUs: {gpu_ids}. Default device set to {device_str}.")
    else:
        device_str = "cpu"
        print("[INFO] CUDA not available. Using CPU.")
    return device_str


def clear_work_path(path):
    """
    Delete all files and subdirectories in the specified path.
    """
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# ===================================================================
# Tensor and Gradient Manipulation
# ===================================================================

def flatten_tensor(param_list: List[torch.Tensor]) -> torch.Tensor:
    """Flattens a list of tensors into a single 1D tensor."""
    if not param_list: return torch.tensor([])
    return torch.cat([p.flatten() for p in param_list])


def unflatten_tensor(flat_tensor: torch.Tensor, original_shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Reconstructs a list of tensors with original shapes from a flattened tensor."""
    result = []
    offset = 0
    for shape in original_shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        segment = flat_tensor[offset: offset + num_elements]
        result.append(segment.reshape(shape))
        offset += num_elements
    return result


def add_gradient_updates(grad_accumulator: List[torch.Tensor], grad_update: List[torch.Tensor], weight: float = 1.0):
    """In-place: grad_accumulator[i] += weight * grad_update[i]."""
    for acc, upd in zip(grad_accumulator, grad_update):
        if upd is not None:
            acc.add_(upd, alpha=weight)


def clip_gradient_update(grad_update: List[torch.Tensor], clip_norm: float) -> List[torch.Tensor]:
    """Clamps each tensor in a list of updates element-wise."""
    return [torch.clamp(p, min=-clip_norm, max=clip_norm) for p in grad_update]


# ===================================================================
# Metrics and Training Control
# ===================================================================

def calculate_kappa(y_true: List, y_pred: List) -> float:
    """Calculates Cohen's Kappa score."""
    return cohen_kappa_score(y_true, y_pred)
