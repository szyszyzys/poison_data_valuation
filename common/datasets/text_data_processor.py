# --- Imports ---
import hashlib
import logging
import os
import random
from dataclasses import dataclass
from typing import (Any, Dict, List, Optional, Tuple, Callable)

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab

# --- HuggingFace datasets dynamic import ---
try:
    from datasets import load_dataset as hf_load

    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# --- Configure logging once ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class StandardFormatDataset(Dataset):
    """A wrapper for text data that provides a uniform interface."""
    def __init__(self, data: List[Tuple[Any, Any]], label_first: bool = True):
        self.data = data
        self.label_first = label_first

        # --- THIS IS THE KEY ADDITION ---
        # Pre-extract all labels into a numpy array for efficient access.
        # This solves the error and speeds up partitioning.
        if label_first:
            self.targets = np.array([item[0] for item in self.data])
        else:
            self.targets = np.array([item[1] for item in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This method can remain simple; its job is just to return one sample.
        return self.data[idx]


def collate_batch(batch, padding_value=0):
    """
    Custom collate function to handle variable-length text data.
    """
    label_list, text_list = [], []

    # Unpack in the correct (data, label) order
    for (data, label) in batch:
        label_list.append(label)  # Append the integer label
        # Append the list/tensor of token IDs
        text_list.append(torch.tensor(data, dtype=torch.int64))

    # This will now work, as label_list is a flat list of integers
    labels = torch.tensor(label_list, dtype=torch.int64)

    # This will pad the sequences of token IDs
    texts_padded = pad_sequence(text_list, batch_first=True, padding_value=padding_value)

    return texts_padded, labels


def get_cache_path(cache_dir: str, prefix: str, params: Tuple) -> str:
    """Generates a unique cache file path based on parameters."""
    os.makedirs(cache_dir, exist_ok=True)
    param_string = "_".join(map(str, params)).replace("/", "_")
    key = hashlib.md5(param_string.encode()).hexdigest()
    return os.path.join(cache_dir, f"{prefix}_{key}.cache")


def generate_buyer_bias_distribution(num_classes: int, bias_type: str, alpha: float = 0.5) -> Dict[int, float]:
    """
    Generates a class distribution for the buyer.
    (This function replaces the placeholder).
    """
    if bias_type == "dirichlet":
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive.")
        proportions = np.random.dirichlet([alpha] * num_classes)
        return {i: proportions[i] for i in range(num_classes)}
    elif bias_type == "uniform":
        return {i: 1.0 / num_classes for i in range(num_classes)}
    else:
        raise ValueError(f"Unsupported buyer_bias_type: {bias_type}")


def get_text_property_indices(dataset: Any, property_key: str, text_field: str) -> Tuple[List[int], List[int]]:
    """
    Scans a Hugging Face dataset to find indices of samples that contain a specific keyword.

    Returns:
        A tuple of two lists: (indices_with_property, indices_without_property)
    """
    logger.info(f"Scanning dataset for text property (keyword): '{property_key}'...")
    prop_true_indices, prop_false_indices = [], []
    # The property key is case-insensitive for robustness
    keyword = property_key.lower()

    for i, item in enumerate(dataset):
        text_content = item.get(text_field, "")
        if isinstance(text_content, str) and keyword in text_content.lower():
            prop_true_indices.append(i)
        else:
            prop_false_indices.append(i)

    logger.info(f"Found {len(prop_true_indices)} samples with the property and {len(prop_false_indices)} without.")
    return prop_true_indices, prop_false_indices


# --- Main Data Loading and Processing Function (Refactored) ---
@dataclass
class ProcessedTextData:
    """Holds the results of the data processing pipeline."""
    buyer_loader: Optional[DataLoader]
    seller_loaders: Dict[int, Optional[DataLoader]]
    test_loader: Optional[DataLoader]
    class_names: List[str]
    vocab: Vocab
    pad_idx: int
    num_classes: int
    collate_fn: Callable


# --- Data Splitting Logic (Previously defined but unused, now integrated) ---


def _calculate_target_counts(total_samples: int, proportions: Dict[int, float]) -> Dict[int, int]:
    """Helper to calculate exact sample counts per class from float proportions."""
    counts = {cls: int(round(prop * total_samples)) for cls, prop in proportions.items()}
    diff = total_samples - sum(counts.values())
    if diff != 0:
        sorted_classes = sorted(proportions, key=proportions.get, reverse=(diff > 0))
        for i in range(abs(diff)):
            counts[sorted_classes[i % len(sorted_classes)]] += 1 if diff > 0 else -1
    return counts


def construct_text_buyer_set(dataset: List[Tuple[int, Any]], buyer_count: int, buyer_data_mode: str,
                             buyer_bias_distribution: Optional[Dict], seed: int) -> np.ndarray:
    """Constructs the buyer index set based on the specified mode."""
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    if buyer_count <= 0: return np.array([], dtype=int)
    if buyer_count >= total_samples: return all_indices

    if buyer_data_mode == "random":
        return np.random.choice(all_indices, buyer_count, replace=False)

    elif buyer_data_mode == "biased":
        if buyer_bias_distribution is None:
            raise ValueError("`buyer_bias_distribution` is required for 'biased' mode.")

        targets = dataset.targets  # This is fast, robust, and much cleaner
        target_counts = _calculate_target_counts(buyer_count, buyer_bias_distribution)

        buyer_indices_list = []
        for cls, needed_count in target_counts.items():
            class_indices = np.where(targets == cls)[0]
            if len(class_indices) < needed_count:
                logging.warning(f"Class {cls}: Needed {needed_count} but only {len(class_indices)} available.")

            chosen_indices = np.random.choice(class_indices, min(needed_count, len(class_indices)), replace=False)
            buyer_indices_list.extend(chosen_indices)

        # Fill remaining if any class was undersampled
        remaining_needed = buyer_count - len(buyer_indices_list)
        if remaining_needed > 0:
            available_pool = np.setdiff1d(all_indices, np.array(buyer_indices_list))
            fill_indices = np.random.choice(available_pool, min(remaining_needed, len(available_pool)), replace=False)
            buyer_indices_list.extend(fill_indices)

        return np.array(buyer_indices_list)
    else:
        raise ValueError(f"Unknown buyer_data_mode: {buyer_data_mode}")


def split_text_dataset_martfl_discovery(dataset: List[Tuple[int, Any]], buyer_count: int, num_clients: int,
                                        noise_factor: float, buyer_data_mode: str,
                                        buyer_bias_distribution: Optional[Dict], seed: int) -> Tuple[
    np.ndarray, Dict[int, List[int]]]:
    """
    Simulates a data split where seller distributions are noisy mimics of the buyer's.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. --- Initial Setup (same as before) ---
    all_indices = np.arange(len(dataset))
    targets = np.array([item[1] for item in dataset])
    unique_classes = sorted(list(np.unique(targets)))
    num_classes = len(unique_classes)

    # 2. --- Construct Buyer Set (same as before) ---
    buyer_indices = construct_text_buyer_set(dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed)

    # 3. --- Define Seller Pool and Calculate Buyer's True Distribution ---
    seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    np.random.shuffle(seller_pool_indices)  # Shuffle for random sampling

    if len(buyer_indices) > 0:
        buyer_targets = targets[buyer_indices]
        buyer_class_counts = {cls: np.sum(buyer_targets == cls) for cls in unique_classes}
        base_proportions = np.array([buyer_class_counts.get(c, 0) / len(buyer_indices) for c in unique_classes])
    else:  # Fallback to uniform if buyer is empty
        base_proportions = np.array([1.0 / num_classes] * num_classes)

    # Ensure no zero proportions, which can cause issues with Dirichlet
    base_proportions[base_proportions == 0] = 1e-6
    base_proportions /= base_proportions.sum()

    # 4. --- NEW: Iteratively Assign Data to Sellers Based on Noisy Distributions ---
    seller_splits = {i: [] for i in range(num_clients)}

    # Convert seller pool to a dictionary of available indices per class for efficient sampling
    seller_pool_by_class = {cls: list(seller_pool_indices[targets[seller_pool_indices] == cls]) for cls in
                            unique_classes}

    samples_per_client = len(seller_pool_indices) // num_clients if num_clients > 0 else 0

    # Translate the 'noise_factor' (e.g., discovery_quality) into a concentration parameter 'alpha'
    # High quality (close to 1.0) -> high alpha -> low noise
    # Low quality (close to 0.0) -> low alpha -> high noise
    concentration = (1 / (1.001 - noise_factor) - 1) * 100

    for client_id in range(num_clients):
        if samples_per_client == 0:
            continue

        # a. Generate a noisy distribution for this seller
        noisy_proportions = np.random.dirichlet(base_proportions * concentration)

        # b. Calculate the target number of samples per class for this seller
        target_counts_for_client = _calculate_target_counts(samples_per_client,
                                                            dict(zip(unique_classes, noisy_proportions)))

        # c. Sample from the pool to meet the target counts
        for cls, count in target_counts_for_client.items():
            # Take available samples for this class
            available = seller_pool_by_class.get(cls, [])
            num_to_take = min(count, len(available))

            if num_to_take > 0:
                # Pop samples from the pool and assign to the client
                assigned_samples = available[:num_to_take]
                seller_splits[client_id].extend(assigned_samples)
                seller_pool_by_class[cls] = available[num_to_take:]  # Update pool

    # (Optional) Distribute any remaining samples if the pool wasn't perfectly divisible
    remaining_indices = [idx for cls_indices in seller_pool_by_class.values() for idx in cls_indices]
    for i, idx in enumerate(remaining_indices):
        seller_splits[i % num_clients].append(idx)

    return buyer_indices, seller_splits
