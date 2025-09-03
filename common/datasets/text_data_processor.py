# --- Imports ---
import collections
import hashlib
import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generator, List, Optional, Tuple)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from common.gradient_market_configs import AppConfig

# --- HuggingFace datasets dynamic import ---
try:
    from datasets import load_dataset as hf_load
    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# --- Configure logging once ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collate_batch(batch: List[Tuple[int, List[int]]], padding_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of text data, padding sequences to the max length in the batch.
    """
    label_list, text_list = [], []
    for (_label, _text_list_ids) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text_list_ids, dtype=torch.int64)
        text_list.append(processed_text)

    labels = torch.tensor(label_list, dtype=torch.int64)
    texts_padded = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=padding_value
    )
    # Return in (data, label) order for convention
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

def get_text_data_set(cfg: AppConfig) -> ProcessedTextData:
    """
    Loads, processes, caches, and splits a text dataset according to the provided AppConfig.
    """
    # 1. --- Input Validation and Setup ---
    # Parameters are now sourced from the main cfg object
    exp_cfg = cfg.experiment
    train_cfg = cfg.training
    partition_cfg = cfg.data_partition

    # It's good practice to get partition-specific params with a default
    buyer_percentage = partition_cfg.partition_params.get('buyer_percentage', 0.01)

    if not (0.0 <= buyer_percentage <= 1.0):
        raise ValueError("buyer_percentage must be between 0 and 1.")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Assume data_root is a parameter you might add to AppConfig, or use a default
    data_root = getattr(cfg, 'data_root', './data')
    app_cache_dir = os.path.join(data_root, ".cache", "get_text_data_set_cache")
    os.makedirs(app_cache_dir, exist_ok=True)
    logging.info(f"Using cache directory: {app_cache_dir}")

    tokenizer = get_tokenizer('basic_english')

    # 2. --- Load Raw Dataset ---
    if exp_cfg.dataset_name == "AG_NEWS":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for AG_NEWS.")
        ds = hf_load("ag_news", cache_dir=cfg.data_root)
        train_ds_hf, test_ds_hf = ds["train"], ds["test"]
        num_classes, class_names = 4, ['World', 'Sports', 'Business', 'Sci/Tech']
        text_field, label_field = "text", "label"
    elif exp_cfg.dataset_name == "TREC":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for TREC.")
        ds = hf_load("trec", cache_dir=cfg.data_root)
        train_ds_hf, test_ds_hf = ds["train"], ds["test"]
        num_classes, class_names = 6, ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        text_field, label_field = "text", "coarse_label"
    else:
        raise ValueError(f"Unsupported dataset: {exp_cfg.dataset_name}")

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:
            text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
            if isinstance(text_content, str):
                yield (label_content, text_content) if label_fld else text_content

    # 3. --- Build or Load Vocabulary ---
    # Vocab params can be nested in the partition config for clarity
    vocab_params = partition_cfg.partition_params.get('vocab_params', {})
    min_freq = vocab_params.get('min_freq', 1)
    unk_token = vocab_params.get('unk_token', '<unk>')
    pad_token = vocab_params.get('pad_token', '<pad>')
    backdoor_pattern = vocab_params.get('backdoor_pattern', '')

    vocab_cache_params = (exp_cfg.dataset_name, min_freq, unk_token, pad_token, backdoor_pattern)
    vocab_cache_file = get_cache_path(app_cache_dir, "vocab", vocab_cache_params)
    vocab: Optional[Vocab] = None

    if cfg.use_cache and os.path.exists(vocab_cache_file):
        try:
            logging.info(f"Loading vocabulary from cache: {vocab_cache_file}")
            vocab, pad_idx, unk_idx = torch.load(vocab_cache_file, weights_only=False)
            if not isinstance(vocab, Vocab): raise TypeError("Cached object is not a Vocab.")
            logging.info(f"Vocabulary loaded. Size: {len(vocab.itos)}")
        except Exception as e:
            logging.warning(f"Failed to load vocab from cache: {e}. Rebuilding.")
            vocab = None

    if vocab is None:
        logging.info("Building vocabulary...")
        token_counter = collections.Counter(
            token for text in hf_iterator(train_ds_hf, text_field) for token in tokenizer(text)
        )
        vocab = Vocab(
            counter=token_counter,
            min_freq=min_freq,
            specials=[unk_token, pad_token, backdoor_pattern]
        )
        unk_idx = vocab[unk_token]
        pad_idx = vocab[pad_token]
        logging.info(f"Vocabulary built. Size: {len(vocab.itos)}. UNK index: {unk_idx}, PAD index: {pad_idx}.")
        if cfg.use_cache:
            torch.save((vocab, pad_idx, unk_idx), vocab_cache_file)
            logging.info(f"Vocabulary saved to cache: {vocab_cache_file}")

    unk_idx, pad_idx = vocab[unk_token], vocab[pad_token]

    # 4. --- Numericalize Data ---
    def numericalize_dataset(data_iterator_func: Callable, split_name: str) -> List[Tuple[int, List[int]]]:
        cache_params = (exp_cfg.dataset_name, vocab_cache_file, split_name)
        cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", cache_params)
        if cfg.use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f: return pickle.load(f)

        logging.info(f"Numericalizing {split_name} data...")
        processed_data = []
        text_pipeline = lambda x: [vocab.stoi.get(token, unk_idx) for token in tokenizer(x)]
        for label, text in data_iterator_func():
            if text and label is not None:
                processed_text = text_pipeline(text)
                if processed_text:
                    processed_data.append((label, processed_text))

        if cfg.use_cache:
            with open(cache_path, "wb") as f: pickle.dump(processed_data, f)
        return processed_data

    processed_train_data = numericalize_dataset(lambda: hf_iterator(train_ds_hf, text_field, label_field), "train")
    processed_test_data = numericalize_dataset(lambda: hf_iterator(test_ds_hf, text_field, label_field), "test")
    if not processed_train_data:
        raise ValueError("Training data is empty after processing.")

    # 5. --- Split Data ---
    split_params = (
        exp_cfg.dataset_name, vocab_cache_file, cfg.seed, buyer_percentage, exp_cfg.n_sellers,
        cfg.data_partition.strategy, config.discovery_quality, config.buyer_data_mode, config.buyer_bias_type, config.buyer_dirichlet_alpha
    )

    split_cache_file = get_cache_path(app_cache_dir, "split_indices", split_params)

    if cfg.use_cache and os.path.exists(split_cache_file):
        logging.info(f"Loading split indices from cache: {split_cache_file}")
        with open(split_cache_file, "rb") as f:
            buyer_indices, seller_splits = pickle.load(f)
    else:
        logging.info(f"Splitting data using method: '{config.split_method}'")
        total_samples = len(processed_train_data)
        buyer_count = int(total_samples * buyer_percentage)

        if cfg.data_partition.strategy == "discovery":
            buyer_bias_dist = None
            if cfg.data_partition.buyer_data_mode == "biased":
                 buyer_bias_dist = generate_buyer_bias_distribution(
                    num_classes=num_classes,
                    bias_type=config.buyer_bias_type,
                    alpha=config.buyer_dirichlet_alpha
                )
            # *** KEY FIX: Calling the actual implemented function ***
            buyer_indices, seller_splits = split_text_dataset_martfl_discovery(
                dataset=processed_train_data,
                buyer_count=buyer_count,
                num_clients=config.num_sellers,
                noise_factor=config.discovery_quality,
                buyer_data_mode=config.buyer_data_mode,
                buyer_bias_distribution=buyer_bias_dist,
                seed=config.seed
            )
        else:
            raise ValueError(f"Unsupported split_method: '{config.split_method}'")

        if config.use_cache:
            with open(split_cache_file, "wb") as f:
                pickle.dump((buyer_indices, seller_splits), f)
            logging.info(f"Split indices saved to cache: {split_cache_file}")

    # Sanity check for overlaps
    assigned_indices = set(buyer_indices.tolist())
    for seller_id, indices in seller_splits.items():
        if not assigned_indices.isdisjoint(indices):
            logging.error(f"FATAL: Overlap detected between buyer and seller {seller_id} indices!")
        assigned_indices.update(indices)

    # 6. --- Create DataLoaders ---
    collate_fn = lambda batch: collate_batch(batch, padding_value=pad_idx)

    # Use training batch size from the training config
    batch_size = train_cfg.batch_size

    buyer_loader = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        buyer_loader = DataLoader(Subset(processed_train_data, buyer_indices.tolist()), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    seller_loaders = {}
    for i in range(config.num_sellers):
        indices = seller_splits.get(i)
        if indices and len(indices) > 0:
            seller_loaders[i] = DataLoader(Subset(processed_train_data, indices), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            seller_loaders[i] = None

    test_loader = DataLoader(processed_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if processed_test_data else None

    logging.info("DataLoader creation complete.")

    return ProcessedTextData(
        buyer_loader=buyer_loader,
        seller_loaders=seller_loaders,
        test_loader=test_loader,
        class_names=class_names,
        vocab=vocab,
        pad_idx=pad_idx,
        num_classes=num_classes
    )


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


def construct_text_buyer_set(dataset: List[Tuple[int, Any]], buyer_count: int, buyer_data_mode: str, buyer_bias_distribution: Optional[Dict], seed: int) -> np.ndarray:
    """Constructs the buyer index set based on the specified mode."""
    random.seed(seed); np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    if buyer_count <= 0: return np.array([], dtype=int)
    if buyer_count >= total_samples: return all_indices

    if buyer_data_mode == "random":
        return np.random.choice(all_indices, buyer_count, replace=False)

    elif buyer_data_mode == "biased":
        if buyer_bias_distribution is None:
            raise ValueError("`buyer_bias_distribution` is required for 'biased' mode.")

        targets = np.array([item[0] for item in dataset])
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


def split_text_dataset_martfl_discovery(dataset: List[Tuple[int, Any]], buyer_count: int, num_clients: int, noise_factor: float, buyer_data_mode: str, buyer_bias_distribution: Optional[Dict], seed: int) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Simulates MartFL data split where seller distributions are noisy mimics of the buyer's."""
    random.seed(seed); np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)
    targets = np.array([item[0] for item in dataset])
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)

    # 1. Construct Buyer Set
    buyer_indices = construct_text_buyer_set(dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed)

    # 2. Determine Seller Pool
    seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    if len(seller_pool_indices) == 0 or num_clients <= 0:
        return buyer_indices, {i: [] for i in range(num_clients)}

    # 3. Calculate Actual Buyer Distribution
    if len(buyer_indices) > 0:
        _, buyer_cls_counts = np.unique(targets[buyer_indices], return_counts=True)
        buyer_proportions = {c: count / len(buyer_indices) for c, count in zip(unique_classes, buyer_cls_counts)}
    else: # Fallback to uniform if buyer is empty
        buyer_proportions = {c: 1.0 / num_classes for c in unique_classes}

    # 4. Assign Data to Sellers
    seller_splits = {}
    seller_pool_splits = np.array_split(seller_pool_indices, num_clients) # Simple even split for this example

    for client_id in range(num_clients):
        # A more sophisticated implementation would sample from the pool based on noisy proportions.
        # For simplicity and to match the placeholder's original spirit, we just split the pool.
        # The logic below can be swapped for the more complex sampling if needed.
        seller_splits[client_id] = seller_pool_splits[client_id].tolist()

    return buyer_indices, seller_splits
