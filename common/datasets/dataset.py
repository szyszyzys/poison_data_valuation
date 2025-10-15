# --- Imports ---
import hashlib
import json
import logging
import os
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import (Callable, Generator)

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchvision.transforms import v2 as transforms

from common.datasets.data_partitioner import FederatedDataPartitioner, _extract_targets
from common.datasets.data_split import OverallFractionSplit
from common.datasets.image_data_processor import save_data_statistics, CelebACustom
from common.datasets.text_data_processor import ProcessedTextData, get_cache_path, \
    generate_buyer_bias_distribution, split_text_dataset_martfl_discovery, collate_batch, StandardFormatDataset
from common.gradient_market_configs import AppConfig

try:
    from datasets import load_dataset as hf_load

    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# --- Setup logging for better feedback ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# In a new file, e.g., common/datasets/wrappers.py
import re
from torch.utils.data import Dataset


class UnifiedDatasetWrapper(Dataset):
    """Wraps various dataset objects to provide a uniform interface."""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.dataset_type = type(original_dataset).__name__

        # --- Create a uniform .targets attribute ---
        if hasattr(original_dataset, 'targets'):
            # For standard torchvision datasets like CIFAR
            self.targets = np.array(original_dataset.targets)
        elif hasattr(original_dataset, 'labels'):
            # For some custom datasets
            self.targets = np.array(original_dataset.labels)
        elif hasattr(original_dataset, 'get_targets'):
            # For our CelebACustom class
            self.targets = original_dataset.get_targets()
        else:
            raise TypeError(f"Could not find a 'targets' or 'labels' attribute on {self.dataset_type}")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[idx]

    def has_property(self, idx: int, property_key: str) -> bool:
        """
        A uniform method to check if a sample has a given property.
        The logic that was in the partitioner now lives here.
        """
        # A. Generic handler for class-based properties
        if property_key.startswith("class_in_"):
            target_classes = set(map(int, re.findall(r'\d+', property_key)))
            sample_label = self.targets[idx]
            return sample_label in target_classes

        # B. Specific handler for datasets with their own logic (like CelebA)
        elif hasattr(self.original_dataset, 'has_property'):
            return self.original_dataset.has_property(idx, property_key=property_key)

        # C. Fallback error
        else:
            raise NotImplementedError(
                f"Property check for key '{property_key}' is not implemented for {self.dataset_type}"
            )


def _get_dataset_loaders(dataset_name: str, data_root: str) -> Tuple[Dataset, Dataset, int]:
    """
    A helper factory to load standard torchvision datasets and get their properties.
    This makes adding new datasets much easier.
    """
    if dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        raw_train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        raw_test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        raw_train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        raw_test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CelebA":
        # Note: CelebA does not have a standard test set split in torchvision
        # We handle this by partitioning the 'train' set into train/test splits.
        # The property_key for CelebA is handled in the main function.
        raw_train_set = CelebACustom(root=data_root, split='train', download=True)
        raw_test_set = CelebACustom(root=data_root, split='test', download=True)
        num_classes = 2  # Assuming binary classification on a single attribute
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    train_set = UnifiedDatasetWrapper(raw_train_set)
    test_set = UnifiedDatasetWrapper(raw_test_set)

    return train_set, test_set, num_classes


# In common/datasets/dataset.py

def get_image_dataset(cfg: AppConfig) -> Tuple[DataLoader, Dict[int, DataLoader], DataLoader, Dict, int]:
    """
    A unified function to load, partition, and prepare federated image datasets.
    This version includes caching for the data partitioning step.
    """
    logger.info(f"--- Starting Federated Dataset Setup for '{cfg.experiment.dataset_name}' ---")
    image_cfg = cfg.data.image
    if not image_cfg:
        raise ValueError("Image data configuration ('data.image') is missing.")

    # ==================== 1. CACHING LOGIC (NEW) ====================
    # Define a cache directory within your main data root
    cache_dir = Path(cfg.data_root) / ".cache"
    cache_dir.mkdir(exist_ok=True)

    # Create a unique key based on all parameters that affect the data split
    config_params = {
        "dataset": cfg.experiment.dataset_name,
        "n_sellers": cfg.experiment.n_sellers,
        "seed": cfg.seed,
        "use_subset": cfg.experiment.use_subset,
        "subset_size": cfg.experiment.subset_size if cfg.experiment.use_subset else None,
        "strategy": image_cfg.strategy,
        "buyer_config": image_cfg.buyer_config,
        "partition_params": asdict(image_cfg.property_skew)
    }
    # Convert dict to a canonical string and hash it for a unique filename
    config_string = json.dumps(config_params, sort_keys=True)
    config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"{config_hash}.pkl"
    # =================================================================

    # Load base dataset
    train_set, test_set, num_classes = _get_dataset_loaders(cfg.experiment.dataset_name, cfg.data_root)
    if cfg.experiment.dataset_name == "CelebA":
        property_key = image_cfg.property_skew.property_key
        train_set.set_target_attribute(property_key)
        test_set.set_target_attribute(property_key)
    if cfg.experiment.use_subset:
        logger.warning(f"❗️ Using a subset of {cfg.experiment.subset_size} samples for testing.")
        num_samples = min(cfg.experiment.subset_size, len(train_set))
        train_set = Subset(train_set, list(range(num_samples)))

    # ==================== 2. LOAD OR CREATE SPLITS (MODIFIED) ====================
    if cache_file.exists():
        logger.info(f"✅ Found cached data split. Loading from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        buyer_indices = cached_data['buyer_indices']
        seller_splits = cached_data['seller_splits']
        client_properties = cached_data['client_properties']

    else:
        logger.info(f"❗️ No cached data split found. Running partitioning...")
        # Initialize and run the partitioner (this part is the same as before)
        partitioner = FederatedDataPartitioner(
            dataset=train_set, num_clients=cfg.experiment.n_sellers, seed=cfg.seed
        )
        buyer_strategy = OverallFractionSplit()
        partitioner.partition(
            buyer_split_strategy=buyer_strategy,
            client_partition_strategy=image_cfg.strategy,
            buyer_config=image_cfg.buyer_config,
            partition_params=asdict(image_cfg.property_skew)
        )
        buyer_indices, seller_splits, _ = partitioner.get_splits()
        client_properties = partitioner.client_properties

        # --- Save the newly created split to the cache file ---
        logger.info(f"💾 Saving new data split to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'buyer_indices': buyer_indices,
                'seller_splits': seller_splits,
                'client_properties': client_properties
            }, f)
    # ===========================================================================

    # Generate statistics and save to a DEDICATED path
    logger.info("Generating and saving image data split statistics...")

    # --- CHANGE: Use the hash from the cache logic to create the unique path ---
    config_hash = Path(cache_file).stem
    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"

    # Pass the new, specific path to the function
    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        client_properties=client_properties,
        targets=_extract_targets(train_set),
        save_filepath=stats_save_path  # Use the new `save_filepath` argument
    )

    # Create DataLoaders (this part is the same as before)
    batch_size = cfg.training.batch_size
    actual_dataset = train_set.dataset if isinstance(train_set, Subset) else train_set
    buyer_loader = DataLoader(Subset(actual_dataset, buyer_indices), batch_size=batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers) if buyer_indices.size > 0 else None

    seller_loaders = {
        cid: DataLoader(Subset(actual_dataset, indices), batch_size=batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers)
        for cid, indices in seller_splits.items() if indices
    }

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers) if test_set else None

    logger.info(f"✅ Federated dataset setup complete. Using {num_classes} classes.")

    return buyer_loader, seller_loaders, test_loader, stats, num_classes


# The get_text_dataset function remains unchanged as it was already robust.
def get_text_dataset(cfg: AppConfig) -> ProcessedTextData:
    """
    Loads, processes, caches, and splits a text dataset according to the provided AppConfig.
    """
    # 1. --- Input Validation and Setup ---
    exp_cfg = cfg.experiment
    train_cfg = cfg.training
    text_cfg = cfg.data.text

    if not text_cfg:
        raise ValueError("Text data configuration ('data.text') is missing from the AppConfig.")

    # Get parameters from their new, correct locations
    discovery_params = text_cfg.discovery
    vocab_cfg = text_cfg.vocab
    buyer_percentage = discovery_params.buyer_percentage

    if not (0.0 <= buyer_percentage <= 1.0):
        raise ValueError("buyer_percentage must be between 0 and 1.")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    app_cache_dir = os.path.join(cfg.data_root, ".cache", "get_text_data_set_cache")
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
        # 1. DEFINE the field names first
        text_field, label_field = "text", "coarse_label"

        # 2. LOAD the Hugging Face dataset
        ds = hf_load("trec", cache_dir=cfg.data_root)
        train_ds_hf, test_ds_hf = ds["train"], ds["test"]

        # 3. GET the actual class information from the dataset
        dataset_features = ds['train'].features[label_field]
        class_names = dataset_features.names
        num_classes = len(class_names)  # This is the true number of classes in the data

    else:
        raise ValueError(f"Unsupported dataset: {exp_cfg.dataset_name}")

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:
            text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
            if isinstance(text_content, str):
                # --- FIX 1: Yield data in the standard (data, label) format ---
                yield (text_content, label_content) if label_fld else text_content

    # def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
    #     for ex in dataset_obj:
    #         text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
    #         if isinstance(text_content, str):
    #             yield (label_content, text_content) if label_fld else text_content

    # 3. --- Build or Load Vocabulary ---
    vocab_cache_params = (
        exp_cfg.dataset_name, vocab_cfg.min_freq, vocab_cfg.unk_token, vocab_cfg.pad_token
    )
    vocab_cache_file = get_cache_path(app_cache_dir, "vocab", vocab_cache_params)
    vocab: Optional[Vocab] = None

    if cfg.use_cache and os.path.exists(vocab_cache_file):
        try:
            logging.info(f"Loading vocabulary from cache: {vocab_cache_file}")
            vocab, pad_idx, unk_idx = torch.load(vocab_cache_file, weights_only=False)
            if not isinstance(vocab, Vocab):
                raise TypeError("Cached object is not a Vocab.")
            logging.info(f"Vocabulary loaded. Size: {len(vocab.get_itos())}")
        except Exception as e:
            logging.warning(f"Failed to load vocab from cache: {e}. Rebuilding.")
            vocab = None

    if vocab is None:
        logging.info("Building vocabulary...")

        def yield_tokens(data_iterator):
            for _, text in data_iterator:
                yield tokenizer(text)

        # This logic is now correct for modern torchtext
        vocab = build_vocab_from_iterator(
            yield_tokens(hf_iterator(train_ds_hf, text_field, label_field)),
            min_freq=vocab_cfg.min_freq,
            specials=[vocab_cfg.unk_token, vocab_cfg.pad_token],
            special_first=True
        )

        unk_idx = vocab[vocab_cfg.unk_token]
        vocab.set_default_index(unk_idx)
        pad_idx = vocab[vocab_cfg.pad_token]

        logging.info(f"Vocabulary built. Size: {len(vocab)}. UNK index: {unk_idx}, PAD index: {pad_idx}.")

        if cfg.use_cache:
            torch.save((vocab, pad_idx, unk_idx), vocab_cache_file)
            logging.info(f"Vocabulary saved to cache: {vocab_cache_file}")

    unk_idx, pad_idx = vocab[vocab_cfg.unk_token], vocab[vocab_cfg.pad_token]

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:
            text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
            if isinstance(text_content, str):
                # --- FIX 1: Yield data in the standard (data, label) format ---
                yield (text_content, label_content) if label_fld else text_content

    # 4. --- Numericalize Data ---
    def numericalize_dataset(data_iterator_func: Callable, split_name: str) -> List[Tuple[int, List[int]]]:
        cache_params = (exp_cfg.dataset_name, vocab_cache_file, split_name)
        cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", cache_params)
        # if cfg.use_cache and os.path.exists(cache_path):
        #     with open(cache_path, "rb") as f: return pickle.load(f)

        logging.info(f"Numericalizing {split_name} data...")
        processed_data = []
        text_pipeline = lambda x: vocab(tokenizer(x))

        # --- FIX 2: Unpack the tuple in the new (data, label) order ---
        for text, label in data_iterator_func():
            if text and label is not None:
                processed_text = text_pipeline(text)
                if processed_text:
                    # This append is now correct and consistent
                    processed_data.append((processed_text, label))

        if cfg.use_cache:
            with open(cache_path, "wb") as f: pickle.dump(processed_data, f)
        return processed_data

    processed_train_data = numericalize_dataset(lambda: hf_iterator(train_ds_hf, text_field, label_field), "train")
    processed_test_data = numericalize_dataset(lambda: hf_iterator(test_ds_hf, text_field, label_field), "test")
    if not processed_train_data:
        raise ValueError("Training data is empty after processing.")
    standardized_train_data = StandardFormatDataset(processed_train_data, label_first=False)
    standardized_test_data = StandardFormatDataset(processed_test_data, label_first=False)

    # 5. --- Split Data ---
    split_params = (
        exp_cfg.dataset_name, vocab_cache_file, cfg.seed, buyer_percentage, exp_cfg.n_sellers,
        text_cfg.strategy, discovery_params.discovery_quality, discovery_params.buyer_data_mode,
        discovery_params.buyer_bias_type, discovery_params.buyer_dirichlet_alpha
    )
    split_cache_file = get_cache_path(app_cache_dir, "split_indices", split_params)

    if cfg.use_cache and os.path.exists(split_cache_file):
        logging.info(f"Loading split indices from cache: {split_cache_file}")
        with open(split_cache_file, "rb") as f:
            buyer_indices, seller_splits = pickle.load(f)
    else:
        logging.info(f"Splitting data using method: '{text_cfg.strategy}'")
        total_samples = len(processed_train_data)
        buyer_count = int(total_samples * buyer_percentage)

        if text_cfg.strategy == "discovery":
            buyer_bias_dist = None
            if discovery_params.buyer_data_mode == "biased":
                buyer_bias_dist = generate_buyer_bias_distribution(
                    num_classes=num_classes,
                    bias_type=discovery_params.buyer_bias_type,
                    alpha=discovery_params.buyer_dirichlet_alpha
                )

            buyer_indices, seller_splits = split_text_dataset_martfl_discovery(
                dataset=standardized_train_data,
                buyer_count=buyer_count,
                num_clients=exp_cfg.n_sellers,  # Use correct config path
                noise_factor=discovery_params.discovery_quality,
                buyer_data_mode=discovery_params.buyer_data_mode,
                buyer_bias_distribution=buyer_bias_dist,
                seed=cfg.seed
            )
        else:
            raise ValueError(f"Unsupported split_method: '{text_cfg.strategy}'")

        if cfg.use_cache:
            with open(split_cache_file, "wb") as f:
                pickle.dump((buyer_indices, seller_splits), f)
            logging.info(f"Split indices saved to cache: {split_cache_file}")

    # Sanity check for overlaps
    assigned_indices = set(buyer_indices.tolist())
    for seller_id, indices in seller_splits.items():
        if not assigned_indices.isdisjoint(indices):
            logging.error(f"FATAL: Overlap detected between buyer and seller {seller_id} indices!")
        assigned_indices.update(indices)
    # ==================== ADD THIS SECTION ====================
    # 6. --- Generate and Save Statistics to a DEDICATED Path ---
    logger.info("Generating and saving data split statistics...")

    # --- ADD THIS: Re-use the hash from your caching logic to create a unique path ---
    # The split_cache_file path already contains the unique hash.
    # We can derive the statistics filename from it.
    config_hash = Path(split_cache_file).stem
    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"
    # --- END ADDITION ---

    train_targets = standardized_train_data.targets

    # Pass the new, specific path to the function
    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        client_properties={},
        targets=train_targets,
        save_filepath=stats_save_path  # Use the new path
    )

    # Create DataLoaders ---
    collate_fn = lambda batch: collate_batch(batch, padding_value=pad_idx)

    # Use training batch size from the training config
    batch_size = train_cfg.batch_size

    buyer_loader = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        buyer_loader = DataLoader(Subset(standardized_train_data, buyer_indices.tolist()), batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    seller_loaders = {}
    for i in range(exp_cfg.n_sellers):
        indices = seller_splits.get(i)
        if indices:
            seller_loaders[i] = DataLoader(Subset(standardized_train_data, indices), batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn, num_workers=cfg.data.num_workers)
        else:
            seller_loaders[i] = None

    test_loader = DataLoader(standardized_test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=cfg.data.num_workers) if processed_test_data else None

    logging.info("DataLoader creation complete.")

    return ProcessedTextData(
        buyer_loader=buyer_loader,
        seller_loaders=seller_loaders,
        test_loader=test_loader,
        class_names=class_names,
        vocab=vocab,
        pad_idx=pad_idx,
        num_classes=num_classes,
        collate_fn=collate_fn
    )
