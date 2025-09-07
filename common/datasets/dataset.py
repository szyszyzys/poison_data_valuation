# --- Imports ---
import logging
import os
import pickle
import random
from dataclasses import asdict
from typing import (Any, Callable, Dict, Generator, List, Optional, Tuple)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

from common.datasets.data_partitioner import FederatedDataPartitioner, _extract_targets
from common.datasets.image_data_processor import load_dataset_with_property, save_data_statistics, CelebACustom
from common.datasets.text_data_processor import ProcessedTextData, hf_datasets_available, get_cache_path, \
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


def get_image_dataset(cfg: AppConfig) -> Tuple[DataLoader, Dict[int, DataLoader], DataLoader, Dict, int]:
    """
    A unified function to load, partition, and prepare federated image datasets.
    """
    logger.info(f"--- Starting Federated Dataset Setup for '{cfg.experiment.dataset_name}' ---")

    # This is the new, correct way to get data-related settings.
    image_cfg = cfg.data.image
    if not image_cfg:
        raise ValueError("Image data configuration ('data.image') is missing from the AppConfig.")

    # 1. Load the base dataset using the helper function
    # The property_key is now accessed from its new location.
    property_key = image_cfg.property_skew.property_key

    # This logic is now simpler, as the key is always present in the config
    train_set, _ = load_dataset_with_property(cfg.experiment.dataset_name, property_key=property_key)

    if cfg.experiment.use_subset:
        logger.warning(
            f"❗️ Using a small subset of the dataset ({cfg.experiment.subset_size} samples) for pipeline testing."
        )
        # Ensure subset_size is not larger than the dataset
        num_samples = min(cfg.experiment.subset_size, len(train_set))
        indices = list(range(num_samples))
        train_set = Subset(train_set, indices)

    # 2. Initialize the partitioner
    partitioner = FederatedDataPartitioner(
        dataset=train_set,
        num_clients=cfg.experiment.n_sellers,
        seed=cfg.seed
    )

    # 3. Execute the partitioning strategy defined in the config
    logger.info(f"Applying '{image_cfg.strategy}' partitioning strategy...")

    partition_params_dict = asdict(image_cfg.property_skew)

    partitioner.partition(
        strategy=image_cfg.strategy,
        buyer_config=image_cfg.buyer_config,
        partition_params=partition_params_dict  # Pass the dictionary here
    )
    buyer_indices, seller_splits, test_indices = partitioner.get_splits()

    # 4. Generate and save statistics (no changes needed here)
    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        client_properties=partitioner.client_properties,
        targets=_extract_targets(train_set),
        output_dir=cfg.experiment.save_path
    )
    actual_dataset = train_set.dataset if isinstance(train_set, Subset) else train_set
    if isinstance(actual_dataset, CelebACustom):
        # For CelebA, the "class" is the binary property (e.g., Blond_Hair vs. not)
        num_classes = 2
    elif hasattr(actual_dataset, 'targets'):
        # For other datasets, calculate from the unique targets
        num_classes = len(np.unique(actual_dataset.targets))
    else:
        # Fallback if a dataset has neither .targets nor a special case
        raise ValueError("Could not determine the number of classes for the dataset.")

    batch_size = cfg.training.batch_size
    actual_dataset = train_set.dataset if isinstance(train_set, Subset) else train_set

    buyer_loader = DataLoader(Subset(actual_dataset, buyer_indices), batch_size=batch_size, shuffle=True) if len(
        buyer_indices) > 0 else None

    seller_loaders = {
        cid: DataLoader(Subset(actual_dataset, indices), batch_size=batch_size, shuffle=True)
        for cid, indices in seller_splits.items() if indices
    }

    # --- UPDATED TEST LOADER LOGIC WITH FALLBACK ---
    if len(test_indices) > 0:
        logger.info(f"Creating test loader from dedicated test set of size {len(test_indices)}.")
        test_loader = DataLoader(Subset(actual_dataset, test_indices), batch_size=batch_size, shuffle=False)
    elif len(buyer_indices) > 0:
        logger.warning(
            "❗️ No test indices found. As a fallback for this test run, "
            "the test loader will use the buyer's data. "
            "Do NOT use this for final results."
        )
        test_loader = DataLoader(Subset(actual_dataset, buyer_indices), batch_size=batch_size, shuffle=False)
    else:
        logger.error("No test or buyer indices available to create a test loader.")
        test_loader = None

    logger.info(f"✅ Federated dataset setup complete. Found {num_classes} classes.")

    return buyer_loader, seller_loaders, test_loader, stats, num_classes


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

    # 4. --- Numericalize Data ---
    def numericalize_dataset(data_iterator_func: Callable, split_name: str) -> List[Tuple[int, List[int]]]:
        cache_params = (exp_cfg.dataset_name, vocab_cache_file, split_name)
        cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", cache_params)
        if cfg.use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f: return pickle.load(f)

        logging.info(f"Numericalizing {split_name} data...")
        processed_data = []
        text_pipeline = lambda x: vocab(tokenizer(x))
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
    standardized_train_data = StandardFormatDataset(processed_train_data, label_first=True)
    standardized_test_data = StandardFormatDataset(processed_test_data, label_first=True)

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

    # 6. --- Create DataLoaders ---
    collate_fn = lambda batch: collate_batch(batch, padding_value=pad_idx)

    # Use training batch size from the training config
    batch_size = train_cfg.batch_size

    buyer_loader = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        buyer_loader = DataLoader(Subset(standardized_train_data, buyer_indices.tolist()), batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)

    seller_loaders = {}
    for i in range(exp_cfg.n_sellers):
        indices = seller_splits.get(i)
        if indices:
            seller_loaders[i] = DataLoader(Subset(standardized_train_data, indices), batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn)
        else:
            seller_loaders[i] = None

    test_loader = DataLoader(standardized_test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn) if processed_test_data else None

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
