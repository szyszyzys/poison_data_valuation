# data_processing_buyer_centric.py
import os
import logging
import random
from typing import Dict, List, Tuple, Any, Optional, Iterator, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets as vision_datasets, transforms

# --- TorchText Imports ---
import torchtext
from torchtext.datasets import AG_NEWS, TREC
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants and Text Helpers ---
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

NUM_CLASSES = { # Add more as needed
    'cifar10': 10, 'cifar': 10, # Allow 'cifar' alias
    'mnist': 10,
    'fmnist': 10,
    'ag_news': 4,
    'trec': 6,
}

def yield_tokens(data_iterator: Iterator, tokenizer) -> Iterator[List[str]]:
    """Helper to yield tokens from text data (label, text)."""
    for _, text in data_iterator:
        yield tokenizer(text)

def build_text_vocab(train_iter: Iterator, tokenizer) -> torchtext.vocab.Vocab:
    """Builds vocabulary from training data iterator."""
    logging.info("Building vocabulary...")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer),
                                      min_freq=1, specials=special_symbols, special_first=True)
    vocab.set_default_index(UNK_IDX)
    logging.info(f"Vocabulary built. Size: {len(vocab)}")
    return vocab

def preprocess_text_dataset(
    raw_dataset_iter: Iterator,
    tokenizer,
    vocab,
    num_classes: int # Needed to adjust TREC labels if necessary
) -> Tuple[List[torch.Tensor], List[int]]:
    """Tokenizes and numericalizes raw text dataset. Adjusts labels to be 0-based."""
    processed_text = []
    processed_labels = []
    logging.info("Preprocessing text data (tokenizing and numericalizing)...")
    max_label_found = -1
    min_label_found = float('inf')

    # Determine label adjustment needed (some datasets like AG_NEWS are 1-based)
    # Peek at first item to see label range if possible (iterators make this hard)
    # Assume adjustment needed if labels seem > num_classes-1
    # Safer: Adjust based on dataset name? AG_NEWS/TREC are 1-based.
    adjust_label = lambda x: int(x) - 1 # Default adjustment

    for label, text in raw_dataset_iter:
        numerical_label = adjust_label(label)
        max_label_found = max(max_label_found, numerical_label)
        min_label_found = min(min_label_found, numerical_label)

        if not (0 <= numerical_label < num_classes):
             logging.warning(f"Unexpected label value after adjustment: {numerical_label} (original {label}) for {num_classes} classes. Check dataset format/adjustment.")
             # Skip? Or clamp? Clamping might hide issues. Let's keep it for now.
             # numerical_label = max(0, min(numerical_label, num_classes - 1))

        processed_labels.append(numerical_label)
        numerical_text = vocab(tokenizer(text))
        processed_text.append(torch.tensor(numerical_text, dtype=torch.int64))

    logging.info(f"Text preprocessing complete. Label range found (adjusted): [{min_label_found}, {max_label_found}]")
    if max_label_found >= num_classes or min_label_found < 0:
         logging.warning("Potential issue with label range after preprocessing.")
    return processed_text, processed_labels

class PreprocessedDataset(Dataset):
    """Simple Dataset wrapper for preprocessed (numericalized) data."""
    def __init__(self, texts: List[torch.Tensor], labels: List[int]):
        self.texts = texts
        self.labels = labels
        if len(texts) != len(labels): raise ValueError("Texts/labels mismatch!")
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]
    # Add .targets attribute for compatibility with logic expecting it
    @property
    def targets(self):
        return self.labels

def create_text_collate_fn(pad_idx: int):
    """Creates a collate_fn for text data loaders (adds padding, BOS/EOS)."""
    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            # Add BOS/EOS tokens
            processed_text = torch.cat([torch.tensor([BOS_IDX]), _text, torch.tensor([EOS_IDX])], dim=0)
            text_list.append(processed_text)
            lengths.append(processed_text.size(0))
        # Pad sequences to the max length in the batch
        padded_text = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
        labels = torch.tensor(label_list, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        return padded_text, labels, lengths # Return lengths for potential use in PackedSequence
    return collate_batch

# --- Dataset Loading (Combined Vision and Text) ---

def _get_vision_transforms(dataset_name: str, normalize: bool) -> transforms.Compose:
    """Gets appropriate transforms for vision datasets."""
    transform_list = [transforms.ToTensor()]
    if normalize:
        if dataset_name == "fmnist":
            transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))
        elif dataset_name == "cifar10" or dataset_name == "cifar":
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        elif dataset_name == "mnist":
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_list)

def _load_raw_dataset(dataset_name: str, data_path: str = './data', normalize: bool = True) -> Tuple[Optional[Any], Optional[Any], Optional[str], Optional[List[str]]]:
    """Loads raw train/test data, returns type and class names."""
    dataset_name = dataset_name.lower()
    os.makedirs(data_path, exist_ok=True)
    dataset_type = None
    class_names = None
    train_data_raw, test_data_raw = None, None
    num_classes = NUM_CLASSES.get(dataset_name)

    if num_classes is None:
        logging.error(f"Dataset '{dataset_name}' not recognized or NUM_CLASSES not defined.")
        return None, None, None, None

    try:
        if dataset_name in ['cifar10', 'cifar', 'mnist', 'fmnist']:
            dataset_type = 'vision'
            transform = _get_vision_transforms(dataset_name, normalize)
            if dataset_name in ['cifar10', 'cifar']:
                train_data_raw = vision_datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
                test_data_raw = vision_datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            elif dataset_name == 'mnist':
                train_data_raw = vision_datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
                test_data_raw = vision_datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
                class_names = [str(i) for i in range(10)]
            elif dataset_name == 'fmnist':
                train_data_raw = vision_datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
                test_data_raw = vision_datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            logging.info(f"{dataset_name.upper()} vision dataset loaded successfully.")

        elif dataset_name in ['ag_news', 'trec']:
            dataset_type = 'text'
            if dataset_name == 'ag_news':
                train_data_raw = AG_NEWS(root=data_path, split='train')
                test_data_raw = AG_NEWS(root=data_path, split='test')
                class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            elif dataset_name == 'trec':
                train_data_raw = TREC(root=data_path, split='train')
                test_data_raw = TREC(root=data_path, split='test')
                class_names = ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Numeric']
            logging.info(f"{dataset_name.upper()} text dataset iterators loaded successfully.")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} loading not implemented.")

        return train_data_raw, test_data_raw, dataset_type, class_names

    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
        return None, None, None, None


# --- Buyer-Centric Distribution and Assignment Logic ---

def _define_buyer_distribution(
    dataset_name: str,
    buyer_type: str,
    bias_params: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """Defines the target probability distribution P_buyer."""
    # ... (Implementation from previous data_utils_buyer_centric.py) ...
    num_classes = NUM_CLASSES.get(dataset_name.lower())
    if num_classes is None: logging.error(f"Num classes undefined: {dataset_name}"); return None
    if buyer_type == 'unbiased':
        distribution = np.ones(num_classes) / num_classes
    elif buyer_type == 'biased':
        if bias_params is None or 'focus_classes' not in bias_params or 'focus_ratio' not in bias_params:
            logging.error("Missing params for biased buyer."); return None
        focus_classes = bias_params['focus_classes']
        focus_ratio = bias_params['focus_ratio']
        if not (0 < focus_ratio < 1): logging.error("focus_ratio invalid."); return None
        if not focus_classes or not all(0 <= c < num_classes for c in focus_classes):
            logging.error(f"Invalid focus_classes: {focus_classes}"); return None
        distribution = np.zeros(num_classes)
        num_focus, num_other = len(focus_classes), num_classes - len(focus_classes)
        prob_per_focus = focus_ratio / num_focus if num_focus > 0 else 0
        prob_per_other = (1 - focus_ratio) / num_other if num_other > 0 else 0
        for i in range(num_classes): distribution[i] = prob_per_focus if i in focus_classes else prob_per_other
        distribution /= distribution.sum()
    else: logging.error(f"Unknown buyer_type: {buyer_type}"); return None
    logging.info(f"Defined buyer distribution ({buyer_type}): {np.round(distribution, 3)}")
    return distribution

def _generate_seller_distributions(
    buyer_distribution: np.ndarray,
    num_sellers: int,
    concentration_k: float,
    seed: int
) -> Optional[List[np.ndarray]]:
    """Generates seller target distributions P_seller_i using Dirichlet."""
    # ... (Implementation from previous data_utils_buyer_centric.py) ...
    if concentration_k <= 0: logging.error("Concentration k must be positive."); return None
    num_classes = len(buyer_distribution)
    alpha = buyer_distribution * concentration_k
    if np.any(alpha <= 0):
        logging.warning(f"Generated alpha has non-positive values: {alpha}. Clamping.")
        alpha = np.maximum(alpha, 1e-6)
    np.random.seed(seed)
    try:
        seller_distributions = np.random.dirichlet(alpha, num_sellers)
        logging.info(f"Generated {num_sellers} seller distributions (k={concentration_k}).")
        return [dist for dist in seller_distributions]
    except Exception as e: logging.error(f"Error generating Dirichlet distributions: {e}", exc_info=True); return None

def _assign_data_based_on_distributions(
    # Takes processed dataset with potentially .targets list or attribute
    processed_dataset: Union[Dataset, PreprocessedDataset],
    data_indices: List[int], # Indices available for sellers
    seller_target_distributions: List[np.ndarray],
    num_sellers: int,
    num_classes: int,
    seed: int
) -> Optional[Dict[str, List[int]]]:
    """Assigns data indices to sellers to match target distributions."""
    # ... (Implementation from previous data_utils_buyer_centric.py) ...
    # Needs access to targets correctly
    try:
        if hasattr(processed_dataset, 'targets'):
            # Handles vision datasets and our PreprocessedDataset
            dataset_targets = np.array(processed_dataset.targets)
        else: # Fallback for datasets without .targets (might be slow)
            logging.warning("Dataset lacks .targets, iterating to get labels for assignment...")
            dataset_targets = np.array([processed_dataset[i][1] for i in range(len(processed_dataset))])
    except Exception as e:
        logging.error(f"Could not get targets for data assignment: {e}")
        return None

    if len(seller_target_distributions) != num_sellers: logging.error("Distributions != num_sellers."); return None
    if not data_indices: logging.warning("No data indices for sellers."); return {f"seller_{i}": [] for i in range(num_sellers)}

    np.random.seed(seed); random.seed(seed)
    indices_in_pool = np.array(data_indices)
    targets_in_pool = dataset_targets[indices_in_pool]
    num_samples_in_pool = len(indices_in_pool)
    samples_per_seller = num_samples_in_pool // num_sellers
    extra_samples = num_samples_in_pool % num_sellers
    idx_by_class_in_pool = {k: [] for k in range(num_classes)}
    for i, original_idx in enumerate(indices_in_pool):
        label = targets_in_pool[i]
        idx_by_class_in_pool[label].append(original_idx)
    for k in range(num_classes): random.shuffle(idx_by_class_in_pool[k])
    class_pointers = {k: 0 for k in range(num_classes)}
    assigned_indices_count = 0
    seller_splits = {f"seller_{i}": [] for i in range(num_sellers)}
    logging.info(f"Assigning {num_samples_in_pool} samples to {num_sellers} sellers...")
    for client_idx in range(num_sellers):
        seller_id = f"seller_{client_idx}"
        target_dist = seller_target_distributions[client_idx]
        target_samples_this_client = samples_per_seller + (1 if client_idx < extra_samples else 0)
        if target_samples_this_client == 0: continue
        target_counts_per_class = (target_dist * target_samples_this_client).astype(float)
        # -- Precise count adjustment logic --
        rounded_counts = np.round(target_counts_per_class).astype(int)
        diff = target_samples_this_client - rounded_counts.sum()
        if diff != 0:
            residuals = target_counts_per_class - rounded_counts
            adjust_order = np.argsort(residuals if diff > 0 else -residuals)
            for i in range(abs(diff)):
                idx_to_adjust = adjust_order[i % num_classes]
                rounded_counts[idx_to_adjust] += np.sign(diff)
            rounded_counts = np.maximum(0, rounded_counts)
            final_sum = rounded_counts.sum()
            if final_sum != target_samples_this_client:
                 rounded_counts[0] += target_samples_this_client - final_sum
        # -- End adjustment --
        client_assigned_indices = []
        for class_id in range(num_classes):
            needed_count = rounded_counts[class_id]
            if needed_count == 0: continue
            start_ptr = class_pointers[class_id]
            available_indices = idx_by_class_in_pool[class_id]
            num_available = len(available_indices) - start_ptr
            actual_count = min(needed_count, num_available)
            if actual_count > 0:
                end_ptr = start_ptr + actual_count
                sampled_original_indices = available_indices[start_ptr:end_ptr]
                client_assigned_indices.extend(sampled_original_indices)
                class_pointers[class_id] = end_ptr
        seller_splits[seller_id] = client_assigned_indices
        assigned_indices_count += len(client_assigned_indices)
    unassigned_count = num_samples_in_pool - assigned_indices_count
    if unassigned_count > 0: logging.warning(f"{unassigned_count} pool samples unassigned.")
    logging.info("Finished assigning data.")
    return seller_splits

def _create_buyer_specific_test_subset(
    # Takes processed test dataset
    processed_test_dataset: Union[Dataset, PreprocessedDataset],
    buyer_distribution: np.ndarray,
    num_test_samples: Optional[int] = None,
    seed: int = 42
) -> Subset:
    """Creates a Subset of the test set matching the buyer distribution."""
    # ... (Implementation from previous data_utils_buyer_centric.py) ...
    # Needs access to targets correctly
    try:
        if hasattr(processed_test_dataset, 'targets'):
            test_targets = np.array(processed_test_dataset.targets)
            original_test_indices = np.arange(len(processed_test_dataset))
        else: # Fallback
            logging.warning("Test dataset lacks .targets, iterating...")
            test_targets = np.array([processed_test_dataset[i][1] for i in range(len(processed_test_dataset))])
            original_test_indices = np.arange(len(processed_test_dataset))
    except Exception as e:
        logging.error(f"Could not get targets for buyer test set creation: {e}. Returning full test set.")
        return Subset(processed_test_dataset, list(range(len(processed_test_dataset))))

    np.random.seed(seed)
    num_classes = len(buyer_distribution)
    test_idx_by_class = {k: original_test_indices[test_targets == k] for k in range(num_classes)}
    total_available_test = len(processed_test_dataset)
    target_total_samples = num_test_samples if num_test_samples is not None else total_available_test
    target_test_counts = (buyer_distribution * target_total_samples).astype(int)
    # -- Precise count adjustment logic (same as assignment) --
    diff = target_total_samples - target_test_counts.sum()
    if diff != 0:
        residuals = (buyer_distribution * target_total_samples) - target_test_counts
        adjust_order = np.argsort(residuals if diff > 0 else -residuals)
        for i in range(abs(diff)):
            idx_to_adjust = adjust_order[i % num_classes]
            target_test_counts[idx_to_adjust] += np.sign(diff)
        target_test_counts = np.maximum(0, target_test_counts)
        target_test_counts[0] += target_total_samples - target_test_counts.sum()
    # -- End adjustment --
    buyer_test_indices = []
    for class_id in range(num_classes):
        needed = target_test_counts[class_id]
        available = test_idx_by_class.get(class_id, np.array([]))
        if len(available) < needed:
            # logging.warning(f"Buyer test set: Class {class_id} needs {needed}, only {len(available)} avail.")
            needed = len(available)
        if needed > 0:
            chosen_indices = np.random.choice(available, needed, replace=False)
            buyer_test_indices.extend(chosen_indices)
    logging.info(f"Created buyer test subset with {len(buyer_test_indices)} samples.")
    return Subset(processed_test_dataset, buyer_test_indices)


# --- Main Function (Replaces get_data_set) ---

def prepare_buyer_centric_loaders(
    dataset_name: str,
    num_sellers: int,
    batch_size: int,
    buyer_distribution_type: str, # 'unbiased' or 'biased'
    client_difference_level: str, # 'low' or 'high'
    buyer_bias_params: Optional[Dict] = None, # if buyer_distribution_type=='biased'
    concentration_k_low: float = 100.0,
    concentration_k_high: float = 1.0,
    buyer_test_set_size: Optional[int] = None,
    normalize_data: bool = True,
    data_path: str = './data',
    seed: int = 42
) -> Tuple[
        Optional[Dict[str, DataLoader]], # seller_loaders
        Optional[DataLoader],            # buyer_test_loader
        Optional[List[str]],             # class_names
        Optional[torchtext.vocab.Vocab], # vocab (for text)
        Optional[callable]               # collate_fn (for text)
    ]:
    """
    Prepares DataLoaders for buyer-centric FL setup.

    Args:
        dataset_name (str): Name like 'CIFAR10', 'MNIST', 'AG_NEWS', 'TREC'.
        num_sellers (int): Number of sellers (clients).
        batch_size (int): Batch size for DataLoaders.
        buyer_distribution_type (str): 'unbiased' or 'biased'.
        client_difference_level (str): 'low' or 'high'. Controls variance of seller dists.
        buyer_bias_params (dict, optional): Params if buyer is biased.
            e.g., {'focus_classes': [0,1], 'focus_ratio': 0.8}.
        concentration_k_low (float): Dirichlet 'k' for 'low' difference level.
        concentration_k_high (float): Dirichlet 'k' for 'high' difference level.
        buyer_test_set_size (int, optional): Target size for buyer test subset. None=use all.
        normalize_data (bool): Apply standard normalization (for vision).
        data_path (str): Path to download/load data.
        seed (int): Random seed.

    Returns:
        Tuple: (seller_loaders, buyer_test_loader, class_names, vocab, collate_fn)
               vocab and collate_fn are None for vision datasets.
               Returns (None, None, None, None, None) on failure.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # 1. Load Raw Data
    train_data_raw, test_data_raw, dataset_type, class_names = _load_raw_dataset(
        dataset_name, data_path, normalize
    )
    if train_data_raw is None: return None, None, None, None, None
    num_classes = NUM_CLASSES[dataset_name.lower()]

    # 2. Preprocess Text Data (if applicable)
    proc_train_data = train_data_raw
    proc_test_data = test_data_raw
    text_collate_fn = None
    vocab = None

    if dataset_type == 'text':
        tokenizer = get_tokenizer('basic_english')
        # Convert iterators to lists to allow multiple passes
        logging.info("Converting text iterators to lists for processing...")
        train_list = list(train_data_raw)
        test_list = list(test_data_raw)
        logging.info("Conversion complete.")

        vocab = build_text_vocab(iter(train_list), tokenizer)
        processed_train_text, processed_train_labels = preprocess_text_dataset(iter(train_list), tokenizer, vocab, num_classes)
        processed_test_text, processed_test_labels = preprocess_text_dataset(iter(test_list), tokenizer, vocab, num_classes)

        proc_train_data = PreprocessedDataset(processed_train_text, processed_train_labels)
        proc_test_data = PreprocessedDataset(processed_test_text, processed_test_labels)
        text_collate_fn = create_text_collate_fn(PAD_IDX)
    elif dataset_type != 'vision':
        logging.error(f"Unsupported dataset type: {dataset_type}")
        return None, None, None, None, None

    # 3. Define Buyer Distribution
    buyer_distribution = _define_buyer_distribution(
        dataset_name, buyer_distribution_type, buyer_bias_params
    )
    if buyer_distribution is None: return None, None, None, None, None

    # 4. Define Concentration Factor 'k'
    concentration_k = concentration_k_low if client_difference_level == 'low' else concentration_k_high
    logging.info(f"Using concentration k = {concentration_k} for '{client_difference_level}' difference level.")

    # 5. Generate Target Seller Distributions
    seller_target_distributions = _generate_seller_distributions(
        buyer_distribution, num_sellers, concentration_k, seed
    )
    if seller_target_distributions is None: return None, None, None, None, None

    # 6. Assign Data to Sellers using the *processed* training data
    all_train_indices = list(range(len(proc_train_data)))
    seller_splits = _assign_data_based_on_distributions(
        processed_dataset=proc_train_data,
        data_indices=all_train_indices,
        seller_target_distributions=seller_target_distributions,
        num_sellers=num_sellers,
        num_classes=num_classes,
        seed=seed
    )
    if seller_splits is None: return None, None, None, None, None

    # 7. Create Seller DataLoaders using the *processed* training data
    seller_loaders = {}
    for seller_id, indices in seller_splits.items():
        if not indices:
            logging.warning(f"Seller {seller_id} has no data assigned.")
            seller_loaders[seller_id] = None
            continue
        seller_subset = Subset(proc_train_data, indices)
        seller_loaders[seller_id] = DataLoader(seller_subset, batch_size=batch_size, shuffle=True,
                                               num_workers=0, collate_fn=text_collate_fn) # Use collate_fn for text
    logging.info(f"Created {len(seller_loaders)} seller DataLoaders.")

    # 8. Create Buyer-Specific Test Loader using the *processed* test data
    buyer_test_subset = _create_buyer_specific_test_subset(
        proc_test_data, buyer_distribution, buyer_test_set_size, seed
    )
    buyer_test_loader = DataLoader(buyer_test_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, collate_fn=text_collate_fn) # Use collate_fn for text
    logging.info("Created buyer-specific test DataLoader.")

    return seller_loaders, buyer_test_loader, class_names, vocab, text_collate_fn

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running Example Usage (Vision) ---")
    sellers_v, buyer_test_v, classes_v, _, _ = prepare_buyer_centric_loaders(
        dataset_name='CIFAR', # Alias for CIFAR10
        num_sellers=5,
        batch_size=32,
        buyer_distribution_type='biased',
        client_difference_level='low',
        buyer_bias_params={'focus_classes': [0, 1], 'focus_ratio': 0.7},
        concentration_k_low=50.0,
        concentration_k_high=0.8,
        buyer_test_set_size=1000,
        seed=123
    )
    if sellers_v and buyer_test_v:
        print(f"Vision Example: {len(sellers_v)} sellers, {len(buyer_test_v.dataset)} buyer test samples.")
    else:
        print("Vision Example Failed.")


    print("\n--- Running Example Usage (Text) ---")
    sellers_t, buyer_test_t, classes_t, vocab_t, collate_t = prepare_buyer_centric_loaders(
        dataset_name='AG_NEWS',
        num_sellers=8,
        batch_size=16,
        buyer_distribution_type='unbiased',
        client_difference_level='high',
        # buyer_bias_params=None, # Not needed for unbiased
        concentration_k_low=50.0,
        concentration_k_high=0.5,
        buyer_test_set_size=500,
        normalize_data=False, # Text doesn't use this normalization
        seed=456
    )
    if sellers_t and buyer_test_t:
        print(f"Text Example: {len(sellers_t)} sellers, {len(buyer_test_t.dataset)} buyer test samples.")
        print(f"Vocab size: {len(vocab_t) if vocab_t else 'N/A'}")
        # Can test the collate function
        # first_batch = next(iter(sellers_t['seller_0']))
        # print("Sample batch shape (text, labels, lengths):", [t.shape for t in first_batch])
    else:
        print("Text Example Failed.")