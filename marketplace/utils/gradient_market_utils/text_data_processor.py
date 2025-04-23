from torch.utils.data import DataLoader, Subset
import logging
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import AG_NEWS, TREC
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Configure logging (optional, but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def placeholder_splitter(*args, **kwargs):
    logging.warning("Using placeholder splitter function!")
    dataset = kwargs.get('dataset', [])
    buyer_count = kwargs.get('buyer_count', 0)
    num_sellers = kwargs.get('num_clients', 1)
    total_len = len(dataset)
    all_indices = np.arange(total_len)
    buyer_indices = np.random.choice(all_indices, buyer_count, replace=False) if buyer_count > 0 else np.array([],
                                                                                                               dtype=int)
    seller_pool = np.setdiff1d(all_indices, buyer_indices)
    seller_splits_list = np.array_split(seller_pool, num_sellers) if num_sellers > 0 else []
    seller_splits = {i: list(split) for i, split in enumerate(seller_splits_list)}
    return buyer_indices, seller_splits


def placeholder_generate_bias(*args, **kwargs):
    logging.warning("Using placeholder bias generation function!")
    num_classes = kwargs.get('num_classes', 2)
    return {i: 1.0 / num_classes for i in range(num_classes)}


split_dataset_martfl_discovery = placeholder_splitter  # Replace with actual import
split_dataset_by_label = placeholder_splitter  # Replace with actual import
split_dataset_buyer_seller_improved = placeholder_splitter  # Replace with actual import
generate_buyer_bias_distribution = placeholder_generate_bias  # Replace with actual import


# Placeholder functions for demonstration if imports fail
def generate_buyer_bias_distribution(num_classes, bias_type, alpha, **kwargs):
    logging.warning("Using placeholder `generate_buyer_bias_distribution`")
    if bias_type == 'dirichlet':
        bias = np.random.dirichlet([alpha] * num_classes)
        return bias
    else:
        return np.ones(num_classes) / num_classes


def split_text_dataset_martfl_discovery(dataset, buyer_count, num_clients, client_data_count, noise_factor,
                                        buyer_data_mode, buyer_bias_distribution, seed, **kwargs):
    logging.warning("Using placeholder `split_text_dataset_martfl_discovery`")
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    buyer_indices = all_indices[:buyer_count]
    seller_all_indices = all_indices[buyer_count:]
    seller_splits_list = np.array_split(seller_all_indices, num_clients)
    seller_splits = {i: list(split) for i, split in enumerate(seller_splits_list)}
    return buyer_indices, seller_splits


def split_dataset_by_label(dataset, buyer_count, num_sellers, label_index, seed, **kwargs):
    logging.warning("Using placeholder `split_dataset_by_label`")
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    buyer_indices = all_indices[:buyer_count]
    seller_all_indices = all_indices[buyer_count:]
    seller_splits_list = np.array_split(seller_all_indices, num_sellers)
    seller_splits = {i: list(split) for i, split in enumerate(seller_splits_list)}
    return buyer_indices, seller_splits


def split_dataset_buyer_seller_improved(dataset, buyer_count, num_sellers, split_method, dirichlet_alpha, n_adversaries,
                                        label_index, seed, **kwargs):
    logging.warning(f"Using placeholder `split_dataset_buyer_seller_improved` for method {split_method}")
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    buyer_indices = all_indices[:buyer_count]
    seller_all_indices = all_indices[buyer_count:]
    seller_splits_list = np.array_split(seller_all_indices, num_sellers)
    seller_splits = {i: list(split) for i, split in enumerate(seller_splits_list)}
    return buyer_indices, seller_splits


# --- End Placeholder Splitters ---


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collate_batch(batch: List[Tuple[int, List[int]]], vocab: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of text data (label, list_of_token_ids).
    Pads sequences to the maximum length in the batch.

    Args:
        batch: A list of tuples, where each tuple contains (label, list_of_token_ids).
        vocab: The vocabulary object used for padding index lookup.

    Returns:
        A tuple containing:
        - labels (torch.Tensor): Tensor of labels (batch_size).
        - padded_texts (torch.Tensor): Tensor of padded text sequences (batch_size, max_seq_len).
    """
    label_list, text_list = [], []
    for (_label, _text_list) in batch:
        label_list.append(_label)
        # Convert list of token IDs to tensor
        processed_text = torch.tensor(_text_list, dtype=torch.int64)
        text_list.append(processed_text)

    labels = torch.tensor(label_list, dtype=torch.int64)

    # Get padding index from vocab
    pad_token = '<pad>'
    try:
        # Standard way for torchtext.vocab.Vocab
        pad_idx = vocab.get_stoi()[pad_token]
    except AttributeError:
        # Fallback if vocab is a simple dict or has a different structure
        logging.warning(f"vocab object doesn't have get_stoi method. Trying direct access vocab['{pad_token}']")
        try:
            pad_idx = vocab[pad_token]
        except KeyError:
            raise ValueError(f"'{pad_token}' token not found in vocabulary.")
        except TypeError:
            raise TypeError(
                f"Vocabulary object (type: {type(vocab)}) is not subscriptable like a dictionary or doesn't contain '{pad_token}'.")

    if pad_idx is None:  # Should ideally be caught by KeyError above, but double check
        raise ValueError(f"'{pad_token}' token not found in vocabulary or resolved to None.")

    padded_texts = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=pad_idx
    )
    return labels, padded_texts


def yield_tokens(data_iter: Any, tokenizer: Any) -> Any:
    """ Helper to yield tokens from a dataset iterator. """
    for _, text in data_iter:
        yield tokenizer(text)


# --- End Text Data Helper Functions ---


def get_text_data_set(
        dataset_name: str,
        buyer_percentage: float = 0.01,
        num_sellers: int = 10,
        batch_size: int = 64,
        split_method: str = "discovery",
        n_adversaries: int = 0,
        # save_path: str = './result', # Path might be needed for stats saving later
        # --- Discovery Split Specific Params ---
        discovery_quality: float = 0.3,
        buyer_data_mode: str = "random",
        buyer_bias_type: str = "dirichlet",
        buyer_dirichlet_alpha: float = 0.3,
        discovery_client_data_count: int = 0,  # Target count per client for discovery (0 for even split)
        # --- Other Split Method Params ---
        seller_dirichlet_alpha: float = 0.7,  # Alpha for non-discovery seller splits
        seed: int = 42
) -> Tuple[Optional[DataLoader], Dict[int, Optional[DataLoader]], Optional[DataLoader], List[
    str], Any, int]:  # MODIFIED return type hint
    """
    Loads, preprocesses, and splits AG_NEWS or TREC text datasets.

    Args:
        dataset_name (str): Name of the dataset ("AG_NEWS" or "TREC").
        buyer_percentage (float): Fraction of training data for the buyer. Must be in [0, 1].
        num_sellers (int): Number of seller clients. Must be positive.
        batch_size (int): Batch size for DataLoaders. Must be positive.
        split_method (str): Method for splitting data ('discovery', 'label', 'dirichlet', etc.).
        n_adversaries (int): Number of adversaries (used by some split methods). Must be non-negative.
        discovery_quality (float): Noise factor for 'discovery' split.
        buyer_data_mode (str): How buyer data is selected ('random', 'biased').
        buyer_bias_type (str): If 'biased', how bias is generated ('dirichlet', etc.).
        buyer_dirichlet_alpha (float): Alpha for Dirichlet distribution buyer bias. Must be positive.
        discovery_client_data_count (int): Target samples per client in discovery split (0 for even split). Non-negative.
        seller_dirichlet_alpha (float): Alpha for Dirichlet distribution seller splits (non-discovery). Must be positive.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple containing:
        - Optional[DataLoader]: Buyer data loader (None if buyer gets no data).
        - Dict[int, Optional[DataLoader]]: Dictionary of seller data loaders (Value is None if seller gets no data).
        - Optional[DataLoader]: Test data loader (None if test set is empty).
        - List[str]: List of class names.
        - Any: The vocabulary object (typically `torchtext.vocab.Vocab`).
        - int: The numerical index used for the padding token ('<pad>'). # MODIFIED return doc

    Raises:
        ValueError: If input parameters are invalid.
        ModuleNotFoundError: If required libraries (e.g., spacy model) are not found.
        ImportError: If required splitting functions cannot be imported.
        RuntimeError: If dataset loading or processing fails unexpectedly.
        TypeError: If vocabulary object has unexpected type in collate_fn.
    """
    # --- Input Validation ---
    # (validation code remains the same)
    if dataset_name not in ["AG_NEWS", "TREC"]:
        raise ValueError(f"Unsupported text dataset: {dataset_name}. Choose 'AG_NEWS' or 'TREC'.")
    if not 0.0 <= buyer_percentage <= 1.0:
        raise ValueError(f"buyer_percentage must be between 0.0 and 1.0, got {buyer_percentage}")
    if num_sellers <= 0:
        raise ValueError(f"num_sellers must be positive, got {num_sellers}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_adversaries < 0:
        raise ValueError(f"n_adversaries cannot be negative, got {n_adversaries}")
    if buyer_dirichlet_alpha <= 0:
        raise ValueError(f"buyer_dirichlet_alpha must be positive, got {buyer_dirichlet_alpha}")
    if discovery_client_data_count < 0:
        raise ValueError(f"discovery_client_data_count cannot be negative, got {discovery_client_data_count}")
    if seller_dirichlet_alpha <= 0:
        raise ValueError(f"seller_dirichlet_alpha must be positive, got {seller_dirichlet_alpha}")

    # --- Setup ---
    # (setup code remains the same)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Constants ---
    label_index_in_tuple = 0
    data_root = './data'
    unk_token = "<unk>"
    pad_token = "<pad>"  # Define pad token string

    # --- Load Raw Data and Tokenizer ---
    # (loading code remains the same)
    logging.info(f"Loading TEXT dataset: {dataset_name}")
    try:
        tokenizer = get_tokenizer('basic_english')
    except ModuleNotFoundError:
        logging.error(
            "Spacy or its 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        raise
    except Exception as e:
        logging.error(f"Error initializing tokenizer: {e}")
        raise

    if dataset_name == "AG_NEWS":
        try:
            train_iter, test_iter = AG_NEWS(root=data_root, split=('train', 'test'))
            num_classes = 4
            class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            label_offset = 1
            logging.info(f"AG_NEWS dataset loaded. Num classes: {num_classes}. Original labels are 1-based.")
        except Exception as e:
            logging.error(f"Failed to load AG_NEWS from {data_root}. Error: {e}")
            raise RuntimeError(f"Failed to load AG_NEWS: {e}") from e
    elif dataset_name == "TREC":
        try:
            train_iter, test_iter = TREC(root=data_root, split=('train', 'test'))
            temp_train_iter_for_labels, _ = TREC(root=data_root, split=('train', 'test'))
            unique_labels = set(label for label, text in temp_train_iter_for_labels)
            del temp_train_iter_for_labels
            num_classes = len(unique_labels)
            expected_trec_classes = 6
            if num_classes == expected_trec_classes:
                class_names = ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Numeric']
            else:
                logging.warning(
                    f"Inferred {num_classes} classes for TREC, expected {expected_trec_classes}. Using numeric names.")
                class_names = [str(i) for i in sorted(list(unique_labels))]
            label_offset = 0
            logging.info(f"TREC dataset loaded. Num classes: {num_classes}. Original labels are 0-based.")
        except Exception as e:
            logging.error(f"Failed to load TREC from {data_root}. Error: {e}")
            raise RuntimeError(f"Failed to load TREC: {e}") from e
    else:
        raise ValueError(f"Dataset loading logic missing for: {dataset_name}")

    # --- Build Vocabulary ---
    logging.info("Building vocabulary...")
    pad_idx: Optional[int] = None  # Initialize pad_idx
    try:
        vocab_train_iter, _ = (AG_NEWS if dataset_name == "AG_NEWS" else TREC)(root=data_root, split=('train', 'test'))
        specials = [unk_token, pad_token]  # Use defined constants

        vocab = build_vocab_from_iterator(yield_tokens(vocab_train_iter, tokenizer), specials=specials)
        vocab.set_default_index(vocab[unk_token])
        del vocab_train_iter

        # --- >>> GET PADDING INDEX HERE <<< ---
        if pad_token not in vocab.get_stoi():
            raise ValueError(f"'{pad_token}' special token was not correctly added to vocabulary.")
        pad_idx = vocab[pad_token]  # Get the index for the padding token
        # --- >>> END GET PADDING INDEX <<< ---

        logging.info(f"Vocabulary built. Size: {len(vocab)}. '{pad_token}' index: {pad_idx}")

    except Exception as e:
        logging.error(f"Failed during vocabulary building: {e}")
        raise RuntimeError(f"Vocabulary building failed: {e}") from e

    # Ensure pad_idx was successfully assigned
    if pad_idx is None:
        raise RuntimeError("Padding index could not be determined after vocabulary building.")

    # --- Define Text Processing Pipeline ---
    text_pipeline = lambda x: vocab(tokenizer(x))

    # --- Pre-process Data into Memory ---
    # (processing code remains the same)
    logging.info("Processing dataset into numerical format (list of (label, token_ids))...")
    processed_train_data: List[Tuple[int, List[int]]] = []
    skipped_train_samples = 0
    for label, text in train_iter:
        processed_text_list = text_pipeline(text)
        if not processed_text_list:
            skipped_train_samples += 1
            continue
        adjusted_label = label - label_offset
        processed_train_data.append((adjusted_label, processed_text_list))
    del train_iter

    processed_test_data: List[Tuple[int, List[int]]] = []
    skipped_test_samples = 0
    for label, text in test_iter:
        processed_text_list = text_pipeline(text)
        if not processed_text_list:
            skipped_test_samples += 1
            continue
        adjusted_label = label - label_offset
        processed_test_data.append((adjusted_label, processed_text_list))
    del test_iter

    logging.info(
        f"Dataset processed. Train samples: {len(processed_train_data)} (skipped {skipped_train_samples}), "
        f"Test samples: {len(processed_test_data)} (skipped {skipped_test_samples})"
    )

    dataset = processed_train_data
    test_set = processed_test_data

    if not dataset:
        raise ValueError("Processed training dataset is empty.")

    # --- Calculate Buyer Count ---
    # (buyer count calculation remains the same)
    total_samples = len(dataset)
    buyer_count = min(int(total_samples * buyer_percentage), total_samples)
    logging.info(f"Total train samples available for splitting: {total_samples}")
    logging.info(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

    # --- Data Splitting ---
    # (splitting logic remains the same, using placeholder functions)
    buyer_indices: np.ndarray = np.array([], dtype=int)
    seller_splits: Dict[int, List[int]] = {}
    buyer_bias_distribution_generated = None

    logging.info(f"Splitting data using method: '{split_method}'")
    if split_method == "discovery":
        # ... (discovery split logic) ...
        if buyer_data_mode == "biased":
            try:
                buyer_bias_distribution_generated = generate_buyer_bias_distribution(num_classes, buyer_bias_type,
                                                                                     buyer_dirichlet_alpha)
                logging.info(f"Generated buyer bias distribution: {buyer_bias_distribution_generated}")
            except NameError:
                raise ImportError("Missing: generate_buyer_bias_distribution")
            except Exception as e:
                raise RuntimeError(f"Buyer bias generation failed: {e}") from e
        elif buyer_data_mode != "random":
            raise ValueError(f"Unknown buyer_data_mode: '{buyer_data_mode}'")
        try:
            buyer_indices, seller_splits = split_text_dataset_martfl_discovery(dataset, buyer_count, num_sellers,
                                                                               discovery_client_data_count,
                                                                               discovery_quality, buyer_data_mode,
                                                                               buyer_bias_distribution_generated, seed)
        except NameError:
            raise ImportError("Missing: split_text_dataset_martfl_discovery")
        except Exception as e:
            raise RuntimeError(f"Discovery split failed: {e}") from e

    elif split_method == "label":
        # ... (label split logic) ...
        try:
            buyer_indices, seller_splits = split_dataset_by_label(dataset, buyer_count, num_sellers,
                                                                  label_index_in_tuple, seed)
        except NameError:
            raise ImportError("Missing: split_dataset_by_label")
        except Exception as e:
            raise RuntimeError(f"Label split failed: {e}") from e

    elif split_method in ["dirichlet", "iid", "quantity"]:
        # ... (other split logic) ...
        try:
            buyer_indices, seller_splits = split_dataset_buyer_seller_improved(dataset, buyer_count, num_sellers,
                                                                               split_method, seller_dirichlet_alpha,
                                                                               n_adversaries, label_index_in_tuple,
                                                                               seed)
        except NameError:
            raise ImportError(f"Missing: split_dataset_buyer_seller_improved")
        except Exception as e:
            raise RuntimeError(f"{split_method.capitalize()} split failed: {e}") from e
    else:
        raise ValueError(f"Unsupported split_method: '{split_method}'.")

    # --- Sanity Checks After Splitting ---
    # (sanity checks remain the same)
    assigned_indices = set(buyer_indices)
    total_seller_samples_assigned = 0
    for seller_id, indices in seller_splits.items():
        if indices is None: continue
        if not isinstance(indices, (list, np.ndarray)):
            logging.warning(f"Seller {seller_id} indices type {type(indices)} not list/array. Assuming empty.")
            indices = []
            seller_splits[seller_id] = indices
        indices_set = set(indices)
        if not assigned_indices.isdisjoint(indices_set):
            logging.error(f"Overlap detected: buyer indices and seller {seller_id} indices!")
        assigned_indices.update(indices_set)
        total_seller_samples_assigned += len(indices)
    logging.info(
        f"Splitting complete. Buyer samples: {len(buyer_indices)}, Total seller samples: {total_seller_samples_assigned}")
    unassigned_count = total_samples - len(assigned_indices)
    if unassigned_count > 0:
        logging.warning(f"{unassigned_count} samples were not assigned.")
    elif unassigned_count < 0:
        logging.error("Error in index accounting: More indices assigned than available.")

    # --- Create DataLoaders ---
    # (DataLoader creation remains the same, using dynamic_collate_fn)
    logging.info("Creating DataLoaders...")
    dynamic_collate_fn = lambda batch: collate_batch(batch, vocab)  # Captures vocab

    buyer_loader: Optional[DataLoader] = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        try:
            buyer_subset = Subset(dataset, buyer_indices)
            buyer_loader = DataLoader(buyer_subset, batch_size=batch_size, shuffle=True, collate_fn=dynamic_collate_fn,
                                      drop_last=False)
            logging.info(f"Buyer DataLoader created with {len(buyer_indices)} samples.")
        except Exception as e:
            raise RuntimeError(f"Buyer DataLoader creation failed: {e}") from e
    else:
        logging.warning("Buyer has no data samples assigned. Buyer DataLoader will be None.")

    seller_loaders: Dict[int, Optional[DataLoader]] = {}
    actual_sellers_with_data = 0
    for i in range(num_sellers):
        indices = seller_splits.get(i)
        if indices is None or len(indices) == 0:
            seller_loaders[i] = None
            if i in seller_splits: logging.warning(f"Seller {i} has no data samples assigned.")
        else:
            try:
                seller_subset = Subset(dataset, indices)
                seller_loaders[i] = DataLoader(seller_subset, batch_size=batch_size, shuffle=True,
                                               collate_fn=dynamic_collate_fn, drop_last=False)
                actual_sellers_with_data += 1
            except Exception as e:
                logging.error(f"Failed to create DataLoader for seller {i}: {e}. Setting to None.")
                seller_loaders[i] = None
    logging.info(
        f"Seller DataLoaders created for {actual_sellers_with_data}/{num_sellers} sellers. Total samples: {total_seller_samples_assigned}")

    test_loader: Optional[DataLoader] = None
    if test_set is not None and len(test_set) > 0:
        try:
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn)
            logging.info(f"Test DataLoader created with {len(test_set)} samples.")
        except Exception as e:
            raise RuntimeError(f"Test DataLoader creation failed: {e}") from e
    else:
        logging.warning("Processed test set is empty or None. Test DataLoader will be None.")

    logging.info("Text data loading, processing, splitting, and DataLoader creation complete.")

    # --- >>> MODIFIED RETURN STATEMENT <<< ---
    return buyer_loader, seller_loaders, test_loader, class_names, vocab, pad_idx


import logging
import random
from typing import List, Dict, Tuple, Optional, Any  # Using Any for dataset elements now
import numpy as np

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Function: Calculate Target Counts (likely unchanged) ---
# This function is generic and usually doesn't depend on data format
def _calculate_target_counts(total_samples: int, proportions: Dict[int, float]) -> Dict[int, int]:
    """Calculates target counts per class ensuring sum matches total_samples."""
    # Ensure proportions sum close to 1 if needed (or handle potential rounding issues)
    # Sanitize proportions - remove any classes with non-positive proportion
    valid_proportions = {cls: p for cls, p in proportions.items() if p > 0}
    if not valid_proportions:
        # If all proportions are zero/negative, distribute uniformly among keys present in original dict
        logging.warning(
            "All proportions were non-positive. Falling back to uniform distribution over specified classes.")
        num_classes = len(proportions) if proportions else 1
        valid_proportions = {cls: 1.0 / num_classes for cls in proportions} if num_classes > 0 else {}
        if not valid_proportions:  # Edge case: empty proportions dict
            return {}

    # Normalize valid proportions if their sum isn't 1 (or close enough)
    prop_sum = sum(valid_proportions.values())
    if not np.isclose(prop_sum, 1.0):
        logging.debug(f"Normalizing proportions (Sum was {prop_sum}).")
        valid_proportions = {cls: p / prop_sum for cls, p in valid_proportions.items()}

    # Calculate initial counts based on valid, normalized proportions
    counts = {cls: int(round(prop * total_samples)) for cls, prop in valid_proportions.items()}
    current_sum = sum(counts.values())
    diff = total_samples - current_sum

    # Adjust counts to exactly match total_samples if needed
    if diff != 0 and valid_proportions:
        # Sort classes by proportion to adjust those with larger/smaller shares first
        sorted_classes = sorted(valid_proportions, key=valid_proportions.get, reverse=(diff > 0))
        idx = 0
        max_adjust_loops = 2 * len(sorted_classes)  # Safety break
        loops = 0
        while diff != 0 and loops < max_adjust_loops:
            cls_to_adjust = sorted_classes[idx % len(sorted_classes)]
            adjustment = 1 if diff > 0 else -1
            # Ensure counts don't go negative
            if counts[cls_to_adjust] + adjustment >= 0:
                counts[cls_to_adjust] += adjustment
                diff -= adjustment
            idx += 1
            loops += 1
        if diff != 0:
            logging.warning(f"Could not exactly match target counts after adjustment. Remaining difference: {diff}")

    # Ensure all classes from the original proportions dict are present, even if with 0 count
    final_counts = {cls: counts.get(cls, 0) for cls in proportions}
    return final_counts


# --- Helper Function: Construct Buyer Set (TEXT specific) ---
def construct_text_buyer_set(
        dataset: List[Tuple[int, Any]],  # Expects list of (label, data)
        buyer_count: int,
        buyer_data_mode: str,
        buyer_bias_distribution: Optional[Dict],
        seed: int
) -> np.ndarray:
    """
    Constructs the buyer set specifically for text data format (label at index 0).

    Args:
        dataset: List of (label, data) tuples.
        buyer_count: Number of samples for the buyer.
        buyer_data_mode: 'random' or 'biased'.
        buyer_bias_distribution: Required if mode is 'biased'. Keys are class labels (int).
        seed: Random seed.

    Returns:
        np.ndarray: Indices for the buyer set relative to the input dataset list.
    """
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    if buyer_count <= 0:
        logging.warning("Buyer count is non-positive. Returning empty buyer set.")
        return np.array([], dtype=int)
    if buyer_count > total_samples:
        logging.warning(
            f"Requested buyer count ({buyer_count}) exceeds total samples ({total_samples}). Using all samples for buyer.")
        return all_indices

    buyer_indices = np.array([], dtype=int)

    if buyer_data_mode == "random":
        buyer_indices = np.random.choice(all_indices, buyer_count, replace=False)
        logging.info(f"Constructed random buyer set with {len(buyer_indices)} samples.")

    elif buyer_data_mode == "biased":
        if buyer_bias_distribution is None:
            raise ValueError("`buyer_bias_distribution` must be provided for 'biased' mode.")

        logging.info(f"Constructing biased buyer set using labels at index 0.")
        try:
            # --- Access label at index 0 ---
            targets = np.array([dataset[i][0] for i in range(total_samples)])
            # -------------------------------
        except (IndexError, TypeError) as e:
            raise ValueError(
                f"Could not extract targets using index 0 in construct_text_buyer_set. "
                f"Ensure dataset items are tuples/lists with label first. Error: {e}"
            ) from e

        # Check if bias distribution keys match actual labels
        dataset_labels = set(targets)
        bias_labels = set(buyer_bias_distribution.keys())
        if not bias_labels.issubset(dataset_labels):
            logging.warning(
                f"Buyer bias distribution contains labels not present in dataset: {bias_labels - dataset_labels}")
        if not dataset_labels.issubset(bias_labels):
            logging.warning(
                f"Dataset contains labels not present in buyer bias distribution: {dataset_labels - bias_labels}. These classes will have 0 proportion.")
            # Ensure all dataset labels are in the distribution, potentially with 0 prop
            for lbl in dataset_labels:
                if lbl not in buyer_bias_distribution:
                    buyer_bias_distribution[lbl] = 0.0

        # Calculate precise target counts for the buyer based on bias distribution
        target_counts = _calculate_target_counts(buyer_count, buyer_bias_distribution)
        logging.debug(f"Buyer target counts: {target_counts}")

        buyer_indices_list = []
        indices_by_class = {int(c): list(np.where(targets == c)[0]) for c in dataset_labels}
        # Shuffle indices within each class to ensure random sampling
        for c in indices_by_class:
            random.shuffle(indices_by_class[c])

        class_pointers = {c: 0 for c in indices_by_class}  # Track usage within each class

        # First pass: try to get exact counts per class
        available_indices_set = set(all_indices)
        for cls, needed_count in target_counts.items():
            if needed_count <= 0 or cls not in indices_by_class: continue

            start_ptr = class_pointers[cls]
            class_idx_list = indices_by_class[cls]
            num_available_in_class = len(class_idx_list) - start_ptr

            num_to_take = min(needed_count, num_available_in_class)

            if num_to_take > 0:
                end_ptr = start_ptr + num_to_take
                sampled_for_class = class_idx_list[start_ptr:end_ptr]
                buyer_indices_list.extend(sampled_for_class)
                class_pointers[cls] = end_ptr  # Update pointer
                # Remove sampled indices from the general available pool
                available_indices_set.difference_update(sampled_for_class)

        # Second pass: If buyer_count not met, fill randomly from remaining pool
        current_count = len(buyer_indices_list)
        remaining_needed = buyer_count - current_count
        if remaining_needed > 0:
            logging.warning(
                f"Could not meet target counts for all classes in biased buyer selection ({current_count}/{buyer_count} sampled). Filling remaining {remaining_needed} randomly.")
            remaining_available_list = list(available_indices_set)  # Convert set to list
            if not remaining_available_list:
                logging.error("No remaining samples available to fill buyer count, but still needed!")
            elif remaining_needed > len(remaining_available_list):
                logging.warning(
                    f"Cannot fill remaining buyer count. Only {len(remaining_available_list)} samples left. Taking all.")
                buyer_indices_list.extend(remaining_available_list)
            else:
                fill_indices = np.random.choice(remaining_available_list, remaining_needed, replace=False)
                buyer_indices_list.extend(fill_indices)

        buyer_indices = np.array(buyer_indices_list)
        np.random.shuffle(buyer_indices)  # Shuffle the final buyer set
        logging.info(f"Biased buyer set constructed with {len(buyer_indices)} samples.")


    else:
        raise ValueError(f"Unknown buyer_data_mode: {buyer_data_mode}")

    return buyer_indices


# --- Main Splitting Function (TEXT specific) ---
def split_text_dataset_martfl_discovery(
        dataset: List[Tuple[int, Any]],  # Expects list of (label, data)
        buyer_count: int,
        num_clients: int,
        client_data_count: int = 0,  # If 0, distribute remaining seller pool evenly
        noise_factor: float = 0.3,
        buyer_data_mode: str = "random",  # Default to random
        buyer_bias_distribution: Optional[Dict] = None,
        seed: int = 42
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Simulates MartFL data split specifically for text data format (label at index 0).
    Seller distributions are noisy mimics of the buyer's distribution.

    Args:
        dataset (List[Tuple[int, Any]]): Input dataset as a list where each item
                                         is a tuple (label, data_features).
        buyer_count (int): Number of samples for the buyer.
        num_clients (int): Number of seller clients.
        client_data_count (int): Target samples per client. If 0, split seller pool evenly.
        noise_factor (float): Multiplicative uniform noise [1-f, 1+f] applied to buyer
                              proportions to generate seller proportions.
        buyer_data_mode (str): How the buyer set is constructed ('random' or 'biased').
        buyer_bias_distribution (Optional[Dict]): Distribution required if buyer_data_mode
                                                  is 'biased'. Keys are class labels (int).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[int, List[int]]]:
            - buyer_indices: NumPy array of indices allocated to the buyer.
            - seller_splits: Dictionary mapping client_id (int) to a list of indices
                             allocated to that seller.
    """
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    logging.info("--- Starting MartFL Discovery Split for Text Data ---")
    logging.info(f"Total samples: {total_samples}, Target buyer count: {buyer_count}, Sellers: {num_clients}")

    # 1. Construct Buyer Set using the text-specific helper
    buyer_indices = construct_text_buyer_set(
        dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed
    )
    actual_buyer_count = len(buyer_indices)
    logging.info(f"Step 1: Buyer set constructed ({buyer_data_mode}). Size: {actual_buyer_count}")

    # 2. Get Targets (using label at index 0) & Determine Seller Pool
    try:
        # --- Access label at index 0 ---
        targets = np.array([dataset[i][0] for i in range(total_samples)], dtype=int)
        # -------------------------------
        unique_classes_in_dataset = np.unique(targets)
        num_classes = len(unique_classes_in_dataset)
        logging.info(
            f"Step 2a: Extracted targets (label @ index 0). Found {num_classes} unique classes: {unique_classes_in_dataset}")

    except (IndexError, TypeError) as e:
        raise ValueError(
            f"Could not extract targets using index 0. "
            f"Ensure dataset is List[Tuple[int, Any]]. Error: {e}"
        ) from e

    seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    num_seller_pool = len(seller_pool_indices)
    logging.info(f"Step 2b: Seller pool identified. Size: {num_seller_pool}")

    # Handle edge cases: empty pool or no clients
    if num_seller_pool == 0:
        logging.warning("Seller pool is empty after buyer set construction. No data for sellers.")
        return buyer_indices, {i: [] for i in range(num_clients)}
    if num_clients <= 0:
        logging.warning("num_clients is zero or negative. No sellers to assign data to.")
        return buyer_indices, {}

    # 3. Calculate Actual Buyer Distribution
    buyer_proportions = {}
    if actual_buyer_count > 0:
        buyer_targets = targets[buyer_indices]
        unique_buyer_classes, buyer_cls_counts = np.unique(buyer_targets, return_counts=True)
        buyer_proportions = {int(c): count / actual_buyer_count for c, count in
                             zip(unique_buyer_classes, buyer_cls_counts)}
        logging.info(f"Step 3: Calculated actual buyer proportions: {buyer_proportions}")
    else:
        logging.warning(
            "Step 3: Buyer set is empty. Cannot calculate buyer proportions. Sellers will be assigned based on uniform distribution if needed.")
        # No proportions available, sellers will likely get uniform distribution based on pool

    # 4. Determine Samples Per Client
    # This logic remains the same
    if client_data_count <= 0:
        # Distribute evenly
        base_samples = num_seller_pool // num_clients
        extra_samples = num_seller_pool % num_clients
        client_sample_counts = [base_samples + 1 if i < extra_samples else base_samples for i in range(num_clients)]
        logging.info(
            f"Step 4: Distributing {num_seller_pool} seller samples evenly across {num_clients} clients. Counts per client: {client_sample_counts}")
    else:
        # Target specific count
        target_samples_per_client = client_data_count
        if target_samples_per_client * num_clients > num_seller_pool:
            logging.warning(
                f"Requested total client samples ({target_samples_per_client * num_clients}) > available seller pool ({num_seller_pool}). Clients might get fewer samples.")
        client_sample_counts = [target_samples_per_client] * num_clients
        logging.info(f"Step 4: Targeting {target_samples_per_client} samples per client.")

    # 5. Index Seller Pool by Class & Prepare Pointers
    pool_by_class = {int(c): [] for c in unique_classes_in_dataset}
    seller_pool_targets = targets[seller_pool_indices]
    for i, original_idx in enumerate(seller_pool_indices):
        label = int(seller_pool_targets[i])
        if label in pool_by_class:  # Check if label is valid (should be)
            pool_by_class[label].append(original_idx)
        else:
            logging.error(
                f"Label {label} found in seller pool targets but not in unique_classes_in_dataset! This indicates an error. Skipping index {original_idx}.")

    # Shuffle indices within each class list for random draws
    for c in pool_by_class:
        random.shuffle(pool_by_class[c])
    class_pointers = {c: 0 for c in pool_by_class}  # Track next available index per class
    logging.info(f"Step 5: Indexed seller pool by class.")

    # 6. Assign Data to Sellers
    seller_splits: Dict[int, List[int]] = {}
    assigned_indices_global = set()  # Use a set to track all assigned indices globally

    for client_id in range(num_clients):
        num_samples_for_this_client = client_sample_counts[client_id]
        client_indices_list = []

        if num_samples_for_this_client == 0:
            logging.debug(f"Client {client_id}: Target count is 0. Assigning empty list.")
            seller_splits[client_id] = []
            continue

        # Calculate noisy target proportions for this client
        noisy_proportions = {}
        if buyer_proportions:  # If buyer proportions exist
            total_noisy_prop = 0
            # Iterate over all classes present in the dataset
            for c in unique_classes_in_dataset:
                expected_prop = buyer_proportions.get(c, 0.0)  # Default to 0 if buyer lacked class
                factor = np.random.uniform(1 - noise_factor, 1 + noise_factor)
                noisy_prop = expected_prop * factor
                noisy_proportions[c] = max(0, noisy_prop)  # Ensure non-negative
                total_noisy_prop += noisy_proportions[c]
            # Normalize
            if total_noisy_prop > 0:
                noisy_proportions = {c: p / total_noisy_prop for c, p in noisy_proportions.items()}
            else:  # Fallback if all noisy props became 0 (unlikely but possible with noise_factor >= 1)
                logging.warning(f"Client {client_id}: All noisy proportions became zero. Falling back to uniform.")
                noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}
        else:  # Fallback if buyer was empty: use uniform distribution over dataset classes
            logging.debug(f"Client {client_id}: No buyer proportions. Using uniform distribution for seller target.")
            noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}

        # Calculate precise target counts for this client
        target_counts = _calculate_target_counts(num_samples_for_this_client, noisy_proportions)
        logging.debug(f"Client {client_id}: Target counts: {target_counts}")

        # Sample data based on target counts, drawing from the pool
        current_client_samples_count = 0
        for cls, needed_count in target_counts.items():
            if needed_count <= 0 or cls not in pool_by_class: continue

            start_ptr = class_pointers.get(cls, 0)
            available_indices_for_class = pool_by_class.get(cls, [])
            num_available_in_class = len(available_indices_for_class) - start_ptr

            num_to_sample = min(needed_count, num_available_in_class)

            if num_to_sample > 0:
                end_ptr = start_ptr + num_to_sample
                # Get candidate indices
                candidate_indices = available_indices_for_class[start_ptr:end_ptr]
                # Filter out any already assigned indices (shouldn't happen with pointer logic, but safe)
                newly_assigned_indices = []
                for idx in candidate_indices:
                    if idx not in assigned_indices_global:
                        newly_assigned_indices.append(idx)
                        assigned_indices_global.add(idx)  # Add to global set
                    else:
                        logging.warning(
                            f"Attempted to re-assign index {idx} (Class {cls}) to client {client_id}. This shouldn't happen with correct pointer logic. Skipping.")

                client_indices_list.extend(newly_assigned_indices)
                class_pointers[cls] = end_ptr  # Move pointer forward by the original num_to_sample
                current_client_samples_count += len(newly_assigned_indices)

        # Log if client got fewer samples than targeted
        if current_client_samples_count < num_samples_for_this_client:
            logging.warning(
                f"Client {client_id} assigned {current_client_samples_count} samples (targeted {num_samples_for_this_client}). This might be due to class data scarcity in the pool or filtered duplicates.")

        np.random.shuffle(client_indices_list)  # Shuffle samples for the client
        seller_splits[client_id] = client_indices_list
        logging.debug(f"Client {client_id}: Assigned {len(client_indices_list)} samples.")

    # Final checks and summary
    assigned_count_total = len(assigned_indices_global)
    unassigned_in_pool = num_seller_pool - assigned_count_total
    logging.info(f"Step 6: Data assignment to sellers complete. Total unique samples assigned: {assigned_count_total}")

    if unassigned_in_pool > 0:
        logging.info(f"{unassigned_in_pool} samples remain unassigned in the seller pool.")
    elif unassigned_in_pool < 0:
        # This indicates a major logic error if indices were assigned multiple times
        logging.error(
            f"Error! More samples assigned ({assigned_count_total}) than available in seller pool ({num_seller_pool}). Check assignment logic and duplicate handling.")

    logging.info("--- MartFL Discovery Split for Text Data Finished ---")
    return buyer_indices, seller_splits
