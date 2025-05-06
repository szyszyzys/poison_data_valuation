import logging
import os
from typing import List, Dict
# Import Optional for type hinting
from typing import Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

from marketplace.utils.gradient_market_utils.data_processor import split_dataset_discovery, \
    print_and_save_data_statistics

# Make sure necessary torchtext components are imported

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


# --- End Text Data Helper Functions ---


def get_text_data_set(
        dataset_name: str,
        buyer_percentage: float = 0.01,
        num_sellers: int = 10,
        batch_size: int = 64,
        data_root="./data",
        split_method: str = "discovery",
        n_adversaries: int = 0,
        save_path: str = './result', # Path might be needed for stats saving later
        # --- Discovery Split Specific Params ---
        discovery_quality: float = 0.3,
        buyer_data_mode: str = "unbiased",
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
    # ── reproducibility ────────────────────────────────────────
    # ── Seed setting ───────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For reproducibility, you might also consider these, but they can affect performance:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.cuda.manual_seed(seed) # Often sufficient if using single GPU
        logging.info(f"CUDA available, setting CUDA seeds to {seed}.")
    else:
        logging.info("CUDA not available.")

    # ── constants & tokenizer ───────────────────────────────────
    unk_token, pad_token = "<unk>", "<pad>"
    min_freq = 1  # Define min_freq for vocab building consistently
    os.makedirs(data_root, exist_ok=True)
    tokenizer = get_tokenizer('basic_english')
    logging.info("Using 'basic_english' tokenizer.")

    # --- Data Loading & Iterator Setup ---
    # We need distinct iterables/iterators for vocab building vs data processing,
    # as iterating consumes the source.
    try:
        from datasets import load_dataset as hf_load
        hf_datasets_available = True
    except ImportError:
        hf_datasets_available = False
        logging.warning("HuggingFace 'datasets' library not found. TREC dataset loading will fail.")

    if dataset_name == "AG_NEWS":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for AG_NEWS but not installed.")
        logging.info(f"Loading AG_NEWS dataset using HuggingFace datasets from {data_root}...")
        # Load using HuggingFace datasets
        ds = hf_load("ag_news", cache_dir=data_root)
        train_ds = ds["train"]  # Dataset object
        test_ds = ds["test"]  # Dataset object

        # --- Inspection (Optional but recommended for first run) ---
        logging.info("Inspecting first few items from AG_NEWS train_ds (HuggingFace):")
        count = 0
        for ex in train_ds:
            print(f"Item {count}: type={type(ex)}, keys={ex.keys() if isinstance(ex, dict) else 'N/A'}")
            if isinstance(ex, dict) and "text" in ex:
                print(f"  - Text part type: {type(ex['text'])}, content={ex['text'][:100]}...")  # Should be string
            if isinstance(ex, dict) and "label" in ex:
                print(f"  - Label part type: {type(ex['label'])}")  # Should be int
            count += 1
            if count >= 3: break
        # --- End Inspection ---

        # For Vocabulary: Generator yielding only text from training set
        # HF datasets usually yield dicts; access the 'text' field. Should be STRING here.
        vocab_source_iter = (ex["text"] for ex in train_ds if isinstance(ex.get("text"), str))

        # For Processing: Generators yielding (label, text) tuples
        # AG_NEWS labels from HF start from 0
        train_iter = ((ex["label"], ex["text"]) for ex in train_ds if isinstance(ex.get("text"), str))
        test_iter = ((ex["label"], ex["text"]) for ex in test_ds if isinstance(ex.get("text"), str))

        num_classes = 4
        # Class names matching HF ag_news labels {0: World, 1: Sports, 2: Business, 3: Sci/Tech}
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        label_offset = 0  # Labels start at 0 with HF dataset
        logging.info("AG_NEWS dataset loaded via HuggingFace datasets.")
    elif dataset_name == "TREC":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for TREC dataset but not installed.")
        logging.info(f"Loading TREC dataset using HuggingFace datasets from {data_root}...")
        # Load HuggingFace dataset object
        ds = hf_load("trec", "default", cache_dir=data_root)
        train_ds = ds["train"]
        test_ds = ds["test"]  # <<< Define test_ds
        logging.info("Inspecting first few items from TREC train_ds:")
        count = 0
        for ex in train_ds:
            print(f"Item {count}: type={type(ex)}, keys={ex.keys() if isinstance(ex, dict) else 'N/A'}")
            if isinstance(ex, dict) and "text" in ex:
                print(f"  - Text part type: {type(ex['text'])}, content={ex['text'][:100]}...")
            count += 1
            if count >= 3: break

        vocab_source_iter = (ex["text"] for ex in train_ds)

        # For Processing: Generators yielding (label, text) tuples
        train_iter = ((ex["coarse_label"], ex["text"]) for ex in train_ds)
        test_iter = ((ex["coarse_label"], ex["text"]) for ex in test_ds)  # <<< Use test_ds here

        num_classes = 6  # Based on coarse_label
        class_names = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        label_offset = 0  # TREC labels start at 0
        logging.info("TREC dataset loaded.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # ── build vocab ─────────────────────────────────────────────
    def yield_tokens(text_iterator):
        """Yields token lists from an iterator of text strings."""
        processed_count = 0
        for text in text_iterator:
            if isinstance(text, str):
                yield tokenizer(text)
                processed_count += 1
            else:
                logging.warning(f"yield_tokens expected a string, but got {type(text)}. Skipping item.")
        logging.info(f"yield_tokens processed {processed_count} text items for vocabulary.")

    specials_list_defined = [unk_token, pad_token]
    # min_freq cannot be applied with this function in this version
    # min_freq_for_vocab = 1

    logging.info("Building vocabulary using build_vocab_from_iterator...")
    # Based on errors, this function likely returns a fully formed Vocab object
    # but doesn't accept configuration arguments like min_freq or specials.

    try:
        token_iterator_for_builder = yield_tokens(vocab_source_iter)

        # Directly get the Vocab object from this function
        vocab = build_vocab_from_iterator(
            token_iterator_for_builder
            # NO ARGUMENTS like min_freq or specials here
        )

    except Exception as e:
        logging.error(f"Error during build_vocab_from_iterator: {e}")
        raise  # Re-raise after logging

    # --- Check the result ---
    if not vocab:
        raise ValueError("Vocabulary building failed (returned None or empty).")
    if not isinstance(vocab, Vocab):
        # This check is important given the confusion
        raise TypeError(f"Expected build_vocab_from_iterator to return a Vocab object, but got {type(vocab)}")

    # --- Vocab object created directly by build_vocab_from_iterator ---
    logging.info("Vocabulary object created directly by build_vocab_from_iterator.")

    # --- Now, work with the 'vocab' object obtained above ---

    # Check if specials were included somehow (e.g., by default)
    if unk_token in vocab:
        vocab.set_default_index(vocab[unk_token])
        unk_idx = vocab[unk_token]
        logging.info(f"Special token '{unk_token}' found. Set default index to {unk_idx}.")
    else:
        # If not found, you CANNOT set the default index this way.
        # You might need to handle unknown tokens differently or accept vocab limitations.
        vocab.set_default_index(-1)  # A fallback, but indicates <unk> wasn't handled as expected.
        logging.warning(
            f"Special token '{unk_token}' NOT found in vocabulary created by build_vocab_from_iterator. Default index set to -1.")

    if pad_token in vocab:
        pad_idx = vocab[pad_token]
        logging.info(f"Special token '{pad_token}' found with index {pad_idx}.")
    else:
        pad_idx = -1  # Fallback
        # Padding might be essential, so log an error or raise one
        logging.error(f"Critical: Special token '{pad_token}' NOT found in vocabulary!")

    logging.info(
        f"Built final vocab (size={len(vocab)}). Note: min_freq/specials config might be limited by torchtext version.")    # ── text → index pipeline ───────────────────────────────────
    # The Vocab object is callable for string-to-index lookup
    # Ensure tokenizer returns a list of tokens suitable for vocab lookup
    text_pipeline = lambda text_string: vocab(tokenizer(text_string))

    # ── numericalize & skip empty ──────────────────────────────
    # Use the separate train_iter and test_iter obtained earlier

    logging.info("Processing and numericalizing train data...")
    processed_train_data = []
    train_processed_count = 0
    train_skipped_count = 0
    for item in train_iter:  # Use the iterator dedicated to training data processing
        try:
            # Expecting (label, text) format from iterators defined above
            lbl, txt = item
            ids = text_pipeline(txt)
            if ids:  # Check if the result is not empty
                processed_train_data.append((lbl - label_offset, ids))  # Apply label offset
                train_processed_count += 1
            else:
                train_skipped_count += 1
                # logging.debug(f"Skipping empty text after processing train item: {txt[:50]}...") # Optional debug log
        except Exception as e:
            logging.warning(f"Error processing train item: {item}. Error: {e}. Skipping item.")
            train_skipped_count += 1
    logging.info(f"Finished processing train data. Processed: {train_processed_count}, Skipped: {train_skipped_count}")

    logging.info("Processing and numericalizing test data...")
    processed_test_data = []
    test_processed_count = 0
    test_skipped_count = 0
    for item in test_iter:  # Use the iterator dedicated to test data processing
        try:
            # Expecting (label, text) format
            lbl, txt = item
            ids = text_pipeline(txt)
            if ids:
                processed_test_data.append((lbl - label_offset, ids))  # Apply label offset
                test_processed_count += 1
            else:
                test_skipped_count += 1
                # logging.debug(f"Skipping empty text after processing test item: {txt[:50]}...") # Optional debug log
        except Exception as e:
            logging.warning(f"Error processing test item: {item}. Error: {e}. Skipping item.")
            test_skipped_count += 1
    logging.info(f"Finished processing test data. Processed: {test_processed_count}, Skipped: {test_skipped_count}")

    logging.info(
        f"Total Processed Samples: train={len(processed_train_data)}, "
        f"test={len(processed_test_data)}"
    )

    # Assign to dataset variables (if needed for clarity or downstream use)
    dataset = processed_train_data
    test_set = processed_test_data

    if not dataset:
        # This check remains valid and important
        raise ValueError("Processed training dataset is empty after filtering. Check data or processing logic.")

    total_samples = len(dataset)
    buyer_count = min(int(total_samples * buyer_percentage), total_samples)
    logging.info(f"Total train samples available for splitting: {total_samples}")
    logging.info(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

    # --- Data Splitting ---
    # (splitting logic remains the same, using placeholder functions)
    buyer_indices: np.ndarray = np.array([], dtype=int)
    seller_splits: Dict[int, List[int]] = {}

    logging.info(f"Splitting data using method: '{split_method}'")
    if split_method == "discovery":
        print(f"Using 'discovery' split method with buyer bias type: '{buyer_bias_type}'")
        # Generate buyer distribution ONLY when needed
        buyer_biased_distribution = generate_buyer_bias_distribution(
            num_classes=num_classes,  # Use derived num_classes
            bias_type=buyer_bias_type,
            alpha=buyer_dirichlet_alpha  # Use argument for alpha
        )
        print(f"Generated buyer bias distribution: {buyer_biased_distribution}")

        buyer_indices, seller_splits = split_dataset_discovery(
            dataset=dataset,
            buyer_count=buyer_count,
            num_clients=num_sellers,
            noise_factor=discovery_quality,
            buyer_data_mode=buyer_data_mode,
            buyer_bias_distribution=buyer_biased_distribution  # Pass generated dist
        )

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
    data_distribution_info = print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results=True,
                                                            output_dir=save_path)
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
