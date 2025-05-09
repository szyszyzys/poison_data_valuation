import os
import random
import logging
from typing import Tuple, Optional, Dict, List, Any, Generator, Callable
import hashlib
import pickle
import collections # For Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# --- TorchText specific imports for version 0.6.0 compatibility ---
import torchtext
if torchtext.__version__ != "0.6.0":
    logging.warning(
        f"This code is specifically tailored for torchtext 0.6.0. "
        f"You are using version {torchtext.__version__}. "
        f"Compatibility issues may arise. Consider updating the code or torchtext."
    )
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab # Explicit import for 0.6.0 style

# --- Placeholder/Assumed imports (these would be in your project) ---
# from your_project.splitting_utils import split_dataset_discovery, generate_buyer_bias_distribution
# from your_project.utils import print_and_save_data_statistics, collate_batch_new

# --- Dummy implementations for missing functions for the sake of running the example ---
def collate_batch_new(batch, pad_idx):
    labels, texts = zip(*batch)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Handle empty batch or batch with no text data
    if not texts or all(not t for t in texts):
        # Return labels and an empty tensor for texts if appropriate
        # Or raise an error if downstream code cannot handle this.
        logging.warning("collate_batch_new received an empty text batch or all texts are empty.")
        return labels_tensor, torch.empty((len(texts), 0), dtype=torch.long)

    max_len = 0
    # Ensure texts is not empty and contains actual sequences before calculating max_len
    valid_texts = [t for t in texts if t] # Filter out None or empty sequences
    if valid_texts:
        max_len = max(len(t) for t in valid_texts)

    if max_len == 0 and texts: # All texts were empty or None, but there were label entries
        return labels_tensor, torch.empty((len(texts), 0), dtype=torch.long)


    padded_texts = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(texts):
        if t: # Ensure text sequence t is not empty
            padded_texts[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        # else: it remains padded (already handled by torch.full)
    return labels_tensor, padded_texts


def generate_buyer_bias_distribution(num_classes, bias_type, alpha):
    logging.info(f"Generating dummy buyer bias: num_classes={num_classes}, type={bias_type}, alpha={alpha}")
    if num_classes <= 0: # Handle case with no classes
        logging.warning("num_classes is 0 or negative, cannot generate bias distribution.")
        return np.array([])
    if bias_type == "dirichlet":
        dist = np.random.dirichlet([alpha] * num_classes)
    else: # unbiased / random
        dist = np.ones(num_classes) / num_classes
    return dist

def split_dataset_discovery(dataset, buyer_count, num_clients, noise_factor, buyer_data_mode, buyer_bias_distribution):
    logging.info(f"Running dummy discovery split: buyer_count={buyer_count}, num_clients={num_clients}, noise={noise_factor}, mode={buyer_data_mode}")
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    actual_buyer_count = min(buyer_count, len(all_indices))
    buyer_indices_list = all_indices[:actual_buyer_count]
    remaining_indices = all_indices[actual_buyer_count:]

    buyer_indices_np = np.array(buyer_indices_list, dtype=int)

    seller_splits = {}
    if num_clients > 0 and remaining_indices:
        samples_per_seller = len(remaining_indices) // num_clients
        # If samples_per_seller is 0 but there are remaining_indices and clients, distribute one by one
        if samples_per_seller == 0 and len(remaining_indices) > 0:
            for i in range(min(num_clients, len(remaining_indices))):
                seller_splits[i] = [remaining_indices[i]]
        else: # Regular distribution
            for i in range(num_clients):
                start = i * samples_per_seller
                if start >= len(remaining_indices): # No more samples to assign
                    break
                end = (i + 1) * samples_per_seller if i < num_clients - 1 else len(remaining_indices)
                if start < end:
                    seller_splits[i] = remaining_indices[start:end]
    elif num_clients > 0 and not remaining_indices:
        logging.info("No remaining indices for sellers after buyer allocation.")


    logging.info(f"Dummy split: Buyer gets {len(buyer_indices_np)}, {len(seller_splits)} sellers with data.")
    return buyer_indices_np, seller_splits


def print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results, output_dir):
    buyer_n = len(buyer_indices) if buyer_indices is not None else 0
    logging.info(f"Dummy stats: Saving to {output_dir}, buyer_n={buyer_n}, sellers_n={len(seller_splits)}")
    if save_results and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return {"info": "dummy stats"}
# --- End Dummy implementations ---

try:
    from datasets import load_dataset as hf_load
    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

def get_cache_path(cache_dir: str, prefix: str, params: Tuple) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    param_string = "_".join(map(str, params)).replace("/", "_") # Sanitize path separators
    key = hashlib.md5(param_string.encode()).hexdigest()
    return os.path.join(cache_dir, f"{prefix}_{key}.cache")

def get_text_data_set(
        dataset_name: str,
        buyer_percentage: float = 0.01,
        num_sellers: int = 10,
        batch_size: int = 64,
        data_root: str = "./data",
        split_method: str = "discovery",
        n_adversaries: int = 0, # Unused in current logic, but kept for signature
        save_path: str = './result',
        # --- Discovery Split Specific Params ---
        discovery_quality: float = 0.3,
        buyer_data_mode: str = "unbiased",
        buyer_bias_type: str = "dirichlet",
        buyer_dirichlet_alpha: float = 0.3,
        discovery_client_data_count: int = 0, # Unused, but kept
        # --- Other Split Method Params ---
        seller_dirichlet_alpha: float = 0.7, # Unused, but kept
        seed: int = 42,
        # --- Caching control ---
        use_cache: bool = True,
        # --- Vocab params ---
        min_freq: int = 1,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
) -> Tuple[Optional[DataLoader], Dict[int, Optional[DataLoader]], Optional[DataLoader], List[str], Vocab, int]:

    if not (0.0 <= buyer_percentage <= 1.0):
        raise ValueError("buyer_percentage must be between 0 and 1.")
    if num_sellers < 0:
        raise ValueError("num_sellers must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if buyer_bias_type == "dirichlet" and buyer_dirichlet_alpha <= 0 :
        raise ValueError("buyer_dirichlet_alpha must be positive for dirichlet bias.")
    if min_freq <=0:
        raise ValueError("min_freq for vocabulary must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info(f"CUDA available, setting CUDA seeds to {seed}.")
    else:
        logging.info("CUDA not available.")

    app_cache_dir = os.path.join(data_root, ".cache", "get_text_data_set_cache_tt060") # Specific cache for this version
    os.makedirs(app_cache_dir, exist_ok=True)
    logging.info(f"Using cache directory: {app_cache_dir}")

    tokenizer = get_tokenizer('basic_english')
    logging.info("Using 'basic_english' tokenizer.")

    if dataset_name == "AG_NEWS":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for AG_NEWS but not installed.")
        logging.info(f"Loading AG_NEWS dataset (raw data cache_dir: {data_root})...")
        ds = hf_load("ag_news", cache_dir=data_root)
        train_ds_hf = ds["train"]
        test_ds_hf = ds["test"]
        num_classes = 4
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        label_offset = 0
        text_field, label_field = "text", "label"
    elif dataset_name == "TREC":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for TREC but not installed.")
        logging.info(f"Loading TREC dataset (raw data cache_dir: {data_root})...")
        ds = hf_load("trec", cache_dir=data_root)
        train_ds_hf = ds["train"]
        test_ds_hf = ds["test"]
        num_classes = 6
        class_names = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        label_offset = 0
        text_field, label_field = "text", "coarse_label"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj: # dataset_obj is a HuggingFace Dataset object
            text_content = ex.get(text_fld)
            if label_fld:
                label_content = ex.get(label_fld)
                if isinstance(text_content, str) and label_content is not None: # Ensure label exists
                    yield (label_content, text_content)
            else: # For vocab building (text only)
                if isinstance(text_content, str):
                    yield text_content

    # ─── 2. Build or Load Vocabulary from Cache (MODIFIED FOR TORCHTEXT 0.6.0) ───
    vocab_cache_params = (dataset_name, min_freq, unk_token, pad_token, "torchtext_0.6.0")
    vocab_cache_file = get_cache_path(app_cache_dir, "vocab", vocab_cache_params)
    vocab: Optional[Vocab] = None
    pad_idx = -1
    unk_idx_val = -1

    if use_cache and os.path.exists(vocab_cache_file):
        try:
            logging.info(f"Attempting to load vocabulary from cache: {vocab_cache_file}")
            # --- MODIFICATION HERE ---
            # For PyTorch versions that default weights_only=True (e.g., >=2.1 or a future version)
            # and you are loading non-weights objects like a Vocab object.
            try:
                # Attempt with weights_only=False if an older PyTorch version doesn't have it or
                # if you know the file is safe and contains pickled Python objects.
                cached_data = torch.load(vocab_cache_file, weights_only=False)
            except TypeError as te: # Older PyTorch versions might not have weights_only argument
                if "weights_only" in str(te):
                    logging.warning(f"torch.load does not support 'weights_only' argument in this PyTorch version ({torch.__version__}). Loading normally.")
                    cached_data = torch.load(vocab_cache_file)
                else:
                    raise # Re-raise other TypeErrors
            # --- END MODIFICATION ---

            if isinstance(cached_data, tuple) and len(cached_data) == 3:
                vocab, pad_idx, unk_idx_val = cached_data
            elif isinstance(cached_data, Vocab): # Backward compatibility for older cache
                vocab = cached_data
                if pad_token in vocab.stoi: pad_idx = vocab.stoi[pad_token]
                else: logging.warning(f"'{pad_token}' not found in loaded vocab.stoi.")
                if unk_token in vocab.stoi: unk_idx_val = vocab.stoi[unk_token]
                elif '<unk>' in vocab.stoi: unk_idx_val = vocab.stoi['<unk>']
                else: logging.warning(f"'{unk_token}' or '<unk>' not found in loaded vocab.stoi.")
            else:
                raise TypeError("Cached vocab data has unexpected format.")

            if not isinstance(vocab, Vocab) or not (isinstance(pad_idx, int) and pad_idx >=0) or not (isinstance(unk_idx_val, int) and unk_idx_val >=0):
                logging.warning(f"Problematic cached vocab/indices: pad_idx={pad_idx}, unk_idx_val={unk_idx_val}. Rebuilding.")
                vocab = None
            else:
                logging.info(f"Vocabulary loaded from cache. Size: {len(vocab.itos)}, Pad index: {pad_idx}, Unk index: {unk_idx_val}")
        except Exception as e:
            logging.warning(f"Failed to load vocab from cache ({vocab_cache_file}): {e}. Rebuilding.")
            vocab = None

    if vocab is None:
        logging.info(f"Building vocabulary for torchtext 0.6.0 with min_freq={min_freq}...")

        def yield_tokens_for_vocab_0_6(text_iterator_func: Callable[[], Generator[str, None, None]]):
            for text_sample in text_iterator_func():
                yield tokenizer(text_sample)

        token_counter = collections.Counter()
        logging.info("Counting token frequencies from dataset...")
        num_docs_processed_for_vocab = 0
        for tokens_list in yield_tokens_for_vocab_0_6(lambda: hf_iterator(train_ds_hf, text_field)):
            token_counter.update(tokens_list)
            num_docs_processed_for_vocab +=1
            if num_docs_processed_for_vocab % 20000 == 0: # Log less frequently for large datasets
                logging.info(f"Processed {num_docs_processed_for_vocab} documents for vocab frequency counting...")
        logging.info(f"Finished counting token frequencies from {num_docs_processed_for_vocab} documents. Total unique tokens before min_freq: {len(token_counter)}")

        vocab = Vocab(
            counter=token_counter,
            min_freq=min_freq,
            specials=[unk_token, pad_token]
        )

        if unk_token in vocab.stoi:
            unk_idx_val = vocab.stoi[unk_token]
        elif '<unk>' in vocab.stoi: # Fallback if custom unk_token wasn't found but default was created
            unk_idx_val = vocab.stoi['<unk>']
            logging.warning(f"Custom unk_token '{unk_token}' not found, using default '<unk>' at index {unk_idx_val}.")
        else:
            raise RuntimeError(f"Neither '{unk_token}' nor '<unk>' found in vocab.stoi after building. OOV handling will fail.")

        if pad_token in vocab.stoi:
            pad_idx = vocab.stoi[pad_token]
        else:
            raise RuntimeError(f"'{pad_token}' not found in vocab.stoi after building. Padding will fail.")

        logging.info(f"Vocabulary built. Size: {len(vocab.itos)}. UNK index: {unk_idx_val}, PAD index: {pad_idx}.")
        if len(vocab.itos) < 20: # If vocab is small, print more of it
            logging.info(f"Vocab itos (first 20 or all): {vocab.itos[:20]}")
        else:
            logging.info(f"Top 10 most frequent tokens (after min_freq): {vocab.itos[:10]}")

        if use_cache:
            try:
                torch.save((vocab, pad_idx, unk_idx_val), vocab_cache_file)
                logging.info(f"Vocabulary (and indices) saved to cache: {vocab_cache_file}")
            except Exception as e:
                logging.error(f"Failed to save vocabulary to cache: {e}")

    # Ensure vocab and indices are valid before proceeding
    if vocab is None: raise RuntimeError("Vocabulary is None after build/load attempt.")
    if not (isinstance(pad_idx, int) and pad_idx >= 0): raise RuntimeError(f"pad_idx ({pad_idx}) is invalid.")
    if not (isinstance(unk_idx_val, int) and unk_idx_val >= 0): raise RuntimeError(f"unk_idx_val ({unk_idx_val}) is invalid.")


    # --- Text pipeline for torchtext 0.6.0 ---
    def text_pipeline_0_6(text_string: str, local_tokenizer: Callable, local_vocab: Vocab, local_unk_idx: int) -> List[int]:
        tokens = local_tokenizer(text_string)
        # In torchtext 0.6.0, vocab[token] should handle OOV by returning unk_idx if unk_token was in specials.
        # Using .get for explicit fallback is safer.
        return [local_vocab.stoi.get(token, local_unk_idx) for token in tokens]

    # ─── 3. Numericalize or Load Numericalized Data from Cache ──────────
    numericalized_cache_key_base = (dataset_name, vocab_cache_file) # vocab_cache_file links to specific vocab build

    def numericalize_dataset(data_iterator_func: Callable[[], Generator[Tuple[int, str], None, None]],
                             split_name: str) -> List[Tuple[int, List[int]]]:
        numericalized_cache_params = numericalized_cache_key_base + (split_name,)
        numericalized_cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", numericalized_cache_params)

        if use_cache and os.path.exists(numericalized_cache_path):
            try:
                logging.info(f"Attempting to load numericalized {split_name} data from {numericalized_cache_path}")
                with open(numericalized_cache_path, "rb") as f:
                    processed_data = pickle.load(f)
                logging.info(f"Numericalized {split_name} data loaded from cache. Samples: {len(processed_data)}")
                return processed_data
            except Exception as e:
                logging.warning(f"Failed to load numericalized {split_name} data from cache: {e}. Re-numericalizing.")

        logging.info(f"Processing and numericalizing {split_name} data (torchtext 0.6.0 method)...")
        processed_data_list = []
        processed_count = 0
        skipped_count = 0

        for item_idx, item_content in enumerate(data_iterator_func()):
            try:
                lbl, txt = item_content
                # Using the correct pipeline with necessary arguments
                ids = text_pipeline_0_6(txt, tokenizer, vocab, unk_idx_val)

                if ids:
                    processed_data_list.append((lbl - label_offset, ids))
                    processed_count += 1
                else: # Text resulted in empty token list after numericalization (e.g. only OOV and no <unk> handling for empty result)
                    # Or tokenizer returned empty list for the text
                    # logging.debug(f"Skipping item {item_idx} in {split_name} as it resulted in empty IDs. Original text: '{txt[:50]}...'")
                    skipped_count += 1
            except Exception as e:
                text_snippet = str(item_content[1])[:70] + "..." if isinstance(item_content, tuple) and len(item_content) > 1 else str(item_content)[:70]
                logging.warning(f"Error processing {split_name} item #{item_idx} (content: '{text_snippet}'). Error: {e}. Skipping.")
                skipped_count += 1
        logging.info(f"Finished numericalizing {split_name} data. Processed: {processed_count}, Skipped: {skipped_count}")

        if use_cache and processed_data_list:
            try:
                with open(numericalized_cache_path, "wb") as f:
                    pickle.dump(processed_data_list, f)
                logging.info(f"Numericalized {split_name} data saved to cache: {numericalized_cache_path}")
            except Exception as e:
                logging.error(f"Failed to save numericalized {split_name} data to cache: {e}")
        return processed_data_list

    processed_train_data = numericalize_dataset(
        lambda: hf_iterator(train_ds_hf, text_field, label_field), "train"
    )
    processed_test_data = numericalize_dataset(
        lambda: hf_iterator(test_ds_hf, text_field, label_field), "test"
    )

    if not processed_train_data:
        raise ValueError("Processed training dataset is empty. Check data, vocab, or numericalization logic.")

    # ─── 4. Split Data or Load Split Indices from Cache ───────────────────
    split_params_tuple_elements = [
        dataset_name, "train_splits_tt060", vocab_cache_file, seed,
        buyer_percentage, num_sellers, split_method
    ]
    if split_method == "discovery":
        split_params_tuple_elements.extend([
            discovery_quality, buyer_data_mode, buyer_bias_type, buyer_dirichlet_alpha
        ])
    split_params_tuple = tuple(split_params_tuple_elements)
    split_indices_cache_file = get_cache_path(app_cache_dir, "split_indices", split_params_tuple)

    buyer_indices_np: Optional[np.ndarray] = None
    seller_splits: Dict[int, List[int]] = {}

    if use_cache and os.path.exists(split_indices_cache_file):
        try:
            logging.info(f"Attempting to load split indices from {split_indices_cache_file}")
            with open(split_indices_cache_file, "rb") as f:
                buyer_indices_np, seller_splits = pickle.load(f)
            if not isinstance(buyer_indices_np, (np.ndarray, type(None))) or not isinstance(seller_splits, dict):
                 raise TypeError("Cached split indices have incorrect type.")
            loaded_buyer_len = len(buyer_indices_np) if buyer_indices_np is not None else 0
            logging.info(f"Split indices loaded from cache. Buyer samples: {loaded_buyer_len}")
        except Exception as e:
            logging.warning(f"Failed to load split indices from cache: {e}. Re-splitting.")
            buyer_indices_np, seller_splits = None, {}

    needs_resplit = buyer_indices_np is None
    if not needs_resplit and num_sellers > 0 and not seller_splits and buyer_percentage < 1.0 : # If sellers expected but no splits
        # (and buyer doesn't take all data)
        logging.info("Seller splits not found or empty in cache with num_sellers > 0. Forcing re-split.")
        needs_resplit = True

    if needs_resplit:
        logging.info(f"Splitting data using method: '{split_method}'")
        total_samples = len(processed_train_data)
        buyer_count = min(int(total_samples * buyer_percentage), total_samples)
        logging.info(f"Total train samples available for splitting: {total_samples}")
        logging.info(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

        if split_method == "discovery":
            buyer_biased_distribution = generate_buyer_bias_distribution(
                num_classes=num_classes,
                bias_type=buyer_bias_type,
                alpha=buyer_dirichlet_alpha
            )
            current_buyer_indices_np, current_seller_splits = split_dataset_discovery(
                dataset=processed_train_data,
                buyer_count=buyer_count,
                num_clients=num_sellers,
                noise_factor=discovery_quality,
                buyer_data_mode=buyer_data_mode,
                buyer_bias_distribution=buyer_biased_distribution
            )
        else:
            raise ValueError(f"Unsupported split_method: '{split_method}'.")

        buyer_indices_np = current_buyer_indices_np
        seller_splits = current_seller_splits

        if use_cache:
            try:
                with open(split_indices_cache_file, "wb") as f:
                    pickle.dump((buyer_indices_np, seller_splits), f)
                logging.info(f"Split indices saved to cache: {split_indices_cache_file}")
            except Exception as e:
                logging.error(f"Failed to save split indices to cache: {e}")

    # Sanity Checks for Splits
    assigned_indices = set(buyer_indices_np.tolist() if buyer_indices_np is not None else [])
    total_seller_samples_assigned = 0
    valid_seller_splits: Dict[int, List[int]] = {}
    for seller_id, indices_list in seller_splits.items():
        if indices_list is None or not isinstance(indices_list, (list, np.ndarray)) or len(indices_list) == 0:
            continue
        indices_set = set(indices_list)
        if buyer_indices_np is not None and not assigned_indices.isdisjoint(indices_set):
            logging.error(f"OVERLAP: Buyer indices and Seller {seller_id} indices overlap!")
        assigned_indices.update(indices_set)
        total_seller_samples_assigned += len(indices_list)
        valid_seller_splits[seller_id] = indices_list
    seller_splits = valid_seller_splits

    buyer_len = len(buyer_indices_np) if buyer_indices_np is not None else 0
    logging.info(
        f"Splitting complete. Buyer samples: {buyer_len}, "
        f"Total seller samples assigned: {total_seller_samples_assigned} across {len(seller_splits)} sellers."
    )
    unassigned_count = len(processed_train_data) - len(assigned_indices)
    if unassigned_count > 0:
        logging.warning(f"{unassigned_count} training samples were not assigned.")
    elif unassigned_count < 0:
        logging.error(f"Index accounting error: {abs(unassigned_count)} MORE indices assigned than available.")

    # ─── 5. Create DataLoaders ───────────────────────────────────────────
    logging.info("Creating DataLoaders...")
    collate_fn_to_use = lambda batch: collate_batch_new(batch, pad_idx)

    buyer_loader: Optional[DataLoader] = None
    if buyer_indices_np is not None and len(buyer_indices_np) > 0:
        buyer_subset = Subset(processed_train_data, buyer_indices_np.tolist())
        buyer_loader = DataLoader(buyer_subset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn_to_use, drop_last=False)
        logging.info(f"Buyer DataLoader created with {len(buyer_indices_np)} samples.")
    else:
        logging.info("Buyer has no data samples assigned. Buyer DataLoader will be None.")

    seller_loaders: Dict[int, Optional[DataLoader]] = {}
    actual_sellers_with_data = 0
    for i in range(num_sellers):
        indices = seller_splits.get(i)
        if indices and len(indices) > 0 : # Ensure indices is not None and not empty
            try:
                seller_subset = Subset(processed_train_data, list(indices)) # Subset expects list
                seller_loaders[i] = DataLoader(seller_subset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn_to_use, drop_last=False)
                actual_sellers_with_data += 1
            except Exception as e:
                logging.error(f"Failed to create DataLoader for seller {i}: {e}. Setting to None.")
                seller_loaders[i] = None
        else:
            seller_loaders[i] = None # No data for this seller
    logging.info(
        f"Seller DataLoaders created. {actual_sellers_with_data}/{num_sellers} sellers have data. "
        f"Total samples in seller loaders: {total_seller_samples_assigned}"
    )

    test_loader: Optional[DataLoader] = None
    if processed_test_data:
        test_loader = DataLoader(processed_test_data, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_to_use)
        logging.info(f"Test DataLoader created with {len(processed_test_data)} samples.")
    else:
        logging.info("Processed test set is empty. Test DataLoader will be None.")

    logging.info("Text data loading, processing, splitting, and DataLoader creation complete.")

    if save_path: # Ensure save_path exists for stats
        os.makedirs(save_path, exist_ok=True)
        data_distribution_info = print_and_save_data_statistics(
            dataset=processed_train_data,
            buyer_indices=buyer_indices_np,
            seller_splits=seller_splits,
            save_results=True,
            output_dir=save_path
        )
        logging.info(f"Data statistics processed. Info: {data_distribution_info}")

    return buyer_loader, seller_loaders, test_loader, class_names, vocab, pad_idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check torchtext version at runtime
    if torchtext.__version__ != "0.6.0":
        print(f"WARNING: This example is primarily for torchtext 0.6.0. Your version is {torchtext.__version__}.")
        print("         You might need to adjust the code if using a significantly different version.")

    print("\n--- Running AG_NEWS (torchtext 0.6.0 compatible) ---")
    try:
        # Using a higher min_freq to avoid very long vocab building times
        b_loader, s_loaders, t_loader, c_names, voc, p_idx = get_text_data_set(
            dataset_name="TREC",
            buyer_percentage=0.02,
            num_sellers=2,
            batch_size=16,
            data_root="./test_data_agnews_tt060", # Use a distinct data root for this test
            save_path='./test_results_agnews_tt060',
            seed=42,
            use_cache=True, # Test caching
            min_freq=5 # Crucial for performance
        )
        logging.info(f"AG_NEWS (tt0.6.0): Buyer DL: {b_loader is not None}, Seller DLs: {len(s_loaders)}, Test DL: {t_loader is not None}")
        if b_loader:
            for lbls, txts in b_loader:
                logging.info(f"AG_NEWS Buyer batch (tt0.6.0): labels shape {lbls.shape}, texts shape {txts.shape}")
                break

        print("\n--- Running AG_NEWS again (should use cache) ---")
        get_text_data_set(
            dataset_name="AG_NEWS",
            buyer_percentage=0.02,
            num_sellers=2,
            batch_size=16,
            data_root="./test_data_agnews_tt060",
            save_path='./test_results_agnews_tt060',
            seed=42,
            use_cache=True,
            min_freq=5 # Same params to hit cache
        )

    except ImportError as e:
        logging.error(f"AG_NEWS example failed due to ImportError: {e}. Is 'datasets' installed?")
    except Exception as e:
        logging.error(f"AG_NEWS example failed: {e}", exc_info=True)


    if hf_datasets_available:
        print("\n--- Running TREC (torchtext 0.6.0 compatible) ---")
        try:
            get_text_data_set(
                dataset_name="TREC",
                buyer_percentage=0.05,
                num_sellers=1,
                batch_size=8,
                data_root="./test_data_trec_tt060",
                save_path='./test_results_trec_tt060',
                seed=77,
                use_cache=True,
                min_freq=3
            )
        except Exception as e:
            logging.error(f"TREC example failed: {e}", exc_info=True)
    else:
        print("\nSkipping TREC example as HuggingFace 'datasets' is not available.")