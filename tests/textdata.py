import os
import random
import logging
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
# no more: Field / LabelField / BucketIterator / legacy

def load_dataset(dataset_name, data_root, seed):
    # ── reproducibility ─────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info("CUDA available, setting CUDA seed.")
    else:
        logging.info("CUDA not available.")

    os.makedirs(data_root, exist_ok=True)
    tokenizer = get_tokenizer('basic_english')

    # common return placeholders
    train_iter = test_iter = None
    num_classes = None
    class_names = None
    label_offset = 0

    if dataset_name == "AG_NEWS":
        # ---- new torchtext API ----
        train_iter, test_iter = AG_NEWS(root=data_root, split=('train','test'))
        logging.info("Loaded AG_NEWS via torchtext.datasets.AG_NEWS")

        num_classes  = 4
        class_names  = ['World','Sports','Business','Sci/Tech']
        label_offset = 1  # labels come 1–4

    elif dataset_name == "TREC":
        # ---- fallback to Hugging-Face Datasets for TREC ----
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("TREC is not in torchtext 0.17; please `pip install datasets` to use HuggingFace loader")

        ds = load_dataset("trec", "default", cache_dir=data_root)
        train_ds, test_ds = ds["train"], ds["test"]
        logging.info("Loaded TREC via HuggingFace `datasets`")

        # make simple Python generators that mimic torchtext's (label, text) pairs:
        train_iter = ((ex["label-coarse"], ex["text"]) for ex in train_ds)
        test_iter  = ((ex["label-coarse"], ex["text"]) for ex in test_ds)

        num_classes  = 6
        class_names  = ['Abbreviation','Entity','Description','Human','Location','Numeric']
        label_offset = 0

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_iter, test_iter, num_classes, class_names, label_offset


def test_load():
    for name in ("AG_NEWS", "TREC"):
        print(f"\n=== Testing {name} ===")
        train_it, test_it, nc, cn, off = load_dataset(
            dataset_name=name,
            data_root=".data",
            seed=42
        )
        # pull one example from train
        example = next(train_it)
        label, text = example
        print(f"{name}: num_classes={nc}, offset={off}")
        print(" sample ->", label, repr(text[:50] + ("…" if len(text)>50 else "")))

if __name__ == "__main__":
    test_load()