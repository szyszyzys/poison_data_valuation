import os
import random
import logging
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS

def load_dataset(dataset_name, data_root, seed):
    # ── reproducibility ─────────────────────────────────────────────────
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

    train_iter = test_iter = None
    num_classes = None
    class_names = None
    label_offset = 0

    if dataset_name == "AG_NEWS":
        # new torchtext API returns DataPipes
        train_dp, test_dp = AG_NEWS(root=data_root, split=('train','test'))
        logging.info("Loaded AG_NEWS via torchtext.datasets.AG_NEWS")

        # **wrap** into iterator
        train_iter = iter(train_dp)
        test_iter  = iter(test_dp)

        num_classes  = 4
        class_names  = ['World','Sports','Business','Sci/Tech']
        label_offset = 1  # labels 1–4

    elif dataset_name == "TREC":
        # use HuggingFace Datasets for TREC
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            raise ImportError("Please install the `datasets` library: pip install datasets")

        ds = hf_load("trec", "default", cache_dir=data_root)
        train_ds, test_ds = ds["train"], ds["test"]
        logging.info("Loaded TREC via HuggingFace `datasets`")

        # make Python generators (these *are* iterators)
        train_iter = ((ex["label-coarse"], ex["text"]) for ex in train_ds)
        test_iter  = ((ex["label-coarse"], ex["text"]) for ex in test_ds)

        num_classes  = 6
        class_names  = ['Abbreviation','Entity','Description','Human','Location','Numeric']
        label_offset = 0

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_iter, test_iter, num_classes, class_names, label_offset


def test_load():
    for name in ("AG_NEWS","TREC"):
        print(f"\n=== Testing {name} ===")
        train_it, test_it, nc, cn, off = load_dataset(
            dataset_name=name,
            data_root=".data",
            seed=42
        )
        # now train_it is a true iterator, so this works:
        label, text = next(train_it)
        print(f"{name}: num_classes={nc}, offset={off}")
        print(" sample label:", label)
        print(" sample text:", repr(text[:80] + ("…" if len(text)>80 else "")))

if __name__ == "__main__":
    test_load()