import logging
import os
import random

import numpy as np
import torch
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.data.utils import get_tokenizer


def load_dataset(dataset_name, data_root, seed, batch_size, device):
    # ── reproducibility ─────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info("CUDA available, setting CUDA seed.")
    else:
        logging.info("CUDA not available.")

    # ── prepare directory ────────────────────────────────────────────────────
    os.makedirs(data_root, exist_ok=True)

    # ── tokenizer + Fields ──────────────────────────────────────────────────
    tokenizer = get_tokenizer('basic_english')
    TEXT = Field(tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True)
    LABEL = LabelField(sequential=False)

    # placeholders
    train_iter = test_iter = None

    # ── AG_NEWS branch (new API) ──────────────────────────────────────────────
    if dataset_name == "AG_NEWS":
        from torchtext.datasets import AG_NEWS as AG_NEWS_Loader

        train_iter, test_iter = AG_NEWS_Loader(root=data_root, split=('train', 'test'))
        logging.info("Using newer torchtext.datasets.AG_NEWS API.")

        # build vocabs
        TEXT.build_vocab([])
        LABEL.build_vocab([])

        num_classes = 4
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        label_offset = 1

    # ── TREC branch (legacy API) ──────────────────────────────────────────────
    elif dataset_name == "TREC":
        from torchtext.datasets.trec import TREC as TREC_Loader

        # splits() returns Dataset objects, not raw iterators
        train_data, test_data = TREC_Loader.splits(
            TEXT, LABEL,
            root=data_root,
            fine_grained=False  # six-way questions; True for 50-way
        )
        logging.info("Using legacy torchtext.datasets.trec.TREC.splits API.")

        # build vocabs on train only
        TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)

        # bucketed iterators
        train_iter, test_iter = BucketIterator.splits(
            (train_data, test_data),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.text),
            device=device
        )

        num_classes = len(LABEL.vocab)
        class_names = LABEL.vocab.itos  # e.g. ['<unk>','Abbreviation',...]
        label_offset = 0

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_iter, test_iter, num_classes, class_names, label_offset


def test_load():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ["AG_NEWS", "TREC"]:
        print(f"\n--- Testing {name} ---")
        train_it, test_it, nc, cn, off = load_dataset(
            dataset_name=name,
            data_root=".data",
            seed=42,
            batch_size=16,
            device=device
        )
        # grab one batch
        batch = next(iter(train_it))
        print(f"{name}: num_classes={nc}, offset={off}")
        print(f"  batch.text shape: {batch.text[0].shape}, batch.label shape: {batch.label.shape}")
        print(f"  sample labels: {batch.label[:5].tolist()}")

if __name__ == "__main__":
    test_load()